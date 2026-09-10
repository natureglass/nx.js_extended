/**
 * WebGL — Phase 2.C: WebGL1 context binding for inline canvases.
 *
 * Phase 2.A laid the shared ES3 context (Skia owns EGL). Phase 2.B realized
 * the coexistence primitive (state save/restore + tenant offscreen FBO +
 * SkImages composite, hardware-proven on Tegra). Phase 2.C grows this file
 * from a null stub into a real WebGL1 context factory: getContext('webgl')
 * returns a non-null object backed by raw GLES3 dispatches into the 2.B
 * tenant FBO, wrapped per-frame in the 2.B bracket.
 *
 * Architecture:
 *   - One module-global WebGLState (`st`). The state matches upstream V8's
 *     beta.5 pattern (commit fb0468f) — WebGL object types (Buffer, Texture,
 *     Program, ...) are wrapped objects holding the GL name; the TS side
 *     passes the JS classes to $.webglInitClass so freshly-minted objects
 *     get the right prototype (instanceof works).
 *   - Each method call lazily ENTERs the per-frame bracket: snapshots Skia's
 *     GL state via nx_gl_state_save, binds the tenant FBO. Subsequent calls
 *     in the same frame stay in the bracket.
 *   - main.cc's present hook calls nx_webgl_compose_if_active() BEFORE
 *     nx_skia_gpu_present(): if the bracket is open, exits it (restores
 *     state + grCtx->resetContext()) then asks the bridge to compose the
 *     FBO into Skia's persistent canvas surface (cheap no-op when the FBO
 *     wasn't touched).
 *   - WebGL-only pixel store flags (UNPACK_FLIP_Y_WEBGL, UNPACK_PREMULTIPLY)
 *     are tracked here; emulation at upload time is a 2.E task — for 2.C
 *     they're stored and otherwise ignored (geometry-cube sets them to
 *     defaults).
 *   - A synthetic error slot backs WebGL-level validation failures
 *     (e.g. invalid object handles); getError() drains it before glGetError.
 *
 * Phase 2.C IS NOT the full 220-method surface. We implement what
 * geometry-cube + Three.js's MeshBasic path empirically calls; 2.E is where
 * the bulk semantics move to brewser-runtime TS and the rest of the surface
 * grows in. Methods the demo doesn't call are not implemented — calling
 * them throws TypeError ("foo is not a function"), which the diagnostic
 * Proxy in webgl-shim.ts logs so the next iteration can add them.
 *
 * Phase 2.G adds WebGL2 (extends this class).
 */
#include "webgl.h"
#include "webgl_bridge.h"
#include "skia_gpu.h"
#include "error.h"
#include "wrap.h"
#include "image.h"

#include <EGL/egl.h>
#include <GLES3/gl3.h>
// Batch 3 (ledger #57) — gl2ext.h declares glClipControlEXT,
// glPolygonOffsetClampEXT, glQueryCounterEXT, glMaxShaderCompilerThreadsKHR,
// and the glEnableiOES/EXT family used by OES_draw_buffers_indexed.
// Multi-draw instanced variants are NOT in gl2ext.h; those land as
// engine-native loop shims (see w_multi_draw_arrays_instanced_webgl et al).
#include <GLES2/gl2ext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "include/gpu/ganesh/GrDirectContext.h"

using namespace v8;

// WebGL-specific enums (not part of GLES headers).
#define NX_GL_UNPACK_FLIP_Y_WEBGL                  0x9240
#define NX_GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL       0x9241
#define NX_GL_UNPACK_COLORSPACE_CONVERSION_WEBGL   0x9243
#define NX_GL_CONTEXT_LOST_WEBGL                   0x9242

// Common WebGL1 pname values returned from getParameter (subset).
#define NX_GL_MAX_VERTEX_ATTRIBS                   0x8869
#define NX_GL_MAX_TEXTURE_SIZE                     0x0D33
#define NX_GL_MAX_TEXTURE_IMAGE_UNITS              0x8872
#define NX_GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS     0x8B4D
#define NX_GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS       0x8B4C
#define NX_GL_MAX_FRAGMENT_UNIFORM_VECTORS         0x8DFD
#define NX_GL_MAX_VERTEX_UNIFORM_VECTORS           0x8DFB
#define NX_GL_MAX_VARYING_VECTORS                  0x8DFC
#define NX_GL_MAX_CUBE_MAP_TEXTURE_SIZE            0x851C
#define NX_GL_MAX_RENDERBUFFER_SIZE                0x84E8
#define NX_GL_MAX_VIEWPORT_DIMS                    0x0D3A
#define NX_GL_ALIASED_LINE_WIDTH_RANGE             0x846E
#define NX_GL_ALIASED_POINT_SIZE_RANGE             0x846D

namespace {

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

enum ObjKind : uint8_t {
	K_BUFFER,
	K_FRAMEBUFFER,
	K_PROGRAM,
	K_RENDERBUFFER,
	K_SHADER,
	K_TEXTURE,
	K_UNIFORM_LOCATION,
	K_ACTIVE_INFO,
	K_SHADER_PRECISION_FORMAT,
	// Phase 2.G.1 cut #3 — v2-only handle kinds. New entries MUST stay
	// BEFORE K_COUNT (which sizes the protos[] array). New entries are
	// stamped on v2 contexts by nx_webgl2_init_class's MAP[] extension.
	K_VERTEX_ARRAY_OBJECT,
	// Phase-1.5-MED handle kinds (ledger #53).
	K_QUERY,
	K_SAMPLER,
	K_SYNC,
	// Phase-1.5-MED-HIGH handle kind (ledger #55).
	K_TRANSFORM_FEEDBACK,
	K_COUNT,
};

struct GLObj {
	uint32_t id;
	int32_t loc; // uniform location for K_UNIFORM_LOCATION
	uint8_t kind;
	// Ledger #95 — WebGL "deletion" is a JS-wrapper concept, not a GL
	// name-freeing one. After `gl.delete<X>(w)`, the JS wrapper `w` stays
	// alive (the app may still hold references), but any subsequent
	// `gl.bind<X>(w)` etc. must generate INVALID_OPERATION per WebGL 1
	// spec §5.13–5.17. Set true by each `w_delete_*` FN; checked at the
	// head of each `w_bind_*` (buffer / texture / renderbuffer /
	// framebuffer) via `obj_deleted()`. Wrapper stays in the #92 cache
	// so `getFramebufferAttachmentParameter(...OBJECT_NAME)` and other
	// queries continue to return the same JS wrapper for identity
	// comparisons (WebGL objects surviving deletion is a spec property,
	// not a leak).
	bool deleted = false;
	// K_SYNC uses a GLsync (opaque pointer), not a GLuint name. Held in
	// its own field to avoid punning `id`+`loc` bits into a 64-bit
	// pointer (which introduces sign-extension traps on the loc int32).
	// For all non-K_SYNC handles this stays zero and is not read.
	GLsync sync = nullptr;
};

struct WebGLState {
	// Per-context WebGL state (multi-WebGL-canvas independence, 2026-09): ONE
	// instance per page <canvas>, allocated in make_context_carrier and swapped
	// into the active module `st` by use_ctx(info.This()) at the top of every
	// method. Stacked canvases (tetr.io #pixi + #pixi-fg) each keep their own
	// user_snap / bound_fbo_js / VAO / dims + their own bridge tenant FBO, so
	// they render independently. The GLOBAL bits — the per-frame Skia bracket
	// (g_bracket_open / g_snap), the JS class prototypes (g_protos), the
	// wrapper cache, enabled extensions, and aliased-link set — are shared
	// across contexts (one Skia-owned EGL context, one set of JS classes) and
	// live at module scope below, NOT here.
	uint32_t tenant_id = 0;   // this canvas's bridge tenant FBO id (0 = default)
	GLenum synthetic_error = GL_NO_ERROR;
	// WebGL-only pixel store emulation state (stored only; 2.E does the work).
	bool unpack_flip_y = false;
	bool unpack_premultiply = false;
	int unpack_alignment = 4;
	int pack_alignment = 4;
	// Full user-side GL state snapshot for cross-bracket persistence.
	// Per the WebGL spec, all GL state a demo sets stays in effect for
	// subsequent calls until the demo explicitly changes it. Skia clobbers
	// GL between the runtime's `exit_bracket()` at compose time and the
	// demo's next `enter_bracket()`; without persistence tracking, a demo
	// that binds program / VAO / textures / viewport / blend func / etc.
	// ONCE at init (rather than per-frame) silently ends up rendering
	// under Skia's state on every subsequent frame. Three.js demos re-emit
	// state per material and accidentally survived the bug; raw-WebGL
	// demos like webgl2demo (Sunset Sea) and spectraplay's audio-reactive
	// visualizer expose it. Design: exit_bracket() saves the demo's live
	// state INTO `user_snap` BEFORE restoring Skia's snap. enter_bracket()
	// saves Skia's live state (as before, into `snap`) and then RESTORES
	// `user_snap` after the cut #15 defaults reset. valid flag gates the
	// first-ever enter_bracket where no user state has been captured yet
	// (cut #15's WebGL-defaults reset handles Three.js's initial cache
	// consistency; user_snap takes over from frame 2 onward).
	nx_gl_state_snap_t user_snap;
	bool user_snap_valid = false;
	// Auto-allocated VAO to hold user's attribute-enable / pointer state
	// for WebGL 1 demos (which never call bindVertexArray). Without this,
	// the demo's `enableVertexAttribArray` + `vertexAttribPointer` calls
	// operate on GL's default VAO (VAO 0), which Ganesh also uses for its
	// own rendering — Skia clobbers the demo's attribute setup between
	// frames, and drawArrays reads stale/wrong attribute pointers. Caught
	// while porting the sensors gyro cube. Allocated once at first
	// enter_bracket and bound before user's first GL call each frame so
	// demo's attribute state has a home Ganesh doesn't touch. WebGL 2
	// demos that call bindVertexArray explicitly override this via the
	// shadow (user_snap.vao) — their intent wins.
	GLuint auto_user_vao = 0;
	// Currently bound DRAW_FRAMEBUFFER as the JS sees it. 0 = "default" =
	// tenant FBO (the only difference from a browser WebGL where 0 = swap
	// chain back buffer). When the user binds a non-null framebuffer object,
	// we forward the FBO name directly.
	uint32_t bound_fbo_js = 0;
	// Bookkeeping for the per-frame "did we touch the FBO?" signal — set when
	// the bound target IS the default (tenant) FBO. The bridge dirties on
	// every draw/clear that lands in tenant; non-default-FBO writes don't.
	bool draw_into_default = true;
	// Drawing buffer dimensions (canvas w/h reported via drawingBufferWidth/
	// drawingBufferHeight, also drives default viewport / scissor).
	int width = 640;
	int height = 360;
	// NOTE (multi-WebGL-canvas independence): protos[], wrapper_cache,
	// enabled_exts, and programs_with_aliased_link were MOVED OUT of this
	// per-context struct to module globals (g_protos / g_wrapper_cache /
	// g_enabled_exts / g_programs_with_aliased_link) below. They are shared
	// across all contexts because: protos are the one set of JS classes
	// ($.webglInitClass runs once at boot, before any context); GL object
	// NAMES are globally unique in the one real GL context so the wrapper
	// cache never key-collides across contexts; enabled_exts sharing was
	// already documented spec-legal; aliased-link is keyed by (unique)
	// program name. Keeping them global avoids per-context duplication and
	// keeps `getX() === originalX` identity working no matter which context
	// is active.
};

WebGLState *st = nullptr;   // ACTIVE per-context state (set by use_ctx)

// ---------------------------------------------------------------------------
// Shared-across-contexts globals (multi-WebGL-canvas independence). One
// Skia-owned EGL context + one set of JS classes back ALL WebGL contexts, so
// these are module-scope, not per-WebGLState. See the NOTE in WebGLState.
// ---------------------------------------------------------------------------
bool g_bracket_open = false;              // the per-frame Skia bracket is global
nx_gl_state_snap_t g_snap;                // Skia's saved GL state (one Skia)
Global<Object> g_protos[K_COUNT];         // WebGL object-class prototypes
std::unordered_map<uint64_t, Global<Object>> g_wrapper_cache;  // ledger #92
std::unordered_set<std::string> g_enabled_exts;               // ledger #67
std::unordered_set<GLuint> g_programs_with_aliased_link;      // ledger #68

// Per-canvas tenant id allocator + deferred-free queue. Each context gets a
// unique tenant id (0 for the first, reusing the legacy default tenant). When
// a context is GC'd its tenant id is queued here and the GL handles are freed
// at the next enter_bracket (GL-current), never in the GC finalizer.
uint32_t g_next_tenant_id = 0;
std::vector<uint32_t> g_tenants_to_free;

// Native GL extension cache. Populated once at first WebGL context
// creation via glGetStringi(GL_EXTENSIONS, i) over GL_NUM_EXTENSIONS —
// the ES3 path; the legacy glGetString(GL_EXTENSIONS) is deprecated
// for 3.x contexts and Mesa returns NULL for it there. Populate emits
// a one-shot [gl-ext-dump] boot log the next hardware session greps
// to resolve all bucket-B extension rows (s3tc, s3tc_srgb, astc,
// anisotropic, EXT_frag_depth, WEBGL_draw_buffers, EXT_shader_texture_lod,
// disjoint_timer_query, etc.) against the actual Mesa-Nouveau / Tegra
// X1 driver capability set. Introspection surface only — no
// advertisement changes to SUPPORTED[] at [webgl.cc:782] this phase;
// that is gated on the hardware dump per the phase-0 plan.
static std::vector<std::string> s_native_ext_list;
static std::string s_native_exts_joined;
static bool s_native_exts_populated = false;

static void populate_native_extensions() {
	if (s_native_exts_populated) return;
	s_native_exts_populated = true;
	// GL error hygiene (rider B) — drain any pre-existing error before the
	// enumeration so any residual we observe below is attributable to our
	// own calls, and drain at the end so downstream engine code (notably
	// the commit-2 [bridge-fbo] completeness assert) never sees a stale
	// error left by us.
	while (glGetError() != GL_NO_ERROR) { /* drain */ }
	GLint n = 0;
	glGetIntegerv(GL_NUM_EXTENSIONS, &n);
	if (n < 0) n = 0;
	s_native_ext_list.reserve((size_t)n);
	for (GLint i = 0; i < n; i++) {
		const GLubyte *e = glGetStringi(GL_EXTENSIONS, (GLuint)i);
		if (e) s_native_ext_list.emplace_back((const char *)e);
	}
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		fprintf(stderr,
		        "[gl-ext-dump:err] enumeration left GL error 0x%x — draining\n",
		        (unsigned)err);
		while (glGetError() != GL_NO_ERROR) { /* drain */ }
	}
	std::sort(s_native_ext_list.begin(), s_native_ext_list.end());
	size_t total = 0;
	for (const auto &e : s_native_ext_list) total += e.size() + 1;
	s_native_exts_joined.reserve(total);
	for (size_t i = 0; i < s_native_ext_list.size(); i++) {
		if (i) s_native_exts_joined.push_back(' ');
		s_native_exts_joined += s_native_ext_list[i];
	}
	// [gl-ext-dump] one-shot extension dump — a boot-time diagnostic (~134
	// lines) used during hardware capture. Disabled for the clean/production
	// build; the joined list above is still built for capability checks. Set
	// GL_EXT_DUMP to 1 to re-enable for a hardware-extension audit.
#if defined(GL_EXT_DUMP) && GL_EXT_DUMP
	fprintf(stderr, "[gl-ext-dump] count=%d\n", (int)s_native_ext_list.size());
	for (const auto &e : s_native_ext_list) {
		fprintf(stderr, "[gl-ext-dump] %s\n", e.c_str());
	}
	fflush(stderr);
#endif
}

// Phase-1 batch-1 helper — probe whether the driver's native GL extension
// list (populated at first context creation by populate_native_extensions()
// per ledger #43) contains the given `GL_*` token as an exact match.
// Cache is std::sort'ed; use binary search. Returns false before the
// cache is populated, which is safe: the only caller is the per-context
// getSupportedExtensions build path, which runs strictly AFTER
// make_context_carrier's eager populate call.
static bool has_native_ext(const char *token) {
	if (!token || !s_native_exts_populated) return false;
	return std::binary_search(s_native_ext_list.begin(),
	                          s_native_ext_list.end(), std::string(token));
}

// Phase-1 batch-2 helper — driver-accept probe for ESSL-100 shader
// `#extension GL_EXT_frag_depth : enable` + `gl_FragDepthEXT` writes.
// Even though `GL_EXT_frag_depth` is in the 134-list on Nouveau, the
// WebGL1 shader form (#version 100 with gl_FragDepthEXT) needs a compile
// probe — a driver that ships the native ES3 extension token might not
// accept the promotion-path GLSL directive. Result cached process-wide;
// probe runs at most once. Emits `[frag-depth-probe] ACCEPT` or
// `[frag-depth-probe] REJECT log=...` to stderr for the batch report.
// Establishes the pattern for batch-3's WEBGL_blend_func_extended
// ESSL-100 probe.
static bool s_frag_depth_probed = false;
static bool s_frag_depth_ok = false;
static bool probe_ext_frag_depth() {
	if (s_frag_depth_probed) return s_frag_depth_ok;
	s_frag_depth_probed = true;
	const char *src =
	    "#version 100\n"
	    "#extension GL_EXT_frag_depth : enable\n"
	    "precision mediump float;\n"
	    "void main() {\n"
	    "    gl_FragColor = vec4(1.0);\n"
	    "    gl_FragDepthEXT = 0.5;\n"
	    "}\n";
	GLuint sh = glCreateShader(GL_FRAGMENT_SHADER);
	if (!sh) {
		fprintf(stderr, "[frag-depth-probe] REJECT reason=create-shader-failed\n");
		fflush(stderr);
		return false;
	}
	glShaderSource(sh, 1, &src, nullptr);
	glCompileShader(sh);
	GLint ok = 0;
	glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
	if (ok) {
		fprintf(stderr, "[frag-depth-probe] ACCEPT\n");
	} else {
		char log[512] = {0};
		GLsizei n = 0;
		glGetShaderInfoLog(sh, sizeof(log) - 1, &n, log);
		fprintf(stderr, "[frag-depth-probe] REJECT log=%.*s\n",
		        (int)(n > 0 ? n : 0), log);
	}
	fflush(stderr);
	glDeleteShader(sh);
	s_frag_depth_ok = (ok != 0);
	return s_frag_depth_ok;
}

// Phase-1 batch-1 helper — v1 vs v2 context detection from the receiver
// object. `make_context_carrier` at [webgl.cc: is_v2 ? Set(__webgl2)]
// stamps `__webgl2 = true` on v2 carriers. All context-shared FNs
// (`w_get_extension`, `w_get_supported_extensions`, `w_get_parameter`)
// use this helper to branch v1/v2 without needing separate FUNCS[]
// registrations. Falls back to false (v1) if the receiver lacks the
// property. Uses `iso->GetCurrentContext()` directly rather than the
// `cur()` helper because `cur()` is declared below this point in the
// file and the ordering has not been reworked yet.
static bool is_v2_context(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Context> c = iso->GetCurrentContext();
	// V8 12.x removed `info.Holder()`; use `info.This()` which returns
	// the receiver Local<Object> for the standard call site.
	Local<Value> this_v = info.This();
	if (!this_v->IsObject()) return false;
	Local<Object> self = this_v.As<Object>();
	Local<Value> v;
	if (!self->Get(c, String::NewFromUtf8(iso, "__webgl2").ToLocalChecked())
	         .ToLocal(&v))
		return false;
	return v->IsBoolean() && v.As<Boolean>()->Value();
}

void free_gl_obj(GLObj *o) { delete o; }

// Ledger #92 — wrapper cache key/lookup helpers. `cache_key` packs kind
// into high 32 + id into low 32. `is_cacheable_kind` gates out kinds
// that shouldn't be cached (uniform-locations use `loc` not `id`; active-
// info / shader-precision-format are transient value objects; sync
// objects use pointers not GLuint names).
static inline uint64_t cache_key(uint8_t kind, GLuint id) {
	return ((uint64_t)kind << 32) | (uint64_t)id;
}
static inline bool is_cacheable_kind(uint8_t kind) {
	return kind != K_UNIFORM_LOCATION && kind != K_ACTIVE_INFO &&
	       kind != K_SHADER_PRECISION_FORMAT && kind != K_SYNC;
}
// Erase a `(kind, id)` entry from the wrapper cache. Called from every
// gl.delete<X>() natives so the JS wrapper reference stops keeping the
// (now-deleted) GL name alive on the JS side. If a subsequent
// gl.createBuffer() happens to reuse the same GL name (spec-legal), the
// re-created wrapper populates a fresh cache entry.
static inline void erase_wrapper_cache(uint8_t kind, GLuint id) {
	if (st && id != 0 && is_cacheable_kind(kind))
		g_wrapper_cache.erase(cache_key(kind, id));
}
Local<Object> new_gl_obj(Isolate *iso, uint8_t kind, GLuint id,
                         GLint loc = -1) {
	// Ledger #92 — cache-first lookup for identity preservation. Same
	// (kind, id) always returns the same JS wrapper across gl.getParameter
	// / gl.getVertexAttrib / gl.getFramebufferAttachmentParameter / etc.
	if (st && id != 0 && is_cacheable_kind(kind)) {
		auto it = g_wrapper_cache.find(cache_key(kind, id));
		if (it != g_wrapper_cache.end() && !it->second.IsEmpty()) {
			return it->second.Get(iso);
		}
	}
	Local<Object> obj = nx::NewWrapped(iso);
	if (st && !g_protos[kind].IsEmpty()) {
		obj->SetPrototype(iso->GetCurrentContext(), g_protos[kind].Get(iso))
		    .Check();
	}
	GLObj *o = new GLObj{id, loc, kind};
	nx::Wrap<GLObj>(iso, obj, o, free_gl_obj);
	if (st && id != 0 && is_cacheable_kind(kind)) {
		g_wrapper_cache.emplace(cache_key(kind, id), Global<Object>(iso, obj));
	}
	return obj;
}

// Ledger #95 — create-path variant of new_gl_obj. Post-#95 the delete
// natives for buffer/texture/renderbuffer/framebuffer keep the JS
// wrapper in the wrapper_cache (to preserve identity for subsequent
// getFramebufferAttachmentParameter etc.); if a driver-side glGenX
// then reuses the freed GL name, `new_gl_obj`'s cache-first lookup
// would silently hand back the DELETED wrapper as the "new" one. Evict
// any stale cache entry with the same (kind, id) FIRST so createX
// always returns a fresh JS object. Called from every w_create_* FN
// so gen-then-reuse-of-freed-name is spec-safe even for kinds that
// don't (yet) use the .deleted mark.
Local<Object> new_gl_obj_create(Isolate *iso, uint8_t kind, GLuint id) {
	if (st && id != 0 && is_cacheable_kind(kind)) {
		g_wrapper_cache.erase(cache_key(kind, id));
	}
	return new_gl_obj(iso, kind, id);
}

GLObj *get_gl_obj(Local<Value> v) {
	if (v.IsEmpty() || !v->IsObject())
		return nullptr;
	return nx::Unwrap<GLObj>(v);
}

GLuint obj_id(Local<Value> v) {
	GLObj *o = get_gl_obj(v);
	return o ? o->id : 0;
}

// Ledger #95 — was this WebGL wrapper deleted via `gl.delete<X>()`?
// Callers use this at the head of `w_bind_*` / other consumer natives
// to reject deleted-object references with INVALID_OPERATION (WebGL 1
// spec §5.13–5.17). Null / undefined / non-GLObj arguments return
// false — bindX(null) is a valid "unbind" and must not be rejected.
inline bool obj_deleted(Local<Value> v) {
	GLObj *o = get_gl_obj(v);
	return o && o->deleted;
}

GLint uniform_loc(Local<Value> v) {
	GLObj *o = get_gl_obj(v);
	return (o && o->kind == K_UNIFORM_LOCATION) ? o->loc : -1;
}

void record_error(GLenum err) {
	if (st && st->synthetic_error == GL_NO_ERROR)
		st->synthetic_error = err;
}

// Ledger #67 — mark a WebGL extension as enabled on this context. Called
// from every success branch in `w_get_extension` (a non-null return means
// the caller opted in to the extension per WebGL spec).
static void record_ext_enabled(const char *name) {
	if (st && name) g_enabled_exts.insert(name);
}

// Ledger #67 — has the caller opted in to this extension via getExtension?
// Consulted by `w_get_parameter`'s extension-gated pname branches so
// unadvertised-or-unenabled extension constants report null + INVALID_ENUM.
static bool is_ext_enabled(const char *name) {
	if (!st || !name) return false;
	return g_enabled_exts.count(std::string(name)) > 0;
}

inline Local<Context> cur(Isolate *iso) { return iso->GetCurrentContext(); }

// ---------------------------------------------------------------------------
// Argument unwrap helpers
// ---------------------------------------------------------------------------

inline uint32_t a_u32(const FunctionCallbackInfo<Value> &info, int i) {
	return info[i]->Uint32Value(cur(info.GetIsolate())).FromMaybe(0);
}
inline int32_t a_i32(const FunctionCallbackInfo<Value> &info, int i) {
	return info[i]->Int32Value(cur(info.GetIsolate())).FromMaybe(0);
}
inline double a_f64(const FunctionCallbackInfo<Value> &info, int i) {
	return info[i]->NumberValue(cur(info.GetIsolate())).FromMaybe(0.0);
}
inline float a_f32(const FunctionCallbackInfo<Value> &info, int i) {
	return (float)a_f64(info, i);
}
inline bool a_bool(const FunctionCallbackInfo<Value> &info, int i) {
	return info[i]->BooleanValue(info.GetIsolate());
}
inline int64_t a_i64(const FunctionCallbackInfo<Value> &info, int i) {
	return info[i]->IntegerValue(cur(info.GetIsolate())).FromMaybe(0);
}

// Returns raw bytes of any ArrayBufferView (honoring byteOffset).
uint8_t *view_bytes(Local<Value> v, size_t *len) {
	if (v.IsEmpty() || !v->IsArrayBufferView()) {
		if (len) *len = 0;
		return nullptr;
	}
	Local<ArrayBufferView> view = v.As<ArrayBufferView>();
	if (len) *len = view->ByteLength();
	return (uint8_t *)view->Buffer()->Data() + view->ByteOffset();
}

// Extract a contiguous f32 list from either Float32Array (zero-copy) or a
// plain JS array (copied into `tmp`).
bool f32_list(Isolate *iso, Local<Value> v, std::vector<float> &tmp,
              const float **out, size_t *n) {
	if (!v.IsEmpty() && v->IsFloat32Array()) {
		Local<Float32Array> ta = v.As<Float32Array>();
		*n = ta->Length();
		*out = (const float *)((uint8_t *)ta->Buffer()->Data() + ta->ByteOffset());
		return true;
	}
	if (!v.IsEmpty() && v->IsArray()) {
		Local<Array> arr = v.As<Array>();
		Local<Context> ctx = cur(iso);
		uint32_t len = arr->Length();
		tmp.resize(len);
		for (uint32_t i = 0; i < len; i++) {
			Local<Value> el;
			tmp[i] = 0.f;
			if (arr->Get(ctx, i).ToLocal(&el))
				tmp[i] = (float)el->NumberValue(ctx).FromMaybe(0.0);
		}
		*out = tmp.data();
		*n = tmp.size();
		return true;
	}
	return false;
}

// Extract an i32 list from Int32Array / plain JS array.
bool i32_list(Isolate *iso, Local<Value> v, std::vector<int32_t> &tmp,
              const int32_t **out, size_t *n) {
	if (!v.IsEmpty() && v->IsInt32Array()) {
		Local<Int32Array> ta = v.As<Int32Array>();
		*n = ta->Length();
		*out = (const int32_t *)((uint8_t *)ta->Buffer()->Data() + ta->ByteOffset());
		return true;
	}
	if (!v.IsEmpty() && v->IsArray()) {
		Local<Array> arr = v.As<Array>();
		Local<Context> ctx = cur(iso);
		uint32_t len = arr->Length();
		tmp.resize(len);
		for (uint32_t i = 0; i < len; i++) {
			Local<Value> el;
			tmp[i] = 0;
			if (arr->Get(ctx, i).ToLocal(&el))
				tmp[i] = el->Int32Value(ctx).FromMaybe(0);
		}
		*out = tmp.data();
		*n = tmp.size();
		return true;
	}
	return false;
}

// Phase-1.5-LOW helper — uint32 array unwrap. Mirror of i32_list but
// accepts Uint32Array (WebGL2's `uniform*uiv` typed-array shape) plus
// generic JS arrays. Used by the UNI_UIV macro for the 8 uint uniform
// vector setters landed in this tier.
bool u32_list(Isolate *iso, Local<Value> v, std::vector<uint32_t> &tmp,
              const uint32_t **out, size_t *n) {
	if (!v.IsEmpty() && v->IsUint32Array()) {
		Local<Uint32Array> ta = v.As<Uint32Array>();
		*n = ta->Length();
		*out = (const uint32_t *)((uint8_t *)ta->Buffer()->Data() +
		                          ta->ByteOffset());
		return true;
	}
	if (!v.IsEmpty() && v->IsArray()) {
		Local<Array> arr = v.As<Array>();
		Local<Context> ctx = cur(iso);
		uint32_t len = arr->Length();
		tmp.resize(len);
		for (uint32_t i = 0; i < len; i++) {
			Local<Value> el;
			tmp[i] = 0;
			if (arr->Get(ctx, i).ToLocal(&el))
				tmp[i] = el->Uint32Value(ctx).FromMaybe(0);
		}
		*out = tmp.data();
		*n = tmp.size();
		return true;
	}
	return false;
}

// UTF-8 string from JS, owned by caller (free with delete[]).
char *take_string(Isolate *iso, Local<Value> v) {
	if (v.IsEmpty() || !v->IsString()) {
		char *empty = new char[1];
		empty[0] = 0;
		return empty;
	}
	Local<String> s = v.As<String>();
	int len = s->Utf8Length(iso);
	char *buf = new char[len + 1];
	s->WriteUtf8(iso, buf, len);
	buf[len] = 0;
	return buf;
}

// ---------------------------------------------------------------------------
// Per-frame bracket — the 2.B contract
// ---------------------------------------------------------------------------

// Ensure the active context's tenant FBO exists at the canvas size and return
// its GL name. Each page canvas owns its own tenant (multi-WebGL-canvas
// independence); tenant_id 0 is the legacy/default tenant.
GLuint cur_tenant_fbo() {
	if (!st) return 0;
	nx_webgl_bridge_tenant_ensure(st->tenant_id, st->width, st->height);
	return nx_webgl_bridge_tenant_fbo(st->tenant_id);
}

// Free any tenants whose contexts were GC'd. Called from enter_bracket where
// the shared GL context is guaranteed current — GL deletes must NOT run in the
// V8 GC finalizer (no current context during GC). See free_webgl_state.
void process_pending_tenant_frees() {
	if (g_tenants_to_free.empty()) return;
	for (uint32_t id : g_tenants_to_free) nx_webgl_bridge_tenant_destroy(id);
	g_tenants_to_free.clear();
}

// Bind the ACTIVE context's render target + restore its intended GL state.
// Called by enter_bracket for a fresh frame AND by use_ctx when a method call
// switches to a different canvas inside an already-open bracket (so stacked
// canvases don't stomp each other's FBO / GL state). Does NOT save Skia state
// — that is the bracket's once-per-frame job in enter_bracket.
void apply_canvas_state() {
	if (!st) return;
	// Bind the user-currently-bound FBO if they switched; otherwise this
	// canvas's own tenant FBO. Either way, our viewport / clear etc. go to the
	// right target without the user noticing the redirect.
	GLuint target_fbo = st->bound_fbo_js == 0 ? cur_tenant_fbo()
	                                          : st->bound_fbo_js;
	glBindFramebuffer(GL_FRAMEBUFFER, target_fbo);
	// Phase 2.G.1 cut #15 — reset the WebGL default state that Skia's
	// Ganesh might have left in a different configuration. Three.js's
	// WebGLState cache initializes assuming these WebGL defaults; if
	// Skia left GL in a different state, Three.js's cache is out of sync
	// and cache short-circuits prevent Three.js from re-emitting the
	// state. Cut #15v log confirmed depth_mask+color_mask+clear_depth
	// are already correct at gl.clear time — meaning the depth CLEAR
	// works. The rejection must be at depth TEST time (depth_func!=LESS,
	// or depth_range inverted). Also reset cull/front-face and stencil
	// state for completeness — these are all cheap.
	glDepthMask(GL_TRUE);
	glStencilMask(0xFF);
	glDepthFunc(GL_LESS);
	glDepthRangef(0.0f, 1.0f);
	glStencilFunc(GL_ALWAYS, 0, 0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	// Auto-allocate this canvas's persistence VAO if we haven't yet. Bind it so
	// user's attribute state (enableVertexAttribArray / vertexAttribPointer)
	// lives in a VAO Ganesh doesn't touch. See auto_user_vao field comment.
	if (st->auto_user_vao == 0) {
		glGenVertexArrays(1, &st->auto_user_vao);
	}
	glBindVertexArray(st->auto_user_vao);
	// Establish user_snap baseline on the very first apply (captures
	// Skia's initial state + cut #15 defaults + auto_user_vao just bound)
	// OR restore accumulated user state on subsequent applies. See
	// exit_bracket() comment for why we no longer save user_snap at exit —
	// Skia's 2D rendering has already clobbered GL state by the time
	// copyBridgeToScreen fires exit_bracket, so live state is not the
	// demo's intent. Instead, each state-modifying w_* setter updates the
	// relevant user_snap field directly, so user_snap always reflects
	// what the demo INTENDED (per WebGL spec, that's what should persist
	// across bracket boundaries).
	if (!st->user_snap_valid) {
		// Ledger #72 — before capturing user_snap for the first time, seed
		// the GL viewport to canvas dimensions per WebGL 1 § 5.14.3
		// ("Initially the viewport rectangle is (0, 0, drawingBufferWidth,
		// drawingBufferHeight)"). Skia paints between context creation and
		// the demo's first WebGL call, leaving whatever viewport it needed —
		// empirically observed at 8×8 during a Tier-A image_bitmap diag,
		// causing full-viewport quads to render into a corner. Seed here
		// (not context_new) because Skia's late paint could clobber a
		// context_new-time seed before we capture user_snap; putting it
		// inside apply_canvas_state() guarantees the seed survives to capture.
		glViewport(0, 0, (GLsizei)st->width, (GLsizei)st->height);
		nx_gl_state_save(&st->user_snap);
		st->user_snap_valid = true;
	} else {
		nx_gl_state_restore(&st->user_snap);
	}
}

// The per-frame bracket is GLOBAL (one Skia-owned EGL context). enter_bracket
// saves Skia's GL state ONCE per frame and applies the active canvas's state;
// mid-frame canvas switches are handled by use_ctx -> apply_canvas_state.
void enter_bracket() {
	if (!st) return;
	if (g_bracket_open) return;
	if (!nx_webgl_bridge_is_initialized()) return;
	process_pending_tenant_frees();
	nx_gl_state_save(&g_snap);
	apply_canvas_state();
	g_bracket_open = true;
}

void exit_bracket() {
	if (!g_bracket_open) return;
	// Do NOT save user_snap here: by the time exit_bracket fires (from
	// w_copy_bridge_to_canvas after Skia's paintLiveOverlay already ran,
	// or from nx_webgl_compose_if_active at present), the shared GL
	// context's live state is Skia-clobbered — it no longer reflects the
	// demo's intent. Instead, user_snap is updated INCREMENTALLY as each
	// state-modifying w_* setter fires, so it captures what the demo
	// actually asked for regardless of what Skia does behind the scenes.
	// The sensors-cube regression (webgl 1 raw cube, "renders broken")
	// pinned this: demo called gl.enable(DEPTH_TEST) at init; Skia's 2D
	// paints disabled DT for compositing; exit_bracket saved DT=0 as
	// "user state"; every subsequent frame's enter_bracket restored DT=0;
	// cube drew without depth test → back faces show through front faces.
	nx_gl_state_restore(&g_snap);
	GrDirectContext *gr = nx_skia_gpu_gr_context();
	if (gr) gr->resetContext();
	g_bracket_open = false;
}

// Select the per-context WebGLState for the receiver of a WebGL method call.
// EVERY method runs this first (via the FN macro) so `st` reflects the canvas
// the call was made on. If a bracket is already open (mid-frame) and the canvas
// actually changed, rebind the incoming canvas's target + restore its intended
// state so two stacked canvases (tetr.io #pixi + #pixi-fg) don't stomp each
// other's FBO / GL state. Robust to a non-context receiver (Unwrap -> nullptr):
// keep the current st, changing nothing.
void use_ctx(Local<Value> recv) {
	WebGLState *c = nx::Unwrap<WebGLState>(recv);
	if (!c || c == st) return;
	st = c;
	if (g_bracket_open) apply_canvas_state();
}

inline void touch_fbo() {
	if (st && st->draw_into_default)
		nx_webgl_bridge_tenant_mark_dirty(st->tenant_id);
}

// ---------------------------------------------------------------------------
// Method implementations — the 2.C allowlist.
// Order matches upstream's table for easy diffing. Methods that the slice
// demo provably doesn't call are NOT here; calling them throws TypeError
// (caught + logged by the diagnostic Proxy). Each iteration adds whichever
// methods the proxy log reveals are missing.
// ---------------------------------------------------------------------------

// Every WebGL method is generated as a thin WRAPPER that first selects the
// receiver's per-context WebGLState via use_ctx(info.This()) — so `st` always
// reflects the canvas the call was made on (multi-WebGL-canvas independence) —
// then calls the real implementation `<name>_impl` that holds the method body.
// FNDECL(name) forward-declares just the wrapper (used where a method is
// referenced before its definition, e.g. the VAO natives in w_get_extension).
#define FNDECL(name) static void name(const FunctionCallbackInfo<Value> &info)
#define FN(name)                                                               \
	static void name##_impl(const FunctionCallbackInfo<Value> &info);          \
	static void name(const FunctionCallbackInfo<Value> &info) {                \
		use_ctx(info.This());                                                  \
		name##_impl(info);                                                     \
	}                                                                          \
	static void name##_impl(const FunctionCallbackInfo<Value> &info)

// ----- State / capability -----
// Shadow-tracking pattern: state-modifying w_* setters update
// st->user_snap.<field> after glCall so user's INTENT is captured
// regardless of Skia clobbers between bracket close/reopen. See
// enter_bracket / exit_bracket comments + [[reference-bracket-state-
// persistence-bug]] for the design rationale.
FN(w_viewport) {
	enter_bracket();
	const GLint x = a_i32(info, 0), y = a_i32(info, 1);
	const GLint w = a_i32(info, 2), h = a_i32(info, 3);
	glViewport(x, y, w, h);
	if (st) {
		st->user_snap.viewport[0] = x;
		st->user_snap.viewport[1] = y;
		st->user_snap.viewport[2] = w;
		st->user_snap.viewport[3] = h;
	}
}
FN(w_scissor) {
	enter_bracket();
	glScissor(a_i32(info, 0), a_i32(info, 1), a_i32(info, 2), a_i32(info, 3));
}
FN(w_enable) {
	enter_bracket();
	const GLenum cap = a_u32(info, 0);
	glEnable(cap);
	if (st) {
		switch (cap) {
			case GL_BLEND:        st->user_snap.blend        = GL_TRUE; break;
			case GL_DEPTH_TEST:   st->user_snap.depth_test   = GL_TRUE; break;
			case GL_CULL_FACE:    st->user_snap.cull         = GL_TRUE; break;
			case GL_SCISSOR_TEST: st->user_snap.scissor      = GL_TRUE; break;
			case GL_STENCIL_TEST: st->user_snap.stencil_test = GL_TRUE; break;
		}
	}
}
FN(w_disable) {
	enter_bracket();
	const GLenum cap = a_u32(info, 0);
	glDisable(cap);
	if (st) {
		switch (cap) {
			case GL_BLEND:        st->user_snap.blend        = GL_FALSE; break;
			case GL_DEPTH_TEST:   st->user_snap.depth_test   = GL_FALSE; break;
			case GL_CULL_FACE:    st->user_snap.cull         = GL_FALSE; break;
			case GL_SCISSOR_TEST: st->user_snap.scissor      = GL_FALSE; break;
			case GL_STENCIL_TEST: st->user_snap.stencil_test = GL_FALSE; break;
		}
	}
}
FN(w_is_enabled) {
	enter_bracket();
	GLboolean b = glIsEnabled(a_u32(info, 0));
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), b == GL_TRUE));
}
FN(w_depth_func) { enter_bracket(); glDepthFunc(a_u32(info, 0)); }
FN(w_depth_mask) {
	enter_bracket();
	const GLboolean m = a_bool(info, 0);
	glDepthMask(m);
	if (st) st->user_snap.depth_mask = m;
}
FN(w_depth_range) { enter_bracket(); glDepthRangef(a_f32(info, 0), a_f32(info, 1)); }
FN(w_cull_face) { enter_bracket(); glCullFace(a_u32(info, 0)); }
FN(w_front_face) { enter_bracket(); glFrontFace(a_u32(info, 0)); }
FN(w_blend_func) {
	enter_bracket();
	const GLenum s = a_u32(info, 0), d = a_u32(info, 1);
	glBlendFunc(s, d);
	if (st) {
		st->user_snap.blend_src_rgb = st->user_snap.blend_src_a = s;
		st->user_snap.blend_dst_rgb = st->user_snap.blend_dst_a = d;
	}
}
FN(w_blend_func_separate) {
	enter_bracket();
	const GLenum sRgb = a_u32(info, 0), dRgb = a_u32(info, 1);
	const GLenum sA = a_u32(info, 2), dA = a_u32(info, 3);
	glBlendFuncSeparate(sRgb, dRgb, sA, dA);
	if (st) {
		st->user_snap.blend_src_rgb = sRgb;
		st->user_snap.blend_dst_rgb = dRgb;
		st->user_snap.blend_src_a   = sA;
		st->user_snap.blend_dst_a   = dA;
	}
}
FN(w_blend_equation) { enter_bracket(); glBlendEquation(a_u32(info, 0)); }
FN(w_blend_equation_separate) {
	enter_bracket();
	glBlendEquationSeparate(a_u32(info, 0), a_u32(info, 1));
}
FN(w_blend_color) {
	enter_bracket();
	glBlendColor(a_f32(info, 0), a_f32(info, 1), a_f32(info, 2), a_f32(info, 3));
}
FN(w_color_mask) {
	enter_bracket();
	const GLboolean r = a_bool(info, 0), g = a_bool(info, 1);
	const GLboolean b = a_bool(info, 2), a = a_bool(info, 3);
	glColorMask(r, g, b, a);
	if (st) {
		st->user_snap.color_mask[0] = r;
		st->user_snap.color_mask[1] = g;
		st->user_snap.color_mask[2] = b;
		st->user_snap.color_mask[3] = a;
	}
}
FN(w_stencil_func) {
	enter_bracket();
	glStencilFunc(a_u32(info, 0), a_i32(info, 1), a_u32(info, 2));
}
FN(w_stencil_func_separate) {
	enter_bracket();
	glStencilFuncSeparate(a_u32(info, 0), a_u32(info, 1), a_i32(info, 2),
	                      a_u32(info, 3));
}
FN(w_stencil_op) {
	enter_bracket();
	glStencilOp(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2));
}
FN(w_stencil_op_separate) {
	enter_bracket();
	glStencilOpSeparate(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2),
	                    a_u32(info, 3));
}
FN(w_stencil_mask) {
	enter_bracket();
	const GLuint m = a_u32(info, 0);
	glStencilMask(m);
	if (st) st->user_snap.stencil_mask = (GLint)m;
}
FN(w_stencil_mask_separate) {
	enter_bracket();
	glStencilMaskSeparate(a_u32(info, 0), a_u32(info, 1));
}
FN(w_polygon_offset) {
	enter_bracket();
	glPolygonOffset(a_f32(info, 0), a_f32(info, 1));
}
FN(w_sample_coverage) {
	enter_bracket();
	glSampleCoverage(a_f32(info, 0), a_bool(info, 1));
}
FN(w_line_width) { enter_bracket(); glLineWidth(a_f32(info, 0)); }
FN(w_hint) { enter_bracket(); glHint(a_u32(info, 0), a_u32(info, 1)); }

// Ledger #100 — WebGL 1 spec §5.14.10 / GLES 2 §4.4.4: drawArrays,
// drawElements, readPixels, copyTexImage2D, copyTexSubImage2D, and
// clear all must generate INVALID_FRAMEBUFFER_OPERATION when the
// currently-bound framebuffer is not FRAMEBUFFER_COMPLETE. Mesa
// Nouveau's native emits the error for clear correctly, but skips it
// for readPixels (observed via `renderbuffers-framebuffer-object-
// attachment` — clear PASSes the getError check, subsequent readPixels
// FAILs). Add a client-side gate for the miss-set. Gate is scoped to
// user-bound FBOs (`bound_fbo_js != 0`); the tenant FBO is always
// complete by construction and doesn't need the check.
static inline bool nx_fbo_complete_or_record_error() {
	if (!st || st->bound_fbo_js == 0) return true;
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		record_error(GL_INVALID_FRAMEBUFFER_OPERATION);
		return false;
	}
	return true;
}
FN(w_clear) {
	enter_bracket();
	if (!nx_fbo_complete_or_record_error()) return;
	glClear(a_u32(info, 0));
	touch_fbo();
}
FN(w_clear_color) {
	enter_bracket();
	const GLfloat r = a_f32(info, 0), g = a_f32(info, 1);
	const GLfloat b = a_f32(info, 2), a = a_f32(info, 3);
	glClearColor(r, g, b, a);
	if (st) {
		st->user_snap.clear_color[0] = r;
		st->user_snap.clear_color[1] = g;
		st->user_snap.clear_color[2] = b;
		st->user_snap.clear_color[3] = a;
	}
}
FN(w_clear_depth) { enter_bracket(); glClearDepthf(a_f32(info, 0)); }
FN(w_clear_stencil) { enter_bracket(); glClearStencil(a_i32(info, 0)); }
FN(w_finish) { enter_bracket(); glFinish(); }
FN(w_flush) { enter_bracket(); glFlush(); }

FN(w_pixel_storei) {
	const GLenum pname = a_u32(info, 0);
	const GLint val = a_i32(info, 1);
	switch (pname) {
	case NX_GL_UNPACK_FLIP_Y_WEBGL:
		if (st) st->unpack_flip_y = (val != 0);
		return;
	case NX_GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL:
		if (st) st->unpack_premultiply = (val != 0);
		return;
	case NX_GL_UNPACK_COLORSPACE_CONVERSION_WEBGL:
		// Ignored; stored value irrelevant for the slice. 2.E can implement.
		return;
	case GL_UNPACK_ALIGNMENT:
		if (st) st->unpack_alignment = val;
		break;
	case GL_PACK_ALIGNMENT:
		if (st) st->pack_alignment = val;
		break;
	}
	enter_bracket();
	glPixelStorei(pname, val);
}

// ----- Query -----
FN(w_get_error) {
	enter_bracket();
	GLenum err = (st && st->synthetic_error != GL_NO_ERROR)
	                 ? st->synthetic_error
	                 : glGetError();
	if (st) st->synthetic_error = GL_NO_ERROR;
	info.GetReturnValue().Set(Uint32::NewFromUnsigned(info.GetIsolate(), err));
}

FN(w_get_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum pname = a_u32(info, 0);
	switch (pname) {
	case GL_VERSION:
	case GL_VENDOR:
	case GL_RENDERER:
	case GL_SHADING_LANGUAGE_VERSION: {
		const GLubyte *s = glGetString(pname);
		const char *cs = s ? (const char *)s : "";
		info.GetReturnValue().Set(
		    String::NewFromUtf8(iso, cs, NewStringType::kNormal)
		        .ToLocalChecked());
		return;
	}
	// Phase-1 batch-1 — WEBGL_debug_renderer_info unmasked pnames.
	// UNMASKED_VENDOR_WEBGL (0x9245) → glGetString(GL_VENDOR);
	// UNMASKED_RENDERER_WEBGL (0x9246) → glGetString(GL_RENDERER).
	//
	// Ledger #67 — gated on getExtension('WEBGL_debug_renderer_info').
	// Pre-#67 this branch returned the string unconditionally, which
	// caused `extensions-webgl-debug-renderer-info` to FAIL its
	// "should not be queryable if extension is disabled" assertion.
	// Three.js + diagnostic tools always successfully vend the ext
	// object before querying, so gating here doesn't regress them.
	case 0x9245: /* UNMASKED_VENDOR_WEBGL */
	case 0x9246: /* UNMASKED_RENDERER_WEBGL */ {
		if (!is_ext_enabled("WEBGL_debug_renderer_info")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		const GLenum native =
		    (pname == 0x9245) ? GL_VENDOR : GL_RENDERER;
		const GLubyte *s = glGetString(native);
		const char *cs = s ? (const char *)s : "";
		info.GetReturnValue().Set(
		    String::NewFromUtf8(iso, cs, NewStringType::kNormal)
		        .ToLocalChecked());
		return;
	}
	// Phase-1 batch-1 — EXT_texture_filter_anisotropic MAX. Spec calls
	// for glGetFloatv (max is a GLfloat like 16.0f), not glGetIntegerv.
	// Ledger #67 — gated on getExtension('EXT_texture_filter_anisotropic')
	// (fixes conformance's "should be null. Was 16." assertion).
	case 0x84FF: /* MAX_TEXTURE_MAX_ANISOTROPY_EXT */ {
		if (!is_ext_enabled("EXT_texture_filter_anisotropic")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLfloat v = 0.f;
		glGetFloatv(pname, &v);
		info.GetReturnValue().Set(Number::New(iso, (double)v));
		return;
	}
	// ============================================================
	// Ledger #67 — extension-gated pname enforcement. WebGL spec § 5.14.3
	// requires each of these pnames to return null (and record
	// INVALID_ENUM) UNTIL getExtension has been called for the gating
	// extension. Pre-#67, the `default:` branch below fell through to
	// glGetIntegerv/glGetFloatv which returned a real value — the
	// conformance corpus explicitly checks for this and FAILs.
	// Each branch below: check is_ext_enabled(<name>); if not, synthesize
	// INVALID_ENUM + return null. If yes, delegate to the appropriate
	// native getter with the appropriate return-value shape.
	// ============================================================
	case 0x935C: /* CLIP_ORIGIN_EXT (EXT_clip_control) */
	case 0x935D: /* CLIP_DEPTH_MODE_EXT (EXT_clip_control) */ {
		if (!is_ext_enabled("EXT_clip_control")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLint v = 0;
		glGetIntegerv(pname, &v);
		info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, (uint32_t)v));
		return;
	}
	case 0x864F: /* DEPTH_CLAMP_EXT (EXT_depth_clamp) — GL enable-cap.
	              * getParameter returns the enable state as a boolean. */ {
		if (!is_ext_enabled("EXT_depth_clamp")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLboolean v = GL_FALSE;
		glGetBooleanv(pname, &v);
		info.GetReturnValue().Set(Boolean::New(iso, v != 0));
		return;
	}
	case 0x8E1B: /* POLYGON_OFFSET_CLAMP_EXT (EXT_polygon_offset_clamp) */ {
		if (!is_ext_enabled("EXT_polygon_offset_clamp")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLfloat v = 0.f;
		glGetFloatv(pname, &v);
		info.GetReturnValue().Set(Number::New(iso, (double)v));
		return;
	}
	case 0x88FC: /* MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL
	              * (WEBGL_blend_func_extended) */ {
		if (!is_ext_enabled("WEBGL_blend_func_extended")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLint v = 0;
		glGetIntegerv(pname, &v);
		info.GetReturnValue().Set(Int32::New(iso, v));
		return;
	}
	case 0x8B8B: /* FRAGMENT_SHADER_DERIVATIVE_HINT_OES
	              * (OES_standard_derivatives; core-in-ES3 but WebGL1 gates
	              * it behind getExtension per spec) */ {
		if (!is_ext_enabled("OES_standard_derivatives")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLint v = 0;
		glGetIntegerv(pname, &v);
		info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, (uint32_t)v));
		return;
	}
	case 0x91B1: /* COMPLETION_STATUS_KHR (KHR_parallel_shader_compile) */ {
		if (!is_ext_enabled("KHR_parallel_shader_compile")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLint v = 0;
		glGetIntegerv(pname, &v);
		info.GetReturnValue().Set(Boolean::New(iso, v != 0));
		return;
	}
	case GL_DEPTH_WRITEMASK:
	case GL_COLOR_WRITEMASK: {
		GLboolean v[4] = {0, 0, 0, 0};
		glGetBooleanv(pname, v);
		if (pname == GL_COLOR_WRITEMASK) {
			Local<Array> arr = Array::New(iso, 4);
			Local<Context> c = cur(iso);
			for (int i = 0; i < 4; i++)
				arr->Set(c, i, Boolean::New(iso, v[i] != 0)).Check();
			info.GetReturnValue().Set(arr);
		} else {
			info.GetReturnValue().Set(Boolean::New(iso, v[0] != 0));
		}
		return;
	}
	case GL_COLOR_CLEAR_VALUE:
	case GL_BLEND_COLOR:
	case GL_DEPTH_RANGE:
	case NX_GL_ALIASED_LINE_WIDTH_RANGE:
	case NX_GL_ALIASED_POINT_SIZE_RANGE: {
		GLfloat v[4] = {0, 0, 0, 0};
		glGetFloatv(pname, v);
		int n = (pname == GL_DEPTH_RANGE || pname == NX_GL_ALIASED_LINE_WIDTH_RANGE ||
		         pname == NX_GL_ALIASED_POINT_SIZE_RANGE) ? 2 : 4;
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, n * 4);
		memcpy(ab->Data(), v, n * 4);
		info.GetReturnValue().Set(Float32Array::New(ab, 0, n));
		return;
	}
	case NX_GL_MAX_VIEWPORT_DIMS: {
		GLint v[2] = {0, 0};
		glGetIntegerv(pname, v);
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, 8);
		memcpy(ab->Data(), v, 8);
		info.GetReturnValue().Set(Int32Array::New(ab, 0, 2));
		return;
	}
	case GL_STENCIL_BITS:
	case GL_DEPTH_BITS: {
		// Phase-0 commit 2 — explicit case for grep-visibility of the
		// STENCIL_BITS/DEPTH_BITS wire. The tenant FBO is currently bound
		// (enter_bracket() above); glGetIntegerv returns THAT FBO's
		// attachment bits. Post-DEPTH24_STENCIL8 upgrade in webgl_bridge.cc
		// create_fbo, STENCIL_BITS = 8 and DEPTH_BITS = 24 on the default
		// (tenant) framebuffer, matching getContextAttributes's advertised
		// {depth:true, stencil:true}. Demo-bound custom FBOs report their
		// own attachment bits — spec-correct without special-casing.
		// See NXJS_PATCHES_NEEDED.md #46 recurrence tell: a hardware regression
		// where getParameter(STENCIL_BITS) reads 0 while the FBO log says
		// [bridge-fbo:complete] stencil=8 = the enter_bracket path decoupled
		// from create_fbo (broken bracket contract), NOT this switch.
		GLint v = 0;
		glGetIntegerv(pname, &v);
		info.GetReturnValue().Set(Int32::New(iso, v));
		return;
	}
	// Ledger #92 — object-returning pnames. WebGL spec: these queries
	// return a WebGLBuffer / WebGLProgram / WebGLTexture / WebGLFramebuffer
	// / WebGLRenderbuffer / WebGLVertexArrayObject wrapper (or `null`
	// when nothing is bound). Pre-#92 they fell through to the default
	// integer branch and callers got raw GLuint numbers — every
	// `bindingObj == originalObj` identity check in the corpus failed.
	// The wrapper cache in `new_gl_obj` ensures returned wrappers are the
	// SAME JS object handed out by the original `gl.create<X>()` call.
	case 0x8894 /* GL_ARRAY_BUFFER_BINDING */:
	case 0x8895 /* GL_ELEMENT_ARRAY_BUFFER_BINDING */: {
		GLint v = 0;
		glGetIntegerv(pname, &v);
		if (v == 0) info.GetReturnValue().SetNull();
		else info.GetReturnValue().Set(new_gl_obj(iso, K_BUFFER, (GLuint)v));
		return;
	}
	case 0x8B8D /* GL_CURRENT_PROGRAM */: {
		GLint v = 0;
		glGetIntegerv(pname, &v);
		if (v == 0) info.GetReturnValue().SetNull();
		else info.GetReturnValue().Set(new_gl_obj(iso, K_PROGRAM, (GLuint)v));
		return;
	}
	case 0x8069 /* GL_TEXTURE_BINDING_2D */:
	case 0x8514 /* GL_TEXTURE_BINDING_CUBE_MAP */:
	case 0x8C1D /* GL_TEXTURE_BINDING_2D_ARRAY (v2) */:
	case 0x806A /* GL_TEXTURE_BINDING_3D (v2) */: {
		GLint v = 0;
		glGetIntegerv(pname, &v);
		if (v == 0) info.GetReturnValue().SetNull();
		else info.GetReturnValue().Set(new_gl_obj(iso, K_TEXTURE, (GLuint)v));
		return;
	}
	case 0x8CA6 /* GL_DRAW_FRAMEBUFFER_BINDING == GL_FRAMEBUFFER_BINDING */:
	case 0x8CAA /* GL_READ_FRAMEBUFFER_BINDING (v2) */: {
		GLint v = 0;
		glGetIntegerv(pname, &v);
		if (v == 0) info.GetReturnValue().SetNull();
		else info.GetReturnValue().Set(new_gl_obj(iso, K_FRAMEBUFFER, (GLuint)v));
		return;
	}
	case 0x8CA7 /* GL_RENDERBUFFER_BINDING */: {
		GLint v = 0;
		glGetIntegerv(pname, &v);
		if (v == 0) info.GetReturnValue().SetNull();
		else info.GetReturnValue().Set(new_gl_obj(iso, K_RENDERBUFFER, (GLuint)v));
		return;
	}
	case 0x85B5 /* GL_VERTEX_ARRAY_BINDING (also VERTEX_ARRAY_BINDING_OES) */: {
		// Ledger #93 — extension-gated on v1: WebGL 1 conformance requires
		// this pname to return null + INVALID_ENUM until getExtension(
		// 'OES_vertex_array_object') has been called (same class as #67's
		// extension-gated pnames). v2 core exposes VERTEX_ARRAY_BINDING
		// unconditionally.
		if (!is_v2_context(info) &&
		    !is_ext_enabled("OES_vertex_array_object")) {
			record_error(GL_INVALID_ENUM);
			info.GetReturnValue().SetNull();
			return;
		}
		GLint v = 0;
		glGetIntegerv(pname, &v);
		if (v == 0) info.GetReturnValue().SetNull();
		else info.GetReturnValue().Set(
		    new_gl_obj(iso, K_VERTEX_ARRAY_OBJECT, (GLuint)v));
		return;
	}
	default: {
		GLint v = 0;
		glGetIntegerv(pname, &v);
		info.GetReturnValue().Set(Int32::New(iso, v));
		return;
	}
	}
}

// Forward decls for VAO natives referenced by w_get_extension's
// OES_vertex_array_object branch (RUNTIME_SHIMS #42 / pre-arm route).
// Actual definitions live in the cut #3 VAO block near line 1580.
FNDECL(w_create_vertex_array);
FNDECL(w_bind_vertex_array);
FNDECL(w_delete_vertex_array);
FNDECL(w_is_vertex_array);
// Phase-1 batch-2 forward decls — the ANGLE_instanced_arrays and
// WEBGL_draw_buffers ext objects vend suffixed method names that alias
// these v2-core natives; the actual FN bodies are defined later in the
// file. Also `w_is_context_lost` used by the WEBGL_lose_context branch.
// Same pattern as the VAO forward decls above.
FNDECL(w_draw_arrays_instanced);
FNDECL(w_draw_elements_instanced);
FNDECL(w_vertex_attrib_divisor);
FNDECL(w_draw_buffers);
FNDECL(w_is_context_lost);

// Batch 3 (ledger #57) forward decls — the ext objects vend these method
// symbols; actual FN bodies live in the batch-3 block near end-of-file.
// Also #53's query-family natives (w_create_query, etc.) used by v1's
// EXT_disjoint_timer_query lifecycle aliasing.
FNDECL(w_create_query);
FNDECL(w_delete_query);
FNDECL(w_is_query);
FNDECL(w_begin_query);
FNDECL(w_end_query);
FNDECL(w_get_query);
FNDECL(w_get_query_parameter);
FNDECL(w_clip_control_ext);
FNDECL(w_polygon_offset_clamp_ext);
FNDECL(w_query_counter_ext);
FNDECL(w_max_shader_compiler_threads_khr);
FNDECL(w_enable_i);
FNDECL(w_disable_i);
FNDECL(w_blend_equation_i);
FNDECL(w_blend_equation_separate_i);
FNDECL(w_blend_func_i);
FNDECL(w_blend_func_separate_i);
FNDECL(w_color_mask_i);
FNDECL(w_is_enabled_i);
FNDECL(w_multi_draw_arrays_webgl);
FNDECL(w_multi_draw_elements_webgl);
FNDECL(w_multi_draw_arrays_instanced_webgl);
FNDECL(w_multi_draw_elements_instanced_webgl);

FN(w_get_extension) {
	Isolate *iso = info.GetIsolate();
	if (info.Length() < 1 || !info[0]->IsString()) return;
	String::Utf8Value name_utf8(iso, info[0]);
	const char *name = *name_utf8;
	if (!name) return;
	Local<Context> c = cur(iso);
	auto make_obj_with = [&](std::initializer_list<std::pair<const char *, uint32_t>> kvs) {
		Local<Object> o = Object::New(iso);
		for (const auto &kv : kvs) {
			o->Set(c, String::NewFromUtf8(iso, kv.first).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, kv.second)).Check();
		}
		info.GetReturnValue().Set(o);
		// Ledger #67 — record the extension as enabled so gated getParameter
		// pnames start reporting real values (they return null + record
		// INVALID_ENUM until getExtension is called successfully).
		record_ext_enabled(name);
	};
	// Ledger #67 — same tracking for the ~12 empty-object success branches
	// below (feature-flag extensions with no constants exposed via the object).
	// Wrapping into a local helper keeps the return + track pair symmetric
	// with make_obj_with above.
	auto set_empty_obj = [&]() {
		info.GetReturnValue().Set(Object::New(iso));
		record_ext_enabled(name);
	};
	// WebGL1 extensions whose enum constants are numerically identical to
	// ES3 core enums — returning an object exposes the constants Three.js
	// queries; subsequent gl.<method>(EXT_CONST) calls reach native ES3
	// (which already implements the underlying capability) transparently.
	// No new engine GL plumbing needed.
	// Phase-1 batch-2 rider-2 — v2 spec-conformance prune. These
	// WebGL1-only extensions return null on v2 to match Chrome / Firefox
	// (Khronos registry marks them WebGL1-only; the underlying capability
	// is WebGL2 core). Coordinates with the removal from
	// `w_get_supported_extensions`'s v2 path. OES_texture_float_linear is
	// KEPT on v2 (still a genuine WebGL2 extension); EXT_texture_filter_
	// anisotropic is KEPT (advertised via the driver-gated block).
	// v1-only names (return null on v2) — Alex's Rider 2 explicit list.
	// Note: EXT_blend_minmax / OES_element_index_uint / EXT_sRGB are
	// ALSO WebGL1-only per registry but Rider 2 did NOT list them
	// explicitly; deferred to a follow-up spec-conformance sweep so
	// this commit matches the exact prune list Alex approved.
	// `v2` is computed here (early) rather than reusing the `v2` decl
	// deeper in the function so this guard runs before any WebGL1-only
	// branch. The batch-1 `const bool v2 = is_v2_context(info)` below
	// stays in place (its scope is the batch-1 branch block).
	const bool v2_rider2 = is_v2_context(info);
	if (v2_rider2 &&
	    (strcmp(name, "OES_standard_derivatives") == 0 ||
	     strcmp(name, "OES_texture_float") == 0 ||
	     strcmp(name, "OES_texture_half_float") == 0 ||
	     // Ledger #110 (2026-07-10) — `OES_texture_half_float_linear`
	     // REMOVED from the WebGL-2 prune list. Rider-2's original claim
	     // that this extension is "promoted to WebGL 2 core" is
	     // incorrect: WebGL 2 §5.30.1 still requires this extension to
	     // be enabled via `getExtension` for LINEAR filtering on any
	     // HALF_FLOAT texture (Chrome and Firefox both advertise it on
	     // their WebGL 2 contexts, matching the Khronos registry). When
	     // the extension is unavailable, sampling a HALF_FLOAT texture
	     // with LINEAR filter makes it "sampler-incomplete" (spec-
	     // mandated) and the texture read returns (0, 0, 0, 1) — Three.js's
	     // EffectComposer post-processing (RGBA16F ping-pong RTs with
	     // LINEAR filter by default) renders a pure-black tenant, cured
	     // by ungating this extension. Companion change in
	     // `w_get_supported_extensions` below.
	     strcmp(name, "WEBGL_depth_texture") == 0)) {
		info.GetReturnValue().SetNull();
		return;
	}
	if (strcmp(name, "EXT_blend_minmax") == 0) {
		make_obj_with({{"MIN_EXT", 0x8007}, {"MAX_EXT", 0x8008}});
		return;
	}
	if (strcmp(name, "OES_element_index_uint") == 0) {
		// No enum values — empty object is the standard signature. Indicates
		// "you may use gl.UNSIGNED_INT (0x1405) with drawElements", which
		// ES3 supports natively.
		set_empty_obj();
		return;
	}
	if (strcmp(name, "OES_standard_derivatives") == 0) {
		make_obj_with({{"FRAGMENT_SHADER_DERIVATIVE_HINT_OES", 0x8B8B}});
		return;
	}
	if (strcmp(name, "OES_texture_float") == 0 ||
	    strcmp(name, "OES_texture_float_linear") == 0 ||
	    strcmp(name, "OES_texture_half_float_linear") == 0) {
		set_empty_obj();
		return;
	}
	if (strcmp(name, "OES_texture_half_float") == 0) {
		// HALF_FLOAT_OES = 0x8D61 in WebGL1's extension, vs 0x140B (HALF_FLOAT)
		// in ES3 core. Three.js v1 code paths use the OES value. ES3 actually
		// accepts BOTH (the driver treats 0x8D61 + 0x140B as aliases on Mesa
		// Nouveau — confirmed in the fork's webgl_egl.c handling).
		make_obj_with({{"HALF_FLOAT_OES", 0x8D61}});
		return;
	}
	if (strcmp(name, "EXT_sRGB") == 0) {
		// Three.js sets texture.colorSpace = SRGB and queries this extension
		// on WebGL1 to know whether to use SRGB_EXT vs SRGB. The constants
		// map cleanly to ES3 (SRGB8 / SRGB8_ALPHA8 internalformats are core).
		make_obj_with({{"SRGB_EXT", 0x8C40},
		                {"SRGB_ALPHA_EXT", 0x8C42},
		                {"SRGB8_ALPHA8_EXT", 0x8C43},
		                {"FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT", 0x8210}});
		return;
	}
	if (strcmp(name, "WEBGL_depth_texture") == 0) {
		make_obj_with({{"UNSIGNED_INT_24_8_WEBGL", 0x84FA}});
		return;
	}
	if (strcmp(name, "OES_vertex_array_object") == 0) {
		// WebGL 1 route for the runtime pre-arm shim (RUNTIME_SHIMS #42).
		// ES3 core has glGenVertexArrays/glBindVertexArray/etc.; the v2
		// context registers them as native methods (webgl.cc line 2359+).
		// v1 has no core VAO API — this ext wires the SAME natives as
		// OES-suffixed methods so the pre-arm's `ext.createVertexArrayOES()
		// + ext.bindVertexArrayOES(vao)` reaches `w_bind_vertex_array`
		// which shadow-writes `user_snap.vao` per patch #36 contract.
		// VERTEX_ARRAY_BINDING_OES = 0x85B5 (ES3 core VERTEX_ARRAY_BINDING).
		Local<Object> o = Object::New(iso);
		o->Set(c, String::NewFromUtf8(iso, "VERTEX_ARRAY_BINDING_OES")
		              .ToLocalChecked(),
		       Uint32::NewFromUnsigned(iso, 0x85B5)).Check();
		o->Set(c, String::NewFromUtf8(iso, "createVertexArrayOES")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_create_vertex_array)
		              ->GetFunction(c).ToLocalChecked()).Check();
		o->Set(c, String::NewFromUtf8(iso, "bindVertexArrayOES")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_bind_vertex_array)
		              ->GetFunction(c).ToLocalChecked()).Check();
		o->Set(c, String::NewFromUtf8(iso, "deleteVertexArrayOES")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_delete_vertex_array)
		              ->GetFunction(c).ToLocalChecked()).Check();
		o->Set(c, String::NewFromUtf8(iso, "isVertexArrayOES")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_is_vertex_array)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// Phase-1 batch-1 — advertising + constants + compressed uploads.
	// Every branch here has a matching row in
	// `w_get_supported_extensions` (v1 or v2 gated per §1.2 of
	// docs/EXTENSION_PORT_PLAN.md). Ordered alphabetically within
	// category (EXT_* then OES_* then WEBGL_*) for grep-friendliness.
	const bool v2 = is_v2_context(info);
	// EXT_depth_clamp — enable-cap 0x864F, use via `gl.enable(DEPTH_CLAMP_EXT)`.
	if (has_native_ext("GL_EXT_depth_clamp") &&
	    strcmp(name, "EXT_depth_clamp") == 0) {
		make_obj_with({{"DEPTH_CLAMP_EXT", 0x864F}});
		return;
	}
	// EXT_float_blend — feature-flag stub (driver auto-enables blending
	// of float render targets when this ext is exposed).
	if (has_native_ext("GL_EXT_float_blend") &&
	    strcmp(name, "EXT_float_blend") == 0) {
		set_empty_obj();
		return;
	}
	// EXT_texture_filter_anisotropic — 2 constants; texParameteri accepts
	// TEXTURE_MAX_ANISOTROPY_EXT as pname (0x84FE), getParameter accepts
	// the MAX (0x84FF) via explicit branch added to `w_get_parameter`.
	if (has_native_ext("GL_EXT_texture_filter_anisotropic") &&
	    strcmp(name, "EXT_texture_filter_anisotropic") == 0) {
		make_obj_with({{"TEXTURE_MAX_ANISOTROPY_EXT", 0x84FE},
		                {"MAX_TEXTURE_MAX_ANISOTROPY_EXT", 0x84FF}});
		return;
	}
	// EXT_texture_compression_bptc — 4 sized internalformats accepted by
	// the new `compressedTexImage2D` native added in this batch.
	if (has_native_ext("GL_EXT_texture_compression_bptc") &&
	    strcmp(name, "EXT_texture_compression_bptc") == 0) {
		make_obj_with({{"COMPRESSED_RGBA_BPTC_UNORM_EXT", 0x8E8C},
		                {"COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT", 0x8E8D},
		                {"COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT", 0x8E8E},
		                {"COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT", 0x8E8F}});
		return;
	}
	// EXT_texture_compression_rgtc — 4 constants.
	if (has_native_ext("GL_EXT_texture_compression_rgtc") &&
	    strcmp(name, "EXT_texture_compression_rgtc") == 0) {
		make_obj_with({{"COMPRESSED_RED_RGTC1_EXT", 0x8DBB},
		                {"COMPRESSED_SIGNED_RED_RGTC1_EXT", 0x8DBC},
		                {"COMPRESSED_RED_GREEN_RGTC2_EXT", 0x8DBD},
		                {"COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT", 0x8DBE}});
		return;
	}
	// EXT_color_buffer_float (v2) — feature-flag stub. Driver-supplied
	// sized RGBA32F etc. render-target formats work through core ES3
	// `renderbufferStorage` / `texStorage2D` paths without an extension
	// method surface.
	if (v2 && has_native_ext("GL_EXT_color_buffer_float") &&
	    strcmp(name, "EXT_color_buffer_float") == 0) {
		set_empty_obj();
		return;
	}
	// EXT_color_buffer_half_float (v2) — feature-flag stub. Same
	// justification as EXT_color_buffer_float; the RGBA16F / RG16F /
	// R16F sized formats are core ES3.
	if (v2 && has_native_ext("GL_EXT_color_buffer_float") &&
	    strcmp(name, "EXT_color_buffer_half_float") == 0) {
		set_empty_obj();
		return;
	}
	// EXT_texture_norm16 (v2) — 8 sized internalformats. Constants only;
	// `texStorage2D` / `texImage2D` accept the values via ES3 core.
	if (v2 && has_native_ext("GL_EXT_texture_norm16") &&
	    strcmp(name, "EXT_texture_norm16") == 0) {
		make_obj_with({{"R16_EXT", 0x822A},
		                {"RG16_EXT", 0x822C},
		                {"RGB16_EXT", 0x8054},
		                {"RGBA16_EXT", 0x805B},
		                {"R16_SNORM_EXT", 0x8F98},
		                {"RG16_SNORM_EXT", 0x8F99},
		                {"RGB16_SNORM_EXT", 0x8F9A},
		                {"RGBA16_SNORM_EXT", 0x8F9B}});
		return;
	}
	// EXT_render_snorm (v2) — feature-flag stub. Snorm sized formats
	// accepted as render targets via ES3 core once the ext is exposed.
	if (v2 && has_native_ext("GL_EXT_render_snorm") &&
	    strcmp(name, "EXT_render_snorm") == 0) {
		set_empty_obj();
		return;
	}
	// WEBGL_color_buffer_float (v1) — v1 alias for the color-buffer-float
	// extension. Empty object per WebGL1 spec (no constants exposed by
	// the WEBGL1 form).
	if (!v2 && has_native_ext("GL_EXT_color_buffer_float") &&
	    strcmp(name, "WEBGL_color_buffer_float") == 0) {
		set_empty_obj();
		return;
	}
	// WEBGL_compressed_texture_s3tc — 4 constants.
	if (has_native_ext("GL_EXT_texture_compression_s3tc") &&
	    strcmp(name, "WEBGL_compressed_texture_s3tc") == 0) {
		make_obj_with({{"COMPRESSED_RGB_S3TC_DXT1_EXT", 0x83F0},
		                {"COMPRESSED_RGBA_S3TC_DXT1_EXT", 0x83F1},
		                {"COMPRESSED_RGBA_S3TC_DXT3_EXT", 0x83F2},
		                {"COMPRESSED_RGBA_S3TC_DXT5_EXT", 0x83F3}});
		return;
	}
	// WEBGL_compressed_texture_s3tc_srgb — 4 sRGB DXT constants.
	if (has_native_ext("GL_EXT_texture_compression_s3tc_srgb") &&
	    strcmp(name, "WEBGL_compressed_texture_s3tc_srgb") == 0) {
		make_obj_with({{"COMPRESSED_SRGB_S3TC_DXT1_EXT", 0x8C4C},
		                {"COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT", 0x8C4D},
		                {"COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT", 0x8C4E},
		                {"COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT", 0x8C4F}});
		return;
	}
	// WEBGL_compressed_texture_etc1 — 1 constant (ES3 core supports ETC2
	// which subsumes ETC1; the OES_compressed_ETC1_RGB8_texture native
	// token still gates advertising per Khronos WEBGL1 extension shape).
	if (has_native_ext("GL_OES_compressed_ETC1_RGB8_texture") &&
	    strcmp(name, "WEBGL_compressed_texture_etc1") == 0) {
		make_obj_with({{"COMPRESSED_RGB_ETC1_WEBGL", 0x8D64}});
		return;
	}
	// WEBGL_compressed_texture_astc — 28 constants (14 LDR + 14 sRGB
	// variants) + `getSupportedProfiles()` method returning the driver's
	// ASTC profile list ("ldr", "hdr", "sliced_3d" per driver tokens).
	if (has_native_ext("GL_KHR_texture_compression_astc_ldr") &&
	    strcmp(name, "WEBGL_compressed_texture_astc") == 0) {
		Local<Object> o = Object::New(iso);
		auto set_u = [&](const char *k, uint32_t v) {
			o->Set(c, String::NewFromUtf8(iso, k).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, v)).Check();
		};
		// LDR sized internalformats (block sizes 4x4..12x12).
		set_u("COMPRESSED_RGBA_ASTC_4x4_KHR", 0x93B0);
		set_u("COMPRESSED_RGBA_ASTC_5x4_KHR", 0x93B1);
		set_u("COMPRESSED_RGBA_ASTC_5x5_KHR", 0x93B2);
		set_u("COMPRESSED_RGBA_ASTC_6x5_KHR", 0x93B3);
		set_u("COMPRESSED_RGBA_ASTC_6x6_KHR", 0x93B4);
		set_u("COMPRESSED_RGBA_ASTC_8x5_KHR", 0x93B5);
		set_u("COMPRESSED_RGBA_ASTC_8x6_KHR", 0x93B6);
		set_u("COMPRESSED_RGBA_ASTC_8x8_KHR", 0x93B7);
		set_u("COMPRESSED_RGBA_ASTC_10x5_KHR", 0x93B8);
		set_u("COMPRESSED_RGBA_ASTC_10x6_KHR", 0x93B9);
		set_u("COMPRESSED_RGBA_ASTC_10x8_KHR", 0x93BA);
		set_u("COMPRESSED_RGBA_ASTC_10x10_KHR", 0x93BB);
		set_u("COMPRESSED_RGBA_ASTC_12x10_KHR", 0x93BC);
		set_u("COMPRESSED_RGBA_ASTC_12x12_KHR", 0x93BD);
		// sRGB variants.
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR", 0x93D0);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR", 0x93D1);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR", 0x93D2);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR", 0x93D3);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR", 0x93D4);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR", 0x93D5);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR", 0x93D6);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR", 0x93D7);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR", 0x93D8);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR", 0x93D9);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR", 0x93DA);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR", 0x93DB);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR", 0x93DC);
		set_u("COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR", 0x93DD);
		// `getSupportedProfiles()` — WebGL extension method returning
		// the driver's profile list. Uses driver token detection to
		// pick "ldr" (always if we're here), "hdr" (never on Mesa
		// Nouveau — token GL_KHR_texture_compression_astc_hdr absent
		// from the 134-list), and "sliced_3d" (gate on
		// GL_KHR_texture_compression_astc_sliced_3d).
		const bool has_sliced_3d =
		    has_native_ext("GL_KHR_texture_compression_astc_sliced_3d");
		// Static string list; V8 owns via FunctionTemplate data.
		auto profiles_cb = [](const FunctionCallbackInfo<Value> &pinfo) {
			Isolate *iso2 = pinfo.GetIsolate();
			Local<Context> c2 = cur(iso2);
			Local<Array> arr =
			    Array::New(iso2, pinfo.Data()->IsBoolean() &&
			                             pinfo.Data().As<Boolean>()->Value()
			                         ? 2
			                         : 1);
			arr->Set(c2, 0,
			         String::NewFromUtf8(iso2, "ldr").ToLocalChecked())
			    .Check();
			if (pinfo.Data()->IsBoolean() &&
			    pinfo.Data().As<Boolean>()->Value()) {
				arr->Set(c2, 1,
				         String::NewFromUtf8(iso2, "sliced_3d").ToLocalChecked())
				    .Check();
			}
			pinfo.GetReturnValue().Set(arr);
		};
		o->Set(c, String::NewFromUtf8(iso, "getSupportedProfiles")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, profiles_cb,
		                             Boolean::New(iso, has_sliced_3d))
		           ->GetFunction(c).ToLocalChecked())
		    .Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// WEBGL_debug_renderer_info — unmasked vendor / renderer via
	// glGetString(GL_VENDOR / GL_RENDERER) through the corresponding
	// `w_get_parameter` branches added in this batch. Constants exposed
	// here; parameter fetch is a follow-up call from the app.
	if (strcmp(name, "WEBGL_debug_renderer_info") == 0) {
		make_obj_with({{"UNMASKED_VENDOR_WEBGL", 0x9245},
		                {"UNMASKED_RENDERER_WEBGL", 0x9246}});
		return;
	}
	// WEBGL_stencil_texturing (v2) — feature-flag stub. Core ES3.1
	// TEXTURE_STENCIL_MODE pname is accepted by w_tex_parameteri
	// pass-through; no method surface.
	if (v2 && strcmp(name, "WEBGL_stencil_texturing") == 0) {
		set_empty_obj();
		return;
	}
	// Phase-1 batch-2 rider-1 — WEBGL_compressed_texture_etc (ETC2/EAC).
	// 10 sized internalformats accepted by ES3 core through the
	// `compressedTexImage2D` native wired in batch 1.
	if (strcmp(name, "WEBGL_compressed_texture_etc") == 0) {
		make_obj_with({
		    {"COMPRESSED_R11_EAC", 0x9270},
		    {"COMPRESSED_SIGNED_R11_EAC", 0x9271},
		    {"COMPRESSED_RG11_EAC", 0x9272},
		    {"COMPRESSED_SIGNED_RG11_EAC", 0x9273},
		    {"COMPRESSED_RGB8_ETC2", 0x9274},
		    {"COMPRESSED_SRGB8_ETC2", 0x9275},
		    {"COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2", 0x9276},
		    {"COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2", 0x9277},
		    {"COMPRESSED_RGBA8_ETC2_EAC", 0x9278},
		    {"COMPRESSED_SRGB8_ALPHA8_ETC2_EAC", 0x9279}});
		return;
	}
	// Phase-1 batch-2 — ANGLE_instanced_arrays (v1). Alias the ES3 core
	// natives already installed on v1's FUNCS[] (batch-2 registers
	// `drawArraysInstanced` / `drawElementsInstanced` / `vertexAttribDivisor`
	// on v1's install_methods table). The `-ANGLE`-suffixed extension
	// methods point at the SAME native handlers; JS-side round-trips
	// through the wrapped surface reach the same GL entry.
	if (!v2 && has_native_ext("GL_EXT_draw_instanced") &&
	    strcmp(name, "ANGLE_instanced_arrays") == 0) {
		Local<Object> o = Object::New(iso);
		o->Set(c, String::NewFromUtf8(iso,
		           "VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE").ToLocalChecked(),
		       Uint32::NewFromUnsigned(iso, 0x88FE)).Check();
		o->Set(c, String::NewFromUtf8(iso, "drawArraysInstancedANGLE")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_draw_arrays_instanced)
		              ->GetFunction(c).ToLocalChecked()).Check();
		o->Set(c, String::NewFromUtf8(iso, "drawElementsInstancedANGLE")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_draw_elements_instanced)
		              ->GetFunction(c).ToLocalChecked()).Check();
		o->Set(c, String::NewFromUtf8(iso, "vertexAttribDivisorANGLE")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_vertex_attrib_divisor)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// Phase-1 batch-2 — WEBGL_draw_buffers (v1). Aliases the v2 core
	// `drawBuffers` native (registered on v1's install_methods in this
	// batch). Constants + `drawBuffersWEBGL` method.
	if (!v2 && has_native_ext("GL_EXT_draw_buffers") &&
	    strcmp(name, "WEBGL_draw_buffers") == 0) {
		Local<Object> o = Object::New(iso);
		auto set_u = [&](const char *k, uint32_t v) {
			o->Set(c, String::NewFromUtf8(iso, k).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, v)).Check();
		};
		set_u("MAX_COLOR_ATTACHMENTS_WEBGL", 0x8CDF);
		set_u("MAX_DRAW_BUFFERS_WEBGL", 0x8824);
		// COLOR_ATTACHMENT{0..15}_WEBGL (0x8CE0..0x8CEF)
		for (int i = 0; i < 16; i++) {
			char k[40];
			snprintf(k, sizeof(k), "COLOR_ATTACHMENT%d_WEBGL", i);
			set_u(k, 0x8CE0 + i);
		}
		// DRAW_BUFFER{0..15}_WEBGL (0x8825..0x8834)
		for (int i = 0; i < 16; i++) {
			char k[40];
			snprintf(k, sizeof(k), "DRAW_BUFFER%d_WEBGL", i);
			set_u(k, 0x8825 + i);
		}
		o->Set(c, String::NewFromUtf8(iso, "drawBuffersWEBGL")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, w_draw_buffers)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// Phase-1.5-LOW rider — OES_fbo_render_mipmap (v1). Advertise-only
	// per spec: framebufferTexture2D with level > 0 is core ES3, so an
	// empty ext object is spec-legal + sufficient. Retires the batch-2
	// report's defect (row 15 of plan §2.6 v1 table was claimed done but
	// not actually advertised).
	if (!v2 && has_native_ext("GL_OES_fbo_render_mipmap") &&
	    strcmp(name, "OES_fbo_render_mipmap") == 0) {
		set_empty_obj();
		return;
	}
	// Phase-1 batch-2 — EXT_frag_depth (v1). Advertise-only per spec
	// (enables `#extension GL_EXT_frag_depth : enable` + `gl_FragDepthEXT`
	// writes in #version 100 shaders). Empty object per Khronos.
	// Compile-probe-gated at advertising time above.
	if (!v2 && has_native_ext("GL_EXT_frag_depth") &&
	    probe_ext_frag_depth() &&
	    strcmp(name, "EXT_frag_depth") == 0) {
		set_empty_obj();
		return;
	}
	// Phase-1 batch-2 — WEBGL_lose_context (both). Software-only,
	// minimal impl: `loseContext` / `restoreContext` are no-ops; the
	// caller can still call them without hitting undefined-method
	// TypeErrors. `isContextLost` piggybacks on the existing
	// `w_is_context_lost` (returns false since brewser never actually
	// loses the context under the shared-EGL bracket contract).
	// Event dispatch (webglcontextlost/restored on the canvas element)
	// is out of scope; if a future demo needs it, a runtime shim can
	// wrap this ext object and intercept the methods.
	if (strcmp(name, "WEBGL_lose_context") == 0) {
		auto noop_cb = [](const FunctionCallbackInfo<Value> &) {};
		Local<Object> o = Object::New(iso);
		o->Set(c, String::NewFromUtf8(iso, "loseContext").ToLocalChecked(),
		       FunctionTemplate::New(iso, noop_cb)
		              ->GetFunction(c).ToLocalChecked()).Check();
		o->Set(c, String::NewFromUtf8(iso, "restoreContext").ToLocalChecked(),
		       FunctionTemplate::New(iso, noop_cb)
		              ->GetFunction(c).ToLocalChecked()).Check();
		o->Set(c, String::NewFromUtf8(iso, "isContextLost").ToLocalChecked(),
		       FunctionTemplate::New(iso, w_is_context_lost)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// Phase-1 batch-2 — WEBGL_debug_shaders (both). Software-only.
	// `getTranslatedShaderSource(shader)` returns the source string
	// verbatim as submitted via `shaderSource(shader, source)`. This
	// stack does no WebGL→ES3 shader translation (unlike ANGLE); the
	// "translated" and "submitted" source are the same. Matches Mesa's
	// GL_ARB_debug_shaders convention.
	if (strcmp(name, "WEBGL_debug_shaders") == 0) {
		auto get_translated_cb = [](const FunctionCallbackInfo<Value> &pinfo) {
			Isolate *iso2 = pinfo.GetIsolate();
			enter_bracket();
			if (pinfo.Length() < 1) return;
			GLObj *o = get_gl_obj(pinfo[0]);
			if (!o || o->kind != K_SHADER) return;
			GLint len = 0;
			glGetShaderiv(o->id, GL_SHADER_SOURCE_LENGTH, &len);
			if (len <= 0) {
				pinfo.GetReturnValue().Set(
				    String::NewFromUtf8(iso2, "").ToLocalChecked());
				return;
			}
			std::vector<char> buf((size_t)len);
			GLsizei got = 0;
			glGetShaderSource(o->id, len, &got, buf.data());
			pinfo.GetReturnValue().Set(
			    String::NewFromUtf8(iso2, buf.data(), NewStringType::kNormal,
			                        (int)got).ToLocalChecked());
		};
		Local<Object> o = Object::New(iso);
		o->Set(c, String::NewFromUtf8(iso, "getTranslatedShaderSource")
		              .ToLocalChecked(),
		       FunctionTemplate::New(iso, get_translated_cb)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// ============================================================
	// Batch 3 (ledger #57) — final extension batch branches.
	// Each branch: driver-gated + returns an object with the
	// extension's constants + method surface (methods alias the
	// prototype-installed FUNCS[] entries via the same underlying
	// natives; installing them on the ext object mirrors the shape
	// Chrome/Firefox return so Three.js's ext.methodEXT and
	// gl.methodEXT both work).
	// ============================================================
	if (strcmp(name, "EXT_clip_control") == 0 &&
	    has_native_ext("GL_EXT_clip_control")) {
		Local<Object> o = Object::New(iso);
		auto U = [&](const char *n, uint32_t v) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, v)).Check();
		};
		U("LOWER_LEFT_EXT",           0x8CA1);
		U("UPPER_LEFT_EXT",           0x8CA2);
		U("NEGATIVE_ONE_TO_ONE_EXT",  0x935E);
		U("ZERO_TO_ONE_EXT",          0x935F);
		U("CLIP_ORIGIN_EXT",          0x935C);
		U("CLIP_DEPTH_MODE_EXT",      0x935D);
		o->Set(c, String::NewFromUtf8(iso, "clipControlEXT").ToLocalChecked(),
		       FunctionTemplate::New(iso, w_clip_control_ext)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	if (strcmp(name, "EXT_polygon_offset_clamp") == 0 &&
	    has_native_ext("GL_EXT_polygon_offset_clamp")) {
		Local<Object> o = Object::New(iso);
		o->Set(c, String::NewFromUtf8(iso, "POLYGON_OFFSET_CLAMP_EXT").ToLocalChecked(),
		       Uint32::NewFromUnsigned(iso, 0x8E1B)).Check();
		o->Set(c, String::NewFromUtf8(iso, "polygonOffsetClampEXT").ToLocalChecked(),
		       FunctionTemplate::New(iso, w_polygon_offset_clamp_ext)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// EXT_disjoint_timer_query (v1) + _webgl2 (v2). v1 exposes the whole
	// query lifecycle with EXT suffix (aliasing the v2 core natives from
	// #53); v2 exposes only queryCounterEXT + timer constants (uses v2
	// core query surface for lifecycle).
	if ((strcmp(name, "EXT_disjoint_timer_query") == 0 && !v2) ||
	    (strcmp(name, "EXT_disjoint_timer_query_webgl2") == 0 && v2)) {
		if (!has_native_ext("GL_EXT_disjoint_timer_query")) {
			info.GetReturnValue().SetNull();
			return;
		}
		Local<Object> o = Object::New(iso);
		auto U = [&](const char *n, uint32_t v) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, v)).Check();
		};
		U("QUERY_COUNTER_BITS_EXT",      0x8864);
		U("CURRENT_QUERY_EXT",            0x8865);
		U("QUERY_RESULT_EXT",             0x8866);
		U("QUERY_RESULT_AVAILABLE_EXT",   0x8867);
		U("TIME_ELAPSED_EXT",             0x88BF);
		U("TIMESTAMP_EXT",                0x8E28);
		U("GPU_DISJOINT_EXT",             0x8FBB);
		auto FN_ = [&](const char *n, FunctionCallback fn) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       FunctionTemplate::New(iso, fn)
			              ->GetFunction(c).ToLocalChecked()).Check();
		};
		if (!v2) {
			FN_("createQueryEXT",  w_create_query);
			FN_("deleteQueryEXT",  w_delete_query);
			FN_("isQueryEXT",      w_is_query);
			FN_("beginQueryEXT",   w_begin_query);
			FN_("endQueryEXT",     w_end_query);
			FN_("getQueryEXT",     w_get_query);
			FN_("getQueryObjectEXT", w_get_query_parameter);
		}
		FN_("queryCounterEXT", w_query_counter_ext);
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// OES_draw_buffers_indexed (v2 only).
	if (v2 && strcmp(name, "OES_draw_buffers_indexed") == 0 &&
	    (has_native_ext("GL_OES_draw_buffers_indexed") ||
	     has_native_ext("GL_EXT_draw_buffers_indexed"))) {
		Local<Object> o = Object::New(iso);
		auto FN_ = [&](const char *n, FunctionCallback fn) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       FunctionTemplate::New(iso, fn)
			              ->GetFunction(c).ToLocalChecked()).Check();
		};
		FN_("enableiOES",                 w_enable_i);
		FN_("disableiOES",                w_disable_i);
		FN_("blendEquationiOES",          w_blend_equation_i);
		FN_("blendEquationSeparateiOES",  w_blend_equation_separate_i);
		FN_("blendFunciOES",              w_blend_func_i);
		FN_("blendFuncSeparateiOES",      w_blend_func_separate_i);
		FN_("colorMaskiOES",              w_color_mask_i);
		FN_("isEnablediOES",              w_is_enabled_i);
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	if (strcmp(name, "KHR_parallel_shader_compile") == 0 &&
	    has_native_ext("GL_KHR_parallel_shader_compile")) {
		Local<Object> o = Object::New(iso);
		o->Set(c, String::NewFromUtf8(iso, "COMPLETION_STATUS_KHR").ToLocalChecked(),
		       Uint32::NewFromUnsigned(iso, 0x91B1)).Check();
		o->Set(c, String::NewFromUtf8(iso, "maxShaderCompilerThreadsKHR").ToLocalChecked(),
		       FunctionTemplate::New(iso, w_max_shader_compiler_threads_khr)
		              ->GetFunction(c).ToLocalChecked()).Check();
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	if (strcmp(name, "WEBGL_blend_func_extended") == 0 &&
	    has_native_ext("GL_EXT_blend_func_extended")) {
		Local<Object> o = Object::New(iso);
		auto U = [&](const char *n, uint32_t v) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, v)).Check();
		};
		U("SRC1_COLOR_WEBGL",                     0x88F9);
		U("SRC1_ALPHA_WEBGL",                     0x8589);
		U("ONE_MINUS_SRC1_COLOR_WEBGL",           0x88FA);
		U("ONE_MINUS_SRC1_ALPHA_WEBGL",           0x88FB);
		U("MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL",   0x88FC);
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// Ledger #87 — WEBGL_multi_draw's Khronos spec REQUIRES gl_DrawID in
	// the vertex shader. Mesa Nouveau on Tegra exposes GL_EXT_multi_draw_
	// arrays (fine for the C-side loop shims below) but NOT
	// GL_ANGLE_multi_draw (which is where gl_DrawID actually lives). Gating
	// on both keeps our advertisement honest: only expose the extension
	// when the driver can back it end-to-end. Pre-#87, we advertised on
	// EXT alone and any test with `#extension GL_ANGLE_multi_draw :
	// require` failed shader compile → 7,728 downstream pixel-check FAILs
	// in the conformance corpus.
	if (strcmp(name, "WEBGL_multi_draw") == 0 &&
	    has_native_ext("GL_EXT_multi_draw_arrays") &&
	    has_native_ext("GL_ANGLE_multi_draw")) {
		Local<Object> o = Object::New(iso);
		auto FN_ = [&](const char *n, FunctionCallback fn) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       FunctionTemplate::New(iso, fn)
			              ->GetFunction(c).ToLocalChecked()).Check();
		};
		FN_("multiDrawArraysWEBGL",              w_multi_draw_arrays_webgl);
		FN_("multiDrawElementsWEBGL",            w_multi_draw_elements_webgl);
		FN_("multiDrawArraysInstancedWEBGL",     w_multi_draw_arrays_instanced_webgl);
		FN_("multiDrawElementsInstancedWEBGL",   w_multi_draw_elements_instanced_webgl);
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	// v2-only advertise-only stubs — constants + feature-flag empty objects.
	if (v2 && strcmp(name, "WEBGL_clip_cull_distance") == 0 &&
	    has_native_ext("GL_EXT_clip_cull_distance")) {
		Local<Object> o = Object::New(iso);
		auto U = [&](const char *n, uint32_t v) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, v)).Check();
		};
		U("MAX_CLIP_DISTANCES_WEBGL",                     0x0D32);
		U("MAX_CULL_DISTANCES_WEBGL",                     0x82F9);
		U("MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL",   0x82FA);
		U("CLIP_DISTANCE0_WEBGL", 0x3000);
		U("CLIP_DISTANCE1_WEBGL", 0x3001);
		U("CLIP_DISTANCE2_WEBGL", 0x3002);
		U("CLIP_DISTANCE3_WEBGL", 0x3003);
		U("CLIP_DISTANCE4_WEBGL", 0x3004);
		U("CLIP_DISTANCE5_WEBGL", 0x3005);
		U("CLIP_DISTANCE6_WEBGL", 0x3006);
		U("CLIP_DISTANCE7_WEBGL", 0x3007);
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}
	if (v2 && strcmp(name, "OES_sample_variables") == 0 &&
	    has_native_ext("GL_OES_sample_variables")) {
		set_empty_obj();
		return;
	}
	if (v2 && strcmp(name, "OES_shader_multisample_interpolation") == 0 &&
	    has_native_ext("GL_OES_shader_multisample_interpolation")) {
		Local<Object> o = Object::New(iso);
		auto U = [&](const char *n, uint32_t v) {
			o->Set(c, String::NewFromUtf8(iso, n).ToLocalChecked(),
			       Uint32::NewFromUnsigned(iso, v)).Check();
		};
		U("MIN_FRAGMENT_INTERPOLATION_OFFSET_OES",  0x8E5B);
		U("MAX_FRAGMENT_INTERPOLATION_OFFSET_OES",  0x8E5C);
		U("FRAGMENT_INTERPOLATION_OFFSET_BITS_OES", 0x8E5D);
		info.GetReturnValue().Set(o);
		record_ext_enabled(name);
		return;
	}

	// Everything else: not advertised yet. Return null (the spec value for
	// "extension not supported"). 2.E will widen this list as the slice
	// demos hit it.
	info.GetReturnValue().SetNull();
}
FN(w_get_supported_extensions) {
	Isolate *iso = info.GetIsolate();
	Local<Context> c = cur(iso);
	const bool v2 = is_v2_context(info);
	// Phase-1 batch-1 — driver-probed advertisement rebuild. Retires the
	// shared static `SUPPORTED[9]` (both context types identical) that
	// shipped through phase-0. Blueprint: pre-migration
	// [nxjs-source/source/webgl.c:2005-2235] (14 always-on statics +
	// ~27 driver-probed `nx_webgl_egl_has_*` gates + v1/v2 branches).
	// This build is the same model on the V8 stack: `has_native_ext()`
	// checks the enumeration populated by ledger #43's
	// `populate_native_extensions()`. Every gated entry has a
	// corresponding branch in `w_get_extension` — the two are
	// consistent by construction.
	std::vector<const char *> out;
	// Phase-1 batch-2 rider-2 — v2 spec-conformance prune. The Khronos
	// registry lists OES_standard_derivatives / OES_texture_float /
	// OES_texture_half_float / OES_texture_half_float_linear /
	// WEBGL_depth_texture as WebGL1-only (their functionality is
	// promoted to WebGL2 core). Chrome and Firefox return null for
	// these on v2 contexts; Brewser now matches. Kept on v2 unchanged:
	// OES_texture_float_linear (a genuine WebGL2 extension per registry).
	// EXT_texture_filter_anisotropic (kept unchanged — advertised by
	// the driver-gated block below, not affected by this prune).
	// Retention behavior for Three.js: `capabilities.floatTextureType`
	// falls back to gl.HALF_FLOAT / gl.FLOAT ES3 core paths cleanly
	// when the pruned extensions are absent; no manual override needed.
	// Guard: batch-2 Citron smoke must include 2-3 curated 13-suite
	// demos on the v2 path; revert this commit alone if any regresses.
	if (!v2) {
		out.push_back("OES_standard_derivatives");
		out.push_back("OES_texture_float");
		out.push_back("OES_texture_half_float");
		out.push_back("WEBGL_depth_texture");
	}
	// KEPT on both: OES_texture_float_linear (registry keeps this as a
	// v2 ext; FLOAT texture sampling with linear filtering is not core).
	out.push_back("OES_texture_float_linear");
	// Ledger #110 (2026-07-10) — KEPT on both: OES_texture_half_float_linear.
	// Same rationale as OES_texture_float_linear: WebGL 2 §5.30.1 still
	// requires this extension to be enabled for LINEAR filtering on
	// HALF_FLOAT textures (rider-2's claim that it was promoted to WebGL 2
	// core was incorrect). Un-pruned to fix Three.js EffectComposer's
	// black-output on RGBA16F ping-pong RTs.
	out.push_back("OES_texture_half_float_linear");
	out.push_back("WEBGL_debug_renderer_info");  // batch-1
	// v1-only statics (v2 has them as core).
	if (!v2) {
		out.push_back("EXT_blend_minmax");
		out.push_back("OES_element_index_uint");
		out.push_back("EXT_sRGB");
	}
	// Batch-1 driver-probed advertising.
	if (has_native_ext("GL_EXT_depth_clamp"))
		out.push_back("EXT_depth_clamp");
	if (has_native_ext("GL_EXT_float_blend"))
		out.push_back("EXT_float_blend");
	if (has_native_ext("GL_EXT_texture_filter_anisotropic"))
		out.push_back("EXT_texture_filter_anisotropic");
	if (has_native_ext("GL_EXT_texture_compression_bptc"))
		out.push_back("EXT_texture_compression_bptc");
	if (has_native_ext("GL_EXT_texture_compression_rgtc"))
		out.push_back("EXT_texture_compression_rgtc");
	if (has_native_ext("GL_EXT_texture_compression_s3tc"))
		out.push_back("WEBGL_compressed_texture_s3tc");
	if (has_native_ext("GL_EXT_texture_compression_s3tc_srgb"))
		out.push_back("WEBGL_compressed_texture_s3tc_srgb");
	if (has_native_ext("GL_OES_compressed_ETC1_RGB8_texture"))
		out.push_back("WEBGL_compressed_texture_etc1");
	if (has_native_ext("GL_KHR_texture_compression_astc_ldr"))
		out.push_back("WEBGL_compressed_texture_astc");
	// v1-only batch-1: WEBGL_color_buffer_float (v2 exposes via
	// EXT_color_buffer_float per spec).
	if (!v2 && has_native_ext("GL_EXT_color_buffer_float"))
		out.push_back("WEBGL_color_buffer_float");
	// v2-only batch-1: color-buffer float/half + norm16 + render_snorm +
	// stencil_texturing (core ES3.1).
	if (v2 && has_native_ext("GL_EXT_color_buffer_float")) {
		out.push_back("EXT_color_buffer_float");
		out.push_back("EXT_color_buffer_half_float");
	}
	if (v2 && has_native_ext("GL_EXT_texture_norm16"))
		out.push_back("EXT_texture_norm16");
	if (v2 && has_native_ext("GL_EXT_render_snorm"))
		out.push_back("EXT_render_snorm");
	if (v2)
		out.push_back("WEBGL_stencil_texturing");  // ES3.1 core
	// Phase-1 batch-2 rider-1 — WEBGL_compressed_texture_etc (ETC2/EAC).
	// Bucket A: core ES3 compressed formats (no driver-token gate; ES3
	// spec guarantees). Batch 1 skipped this by OVERSIGHT — the plan's
	// §4.1 add-list correctly included it, but the batch-1
	// implementation only wired ETC1 (via GL_OES_compressed_ETC1_RGB8_
	// texture). Batch 2 closes the omission.
	out.push_back("WEBGL_compressed_texture_etc");
	// Phase-1 batch-2 — Unity-P1 v1 function surfaces.
	if (!v2 && has_native_ext("GL_EXT_draw_instanced"))
		out.push_back("ANGLE_instanced_arrays");
	if (!v2 && has_native_ext("GL_EXT_draw_buffers"))
		out.push_back("WEBGL_draw_buffers");
	// #42 list-flip: OES_vertex_array_object was intentionally list-hidden
	// pre-batch-2 (see [NXJS_PATCHES_NEEDED.md #42] rationale). Batch 2
	// retires the asymmetry: advertise on v1 so `getSupportedExtensions`
	// and `getExtension` agree. The runtime pre-arm route still calls
	// `getExtension('OES_vertex_array_object')` by name; that path is
	// unaffected (the vended object is the same). Three.js path-change
	// risk documented in plan §1.3 — suite guard on this batch's smoke.
	if (!v2 && has_native_ext("GL_OES_vertex_array_object"))
		out.push_back("OES_vertex_array_object");
	// Phase-1.5-LOW rider — OES_fbo_render_mipmap on v1 (batch-2 report
	// defect; plan §2.6 row 15 = A, batch 2). Bucket A: `framebufferTexture2D`
	// with `level > 0` is core ES3, so advertising alone unblocks Three.js's
	// v1 capability probe. No engine plumbing needed beyond the ext-object
	// branch below.
	if (!v2 && has_native_ext("GL_OES_fbo_render_mipmap"))
		out.push_back("OES_fbo_render_mipmap");
	// EXT_frag_depth (v1): compile-probe-gated per plan §2.4. Native
	// token IS in the 134-list, but only advertise if ESSL-100 accepts
	// the extension directive. Probe result cached; runs at most once.
	if (!v2 && has_native_ext("GL_EXT_frag_depth") &&
	    probe_ext_frag_depth())
		out.push_back("EXT_frag_depth");
	// WEBGL_lose_context (both) — software-only per plan §2.1. Engine
	// vends a spec-legal minimal object (no-op loseContext/restoreContext
	// + isContextLost returning false). Event dispatch on canvas element
	// is out of scope for a minimal impl; adding events later goes in a
	// runtime shim without changing this advertising row.
	out.push_back("WEBGL_lose_context");
	// WEBGL_debug_shaders (both) — software-only. `getTranslatedShaderSource`
	// returns the source as submitted (this stack does no WebGL→ES3
	// translation), consistent with the Mesa GL_ARB_debug_shaders impl.
	out.push_back("WEBGL_debug_shaders");

	// Batch 3 (ledger #57) — final extension batch. All driver-gated
	// against the enumeration populated at bridge init (ledger #43).
	if (has_native_ext("GL_EXT_clip_control"))
		out.push_back("EXT_clip_control");
	if (has_native_ext("GL_EXT_polygon_offset_clamp"))
		out.push_back("EXT_polygon_offset_clamp");
	if (has_native_ext("GL_KHR_parallel_shader_compile"))
		out.push_back("KHR_parallel_shader_compile");
	// Ledger #87 — WEBGL_multi_draw requires gl_DrawID via GL_ANGLE_multi_draw;
	// GL_EXT_multi_draw_arrays alone is not spec-conforming.
	if (has_native_ext("GL_EXT_multi_draw_arrays") &&
	    has_native_ext("GL_ANGLE_multi_draw"))
		out.push_back("WEBGL_multi_draw");
	if (has_native_ext("GL_EXT_blend_func_extended"))
		out.push_back("WEBGL_blend_func_extended");
	if (has_native_ext("GL_EXT_disjoint_timer_query")) {
		if (v2) out.push_back("EXT_disjoint_timer_query_webgl2");
		else    out.push_back("EXT_disjoint_timer_query");
	}
	// v2-only b3 items
	if (v2 && (has_native_ext("GL_OES_draw_buffers_indexed") ||
	           has_native_ext("GL_EXT_draw_buffers_indexed")))
		out.push_back("OES_draw_buffers_indexed");
	if (v2 && has_native_ext("GL_EXT_clip_cull_distance"))
		out.push_back("WEBGL_clip_cull_distance");
	if (v2 && has_native_ext("GL_OES_sample_variables"))
		out.push_back("OES_sample_variables");
	if (v2 && has_native_ext("GL_OES_shader_multisample_interpolation"))
		out.push_back("OES_shader_multisample_interpolation");

	Local<Array> arr = Array::New(iso, (int)out.size());
	for (size_t i = 0; i < out.size(); i++) {
		arr->Set(c, (uint32_t)i,
		         String::NewFromUtf8(iso, out[i]).ToLocalChecked()).Check();
	}
	info.GetReturnValue().Set(arr);
}

// Phase-0 gap fill — restore native GL extension visibility the QuickJS-
// era engine exposed via `gl.getBackendInfo().glExtensions` and which the
// V8 migration dropped. Internal-only native (leading underscore names
// signal "shim consumers only, not spec surface"). Returns the joined
// space-separated native GL extension string (sorted alphabetically) so
// the brewser-runtime getBackendInfo shim can pass it through unchanged
// to com.natureglass.webglreport, which splits it back into tokens at
// [webglreport.js:199]. See WEBGL_EXTENSION_GAP.md §Broken-introspection A
// and phase-0 ledger entry.
FN(w_get_native_extensions_string) {
	populate_native_extensions();
	Isolate *iso = info.GetIsolate();
	info.GetReturnValue().Set(
	    String::NewFromUtf8(iso, s_native_exts_joined.c_str(),
	                        NewStringType::kNormal,
	                        (int)s_native_exts_joined.size())
	        .ToLocalChecked());
}

// Phase-0 gap fill — EGL version string for the getBackendInfo shim's
// `eglMajor`/`eglMinor` fields (matches pre-migration schema at
// [nxjs-source/source/webgl_egl.c:8433-8509]). Uses eglQueryString
// (idempotent, no re-init) on the display Skia already brought up.
// Returned form: "major.minor" (e.g. "1.5"); trailing vendor blob per
// EGL spec is stripped so the shim's parse is a single split-on-dot.
// Empty string means the display isn't up yet (never happens at
// getContext time; guard is defensive).
FN(w_get_egl_version) {
	Isolate *iso = info.GetIsolate();
	EGLDisplay dpy = nx_skia_gpu_egl_display();
	const char *raw = dpy ? eglQueryString(dpy, EGL_VERSION) : nullptr;
	std::string out;
	if (raw) {
		const char *end = raw;
		while (*end && *end != ' ' && *end != '\t') end++;
		out.assign(raw, (size_t)(end - raw));
	}
	info.GetReturnValue().Set(
	    String::NewFromUtf8(iso, out.c_str(), NewStringType::kNormal,
	                        (int)out.size())
	        .ToLocalChecked());
}

FN(w_is_context_lost) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), false));
}
FN(w_get_context_attributes) {
	Isolate *iso = info.GetIsolate();
	Local<Context> c = cur(iso);
	Local<Object> o = Object::New(iso);
	auto setb = [&](const char *k, bool v) {
		o->Set(c, String::NewFromUtf8(iso, k).ToLocalChecked(),
		       Boolean::New(iso, v)).Check();
	};
	setb("alpha", true);
	setb("antialias", false);
	setb("depth", true);
	setb("stencil", true);
	setb("premultipliedAlpha", true);
	setb("preserveDrawingBuffer", false);
	setb("desynchronized", false);
	setb("failIfMajorPerformanceCaveat", false);
	o->Set(c, String::NewFromUtf8(iso, "powerPreference").ToLocalChecked(),
	       String::NewFromUtf8(iso, "default").ToLocalChecked()).Check();
	info.GetReturnValue().Set(o);
}

FN(w_get_shader_precision_format) {
	Isolate *iso = info.GetIsolate();
	GLint range[2] = {0, 0};
	GLint precision = 0;
	glGetShaderPrecisionFormat(a_u32(info, 0), a_u32(info, 1), range, &precision);
	Local<Object> obj = new_gl_obj(iso, K_SHADER_PRECISION_FORMAT, 0);
	Local<Context> c = cur(iso);
	obj->Set(c, String::NewFromUtf8(iso, "rangeMin").ToLocalChecked(),
	         Int32::New(iso, range[0])).Check();
	obj->Set(c, String::NewFromUtf8(iso, "rangeMax").ToLocalChecked(),
	         Int32::New(iso, range[1])).Check();
	obj->Set(c, String::NewFromUtf8(iso, "precision").ToLocalChecked(),
	         Int32::New(iso, precision)).Check();
	info.GetReturnValue().Set(obj);
}

// ----- Shader -----
FN(w_create_shader) {
	enter_bracket();
	GLuint s = glCreateShader(a_u32(info, 0));
	info.GetReturnValue().Set(new_gl_obj(info.GetIsolate(), K_SHADER, s));
}
FN(w_delete_shader) {
	GLuint id = obj_id(info[0]);
	if (id) {
		glDeleteShader(id);
		erase_wrapper_cache(K_SHADER, id);
	}
}
FN(w_is_shader) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsShader(obj_id(info[0])) == GL_TRUE));
}

// Phase 2.F.1: PMREMGGXConvolution FS replacement. Mesa-Nouveau on Tegra X1
// aborts glDrawArrays when the original Three.js r184 PMREM convolution FS
// (uint + bitwise Hammersley radicalInverse_VdC in the GGX importance-sample
// loop) is compiled, even when the loop is statically unreachable. The
// replacement keeps the cubeUV math (getFace, getUV, inline bilinearCubeUV)
// but uses a 5-tap unrolled cross blur using a tangent/bitangent basis. No
// uint, no bitwise, no for, no sin/cos, no early-return conditional — every
// construct that crashed an iteration during the QuickJS-era 13-iteration
// bisect is excluded. Verbatim re-port from
// nxjs-source/source/webgl_egl.c:8629-8800 per
// [[reference-pmrem-tegra-compiler-workaround]].
//
// Per-pass roughness scales the kernel radius; across PMREM's cumulative
// passes this produces a widening hemisphere coverage in the cube_uv mip
// chain — visually inferior to upstream's roughness-based GGX blur (perfect-
// mirror-ish look at all roughness levels) but the only way to get PMREM
// through Tegra/Mesa GLES without driver abort.
//
// Texel constants (CUBEUV_MAX_MIP / CUBEUV_TEXEL_WIDTH / CUBEUV_TEXEL_HEIGHT)
// are PARSED from the Three.js-emitted source — hardcoding them only matched
// cubeSize=256 (lodMax=8); for 2048-wide equirects → cubeSize=512 → lodMax=9
// the hardcoded constants put UVs outside the texture and multi-tap writes
// crashed Mesa.
//
// Gate: shader_type == GL_FRAGMENT_SHADER AND source contains
// "PMREMGGXConvolution". Both conditions must hold; the substring guard
// alone is r184-PMREM-specific (Three.js r162's PMREM FS doesn't contain
// this name) and the type guard makes accidental match in a vertex shader
// (extremely unlikely) safe.
//
// Returns a heap-allocated replacement source (caller delete[]) when the
// substitution fires, or nullptr when the source should be passed through
// unchanged.
static char *maybe_replace_pmrem_fs(GLuint shader, const char *src) {
	GLint shader_type = 0;
	glGetShaderiv(shader, GL_SHADER_TYPE, &shader_type);
	if (shader_type != GL_FRAGMENT_SHADER) return nullptr;
	if (strstr(src, "PMREMGGXConvolution") == nullptr) return nullptr;

	double max_mip = 8.0;
	double texel_w = 1.0 / 1536.0;
	double texel_h = 1.0 / 2048.0;
	if (const char *p = strstr(src, "#define CUBEUV_MAX_MIP ")) {
		max_mip = strtod(p + strlen("#define CUBEUV_MAX_MIP "), nullptr);
	}
	if (const char *p = strstr(src, "#define CUBEUV_TEXEL_WIDTH ")) {
		texel_w = strtod(p + strlen("#define CUBEUV_TEXEL_WIDTH "), nullptr);
	}
	if (const char *p = strstr(src, "#define CUBEUV_TEXEL_HEIGHT ")) {
		texel_h = strtod(p + strlen("#define CUBEUV_TEXEL_HEIGHT "), nullptr);
	}

	char *built = new char[16384];
	const int n = snprintf(built, 16384,
		"#version 300 es\n"
		"precision highp float;\n"
		"precision highp sampler2D;\n"
		"in vec3 vOutputDirection;\n"
		"layout(location = 0) out highp vec4 pc_fragColor;\n"
		"uniform sampler2D envMap;\n"
		"uniform float roughness;\n"
		"uniform float mipInt;\n"
		"#define cubeUV_minMipLevel 4.0\n"
		"#define cubeUV_minTileSize 16.0\n"
		"#define CUBEUV_TEXEL_WIDTH %.10f\n"
		"#define CUBEUV_TEXEL_HEIGHT %.10f\n"
		"#define CUBEUV_MAX_MIP %.4f\n"
		"float getFace(vec3 d) {\n"
		"  vec3 a = abs(d);\n"
		"  if (a.x > a.z) {\n"
		"    if (a.x > a.y) return d.x > 0.0 ? 0.0 : 3.0;\n"
		"    return d.y > 0.0 ? 1.0 : 4.0;\n"
		"  }\n"
		"  if (a.z > a.y) return d.z > 0.0 ? 2.0 : 5.0;\n"
		"  return d.y > 0.0 ? 1.0 : 4.0;\n"
		"}\n"
		"vec2 getUV(vec3 d, float f) {\n"
		"  vec2 uv;\n"
		"  if (f == 0.0) uv = vec2(d.z, d.y) / abs(d.x);\n"
		"  else if (f == 1.0) uv = vec2(-d.x, -d.z) / abs(d.y);\n"
		"  else if (f == 2.0) uv = vec2(-d.x, d.y) / abs(d.z);\n"
		"  else if (f == 3.0) uv = vec2(-d.z, d.y) / abs(d.x);\n"
		"  else if (f == 4.0) uv = vec2(-d.x, d.z) / abs(d.y);\n"
		"  else uv = vec2(d.x, d.y) / abs(d.z);\n"
		"  return 0.5 * (uv + 1.0);\n"
		"}\n"
		"void main() {\n"
		"  vec3 N = normalize(vOutputDirection);\n"
		"  vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);\n"
		"  vec3 T = normalize(cross(up, N));\n"
		"  vec3 B = cross(N, T);\n"
		"  float blur = roughness * 5.0;\n"
		"  vec3 D0 = N;\n"
		"  vec3 D1 = normalize(N + T * blur);\n"
		"  vec3 D2 = normalize(N - T * blur);\n"
		"  vec3 D3 = normalize(N + B * blur);\n"
		"  vec3 D4 = normalize(N - B * blur);\n"
		"  vec3 acc = vec3(0.0);\n"
		"  { vec3 d = D0; float mi = mipInt; float face = getFace(d);\n"
		"    float fi = max(cubeUV_minMipLevel - mi, 0.0);\n"
		"    mi = max(mi, cubeUV_minMipLevel); float fs = exp2(mi);\n"
		"    vec2 uv = getUV(d, face) * (fs - 2.0) + 1.0;\n"
		"    if (face > 2.0) { uv.y += fs; face -= 3.0; }\n"
		"    uv.x += face * fs; uv.x += fi * 3.0 * cubeUV_minTileSize;\n"
		"    uv.y += 4.0 * (exp2(CUBEUV_MAX_MIP) - fs);\n"
		"    uv.x *= CUBEUV_TEXEL_WIDTH; uv.y *= CUBEUV_TEXEL_HEIGHT;\n"
		"    acc += texture(envMap, uv).rgb; }\n"
		"  { vec3 d = D1; float mi = mipInt; float face = getFace(d);\n"
		"    float fi = max(cubeUV_minMipLevel - mi, 0.0);\n"
		"    mi = max(mi, cubeUV_minMipLevel); float fs = exp2(mi);\n"
		"    vec2 uv = getUV(d, face) * (fs - 2.0) + 1.0;\n"
		"    if (face > 2.0) { uv.y += fs; face -= 3.0; }\n"
		"    uv.x += face * fs; uv.x += fi * 3.0 * cubeUV_minTileSize;\n"
		"    uv.y += 4.0 * (exp2(CUBEUV_MAX_MIP) - fs);\n"
		"    uv.x *= CUBEUV_TEXEL_WIDTH; uv.y *= CUBEUV_TEXEL_HEIGHT;\n"
		"    acc += texture(envMap, uv).rgb; }\n"
		"  { vec3 d = D2; float mi = mipInt; float face = getFace(d);\n"
		"    float fi = max(cubeUV_minMipLevel - mi, 0.0);\n"
		"    mi = max(mi, cubeUV_minMipLevel); float fs = exp2(mi);\n"
		"    vec2 uv = getUV(d, face) * (fs - 2.0) + 1.0;\n"
		"    if (face > 2.0) { uv.y += fs; face -= 3.0; }\n"
		"    uv.x += face * fs; uv.x += fi * 3.0 * cubeUV_minTileSize;\n"
		"    uv.y += 4.0 * (exp2(CUBEUV_MAX_MIP) - fs);\n"
		"    uv.x *= CUBEUV_TEXEL_WIDTH; uv.y *= CUBEUV_TEXEL_HEIGHT;\n"
		"    acc += texture(envMap, uv).rgb; }\n"
		"  { vec3 d = D3; float mi = mipInt; float face = getFace(d);\n"
		"    float fi = max(cubeUV_minMipLevel - mi, 0.0);\n"
		"    mi = max(mi, cubeUV_minMipLevel); float fs = exp2(mi);\n"
		"    vec2 uv = getUV(d, face) * (fs - 2.0) + 1.0;\n"
		"    if (face > 2.0) { uv.y += fs; face -= 3.0; }\n"
		"    uv.x += face * fs; uv.x += fi * 3.0 * cubeUV_minTileSize;\n"
		"    uv.y += 4.0 * (exp2(CUBEUV_MAX_MIP) - fs);\n"
		"    uv.x *= CUBEUV_TEXEL_WIDTH; uv.y *= CUBEUV_TEXEL_HEIGHT;\n"
		"    acc += texture(envMap, uv).rgb; }\n"
		"  { vec3 d = D4; float mi = mipInt; float face = getFace(d);\n"
		"    float fi = max(cubeUV_minMipLevel - mi, 0.0);\n"
		"    mi = max(mi, cubeUV_minMipLevel); float fs = exp2(mi);\n"
		"    vec2 uv = getUV(d, face) * (fs - 2.0) + 1.0;\n"
		"    if (face > 2.0) { uv.y += fs; face -= 3.0; }\n"
		"    uv.x += face * fs; uv.x += fi * 3.0 * cubeUV_minTileSize;\n"
		"    uv.y += 4.0 * (exp2(CUBEUV_MAX_MIP) - fs);\n"
		"    uv.x *= CUBEUV_TEXEL_WIDTH; uv.y *= CUBEUV_TEXEL_HEIGHT;\n"
		"    acc += texture(envMap, uv).rgb; }\n"
		"  pc_fragColor = vec4(acc * 0.2, 1.0);\n"
		"}\n",
		texel_w, texel_h, max_mip);
	if (n <= 0 || n >= 16384) {
		delete[] built;
		return nullptr;
	}
	return built;
}

// Ledger #90 — WebGL 1 spec §5 GLSL identifier reservation. `_webgl_` and
// `webgl_` are reserved prefixes; shaders using them for any identifier
// (variable, function, struct, struct field, etc.) MUST fail compilation.
// Mesa Nouveau's GLSL compiler doesn't enforce this, so the Khronos
// glsl-reserved-{_,}webgl_{field,function,struct,variable} cluster (8
// tests) all FAIL the "[unexpected link status] (expected: false)" check.
// Detect at boundary (start-of-string OR preceded by non-identifier char)
// to avoid false-positives on legit identifiers that happen to contain
// "webgl_" as a substring (e.g. "myapp_webgl_foo" is legal per spec —
// only the LEADING prefix is reserved).
static bool has_reserved_webgl_identifier(const char *src) {
	if (!src) return false;
	for (const char *p = src; *p; p++) {
		char prev = (p == src) ? '\0' : p[-1];
		bool at_boundary =
		    !((prev >= 'a' && prev <= 'z') || (prev >= 'A' && prev <= 'Z') ||
		      (prev >= '0' && prev <= '9') || prev == '_');
		if (!at_boundary) continue;
		if (p[0] == '_' && !strncmp(p + 1, "webgl_", 6)) return true;
		if (!strncmp(p, "webgl_", 6)) return true;
	}
	return false;
}
FN(w_shader_source) {
	GLuint s = obj_id(info[0]);
	char *src = take_string(info.GetIsolate(), info[1]);
	// Ledger #90 — replace source with a #error directive that halts the
	// GLSL preprocessor. glCompileShader then reports COMPILE_STATUS=false;
	// attachShader + linkProgram propagate that to LINK_STATUS=false,
	// which is what the glsl-reserved cluster's info.linkSuccess=false
	// expectation requires.
	if (has_reserved_webgl_identifier(src)) {
		const char *err_src =
		    "#error nxjs: WebGL 1 spec \xC2\xA75 reserves _webgl_/webgl_ "
		    "prefix\nvoid main(){gl_Position=vec4(0);}\n";
		glShaderSource(s, 1, &err_src, nullptr);
		delete[] src;
		return;
	}
	char *pmrem_repl = maybe_replace_pmrem_fs(s, src);
	const char *p = pmrem_repl ? pmrem_repl : src;
	glShaderSource(s, 1, &p, nullptr);
	delete[] src;
	if (pmrem_repl) delete[] pmrem_repl;
}
FN(w_compile_shader) { enter_bracket(); glCompileShader(obj_id(info[0])); }
FN(w_get_shader_parameter) {
	GLuint s = obj_id(info[0]);
	GLenum pname = a_u32(info, 1);
	GLint v = 0;
	glGetShaderiv(s, pname, &v);
	if (pname == GL_DELETE_STATUS || pname == GL_COMPILE_STATUS) {
		info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), v != 0));
	} else {
		info.GetReturnValue().Set(Int32::New(info.GetIsolate(), v));
	}
}
FN(w_get_shader_info_log) {
	GLuint s = obj_id(info[0]);
	GLint len = 0;
	glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
	std::vector<char> buf(len > 0 ? len : 1);
	GLsizei written = 0;
	if (len > 0) glGetShaderInfoLog(s, len, &written, buf.data());
	info.GetReturnValue().Set(
	    String::NewFromUtf8(info.GetIsolate(),
	                        buf.data(), NewStringType::kNormal,
	                        written).ToLocalChecked());
}
FN(w_get_shader_source) {
	GLuint s = obj_id(info[0]);
	GLint len = 0;
	glGetShaderiv(s, GL_SHADER_SOURCE_LENGTH, &len);
	std::vector<char> buf(len > 0 ? len : 1);
	GLsizei written = 0;
	if (len > 0) glGetShaderSource(s, len, &written, buf.data());
	info.GetReturnValue().Set(
	    String::NewFromUtf8(info.GetIsolate(),
	                        buf.data(), NewStringType::kNormal,
	                        written).ToLocalChecked());
}

// ----- Program -----
FN(w_create_program) {
	GLuint p = glCreateProgram();
	info.GetReturnValue().Set(new_gl_obj(info.GetIsolate(), K_PROGRAM, p));
}
FN(w_delete_program) {
	GLuint id = obj_id(info[0]);
	if (id) {
		glDeleteProgram(id);
		// Ledger #68 — clear any aliased-link record for this program name
		// so a fresh program allocated later with the same GLuint doesn't
		// inherit the stale aliased state (glGenProgram reuse is spec-legal
		// after delete).
		if (st) g_programs_with_aliased_link.erase(id);
		erase_wrapper_cache(K_PROGRAM, id);
	}
}
FN(w_is_program) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsProgram(obj_id(info[0])) == GL_TRUE));
}
FN(w_attach_shader) {
	glAttachShader(obj_id(info[0]), obj_id(info[1]));
}
FN(w_detach_shader) {
	glDetachShader(obj_id(info[0]), obj_id(info[1]));
}
// Ledger #68 — post-link scan for aliased attribute locations. WebGL spec
// §5.14.9: linkProgram MUST fail when two active attributes end up bound
// to the same location. Some drivers (Mesa-Nouveau observed) succeed the
// link and let both names read at the same location. This helper walks
// the program's active attribs, resolves each name's real location via
// glGetAttribLocation, and if two active attribs share a location it
// records the program as aliased-link-failed. `w_get_program_parameter`
// then reports GL_LINK_STATUS = false for the recorded programs.
//
// Runs only when the driver reported success (there's nothing to override
// if the driver already failed). Skips inactive/-1 attribs (they're not
// what the spec's aliasing rule targets).
static void nx_detect_link_attrib_aliasing(GLuint program) {
	if (!st || program == 0) return;
	GLint link_ok = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
	if (!link_ok) {
		g_programs_with_aliased_link.erase(program);
		return;
	}
	GLint active_count = 0;
	glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &active_count);
	GLint max_name_len = 0;
	glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &max_name_len);
	if (max_name_len < 1) max_name_len = 1;
	std::vector<char> name_buf((size_t)max_name_len + 1);
	std::unordered_set<GLint> seen_locations;
	bool aliased = false;
	for (GLint i = 0; i < active_count; i++) {
		GLsizei nlen = 0;
		GLint size = 0;
		GLenum type = 0;
		glGetActiveAttrib(program, (GLuint)i, max_name_len, &nlen,
		                   &size, &type, name_buf.data());
		if (nlen <= 0) continue;
		// Skip GL-reserved names — they're pre-linked to fixed pipeline
		// slots and can't participate in user aliasing.
		if (name_buf[0] == 'g' && name_buf[1] == 'l' && name_buf[2] == '_')
			continue;
		const GLint loc = glGetAttribLocation(program, name_buf.data());
		if (loc < 0) continue;
		if (!seen_locations.insert(loc).second) {
			// Duplicate location — two active attribs aliased.
			aliased = true;
			break;
		}
	}
	if (aliased) {
		g_programs_with_aliased_link.insert(program);
	} else {
		g_programs_with_aliased_link.erase(program);
	}
}

FN(w_link_program) {
	enter_bracket();
	const GLuint p = obj_id(info[0]);
	glLinkProgram(p);
	// Ledger #68 — post-link aliased-attribute check. Runs unconditionally
	// (whether the driver reported success or failure) so a subsequent
	// linkProgram on the same program clears any stale aliased flag.
	nx_detect_link_attrib_aliasing(p);
}
FN(w_validate_program) { enter_bracket(); glValidateProgram(obj_id(info[0])); }
FN(w_use_program) {
	enter_bracket();
	const GLuint p = obj_id(info[0]);
	glUseProgram(p);
	if (st) st->user_snap.program = (GLint)p;
}
FN(w_get_program_parameter) {
	GLuint p = obj_id(info[0]);
	GLenum pname = a_u32(info, 1);
	GLint v = 0;
	glGetProgramiv(p, pname, &v);
	if (pname == GL_DELETE_STATUS || pname == GL_LINK_STATUS ||
	    pname == GL_VALIDATE_STATUS) {
		// Ledger #68 — LINK_STATUS override for aliased-attribute programs.
		// The driver may have succeeded the link despite spec violation;
		// nx_detect_link_attrib_aliasing (run at the tail of w_link_program)
		// marks such programs and we report FALSE here so
		// `attribs-gl-bindAttribLocation-aliasing` sees a spec-correct
		// failure verdict.
		if (pname == GL_LINK_STATUS && v != 0 && st &&
		    g_programs_with_aliased_link.count(p) > 0) {
			v = 0;
		}
		info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), v != 0));
	} else {
		info.GetReturnValue().Set(Int32::New(info.GetIsolate(), v));
	}
}
FN(w_get_program_info_log) {
	GLuint p = obj_id(info[0]);
	GLint len = 0;
	glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
	std::vector<char> buf(len > 0 ? len : 1);
	GLsizei written = 0;
	if (len > 0) glGetProgramInfoLog(p, len, &written, buf.data());
	info.GetReturnValue().Set(
	    String::NewFromUtf8(info.GetIsolate(),
	                        buf.data(), NewStringType::kNormal,
	                        written).ToLocalChecked());
}
FN(w_get_attrib_location) {
	GLuint p = obj_id(info[0]);
	char *name = take_string(info.GetIsolate(), info[1]);
	GLint loc = glGetAttribLocation(p, name);
	delete[] name;
	info.GetReturnValue().Set(Int32::New(info.GetIsolate(), loc));
}
FN(w_get_uniform_location) {
	GLuint p = obj_id(info[0]);
	char *name = take_string(info.GetIsolate(), info[1]);
	// Ledger #88 — WebGL 1 spec 5.14 / 3.7.16: array indices in the
	// uniform name (e.g. "colora[N]") must parse as a non-negative
	// decimal integer that fits in int32_t. Out-of-range indices return
	// null. The Khronos test
	// `uniforms-no-over-optimization-on-uniform-array-{00..17}` exercises
	// this with indices near uint32 max (4294967296 = 2^32, plus 4294967518
	// and 4294967519). Mesa-Nouveau's native glGetUniformLocation returns
	// a non-null location for those names (driver leniency — the parser
	// wraps or clamps), so without this client-side validation all 18
	// tests FAIL the single assertion "Requesting colora[BIG] uniform
	// should return a null uniform location". Ported verbatim from the
	// QuickJS-era engine at ../nxjs-source/source/webgl.c:7591-7618.
	const char *lbracket = strchr(name, '[');
	if (lbracket) {
		const char *rbracket = strchr(lbracket + 1, ']');
		if (!rbracket || rbracket == lbracket + 1) {
			delete[] name;
			info.GetReturnValue().SetNull();
			return;
		}
		uint64_t idx = 0;
		bool valid = true;
		for (const char *q = lbracket + 1; q < rbracket; q++) {
			if (*q < '0' || *q > '9') { valid = false; break; }
			idx = idx * 10u + (uint32_t)(*q - '0');
			if (idx > (uint64_t)0x7FFFFFFF /*INT32_MAX*/) {
				valid = false;
				break;
			}
		}
		if (!valid) {
			delete[] name;
			info.GetReturnValue().SetNull();
			return;
		}
	}
	GLint loc = glGetUniformLocation(p, name);
	delete[] name;
	if (loc < 0) {
		info.GetReturnValue().SetNull();
		return;
	}
	info.GetReturnValue().Set(
	    new_gl_obj(info.GetIsolate(), K_UNIFORM_LOCATION, 0, loc));
}
FN(w_bind_attrib_location) {
	GLuint p = obj_id(info[0]);
	GLuint idx = a_u32(info, 1);
	char *name = take_string(info.GetIsolate(), info[2]);
	glBindAttribLocation(p, idx, name);
	delete[] name;
}
FN(w_get_active_attrib) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	GLuint p = obj_id(info[0]);
	GLuint idx = a_u32(info, 1);
	char name[256];
	GLsizei nlen = 0;
	GLint size = 0;
	GLenum type = 0;
	glGetActiveAttrib(p, idx, sizeof(name), &nlen, &size, &type, name);
	Local<Object> obj = new_gl_obj(iso, K_ACTIVE_INFO, 0);
	Local<Context> c = cur(iso);
	obj->Set(c, String::NewFromUtf8(iso, "name").ToLocalChecked(),
	         String::NewFromUtf8(iso, name, NewStringType::kNormal,
	                              nlen).ToLocalChecked()).Check();
	obj->Set(c, String::NewFromUtf8(iso, "size").ToLocalChecked(),
	         Int32::New(iso, size)).Check();
	obj->Set(c, String::NewFromUtf8(iso, "type").ToLocalChecked(),
	         Uint32::NewFromUnsigned(iso, type)).Check();
	info.GetReturnValue().Set(obj);
}
FN(w_get_active_uniform) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	GLuint p = obj_id(info[0]);
	GLuint idx = a_u32(info, 1);
	char name[256];
	GLsizei nlen = 0;
	GLint size = 0;
	GLenum type = 0;
	glGetActiveUniform(p, idx, sizeof(name), &nlen, &size, &type, name);
	Local<Object> obj = new_gl_obj(iso, K_ACTIVE_INFO, 0);
	Local<Context> c = cur(iso);
	obj->Set(c, String::NewFromUtf8(iso, "name").ToLocalChecked(),
	         String::NewFromUtf8(iso, name, NewStringType::kNormal,
	                              nlen).ToLocalChecked()).Check();
	obj->Set(c, String::NewFromUtf8(iso, "size").ToLocalChecked(),
	         Int32::New(iso, size)).Check();
	obj->Set(c, String::NewFromUtf8(iso, "type").ToLocalChecked(),
	         Uint32::NewFromUnsigned(iso, type)).Check();
	info.GetReturnValue().Set(obj);
}

// ----- Buffer -----
FN(w_create_buffer) {
	GLuint b = 0;
	glGenBuffers(1, &b);
	// Ledger #95 — new_gl_obj_create evicts any stale (deleted-but-cached)
	// wrapper with the same GL name before creating so a driver-side gen-
	// after-delete reuse of the same name doesn't hand back the marked-
	// deleted wrapper.
	info.GetReturnValue().Set(
	    new_gl_obj_create(info.GetIsolate(), K_BUFFER, b));
}
FN(w_delete_buffer) {
	// Ledger #95 — keep the wrapper in the #92 cache; mark deleted so
	// bindBuffer / bufferData etc. can reject with INVALID_OPERATION.
	// Identity of the wrapper survives into subsequent
	// getFramebufferAttachmentParameter / getVertexAttrib queries per
	// WebGL 1 spec §5.14.
	GLObj *o = get_gl_obj(info[0]);
	if (o && o->id) {
		glDeleteBuffers(1, &o->id);
		o->deleted = true;
	}
}
FN(w_is_buffer) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsBuffer(obj_id(info[0])) == GL_TRUE));
}
FN(w_bind_buffer) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	// Ledger #95 — WebGL 1 spec §5.14: bindBuffer on a deleted buffer
	// generates INVALID_OPERATION and MUST NOT change the current
	// binding. glDeleteBuffers already ran on the underlying GL name;
	// the reject-here contract is a WebGL-layer promise, not a GL one.
	if (obj_deleted(info[1])) {
		record_error(GL_INVALID_OPERATION);
		return;
	}
	const GLuint buf = obj_id(info[1]);
	glBindBuffer(target, buf);
	if (st && target == GL_ARRAY_BUFFER) st->user_snap.array_buffer = (GLint)buf;
}
// WebGL2 (srcData, srcOffset[, length]) overload support. Emscripten's GLES3
// path passes the WHOLE HEAP typed-array as srcData plus a byte srcOffset
// (and, for buffer/compressed uploads, a length) — see WebGL2 §5.14 bufferData
// / bufferSubData / texImage* overloads. Without honoring srcOffset/length the
// handler would upload the whole heap from offset 0 (all zeros), which made
// every emscripten/Unity WebGL2 vertex + texture upload arrive blank (black
// screen). WebGL1's emscripten path uses HEAPU8.subarray(...) (already offset)
// so it was unaffected. Three.js passes the real (already-offset) view in the
// 3-arg form, so this no-ops for it.
static size_t ta_elem_size(Local<Value> v) {
	if (!v.IsEmpty() && v->IsTypedArray()) {
		Local<TypedArray> ta = v.As<TypedArray>();
		size_t n = ta->Length();
		if (n > 0) return ta->ByteLength() / n;
	}
	return 1; // DataView / Uint8Array / fallback = byte-addressed
}
// Adjust (*pp,*plen) for a call whose srcData is at `srcIdx` and whose numeric
// srcOffset is at `offIdx` (length, if any, at offIdx+1). srcOffset/length are
// in ELEMENTS of srcData. No-op unless a numeric srcOffset is actually present.
static void apply_webgl2_src_offset(const FunctionCallbackInfo<Value> &info,
                                    int srcIdx, int offIdx, uint8_t **pp,
                                    size_t *plen, bool have_length) {
	if (!*pp) return;
	if (info.Length() <= offIdx || !info[offIdx]->IsNumber()) return;
	const size_t es = ta_elem_size(info[srcIdx]);
	const size_t srcOff = (size_t)a_i64(info, offIdx) * es;
	if (srcOff > *plen) { *plen = 0; return; }
	*pp += srcOff;
	*plen -= srcOff;
	if (have_length && info.Length() > offIdx + 1 &&
	    info[offIdx + 1]->IsNumber()) {
		const int64_t lenEl = a_i64(info, offIdx + 1);
		if (lenEl > 0) {
			const size_t sub = (size_t)lenEl * es;
			if (sub < *plen) *plen = sub;
		}
	}
}
FN(w_buffer_data) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLenum usage = a_u32(info, 2);
	if (info[1]->IsNumber()) {
		// (target, size, usage) — allocate uninitialized storage.
		const int64_t size = a_i64(info, 1);
		glBufferData(target, (GLsizeiptr)size, nullptr, usage);
		return;
	}
	size_t len = 0;
	uint8_t *p = view_bytes(info[1], &len);
	if (info[1]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[1].As<ArrayBuffer>();
		p = (uint8_t *)ab->Data();
		len = ab->ByteLength();
	}
	apply_webgl2_src_offset(info, 1, 3, &p, &len, true);
	glBufferData(target, (GLsizeiptr)len, p, usage);
}
FN(w_buffer_sub_data) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLintptr offset = (GLintptr)a_i64(info, 1);
	size_t len = 0;
	uint8_t *p = view_bytes(info[2], &len);
	if (info[2]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[2].As<ArrayBuffer>();
		p = (uint8_t *)ab->Data();
		len = ab->ByteLength();
	}
	apply_webgl2_src_offset(info, 2, 3, &p, &len, true);
	if (p) glBufferSubData(target, offset, (GLsizeiptr)len, p);
}

// ----- Vertex attrib -----
FN(w_enable_vertex_attrib_array) {
	enter_bracket();
	glEnableVertexAttribArray(a_u32(info, 0));
}
FN(w_disable_vertex_attrib_array) {
	enter_bracket();
	glDisableVertexAttribArray(a_u32(info, 0));
}
FN(w_vertex_attrib_pointer) {
	enter_bracket();
	const GLuint index = a_u32(info, 0);
	const GLint size = a_i32(info, 1);
	const GLenum type = a_u32(info, 2);
	const GLboolean normalized = a_bool(info, 3) ? GL_TRUE : GL_FALSE;
	const GLsizei stride = a_i32(info, 4);
	const int64_t offset = a_i64(info, 5);
	// Ledger #98 — WebGL 1 spec §5.14.10: vertexAttribPointer's `type`
	// arg only accepts BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, FLOAT.
	// INT / UNSIGNED_INT / FIXED are all invalid on WebGL 1 (they're
	// added on WebGL 2 by ES3 core, plus INT_2_10_10_10_REV via ES3).
	// Mesa Nouveau's native glVertexAttribPointer runs an ES3.2 driver so
	// it accepts INT/UNSIGNED_INT silently (FIXED via GL_OES_fixed_point
	// / ES1-legacy on some builds). Guard client-side to match spec.
	// OES_vertex_type_2_10_10_10_REV / OES_vertex_half_float would extend
	// this list at extension-enable time; not implemented in this ledger
	// because the conformance test doesn't exercise them on WebGL 1.
	int bytes_per_component = 0;
	if (!is_v2_context(info)) {
		switch (type) {
		case GL_BYTE:           bytes_per_component = 1; break;
		case GL_UNSIGNED_BYTE:  bytes_per_component = 1; break;
		case GL_SHORT:          bytes_per_component = 2; break;
		case GL_UNSIGNED_SHORT: bytes_per_component = 2; break;
		case GL_FLOAT:          bytes_per_component = 4; break;
		default:
			record_error(GL_INVALID_ENUM);
			return;
		}
		// Ledger #98 — WebGL 1 spec §5.14.10 additional validation the
		// native driver doesn't enforce:
		//   - `stride` must be in [0, 255]. GLES ES3 relaxes this to
		//     INT_MAX for driver-side memory reasons; WebGL 1 keeps the
		//     original ES2 range. `> 255` → INVALID_VALUE.
		//   - `stride % bytesPerComponent != 0` → INVALID_OPERATION
		//     (misaligned inter-vertex step for the primitive size).
		//   - `offset % bytesPerComponent != 0` → INVALID_OPERATION
		//     (misaligned per-attribute base).
		//   - `size` must be in [1, 4] (already partially enforced by
		//     driver via INVALID_VALUE).
		//   - `offset < 0` → INVALID_VALUE. (Never triggered by JS
		//     numeric coercion since a_i64 caps at int64 min, but the
		//     check is cheap and spec-mandated.)
		if (stride < 0 || stride > 255) {
			record_error(GL_INVALID_VALUE);
			return;
		}
		if (size < 1 || size > 4) {
			record_error(GL_INVALID_VALUE);
			return;
		}
		if (offset < 0) {
			record_error(GL_INVALID_VALUE);
			return;
		}
		if (bytes_per_component > 1) {
			if ((stride % bytes_per_component) != 0) {
				record_error(GL_INVALID_OPERATION);
				return;
			}
			if ((offset % bytes_per_component) != 0) {
				record_error(GL_INVALID_OPERATION);
				return;
			}
		}
	}
	glVertexAttribPointer(index, size, type, normalized, stride,
	                      (const void *)(intptr_t)offset);
}
FN(w_vertex_attrib_1f) { enter_bracket(); glVertexAttrib1f(a_u32(info, 0), a_f32(info, 1)); }
FN(w_vertex_attrib_2f) { enter_bracket(); glVertexAttrib2f(a_u32(info, 0), a_f32(info, 1), a_f32(info, 2)); }
FN(w_vertex_attrib_3f) { enter_bracket(); glVertexAttrib3f(a_u32(info, 0), a_f32(info, 1), a_f32(info, 2), a_f32(info, 3)); }
FN(w_vertex_attrib_4f) { enter_bracket(); glVertexAttrib4f(a_u32(info, 0), a_f32(info, 1), a_f32(info, 2), a_f32(info, 3), a_f32(info, 4)); }

// ----- Uniform -----
FN(w_uniform_1f) { enter_bracket(); glUniform1f(uniform_loc(info[0]), a_f32(info, 1)); }
FN(w_uniform_2f) { enter_bracket(); glUniform2f(uniform_loc(info[0]), a_f32(info, 1), a_f32(info, 2)); }
FN(w_uniform_3f) { enter_bracket(); glUniform3f(uniform_loc(info[0]), a_f32(info, 1), a_f32(info, 2), a_f32(info, 3)); }
FN(w_uniform_4f) { enter_bracket(); glUniform4f(uniform_loc(info[0]), a_f32(info, 1), a_f32(info, 2), a_f32(info, 3), a_f32(info, 4)); }
FN(w_uniform_1i) { enter_bracket(); glUniform1i(uniform_loc(info[0]), a_i32(info, 1)); }
FN(w_uniform_2i) { enter_bracket(); glUniform2i(uniform_loc(info[0]), a_i32(info, 1), a_i32(info, 2)); }
FN(w_uniform_3i) { enter_bracket(); glUniform3i(uniform_loc(info[0]), a_i32(info, 1), a_i32(info, 2), a_i32(info, 3)); }
FN(w_uniform_4i) { enter_bracket(); glUniform4i(uniform_loc(info[0]), a_i32(info, 1), a_i32(info, 2), a_i32(info, 3), a_i32(info, 4)); }

#define UNI_FV(N) \
FN(w_uniform_##N##fv) { \
	enter_bracket(); \
	std::vector<float> tmp; \
	const float *p = nullptr; \
	size_t n = 0; \
	if (!f32_list(info.GetIsolate(), info[1], tmp, &p, &n)) return; \
	glUniform##N##fv(uniform_loc(info[0]), (GLsizei)(n / N), p); \
}
UNI_FV(1)
UNI_FV(2)
UNI_FV(3)
UNI_FV(4)
#undef UNI_FV

#define UNI_IV(N) \
FN(w_uniform_##N##iv) { \
	enter_bracket(); \
	std::vector<int32_t> tmp; \
	const int32_t *p = nullptr; \
	size_t n = 0; \
	if (!i32_list(info.GetIsolate(), info[1], tmp, &p, &n)) return; \
	glUniform##N##iv(uniform_loc(info[0]), (GLsizei)(n / N), p); \
}
UNI_IV(1)
UNI_IV(2)
UNI_IV(3)
UNI_IV(4)
#undef UNI_IV

#define UNI_MAT_FV(N) \
FN(w_uniform_matrix_##N##fv) { \
	enter_bracket(); \
	std::vector<float> tmp; \
	const float *p = nullptr; \
	size_t n = 0; \
	if (!f32_list(info.GetIsolate(), info[2], tmp, &p, &n)) return; \
	const GLsizei count = (GLsizei)(n / (N * N)); \
	glUniformMatrix##N##fv(uniform_loc(info[0]), count, \
	                        a_bool(info, 1) ? GL_TRUE : GL_FALSE, p); \
}
UNI_MAT_FV(2)
UNI_MAT_FV(3)
UNI_MAT_FV(4)
#undef UNI_MAT_FV

// ----- Texture -----
FN(w_create_texture) {
	GLuint t = 0;
	glGenTextures(1, &t);
	// Ledger #95 — see w_create_buffer for the create-path eviction
	// rationale.
	info.GetReturnValue().Set(
	    new_gl_obj_create(info.GetIsolate(), K_TEXTURE, t));
}
FN(w_delete_texture) {
	// Ledger #95 — see w_delete_buffer.
	GLObj *o = get_gl_obj(info[0]);
	if (o && o->id) {
		glDeleteTextures(1, &o->id);
		o->deleted = true;
	}
}
FN(w_is_texture) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsTexture(obj_id(info[0])) == GL_TRUE));
}
FN(w_bind_texture) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	// Ledger #95 — see w_bind_buffer.
	if (obj_deleted(info[1])) {
		record_error(GL_INVALID_OPERATION);
		return;
	}
	const GLuint tex = obj_id(info[1]);
	glBindTexture(target, tex);
	// Track TU0 TEXTURE_2D binding to match nx_gl_state_snap_t coverage.
	if (st && target == GL_TEXTURE_2D &&
		st->user_snap.active_tex == (GLint)GL_TEXTURE0) {
		st->user_snap.tex2d_binding = (GLint)tex;
	}
}
FN(w_active_texture) {
	enter_bracket();
	const GLenum unit = a_u32(info, 0);
	glActiveTexture(unit);
	if (st) st->user_snap.active_tex = (GLint)unit;
}

// Ledger #95b — WebGL 1 spec §5.14.8: texParameter{i,f} on a target with no
// currently-bound texture generates INVALID_OPERATION. GLES 2 native is
// permissive here (glTexParameter{i,f} silently no-ops when nothing is bound),
// so a client-side guard is needed to match the spec assertion the
// misc-object-deletion-behaviour test relies on: after `deleteTexture(t)` on
// t bound to TEXTURE_2D, WebGL 1 §5.14.5 auto-unbinds t to null; #95's
// bindTexture(deleted t) then also refuses without re-binding; the resulting
// "no texture bound" state must reject subsequent texParameteri with 0x502.
// Query the binding via glGetIntegerv(TEXTURE_BINDING_<target>). Only guards
// TEXTURE_2D and TEXTURE_CUBE_MAP (WebGL 1); WebGL 2's 3D / 2D_ARRAY targets
// fall through unchanged for now — their misc-object-deletion equivalents
// aren't in this test's scope.
static inline bool nx_binding_pname_for_tex_target(GLenum target, GLenum *out) {
	switch (target) {
	case GL_TEXTURE_2D:
		*out = 0x8069; /* GL_TEXTURE_BINDING_2D */
		return true;
	case GL_TEXTURE_CUBE_MAP:
		*out = 0x8514; /* GL_TEXTURE_BINDING_CUBE_MAP */
		return true;
	default:
		return false;
	}
}
FN(w_tex_parameteri) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	GLenum bpname;
	if (nx_binding_pname_for_tex_target(target, &bpname)) {
		GLint bound = 0;
		glGetIntegerv(bpname, &bound);
		if (bound == 0) {
			record_error(GL_INVALID_OPERATION);
			return;
		}
	}
	glTexParameteri(target, a_u32(info, 1), a_i32(info, 2));
}
FN(w_tex_parameterf) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	GLenum bpname;
	if (nx_binding_pname_for_tex_target(target, &bpname)) {
		GLint bound = 0;
		glGetIntegerv(bpname, &bound);
		if (bound == 0) {
			record_error(GL_INVALID_OPERATION);
			return;
		}
	}
	glTexParameterf(target, a_u32(info, 1), a_f32(info, 2));
}
// Ledger #91 — WebGL 1 spec §5.14.8: non-power-of-two textures have
// restrictions. `is_pot` is the canonical "n is a positive power of two"
// check; `n == 0` returns false so a zero-dim texture (never uploaded)
// doesn't accidentally accept a NPOT-guarded call.
//
// generateMipmap-side NPOT check needs the currently-bound texture's
// dimensions, which GLES 2 doesn't expose via glGetTexLevelParameteriv
// (a GLES 3.1+ / desktop-GL primitive). Would require per-texture
// dimension tracking in nx-side state; deferred as a follow-up. This
// ledger only tackles the `w_tex_image_2d` and `w_copy_tex_image_2d`
// level>0-and-NPOT paths — those get width/height from call args
// directly and don't need any state lookup.
static inline bool is_pot(GLint n) { return n > 0 && (n & (n - 1)) == 0; }
FN(w_generate_mipmap) {
	enter_bracket();
	glGenerateMipmap(a_u32(info, 0));
}
// texImage2D — buffer-source overload only for 2.C
//   (target, level, internalformat, width, height, border, format, type, pixels)
//
// Bucket E translate (re-applies dropped fork patches; see references in
// auto-memory):
//   * WebGL1 EXT_sRGB unsized formats → ES3 sized internalformat + unsized
//     format. Three.js's WebGL1 path uses SRGB_EXT/SRGB_ALPHA_EXT (per the
//     EXT_sRGB spec, which requires internalformat==format). ES3-core does
//     not recognize 0x8C40/0x8C42 as internalformats and returns
//     GL_INVALID_ENUM — confirmed empirically via the bucket-e:texImage2D
//     diagnostic (err=0x0500 on 0x8C42 calls; 0x0000 on RGBA calls). Reapplies
//     [[reference-brewser-v1-black-texture-demos]].
//   * WebGL1 unsized internalformat + HALF_FLOAT/FLOAT type → ES3 sized
//     internalformat (RGBA→RGBA16F/32F, RGB→RGB16F/32F, RG→RG16F/32F,
//     R→R16F/32F). ES3 requires a sized internalformat for HALF_FLOAT/FLOAT
//     types — unsized RGBA + HALF_FLOAT is INVALID_OPERATION in ES3. This is
//     a precondition for Bucket F PMREM (cube→2D atlas intermediates upload
//     half-float). Reapplies [[reference-dfglut-rg16f-accept-list-fix]] as a
//     translate (the V8 engine has no accept-list to widen — the V8 analog
//     is the unsized→sized translation).
// Both translations are guarded by `internalformat == format` so we only
// touch the WebGL1-style ES2 call shape; if a caller already passes a sized
// internalformat, we leave it alone.
static inline void bucket_e_translate_tex_image(
    GLint *internalformat, GLenum *format, GLenum *type) {
	if (*internalformat == *format) {
		if (*internalformat == 0x8C42 /* SRGB_ALPHA_EXT */) {
			*internalformat = 0x8C43; /* SRGB8_ALPHA8 */
			*format         = 0x1908; /* GL_RGBA */
			return;
		}
		if (*internalformat == 0x8C40 /* SRGB_EXT */) {
			*internalformat = 0x8C41; /* SRGB8 */
			*format         = 0x1907; /* GL_RGB */
			return;
		}
		if (*type == 0x140B /* HALF_FLOAT */ ||
		    *type == 0x8D61 /* HALF_FLOAT_OES */) {
			switch (*internalformat) {
			case 0x1908: *internalformat = 0x881A; break; /* RGBA→RGBA16F */
			case 0x1907: *internalformat = 0x881B; break; /* RGB→RGB16F */
			case 0x8227: *internalformat = 0x822F; break; /* RG→RG16F */
			case 0x1903: *internalformat = 0x822D; break; /* RED→R16F */
			}
		}
		if (*type == 0x1406 /* GL_FLOAT */) {
			switch (*internalformat) {
			case 0x1908: *internalformat = 0x8814; break; /* RGBA→RGBA32F */
			case 0x1907: *internalformat = 0x8815; break; /* RGB→RGB32F */
			case 0x8227: *internalformat = 0x8230; break; /* RG→RG32F */
			case 0x1903: *internalformat = 0x822E; break; /* RED→R32F */
			}
		}
	}
	// F.1: Normalize HALF_FLOAT_OES (WebGL1 OES extension token, 0x8D61) to
	// HALF_FLOAT (ES3-core token, 0x140B) when paired with a sized half-float
	// internalformat (RGBA16F/RGB16F/RG16F/R16F). The QuickJS-era engine
	// comment that said "Mesa Nouveau treats 0x8D61 and 0x140B as aliases"
	// is true for the WebGL1 ES2-style unsized call (internalformat==format
	// unsized), but ES3 spec REQUIRES type==HALF_FLOAT for sized half-float
	// internalformats — pairing HALF_FLOAT_OES with sized internalformat
	// returns INVALID_OPERATION on Mesa-Nouveau-on-Citron (confirmed via F.1
	// `[f1:texImage2D-hf]` diagnostic on webgl-loader-gltf's r162 PMREM
	// RGBA16F intermediate uploads). Three.js r162's PMREM passes already-
	// sized RGBA16F + HALF_FLOAT_OES; this normalization makes that valid.
	if (*type == 0x8D61 /* HALF_FLOAT_OES */) {
		switch (*internalformat) {
		case 0x881A /* RGBA16F */:
		case 0x881B /* RGB16F */:
		case 0x822F /* RG16F */:
		case 0x822D /* R16F */:
			*type = 0x140B; /* HALF_FLOAT */
			break;
		}
	}
}

// texSubImage2D has no internalformat parameter — storage was allocated by a
// prior texImage2D/texStorage2D. The sub-upload's format should be unsized
// (the storage carries the sizing). The only translation needed mirrors the
// SRGB format-side normalization (in case Three.js passes SRGB_ALPHA_EXT as
// the format here too, mirroring the texImage2D pattern).
static inline void bucket_e_translate_tex_sub_image(
    GLenum *format, GLenum *type) {
	if (*format == 0x8C42 /* SRGB_ALPHA_EXT */) {
		*format = 0x1908; /* GL_RGBA */
	} else if (*format == 0x8C40 /* SRGB_EXT */) {
		*format = 0x1907; /* GL_RGB */
	}
	// F.1: Normalize HALF_FLOAT_OES → HALF_FLOAT. texSubImage2D uploads
	// against storage that was allocated by a prior texImage2D or
	// texStorage2D — if the storage was sized half-float (RGBA16F/etc.),
	// the sub-upload must use the ES3-core HALF_FLOAT token, not the
	// WebGL1 OES alias. Safe to normalize unconditionally: HALF_FLOAT is
	// the ES3-canonical token and Mesa accepts it for any half-float
	// storage shape; the unsized-WebGL1 case still works post-normalize.
	if (*type == 0x8D61 /* HALF_FLOAT_OES */) {
		*type = 0x140B; /* HALF_FLOAT */
	}
}

// Ledger #69 — texImage2D / texSubImage2D ImageBitmap + Image source support.
//
// nx_image_t holds premultiplied BGRA pixels (byte order B,G,R,A, row-major,
// width*4 stride, top-to-bottom — see image.h / image.cc). Pre-#69, the
// texImage2D / texSubImage2D bodies handled only ArrayBuffer / ArrayBufferView
// sources; nx_image_t fell through the null-source branch, so
// `gl.texImage2D(target, 0, format, format, type, imageBitmap)` uploaded
// nothing and the destination texture stayed cleared. WebGL 1 conformance
// hit this across ~40 textures-image_bitmap_from_* tests (all 5 sub-clusters
// × 8 texture formats) with the signature `shouldBe 255,0,0 was 0,0,0`.
//
// This helper converts BGRA → the caller's (format, type) into a scratch
// buffer. Returns scratch.data() on success (scratch owns lifetime), nullptr
// on unsupported (format, type). Honors UNPACK_FLIP_Y_WEBGL (row-reverse)
// and UNPACK_PREMULTIPLY_ALPHA_WEBGL (un-premultiply when the caller asked
// for false — source is already premultiplied).
//
// MVP format matrix (matches the Tier-A conformance iteration):
//   RGBA/UNSIGNED_BYTE, RGB/UNSIGNED_BYTE,
//   RGBA/UNSIGNED_SHORT_4_4_4_4, RGBA/UNSIGNED_SHORT_5_5_5_1,
//   RGB/UNSIGNED_SHORT_5_6_5,
//   LUMINANCE/UNSIGNED_BYTE, LUMINANCE_ALPHA/UNSIGNED_BYTE,
//   ALPHA/UNSIGNED_BYTE.
// Non-MVP (HALF_FLOAT/FLOAT source-uploads, OffscreenCanvas-as-source,
// PBO offset overload) fall through to nullptr → null upload, matching
// pre-#69 behavior. See ledger #69 for deferral rationale.
static uint8_t *convert_image_source_to_gl_pixels(
    nx_image_t *img, GLenum format, GLenum type,
    bool flip_y, bool un_premultiply,
    std::vector<uint8_t> &scratch) {
	if (!img || !img->data || img->width == 0 || img->height == 0) return nullptr;
	const uint32_t W = img->width;
	const uint32_t H = img->height;
	// Bytes per destination pixel for the supported (format, type) matrix.
	size_t dst_bpp = 0;
	if (type == GL_UNSIGNED_BYTE) {
		switch (format) {
		case GL_RGBA:            dst_bpp = 4; break;
		case GL_RGB:             dst_bpp = 3; break;
		case GL_LUMINANCE_ALPHA: dst_bpp = 2; break;
		case GL_LUMINANCE:       dst_bpp = 1; break;
		case GL_ALPHA:           dst_bpp = 1; break;
		default: return nullptr;
		}
	} else if (type == GL_UNSIGNED_SHORT_4_4_4_4 && format == GL_RGBA) {
		dst_bpp = 2;
	} else if (type == GL_UNSIGNED_SHORT_5_5_5_1 && format == GL_RGBA) {
		dst_bpp = 2;
	} else if (type == GL_UNSIGNED_SHORT_5_6_5 && format == GL_RGB) {
		dst_bpp = 2;
	} else {
		return nullptr;
	}
	scratch.assign((size_t)W * (size_t)H * dst_bpp, 0);
	uint8_t *dst = scratch.data();
	for (uint32_t y = 0; y < H; ++y) {
		const uint32_t src_y = flip_y ? (H - 1 - y) : y;
		const uint8_t *src_row = img->data + (size_t)src_y * (size_t)W * 4;
		uint8_t *dst_row = dst + (size_t)y * (size_t)W * dst_bpp;
		for (uint32_t x = 0; x < W; ++x) {
			// Source is premultiplied BGRA.
			uint8_t b = src_row[x * 4 + 0];
			uint8_t g = src_row[x * 4 + 1];
			uint8_t r = src_row[x * 4 + 2];
			uint8_t a = src_row[x * 4 + 3];
			if (un_premultiply && a != 0 && a != 255) {
				// Divide premultiplied channels back out. `(c*255 + a/2)/a`
				// rounds to nearest and clamps at 255 (input is guaranteed
				// c <= a because of premultiplication).
				r = (uint8_t)std::min(255, (r * 255 + a / 2) / a);
				g = (uint8_t)std::min(255, (g * 255 + a / 2) / a);
				b = (uint8_t)std::min(255, (b * 255 + a / 2) / a);
			}
			if (type == GL_UNSIGNED_BYTE) {
				switch (format) {
				case GL_RGBA:
					dst_row[x * 4 + 0] = r;
					dst_row[x * 4 + 1] = g;
					dst_row[x * 4 + 2] = b;
					dst_row[x * 4 + 3] = a;
					break;
				case GL_RGB:
					dst_row[x * 3 + 0] = r;
					dst_row[x * 3 + 1] = g;
					dst_row[x * 3 + 2] = b;
					break;
				case GL_LUMINANCE_ALPHA:
					// Ledger #79 — WebGL 1 spec Table 5.14.6.1 (RGBA source
					// → LUMINANCE_ALPHA target) is L = R, A = A. Not
					// Rec.601 luma. Chrome + conformance both check L=R:
					// the `_from_*-tex-2d-luminance*` tests upload
					// (255, 0, 0) and expect a sample of (255, 255, 255).
					// Rec.601 gives L=76 (fails ±10 tolerance). #69's
					// original comment claimed Rec.601 was spec-compliant
					// — it wasn't; the WebGL 1 §5.14.6 table has been L=R
					// since spec inception, and the ES 2.0 pixel-transfer
					// rules the comment referenced don't apply to
					// TexImageSource uploads (only to typed-array uploads
					// where format must match source).
					dst_row[x * 2 + 0] = r;
					dst_row[x * 2 + 1] = a;
					break;
				case GL_LUMINANCE:
					// Ledger #79 — L = R per WebGL 1 spec Table 5.14.6.1.
					dst_row[x] = r;
					break;
				case GL_ALPHA:
					dst_row[x] = a;
					break;
				}
			} else if (type == GL_UNSIGNED_SHORT_4_4_4_4) {
				// Big-endian in memory per WebGL: (R<<12) | (G<<8) | (B<<4) | A.
				uint16_t p = (uint16_t)(((r >> 4) << 12) | ((g >> 4) << 8) |
				                       ((b >> 4) << 4) | (a >> 4));
				dst_row[x * 2 + 0] = (uint8_t)(p & 0xFF);
				dst_row[x * 2 + 1] = (uint8_t)(p >> 8);
			} else if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
				uint16_t p = (uint16_t)(((r >> 3) << 11) | ((g >> 3) << 6) |
				                       ((b >> 3) << 1) | (a >> 7));
				dst_row[x * 2 + 0] = (uint8_t)(p & 0xFF);
				dst_row[x * 2 + 1] = (uint8_t)(p >> 8);
			} else if (type == GL_UNSIGNED_SHORT_5_6_5) {
				uint16_t p = (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) |
				                       (b >> 3));
				dst_row[x * 2 + 0] = (uint8_t)(p & 0xFF);
				dst_row[x * 2 + 1] = (uint8_t)(p >> 8);
			}
		}
	}
	return dst;
}

// Ledger #108 — UNPACK_FLIP_Y_WEBGL honor for typed-array pixel uploads.
//
// Per WebGL 1 §6.10 and WebGL 2 §5.30, UNPACK_FLIP_Y_WEBGL "applies to any
// type of unpacked pixel data, including array buffer views" — Chrome and
// Firefox row-reverse raw pixel bytes when the flag is set, not just
// TexImageSource inputs. Pre-#108 the fork only flipped inside
// convert_image_source_to_gl_pixels (the nx_image_t branch), so any path
// that funneled to native as bytes (WebGL 2's sourceToPixels shim
// rasterizes ImageBitmap/OffscreenCanvas → getImageData → typed array) got
// unflipped uploads even when the caller had UNPACK_FLIP_Y_WEBGL=1.
//
// Symptom that surfaced this: webgl_materials_video with the fixed post-
// seek clock (#107) renders 200 THREE.VideoTexture cubes upside-down.
// Three.js `WebGLTextures.setTexture2D` sets `pixelStorei(UNPACK_FLIP_Y_
// WEBGL, texture.flipY)` — default `true` for VideoTexture. WebGL2's shim
// (webgl2-rendering-context.ts) rasterizes the ImageBitmap through an
// OffscreenCanvas, hands the RGBA bytes to native. Native's raw-pixel
// path emits glTexImage2D directly with UNPACK_FLIP_Y ignored → V=0 of
// the texture receives the top row → sampled upside-down.
//
// Fix: after the nx_image_t branch declines (pixels came in as a typed
// array), row-reverse pixels into scratch when UNPACK_FLIP_Y_WEBGL is set.
// Row pitch honors UNPACK_ALIGNMENT (already stored on WebGLState). Only
// the (format, type) matrix nx.js can compute is handled — HALF_FLOAT /
// FLOAT / integer formats + BYTE/RGB/RGBA/L/LA/A × UNSIGNED_BYTE +
// packed 16-bit; anything else falls through to pre-fix behavior (no
// flip), matching the pre-fix semantics but not blocking new upload paths.
static size_t nx_gl_pixel_bpp(GLenum format, GLenum type) {
	int channels = 0;
	switch (format) {
	case GL_RGBA: channels = 4; break;
	case GL_RGB:  channels = 3; break;
	case GL_LUMINANCE_ALPHA: channels = 2; break;
	case GL_LUMINANCE:
	case GL_ALPHA:
	case GL_RED:
		channels = 1;
		break;
	default:
		return 0;
	}
	switch (type) {
	case GL_UNSIGNED_BYTE:
		return (size_t)channels;
	case GL_BYTE:
		return (size_t)channels;
	case GL_UNSIGNED_SHORT_4_4_4_4:
	case GL_UNSIGNED_SHORT_5_5_5_1:
	case GL_UNSIGNED_SHORT_5_6_5:
		return 2;
	case GL_UNSIGNED_SHORT:
	case GL_SHORT:
	case 0x8D61 /* GL_HALF_FLOAT_OES */:
	case 0x140B /* GL_HALF_FLOAT */:
		return (size_t)channels * 2;
	case GL_FLOAT:
	case GL_UNSIGNED_INT:
	case GL_INT:
		return (size_t)channels * 4;
	default:
		return 0;
	}
}
// Row-reverse `pixels` into `scratch`. Returns scratch.data() on success,
// nullptr when the (format, type) combo isn't in the bpp matrix (caller
// falls back to pre-fix "no flip" — spec-nonconformant for that combo but
// no worse than the shipping engine). Row pitch = ceil(width * bpp,
// alignment); the copy preserves inter-row padding so the driver's row-
// stride reads keep matching.
static const void *nx_gl_flip_pixels_y(
    const void *pixels, size_t len, GLsizei width, GLsizei height,
    GLenum format, GLenum type, int alignment,
    std::vector<uint8_t> &scratch) {
	if (!pixels || width <= 0 || height <= 0) return nullptr;
	const size_t bpp = nx_gl_pixel_bpp(format, type);
	if (bpp == 0) return nullptr;
	if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8)
		alignment = 4;
	const size_t unaligned_row = bpp * (size_t)width;
	const size_t row_pitch =
	    (unaligned_row + (size_t)alignment - 1) &
	    ~((size_t)alignment - 1);
	// The last row has no trailing pad byte requirement (glTexImage2D only
	// reads `unaligned_row` bytes on the last line), so the smallest buffer
	// glTexImage2D would accept is: (H-1) * row_pitch + unaligned_row.
	const size_t expected =
	    (size_t)(height - 1) * row_pitch + unaligned_row;
	if (len < expected) return nullptr;
	scratch.assign(row_pitch * (size_t)height, 0);
	const uint8_t *src = (const uint8_t *)pixels;
	for (int y = 0; y < height; ++y) {
		const size_t src_row_ofs = (size_t)y * row_pitch;
		const size_t dst_row_ofs = (size_t)(height - 1 - y) * row_pitch;
		// The last input row may lack padding bytes; clamp the copy so we
		// don't read past `len`. Padding bytes on the reversed row stay 0
		// from the assign — driver never reads them past unaligned_row.
		size_t copy = row_pitch;
		if (src_row_ofs + copy > len) copy = len - src_row_ofs;
		memcpy(scratch.data() + dst_row_ofs, src + src_row_ofs, copy);
	}
	return scratch.data();
}

FN(w_tex_image_2d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	GLint internalformat = a_i32(info, 2);
	GLsizei width = a_i32(info, 3);
	GLsizei height = a_i32(info, 4);
	const GLint border = a_i32(info, 5);
	GLenum format = a_u32(info, 6);
	GLenum type = a_u32(info, 7);
	// Ledger #91 — WebGL 1 spec §5.14.8: texImage2D at level > 0 with
	// NPOT dimensions must return INVALID_VALUE. WebGL 2 accepts NPOT at
	// any level. Guard emits the error + short-circuits before any pixel
	// conversion, avoiding wasted work on a call that will fail.
	if (!is_v2_context(info) && level > 0 &&
	    (!is_pot(width) || !is_pot(height))) {
		record_error(GL_INVALID_VALUE);
		return;
	}
	size_t len = 0;
	void *pixels = view_bytes(info[8], &len);
	if (info[8]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[8].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	if (info[8]->IsNullOrUndefined()) pixels = nullptr;
	// Ledger #101 — WEBGL_depth_texture (WebGL 1) spec §4.1: texImage2D
	// with format ∈ {DEPTH_COMPONENT, DEPTH_STENCIL} has three additional
	// constraints beyond the base spec (Mesa Nouveau's ES3.2 native
	// accepts all these silently):
	//   - target MUST be TEXTURE_2D. Cube-face targets → INVALID_OPERATION.
	//   - level MUST be 0. Non-zero level → INVALID_OPERATION.
	//   - pixels MUST be null. Non-null pixels → INVALID_OPERATION.
	// v2 (WebGL 2 / ES3) lifts these; the gate is v1-only.
	//
	// Note the "null" check uses `info[8]->IsNullOrUndefined()`, not the
	// C++ `pixels != nullptr` — canvas / ImageBitmap / Image sources go
	// through the image-conversion path further down in this FN which
	// derives `pixels` from `nx_get_image` after the gate. At gate time
	// the local `pixels` is still nullptr from `view_bytes` (returns null
	// for non-ArrayBufferView). Checking the raw JS arg's null-ness
	// catches typed arrays AND image sources uniformly.
	if (!is_v2_context(info) &&
	    (format == 0x1902 /* DEPTH_COMPONENT */ ||
	     format == 0x84F9 /* DEPTH_STENCIL */)) {
		if (target != GL_TEXTURE_2D) {
			record_error(GL_INVALID_OPERATION);
			return;
		}
		if (level != 0) {
			record_error(GL_INVALID_OPERATION);
			return;
		}
		if (!info[8]->IsNullOrUndefined()) {
			record_error(GL_INVALID_OPERATION);
			return;
		}
	}
	// Ledger #69 — ImageBitmap / Image (nx_image_t) source. If the JS arg
	// wraps an nx_image_t, override width/height from the source (WebGL
	// spec: source dimensions win for TexImageSource overloads) and
	// convert premultiplied BGRA into the caller's (format, type) via
	// scratch. Override UNPACK_ALIGNMENT to 1 around the upload so
	// tightly-packed row layouts (e.g. RGB/UNSIGNED_BYTE with odd width,
	// LUMINANCE) don't trip the driver's row-stride check; restore after.
	std::vector<uint8_t> scratch;
	GLint saved_alignment = 4;
	bool alignment_overridden = false;
	if (!pixels) {
		nx_image_t *img = nx_get_image(info.GetIsolate(), info[8]);
		if (img) {
			const bool flip_y = st ? st->unpack_flip_y : false;
			// Ledger #72 — image_bitmap conformance tests do NOT call
			// pixelStorei(UNPACK_PREMULTIPLY_ALPHA_WEBGL, ...); state
			// stays default FALSE. Test expectations assume the bitmap's
			// own premul state is preserved through upload (no driver-side
			// conversion). Our nx_image_t is premultiplied BGRA;
			// un-multiplying with flag=FALSE produces 255,0,0 where test
			// expects 128,0,0 for half-alpha red. Keep source as-is.
			const bool un_premultiply = false;
			uint8_t *conv = convert_image_source_to_gl_pixels(
			    img, format, type, flip_y, un_premultiply, scratch);
			if (conv) {
				width = (GLsizei)img->width;
				height = (GLsizei)img->height;
				pixels = conv;
				glGetIntegerv(GL_UNPACK_ALIGNMENT, &saved_alignment);
				glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
				alignment_overridden = true;
			}
		}
	}
	// Ledger #108 — UNPACK_FLIP_Y_WEBGL honor for typed-array pixel data.
	// Only fires when: (a) pixels came from a real typed array (not the
	// nx_image_t branch above, which already flipped internally); (b) the
	// caller set UNPACK_FLIP_Y_WEBGL=1; (c) width/height are known and > 0.
	// Handles the WebGL2 shim's ImageBitmap-rasterization path used by
	// Three.js's VideoTexture (webgl_materials_video renders upside-down
	// without this fix even with `<video>.flipY=true` default).
	std::vector<uint8_t> flip_scratch;
	if (pixels && !scratch.size() && st && st->unpack_flip_y &&
	    width > 0 && height > 0) {
		const void *flipped = nx_gl_flip_pixels_y(
		    pixels, len, width, height, format, type,
		    st ? st->unpack_alignment : 4, flip_scratch);
		if (flipped) pixels = (void *)flipped;
	}
	bucket_e_translate_tex_image(&internalformat, &format, &type);
	glTexImage2D(target, level, internalformat, width, height, border, format,
	             type, pixels);
	if (alignment_overridden) {
		glPixelStorei(GL_UNPACK_ALIGNMENT, saved_alignment);
	}
}
FN(w_tex_sub_image_2d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLint xoff = a_i32(info, 2);
	const GLint yoff = a_i32(info, 3);
	GLsizei width = a_i32(info, 4);
	GLsizei height = a_i32(info, 5);
	GLenum format = a_u32(info, 6);
	GLenum type = a_u32(info, 7);
	// Ledger #101 — WEBGL_depth_texture (WebGL 1) spec §4.1: texSubImage2D
	// with format ∈ {DEPTH_COMPONENT, DEPTH_STENCIL} is INVALID_OPERATION.
	// Depth textures MUST be uploaded via texImage2D with pixels=null;
	// the driver renders into them via a depth-attachment FBO. Sub-upload
	// would require format conversion the driver can't do. v2 unchanged.
	if (!is_v2_context(info) &&
	    (format == 0x1902 /* DEPTH_COMPONENT */ ||
	     format == 0x84F9 /* DEPTH_STENCIL */)) {
		record_error(GL_INVALID_OPERATION);
		return;
	}
	size_t len = 0;
	void *pixels = view_bytes(info[8], &len);
	if (info[8]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[8].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	// Ledger #69 — ImageBitmap / Image source path mirrors w_tex_image_2d
	// (no internalformat here — storage was allocated by a prior
	// texImage2D/texStorage2D). Sub-uploads still override width/height
	// from the source: WebGL spec allows either the explicit args or the
	// source dimensions, and Three.js / conformance both pass the source's
	// natural dimensions.
	std::vector<uint8_t> scratch;
	GLint saved_alignment = 4;
	bool alignment_overridden = false;
	if (!pixels) {
		nx_image_t *img = nx_get_image(info.GetIsolate(), info[8]);
		if (img) {
			const bool flip_y = st ? st->unpack_flip_y : false;
			const bool un_premultiply = false; // Ledger #72 — see w_tex_image_2d.
			uint8_t *conv = convert_image_source_to_gl_pixels(
			    img, format, type, flip_y, un_premultiply, scratch);
			if (conv) {
				width = (GLsizei)img->width;
				height = (GLsizei)img->height;
				pixels = conv;
				glGetIntegerv(GL_UNPACK_ALIGNMENT, &saved_alignment);
				glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
				alignment_overridden = true;
			}
		}
	}
	// Ledger #108 — UNPACK_FLIP_Y_WEBGL honor for typed-array pixel data.
	// Sibling of the w_tex_image_2d branch; see there for rationale.
	std::vector<uint8_t> flip_scratch;
	if (pixels && !scratch.size() && st && st->unpack_flip_y &&
	    width > 0 && height > 0) {
		const void *flipped = nx_gl_flip_pixels_y(
		    pixels, len, width, height, format, type,
		    st ? st->unpack_alignment : 4, flip_scratch);
		if (flipped) pixels = (void *)flipped;
	}
	bucket_e_translate_tex_sub_image(&format, &type);
	glTexSubImage2D(target, level, xoff, yoff, width, height, format, type,
	                pixels);
	if (alignment_overridden) {
		glPixelStorei(GL_UNPACK_ALIGNMENT, saved_alignment);
	}
}

// Tier 4 (ledger #65) — validation gate for compressed-texture uploads.
//
// Maps each sized compressed internalformat to the extension token that must
// currently be advertised (via `has_native_ext`) for the format to be legal
// input to `glCompressedTexImage2D` / `glCompressedTexSubImage2D`. Returns
// true if the format IS advertised (either by a driver-gated extension or
// as ES3 core, e.g. ETC2/EAC), false otherwise.
//
// Why this gate exists. The WebGL 1 conformance corpus contains a helper
// `testCompressedFormatsUnavailableWhenExtensionDisabled` that intentionally
// calls `glCompressedTexImage2D(target, 0, <unadvertised-format>, 4, 4, 0,
// data)` and asserts the call returns INVALID_ENUM instead of doing the
// upload. On Citron running Mesa-Nouveau, an unadvertised compressed format
// does NOT return INVALID_ENUM at the driver — the WebGL runner hangs inside
// `runOneTest` before the test's START diag fires (CITRON-observed hang;
// hardware stall behavior unverified — Citron is a functional-iteration
// authority, never a driver-truth authority per the house rule reinforced
// in the 2026-07-04 cliff diagnosis). Independent of whether real Tegra
// stalls, the fix is spec-required: WebGL implementations MUST validate the
// internalformat against the currently-advertised extension set before
// dispatching to the driver. This gate makes us spec-compliant regardless
// of driver behavior.
//
// Format-token → advertising extension map (all sized ES3 internalformats):
//   0x83F0..3         S3TC  DXT1/DXT1a/DXT3/DXT5     GL_EXT_texture_compression_s3tc
//   0x8C4C..F         S3TC sRGB  DXT1/DXT1a/DXT3/DXT5 GL_EXT_texture_compression_s3tc_srgb
//   0x8D64            ETC1 RGB8                       GL_OES_compressed_ETC1_RGB8_texture
//   0x9270..9         ETC2 / EAC (10 formats)          ES3 core (always allowed)
//   0x8DBB..E         RGTC (4 formats)                 GL_EXT_texture_compression_rgtc
//   0x8E8C..F         BPTC (4 formats)                 GL_EXT_texture_compression_bptc
//   0x93B0..D         ASTC LDR (14 formats)            GL_KHR_texture_compression_astc_ldr
//   0x93D0..D         ASTC sRGB (14 formats)           GL_KHR_texture_compression_astc_ldr
//
// PVRTC (0x8C00..3) is deliberately absent — Mesa-Nouveau does not expose
// GL_IMG_texture_compression_pvrtc, so any PVRTC format is unadvertised and
// falls through to the `default` branch (returns false → INVALID_ENUM).
static bool has_compressed_format_advertised(GLenum internalformat) {
	switch (internalformat) {
	// S3TC
	case 0x83F0:
	case 0x83F1:
	case 0x83F2:
	case 0x83F3:
		return has_native_ext("GL_EXT_texture_compression_s3tc");
	// S3TC sRGB
	case 0x8C4C:
	case 0x8C4D:
	case 0x8C4E:
	case 0x8C4F:
		return has_native_ext("GL_EXT_texture_compression_s3tc_srgb");
	// ETC1
	case 0x8D64:
		return has_native_ext("GL_OES_compressed_ETC1_RGB8_texture");
	// ETC2 / EAC — ES3 core; no ext gate. These are always allowed.
	case 0x9270:
	case 0x9271:
	case 0x9272:
	case 0x9273:
	case 0x9274:
	case 0x9275:
	case 0x9276:
	case 0x9277:
	case 0x9278:
	case 0x9279:
		return true;
	// RGTC
	case 0x8DBB:
	case 0x8DBC:
	case 0x8DBD:
	case 0x8DBE:
		return has_native_ext("GL_EXT_texture_compression_rgtc");
	// BPTC
	case 0x8E8C:
	case 0x8E8D:
	case 0x8E8E:
	case 0x8E8F:
		return has_native_ext("GL_EXT_texture_compression_bptc");
	// ASTC LDR (14 sized formats)
	case 0x93B0: case 0x93B1: case 0x93B2: case 0x93B3:
	case 0x93B4: case 0x93B5: case 0x93B6: case 0x93B7:
	case 0x93B8: case 0x93B9: case 0x93BA: case 0x93BB:
	case 0x93BC: case 0x93BD:
	// ASTC sRGB (14 sized formats)
	case 0x93D0: case 0x93D1: case 0x93D2: case 0x93D3:
	case 0x93D4: case 0x93D5: case 0x93D6: case 0x93D7:
	case 0x93D8: case 0x93D9: case 0x93DA: case 0x93DB:
	case 0x93DC: case 0x93DD:
		return has_native_ext("GL_KHR_texture_compression_astc_ldr");
	default:
		// Any format not in the map — including PVRTC and any future
		// unaudited codec — is treated as unadvertised. WebGL implementations
		// MUST return INVALID_ENUM on this path per the conformance corpus's
		// testCompressedFormatsUnavailableWhenExtensionDisabled contract.
		return false;
	}
}

// Phase-1 batch-1 — compressed texture 2D uploads. Native
// glCompressedTexImage2D takes an explicit imageSize argument; in WebGL
// this comes from the ArrayBufferView's byteLength.
// WebGL1 signature (7 args): (target, level, internalformat, width, height,
// border, data). WebGL2 adds 8/9-arg PBO / typed-array-offset overloads
// (imageSize + offset / srcOffset + srcLengthOverride) — deferred to a
// later cut. Batch-1 supports the WebGL1 7-arg form on both context types;
// spec-legal (v2 falls back to the ArrayBufferView shape when the caller
// doesn't pass the extra args). Ledger row per plan §3.1's discovered
// missing-native gap.
//
// Tier 4 (ledger #65) — compressed-format validation gate. Rejects
// unadvertised sized compressed internalformats with INVALID_ENUM BEFORE
// touching the driver. Prevents CITRON-observed hangs in
// testCompressedFormatsUnavailableWhenExtensionDisabled (hardware stall
// behavior unverified per Citron / driver-authority house rule).
FN(w_compressed_tex_image_2d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLenum internalformat = a_u32(info, 2);
	const GLsizei width = a_i32(info, 3);
	const GLsizei height = a_i32(info, 4);
	const GLint border = a_i32(info, 5);
	// Ledger #65 gate — return INVALID_ENUM without touching the driver if
	// the requested internalformat is not currently advertised. Spec-required
	// and dodges the CITRON-observed hang in
	// testCompressedFormatsUnavailableWhenExtensionDisabled.
	if (!has_compressed_format_advertised(internalformat)) {
		record_error(GL_INVALID_ENUM);
		return;
	}
	size_t len = 0;
	void *pixels = view_bytes(info[6], &len);
	if (info[6]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[6].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	if (info[6]->IsNullOrUndefined()) { pixels = nullptr; len = 0; }
	glCompressedTexImage2D(target, level, internalformat, width, height,
	                       border, (GLsizei)len, pixels);
}
FN(w_compressed_tex_sub_image_2d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLint xoff = a_i32(info, 2);
	const GLint yoff = a_i32(info, 3);
	const GLsizei width = a_i32(info, 4);
	const GLsizei height = a_i32(info, 5);
	const GLenum format = a_u32(info, 6);
	// Ledger #65 gate — mirrors the w_compressed_tex_image_2d branch. The
	// format arg here is spec-named `format` (parity with texSubImage2D) but
	// it identifies the same sized compressed internalformat as the allocated
	// storage, so the same advertising check applies. Spec-required
	// INVALID_ENUM on unadvertised formats.
	if (!has_compressed_format_advertised(format)) {
		record_error(GL_INVALID_ENUM);
		return;
	}
	size_t len = 0;
	void *pixels = view_bytes(info[7], &len);
	if (info[7]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[7].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	if (info[7]->IsNullOrUndefined()) { pixels = nullptr; len = 0; }
	glCompressedTexSubImage2D(target, level, xoff, yoff, width, height,
	                          format, (GLsizei)len, pixels);
}

// Phase 2.G.1 cut #25 (2026-07-01) — bound because cube-route-shim's
// cube-RT-readback rescue needs it to blit scratch → atlas at face
// offset (per NXJS_PATCHES_NEEDED.md #24). Never bound in QuickJS-era
// either; not load-bearing for any current v1/v2 demo OTHER than the
// runtime rescue's flush path, so this is a pure additive spec-hole
// close. Same signature as glCopyTexSubImage2D core-GLES2 — reads from
// current READ_FRAMEBUFFER (or FRAMEBUFFER on ES2) at (x,y) w×h and
// writes to the currently-bound target texture at (xoffset,yoffset).
FN(w_copy_tex_sub_image_2d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLint xoff = a_i32(info, 2);
	const GLint yoff = a_i32(info, 3);
	const GLint x = a_i32(info, 4);
	const GLint y = a_i32(info, 5);
	const GLsizei width = a_i32(info, 6);
	const GLsizei height = a_i32(info, 7);
	glCopyTexSubImage2D(target, level, xoff, yoff, x, y, width, height);
}

// Phase 2.G.1 cut #27 (2026-07-01) — cube-route-shim needs blitFramebuffer
// to Y-flip the scratch → atlas copy in one GPU-side pass (avoids CPU
// readPixels/texSubImage2D roundtrip and its per-format bpp mapping). WebGL2
// only (glBlitFramebuffer is core GLES3, not ES2). Passed directly through
// to the GLES3 entrypoint; Mesa-Nouveau supports Y-flip via reversed dst
// rectangle (dstY0 > dstY1). Signature matches WebGL2 spec 1:1.
FN(w_blit_framebuffer) {
	enter_bracket();
	const GLint sx0 = a_i32(info, 0);
	const GLint sy0 = a_i32(info, 1);
	const GLint sx1 = a_i32(info, 2);
	const GLint sy1 = a_i32(info, 3);
	const GLint dx0 = a_i32(info, 4);
	const GLint dy0 = a_i32(info, 5);
	const GLint dx1 = a_i32(info, 6);
	const GLint dy1 = a_i32(info, 7);
	const GLbitfield mask = a_u32(info, 8);
	const GLenum filter = a_u32(info, 9);
	glBlitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
}

// Phase 2.G.1 cut #3 — v2-only method impls. Grouped before texImage3D to
// keep all v2-only additions in one contiguous block of the file.
//
// VAO (cut #3a) — Three.js v2 uses VAOs unconditionally for every Mesh.
// Four spec methods: create / delete / is / bind. Direct GLES3 passthroughs;
// the K_VERTEX_ARRAY_OBJECT handle kind (declared above K_COUNT) carries
// the GL name through the JS object via the existing GLObj wrapper.
FN(w_create_vertex_array) {
	GLuint v = 0;
	glGenVertexArrays(1, &v);
	info.GetReturnValue().Set(new_gl_obj(info.GetIsolate(),
	                                      K_VERTEX_ARRAY_OBJECT, v));
}
FN(w_delete_vertex_array) {
	GLuint id = obj_id(info[0]);
	if (id) {
		glDeleteVertexArrays(1, &id);
		erase_wrapper_cache(K_VERTEX_ARRAY_OBJECT, id);
	}
}
FN(w_is_vertex_array) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsVertexArray(obj_id(info[0])) == GL_TRUE));
}
FN(w_bind_vertex_array) {
	enter_bracket();
	const GLuint v = obj_id(info[0]);
	glBindVertexArray(v);
	if (st) st->user_snap.vao = (GLint)v;
}

// UBO core (cut #3b) — minimum surface for Three.js's UBO setup: index
// query + binding-point assignment + buffer-to-binding bind. The webgl2-ubo
// demo calls this pair after THREE.UniformsGroup creation:
//   const idx = gl.getUniformBlockIndex(program, 'ViewData');
//   gl.uniformBlockBinding(program, idx, 0);
//   gl.bindBufferBase(GL_UNIFORM_BUFFER, 0, viewDataBuffer);
// `bindBufferRange` is the offset+size variant; bound in case Three.js's
// uniformsGroup uses it for sub-region UBO uploads (it doesn't by default
// in r184 but cheap to ship together).
FN(w_bind_buffer_base) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLuint index = a_u32(info, 1);
	const GLuint buffer = obj_id(info[2]);
	glBindBufferBase(target, index, buffer);
}
FN(w_bind_buffer_range) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLuint index = a_u32(info, 1);
	const GLuint buffer = obj_id(info[2]);
	const GLintptr offset = (GLintptr)a_i64(info, 3);
	const GLsizeiptr size = (GLsizeiptr)a_i64(info, 4);
	glBindBufferRange(target, index, buffer, offset, size);
}
FN(w_get_uniform_block_index) {
	Isolate *iso = info.GetIsolate();
	const GLuint prog = obj_id(info[0]);
	String::Utf8Value name(iso, info[1]);
	const char *cn = *name ? *name : "";
	GLuint idx = glGetUniformBlockIndex(prog, cn);
	info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, idx));
}
FN(w_uniform_block_binding) {
	enter_bracket();
	const GLuint prog = obj_id(info[0]);
	const GLuint blockIndex = a_u32(info, 1);
	const GLuint blockBinding = a_u32(info, 2);
	glUniformBlockBinding(prog, blockIndex, blockBinding);
}

// texStorage2D (cut #3c) — Three.js v2 prefers immutable texture storage;
// `texStorage2D` allocates ALL mip levels at once and freezes the dimensions.
// Subsequent uploads must go through texSubImage2D. Five-arg passthrough.
FN(w_tex_storage_2d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLsizei levels = a_i32(info, 1);
	const GLenum internalformat = a_u32(info, 2);
	const GLsizei width = a_i32(info, 3);
	const GLsizei height = a_i32(info, 4);
	glTexStorage2D(target, levels, internalformat, width, height);
}

// Instanced drawing (cut #4) — Three.js's InstancedMesh path uses
// drawElementsInstanced + vertexAttribDivisor for per-instance attribute
// stepping. drawArraysInstanced is the unindexed sibling, shipped together.
// All three are direct GLES3 passthroughs. Both draw impls use touch_fbo()
// (defined above) to mark the bridge FBO dirty when drawing into the default
// target — matches the v1 w_draw_arrays / w_draw_elements pattern.
FN(w_draw_arrays_instanced) {
	enter_bracket();
	glDrawArraysInstanced(a_u32(info, 0), a_i32(info, 1), a_i32(info, 2),
	                      a_i32(info, 3));
	touch_fbo();
}
FN(w_draw_elements_instanced) {
	enter_bracket();
	glDrawElementsInstanced(a_u32(info, 0), a_i32(info, 1), a_u32(info, 2),
	                        (const void *)(intptr_t)a_i64(info, 3),
	                        a_i32(info, 4));
	touch_fbo();
}
FN(w_vertex_attrib_divisor) {
	enter_bracket();
	glVertexAttribDivisor(a_u32(info, 0), a_u32(info, 1));
}

// drawBuffers (cut #5) — Three.js v2's WebGLState calls
// `gl.drawBuffers([gl.BACK])` on the very first render's bindFramebuffer
// transition (defaultDrawbuffers starts [] so the BACK-set is unconditional
// on the first frame; subsequent frames are gated by needsUpdate). Without
// this method, calling it through the demo's Proxy resolves to `undefined`
// → TypeError "drawBuffers is not a function" → which Three.js's `state`
// function does NOT wrap in try/catch (unlike texStorage2D/texImage2D/etc.
// which ARE wrapped), so the throw propagates up out of renderer.render()
// and aborts the render path; subsequent frames continue ticking the
// animate loop (animation tween + fps counter), masking it. Symptom:
// canvas shows the clear color only (the gl.clear that runs BEFORE
// state.drawBuffers in the frame), no geometry rasterizes, no JS error
// surfaces because the Three.js try/catch upstream isn't on this path.
// GLES3 signature: glDrawBuffers(GLsizei n, const GLenum *bufs).
//
// Ledger #109 (2026-07-10) — translate `GL_BACK` to `GL_COLOR_ATTACHMENT0`
// when the JS view says "default framebuffer" (`bound_fbo_js == 0`). Per
// ES3 §16.1.4, `BACK` is only valid for the true default framebuffer
// (name 0); on a user FBO — which our tenant is, since we redirect the
// JS null-bind to a real user FBO with `COLOR_ATTACHMENT0` — passing
// `BACK` is INVALID_OPERATION. Symptom the fix cures: Three.js's
// EffectComposer post-processing pipeline (`RenderPass` + `BloomPass` +
// `OutputPass`) renders black because OutputPass's `setRenderTarget(null)`
// triggers `state.drawBuffers(null, null)` which emits `[BACK]`. The
// errored `glDrawBuffers` call combined with the subsequent draw ends
// up not writing to the tenant (empirically observed even though ES3
// spec says the state should stay unchanged on error — Mesa Nouveau's
// behavior in the errored path is what matters here, not the spec).
// The translation makes the WebGL 2 tenant model spec-conformant from
// the user's perspective: `[BACK]` on the "default" framebuffer maps to
// the tenant's `COLOR_ATTACHMENT0`, which IS the tenant's back buffer
// from the user's POV.
FN(w_draw_buffers) {
	enter_bracket();
	std::vector<int32_t> tmp;
	const int32_t *p = nullptr;
	size_t n = 0;
	if (!i32_list(info.GetIsolate(), info[0], tmp, &p, &n)) return;
	// Ledger #109 — when JS-null is bound (redirected to tenant), translate
	// any GL_BACK to GL_COLOR_ATTACHMENT0. Buffer array is small (typically
	// 1 entry); inline scratch avoids an allocation on the fast path.
	GLenum bufs_scratch[16];
	const GLenum *out = (const GLenum *)p;
	if (st && st->bound_fbo_js == 0 && n > 0 && n <= 16) {
		bool needs_translate = false;
		for (size_t i = 0; i < n; ++i) {
			if ((GLenum)p[i] == GL_BACK) { needs_translate = true; break; }
		}
		if (needs_translate) {
			for (size_t i = 0; i < n; ++i) {
				bufs_scratch[i] = ((GLenum)p[i] == GL_BACK)
				                       ? GL_COLOR_ATTACHMENT0
				                       : (GLenum)p[i];
			}
			out = bufs_scratch;
		}
	}
	glDrawBuffers((GLsizei)n, out);
}

// Phase 2.G.1 cut #2 — texImage3D for WebGL2. Three.js's v2 path calls this
// at WebGLRenderer init time to create 1×1 placeholder textures for the
// default-bound TEXTURE_3D / TEXTURE_2D_ARRAY samplers (the v2 analog of v1's
// _emptyCubeTexture). Without this binding the demo throws before reaching
// its scene setup. Signature: (target, level, internalformat, width, height,
// depth, border, format, type, pixels). Mirrors w_tex_image_2d but with
// `depth` between height and border. No bucket-E format widening yet — the
// Three.js call site uses canonical RGBA+UByte which Mesa Nouveau accepts
// directly; format-widening for v2 3D-texture uploads is a future cut if/
// when a demo surfaces a probe-rejected format.
FN(w_tex_image_3d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	GLint internalformat = a_i32(info, 2);
	const GLsizei width = a_i32(info, 3);
	const GLsizei height = a_i32(info, 4);
	const GLsizei depth = a_i32(info, 5);
	const GLint border = a_i32(info, 6);
	GLenum format = a_u32(info, 7);
	GLenum type = a_u32(info, 8);
	size_t len = 0;
	void *pixels = view_bytes(info[9], &len);
	if (info[9]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[9].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	if (info[9]->IsNullOrUndefined()) pixels = nullptr;
	glTexImage3D(target, level, internalformat, width, height, depth, border,
	             format, type, pixels);
}

// Phase 2.G.1 cut #32 (2026-07-01) — bound because Three.js r184's WebGL2
// backend unconditionally calls state.texStorage3D + state.texSubImage3D
// for DataArrayTexture/Data3DTexture uploads (WebGLTextures.js:1174/1190/
// 1198). Both wrappers try/catch and silently swallow "gl.texStorage3D is
// not a function" errors, so without these bindings the array-texture
// storage is never allocated → sampler2DArray/sampler3D reads return
// vec4(0) → webgl2-texture2darray renders black on both Citron and
// hardware. Direct passthrough — no format massaging required (GLES3
// spec matches WebGL2 spec 1:1 for these entry points).
FN(w_tex_storage_3d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLsizei levels = a_i32(info, 1);
	const GLenum internalformat = a_u32(info, 2);
	const GLsizei width = a_i32(info, 3);
	const GLsizei height = a_i32(info, 4);
	const GLsizei depth = a_i32(info, 5);
	glTexStorage3D(target, levels, internalformat, width, height, depth);
}

FN(w_tex_sub_image_3d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLint xoff = a_i32(info, 2);
	const GLint yoff = a_i32(info, 3);
	const GLint zoff = a_i32(info, 4);
	const GLsizei width = a_i32(info, 5);
	const GLsizei height = a_i32(info, 6);
	const GLsizei depth = a_i32(info, 7);
	const GLenum format = a_u32(info, 8);
	const GLenum type = a_u32(info, 9);
	size_t len = 0;
	void *pixels = view_bytes(info[10], &len);
	if (info[10]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[10].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	if (info[10]->IsNullOrUndefined()) pixels = nullptr;
	glTexSubImage3D(target, level, xoff, yoff, zoff, width, height, depth,
	                format, type, pixels);
}

// ----- Framebuffer / Renderbuffer -----
FN(w_create_framebuffer) {
	GLuint f = 0;
	glGenFramebuffers(1, &f);
	// Ledger #95 — see w_create_buffer.
	info.GetReturnValue().Set(
	    new_gl_obj_create(info.GetIsolate(), K_FRAMEBUFFER, f));
}
FN(w_delete_framebuffer) {
	// Ledger #95 — see w_delete_buffer for the wrapper-lifecycle rationale.
	// Also: WebGL 1 spec §5.14.2 (mirroring GLES 2 §4.4.4) — if the deleted
	// framebuffer was currently bound, the binding falls back to the
	// default framebuffer (name 0). Our shim redirects "JS-null / 0" to
	// the tenant FBO via `bound_fbo_js == 0` in enter_bracket; without
	// resetting the shadow here, subsequent JS calls' enter_bracket would
	// glBindFramebuffer(deletedName) and Mesa-Nouveau re-conjures a fresh
	// empty FBO under that name — every draw / clear / readPixels then
	// hits an incomplete FBO and the observable state (colors on the
	// tenant surface, `getFramebufferAttachmentParameter` etc.) diverges
	// from spec. Fix: fall the shadow back to default on delete-of-bound.
	GLObj *o = get_gl_obj(info[0]);
	if (o && o->id) {
		if (st && st->bound_fbo_js == o->id) {
			st->bound_fbo_js = 0;
			st->draw_into_default = true;
			// user_snap.fbo is the GL-side actual (post-redirect) name we
			// re-emit at every enter_bracket. Steer it back to the tenant
			// FBO id so the next bracket doesn't glBindFramebuffer(0)
			// (native default, ≠ tenant) and lose the redirect.
			st->user_snap.fbo = (GLint)cur_tenant_fbo();
			st->user_snap.read_fbo = (GLint)cur_tenant_fbo();
		}
		glDeleteFramebuffers(1, &o->id);
		o->deleted = true;
	}
}
FN(w_is_framebuffer) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsFramebuffer(obj_id(info[0])) == GL_TRUE));
}
FN(w_bind_framebuffer) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	// Ledger #95 — see w_bind_buffer.
	if (obj_deleted(info[1])) {
		record_error(GL_INVALID_OPERATION);
		return;
	}
	GLuint fbo = obj_id(info[1]);
	if (st) {
		st->bound_fbo_js = fbo;
		st->draw_into_default = (fbo == 0);
	}
	// JS sees null/0 as "default" framebuffer; we redirect to THIS canvas's
	// tenant FBO (multi-WebGL-canvas independence).
	GLuint actual = (fbo == 0) ? cur_tenant_fbo() : fbo;
	glBindFramebuffer(target, actual);
	if (st) {
		// GL_FRAMEBUFFER binds both draw and read; the other two target
		// only one endpoint.
		if (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER) {
			st->user_snap.fbo = (GLint)actual;
		}
		if (target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) {
			st->user_snap.read_fbo = (GLint)actual;
		}
	}
}
FN(w_framebuffer_texture_2d) {
	enter_bracket();
	glFramebufferTexture2D(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2),
	                       obj_id(info[3]), a_i32(info, 4));
}
FN(w_framebuffer_renderbuffer) {
	enter_bracket();
	glFramebufferRenderbuffer(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2),
	                          obj_id(info[3]));
}
FN(w_check_framebuffer_status) {
	enter_bracket();
	info.GetReturnValue().Set(Uint32::NewFromUnsigned(info.GetIsolate(),
	    glCheckFramebufferStatus(a_u32(info, 0))));
}
FN(w_create_renderbuffer) {
	GLuint r = 0;
	glGenRenderbuffers(1, &r);
	// Ledger #95 — see w_create_buffer.
	info.GetReturnValue().Set(
	    new_gl_obj_create(info.GetIsolate(), K_RENDERBUFFER, r));
}
FN(w_delete_renderbuffer) {
	// Ledger #95 — see w_delete_buffer.
	GLObj *o = get_gl_obj(info[0]);
	if (o && o->id) {
		glDeleteRenderbuffers(1, &o->id);
		o->deleted = true;
	}
}
FN(w_is_renderbuffer) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsRenderbuffer(obj_id(info[0])) == GL_TRUE));
}
FN(w_bind_renderbuffer) {
	enter_bracket();
	// Ledger #95 — see w_bind_buffer.
	if (obj_deleted(info[1])) {
		record_error(GL_INVALID_OPERATION);
		return;
	}
	glBindRenderbuffer(a_u32(info, 0), obj_id(info[1]));
}
FN(w_renderbuffer_storage) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	GLenum internalformat = a_u32(info, 1);
	const GLsizei width = a_i32(info, 2);
	const GLsizei height = a_i32(info, 3);
	// Ledger #100 — WebGL 1's `DEPTH_STENCIL` (0x84F9) is an unsized
	// enum that GLES 2 native rejects with INVALID_ENUM (packed depth-
	// stencil is only available via GL_OES_packed_depth_stencil which
	// uses `DEPTH24_STENCIL8_OES` = 0x88F0). Translate the unsized
	// WebGL 1 token to the OES/ES3-sized one so Mesa Nouveau (ES3.2)
	// accepts it. Same class as #10's SRGB / HALF_FLOAT translations
	// for texImage2D — WebGL 1's unsized enums silently rewritten to
	// their ES3-core sized equivalents.
	if (internalformat == 0x84F9 /* GL_DEPTH_STENCIL */) {
		internalformat = 0x88F0; /* GL_DEPTH24_STENCIL8 */
	}
	glRenderbufferStorage(target, internalformat, width, height);
}
// Ledger #100 — WebGL 1 spec §5.14.7: `getRenderbufferParameter(target,
// pname)` returns an integer for every supported pname (WIDTH, HEIGHT,
// INTERNAL_FORMAT, RED/GREEN/BLUE/ALPHA/DEPTH/STENCIL_SIZE, SAMPLES,
// STENCIL). No object-returning cases on WebGL 1. Wraps
// glGetRenderbufferParameteriv unchanged; error handling is via the
// driver's own pname validation (INVALID_ENUM for bad pname, INVALID_
// OPERATION when no renderbuffer bound to target).
FN(w_get_renderbuffer_parameter) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLenum pname = a_u32(info, 1);
	GLint value = 0;
	glGetRenderbufferParameteriv(target, pname, &value);
	// Ledger #100 — round-trip translation for RENDERBUFFER_INTERNAL_
	// FORMAT: w_renderbuffer_storage translates WebGL 1's `DEPTH_STENCIL`
	// (0x84F9) → GLES 2 native's `DEPTH24_STENCIL8` (0x88F0). The query
	// side has to reverse the mapping so WebGL 1 apps see the token they
	// originally passed. The v1 spec §5.14.7 example expects the storage
	// query to return the WebGL 1 token, not the ES3-sized one.
	if (pname == 0x8D44 /* GL_RENDERBUFFER_INTERNAL_FORMAT */ &&
	    value == 0x88F0 /* GL_DEPTH24_STENCIL8 */) {
		value = 0x84F9; /* GL_DEPTH_STENCIL */
	}
	info.GetReturnValue().Set(Int32::New(info.GetIsolate(), value));
}

// ----- Draw -----

FN(w_draw_arrays) {
	enter_bracket();
	// Ledger #100 — see nx_fbo_complete_or_record_error.
	if (!nx_fbo_complete_or_record_error()) return;
	glDrawArrays(a_u32(info, 0), a_i32(info, 1), a_i32(info, 2));
	touch_fbo();
}
FN(w_draw_elements) {
	enter_bracket();
	if (!nx_fbo_complete_or_record_error()) return;
	glDrawElements(a_u32(info, 0), a_i32(info, 1), a_u32(info, 2),
	               (const void *)(intptr_t)a_i64(info, 3));
	touch_fbo();
}

// ----- Readback -----
FN(w_read_pixels) {
	enter_bracket();
	// Ledger #100 — see nx_fbo_complete_or_record_error. Mesa Nouveau
	// does NOT emit INVALID_FRAMEBUFFER_OPERATION for readPixels on an
	// incomplete FBO (unlike its behavior for `clear`), so the client-
	// side gate is load-bearing here specifically.
	if (!nx_fbo_complete_or_record_error()) return;
	const GLint x = a_i32(info, 0);
	const GLint y = a_i32(info, 1);
	const GLsizei width = a_i32(info, 2);
	const GLsizei height = a_i32(info, 3);
	const GLenum format = a_u32(info, 4);
	const GLenum type = a_u32(info, 5);
	size_t len = 0;
	void *pixels = view_bytes(info[6], &len);
	if (pixels) glReadPixels(x, y, width, height, format, type, pixels);
}

// ----- Fork-specific hooks required by brewser-runtime canvas-runner -----
FN(w_enable_gpu_bridge_prototype) {
	// Canvas-runner refuses to use the context if this method is absent
	// (see brewser-runtime canvas-runner.ts:278). The QuickJS-era fork's
	// bridge had this as a runtime toggle; the V8-migration bridge is always
	// on (the only path that exists), so this is a no-op that returns true.
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), true));
}
FN(w_set_bridge_auto_flush) {
	// brewser-runtime canvas-runner.ts calls gl.setBridgeAutoFlush(false) to
	// tell the engine "I drive my own per-canvas paint via copyBridgeToCanvas
	// at each <canvas>'s CSS layout slot — do NOT auto-stomp the whole FBO
	// onto the screen on top of that". Forward the bool to the bridge;
	// default-true (the no-arg / non-bool call) preserves the 2.B test-FBO
	// smoke path that never calls this.
	Isolate *iso = info.GetIsolate();
	bool v = true;
	if (info.Length() >= 1)
		v = info[0]->BooleanValue(iso);
	nx_webgl_bridge_set_auto_flush(v);
	info.GetReturnValue().Set(Boolean::New(iso, true));
}

FN(w_copy_bridge_to_canvas) {
	// gl.copyBridgeToCanvas(srcX, srcY, srcW, srcH, dst_canvas, dstX, dstY)
	//   → bool ok
	//
	// Engine-side half of the runtime's per-canvas paint path. The runtime
	// (brewser-runtime canvas-runner.ts → overlayLiveAnimatedCanvases)
	// computes each inline <canvas>'s CSS layout slot and calls this to blit
	// that sub-rect of the shared bridge FBO onto the destination canvas
	// surface at the slot's screen-coord position. Pair with
	// setBridgeAutoFlush(false) so the engine's whole-FBO compose path stays
	// out of the way.
	//
	// Phase 2.D scope: dst_canvas is ignored — the only destination the
	// runtime passes is screen (nxScreen()), and the bridge composes onto
	// the persistent canvas surface from nx_skia_gpu_canvas_surface(). When
	// Phase 2.E / 2.G needs per-OffscreenCanvas destinations, extract the
	// dst_canvas's Skia surface from info[4].
	Isolate *iso = info.GetIsolate();
	if (info.Length() < 7) {
		info.GetReturnValue().Set(Boolean::New(iso, false));
		return;
	}
	const int sx = a_i32(info, 0);
	const int sy = a_i32(info, 1);
	const int sw = a_i32(info, 2);
	const int sh = a_i32(info, 3);
	const int dx = a_i32(info, 5);
	const int dy = a_i32(info, 6);

	// Close any open per-frame WebGL bracket so Skia draws against its own
	// cached GL state, not the WebGL pass's (FBO/viewport/program/etc. saved
	// in g_snap). Same discipline nx_webgl_compose_if_active uses at
	// present time.
	if (g_bracket_open) exit_bracket();

	SkSurface *target = nx_skia_gpu_canvas_surface();
	if (!target) {
		info.GetReturnValue().Set(Boolean::New(iso, false));
		return;
	}
	// Compose THIS canvas's own tenant FBO into its DOM slot (use_ctx set `st`
	// from info.This()), so stacked WebGL canvases each show their own pixels.
	const uint32_t tid = st ? st->tenant_id : 0;
	const bool ok = nx_webgl_bridge_tenant_compose_rect(target, tid, sx, sy, sw,
	                                                     sh, dx, dy);
	info.GetReturnValue().Set(Boolean::New(iso, ok));
}

// Runner-driven resetSharedContext hook. The full-webgl1-conformance runner
// (D:/Workspace/brewser-apps/apps/experimental/com.natureglass.webglconformtest/
// full-webgl1-conformance/assets/runner.js:3340) calls this every 25 test
// iterations to relieve cumulative Skia + GL pressure that otherwise OOMs the
// run around test #325 (V8 heap allocation failure inside renderStatus,
// classically a ~217KB font/glyph-atlas paint alloc). The QuickJS-era
// resetSharedContext destroyed the WebGL EGL context (see
// D:/Workspace/nxjs-source/source/webgl.c:14422 + webgl_egl.c:3087) but this
// bridge shares Skia's EGL context per Phase 2.A, so the QuickJS pattern would
// take Skia's rendering down with it. Instead, do the three cheap operations
// that actually recover the failing allocation site:
//   1. Purge Skia's Ganesh caches (atlas + glyph cache + unlocked GPU
//      resources). This is the primary recovery — the failing 217KB
//      allocation is a paint-side Skia alloc.
//   2. Reset Ganesh's cached GL-state model so it rebinds its program/VAO on
//      the next paint (WebGL may have left GL state Ganesh still thinks it
//      owns).
//   3. glFlush so the driver retires already-submitted work rather than
//      queueing indefinitely.
//   4. Bump s_gl_reset_generation as a forward-compat hook for future
//      per-object staleness checks; nothing currently reads it, but exporting
//      it via the same mechanism as the QuickJS bridge's context_generation
//      lets us add stale-handle guards later without changing the JS API.
// Returns true on success; false only if Skia isn't up yet (call before
// nx_skia_gpu_screen_init succeeded, or after nx_skia_gpu_screen_exit).
static uint32_t s_gl_reset_generation = 0;
FN(w_reset_shared_context) {
	Isolate *iso = info.GetIsolate();
	GrDirectContext *gr = nx_skia_gpu_gr_context();
	if (!gr) {
		info.GetReturnValue().Set(Boolean::New(iso, false));
		return;
	}
	nx_skia_gpu_free_gpu_resources();
	gr->resetContext();
	glFlush();
	++s_gl_reset_generation;
	info.GetReturnValue().Set(Boolean::New(iso, true));
}

// ---------------------------------------------------------------------------
// Class init: receive prototype carriers + install methods on them.
//
// Phase 2.G.0 — table-split shape DECISION: separate FUNCS[] tables for v1 and
// v2 (this file holds the v1 table; install_methods_v2 below holds the v2
// table). Rationale, against the JIT-safety lesson from NXJS_PATCHES_NEEDED.md
// #8:
//
//   1. The v1 install path is hardware-verified clean (Phase 2.C hardware
//      gate, jit=on default re-enabled by #8 fix). Leaving the v1 FUNCS[]
//      table + install loop UNCHANGED means the JS shape that JIT-tier'd up
//      successfully on Tegra is preserved byte-for-byte. A separate v2
//      table cannot disturb v1's compiled code.
//   2. The v2 install path is its OWN predictable JS shape: same install
//      loop pattern (C++ for-each over a static-storage FUNCS[] with
//      proto->Set + FunctionTemplate::New + GetFunction). The hardware
//      verification step for 2.G.0 is "does this NEW shape JIT-compile
//      clean too?" — and if it doesn't, the regression is bisectable
//      because v1 is untouched.
//   3. Shape mutation discipline. The #8 fix was about a TS-side
//      Object.defineProperty pattern that compiled into a JIT codegen
//      issue; that fix replaced ~1160 per-key calls with bulk
//      Object.defineProperties on each target. v1 and v2 both already
//      use that fixed install shape for constants (webgl-rendering-
//      context.ts:577-585 and webgl2-rendering-context.ts:1102-1110).
//      The METHOD install at install_methods() is engine-side, runs once
//      at boot, NOT TS-JIT-hot — but the same discipline applies: keep
//      shapes predictable, do not gate behavior at install time. Separate
//      tables avoid a "select v1 vs v2 subset at install time" branch
//      that could feed inconsistent function objects into V8's hidden
//      class for the prototype.
//   4. The 9 v2-only extensions (EXT_disjoint_timer_query_webgl2,
//      EXT_texture_norm16, WEBGL_clip_cull_distance, EXT_float_blend,
//      EXT_render_snorm, OES_sample_variables, OES_draw_buffers_indexed,
//      WEBGL_blend_func_extended, WEBGL_compressed_texture_etc) belong
//      in v2's table only; a shared table would have to gate them
//      at runtime, growing the binding surface.
//   5. Three.js detects v1/v2 via gl.constructor.name === 'WebGL{2}
//      RenderingContext' — independent prototype chains match the
//      runtime contract.
//
// COST: when 2.G.1+ ports the ~95 v2-only methods plus duplicates the ~95
// v1-shared methods into the v2 table, the v2 FUNCS[] will be ~190 entries.
// The duplicate v1 entries point at the SAME C++ impl functions; no code
// duplication, only Spec-entry duplication.
//
// For 2.G.0 the v2 table is EMPTY (no methods bound). The v2 prototype
// scaffolding is correct (the bulk-defineProperties constants install on
// the TS side already populates 387 v2 constants on the prototype). Method
// calls on a v2 instance throw `TypeError: X is not a function`.
// ---------------------------------------------------------------------------

// ============================================================================
// Phase-1.5-LOW — 30 core WebGL2 methods (tier LOW per plan §0.1).
// All v2-only; installed on install_methods_v2 FUNCS[] only. Grouped by
// Khronos family for grep-audit + verify-patches.sh check-family
// alignment.
// ============================================================================

// #56b sync-guard helper — force GPU-command completion + buffer-store
// coherency for a subsequent MAP_READ. Used before glMapBufferRange in
// every GPU→CPU readback path that goes through a mapped buffer store.
//
// LEDGER RATIONALE (empirical, NOT spec-guaranteed):
// A spec-conformant GLES3 driver needs neither glFenceSync nor extra
// coherency work beyond what glFinish already provides — glMapBufferRange
// with GL_MAP_READ_BIT implicitly synchronizes per ES3 §2.9.5. This
// helper is an EMPIRICAL workaround for a Mesa Nouveau NV120
// map-coherency bug documented in NXJS_PATCHES_NEEDED #56b (hardware
// smoke #3, strict re-run on cached WebGL2 context: fresh-context first
// readback works with glFinish alone; subsequent readbacks on the same
// live context read stale bytes). The fence path apparently exercises
// a different driver flushing code path that reaches the mapped-region
// cache; glFinish alone happens to skip that flush on re-invocation.
//
// Non-conformant reliance on the fence-flush combo is documented per
// #56b's escalation ladder: if this stops carrying on future hardware,
// rung 2 adds MAP_INVALIDATE_RANGE_BIT cache-bust pre-map; rung 3 is a
// staging-copy through a per-call fresh buffer.
//
// Timeout 100ms — long enough to swallow any real GPU stall on a small
// buffer readback, short enough to keep the JS event loop responsive if
// something has gone catastrophically wrong (log + proceed, don't hang).
static void nx_56b_readback_sync_guard(const char *site_tag) {
	GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
	if (!sync) {
		fprintf(stderr, "[#56b] %s: glFenceSync returned NULL "
		                "— proceeding without fence guard\n", site_tag);
		fflush(stderr);
		return;
	}
	const GLuint64 timeout_ns = 100000000ULL; // 100ms
	GLenum r = glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, timeout_ns);
	if (r == GL_TIMEOUT_EXPIRED) {
		fprintf(stderr, "[#56b] %s: clientWaitSync TIMEOUT after 100ms "
		                "— proceeding to map anyway (see escalation ladder in "
		                "NXJS_PATCHES_NEEDED #56b)\n", site_tag);
		fflush(stderr);
	} else if (r == GL_WAIT_FAILED) {
		fprintf(stderr, "[#56b] %s: clientWaitSync WAIT_FAILED "
		                "— proceeding to map anyway\n", site_tag);
		fflush(stderr);
	}
	glDeleteSync(sync);
}

// #56 fix path — runtime-resolved `glGetBufferSubData`. The GLES3 header
// set on devkitPro doesn't declare glGetBufferSubData (it's a desktop-GL
// entry point), but Mesa's Nouveau NV120 driver exports the symbol at
// runtime — the QuickJS-era pre-migration nx_webgl_egl_get_buffer_sub_data
// (nxjs-source/source/webgl_egl.c:9925) used it via a runtime-resolved
// function pointer and worked on the same hardware. Cached at first
// getBufferSubData call. Diagnostic banner logs on resolution (once per
// boot).
typedef void (GL_APIENTRY *pfn_glGetBufferSubData_t)(GLenum target,
                                                     GLintptr offset,
                                                     GLsizeiptr size,
                                                     void *data);
static pfn_glGetBufferSubData_t s_pfn_get_buffer_sub_data = nullptr;
static bool s_pfn_get_buffer_sub_data_resolved = false;
static void resolve_pfn_get_buffer_sub_data() {
	if (s_pfn_get_buffer_sub_data_resolved) return;
	s_pfn_get_buffer_sub_data_resolved = true;
	s_pfn_get_buffer_sub_data = (pfn_glGetBufferSubData_t)
	    eglGetProcAddress("glGetBufferSubData");
	fprintf(stderr, "[#56] glGetBufferSubData proc-address resolved: %p "
	                "(non-null = hardware fallback available; QuickJS-era "
	                "reference used this successfully on Mesa Nouveau NV120)\n",
	        (void *)(uintptr_t)s_pfn_get_buffer_sub_data);
	fflush(stderr);
}

// ----- Buffer ops (v2 adds) -----
FN(w_get_buffer_sub_data) {
	// getBufferSubData(target, srcByteOffset, dstBuffer, dstOffset?, length?)
	//
	// #56 diagnosis-and-fix (2026-07-03 hardware smoke revealed Mesa
	// Nouveau NV120 silently returns zeros through the glMapBufferRange
	// path for GL_COPY_WRITE_BUFFER — Citron/AMD Vulkan translation of
	// the same path works). Ship BOTH candidate mitigations in a single
	// commit + NX_56_DEBUG instrumentation so the next hardware boot
	// pins down which one carried the fix:
	//
	//   (b) glFinish() before glMapBufferRange — force GPU pipeline
	//       drain. Spec says MAP_READ_BIT implicitly syncs, but Nouveau
	//       drivers historically underimplement this for the copy-and-
	//       read pattern. Perf note: this stalls the GPU on every
	//       getBufferSubData; readback is inherently sync, so the extra
	//       latency is acceptable for a spec-conformance path.
	//
	//   (a) glGetBufferSubData proc-address fallback — if
	//       glMapBufferRange returns NULL, resolve glGetBufferSubData
	//       via eglGetProcAddress and use it directly. Matches the
	//       QuickJS-era reference implementation that worked on the
	//       same hardware.
	//
	// Build with -DNX_56_DEBUG for boot-log instrumentation on each
	// getBufferSubData call (mapped ptr, first 16 bytes, glGetError
	// results before/after map). Standard build stays quiet after the
	// one-shot proc-address resolution banner.
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLintptr src_off = (GLintptr)a_i32(info, 1);
	if (info.Length() < 3) return;
	size_t view_len = 0;
	void *dst = view_bytes(info[2], &view_len);
	if (!dst) return;
	size_t dst_off = 0;
	size_t len = view_len;
	if (info.Length() >= 4 && info[3]->IsNumber())
		dst_off = (size_t)a_i32(info, 3);
	if (info.Length() >= 5 && info[4]->IsNumber())
		len = (size_t)a_i32(info, 4);
	if (dst_off + len > view_len) len = (view_len > dst_off) ?
	    (view_len - dst_off) : 0;
	if (len == 0) return;
	uint8_t *dst_bytes = (uint8_t *)dst + dst_off;

	// #56 candidate (b) — sync barrier before map. Cheap on frame-cost
	// terms; getBufferSubData is inherently a sync operation from the
	// JS perspective (the returned bytes must be up-to-date). RETAINED
	// alongside the #56b fence-guard below: glFinish handles the
	// fresh-context first-readback path (verified 2026-07-03 smoke #3).
	glFinish();

	// #56b sync-guard — force buffer-store coherency for the subsequent
	// map. Handles the write-visibility race on re-invocation that
	// hardware smoke #3 strict-run exposed (verified 2026-07-03: cached
	// WebGL2 context, second BUFFER probe invocation reads stale bytes
	// from mapped region despite full re-execution of bufferData +
	// copyBufferSubData + glFinish). Empirical Nouveau workaround per
	// NXJS_PATCHES_NEEDED #56b — the fence path exercises a different
	// driver flushing code path.
	nx_56b_readback_sync_guard("getBufferSubData");

	// #56 candidate (a1) — per-target rebind fallback for map. 2026-07-03
	// second hardware smoke on Mesa Nouveau NV120 confirmed: Arm A
	// (ARRAY_BUFFER direct readback) PASSes with candidates (a)+(b), but
	// Arm B (COPY_WRITE_BUFFER via copy) still FAILs — the map returns
	// non-NULL but points at unrelated memory (Boot A read got=0, Boot B
	// read got=42; both spurious). Target-specific defect. Workaround:
	// for COPY_WRITE_BUFFER / COPY_READ_BUFFER / PIXEL_PACK_BUFFER, look
	// up the buffer name bound to that target, temporarily rebind to
	// ARRAY_BUFFER (per-target binding — no data movement, driver
	// side-effect-free), map from ARRAY_BUFFER (verified to work by
	// Arm A), unmap, restore original ARRAY_BUFFER binding.
	GLenum map_target = target;
	GLint saved_array_binding = 0;
	bool did_rebind = false;
	if (target == GL_COPY_WRITE_BUFFER || target == GL_COPY_READ_BUFFER ||
	    target == GL_PIXEL_PACK_BUFFER) {
		GLenum binding_pname = 0;
		if (target == GL_COPY_WRITE_BUFFER) binding_pname = GL_COPY_WRITE_BUFFER_BINDING;
		else if (target == GL_COPY_READ_BUFFER) binding_pname = GL_COPY_READ_BUFFER_BINDING;
		else if (target == GL_PIXEL_PACK_BUFFER) binding_pname = GL_PIXEL_PACK_BUFFER_BINDING;
		GLint target_binding = 0;
		glGetIntegerv(binding_pname, &target_binding);
		if (target_binding != 0) {
			glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &saved_array_binding);
			glBindBuffer(GL_ARRAY_BUFFER, (GLuint)target_binding);
			map_target = GL_ARRAY_BUFFER;
			did_rebind = true;
#ifdef NX_56_DEBUG
			fprintf(stderr, "[#56] per-target rebind: target=0x%x bind=%d "
			                "saved_array=%d\n", target, target_binding,
			        saved_array_binding);
			fflush(stderr);
#endif
		}
	}

#ifdef NX_56_DEBUG
	while (glGetError() != GL_NO_ERROR);
#endif
	void *mapped = glMapBufferRange(map_target, src_off, (GLsizeiptr)len,
	                                GL_MAP_READ_BIT);
#ifdef NX_56_DEBUG
	GLenum e_map = glGetError();
	fprintf(stderr, "[#56] glMapBufferRange target=0x%x (mapped as 0x%x) off=%td len=%zu "
	                "-> mapped=%p err=0x%x\n",
	        target, map_target, (ptrdiff_t)src_off, len, mapped, e_map);
	if (mapped) {
		const uint8_t *b = (const uint8_t *)mapped;
		size_t log_len = len < 16 ? len : 16;
		fprintf(stderr, "[#56]   mapped first %zu bytes:", log_len);
		for (size_t i = 0; i < log_len; i++) fprintf(stderr, " %02x", b[i]);
		fprintf(stderr, "\n");
	}
	fflush(stderr);
#endif

	if (mapped) {
		memcpy(dst_bytes, mapped, len);
		glUnmapBuffer(map_target);
		if (did_rebind) {
			glBindBuffer(GL_ARRAY_BUFFER, (GLuint)saved_array_binding);
		}
		return;
	}

	// Restore original ARRAY_BUFFER binding before falling through to the
	// proc-address fallback (which uses the caller's target directly and
	// doesn't need the rebind).
	if (did_rebind) {
		glBindBuffer(GL_ARRAY_BUFFER, (GLuint)saved_array_binding);
	}

	// #56 candidate (a) fallback — glMapBufferRange returned NULL (Mesa
	// Nouveau NV120 may not implement mapping for GL_COPY_WRITE_BUFFER
	// or other v2-only targets). Try glGetBufferSubData via proc-address.
	resolve_pfn_get_buffer_sub_data();
	if (s_pfn_get_buffer_sub_data) {
		s_pfn_get_buffer_sub_data(target, src_off, (GLsizeiptr)len, dst_bytes);
#ifdef NX_56_DEBUG
		GLenum e_read = glGetError();
		fprintf(stderr, "[#56] fallback glGetBufferSubData target=0x%x off=%td "
		                "len=%zu called, err=0x%x\n",
		        target, (ptrdiff_t)src_off, len, e_read);
		const uint8_t *b = (const uint8_t *)dst_bytes;
		size_t log_len = len < 16 ? len : 16;
		fprintf(stderr, "[#56]   readback first %zu bytes:", log_len);
		for (size_t i = 0; i < log_len; i++) fprintf(stderr, " %02x", b[i]);
		fprintf(stderr, "\n");
		fflush(stderr);
#endif
	} else {
#ifdef NX_56_DEBUG
		fprintf(stderr, "[#56] BOTH PATHS FAILED: glMapBufferRange returned "
		                "NULL AND glGetBufferSubData proc-address unresolved. "
		                "dst stays zero-initialized.\n");
		fflush(stderr);
#endif
	}
}
FN(w_copy_buffer_sub_data) {
	// copyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size)
	enter_bracket();
	const GLenum r_target = a_u32(info, 0);
	const GLenum w_target = a_u32(info, 1);
	const GLintptr r_off = (GLintptr)a_i32(info, 2);
	const GLintptr w_off = (GLintptr)a_i32(info, 3);
	const GLsizeiptr size = (GLsizeiptr)a_i32(info, 4);
	glCopyBufferSubData(r_target, w_target, r_off, w_off, size);
}

// ----- Framebuffer thin (v2 adds) -----
FN(w_framebuffer_texture_layer) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLenum attachment = a_u32(info, 1);
	const GLuint tex = obj_id(info[2]);
	const GLint level = a_i32(info, 3);
	const GLint layer = a_i32(info, 4);
	glFramebufferTextureLayer(target, attachment, tex, level, layer);
}
// The JS "default framebuffer" (name 0) is redirected to our tenant USER FBO.
// GL_COLOR/GL_DEPTH/GL_STENCIL are only valid attachment tokens on the true
// default framebuffer; on a user FBO they are GL_INVALID_ENUM. Translate them
// to the user-FBO forms when the tenant is bound. GLES3-only ops (invalidate*)
// → v2 context only, which is part of why WebGL1 rendered and WebGL2 didn't.
// Same class as w_draw_buffers Ledger #109.
static GLenum xlate_default_fb_attachment(GLenum a) {
	switch (a) {
	case GL_COLOR:   return GL_COLOR_ATTACHMENT0;  // 0x1800 -> 0x8CE0
	case GL_DEPTH:   return GL_DEPTH_ATTACHMENT;   // 0x1801 -> 0x8D00
	case GL_STENCIL: return GL_STENCIL_ATTACHMENT; // 0x1802 -> 0x8D20
	default:         return a;
	}
}
static const GLenum *xlate_default_fb_attachments(const int32_t *p, size_t n,
                                                  GLenum scratch[16]) {
	if (!(st && st->bound_fbo_js == 0) || n == 0 || n > 16)
		return (const GLenum *)p;
	for (size_t i = 0; i < n; ++i)
		scratch[i] = xlate_default_fb_attachment((GLenum)p[i]);
	return scratch;
}
FN(w_invalidate_framebuffer) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	std::vector<int32_t> tmp;
	const int32_t *p = nullptr;
	size_t n = 0;
	if (!i32_list(info.GetIsolate(), info[1], tmp, &p, &n)) return;
	GLenum scratch[16];
	const GLenum *out = xlate_default_fb_attachments(p, n, scratch);
	glInvalidateFramebuffer(target, (GLsizei)n, out);
}
FN(w_invalidate_sub_framebuffer) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	std::vector<int32_t> tmp;
	const int32_t *p = nullptr;
	size_t n = 0;
	if (!i32_list(info.GetIsolate(), info[1], tmp, &p, &n)) return;
	const GLint x = a_i32(info, 2);
	const GLint y = a_i32(info, 3);
	const GLsizei w = a_i32(info, 4);
	const GLsizei h = a_i32(info, 5);
	GLenum scratch[16];
	const GLenum *out = xlate_default_fb_attachments(p, n, scratch);
	glInvalidateSubFramebuffer(target, (GLsizei)n, out, x, y, w, h);
}
FN(w_read_buffer) {
	enter_bracket();
	GLenum src = a_u32(info, 0);
	// Tenant redirect: glReadBuffer(GL_BACK) is INVALID_OPERATION on a user
	// FBO; map BACK → COLOR_ATTACHMENT0 (the tenant's back buffer). Same class
	// as w_draw_buffers Ledger #109. GLES3-only → v2 context only.
	if (st && st->bound_fbo_js == 0 && src == GL_BACK)
		src = GL_COLOR_ATTACHMENT0;
	glReadBuffer(src);
}
FN(w_renderbuffer_storage_multisample) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLsizei samples = a_i32(info, 1);
	const GLenum internalformat = a_u32(info, 2);
	const GLsizei w = a_i32(info, 3);
	const GLsizei h = a_i32(info, 4);
	glRenderbufferStorageMultisample(target, samples, internalformat, w, h);
}
FN(w_get_frag_data_location) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint prog = obj_id(info[0]);
	char *name = take_string(iso, info[1]);
	GLint loc = glGetFragDataLocation(prog, name);
	delete[] name;
	info.GetReturnValue().Set(Int32::New(iso, loc));
}

// ----- 3D texture family (v2 adds — 3D-copy + compressed 3D) -----
FN(w_copy_tex_sub_image_3d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLint xoff = a_i32(info, 2);
	const GLint yoff = a_i32(info, 3);
	const GLint zoff = a_i32(info, 4);
	const GLint x = a_i32(info, 5);
	const GLint y = a_i32(info, 6);
	const GLsizei w = a_i32(info, 7);
	const GLsizei h = a_i32(info, 8);
	glCopyTexSubImage3D(target, level, xoff, yoff, zoff, x, y, w, h);
}
FN(w_compressed_tex_image_3d) {
	// compressedTexImage3D(target, level, internalformat, w, h, d, border, data)
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLenum internalformat = a_u32(info, 2);
	const GLsizei w = a_i32(info, 3);
	const GLsizei h = a_i32(info, 4);
	const GLsizei d = a_i32(info, 5);
	const GLint border = a_i32(info, 6);
	size_t len = 0;
	void *pixels = view_bytes(info[7], &len);
	if (info[7]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[7].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	if (info[7]->IsNullOrUndefined()) { pixels = nullptr; len = 0; }
	glCompressedTexImage3D(target, level, internalformat, w, h, d, border,
	                        (GLsizei)len, pixels);
}
FN(w_compressed_tex_sub_image_3d) {
	// compressedTexSubImage3D(target, level, xo, yo, zo, w, h, d, format, data)
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLint xo = a_i32(info, 2);
	const GLint yo = a_i32(info, 3);
	const GLint zo = a_i32(info, 4);
	const GLsizei w = a_i32(info, 5);
	const GLsizei h = a_i32(info, 6);
	const GLsizei d = a_i32(info, 7);
	const GLenum format = a_u32(info, 8);
	size_t len = 0;
	void *pixels = view_bytes(info[9], &len);
	if (info[9]->IsArrayBuffer()) {
		Local<ArrayBuffer> ab = info[9].As<ArrayBuffer>();
		pixels = ab->Data();
		len = ab->ByteLength();
	}
	if (info[9]->IsNullOrUndefined()) { pixels = nullptr; len = 0; }
	glCompressedTexSubImage3D(target, level, xo, yo, zo, w, h, d, format,
	                           (GLsizei)len, pixels);
}

// ----- UInt uniforms (v2 adds) -----
FN(w_uniform_1ui) { enter_bracket(); glUniform1ui(uniform_loc(info[0]), a_u32(info, 1)); }
FN(w_uniform_2ui) { enter_bracket(); glUniform2ui(uniform_loc(info[0]), a_u32(info, 1), a_u32(info, 2)); }
FN(w_uniform_3ui) { enter_bracket(); glUniform3ui(uniform_loc(info[0]), a_u32(info, 1), a_u32(info, 2), a_u32(info, 3)); }
FN(w_uniform_4ui) { enter_bracket(); glUniform4ui(uniform_loc(info[0]), a_u32(info, 1), a_u32(info, 2), a_u32(info, 3), a_u32(info, 4)); }

#define UNI_UIV(N) \
FN(w_uniform_##N##uiv) { \
	enter_bracket(); \
	std::vector<uint32_t> tmp; \
	const uint32_t *p = nullptr; \
	size_t n = 0; \
	if (!u32_list(info.GetIsolate(), info[1], tmp, &p, &n)) return; \
	glUniform##N##uiv(uniform_loc(info[0]), (GLsizei)(n / N), p); \
}
UNI_UIV(1)
UNI_UIV(2)
UNI_UIV(3)
UNI_UIV(4)
#undef UNI_UIV

// ----- Non-square matrix uniforms (v2 adds) -----
// glUniformMatrix{R}x{C}fv — R rows, C columns, submitted column-major.
// Element count per matrix = R * C = 6 (2x3, 3x2) / 8 (2x4, 4x2) / 12 (3x4, 4x3).
#define UNI_MAT_RxC(R, C) \
FN(w_uniform_matrix_##R##x##C##fv) { \
	enter_bracket(); \
	std::vector<float> tmp; \
	const float *p = nullptr; \
	size_t n = 0; \
	if (!f32_list(info.GetIsolate(), info[2], tmp, &p, &n)) return; \
	const GLsizei count = (GLsizei)(n / (R * C)); \
	glUniformMatrix##R##x##C##fv(uniform_loc(info[0]), count, \
	                              a_bool(info, 1) ? GL_TRUE : GL_FALSE, p); \
}
UNI_MAT_RxC(2, 3)
UNI_MAT_RxC(3, 2)
UNI_MAT_RxC(2, 4)
UNI_MAT_RxC(4, 2)
UNI_MAT_RxC(3, 4)
UNI_MAT_RxC(4, 3)
#undef UNI_MAT_RxC

// ----- Clear buffer family (v2 adds) -----
// clearBufferiv(buffer, drawbuffer, values) — 4 int values (RGBA/depth/stencil
// per buffer target). Same shape for uiv / fv. The `fi` variant takes 2
// scalars (depth + stencil) rather than an array.
FN(w_clear_buffer_iv) {
	enter_bracket();
	const GLenum buffer = a_u32(info, 0);
	const GLint drawbuffer = a_i32(info, 1);
	std::vector<int32_t> tmp;
	const int32_t *p = nullptr;
	size_t n = 0;
	if (!i32_list(info.GetIsolate(), info[2], tmp, &p, &n)) return;
	glClearBufferiv(buffer, drawbuffer, p);
	(void)n;
}
FN(w_clear_buffer_uiv) {
	enter_bracket();
	const GLenum buffer = a_u32(info, 0);
	const GLint drawbuffer = a_i32(info, 1);
	std::vector<uint32_t> tmp;
	const uint32_t *p = nullptr;
	size_t n = 0;
	if (!u32_list(info.GetIsolate(), info[2], tmp, &p, &n)) return;
	glClearBufferuiv(buffer, drawbuffer, p);
	(void)n;
}
FN(w_clear_buffer_fv) {
	enter_bracket();
	const GLenum buffer = a_u32(info, 0);
	const GLint drawbuffer = a_i32(info, 1);
	std::vector<float> tmp;
	const float *p = nullptr;
	size_t n = 0;
	if (!f32_list(info.GetIsolate(), info[2], tmp, &p, &n)) return;
	glClearBufferfv(buffer, drawbuffer, p);
	(void)n;
}
FN(w_clear_buffer_fi) {
	enter_bracket();
	const GLenum buffer = a_u32(info, 0);
	const GLint drawbuffer = a_i32(info, 1);
	const GLfloat depth = a_f32(info, 2);
	const GLint stencil = a_i32(info, 3);
	glClearBufferfi(buffer, drawbuffer, depth, stencil);
}

// ----- Draw range (v2 adds) -----
//
// #52a fallback gate: define NX_52A_DISABLE_FALLBACK at build time to
// call glDrawRangeElements directly (needed for the HW_SESSION_RUNBOOK
// hardware probe that records whether real Tegra Nouveau also silent-
// no-ops or if the behavior is Citron-only). Default (undefined) keeps
// the spec-legal glDrawElements substitution shipping.
#ifndef NX_52A_DISABLE_FALLBACK
#define NX_52A_DISABLE_FALLBACK 0
#endif
FN(w_draw_range_elements) {
	enter_bracket();
	// NXJS_PATCHES_NEEDED #52a — Citron-observed silent no-op:
	// gl-probes v0.3.0 discriminator (sentinel clear color + isolation-
	// mode + per-step glGetError) showed the failure reproduces on a
	// fresh WebGL2 context with no prior probes: baseline glDrawElements
	// PASSES on the same VAO/FBO/shader, then clearColor(sentinel)+
	// clear+glDrawRangeElements yields FBO texture untouched (readback
	// [0,0,0,0]) with all glGetError = 0. Whether this is a Mesa
	// Nouveau driver behavior or a Citron GPU translation issue is
	// HARDWARE-PENDING per the standing "Citron is functional iteration,
	// not driver truth" rule — the hardware probe recipe is in
	// docs/HW_SESSION_RUNBOOK.md §#52a.
	// ES3 §2.8.3 makes the (start, end) range a driver optimization
	// hint only — glDrawElements(mode, count, type, indices) is the
	// spec-legal equivalent. Fallback stays shipped regardless of
	// hardware outcome (belt-and-suspenders); hardware verdict decides
	// whether it becomes defensive-only or remains load-bearing.
	const GLenum mode = a_u32(info, 0);
	const GLuint start = a_u32(info, 1); // used only if fallback disabled
	const GLuint end   = a_u32(info, 2);
	const GLsizei count = a_i32(info, 3);
	const GLenum type = a_u32(info, 4);
	const GLintptr offset = (GLintptr)a_i32(info, 5);
	static bool s_boot_logged = false;
	if (!s_boot_logged) {
#if NX_52A_DISABLE_FALLBACK
		fprintf(stderr, "[#52a] drawRangeElements DIRECT (fallback DISABLED "
		                "via NX_52A_DISABLE_FALLBACK build) — hardware probe mode\n");
#else
		fprintf(stderr, "[#52a] drawRangeElements -> drawElements fallback "
		                "(range hint dropped; see NXJS_PATCHES_NEEDED #52a)\n");
#endif
		fflush(stderr);
		s_boot_logged = true;
	}
#if NX_52A_DISABLE_FALLBACK
	glDrawRangeElements(mode, start, end, count, type, (const void *)offset);
#else
	(void)start; (void)end;
	glDrawElements(mode, count, type, (const void *)offset);
#endif
	touch_fbo();
}

// ============================================================================
// End phase-1.5-LOW block.
// ============================================================================

// ============================================================================
// Phase-1.5-LOW-MED — 6 methods (tier LOW-MED per plan §0.1). All v2-only;
// installed on install_methods_v2 FUNCS[] only.
//
// Integer vertex attribs (5) — mirror of the float `vertexAttrib*f` family
// but writing to an integer attribute slot (accessed as `in ivec4` /
// `in uvec4` / `in int` / `in uint` in ESSL 3.00). `vertexAttribIPointer`
// is the buffer-backed counterpart of `vertexAttribPointer` for integer
// attributes — it MUST NOT normalize, so there's no `normalized` bool
// parameter (unlike the float pointer version).
//
// getInternalformatParameter(target, internalformat, pname) — WebGL2's
// exposure of `glGetInternalformativ`. Currently the only pnames the spec
// defines are SAMPLES (returns Int32Array of driver-supported MSAA counts
// for the format, high→low) and NUM_SAMPLE_COUNTS (single int). We look
// up NUM_SAMPLE_COUNTS first to size the SAMPLES buffer since the ES3
// spec doesn't publish a fixed upper bound.
// ============================================================================

// ----- Integer vertex attribs (v2 adds) -----
FN(w_vertex_attrib_i4i) {
	enter_bracket();
	glVertexAttribI4i(a_u32(info, 0), a_i32(info, 1), a_i32(info, 2),
	                  a_i32(info, 3), a_i32(info, 4));
}
FN(w_vertex_attrib_i4ui) {
	enter_bracket();
	glVertexAttribI4ui(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2),
	                   a_u32(info, 3), a_u32(info, 4));
}
FN(w_vertex_attrib_i4iv) {
	enter_bracket();
	std::vector<int32_t> tmp;
	const int32_t *p = nullptr;
	size_t n = 0;
	if (!i32_list(info.GetIsolate(), info[1], tmp, &p, &n)) return;
	// glVertexAttribI4iv reads exactly 4 ints. Under-length input is a
	// caller bug (WebGL2 spec: TypeError). Guard against under-read but
	// otherwise forward the first 4; over-length input is trimmed to 4.
	if (n < 4) return;
	glVertexAttribI4iv(a_u32(info, 0), p);
}
FN(w_vertex_attrib_i4uiv) {
	enter_bracket();
	std::vector<uint32_t> tmp;
	const uint32_t *p = nullptr;
	size_t n = 0;
	if (!u32_list(info.GetIsolate(), info[1], tmp, &p, &n)) return;
	if (n < 4) return;
	glVertexAttribI4uiv(a_u32(info, 0), p);
}
FN(w_vertex_attrib_i_pointer) {
	// vertexAttribIPointer(index, size, type, stride, offset). Note the
	// missing `normalized` parameter vs. vertexAttribPointer — integer
	// attributes cannot normalize by definition (they carry ivec/uvec
	// through the pipeline; no float conversion).
	enter_bracket();
	glVertexAttribIPointer(a_u32(info, 0), a_i32(info, 1), a_u32(info, 2),
	                       a_i32(info, 3),
	                       (const void *)(intptr_t)a_i64(info, 4));
}

// ----- getInternalformatParameter (v2 adds) -----
FN(w_get_internalformat_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum target = a_u32(info, 0);
	const GLenum internalformat = a_u32(info, 1);
	const GLenum pname = a_u32(info, 2);

	// Fast-path for NUM_SAMPLE_COUNTS: single-value return, still shaped
	// as an Int32Array of length 1 to match the WebGL2 spec's
	// "always-returns-a-typed-array" contract.
	if (pname == 0x9380 /* GL_NUM_SAMPLE_COUNTS */) {
		GLint n = 0;
		glGetInternalformativ(target, internalformat, pname, 1, &n);
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, 4);
		int32_t *out = (int32_t *)ab->Data();
		out[0] = n;
		info.GetReturnValue().Set(Int32Array::New(ab, 0, 1));
		return;
	}

	// SAMPLES (or any array-shaped pname): probe NUM_SAMPLE_COUNTS to
	// know the return size, then materialize the actual values. Cap at
	// 32 to bound the transient allocation on drivers that pretend to
	// support absurd sample counts (Mesa Nouveau reports ≤ 8 in practice
	// — 32 is comfortable headroom).
	GLint count = 0;
	glGetInternalformativ(target, internalformat, 0x9380 /*NUM_SAMPLE_COUNTS*/,
	                       1, &count);
	if (count < 0) count = 0;
	if (count > 32) count = 32;

	Local<ArrayBuffer> ab = ArrayBuffer::New(iso, count * 4);
	int32_t *out = (int32_t *)ab->Data();
	if (count > 0) {
		glGetInternalformativ(target, internalformat, pname, count, out);
	}
	info.GetReturnValue().Set(Int32Array::New(ab, 0, count));
}

// ============================================================================
// End phase-1.5-LOW-MED block.
// ============================================================================

// ============================================================================
// #52b — WebGL1 core getTexParameter (both v1 + v2 FUNCS[]).
// Discovered by gl-probes v0.1.0 EXT_ANISO. TEXTURE_MAX_ANISOTROPY_EXT
// (0x84FE), TEXTURE_MAX_LOD (0x813B), TEXTURE_MIN_LOD (0x813A) are float-
// valued; everything else int-valued.
// ============================================================================
FN(w_get_tex_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum target = a_u32(info, 0);
	const GLenum pname = a_u32(info, 1);
	if (pname == 0x84FE || pname == GL_TEXTURE_MAX_LOD ||
	    pname == GL_TEXTURE_MIN_LOD) {
		GLfloat f = 0.0f;
		glGetTexParameterfv(target, pname, &f);
		info.GetReturnValue().Set(Number::New(iso, f));
	} else {
		GLint v = 0;
		glGetTexParameteriv(target, pname, &v);
		info.GetReturnValue().Set(Int32::New(iso, v));
	}
}

// ============================================================================
// Phase-1.5-MED — 25 core WebGL2 methods (ledger #53).
// Four families: sampler (7), sync (6), query (7), UBO introspection (5).
// 3 new K_* handle kinds: K_QUERY, K_SAMPLER, K_SYNC (declared in the
// ObjKind enum + registered in nx_webgl2_init_class MAP[]). All v2-only;
// installed on install_methods_v2 FUNCS[] only. Runtime teardown tracking
// for the new handle kinds already lands via brewser-runtime-v8's
// gl-teardown.ts (samplers/queries/syncs/transformFeedbacks pre-wired).
// Counter jump 53 → 78/88.
// ============================================================================

// ----- Sampler family (7) -----
FN(w_create_sampler) {
	GLuint s = 0;
	glGenSamplers(1, &s);
	info.GetReturnValue().Set(new_gl_obj(info.GetIsolate(), K_SAMPLER, s));
}
FN(w_delete_sampler) {
	GLuint id = obj_id(info[0]);
	if (id) {
		glDeleteSamplers(1, &id);
		erase_wrapper_cache(K_SAMPLER, id);
	}
}
FN(w_is_sampler) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsSampler(obj_id(info[0])) == GL_TRUE));
}
FN(w_bind_sampler) {
	enter_bracket();
	glBindSampler(a_u32(info, 0), obj_id(info[1]));
}
FN(w_sampler_parameter_i) {
	enter_bracket();
	glSamplerParameteri(obj_id(info[0]), a_u32(info, 1), a_i32(info, 2));
}
FN(w_sampler_parameter_f) {
	enter_bracket();
	glSamplerParameterf(obj_id(info[0]), a_u32(info, 1), a_f32(info, 2));
}
FN(w_get_sampler_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint s = obj_id(info[0]);
	const GLenum pname = a_u32(info, 1);
	if (pname == GL_TEXTURE_MAX_LOD || pname == GL_TEXTURE_MIN_LOD) {
		GLfloat f = 0.0f;
		glGetSamplerParameterfv(s, pname, &f);
		info.GetReturnValue().Set(Number::New(iso, f));
	} else {
		GLint v = 0;
		glGetSamplerParameteriv(s, pname, &v);
		info.GetReturnValue().Set(Int32::New(iso, v));
	}
}

// ----- Sync family (6) -----
FN(w_fence_sync) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum condition = a_u32(info, 0);
	const GLbitfield flags = a_u32(info, 1);
	GLsync s = glFenceSync(condition, flags);
	if (!s) { info.GetReturnValue().SetNull(); return; }
	Local<Object> obj = new_gl_obj(iso, K_SYNC, 0);
	GLObj *o = nx::Unwrap<GLObj>(obj);
	if (o) o->sync = s;
	info.GetReturnValue().Set(obj);
}
FN(w_is_sync) {
	GLObj *o = get_gl_obj(info[0]);
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    (o && o->kind == K_SYNC && o->sync) ? (glIsSync(o->sync) == GL_TRUE) : false));
}
FN(w_delete_sync) {
	GLObj *o = get_gl_obj(info[0]);
	if (o && o->kind == K_SYNC && o->sync) {
		glDeleteSync(o->sync);
		o->sync = nullptr;
	}
}
FN(w_client_wait_sync) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	GLObj *o = get_gl_obj(info[0]);
	if (!o || o->kind != K_SYNC || !o->sync) {
		info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, GL_WAIT_FAILED));
		return;
	}
	const GLbitfield flags = a_u32(info, 1);
	// timeout is a GLuint64. JS Number can express it directly (up to
	// 2^53 — WebGL2 typically uses small values like 0 or a few ms in ns).
	const GLuint64 timeout = (GLuint64)a_f64(info, 2);
	GLenum r = glClientWaitSync(o->sync, flags, timeout);
	info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, r));
}
FN(w_wait_sync) {
	enter_bracket();
	GLObj *o = get_gl_obj(info[0]);
	if (!o || o->kind != K_SYNC || !o->sync) return;
	const GLbitfield flags = a_u32(info, 1);
	// waitSync's timeout is required to be GL_TIMEOUT_IGNORED per WebGL2
	// spec, but we forward whatever the caller passed for parity with
	// desktop drivers. GL_TIMEOUT_IGNORED = 0xFFFFFFFFFFFFFFFF.
	const GLuint64 timeout = (GLuint64)a_f64(info, 2);
	glWaitSync(o->sync, flags, timeout);
}
FN(w_get_sync_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	GLObj *o = get_gl_obj(info[0]);
	if (!o || o->kind != K_SYNC || !o->sync) { info.GetReturnValue().SetNull(); return; }
	const GLenum pname = a_u32(info, 1);
	GLint v = 0;
	GLsizei written = 0;
	glGetSynciv(o->sync, pname, 1, &written, &v);
	info.GetReturnValue().Set(Int32::New(iso, v));
}

// ----- Query family (7) -----
// Note: batch-3's EXT_disjoint_timer_query builds ON this — the query
// ext object exposes suffixed methods that alias these core natives +
// adds queryCounterEXT + timer-EXT constants. This is the batch-3 dedup
// base per plan §0.1.1.
FN(w_create_query) {
	GLuint q = 0;
	glGenQueries(1, &q);
	info.GetReturnValue().Set(new_gl_obj(info.GetIsolate(), K_QUERY, q));
}
FN(w_delete_query) {
	GLuint id = obj_id(info[0]);
	if (id) {
		glDeleteQueries(1, &id);
		erase_wrapper_cache(K_QUERY, id);
	}
}
FN(w_is_query) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsQuery(obj_id(info[0])) == GL_TRUE));
}
FN(w_begin_query) {
	enter_bracket();
	glBeginQuery(a_u32(info, 0), obj_id(info[1]));
}
FN(w_end_query) {
	enter_bracket();
	glEndQuery(a_u32(info, 0));
}
FN(w_get_query) {
	// getQuery(target, pname) — pname is CURRENT_QUERY. Returns the
	// currently-active query for the target, or null if none active.
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum target = a_u32(info, 0);
	const GLenum pname = a_u32(info, 1);
	GLint v = 0;
	glGetQueryiv(target, pname, &v);
	if (v == 0) { info.GetReturnValue().SetNull(); return; }
	info.GetReturnValue().Set(new_gl_obj(iso, K_QUERY, (GLuint)v));
}
FN(w_get_query_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint q = obj_id(info[0]);
	const GLenum pname = a_u32(info, 1);
	// QUERY_RESULT_AVAILABLE returns bool; QUERY_RESULT returns int (spec:
	// GLuint64 but WebGL2 exposes as regular number).
	GLuint v = 0;
	glGetQueryObjectuiv(q, pname, &v);
	if (pname == GL_QUERY_RESULT_AVAILABLE) {
		info.GetReturnValue().Set(Boolean::New(iso, v != 0));
	} else {
		info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, v));
	}
}

// ----- UBO introspection (5) -----
FN(w_get_indexed_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum target = a_u32(info, 0);
	const GLuint index = a_u32(info, 1);
	// TRANSFORM_FEEDBACK_BUFFER_BINDING / UNIFORM_BUFFER_BINDING return
	// a buffer object; the SIZE / START pnames return int64 but WebGL2
	// exposes as regular number.
	if (target == GL_UNIFORM_BUFFER_BINDING ||
	    target == GL_TRANSFORM_FEEDBACK_BUFFER_BINDING) {
		GLint v = 0;
		glGetIntegeri_v(target, index, &v);
		if (v == 0) { info.GetReturnValue().SetNull(); return; }
		info.GetReturnValue().Set(new_gl_obj(iso, K_BUFFER, (GLuint)v));
		return;
	}
	// SIZE / START / etc. return integer values.
	GLint64 v = 0;
	glGetInteger64i_v(target, index, &v);
	info.GetReturnValue().Set(Number::New(iso, (double)v));
}
FN(w_get_uniform_indices) {
	// getUniformIndices(program, names[]) — returns Uint32Array of
	// indices, one per input name.
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint program = obj_id(info[0]);
	if (info.Length() < 2 || !info[1]->IsArray()) return;
	Local<Array> names = info[1].As<Array>();
	const uint32_t n = names->Length();
	if (n == 0) {
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, 0);
		info.GetReturnValue().Set(Uint32Array::New(ab, 0, 0));
		return;
	}
	Local<Context> ctx = cur(iso);
	std::vector<char *> owned(n, nullptr);
	std::vector<const char *> ptrs(n, nullptr);
	for (uint32_t i = 0; i < n; i++) {
		Local<Value> nm;
		if (!names->Get(ctx, i).ToLocal(&nm)) continue;
		owned[i] = take_string(iso, nm);
		ptrs[i] = owned[i] ? owned[i] : "";
	}
	Local<ArrayBuffer> ab = ArrayBuffer::New(iso, n * 4);
	glGetUniformIndices(program, (GLsizei)n, ptrs.data(),
	                     (GLuint *)ab->Data());
	for (uint32_t i = 0; i < n; i++) if (owned[i]) delete[] owned[i];
	info.GetReturnValue().Set(Uint32Array::New(ab, 0, n));
}
FN(w_get_active_uniforms) {
	// getActiveUniforms(program, uniformIndices, pname) — returns array
	// of results, one per input index. Result type depends on pname:
	// TYPE / SIZE / OFFSET / etc. return Int32Array; IS_ROW_MAJOR
	// returns bool array.
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint program = obj_id(info[0]);
	std::vector<uint32_t> tmp;
	const uint32_t *indices = nullptr;
	size_t n = 0;
	if (!u32_list(iso, info[1], tmp, &indices, &n)) return;
	const GLenum pname = a_u32(info, 2);
	if (n == 0) {
		info.GetReturnValue().Set(Array::New(iso, 0));
		return;
	}
	std::vector<GLint> out(n);
	glGetActiveUniformsiv(program, (GLsizei)n, indices, pname, out.data());
	Local<Context> ctx = cur(iso);
	Local<Array> arr = Array::New(iso, (int)n);
	const bool is_bool = (pname == GL_UNIFORM_IS_ROW_MAJOR);
	for (size_t i = 0; i < n; i++) {
		Local<Value> v = is_bool
		    ? (Local<Value>)Boolean::New(iso, out[i] != 0)
		    : (Local<Value>)Int32::New(iso, out[i]);
		arr->Set(ctx, (uint32_t)i, v).Check();
	}
	info.GetReturnValue().Set(arr);
}
FN(w_get_active_uniform_block_parameter) {
	// getActiveUniformBlockParameter(program, blockIndex, pname) —
	// pname determines return shape:
	//   UNIFORM_BLOCK_BINDING / _DATA_SIZE / _ACTIVE_UNIFORMS → int
	//   UNIFORM_BLOCK_REFERENCED_BY_VERTEX/_FRAGMENT_SHADER → bool
	//   UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES → Uint32Array of length
	//     ACTIVE_UNIFORMS
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint program = obj_id(info[0]);
	const GLuint blockIndex = a_u32(info, 1);
	const GLenum pname = a_u32(info, 2);
	if (pname == GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER ||
	    pname == GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER) {
		GLint v = 0;
		glGetActiveUniformBlockiv(program, blockIndex, pname, &v);
		info.GetReturnValue().Set(Boolean::New(iso, v != 0));
		return;
	}
	if (pname == GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES) {
		GLint count = 0;
		glGetActiveUniformBlockiv(program, blockIndex,
		    GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &count);
		if (count < 0) count = 0;
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, count * 4);
		if (count > 0) {
			glGetActiveUniformBlockiv(program, blockIndex, pname,
			    (GLint *)ab->Data());
		}
		info.GetReturnValue().Set(Uint32Array::New(ab, 0, count));
		return;
	}
	// Scalar int pnames.
	GLint v = 0;
	glGetActiveUniformBlockiv(program, blockIndex, pname, &v);
	info.GetReturnValue().Set(Int32::New(iso, v));
}
FN(w_get_active_uniform_block_name) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint program = obj_id(info[0]);
	const GLuint blockIndex = a_u32(info, 1);
	GLint len = 0;
	glGetActiveUniformBlockiv(program, blockIndex,
	    GL_UNIFORM_BLOCK_NAME_LENGTH, &len);
	std::vector<char> buf(len > 0 ? len : 1);
	GLsizei written = 0;
	if (len > 0) glGetActiveUniformBlockName(program, blockIndex, len,
	                                          &written, buf.data());
	info.GetReturnValue().Set(
	    String::NewFromUtf8(iso, buf.data(), NewStringType::kNormal,
	                         written).ToLocalChecked());
}

// ============================================================================
// End phase-1.5-MED block.
// ============================================================================

// ============================================================================
// Phase-1.5-MED-HIGH — 10 transform-feedback methods (ledger #55).
// Closes the WebGL2 spec function counter: 78 → 88/88.
// One new handle kind: K_TRANSFORM_FEEDBACK. Runtime teardown side already
// tracks transformFeedbacks in gl-teardown.ts (pre-wired defensive
// addition when VAO tracking landed — same as the K_QUERY/K_SAMPLER/
// K_SYNC additions in #53). No runtime commit needed.
//
// Spec notes:
// - transformFeedbackVaryings binds do NOT take effect until the program
//   is (re-)linked. Callers MUST call linkProgram AFTER
//   transformFeedbackVaryings; otherwise the varyings-capture list is
//   silently the previous link's contents (or empty for a never-linked
//   program). Standard WebGL2 pattern.
// - getTransformFeedbackVarying returns a WebGLActiveInfo (name/size/type) —
//   reuses the existing K_ACTIVE_INFO handle kind + JS class.
// - RASTERIZER_DISCARD (0x8C89) is a passthrough enable/disable cap. The
//   existing w_enable / w_disable forward the raw cap to glEnable/glDisable
//   without a whitelist (only shadow-track the BLEND/DEPTH_TEST/CULL/
//   SCISSOR/STENCIL caps for user_snap restore); RASTERIZER_DISCARD lives
//   only for the duration of the user's begin/end bracket in typical use,
//   so lack of shadow tracking is spec-conformant. A future extension of
//   nx_gl_state_snap_t could add RASTERIZER_DISCARD for long-running
//   demos that leave discard on across frames.
// ============================================================================

FN(w_create_transform_feedback) {
	GLuint t = 0;
	glGenTransformFeedbacks(1, &t);
	info.GetReturnValue().Set(new_gl_obj(info.GetIsolate(),
	                                      K_TRANSFORM_FEEDBACK, t));
}
FN(w_delete_transform_feedback) {
	GLuint id = obj_id(info[0]);
	if (id) {
		glDeleteTransformFeedbacks(1, &id);
		erase_wrapper_cache(K_TRANSFORM_FEEDBACK, id);
	}
}
FN(w_is_transform_feedback) {
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    glIsTransformFeedback(obj_id(info[0])) == GL_TRUE));
}
FN(w_bind_transform_feedback) {
	enter_bracket();
	glBindTransformFeedback(a_u32(info, 0), obj_id(info[1]));
}
FN(w_begin_transform_feedback) {
	enter_bracket();
	glBeginTransformFeedback(a_u32(info, 0));
}
FN(w_end_transform_feedback) {
	enter_bracket();
	glEndTransformFeedback();
}
FN(w_transform_feedback_varyings) {
	// transformFeedbackVaryings(program, varyings, bufferMode). `varyings`
	// is a JS Array of strings. Effect is stored on the program object;
	// applied at the next glLinkProgram — callers MUST relink for the
	// binding to take effect.
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint program = obj_id(info[0]);
	if (info.Length() < 2 || !info[1]->IsArray()) return;
	Local<Array> names = info[1].As<Array>();
	const uint32_t n = names->Length();
	const GLenum bufferMode = a_u32(info, 2);
	if (n == 0) {
		glTransformFeedbackVaryings(program, 0, nullptr, bufferMode);
		return;
	}
	Local<Context> ctx = cur(iso);
	std::vector<char *> owned(n, nullptr);
	std::vector<const char *> ptrs(n, nullptr);
	for (uint32_t i = 0; i < n; i++) {
		Local<Value> nm;
		if (!names->Get(ctx, i).ToLocal(&nm)) continue;
		owned[i] = take_string(iso, nm);
		ptrs[i] = owned[i] ? owned[i] : "";
	}
	glTransformFeedbackVaryings(program, (GLsizei)n, ptrs.data(), bufferMode);
	for (uint32_t i = 0; i < n; i++) if (owned[i]) delete[] owned[i];
}
FN(w_get_transform_feedback_varying) {
	// getTransformFeedbackVarying(program, index) → WebGLActiveInfo|null.
	// Reuses the K_ACTIVE_INFO handle + JS class (same shape as
	// getActiveAttrib / getActiveUniform).
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint program = obj_id(info[0]);
	const GLuint index = a_u32(info, 1);
	char name[256];
	GLsizei nlen = 0;
	GLint size = 0;
	GLenum type = 0;
	glGetTransformFeedbackVarying(program, index, sizeof(name),
	                               &nlen, &size, &type, name);
	if (nlen == 0) { info.GetReturnValue().SetNull(); return; }
	Local<Object> obj = new_gl_obj(iso, K_ACTIVE_INFO, 0);
	Local<Context> c = cur(iso);
	obj->Set(c, String::NewFromUtf8(iso, "name").ToLocalChecked(),
	         String::NewFromUtf8(iso, name, NewStringType::kNormal,
	                              nlen).ToLocalChecked()).Check();
	obj->Set(c, String::NewFromUtf8(iso, "size").ToLocalChecked(),
	         Int32::New(iso, size)).Check();
	obj->Set(c, String::NewFromUtf8(iso, "type").ToLocalChecked(),
	         Uint32::NewFromUnsigned(iso, type)).Check();
	info.GetReturnValue().Set(obj);
}
FN(w_pause_transform_feedback) {
	enter_bracket();
	glPauseTransformFeedback();
}
FN(w_resume_transform_feedback) {
	enter_bracket();
	glResumeTransformFeedback();
}

// ============================================================================
// End phase-1.5-MED-HIGH block.
// ============================================================================

// ============================================================================
// Batch 3 — final extension batch (ledger #57).
// Advertising + entry-point surface for the last 10 rows in the pre-migration
// audit (plan §2 tier assignments, all Ba=3):
//   EXT_disjoint_timer_query (v1) + EXT_disjoint_timer_query_webgl2 (v2)
//     — full timer surface (queryCounterEXT + timer-EXT constants +
//     GPU_DISJOINT_EXT via getParameter). v1 additionally re-exposes the
//     query lifecycle methods with EXT suffix (v1 doesn't have core query
//     surface; v2 uses #53's core query methods).
//   EXT_polygon_offset_clamp — polygonOffsetClampEXT
//   EXT_clip_control          — clipControlEXT
//   OES_draw_buffers_indexed  — 8 indexed methods (v2)
//   KHR_parallel_shader_compile — maxShaderCompilerThreadsKHR +
//     COMPLETION_STATUS_KHR pname (accepted by existing
//     glGetProgramiv / glGetShaderiv paths).
//   WEBGL_blend_func_extended — SRC1_* constants + MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL;
//     v1 compile-probe-gated (per plan §2.5 defensive check).
//   WEBGL_multi_draw          — 4 engine-native loop shims.
//   WEBGL_clip_cull_distance (v2)   — 11 constants (advertise-only).
//   OES_sample_variables (v2)       — advertise-only feature-flag stub.
//   OES_shader_multisample_interpolation (v2) — 3 constants (advertise-only).
//
// Counter impact: 0 — batch 3 items are all extension-suffixed methods, none
// in the 88-list. WebGL2 spec function counter stays at 88/88.
// ============================================================================

// Batch-3 native entry-point resolution — the extension entry points are
// declared in gl2ext.h but not always linked; use eglGetProcAddress for
// robust availability. Cached once per boot.
typedef void (GL_APIENTRY *pfn_clip_control_t)(GLenum origin, GLenum depth);
typedef void (GL_APIENTRY *pfn_polygon_offset_clamp_t)(GLfloat, GLfloat, GLfloat);
typedef void (GL_APIENTRY *pfn_query_counter_t)(GLuint id, GLenum target);
typedef void (GL_APIENTRY *pfn_max_shader_threads_t)(GLuint count);
typedef void (GL_APIENTRY *pfn_enablei_t)(GLenum, GLuint);
typedef void (GL_APIENTRY *pfn_disablei_t)(GLenum, GLuint);
typedef void (GL_APIENTRY *pfn_blend_eq_i_t)(GLuint, GLenum);
typedef void (GL_APIENTRY *pfn_blend_eq_sep_i_t)(GLuint, GLenum, GLenum);
typedef void (GL_APIENTRY *pfn_blend_func_i_t)(GLuint, GLenum, GLenum);
typedef void (GL_APIENTRY *pfn_blend_func_sep_i_t)(GLuint, GLenum, GLenum, GLenum, GLenum);
typedef void (GL_APIENTRY *pfn_color_mask_i_t)(GLuint, GLboolean, GLboolean, GLboolean, GLboolean);
typedef GLboolean (GL_APIENTRY *pfn_is_enabled_i_t)(GLenum, GLuint);

static pfn_clip_control_t         s_pfn_clip_control = nullptr;
static pfn_polygon_offset_clamp_t s_pfn_polygon_offset_clamp = nullptr;
static pfn_query_counter_t        s_pfn_query_counter = nullptr;
static pfn_max_shader_threads_t   s_pfn_max_shader_threads = nullptr;
static pfn_enablei_t              s_pfn_enablei = nullptr;
static pfn_disablei_t             s_pfn_disablei = nullptr;
static pfn_blend_eq_i_t           s_pfn_blend_eq_i = nullptr;
static pfn_blend_eq_sep_i_t       s_pfn_blend_eq_sep_i = nullptr;
static pfn_blend_func_i_t         s_pfn_blend_func_i = nullptr;
static pfn_blend_func_sep_i_t     s_pfn_blend_func_sep_i = nullptr;
static pfn_color_mask_i_t         s_pfn_color_mask_i = nullptr;
static pfn_is_enabled_i_t         s_pfn_is_enabled_i = nullptr;
static bool s_b3_pfns_resolved = false;

static void resolve_b3_pfns() {
	if (s_b3_pfns_resolved) return;
	s_b3_pfns_resolved = true;
	s_pfn_clip_control         = (pfn_clip_control_t)eglGetProcAddress("glClipControlEXT");
	s_pfn_polygon_offset_clamp = (pfn_polygon_offset_clamp_t)eglGetProcAddress("glPolygonOffsetClampEXT");
	s_pfn_query_counter        = (pfn_query_counter_t)eglGetProcAddress("glQueryCounterEXT");
	s_pfn_max_shader_threads   = (pfn_max_shader_threads_t)eglGetProcAddress("glMaxShaderCompilerThreadsKHR");
	// Try OES first (Nouveau native token), fall back to EXT. Both entry
	// points bind to the same driver symbol on Mesa (documented aliasing).
	s_pfn_enablei              = (pfn_enablei_t)eglGetProcAddress("glEnableiOES");
	if (!s_pfn_enablei) s_pfn_enablei = (pfn_enablei_t)eglGetProcAddress("glEnableiEXT");
	s_pfn_disablei             = (pfn_disablei_t)eglGetProcAddress("glDisableiOES");
	if (!s_pfn_disablei) s_pfn_disablei = (pfn_disablei_t)eglGetProcAddress("glDisableiEXT");
	s_pfn_blend_eq_i           = (pfn_blend_eq_i_t)eglGetProcAddress("glBlendEquationiOES");
	if (!s_pfn_blend_eq_i) s_pfn_blend_eq_i = (pfn_blend_eq_i_t)eglGetProcAddress("glBlendEquationiEXT");
	s_pfn_blend_eq_sep_i       = (pfn_blend_eq_sep_i_t)eglGetProcAddress("glBlendEquationSeparateiOES");
	if (!s_pfn_blend_eq_sep_i) s_pfn_blend_eq_sep_i = (pfn_blend_eq_sep_i_t)eglGetProcAddress("glBlendEquationSeparateiEXT");
	s_pfn_blend_func_i         = (pfn_blend_func_i_t)eglGetProcAddress("glBlendFunciOES");
	if (!s_pfn_blend_func_i) s_pfn_blend_func_i = (pfn_blend_func_i_t)eglGetProcAddress("glBlendFunciEXT");
	s_pfn_blend_func_sep_i     = (pfn_blend_func_sep_i_t)eglGetProcAddress("glBlendFuncSeparateiOES");
	if (!s_pfn_blend_func_sep_i) s_pfn_blend_func_sep_i = (pfn_blend_func_sep_i_t)eglGetProcAddress("glBlendFuncSeparateiEXT");
	s_pfn_color_mask_i         = (pfn_color_mask_i_t)eglGetProcAddress("glColorMaskiOES");
	if (!s_pfn_color_mask_i) s_pfn_color_mask_i = (pfn_color_mask_i_t)eglGetProcAddress("glColorMaskiEXT");
	s_pfn_is_enabled_i         = (pfn_is_enabled_i_t)eglGetProcAddress("glIsEnablediOES");
	if (!s_pfn_is_enabled_i) s_pfn_is_enabled_i = (pfn_is_enabled_i_t)eglGetProcAddress("glIsEnablediEXT");
	fprintf(stderr, "[b3] extension entry-point resolution: clipControl=%p polygonOffsetClamp=%p "
	                "queryCounter=%p maxShaderThreads=%p enablei=%p disablei=%p "
	                "blendEqI=%p blendFuncI=%p colorMaskI=%p isEnabledI=%p\n",
	        (void*)(uintptr_t)s_pfn_clip_control,
	        (void*)(uintptr_t)s_pfn_polygon_offset_clamp,
	        (void*)(uintptr_t)s_pfn_query_counter,
	        (void*)(uintptr_t)s_pfn_max_shader_threads,
	        (void*)(uintptr_t)s_pfn_enablei,
	        (void*)(uintptr_t)s_pfn_disablei,
	        (void*)(uintptr_t)s_pfn_blend_eq_i,
	        (void*)(uintptr_t)s_pfn_blend_func_i,
	        (void*)(uintptr_t)s_pfn_color_mask_i,
	        (void*)(uintptr_t)s_pfn_is_enabled_i);
	fflush(stderr);
}

// ----- EXT_clip_control (v1+v2) -----
FN(w_clip_control_ext) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_clip_control) return;
	s_pfn_clip_control(a_u32(info, 0), a_u32(info, 1));
}

// ----- EXT_polygon_offset_clamp (v1+v2) -----
FN(w_polygon_offset_clamp_ext) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_polygon_offset_clamp) return;
	s_pfn_polygon_offset_clamp(a_f32(info, 0), a_f32(info, 1), a_f32(info, 2));
}

// ----- EXT_disjoint_timer_query (v1) + _webgl2 (v2) — queryCounter -----
FN(w_query_counter_ext) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_query_counter) return;
	s_pfn_query_counter(obj_id(info[0]), a_u32(info, 1));
}

// ----- KHR_parallel_shader_compile (v1+v2) -----
FN(w_max_shader_compiler_threads_khr) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_max_shader_threads) return;
	s_pfn_max_shader_threads(a_u32(info, 0));
}

// ----- OES_draw_buffers_indexed (v2) — 8 methods -----
FN(w_enable_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_enablei) return;
	s_pfn_enablei(a_u32(info, 0), a_u32(info, 1));
}
FN(w_disable_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_disablei) return;
	s_pfn_disablei(a_u32(info, 0), a_u32(info, 1));
}
FN(w_blend_equation_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_blend_eq_i) return;
	s_pfn_blend_eq_i(a_u32(info, 0), a_u32(info, 1));
}
FN(w_blend_equation_separate_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_blend_eq_sep_i) return;
	s_pfn_blend_eq_sep_i(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2));
}
FN(w_blend_func_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_blend_func_i) return;
	s_pfn_blend_func_i(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2));
}
FN(w_blend_func_separate_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_blend_func_sep_i) return;
	s_pfn_blend_func_sep_i(a_u32(info, 0), a_u32(info, 1), a_u32(info, 2),
	                       a_u32(info, 3), a_u32(info, 4));
}
FN(w_color_mask_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_color_mask_i) return;
	s_pfn_color_mask_i(a_u32(info, 0),
	                   a_bool(info, 1) ? GL_TRUE : GL_FALSE,
	                   a_bool(info, 2) ? GL_TRUE : GL_FALSE,
	                   a_bool(info, 3) ? GL_TRUE : GL_FALSE,
	                   a_bool(info, 4) ? GL_TRUE : GL_FALSE);
}
FN(w_is_enabled_i) {
	enter_bracket();
	resolve_b3_pfns();
	if (!s_pfn_is_enabled_i) {
		info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), false));
		return;
	}
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(),
	    s_pfn_is_enabled_i(a_u32(info, 0), a_u32(info, 1)) == GL_TRUE));
}

// ----- WEBGL_multi_draw (v1+v2) — 4 engine-native loop shims -----
// Per plan §3.3 — spec-conformant behavior via iteration + per-draw
// glDrawArrays / glDrawElements. Perf caveat: no batching benefit vs
// a native glMultiDrawArraysEXT, but the driver's multi-draw is only
// available via EXT and only for the non-instanced pair. Loop shim
// covers all four variants uniformly.
FN(w_multi_draw_arrays_webgl) {
	// multiDrawArraysWEBGL(mode, firsts, offsetF, counts, offsetC, drawcount)
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum mode = a_u32(info, 0);
	std::vector<int32_t> firstsTmp, countsTmp;
	const int32_t *firsts = nullptr, *counts = nullptr;
	size_t firstsN = 0, countsN = 0;
	if (!i32_list(iso, info[1], firstsTmp, &firsts, &firstsN)) return;
	const size_t offsetF = (size_t)a_i32(info, 2);
	if (!i32_list(iso, info[3], countsTmp, &counts, &countsN)) return;
	const size_t offsetC = (size_t)a_i32(info, 4);
	const size_t drawcount = (size_t)a_i32(info, 5);
	for (size_t i = 0; i < drawcount; i++) {
		if (offsetF + i >= firstsN || offsetC + i >= countsN) break;
		glDrawArrays(mode, firsts[offsetF + i], counts[offsetC + i]);
	}
	touch_fbo();
}
FN(w_multi_draw_elements_webgl) {
	// multiDrawElementsWEBGL(mode, counts, offsetC, type, offsets, offsetO, drawcount)
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum mode = a_u32(info, 0);
	std::vector<int32_t> countsTmp, offsetsTmp;
	const int32_t *counts = nullptr, *offsets = nullptr;
	size_t countsN = 0, offsetsN = 0;
	if (!i32_list(iso, info[1], countsTmp, &counts, &countsN)) return;
	const size_t offsetC = (size_t)a_i32(info, 2);
	const GLenum type = a_u32(info, 3);
	if (!i32_list(iso, info[4], offsetsTmp, &offsets, &offsetsN)) return;
	const size_t offsetO = (size_t)a_i32(info, 5);
	const size_t drawcount = (size_t)a_i32(info, 6);
	for (size_t i = 0; i < drawcount; i++) {
		if (offsetC + i >= countsN || offsetO + i >= offsetsN) break;
		glDrawElements(mode, counts[offsetC + i], type,
		               (const void *)(intptr_t)offsets[offsetO + i]);
	}
	touch_fbo();
}
FN(w_multi_draw_arrays_instanced_webgl) {
	// multiDrawArraysInstancedWEBGL(mode, firsts, offsetF, counts, offsetC,
	//                                instanceCounts, offsetI, drawcount)
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum mode = a_u32(info, 0);
	std::vector<int32_t> firstsTmp, countsTmp, instTmp;
	const int32_t *firsts = nullptr, *counts = nullptr, *inst = nullptr;
	size_t firstsN = 0, countsN = 0, instN = 0;
	if (!i32_list(iso, info[1], firstsTmp, &firsts, &firstsN)) return;
	const size_t offsetF = (size_t)a_i32(info, 2);
	if (!i32_list(iso, info[3], countsTmp, &counts, &countsN)) return;
	const size_t offsetC = (size_t)a_i32(info, 4);
	if (!i32_list(iso, info[5], instTmp, &inst, &instN)) return;
	const size_t offsetI = (size_t)a_i32(info, 6);
	const size_t drawcount = (size_t)a_i32(info, 7);
	for (size_t i = 0; i < drawcount; i++) {
		if (offsetF + i >= firstsN || offsetC + i >= countsN ||
		    offsetI + i >= instN) break;
		glDrawArraysInstanced(mode, firsts[offsetF + i], counts[offsetC + i],
		                       inst[offsetI + i]);
	}
	touch_fbo();
}
FN(w_multi_draw_elements_instanced_webgl) {
	// multiDrawElementsInstancedWEBGL(mode, counts, offsetC, type, offsets,
	//                                  offsetO, instanceCounts, offsetI, drawcount)
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum mode = a_u32(info, 0);
	std::vector<int32_t> countsTmp, offsetsTmp, instTmp;
	const int32_t *counts = nullptr, *offsets = nullptr, *inst = nullptr;
	size_t countsN = 0, offsetsN = 0, instN = 0;
	if (!i32_list(iso, info[1], countsTmp, &counts, &countsN)) return;
	const size_t offsetC = (size_t)a_i32(info, 2);
	const GLenum type = a_u32(info, 3);
	if (!i32_list(iso, info[4], offsetsTmp, &offsets, &offsetsN)) return;
	const size_t offsetO = (size_t)a_i32(info, 5);
	if (!i32_list(iso, info[6], instTmp, &inst, &instN)) return;
	const size_t offsetI = (size_t)a_i32(info, 7);
	const size_t drawcount = (size_t)a_i32(info, 8);
	for (size_t i = 0; i < drawcount; i++) {
		if (offsetC + i >= countsN || offsetO + i >= offsetsN ||
		    offsetI + i >= instN) break;
		glDrawElementsInstanced(mode, counts[offsetC + i], type,
		                         (const void *)(intptr_t)offsets[offsetO + i],
		                         inst[offsetI + i]);
	}
	touch_fbo();
}

// ============================================================================
// End batch 3 block.
// ============================================================================

// ============================================================================
// Tier 1 batch (ledger #58) — WebGL1 conformance ERROR-bucket engine fill.
// Cold-restart batched Citron baseline surfaced 7 methods absent from the
// v1/v2 FUNCS[] tables (see webgl1-results.json). getUniform is the largest
// win: 20 tests from the `uniforms-no-over-optimization-on-uniform-array-*`
// cluster ERROR on `gl.getUniform is not a function` before any assertion
// runs. All 7 methods are core WebGL1 spec surface — pure spec-hole fills
// with zero Brewser coupling, i.e. upstream-PR-ready.
// ============================================================================

// #58 — getUniform(program, location). Reads back the current value of a
// uniform on the given program. The 20 failing tests exercise the array-
// element form: getUniformLocation("u[i]") gives a per-element location, and
// getUniform on that location must return ONLY element i — never the whole
// array. That's spec-required and is what the "no-over-optimization" cluster
// checks (a compiler that collapses the array to a single storage location
// would return the same value for every element; the test proves it doesn't).
//
// Implementation shape: WebGL doesn't hand us the uniform's TYPE at call
// time — we're given only (program, location). GL exposes the type only via
// glGetActiveUniform(program, index). So we walk the program's active-
// uniform list, resolve each uniform's base location via glGetUniformLocation,
// and match `location` against [baseLoc, baseLoc + size). Once the type is
// known, dispatch the matching glGetUniform*v variant and box the JS return:
//
//   float / int / uint / bool scalar   → number / boolean
//   {float,int,uint}Vec{2,3,4}         → {Float32,Int32,Uint32}Array(N)
//   boolVec{2,3,4}                     → plain JS Array of booleans
//   floatMat{2,3,4}                    → Float32Array(N*N)
//   floatMat{2x3,2x4,3x2,3x4,4x2,4x3}  → Float32Array(rows*cols) (WebGL2)
//   sampler*                           → number (GLint TU index)
//
// The scan is O(active_uniforms) per call — acceptable because active-uniform
// counts are typically small (<20) and getUniform is not on a hot path
// (introspection / debug surface only). No program-state cache: driver-
// direct glGetActiveUniform queries stay authoritative per the stop-and-
// report policy in the ledger #58 spec.
//
// GL_* constants below are stable across all GLES3 headers (Khronos-assigned
// values in the WebGL / ES3 spec); using hex literals here avoids a header-
// availability dance if a future toolchain hides one behind a version gate.
FN(w_get_uniform) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	Local<Context> c = cur(iso);
	const GLuint program = obj_id(info[0]);
	const GLint location = uniform_loc(info[1]);
	if (program == 0 || location < 0) {
		info.GetReturnValue().SetNull();
		return;
	}
	// Resolve the type by scanning active uniforms until one whose location
	// range [base, base + size) contains the requested location.
	GLint active_count = 0;
	glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &active_count);
	GLint max_name_len = 0;
	glGetProgramiv(program, GL_ACTIVE_UNIFORM_MAX_LENGTH, &max_name_len);
	if (max_name_len < 1) max_name_len = 1;
	std::vector<char> name_buf((size_t)max_name_len + 1);
	GLenum uni_type = 0;
	GLint uni_size = 0;
	bool found = false;
	for (GLint i = 0; i < active_count; i++) {
		GLsizei nlen = 0;
		GLint size = 0;
		GLenum type = 0;
		glGetActiveUniform(program, (GLuint)i, max_name_len, &nlen,
		                    &size, &type, name_buf.data());
		if (nlen <= 0) continue;
		const GLint base = glGetUniformLocation(program, name_buf.data());
		if (base < 0) continue;
		if (location >= base && location < base + size) {
			uni_type = type;
			uni_size = size;
			found = true;
			break;
		}
	}
	if (!found) {
		// Location doesn't map to any active uniform — spec: INVALID_OPERATION.
		record_error(GL_INVALID_OPERATION);
		info.GetReturnValue().SetNull();
		return;
	}
	(void)uni_size;

	// Component-count table by uniform type. n_components is the number of
	// scalars glGetUniform*v writes into the destination for ONE location
	// (not the whole array — location is per-element per WebGL spec).
	auto box_float_array = [&](GLint n) {
		std::vector<GLfloat> buf((size_t)n, 0.0f);
		glGetUniformfv(program, location, buf.data());
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, (size_t)n * sizeof(GLfloat));
		memcpy(ab->Data(), buf.data(), (size_t)n * sizeof(GLfloat));
		info.GetReturnValue().Set(Float32Array::New(ab, 0, n));
	};
	auto box_int_array = [&](GLint n) {
		std::vector<GLint> buf((size_t)n, 0);
		glGetUniformiv(program, location, buf.data());
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, (size_t)n * sizeof(GLint));
		memcpy(ab->Data(), buf.data(), (size_t)n * sizeof(GLint));
		info.GetReturnValue().Set(Int32Array::New(ab, 0, n));
	};
	auto box_uint_array = [&](GLint n) {
		std::vector<GLuint> buf((size_t)n, 0u);
		glGetUniformuiv(program, location, buf.data());
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, (size_t)n * sizeof(GLuint));
		memcpy(ab->Data(), buf.data(), (size_t)n * sizeof(GLuint));
		info.GetReturnValue().Set(Uint32Array::New(ab, 0, n));
	};
	auto box_bool_scalar = [&]() {
		GLint v = 0;
		glGetUniformiv(program, location, &v);
		info.GetReturnValue().Set(Boolean::New(iso, v != 0));
	};
	auto box_bool_vec = [&](GLint n) {
		std::vector<GLint> buf((size_t)n, 0);
		glGetUniformiv(program, location, buf.data());
		Local<Array> arr = Array::New(iso, n);
		for (GLint i = 0; i < n; i++)
			arr->Set(c, (uint32_t)i, Boolean::New(iso, buf[(size_t)i] != 0)).Check();
		info.GetReturnValue().Set(arr);
	};
	auto box_float_scalar = [&]() {
		GLfloat v = 0.0f;
		glGetUniformfv(program, location, &v);
		info.GetReturnValue().Set(Number::New(iso, (double)v));
	};
	auto box_int_scalar = [&]() {
		GLint v = 0;
		glGetUniformiv(program, location, &v);
		info.GetReturnValue().Set(Int32::New(iso, v));
	};
	auto box_uint_scalar = [&]() {
		GLuint v = 0u;
		glGetUniformuiv(program, location, &v);
		info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, v));
	};

	switch (uni_type) {
	// Float scalar + vector
	case 0x1406 /* GL_FLOAT */:      box_float_scalar(); return;
	case 0x8B50 /* GL_FLOAT_VEC2 */: box_float_array(2); return;
	case 0x8B51 /* GL_FLOAT_VEC3 */: box_float_array(3); return;
	case 0x8B52 /* GL_FLOAT_VEC4 */: box_float_array(4); return;
	// Int scalar + vector
	case 0x1404 /* GL_INT */:      box_int_scalar(); return;
	case 0x8B53 /* GL_INT_VEC2 */: box_int_array(2); return;
	case 0x8B54 /* GL_INT_VEC3 */: box_int_array(3); return;
	case 0x8B55 /* GL_INT_VEC4 */: box_int_array(4); return;
	// Uint scalar + vector (WebGL2)
	case 0x1405 /* GL_UNSIGNED_INT */:      box_uint_scalar(); return;
	case 0x8DC6 /* GL_UNSIGNED_INT_VEC2 */: box_uint_array(2); return;
	case 0x8DC7 /* GL_UNSIGNED_INT_VEC3 */: box_uint_array(3); return;
	case 0x8DC8 /* GL_UNSIGNED_INT_VEC4 */: box_uint_array(4); return;
	// Bool scalar + vector — WebGL spec: scalar→boolean, vecN→Array of boolean.
	case 0x8B56 /* GL_BOOL */:      box_bool_scalar(); return;
	case 0x8B57 /* GL_BOOL_VEC2 */: box_bool_vec(2); return;
	case 0x8B58 /* GL_BOOL_VEC3 */: box_bool_vec(3); return;
	case 0x8B59 /* GL_BOOL_VEC4 */: box_bool_vec(4); return;
	// Square float matrices
	case 0x8B5A /* GL_FLOAT_MAT2 */: box_float_array(2 * 2); return;
	case 0x8B5B /* GL_FLOAT_MAT3 */: box_float_array(3 * 3); return;
	case 0x8B5C /* GL_FLOAT_MAT4 */: box_float_array(4 * 4); return;
	// Non-square float matrices (WebGL2). Naming convention: MAT<C>x<R> means
	// C columns × R rows — total scalars = C * R.
	case 0x8B65 /* GL_FLOAT_MAT2x3 */: box_float_array(2 * 3); return;
	case 0x8B66 /* GL_FLOAT_MAT2x4 */: box_float_array(2 * 4); return;
	case 0x8B67 /* GL_FLOAT_MAT3x2 */: box_float_array(3 * 2); return;
	case 0x8B68 /* GL_FLOAT_MAT3x4 */: box_float_array(3 * 4); return;
	case 0x8B69 /* GL_FLOAT_MAT4x2 */: box_float_array(4 * 2); return;
	case 0x8B6A /* GL_FLOAT_MAT4x3 */: box_float_array(4 * 3); return;
	default:
		// All sampler / opaque types — WebGL spec returns the bound texture
		// unit index as a plain number. Covers SAMPLER_2D (0x8B5E), SAMPLER_CUBE
		// (0x8B60), SAMPLER_3D (0x8B5F), SAMPLER_2D_SHADOW (0x8B62), sampler
		// array/shadow variants (0x8DC1..0x8DCA), integer/unsigned samplers
		// (0x8DC9..0x8DD7), etc. — everything reads through glGetUniformiv as
		// one GLint. Safer than enumerating each sampler variant one-by-one
		// (new sampler types could be added by a future extension).
		box_int_scalar();
		return;
	}
}

// Ledger #93 — WebGL 1 `getVertexAttribOffset(index, pname)`. Returns the
// offset (as a GLintptr) of the vertex-attribute-array pointer for the
// given attribute index. Only pname is `GL_VERTEX_ATTRIB_ARRAY_POINTER`
// (0x8645). Missing pre-#93 → conformance's `extensions-oes-vertex-array-
// object` and `attribs-gl-vertexattribpointer-offsets` FAIL with
// `TypeError: gl.getVertexAttribOffset is not a function`.
FN(w_get_vertex_attrib_offset) {
	enter_bracket();
	const GLuint index = a_u32(info, 0);
	const GLenum pname = a_u32(info, 1);
	void *ptr = nullptr;
	glGetVertexAttribPointerv(index, pname, &ptr);
	info.GetReturnValue().Set(Number::New(info.GetIsolate(),
	                                       (double)(intptr_t)ptr));
}

// #59 — copyTexImage2D(target, level, internalformat, x, y, w, h, border).
// Thin glCopyTexImage2D wrapper. Reads from the current READ_FRAMEBUFFER
// (ES3) or FRAMEBUFFER (ES2) at (x, y) w×h and allocates fresh mipmap-level
// storage in the currently-bound target texture using the given
// internalformat. Sibling of w_copy_tex_sub_image_2d (ledger 2.G.1 cut #25)
// which uses xoffset/yoffset into existing storage; this variant allocates.
// Unlocks 7 tests: misc-uninitialized-test, rendering-clear-after-
// copyTexImage2D, textures-misc-copy-tex-image-2d-formats, -crash,
// -texture-copying-and-deletion, -feedback-loops, -texture-npot.
FN(w_copy_tex_image_2d) {
	enter_bracket();
	const GLenum target = a_u32(info, 0);
	const GLint level = a_i32(info, 1);
	const GLenum internalformat = a_u32(info, 2);
	const GLint x = a_i32(info, 3);
	const GLint y = a_i32(info, 4);
	const GLsizei width = a_i32(info, 5);
	const GLsizei height = a_i32(info, 6);
	const GLint border = a_i32(info, 7);
	// Ledger #91 — WebGL 1 spec §5.14.8: copyTexImage2D at level > 0 with
	// NPOT dimensions must return INVALID_VALUE. See w_tex_image_2d
	// above for rationale.
	if (!is_v2_context(info) && level > 0 &&
	    (!is_pot(width) || !is_pot(height))) {
		record_error(GL_INVALID_VALUE);
		return;
	}
	glCopyTexImage2D(target, level, internalformat, x, y, width, height,
	                 border);
}

// #60 — getVertexAttrib(index, pname). Pname-switched read of the vertex-
// attribute-array state at `index`. GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING
// returns a WebGLBuffer wrapper (never a raw GLuint), so the caller can
// `instanceof WebGLBuffer` — that's what extensions-angle-instanced-arrays
// checks. CURRENT_VERTEX_ATTRIB returns a Float32Array(4). Everything else
// is an integer- or boolean-typed pname.
//
// Hex constants used inline where the token isn't guaranteed to be defined
// under gl3.h (INTEGER + DIVISOR are ES3-core; TYPE / STRIDE etc are ES2).
FN(w_get_vertex_attrib) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLuint index = a_u32(info, 0);
	const GLenum pname = a_u32(info, 1);
	switch (pname) {
	case 0x889F /* GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING */: {
		GLint v = 0;
		glGetVertexAttribiv(index, pname, &v);
		if (v == 0) {
			info.GetReturnValue().SetNull();
		} else {
			info.GetReturnValue().Set(
			    new_gl_obj(iso, K_BUFFER, (GLuint)v));
		}
		return;
	}
	case 0x8622 /* GL_VERTEX_ATTRIB_ARRAY_ENABLED */:
	case 0x886A /* GL_VERTEX_ATTRIB_ARRAY_NORMALIZED */:
	case 0x88FD /* GL_VERTEX_ATTRIB_ARRAY_INTEGER (WebGL2) */: {
		GLint v = 0;
		glGetVertexAttribiv(index, pname, &v);
		info.GetReturnValue().Set(Boolean::New(iso, v != 0));
		return;
	}
	case 0x8623 /* GL_VERTEX_ATTRIB_ARRAY_SIZE */:
	case 0x8624 /* GL_VERTEX_ATTRIB_ARRAY_STRIDE */:
	case 0x88FE /* GL_VERTEX_ATTRIB_ARRAY_DIVISOR (WebGL2 core) */: {
		GLint v = 0;
		glGetVertexAttribiv(index, pname, &v);
		info.GetReturnValue().Set(Int32::New(iso, v));
		return;
	}
	case 0x8625 /* GL_VERTEX_ATTRIB_ARRAY_TYPE */: {
		GLint v = 0;
		glGetVertexAttribiv(index, pname, &v);
		info.GetReturnValue().Set(Uint32::NewFromUnsigned(iso, (uint32_t)v));
		return;
	}
	case 0x8626 /* GL_CURRENT_VERTEX_ATTRIB */: {
		GLfloat v[4] = {0, 0, 0, 0};
		glGetVertexAttribfv(index, pname, v);
		Local<ArrayBuffer> ab = ArrayBuffer::New(iso, 4 * sizeof(GLfloat));
		memcpy(ab->Data(), v, 4 * sizeof(GLfloat));
		info.GetReturnValue().Set(Float32Array::New(ab, 0, 4));
		return;
	}
	default: {
		// Unknown pname — spec: INVALID_ENUM, return null.
		record_error(GL_INVALID_ENUM);
		info.GetReturnValue().SetNull();
		return;
	}
	}
}

// #61 — getFramebufferAttachmentParameter(target, attachment, pname). Pname-
// switched read of FBO attachment metadata. ATTACHMENT_OBJECT_NAME returns a
// wrapper of the attached object — either K_TEXTURE or K_RENDERBUFFER,
// depending on ATTACHMENT_OBJECT_TYPE. Everything else is integer.
FN(w_get_framebuffer_attachment_parameter) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	const GLenum target = a_u32(info, 0);
	const GLenum attachment = a_u32(info, 1);
	const GLenum pname = a_u32(info, 2);
	if (pname == 0x8CD1 /* GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME */) {
		// Preflight the OBJECT_TYPE to decide which wrapper kind to use.
		GLint obj_type = 0;
		glGetFramebufferAttachmentParameteriv(
		    target, attachment,
		    0x8CD0 /* GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE */, &obj_type);
		GLint name = 0;
		glGetFramebufferAttachmentParameteriv(target, attachment, pname,
		                                       &name);
		if (obj_type == 0 /* GL_NONE */ || name == 0) {
			// Ledger #95 — WebGL 1 spec §5.14.3: if the framebuffer
			// attachment type is NONE, querying OBJECT_NAME generates
			// INVALID_ENUM and the return value is null. WebGL 2 §3.7.3
			// widens this: OBJECT_NAME with NONE attachment returns null
			// WITHOUT generating an error. The distinction is spec-
			// mandated and load-bearing for the `misc-object-deletion-
			// behaviour` test's WebGL 1 assertion arm.
			if (!is_v2_context(info)) {
				record_error(GL_INVALID_ENUM);
			}
			info.GetReturnValue().SetNull();
			return;
		}
		if (obj_type == (GLint)GL_TEXTURE) {
			info.GetReturnValue().Set(
			    new_gl_obj(iso, K_TEXTURE, (GLuint)name));
			return;
		}
		if (obj_type == (GLint)GL_RENDERBUFFER) {
			info.GetReturnValue().Set(
			    new_gl_obj(iso, K_RENDERBUFFER, (GLuint)name));
			return;
		}
		// Unknown attached-object type — spec doesn't define this, but
		// returning null is safest (matches Chrome / Firefox).
		info.GetReturnValue().SetNull();
		return;
	}
	// All other pnames are integer-valued (TYPE / LEVEL / CUBE_MAP_FACE /
	// LAYER / RED_SIZE / GREEN_SIZE / BLUE_SIZE / ALPHA_SIZE / DEPTH_SIZE /
	// STENCIL_SIZE / COMPONENT_TYPE / COLOR_ENCODING). Return as number.
	GLint v = 0;
	glGetFramebufferAttachmentParameteriv(target, attachment, pname, &v);
	info.GetReturnValue().Set(Int32::New(iso, v));
}

// #62 — getAttachedShaders(program). Returns a JS Array of WebGLShader
// wrappers for the shaders currently attached to `program`. Empty array
// when nothing is attached (not null — spec).
FN(w_get_attached_shaders) {
	enter_bracket();
	Isolate *iso = info.GetIsolate();
	Local<Context> c = cur(iso);
	const GLuint program = obj_id(info[0]);
	if (program == 0) {
		info.GetReturnValue().SetNull();
		return;
	}
	GLint count = 0;
	glGetProgramiv(program, GL_ATTACHED_SHADERS, &count);
	if (count < 0) count = 0;
	std::vector<GLuint> names((size_t)count, 0u);
	GLsizei got = 0;
	if (count > 0) {
		glGetAttachedShaders(program, (GLsizei)count, &got, names.data());
		if (got < 0) got = 0;
		if ((GLint)got > count) got = (GLsizei)count;
	}
	Local<Array> arr = Array::New(iso, (int)got);
	for (GLsizei i = 0; i < got; i++) {
		arr->Set(c, (uint32_t)i,
		          new_gl_obj(iso, K_SHADER, names[(size_t)i])).Check();
	}
	info.GetReturnValue().Set(arr);
}

// #63 — vertexAttrib{1,2,3,4}fv(index, values). Typed-array pointer variants
// of the scalar setters at line ~2273. Same unwrap pattern as uniform_{N}fv
// (via the shared f32_list helper — handles both Float32Array and plain JS
// arrays). glVertexAttrib1fv accepts a pointer to one float, 2fv two, etc.
#define VA_FV(N) \
FN(w_vertex_attrib_##N##fv) { \
	enter_bracket(); \
	Isolate *iso = info.GetIsolate(); \
	const GLuint index = a_u32(info, 0); \
	std::vector<float> tmp; \
	const float *p = nullptr; \
	size_t n = 0; \
	if (!f32_list(iso, info[1], tmp, &p, &n)) return; \
	if (n < N) return; \
	glVertexAttrib##N##fv(index, p); \
}
VA_FV(1)
VA_FV(2)
VA_FV(3)
VA_FV(4)
#undef VA_FV

// ============================================================================
// End Tier 1 block.
// ============================================================================

static void install_methods(Isolate *iso, Local<Object> proto) {
	struct Spec { const char *name; FunctionCallback fn; };
	static const Spec FUNCS[] = {
	    {"viewport", w_viewport},
	    {"scissor", w_scissor},
	    {"enable", w_enable},
	    {"disable", w_disable},
	    {"isEnabled", w_is_enabled},
	    {"depthFunc", w_depth_func},
	    {"depthMask", w_depth_mask},
	    {"depthRange", w_depth_range},
	    {"cullFace", w_cull_face},
	    {"frontFace", w_front_face},
	    {"blendFunc", w_blend_func},
	    {"blendFuncSeparate", w_blend_func_separate},
	    {"blendEquation", w_blend_equation},
	    {"blendEquationSeparate", w_blend_equation_separate},
	    {"blendColor", w_blend_color},
	    {"colorMask", w_color_mask},
	    {"stencilFunc", w_stencil_func},
	    {"stencilFuncSeparate", w_stencil_func_separate},
	    {"stencilOp", w_stencil_op},
	    {"stencilOpSeparate", w_stencil_op_separate},
	    {"stencilMask", w_stencil_mask},
	    {"stencilMaskSeparate", w_stencil_mask_separate},
	    {"polygonOffset", w_polygon_offset},
	    {"sampleCoverage", w_sample_coverage},
	    {"lineWidth", w_line_width},
	    {"hint", w_hint},
	    {"clear", w_clear},
	    {"clearColor", w_clear_color},
	    {"clearDepth", w_clear_depth},
	    {"clearStencil", w_clear_stencil},
	    {"finish", w_finish},
	    {"flush", w_flush},
	    {"pixelStorei", w_pixel_storei},
	    {"getError", w_get_error},
	    {"getParameter", w_get_parameter},
	    {"getTexParameter", w_get_tex_parameter},
	    {"getExtension", w_get_extension},
	    {"getSupportedExtensions", w_get_supported_extensions},
	    // Phase-0 gap fill — internal natives for the runtime's
	    // getBackendInfo shim. Leading underscore signals "shim
	    // consumers only, not a WebGL spec surface". See
	    // populate_native_extensions() rationale + phase-0 ledger entry.
	    {"_getNativeExtensionsString", w_get_native_extensions_string},
	    {"_getEglVersion", w_get_egl_version},
	    {"isContextLost", w_is_context_lost},
	    {"getContextAttributes", w_get_context_attributes},
	    {"getShaderPrecisionFormat", w_get_shader_precision_format},
	    {"createShader", w_create_shader},
	    {"deleteShader", w_delete_shader},
	    {"isShader", w_is_shader},
	    {"shaderSource", w_shader_source},
	    {"compileShader", w_compile_shader},
	    {"getShaderParameter", w_get_shader_parameter},
	    {"getShaderInfoLog", w_get_shader_info_log},
	    {"getShaderSource", w_get_shader_source},
	    {"createProgram", w_create_program},
	    {"deleteProgram", w_delete_program},
	    {"isProgram", w_is_program},
	    {"attachShader", w_attach_shader},
	    {"detachShader", w_detach_shader},
	    {"linkProgram", w_link_program},
	    {"validateProgram", w_validate_program},
	    {"useProgram", w_use_program},
	    {"getProgramParameter", w_get_program_parameter},
	    {"getProgramInfoLog", w_get_program_info_log},
	    {"getAttribLocation", w_get_attrib_location},
	    {"getUniformLocation", w_get_uniform_location},
	    {"bindAttribLocation", w_bind_attrib_location},
	    {"getActiveAttrib", w_get_active_attrib},
	    {"getActiveUniform", w_get_active_uniform},
	    {"createBuffer", w_create_buffer},
	    {"deleteBuffer", w_delete_buffer},
	    {"isBuffer", w_is_buffer},
	    {"bindBuffer", w_bind_buffer},
	    {"bufferData", w_buffer_data},
	    {"bufferSubData", w_buffer_sub_data},
	    {"enableVertexAttribArray", w_enable_vertex_attrib_array},
	    {"disableVertexAttribArray", w_disable_vertex_attrib_array},
	    {"vertexAttribPointer", w_vertex_attrib_pointer},
	    {"vertexAttrib1f", w_vertex_attrib_1f},
	    {"vertexAttrib2f", w_vertex_attrib_2f},
	    {"vertexAttrib3f", w_vertex_attrib_3f},
	    {"vertexAttrib4f", w_vertex_attrib_4f},
	    {"uniform1f", w_uniform_1f},
	    {"uniform2f", w_uniform_2f},
	    {"uniform3f", w_uniform_3f},
	    {"uniform4f", w_uniform_4f},
	    {"uniform1i", w_uniform_1i},
	    {"uniform2i", w_uniform_2i},
	    {"uniform3i", w_uniform_3i},
	    {"uniform4i", w_uniform_4i},
	    {"uniform1fv", w_uniform_1fv},
	    {"uniform2fv", w_uniform_2fv},
	    {"uniform3fv", w_uniform_3fv},
	    {"uniform4fv", w_uniform_4fv},
	    {"uniform1iv", w_uniform_1iv},
	    {"uniform2iv", w_uniform_2iv},
	    {"uniform3iv", w_uniform_3iv},
	    {"uniform4iv", w_uniform_4iv},
	    {"uniformMatrix2fv", w_uniform_matrix_2fv},
	    {"uniformMatrix3fv", w_uniform_matrix_3fv},
	    {"uniformMatrix4fv", w_uniform_matrix_4fv},
	    {"createTexture", w_create_texture},
	    {"deleteTexture", w_delete_texture},
	    {"isTexture", w_is_texture},
	    {"bindTexture", w_bind_texture},
	    {"activeTexture", w_active_texture},
	    {"texParameteri", w_tex_parameteri},
	    {"texParameterf", w_tex_parameterf},
	    {"generateMipmap", w_generate_mipmap},
	    {"texImage2D", w_tex_image_2d},
	    {"texSubImage2D", w_tex_sub_image_2d},
	    // Phase-1 batch-1 — compressed texture 2D upload path.
	    // Missing FUNCS[] entries pre-batch-1 meant advertising any
	    // compressed-format extension (S3TC/RGTC/BPTC/ETC1/ASTC) would
	    // have been a fake — the constants would exist but the upload
	    // native would be a TypeError. Added in the same batch as the
	    // advertising rows for that family.
	    {"compressedTexImage2D", w_compressed_tex_image_2d},
	    {"compressedTexSubImage2D", w_compressed_tex_sub_image_2d},
	    {"copyTexSubImage2D", w_copy_tex_sub_image_2d},
	    {"createFramebuffer", w_create_framebuffer},
	    {"deleteFramebuffer", w_delete_framebuffer},
	    {"isFramebuffer", w_is_framebuffer},
	    {"bindFramebuffer", w_bind_framebuffer},
	    {"framebufferTexture2D", w_framebuffer_texture_2d},
	    {"framebufferRenderbuffer", w_framebuffer_renderbuffer},
	    {"checkFramebufferStatus", w_check_framebuffer_status},
	    {"createRenderbuffer", w_create_renderbuffer},
	    {"deleteRenderbuffer", w_delete_renderbuffer},
	    {"isRenderbuffer", w_is_renderbuffer},
	    {"bindRenderbuffer", w_bind_renderbuffer},
	    {"renderbufferStorage", w_renderbuffer_storage},
	    // Ledger #100 — getRenderbufferParameter (WebGL 1 §5.14.7).
	    {"getRenderbufferParameter", w_get_renderbuffer_parameter},
	    {"drawArrays", w_draw_arrays},
	    {"drawElements", w_draw_elements},
	    {"readPixels", w_read_pixels},
	    // Phase-1 batch-2 — Unity-P1 v1 function surfaces. Aliases the v2
	    // core natives so the WebGL1 ext objects (ANGLE_instanced_arrays,
	    // WEBGL_draw_buffers) can vend `-ANGLE` / `-WEBGL` suffixed
	    // methods that point at the SAME native handlers. Also flips
	    // the report's `extInstancedArraysPresent: false` on v1 to true
	    // (getBackendInfo shim probes `typeof gl.drawArraysInstanced ===
	    // 'function'`; before this batch the v1 table lacked those).
	    {"drawArraysInstanced", w_draw_arrays_instanced},
	    {"drawElementsInstanced", w_draw_elements_instanced},
	    {"vertexAttribDivisor", w_vertex_attrib_divisor},
	    {"drawBuffers", w_draw_buffers},
	    // ============================================================
	    // Batch 3 (ledger #57) — v1 extension entry-point wiring.
	    // v1 doesn't have the core query family (v2's #53), so
	    // EXT_disjoint_timer_query on v1 exposes the whole query
	    // lifecycle with EXT suffix, aliasing to the v2 natives.
	    // ============================================================
	    // EXT_disjoint_timer_query lifecycle (v1) — aliases #53 natives
	    {"createQueryEXT", w_create_query},
	    {"deleteQueryEXT", w_delete_query},
	    {"isQueryEXT", w_is_query},
	    {"beginQueryEXT", w_begin_query},
	    {"endQueryEXT", w_end_query},
	    {"getQueryEXT", w_get_query},
	    {"getQueryObjectEXT", w_get_query_parameter},
	    {"queryCounterEXT", w_query_counter_ext},
	    // EXT_clip_control
	    {"clipControlEXT", w_clip_control_ext},
	    // EXT_polygon_offset_clamp
	    {"polygonOffsetClampEXT", w_polygon_offset_clamp_ext},
	    // KHR_parallel_shader_compile
	    {"maxShaderCompilerThreadsKHR", w_max_shader_compiler_threads_khr},
	    // WEBGL_multi_draw — 4 loop-shim methods (v1 aliases v2)
	    {"multiDrawArraysWEBGL", w_multi_draw_arrays_webgl},
	    {"multiDrawElementsWEBGL", w_multi_draw_elements_webgl},
	    {"multiDrawArraysInstancedWEBGL", w_multi_draw_arrays_instanced_webgl},
	    {"multiDrawElementsInstancedWEBGL", w_multi_draw_elements_instanced_webgl},
	    // ============================================================
	    // Tier 1 (ledger #58–#63) — WebGL1 spec-hole fill for the
	    // conformance ERROR bucket. See the Tier 1 block comment above
	    // for scope. Same entries land on both v1 + v2 FUNCS[].
	    // ============================================================
	    {"getUniform", w_get_uniform},
	    {"copyTexImage2D", w_copy_tex_image_2d},
	    {"getVertexAttrib", w_get_vertex_attrib},
	    {"getVertexAttribOffset", w_get_vertex_attrib_offset},
	    {"getFramebufferAttachmentParameter", w_get_framebuffer_attachment_parameter},
	    {"getAttachedShaders", w_get_attached_shaders},
	    {"vertexAttrib1fv", w_vertex_attrib_1fv},
	    {"vertexAttrib2fv", w_vertex_attrib_2fv},
	    {"vertexAttrib3fv", w_vertex_attrib_3fv},
	    {"vertexAttrib4fv", w_vertex_attrib_4fv},
	    // Fork-specific hooks (canvas-runner expects them).
	    {"enableGpuBridgePrototype", w_enable_gpu_bridge_prototype},
	    {"setBridgeAutoFlush", w_set_bridge_auto_flush},
	    {"copyBridgeToCanvas", w_copy_bridge_to_canvas},
	    {"resetSharedContext", w_reset_shared_context},
	};
	Local<Context> ctx = iso->GetCurrentContext();
	for (const auto &s : FUNCS) {
		proto->Set(ctx, nx_str(iso, s.name),
		           FunctionTemplate::New(iso, s.fn)
		               ->GetFunction(ctx).ToLocalChecked()).Check();
	}
}

// Phase 2.G.1 cut #1 — v2 method table = v1 95-method base (verbatim copy
// of install_methods's FUNCS[]). All w_* impl functions operate on the
// process-wide WebGLState `st` which is identical between v1 and v2 (one
// bridge, one tenant FBO, one shared ES3 context per Phase 2.A), so binding
// the same impl functions on the v2 prototype is semantically correct —
// each call goes through the same lazy bracket enter / Skia composite
// exit machinery a v1 call would. Entries duplicate the Spec rows; the
// underlying C++ functions are SHARED (no impl duplication).
//
// 2.G.1 cut #2+ will add v2-only methods (VAO, UBO, sync, queries,
// texStorage2D, ...) at the bottom of this table as the webgl2-ubo slice's
// diag-proxy reports them. The intentional code duplication of the Spec
// rows is the table-split-shape discipline from NXJS_PATCHES_NEEDED.md #15
// — DO NOT refactor to dedup; refactoring re-introduces the JIT-safety
// risk per the rationale block above install_methods().
static void install_methods_v2(Isolate *iso, Local<Object> proto) {
	struct Spec { const char *name; FunctionCallback fn; };
	static const Spec FUNCS[] = {
	    {"viewport", w_viewport},
	    {"scissor", w_scissor},
	    {"enable", w_enable},
	    {"disable", w_disable},
	    {"isEnabled", w_is_enabled},
	    {"depthFunc", w_depth_func},
	    {"depthMask", w_depth_mask},
	    {"depthRange", w_depth_range},
	    {"cullFace", w_cull_face},
	    {"frontFace", w_front_face},
	    {"blendFunc", w_blend_func},
	    {"blendFuncSeparate", w_blend_func_separate},
	    {"blendEquation", w_blend_equation},
	    {"blendEquationSeparate", w_blend_equation_separate},
	    {"blendColor", w_blend_color},
	    {"colorMask", w_color_mask},
	    {"stencilFunc", w_stencil_func},
	    {"stencilFuncSeparate", w_stencil_func_separate},
	    {"stencilOp", w_stencil_op},
	    {"stencilOpSeparate", w_stencil_op_separate},
	    {"stencilMask", w_stencil_mask},
	    {"stencilMaskSeparate", w_stencil_mask_separate},
	    {"polygonOffset", w_polygon_offset},
	    {"sampleCoverage", w_sample_coverage},
	    {"lineWidth", w_line_width},
	    {"hint", w_hint},
	    {"clear", w_clear},
	    {"clearColor", w_clear_color},
	    {"clearDepth", w_clear_depth},
	    {"clearStencil", w_clear_stencil},
	    {"finish", w_finish},
	    {"flush", w_flush},
	    {"pixelStorei", w_pixel_storei},
	    {"getError", w_get_error},
	    {"getParameter", w_get_parameter},
	    {"getTexParameter", w_get_tex_parameter},
	    {"getExtension", w_get_extension},
	    {"getSupportedExtensions", w_get_supported_extensions},
	    // Phase-0 gap fill — internal natives for the runtime's
	    // getBackendInfo shim. Leading underscore signals "shim
	    // consumers only, not a WebGL spec surface". See
	    // populate_native_extensions() rationale + phase-0 ledger entry.
	    {"_getNativeExtensionsString", w_get_native_extensions_string},
	    {"_getEglVersion", w_get_egl_version},
	    {"isContextLost", w_is_context_lost},
	    {"getContextAttributes", w_get_context_attributes},
	    {"getShaderPrecisionFormat", w_get_shader_precision_format},
	    {"createShader", w_create_shader},
	    {"deleteShader", w_delete_shader},
	    {"isShader", w_is_shader},
	    {"shaderSource", w_shader_source},
	    {"compileShader", w_compile_shader},
	    {"getShaderParameter", w_get_shader_parameter},
	    {"getShaderInfoLog", w_get_shader_info_log},
	    {"getShaderSource", w_get_shader_source},
	    {"createProgram", w_create_program},
	    {"deleteProgram", w_delete_program},
	    {"isProgram", w_is_program},
	    {"attachShader", w_attach_shader},
	    {"detachShader", w_detach_shader},
	    {"linkProgram", w_link_program},
	    {"validateProgram", w_validate_program},
	    {"useProgram", w_use_program},
	    {"getProgramParameter", w_get_program_parameter},
	    {"getProgramInfoLog", w_get_program_info_log},
	    {"getAttribLocation", w_get_attrib_location},
	    {"getUniformLocation", w_get_uniform_location},
	    {"bindAttribLocation", w_bind_attrib_location},
	    {"getActiveAttrib", w_get_active_attrib},
	    {"getActiveUniform", w_get_active_uniform},
	    {"createBuffer", w_create_buffer},
	    {"deleteBuffer", w_delete_buffer},
	    {"isBuffer", w_is_buffer},
	    {"bindBuffer", w_bind_buffer},
	    {"bufferData", w_buffer_data},
	    {"bufferSubData", w_buffer_sub_data},
	    {"enableVertexAttribArray", w_enable_vertex_attrib_array},
	    {"disableVertexAttribArray", w_disable_vertex_attrib_array},
	    {"vertexAttribPointer", w_vertex_attrib_pointer},
	    {"vertexAttrib1f", w_vertex_attrib_1f},
	    {"vertexAttrib2f", w_vertex_attrib_2f},
	    {"vertexAttrib3f", w_vertex_attrib_3f},
	    {"vertexAttrib4f", w_vertex_attrib_4f},
	    {"uniform1f", w_uniform_1f},
	    {"uniform2f", w_uniform_2f},
	    {"uniform3f", w_uniform_3f},
	    {"uniform4f", w_uniform_4f},
	    {"uniform1i", w_uniform_1i},
	    {"uniform2i", w_uniform_2i},
	    {"uniform3i", w_uniform_3i},
	    {"uniform4i", w_uniform_4i},
	    {"uniform1fv", w_uniform_1fv},
	    {"uniform2fv", w_uniform_2fv},
	    {"uniform3fv", w_uniform_3fv},
	    {"uniform4fv", w_uniform_4fv},
	    {"uniform1iv", w_uniform_1iv},
	    {"uniform2iv", w_uniform_2iv},
	    {"uniform3iv", w_uniform_3iv},
	    {"uniform4iv", w_uniform_4iv},
	    {"uniformMatrix2fv", w_uniform_matrix_2fv},
	    {"uniformMatrix3fv", w_uniform_matrix_3fv},
	    {"uniformMatrix4fv", w_uniform_matrix_4fv},
	    {"createTexture", w_create_texture},
	    {"deleteTexture", w_delete_texture},
	    {"isTexture", w_is_texture},
	    {"bindTexture", w_bind_texture},
	    {"activeTexture", w_active_texture},
	    {"texParameteri", w_tex_parameteri},
	    {"texParameterf", w_tex_parameterf},
	    {"generateMipmap", w_generate_mipmap},
	    {"texImage2D", w_tex_image_2d},
	    {"texSubImage2D", w_tex_sub_image_2d},
	    // Phase-1 batch-1 — compressed texture 2D upload path.
	    // Missing FUNCS[] entries pre-batch-1 meant advertising any
	    // compressed-format extension (S3TC/RGTC/BPTC/ETC1/ASTC) would
	    // have been a fake — the constants would exist but the upload
	    // native would be a TypeError. Added in the same batch as the
	    // advertising rows for that family.
	    {"compressedTexImage2D", w_compressed_tex_image_2d},
	    {"compressedTexSubImage2D", w_compressed_tex_sub_image_2d},
	    {"copyTexSubImage2D", w_copy_tex_sub_image_2d},
	    {"createFramebuffer", w_create_framebuffer},
	    {"deleteFramebuffer", w_delete_framebuffer},
	    {"isFramebuffer", w_is_framebuffer},
	    {"bindFramebuffer", w_bind_framebuffer},
	    {"framebufferTexture2D", w_framebuffer_texture_2d},
	    {"framebufferRenderbuffer", w_framebuffer_renderbuffer},
	    {"checkFramebufferStatus", w_check_framebuffer_status},
	    {"createRenderbuffer", w_create_renderbuffer},
	    {"deleteRenderbuffer", w_delete_renderbuffer},
	    {"isRenderbuffer", w_is_renderbuffer},
	    {"bindRenderbuffer", w_bind_renderbuffer},
	    {"renderbufferStorage", w_renderbuffer_storage},
	    // Ledger #100 — getRenderbufferParameter (WebGL 1 §5.14.7).
	    {"getRenderbufferParameter", w_get_renderbuffer_parameter},
	    {"drawArrays", w_draw_arrays},
	    {"drawElements", w_draw_elements},
	    {"readPixels", w_read_pixels},
	    // Fork-specific hooks (canvas-runner expects them).
	    {"enableGpuBridgePrototype", w_enable_gpu_bridge_prototype},
	    {"setBridgeAutoFlush", w_set_bridge_auto_flush},
	    {"copyBridgeToCanvas", w_copy_bridge_to_canvas},
	    {"resetSharedContext", w_reset_shared_context},

	    // ---- v2-only adds (Phase 2.G.1 cut #2+) ----
	    // cut #2 (2026-06-30) — texImage3D: Three.js v2 WebGLRenderer init
	    // creates 1×1 placeholder textures for default-bound TEXTURE_3D /
	    // TEXTURE_2D_ARRAY samplers (v2 analog of v1's _emptyCubeTexture).
	    {"texImage3D", w_tex_image_3d},
	    // cut #32 (2026-07-01) — Three.js r184 WebGL2 uses these for every
	    // DataArrayTexture/Data3DTexture upload; without them, sampler2DArray
	    // reads return vec4(0) → webgl2-texture2darray renders black.
	    {"texStorage3D", w_tex_storage_3d},
	    {"texSubImage3D", w_tex_sub_image_3d},

	    // cut #3 (2026-06-30) — VAO + UBO core + texStorage2D. Three.js v2
	    // uses VAOs unconditionally for every Mesh, and the webgl2-ubo demo
	    // exercises the UBO surface directly. texStorage2D is Three.js v2's
	    // preferred immutable-allocation path for textures.
	    {"createVertexArray", w_create_vertex_array},
	    {"deleteVertexArray", w_delete_vertex_array},
	    {"isVertexArray", w_is_vertex_array},
	    {"bindVertexArray", w_bind_vertex_array},
	    {"bindBufferBase", w_bind_buffer_base},
	    {"bindBufferRange", w_bind_buffer_range},
	    {"getUniformBlockIndex", w_get_uniform_block_index},
	    {"uniformBlockBinding", w_uniform_block_binding},
	    {"texStorage2D", w_tex_storage_2d},

	    // cut #27 (2026-07-01) — blitFramebuffer for cube-route-shim's
	    // Y-flipped scratch → atlas copy in the cube-RT-readback rescue.
	    // WebGL2-only (glBlitFramebuffer is core GLES3, no ES2 equivalent).
	    {"blitFramebuffer", w_blit_framebuffer},

	    // cut #4 (2026-06-30) — instanced-drawing trio. Three.js's
	    // InstancedMesh path uses drawElementsInstanced + vertexAttribDivisor
	    // for per-instance attribute stepping; drawArraysInstanced is the
	    // unindexed sibling.
	    {"drawArraysInstanced", w_draw_arrays_instanced},
	    {"drawElementsInstanced", w_draw_elements_instanced},
	    {"vertexAttribDivisor", w_vertex_attrib_divisor},

	    // cut #5 (2026-06-30) — drawBuffers. Three.js v2 WebGLState calls
	    // gl.drawBuffers([gl.BACK]) on first-render bindFramebuffer transition
	    // (un-try/catch'd in the upstream state.drawBuffers function);
	    // silently aborts the render path when undefined. instancing-dynamic
	    // surfaced this as "render fps stays 49 + clear color shows + nothing
	    // else draws + demo error stays null" — gl.clear runs FIRST in the
	    // frame so the clear color reaches the FBO; the throw from
	    // state.drawBuffers after that loses everything downstream, but
	    // Three.js's logging path (`error(...)`) was muted by the demo's
	    // console.error override, so no surfaced error. See drawBuffers impl
	    // comment above for the full chain.
	    {"drawBuffers", w_draw_buffers},
	    // ============================================================
	    // Phase-1.5-LOW — 30 core WebGL2 methods (tier LOW per plan §0.1).
	    // Grouped by Khronos family to align with verify-patches.sh
	    // check-family layout. Counter jump 17 → 47 / 88 (all in this
	    // tier are v2-only, so v1 install_methods FUNCS[] is untouched).
	    // ============================================================
	    // Buffer ops (2)
	    {"getBufferSubData", w_get_buffer_sub_data},
	    {"copyBufferSubData", w_copy_buffer_sub_data},
	    // Framebuffer thin (6)
	    {"framebufferTextureLayer", w_framebuffer_texture_layer},
	    {"invalidateFramebuffer", w_invalidate_framebuffer},
	    {"invalidateSubFramebuffer", w_invalidate_sub_framebuffer},
	    {"readBuffer", w_read_buffer},
	    {"renderbufferStorageMultisample", w_renderbuffer_storage_multisample},
	    {"getFragDataLocation", w_get_frag_data_location},
	    // 3D texture family (3)
	    {"copyTexSubImage3D", w_copy_tex_sub_image_3d},
	    {"compressedTexImage3D", w_compressed_tex_image_3d},
	    {"compressedTexSubImage3D", w_compressed_tex_sub_image_3d},
	    // UInt uniforms (8)
	    {"uniform1ui", w_uniform_1ui},
	    {"uniform2ui", w_uniform_2ui},
	    {"uniform3ui", w_uniform_3ui},
	    {"uniform4ui", w_uniform_4ui},
	    {"uniform1uiv", w_uniform_1uiv},
	    {"uniform2uiv", w_uniform_2uiv},
	    {"uniform3uiv", w_uniform_3uiv},
	    {"uniform4uiv", w_uniform_4uiv},
	    // Non-square matrix uniforms (6)
	    {"uniformMatrix2x3fv", w_uniform_matrix_2x3fv},
	    {"uniformMatrix3x2fv", w_uniform_matrix_3x2fv},
	    {"uniformMatrix2x4fv", w_uniform_matrix_2x4fv},
	    {"uniformMatrix4x2fv", w_uniform_matrix_4x2fv},
	    {"uniformMatrix3x4fv", w_uniform_matrix_3x4fv},
	    {"uniformMatrix4x3fv", w_uniform_matrix_4x3fv},
	    // Clear buffer family (4)
	    {"clearBufferiv", w_clear_buffer_iv},
	    {"clearBufferuiv", w_clear_buffer_uiv},
	    {"clearBufferfv", w_clear_buffer_fv},
	    {"clearBufferfi", w_clear_buffer_fi},
	    // Draw range (1)
	    {"drawRangeElements", w_draw_range_elements},
	    // ============================================================
	    // Phase-1.5-LOW-MED — 6 methods (tier LOW-MED per plan §0.1).
	    // Counter jump 47 → 53 / 88 (all v2-only, so v1 FUNCS[] stays
	    // untouched — same as phase-1.5-LOW).
	    // ============================================================
	    // Integer vertex attribs (5)
	    {"vertexAttribI4i", w_vertex_attrib_i4i},
	    {"vertexAttribI4ui", w_vertex_attrib_i4ui},
	    {"vertexAttribI4iv", w_vertex_attrib_i4iv},
	    {"vertexAttribI4uiv", w_vertex_attrib_i4uiv},
	    {"vertexAttribIPointer", w_vertex_attrib_i_pointer},
	    // getInternalformatParameter (1)
	    {"getInternalformatParameter", w_get_internalformat_parameter},
	    // ============================================================
	    // Phase-1.5-MED — 25 methods (ledger #53). Counter jump 53 → 78/88.
	    // Sampler (7) + sync (6) + query (7) + UBO introspection (5).
	    // ============================================================
	    // Sampler (7)
	    {"createSampler", w_create_sampler},
	    {"deleteSampler", w_delete_sampler},
	    {"isSampler", w_is_sampler},
	    {"bindSampler", w_bind_sampler},
	    {"samplerParameteri", w_sampler_parameter_i},
	    {"samplerParameterf", w_sampler_parameter_f},
	    {"getSamplerParameter", w_get_sampler_parameter},
	    // Sync (6)
	    {"fenceSync", w_fence_sync},
	    {"isSync", w_is_sync},
	    {"deleteSync", w_delete_sync},
	    {"clientWaitSync", w_client_wait_sync},
	    {"waitSync", w_wait_sync},
	    {"getSyncParameter", w_get_sync_parameter},
	    // Query (7) — batch-3 dedup base per plan §0.1.1
	    {"createQuery", w_create_query},
	    {"deleteQuery", w_delete_query},
	    {"isQuery", w_is_query},
	    {"beginQuery", w_begin_query},
	    {"endQuery", w_end_query},
	    {"getQuery", w_get_query},
	    {"getQueryParameter", w_get_query_parameter},
	    // UBO introspection (5)
	    {"getIndexedParameter", w_get_indexed_parameter},
	    {"getUniformIndices", w_get_uniform_indices},
	    {"getActiveUniforms", w_get_active_uniforms},
	    {"getActiveUniformBlockParameter", w_get_active_uniform_block_parameter},
	    {"getActiveUniformBlockName", w_get_active_uniform_block_name},
	    // #52b — WebGL1 core getTexParameter (also folded into v1
	    // install_methods below).
	    {"getTexParameter", w_get_tex_parameter},
	    // ============================================================
	    // Phase-1.5-MED-HIGH — 10 transform-feedback methods (ledger #55).
	    // Counter jump 78 → 88/88 (closes the WebGL2 spec counter).
	    // ============================================================
	    {"createTransformFeedback", w_create_transform_feedback},
	    {"deleteTransformFeedback", w_delete_transform_feedback},
	    {"isTransformFeedback", w_is_transform_feedback},
	    {"bindTransformFeedback", w_bind_transform_feedback},
	    {"beginTransformFeedback", w_begin_transform_feedback},
	    {"endTransformFeedback", w_end_transform_feedback},
	    {"transformFeedbackVaryings", w_transform_feedback_varyings},
	    {"getTransformFeedbackVarying", w_get_transform_feedback_varying},
	    {"pauseTransformFeedback", w_pause_transform_feedback},
	    {"resumeTransformFeedback", w_resume_transform_feedback},
	    // ============================================================
	    // Batch 3 (ledger #57) — extension entry-point wiring on v2's
	    // FUNCS[]. The methods with a suffix (queryCounterEXT,
	    // clipControlEXT, etc.) are also vended via the extension
	    // object in w_get_extension; landing them on the prototype
	    // gives Three.js and Unity WebGL emitters (which sometimes
	    // call gl.methodEXT directly on the context) a functional
	    // symbol regardless of getExtension-object access order.
	    // ============================================================
	    // EXT_disjoint_timer_query_webgl2 (v2) — queryCounterEXT
	    {"queryCounterEXT", w_query_counter_ext},
	    // EXT_clip_control
	    {"clipControlEXT", w_clip_control_ext},
	    // EXT_polygon_offset_clamp
	    {"polygonOffsetClampEXT", w_polygon_offset_clamp_ext},
	    // KHR_parallel_shader_compile
	    {"maxShaderCompilerThreadsKHR", w_max_shader_compiler_threads_khr},
	    // OES_draw_buffers_indexed — 8 methods (v2 only)
	    {"enableiOES", w_enable_i},
	    {"disableiOES", w_disable_i},
	    {"blendEquationiOES", w_blend_equation_i},
	    {"blendEquationSeparateiOES", w_blend_equation_separate_i},
	    {"blendFunciOES", w_blend_func_i},
	    {"blendFuncSeparateiOES", w_blend_func_separate_i},
	    {"colorMaskiOES", w_color_mask_i},
	    {"isEnablediOES", w_is_enabled_i},
	    // WEBGL_multi_draw — 4 loop-shim methods
	    {"multiDrawArraysWEBGL", w_multi_draw_arrays_webgl},
	    {"multiDrawElementsWEBGL", w_multi_draw_elements_webgl},
	    {"multiDrawArraysInstancedWEBGL", w_multi_draw_arrays_instanced_webgl},
	    {"multiDrawElementsInstancedWEBGL", w_multi_draw_elements_instanced_webgl},
	    // ============================================================
	    // Tier 1 (ledger #58–#63) — v2 mirror of the v1 spec-hole fills.
	    // Same body impl functions; shared across contexts.
	    // ============================================================
	    {"getUniform", w_get_uniform},
	    {"copyTexImage2D", w_copy_tex_image_2d},
	    {"getVertexAttrib", w_get_vertex_attrib},
	    {"getVertexAttribOffset", w_get_vertex_attrib_offset},
	    {"getFramebufferAttachmentParameter", w_get_framebuffer_attachment_parameter},
	    {"getAttachedShaders", w_get_attached_shaders},
	    {"vertexAttrib1fv", w_vertex_attrib_1fv},
	    {"vertexAttrib2fv", w_vertex_attrib_2fv},
	    {"vertexAttrib3fv", w_vertex_attrib_3fv},
	    {"vertexAttrib4fv", w_vertex_attrib_4fv},
	};
	Local<Context> ctx = iso->GetCurrentContext();
	for (const auto &s : FUNCS) {
		if (!s.name || !s.fn) continue;
		proto->Set(ctx, nx_str(iso, s.name),
		           FunctionTemplate::New(iso, s.fn)
		               ->GetFunction(ctx).ToLocalChecked()).Check();
	}
}

// $.webglInitClass(WebGLRenderingContext, {WebGLBuffer, WebGLProgram, ...})
//   — receives the JS class + the helper-class map; stashes prototype
//   carriers for new_gl_obj() to use, and installs the WebGL methods on
//   the class's prototype.
void nx_webgl_init_class(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	// (no per-context state needed here — g_protos + method install are global;
	// the per-canvas WebGLState is created in make_context_carrier.)
	if (info.Length() < 1 || !info[0]->IsFunction()) return;
	Local<Function> cls = info[0].As<Function>();
	Local<Context> ctx = cur(iso);
	Local<Value> proto_v;
	if (!cls->Get(ctx, nx_str(iso, "prototype")).ToLocal(&proto_v)) return;
	if (!proto_v->IsObject()) return;
	Local<Object> proto = proto_v.As<Object>();
	install_methods(iso, proto);

	if (info.Length() < 2 || !info[1]->IsObject()) return;
	Local<Object> classes = info[1].As<Object>();
	struct KV { const char *name; ObjKind kind; };
	static const KV MAP[] = {
	    {"WebGLBuffer", K_BUFFER},
	    {"WebGLFramebuffer", K_FRAMEBUFFER},
	    {"WebGLProgram", K_PROGRAM},
	    {"WebGLRenderbuffer", K_RENDERBUFFER},
	    {"WebGLShader", K_SHADER},
	    {"WebGLTexture", K_TEXTURE},
	    {"WebGLUniformLocation", K_UNIFORM_LOCATION},
	    {"WebGLActiveInfo", K_ACTIVE_INFO},
	    {"WebGLShaderPrecisionFormat", K_SHADER_PRECISION_FORMAT},
	};
	for (const auto &kv : MAP) {
		Local<Value> jc;
		if (!classes->Get(ctx, nx_str(iso, kv.name)).ToLocal(&jc)) continue;
		if (!jc->IsFunction()) continue;
		Local<Value> p;
		if (!jc.As<Function>()->Get(ctx, nx_str(iso, "prototype")).ToLocal(&p))
			continue;
		if (!p->IsObject()) continue;
		g_protos[kv.kind].Reset(iso, p.As<Object>());
	}
}

// Phase 2.G.0 — $.webgl2InitClass(WebGL2RenderingContext, { handle map })
// receives the JS WebGL2RenderingContext class + the helper-class map;
// stashes prototype carriers (additive — v2 handle set is a superset of v1,
// shares the same K_* slots) and installs the v2 method table on the class's
// prototype. Separate symbol from nx_webgl_init_class to keep the v1 install
// path byte-identical to its hardware-verified shape (see install_methods()'s
// JIT-safety rationale block).
void nx_webgl2_init_class(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	// (no per-context state needed here — g_protos + method install are global;
	// the per-canvas WebGLState is created in make_context_carrier.)
	if (info.Length() < 1 || !info[0]->IsFunction()) return;
	Local<Function> cls = info[0].As<Function>();
	Local<Context> ctx = cur(iso);
	Local<Value> proto_v;
	if (!cls->Get(ctx, nx_str(iso, "prototype")).ToLocal(&proto_v)) return;
	if (!proto_v->IsObject()) return;
	Local<Object> proto = proto_v.As<Object>();
	install_methods_v2(iso, proto);

	// Handle class map. v2 adds v2-only handle kinds (WebGLVertexArrayObject
	// cut #3; WebGLQuery, WebGLSampler, WebGLSync, WebGLTransformFeedback
	// in future cuts as the slice's diag-proxy reports each create*).
	if (info.Length() < 2 || !info[1]->IsObject()) return;
	Local<Object> classes = info[1].As<Object>();
	struct KV { const char *name; ObjKind kind; };
	static const KV MAP[] = {
	    {"WebGLBuffer", K_BUFFER},
	    {"WebGLFramebuffer", K_FRAMEBUFFER},
	    {"WebGLProgram", K_PROGRAM},
	    {"WebGLRenderbuffer", K_RENDERBUFFER},
	    {"WebGLShader", K_SHADER},
	    {"WebGLTexture", K_TEXTURE},
	    {"WebGLUniformLocation", K_UNIFORM_LOCATION},
	    {"WebGLActiveInfo", K_ACTIVE_INFO},
	    {"WebGLShaderPrecisionFormat", K_SHADER_PRECISION_FORMAT},
	    // v2-only handle kinds (cut #3+):
	    {"WebGLVertexArrayObject", K_VERTEX_ARRAY_OBJECT},
	    // Phase-1.5-MED handle kinds (ledger #53):
	    {"WebGLQuery", K_QUERY},
	    {"WebGLSampler", K_SAMPLER},
	    {"WebGLSync", K_SYNC},
	    // Phase-1.5-MED-HIGH handle kind (ledger #55):
	    {"WebGLTransformFeedback", K_TRANSFORM_FEEDBACK},
	};
	for (const auto &kv : MAP) {
		Local<Value> jc;
		if (!classes->Get(ctx, nx_str(iso, kv.name)).ToLocal(&jc)) continue;
		if (!jc->IsFunction()) continue;
		Local<Value> p;
		if (!jc.As<Function>()->Get(ctx, nx_str(iso, "prototype")).ToLocal(&p))
			continue;
		if (!p->IsObject()) continue;
		// Additive-safe: WebGLBuffer / WebGLFramebuffer / etc. are EXPORTED
		// from webgl2-rendering-context.ts and IMPORTED by v1's
		// webgl-rendering-context.ts at line 24-46 — one class per kind,
		// shared symbol. Module evaluation order is v2 first (v1 depends on
		// it for the handle classes), v1 second. So when this runs at v2's
		// init, the slot is still empty; when v1's init runs afterward, it
		// unconditionally Reset()s the same slot to the SAME prototype
		// pointer. Net effect: identical prototype stashed regardless of
		// order. The IsEmpty() check here is a defensive no-op against
		// future re-orderings (e.g. if v1 is ever decoupled from v2's
		// handle exports and runs first).
		if (g_protos[kv.kind].IsEmpty()) {
			g_protos[kv.kind].Reset(iso, p.As<Object>());
		}
	}
	fprintf(stderr, "[webgl2] init_class ok (empty v2 method table; "
	                "shape verification only)\n");
	fflush(stderr);
}

// Internal shared helper for both v1 and v2 context factories. Returns an
// empty Local<Object>() on failure (caller sets info return to undefined ->
// TS createWebGL*Context returns null). `is_v2` tags the carrier so engine-
// side dispatchers added in 2.G.1+ can branch on context kind.
// GC finalizer for a per-context WebGLState (attached via nx::Wrap in
// make_context_carrier). Runs during V8 GC where the shared GL context is NOT
// current — so it must NOT issue any GL calls. It queues the context's tenant
// FBO id for deletion at the next GL-current point (enter_bracket /
// make_context_carrier via process_pending_tenant_frees) and frees the CPU
// struct. The tenant's auto_user_vao + any GL names the context created leak
// until bridge teardown (same as the pre-multi-context behavior — the finalizer
// has no GL context); the tenant FBO is the significant allocation and IS
// reclaimed. tenant_id 0 (the shared default) is never freed here.
void free_webgl_state(WebGLState *s) {
	if (!s) return;
	if (s->tenant_id != 0) g_tenants_to_free.push_back(s->tenant_id);
	if (st == s) st = nullptr; // don't leave `st` dangling at the freed state
	delete s;
}

static Local<Object> make_context_carrier(Isolate *iso,
                                          const FunctionCallbackInfo<Value> &info,
                                          bool is_v2) {
	// Skia must be up — without the shared ES3 context + GrDirectContext,
	// the bridge can't init. Caller (TS) treats this as "no GL available".
	if (!nx_skia_gpu_egl_context() || !nx_skia_gpu_gr_context()) {
		fprintf(stderr, "[webgl%s] context_new refused: skia_gpu not ready\n",
		        is_v2 ? "2" : "");
		fflush(stderr);
		return Local<Object>();
	}

	// Reclaim any tenants whose contexts were GC'd (GL is current here — the
	// GC finalizer only queued the ids). See free_webgl_state.
	process_pending_tenant_frees();

	// Read canvas dimensions if a canvas was passed.
	int w = 640, h = 360;
	if (info.Length() >= 1 && info[0]->IsObject()) {
		Local<Object> canvas = info[0].As<Object>();
		Local<Context> ctx = cur(iso);
		Local<Value> vw, vh;
		if (canvas->Get(ctx, nx_str(iso, "width")).ToLocal(&vw) &&
		    vw->IsNumber())
			w = vw->Int32Value(ctx).FromMaybe(w);
		if (canvas->Get(ctx, nx_str(iso, "height")).ToLocal(&vh) &&
		    vh->IsNumber())
			h = vh->Int32Value(ctx).FromMaybe(h);
	}
	if (w <= 0) w = 640;
	if (h <= 0) h = 360;

	// Allocate a FRESH per-context WebGLState (multi-WebGL-canvas independence)
	// and make it active. Each page <canvas> gets its own state + its own
	// bridge tenant id: 0 for the first context (reuses the legacy default
	// tenant, zero extra GPU), 1,2,... for stacked canvases. The state is
	// attached to the returned carrier via nx::Wrap so use_ctx(info.This())
	// selects it on every method call; free_webgl_state frees it at GC.
	WebGLState *cst = new WebGLState();
	cst->tenant_id = g_next_tenant_id++;
	cst->width = w;
	cst->height = h;
	st = cst;

	// Bring up the bridge (creates the default tenant 0) on the first context.
	if (!nx_webgl_bridge_is_initialized()) {
		if (!nx_webgl_bridge_init(w, h)) {
			fprintf(stderr,
			        "[webgl%s] context_new refused: bridge_init failed\n",
			        is_v2 ? "2" : "");
			fflush(stderr);
			st = nullptr;
			delete cst;
			return Local<Object>();
		}
	}
	// Ensure THIS canvas's tenant FBO exists (id 0 == the default just created;
	// >0 == a dedicated per-canvas FBO so stacked WebGL canvases are isolated).
	if (!nx_webgl_bridge_tenant_ensure(cst->tenant_id, w, h)) {
		fprintf(stderr,
		        "[webgl%s] context_new refused: tenant %u ensure failed\n",
		        is_v2 ? "2" : "", cst->tenant_id);
		fflush(stderr);
		st = nullptr;
		delete cst;
		return Local<Object>();
	}
	nx_webgl_bridge_set_webgl_owned(true);

	// ── Multi-canvas set-once persistence fix (2026-09-10) ──────────────────
	// Seed THIS fresh context's user_snap with clean WebGL default state NOW, at
	// creation, and mark it valid — so apply_canvas_state ALWAYS takes the
	// RESTORE branch for this context and NEVER the lazy SAVE branch.
	//
	// The bug this fixes (pinned by per-op native instrumentation): the SAVE
	// branch used to fire LATE — at the context's first *fresh* per-frame
	// bracket, which for a page context lands mid-app-init because the shell's
	// bracket is already open during the app's setup calls (so those setters'
	// enter_bracket early-returns and never triggers the SAVE). By the time the
	// SAVE finally ran, the app had already shadow-written its intended state
	// into user_snap (program=25, clearColor=0.04, its own VAO), and
	// nx_gl_state_save OVERWROTE all of it with whatever Skia had left live
	// (program 7, clearColor 0,0,0, the auto VAO). Every later frame then
	// RESTORED that clobbered snapshot → the app's program was never current
	// (uniform* → GL_INVALID_OPERATION 0x502) and clears were black. Re-emit
	// apps (three.js / dusk / the shell wallpaper / the multiwebgl demo) only
	// survived by re-issuing full GL state every frame, which repaired the
	// snapshot each frame — set-once apps (compass / midisurface /
	// nxjswebgl2test / gravityballs / glsingletest) had nothing to repair it.
	//
	// Seeding here — before ANY app GL call — establishes valid=true up front,
	// so the destructive SAVE never runs and the app's init shadow-writes
	// accumulate on a correct baseline and RESTORE faithfully every frame. This
	// mirrors why the pre-regression shared context (tenant 0) always worked:
	// its user_snap was captured while GL was in clean defaults. Uses
	// nx_gl_state_save (not hand-filled fields) so the dynamic entries — tenant
	// FBO id, the auto VAO name, GL_TEXTURE0 as the active unit — are captured
	// correctly rather than guessed. Skia's live GL state is saved and restored
	// around the seed so the shell's in-flight frame is undisturbed.
	{
		nx_gl_state_snap_t skia_snap;
		nx_gl_state_save(&skia_snap);
		// Bind this context's tenant FBO (read+draw) so the captured
		// user_snap.fbo / read_fbo point at the tenant, matching apply's bind.
		glBindFramebuffer(GL_FRAMEBUFFER, nx_webgl_bridge_tenant_fbo(cst->tenant_id));
		// Allocate + bind the persistence VAO so WebGL-1 attribute state has a
		// home Ganesh doesn't touch; the capture records it as the baseline VAO.
		if (cst->auto_user_vao == 0) glGenVertexArrays(1, &cst->auto_user_vao);
		glBindVertexArray(cst->auto_user_vao);
		// Drive live GL to WebGL spec defaults, then capture into user_snap.
		glUseProgram(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDepthMask(GL_TRUE);
		glStencilMask(0xFFFFFFFFu);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glDisable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_STENCIL_TEST);
		glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ONE, GL_ZERO);
		glActiveTexture(GL_TEXTURE0);
		glBindSampler(0, 0);           // ES 3.x shared ctx; safe for v1 + v2
		glBindTexture(GL_TEXTURE_2D, 0);
		glViewport(0, 0, (GLsizei)w, (GLsizei)h);
		nx_gl_state_save(&cst->user_snap);
		cst->user_snap_valid = true;
		// Put Skia's live GL state back — do NOT leave the shell disturbed.
		nx_gl_state_restore(&skia_snap);
		GrDirectContext *gr = nx_skia_gpu_gr_context();
		if (gr) gr->resetContext();
	}

	// WebGL-1 attribute-VAO fix (2026-09-10): if the shell's per-frame bracket
	// is ALREADY OPEN when the app calls getContext (the common case — the app
	// inits mid-frame), the app's upcoming init GL calls will find st==cst AND
	// the bracket open, so NEITHER use_ctx (no context switch — st is already
	// cst) NOR enter_bracket (already open) runs apply_canvas_state(). That
	// leaves the PREVIOUS context's VAO bound, so a WebGL-1 app's
	// enableVertexAttribArray/vertexAttribPointer (which have no explicit VAO —
	// they rely on this context's auto_user_vao) land on the WRONG VAO. At draw
	// the correct-but-empty auto VAO is bound → no geometry (this was proven by
	// per-op native instrumentation: the app's attribute setup bound the
	// previous context's VAO, and attrib 0 was disabled with no buffer at draw).
	// WebGL-2 apps escape this because they bindVertexArray(ownVAO) explicitly.
	// Apply cst's canvas state NOW (binds its tenant FBO + auto VAO + seeded
	// snapshot) so init calls land on the right VAO. Mirrors use_ctx's
	// mid-bracket apply; harmless when the bracket is closed (skipped — the
	// app's first GL call will open a bracket and apply then).
	if (g_bracket_open) apply_canvas_state();

	// Phase-0 — populate the native GL extension cache and emit the
	// [gl-ext-dump] one-shot boot log. Bridge init above guarantees the
	// shared ES3 EGL context is current; glGetIntegerv(GL_NUM_EXTENSIONS)
	// + glGetStringi(GL_EXTENSIONS, i) is legal at this point. First
	// context creation fires the dump; subsequent creations are cheap
	// no-ops via the `s_native_exts_populated` guard.
	populate_native_extensions();

	// Mint the context carrier object. The TS factory sets its prototype to
	// WebGL{2}RenderingContext, so install_methods{,_v2} having populated the
	// prototype is what makes instance methods reachable.
	Local<Object> ctx_obj = nx::NewWrapped(iso);
	// Attach THIS context's per-canvas WebGLState so use_ctx(info.This()) can
	// select it on every method call (multi-WebGL-canvas independence). GC of
	// the carrier frees the state via free_webgl_state (which queues the tenant
	// FBO for a GL-current delete).
	nx::Wrap<WebGLState>(iso, ctx_obj, cst, free_webgl_state);
	Local<Context> jctx = cur(iso);
	ctx_obj->Set(jctx, nx_str(iso, "drawingBufferWidth"),
	             Int32::New(iso, w)).Check();
	ctx_obj->Set(jctx, nx_str(iso, "drawingBufferHeight"),
	             Int32::New(iso, h)).Check();
	if (is_v2) {
		ctx_obj->Set(jctx, nx_str(iso, "__webgl2"),
		             Boolean::New(iso, true)).Check();
	}
	fprintf(stderr, "[webgl%s] context_new ok %dx%d%s\n",
	        is_v2 ? "2" : "", w, h,
	        is_v2 ? " (v2 wrapper, empty methods — 2.G.0)" : "");
	fflush(stderr);
	return ctx_obj;
}

// $.webglContextNew(canvas) — main factory. Returns a wrapped object that
// the TS side adds the WebGLRenderingContext prototype to. Returns undefined
// on failure (TS returns null from getContext).
void nx_webgl_context_new(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Object> obj = make_context_carrier(iso, info, /*is_v2=*/false);
	if (obj.IsEmpty()) return;
	info.GetReturnValue().Set(obj);
}

// Phase 2.G.0 — $.webgl2ContextNew(canvas). Separate factory symbol from v1.
// Currently shares engine state (WebGLState `st` is process-wide) with v1 —
// there is no per-context state divergence in 2.G.0. The separate symbol is
// the load-bearing structural choice: createWebGL2Context calls this,
// createWebGLContext calls nx_webgl_context_new, and the install paths via
// $.webgl{2}InitClass are wholly distinct. Returns undefined on failure.
void nx_webgl2_context_new(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Object> obj = make_context_carrier(iso, info, /*is_v2=*/true);
	if (obj.IsEmpty()) return;
	info.GetReturnValue().Set(obj);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API used by main.cc — the present-time integration.
// ---------------------------------------------------------------------------

// True only when this file owns the screen (legacy: WebGL2-screen path; in
// 2.C we render INSIDE Skia's canvas, so this stays false — the per-frame
// compose hook below is what 2.C uses, not the WebGL-owns-NWindow path).
bool nx_webgl_active(void) { return false; }
void nx_webgl_present(void) {}
void nx_webgl_exit(void) {}

// Called from main.cc's GPU canvas present branch, BEFORE nx_skia_gpu_present.
// Closes the per-frame bracket if it's open (restores Skia's GL state +
// resetContext), then asks the bridge to composite the tenant FBO into
// Skia's persistent canvas surface. The bridge's compose is dirty-gated, so
// the call is cheap on frames where WebGL didn't draw.
void nx_webgl_compose_if_active(SkSurface *target) {
	if (!st) return;
	if (g_bracket_open) exit_bracket();
	if (target) nx_webgl_bridge_compose(target);
}

// Tier 1 (ledger #64) — WebGL-surface framebuffer readback for Screen.toDataURL.
//
// Reads the tenant FBO's current color contents to a heap-allocated BGRA
// buffer, top-down (Y-flipped relative to GL's bottom-up), so canvas.cc's
// `encode_pixels` path (Skia PNG encoder + libjpeg-turbo + libwebp — all
// consuming a top-down BGRA pixmap) can hand back a data URL over what
// WebGL rendered without needing a separate encode surface.
//
// Ownership: on success, `*out_bgra` is a fresh malloc()'d buffer of size
// `(*out_w) * (*out_h) * 4` bytes; the caller `free()`s it. On failure,
// nothing is written to the out-params and the function returns false.
//
// State discipline. The function saves + restores GL_READ_FRAMEBUFFER_BINDING
// and GL_PACK_ALIGNMENT around the call. It deliberately does NOT enter the
// per-frame bracket (`enter_bracket()`), because:
//   - `enter_bracket()` binds DRAW_FRAMEBUFFER (not READ_FRAMEBUFFER) to the
//     tenant FBO — Skia's rendering uses DRAW_FRAMEBUFFER, and the two are
//     independent in ES3. Reading from the tenant FBO via READ_FRAMEBUFFER
//     doesn't disturb Skia's DRAW state.
//   - The 2.B nx_gl_state_snap_t does NOT cover READ_FRAMEBUFFER (patch #17
//     later added it only for the ACTIVE probe path). Not entering the
//     bracket keeps this readback outside the frozen bracket-machinery
//     contract (blast-radius rule in this session).
//   - PACK_ALIGNMENT is not in the snap either; we save+restore it locally.
//
// Returns false if the bridge isn't initialized, the FBO has zero dimensions,
// or the malloc fails. Caller falls through to the existing raster path.
bool nx_webgl_snapshot_bridge_rgba8(int *out_w, int *out_h,
                                     uint8_t **out_bgra) {
	if (!out_w || !out_h || !out_bgra) return false;
	if (!nx_webgl_bridge_is_initialized()) return false;
	// Snapshot the ACTIVE context's tenant FBO (multi-WebGL-canvas: `st` is the
	// last-active context). Falls back to the default tenant when no context.
	const uint32_t tid = st ? st->tenant_id : 0;
	GLuint fbo = nx_webgl_bridge_tenant_fbo(tid);
	if (fbo == 0) return false;
	int fbo_w = 0, fbo_h = 0;
	nx_webgl_bridge_tenant_size(tid, &fbo_w, &fbo_h);
	if (fbo_w <= 0 || fbo_h <= 0) return false;

	// Save the two GL states we're about to touch. Values 0 (default binding)
	// / 4 (default alignment) are safe restore defaults if the query fails.
	GLint saved_read_fbo = 0;
	GLint saved_pack_align = 4;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &saved_read_fbo);
	glGetIntegerv(GL_PACK_ALIGNMENT, &saved_pack_align);

	// Drain any pre-existing GL error so a post-read glGetError reflects only
	// this readback.
	while (glGetError() != GL_NO_ERROR) { /* drain */ }

	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	const size_t px_count = (size_t)fbo_w * (size_t)fbo_h;
	const size_t byte_count = px_count * 4;
	uint8_t *rgba = (uint8_t *)malloc(byte_count);
	if (!rgba) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)saved_read_fbo);
		glPixelStorei(GL_PACK_ALIGNMENT, saved_pack_align);
		return false;
	}
	glReadPixels(0, 0, fbo_w, fbo_h, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
	const GLenum read_err = glGetError();

	// Restore state before allocating the destination buffer, so an OOM on
	// the second malloc doesn't leave the readback bindings dangling.
	glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)saved_read_fbo);
	glPixelStorei(GL_PACK_ALIGNMENT, saved_pack_align);

	if (read_err != GL_NO_ERROR) {
		free(rgba);
		return false;
	}

	uint8_t *bgra = (uint8_t *)malloc(byte_count);
	if (!bgra) {
		free(rgba);
		return false;
	}
	// Y-flip + RGBA→BGRA swizzle in one pass. GL returns bottom-up; PNG/JPEG
	// encoders + canvas.cc's Skia paint pixmap all expect top-down BGRA.
	for (int y = 0; y < fbo_h; y++) {
		const uint8_t *src_row = rgba + (size_t)(fbo_h - 1 - y) * (size_t)fbo_w * 4;
		uint8_t *dst_row = bgra + (size_t)y * (size_t)fbo_w * 4;
		for (int x = 0; x < fbo_w; x++) {
			const uint8_t r = src_row[x * 4 + 0];
			const uint8_t g = src_row[x * 4 + 1];
			const uint8_t b = src_row[x * 4 + 2];
			const uint8_t a = src_row[x * 4 + 3];
			dst_row[x * 4 + 0] = b;
			dst_row[x * 4 + 1] = g;
			dst_row[x * 4 + 2] = r;
			dst_row[x * 4 + 3] = a;
		}
	}
	free(rgba);
	*out_w = fbo_w;
	*out_h = fbo_h;
	*out_bgra = bgra;
	return true;
}

void nx_init_webgl(v8::Isolate *iso, v8::Local<v8::Object> init_obj) {
	NX_SET_FUNC(init_obj, "webglContextNew", nx_webgl_context_new);
	NX_SET_FUNC(init_obj, "webglInitClass", nx_webgl_init_class);
	// Phase 2.G.0 — separate v2 factory + init pair. Empty method table
	// for 2.G.0 (shape verification only); methods land in 2.G.1.
	NX_SET_FUNC(init_obj, "webgl2ContextNew", nx_webgl2_context_new);
	NX_SET_FUNC(init_obj, "webgl2InitClass", nx_webgl2_init_class);
}
