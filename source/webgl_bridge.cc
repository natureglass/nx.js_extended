#include "webgl_bridge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unordered_map>

#include "include/core/SkBlendMode.h"
#include "include/core/SkCanvas.h"
#include "include/core/SkColor.h"
#include "include/core/SkColorSpace.h"
#include "include/core/SkFont.h"
#include "include/core/SkImage.h"
#include "include/core/SkPaint.h"
#include "include/core/SkRect.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkSamplingOptions.h"
#include "include/gpu/GpuTypes.h"
#include "include/gpu/ganesh/GrBackendSurface.h"
#include "include/gpu/ganesh/GrDirectContext.h"
#include "include/gpu/ganesh/SkImageGanesh.h"
#include "include/gpu/ganesh/gl/GrGLBackendSurface.h"
#include "include/gpu/ganesh/gl/GrGLTypes.h"

namespace {

bool s_initialized = false;
int s_fbo_w = 0;
int s_fbo_h = 0;

// Tenant GL handles. Owned by webgl_bridge; freed in nx_webgl_bridge_exit().
GLuint s_fbo = 0;
GLuint s_color_tex = 0;
GLuint s_depth_rb = 0;

// Phase 2.C ownership flag. When true (WebGL context exists), the 2.B test
// driver yields its render step; compose path is shared.
bool s_webgl_owned = false;

// Per-frame dirty flag: set when the tenant FBO has been written to since the
// last compose, cleared by nx_webgl_bridge_compose. Drops the SkImage draw on
// idle frames (no WebGL traffic + no test_fbo).
bool s_fbo_dirty = false;

// Auto-flush gate: when true (default), nx_webgl_bridge_compose stomps the
// dirty FBO onto Skia's canvas surface every frame. When false, the runtime
// is opting to drive its own per-canvas readback (canvas-runner.ts reads
// gl.readPixels into each <canvas>'s OffscreenCanvas, which Skia paints at
// the canvas's CSS layout slot) and the engine compose MUST stay out of the
// way or it overwrites the correctly-placed DOM. The 2.B test-FBO smoke path
// never calls setBridgeAutoFlush so its default-true behavior is preserved.
// Toggled by gl.setBridgeAutoFlush(bool) via webgl.cc::w_set_bridge_auto_flush.
bool s_auto_flush = true;

// 2.B test program. Compiled once at init; not WebGL — just enough hand-
// written GL to land a visible draw in the FBO that 2.D can read back.
GLuint s_test_prog = 0;
GLuint s_test_vbo = 0;
GLuint s_test_vao = 0;
GLint s_test_u_color = -1;

// Compose path: SkImage wrapping the tenant FBO color texture, allocated
// once and reused every frame. Re-borrowed if the FBO changes.
sk_sp<SkImage> s_compose_image;

// Animation state for the 2.B test draw + perf metering.
uint64_t s_t_start_ns = 0;
uint64_t s_frame_count = 0;
double s_boundary_us_accum = 0.0;

// Phase 2.G.0 state-contract probe gate. Off by default; set true at engine
// boot via nx_webgl_state_probe_enable(cfg->webgl_state_probe). The probe is
// read-only.
bool s_state_probe_on = false;

// #16-ACTIVE state-leak probe gate. Off by default; opt-in via [webgl]
// state_probe_active = true (see nx_webgl_state_probe_active_enable). Runs
// ONCE per launch — SET on compose #1, READ_BACK on compose #2, then
// s_active_probe_done latches and subsequent composes skip. See
// NXJS_PATCHES_NEEDED.md #16-ACTIVE spec for interpretation of verdicts.
bool s_state_probe_active_on = false;
bool s_active_probe_done = false;

// Saved state for the four #16-ACTIVE candidates. Only valid between the
// compose-#1 SET and compose-#2 READ_BACK.
struct nx_active_probe_state_t {
	// (a) UBO slot 3 (Ganesh docs use 0..2; slot 3 is first Three.js contender)
	GLuint ubo_probe_obj;
	GLint ubo3_pre;
	GLint ubo3_set;
	GLint ubo0_pre; // slot 0 only READ (Ganesh-touched)
	// (b) Sampler unit 0
	GLuint samp_probe_obj;
	GLint samp0_pre;
	GLint samp0_set;
	// (c) READ_FRAMEBUFFER (separate from DRAW — passive probe showed if the
	// two diverge; active tests whether Skia binds both together)
	GLuint rfb_probe_obj;
	GLint read_fbo_pre;
	GLint read_fbo_set;
	// (d) Transform feedback + rasterizer-discard
	GLuint tf_probe_obj;
	GLint tf_pre;
	GLint tf_set;
	GLboolean rd_pre;
	GLboolean rd_set;
};
nx_active_probe_state_t s_ap_state = {};

// Frame counter for the compose-path probe sampling. Independent of
// s_frame_count (which only counts test_fbo frames); this one ticks on
// every nx_webgl_bridge_compose call regardless of dirty/auto-flush. Sampled
// at 1, 60, 600 — early baseline + 1s + 10s of steady state.
uint64_t s_probe_compose_n = 0;

bool probe_sample_this_frame(uint64_t n) {
	return n == 1 || n == 60 || n == 600;
}

uint64_t ts_ns() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

GLuint compile_shader(GLenum type, const char *src) {
	GLuint sh = glCreateShader(type);
	glShaderSource(sh, 1, &src, nullptr);
	glCompileShader(sh);
	GLint ok = 0;
	glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[1024];
		GLsizei n = 0;
		glGetShaderInfoLog(sh, sizeof(log), &n, log);
		fprintf(stderr, "[webgl-bridge] shader compile fail: %.*s\n", (int)n,
		        log);
		fflush(stderr);
		glDeleteShader(sh);
		return 0;
	}
	return sh;
}

// Create a color-texture + depth24-stencil8 + FBO trio of the given size into
// caller-owned out-params. Generalized from the original single-tenant
// create_fbo so per-canvas tenants (multi-WebGL-canvas independence) allocate
// byte-identical resources. Leaves FBO 0 bound on both success and failure.
bool create_fbo_resources(int w, int h, GLuint *out_fbo, GLuint *out_color,
                          GLuint *out_depth) {
	GLuint color_tex = 0, depth_rb = 0, fbo = 0;
	glGenTextures(1, &color_tex);
	glBindTexture(GL_TEXTURE_2D, color_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
	             GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Phase-0 commit 2 — combined depth/stencil renderbuffer to honor the
	// context's advertised `stencil: true` and produce 8 bits at
	// `getParameter(STENCIL_BITS)`. Nouveau typically stores DEPTH_COMPONENT24
	// as padded D24X8 anyway, so DEPTH24_STENCIL8 claims already-allocated
	// bits — no measurable memory delta on Tegra. Unlocks Unity RectMask2D
	// and Phaser stencil masks post-hardware-verify. Depth path preserved:
	// depth attachment goes through the SAME renderbuffer via the combined
	// DEPTH_STENCIL_ATTACHMENT point (single attach, ES3-preferred), so
	// DEPTH_BITS still reports 24. See NXJS_PATCHES_NEEDED.md #46.
	glGenRenderbuffers(1, &depth_rb);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
	                       color_tex, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
	                          GL_RENDERBUFFER, depth_rb);
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	// Unbind BEFORE checking status so the failure path leaves Skia's FBO 0
	// current rather than our incomplete tenant.
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		// Distinctive tag [bridge-fbo:INCOMPLETE] — grep target for
		// hardware-session gate (a) per phase-0 commit-2 rider 3. Citron
		// is authoritative for THIS assert (functional).
		fprintf(stderr,
		        "[bridge-fbo:INCOMPLETE] status=0x%x — DEPTH24_STENCIL8 attach failed\n",
		        (unsigned)status);
		fflush(stderr);
		if (fbo) glDeleteFramebuffers(1, &fbo);
		if (color_tex) glDeleteTextures(1, &color_tex);
		if (depth_rb) glDeleteRenderbuffers(1, &depth_rb);
		return false;
	}
	// Positive-path breadcrumb for the same gate; the assert holds when this
	// fires and [bridge-fbo:INCOMPLETE] does not.
	fprintf(stderr,
	        "[bridge-fbo:complete] %dx%d color=RGBA8 depth=24 stencil=8 (combined attach)\n",
	        w, h);
	fflush(stderr);
	*out_fbo = fbo;
	*out_color = color_tex;
	*out_depth = depth_rb;
	return true;
}

bool create_fbo(int w, int h) {
	return create_fbo_resources(w, h, &s_fbo, &s_color_tex, &s_depth_rb);
}

bool create_test_program(void) {
	// Solid-color triangle program. Uses ES 3 `#version 300 es` because the
	// shared context is ES3 (Phase 2.A) — that matches the spike. The 2D
	// vertex layout makes the triangle visible regardless of MVP setup.
	const char *vs = "#version 300 es\n"
	                 "layout(location=0) in vec2 a_pos;\n"
	                 "void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
	const char *fs = "#version 300 es\n"
	                 "precision mediump float;\n"
	                 "uniform vec4 u_color;\n"
	                 "out vec4 outColor;\n"
	                 "void main(){ outColor = u_color; }\n";
	GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
	if (!v) return false;
	GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
	if (!f) { glDeleteShader(v); return false; }
	s_test_prog = glCreateProgram();
	glAttachShader(s_test_prog, v);
	glAttachShader(s_test_prog, f);
	glLinkProgram(s_test_prog);
	GLint ok = 0;
	glGetProgramiv(s_test_prog, GL_LINK_STATUS, &ok);
	glDeleteShader(v);
	glDeleteShader(f);
	if (!ok) {
		char log[1024];
		GLsizei n = 0;
		glGetProgramInfoLog(s_test_prog, sizeof(log), &n, log);
		fprintf(stderr, "[webgl-bridge] program link fail: %.*s\n", (int)n,
		        log);
		fflush(stderr);
		glDeleteProgram(s_test_prog);
		s_test_prog = 0;
		return false;
	}
	s_test_u_color = glGetUniformLocation(s_test_prog, "u_color");

	// Triangle in clip space. Big enough to mostly fill the FBO while
	// leaving a tinted border (the clear color) so we can confirm BOTH the
	// glClear AND the draw made it through compositing.
	const float verts[] = {-0.7f, -0.7f, 0.7f, -0.7f, 0.0f, 0.8f};
	glGenVertexArrays(1, &s_test_vao);
	glBindVertexArray(s_test_vao);
	glGenBuffers(1, &s_test_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, s_test_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return true;
}

// Render an animated clear-+-triangle into the tenant FBO. Caller has already
// captured GL state via nx_gl_state_save(); we leave state mutated but the
// caller's restore handles it.
void render_test_into_fbo(float t) {
	glBindFramebuffer(GL_FRAMEBUFFER, s_fbo);
	glViewport(0, 0, s_fbo_w, s_fbo_h);
	// Animate clear color so a stuck-frame is obvious (constant fill = stale
	// frame; cycling = present is actually advancing).
	float r = 0.05f + 0.05f * sinf(t * 0.7f);
	float g = 0.10f + 0.05f * cosf(t * 0.9f);
	float b = 0.25f + 0.05f * sinf(t * 1.1f);
	glClearColor(r, g, b, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// Disable Skia's typical enables so our minimal draw isn't accidentally
	// dependent on inherited state. The save snap captured what to restore.
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glUseProgram(s_test_prog);
	// Triangle color cycles too — magenta-ish, contrasts with the dark clear.
	float tr = 0.7f + 0.3f * sinf(t * 2.0f);
	float tg = 0.2f + 0.2f * cosf(t * 1.4f);
	float tb = 0.8f;
	glUniform4f(s_test_u_color, tr, tg, tb, 1.0f);
	glBindVertexArray(s_test_vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Borrow a color texture as an SkImage. Borrowing means Skia does NOT own the
// GL handle — webgl_bridge keeps owning it, must outlive any Skia draw that
// references the image, and frees it before the GrDirectContext goes away.
// Generalized from wrap_compose_image so per-canvas tenants can each borrow
// their own color texture.
sk_sp<SkImage> borrow_fbo_image(GLuint color_tex, int w, int h) {
	GrGLTextureInfo tinfo;
	tinfo.fTarget = GL_TEXTURE_2D;
	tinfo.fID = color_tex;
	tinfo.fFormat = 0x8058; // GL_RGBA8
	GrBackendTexture btex = GrBackendTextures::MakeGL(w, h,
	                                                  skgpu::Mipmapped::kNo,
	                                                  tinfo);
	GrDirectContext *gr = nx_skia_gpu_gr_context();
	if (!gr) return nullptr;
	sk_sp<SkImage> img = SkImages::BorrowTextureFrom(
	    gr, btex, kBottomLeft_GrSurfaceOrigin, kRGBA_8888_SkColorType,
	    kPremul_SkAlphaType, nullptr);
	if (!img) {
		fprintf(stderr, "[webgl-bridge] SkImages::BorrowTextureFrom failed\n");
		fflush(stderr);
	}
	return img;
}

void wrap_compose_image(void) {
	s_compose_image = borrow_fbo_image(s_color_tex, s_fbo_w, s_fbo_h);
}

// ---------------------------------------------------------------------------
// Per-canvas tenant registry (multi-WebGL-canvas independence).
//
// tenant_id 0 is the legacy/default tenant backed by the globals above
// (s_fbo / s_color_tex / s_depth_rb / s_fbo_w/h / s_fbo_dirty /
// s_compose_image) — single-canvas apps and the shell are byte-for-byte
// unchanged. tenant_id > 0 lives in this map, one entry per additional page
// WebGL canvas, so two stacked canvases draw into independent FBOs and each
// composites its own contents into its own DOM slot.
// ---------------------------------------------------------------------------
struct ExtraTenant {
	GLuint fbo = 0;
	GLuint color_tex = 0;
	GLuint depth_rb = 0;
	int w = 0;
	int h = 0;
	bool dirty = false;
	sk_sp<SkImage> image;
};
std::unordered_map<uint32_t, ExtraTenant> s_extra_tenants;

// Free an ExtraTenant's GL handles + SkImage. GL must be current.
void destroy_extra_tenant(ExtraTenant &t) {
	t.image.reset();
	if (t.fbo) { glDeleteFramebuffers(1, &t.fbo); t.fbo = 0; }
	if (t.color_tex) { glDeleteTextures(1, &t.color_tex); t.color_tex = 0; }
	if (t.depth_rb) { glDeleteRenderbuffers(1, &t.depth_rb); t.depth_rb = 0; }
}

// Shared sub-rect blit of a borrowed FBO SkImage onto `target`. Factored from
// nx_webgl_bridge_compose_rect so per-canvas tenants reuse the exact same
// Y-flip + clip + 3-arg-drawImageRect discipline (see the cut #12/#13 notes on
// the caller). `fbo_w/h` are the tenant's dims (drive the Y-flip complement).
bool compose_image_rect(SkSurface *target, const sk_sp<SkImage> &img,
                        int fbo_w, int fbo_h, int src_x, int src_y, int src_w,
                        int src_h, int dst_x, int dst_y) {
	if (!target || !img) return false;
	if (src_w <= 0 || src_h <= 0) return false;
	SkCanvas *c = target->getCanvas();
	if (!c) return false;
	const int visual_sy = fbo_h - src_y - src_h;
	const float draw_dx = (float)dst_x - (float)src_x;
	const float draw_dy = (float)dst_y - (float)visual_sy;
	c->save();
	c->clipRect(SkRect::MakeXYWH((float)dst_x, (float)dst_y, (float)src_w,
	                             (float)src_h));
	SkPaint paint;
	c->drawImageRect(img.get(),
	                 SkRect::MakeXYWH(draw_dx, draw_dy, (float)fbo_w,
	                                  (float)fbo_h),
	                 SkSamplingOptions(), &paint);
	c->restore();
	return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Phase 2.G.0 — state-contract probe (read-only).
// ---------------------------------------------------------------------------
//
// Purpose: before extending the FROZEN 2.B nx_gl_state_snap_t, determine via
// hardware probe which of four candidate GL bindings actually leak across
// our shared-context EGL architecture. The shared-context model (Skia and
// WebGL share one EGL context) was the root cause across multiple prior
// bugs — it may make some of the QuickJS-era snap-set assumptions moot
// (Skia/Ganesh-GL never touching the binding => zero leak risk) or
// inversely surface new ones.
//
// The probe queries each candidate at known hook points and prints to
// stderr. The user reads the log on hardware; comparing the same-binding
// value across two consecutive tags reveals whether Skia mutated it between
// those tags. A binding whose value stays constant across all tags does
// NOT need to be in nx_gl_state_snap_t. A binding that changes between
// `init`/`compose-pre`/`compose-post` tags is a candidate for the batched
// snap extension proposed in Deliverable 2.

// GL enums not necessarily in older GLES3 headers — define numerically to
// avoid macro dependency on the gl3.h version vendored upstream.
#ifndef GL_UNIFORM_BUFFER_BINDING
#define GL_UNIFORM_BUFFER_BINDING                  0x8A28
#endif
#ifndef GL_SAMPLER_BINDING
#define GL_SAMPLER_BINDING                         0x8919
#endif
#ifndef GL_READ_FRAMEBUFFER_BINDING
#define GL_READ_FRAMEBUFFER_BINDING                0x8CAA
#endif
#ifndef GL_DRAW_FRAMEBUFFER_BINDING
#define GL_DRAW_FRAMEBUFFER_BINDING                0x8CA6
#endif
#ifndef GL_TRANSFORM_FEEDBACK_BINDING
#define GL_TRANSFORM_FEEDBACK_BINDING              0x8E25
#endif
#ifndef GL_RASTERIZER_DISCARD
#define GL_RASTERIZER_DISCARD                      0x8C89
#endif

void nx_webgl_state_probe_enable(bool on) {
	s_state_probe_on = on;
	if (on) {
		fprintf(stderr, "[webgl-bridge:probe] enabled (Phase 2.G.0 state-contract probe; "
		                "read-only)\n");
		fflush(stderr);
	}
}

bool nx_webgl_state_probe_enabled(void) { return s_state_probe_on; }

void nx_webgl_state_probe_log(const char *tag) {
	if (!s_state_probe_on) return;

	// Defensive: restore GL_ACTIVE_TEXTURE in case any future probe ever
	// changes it. The current probe only reads with the unit already at
	// whatever the caller had selected — we sample SAMPLER_BINDING for
	// the unit that's currently active (typically unit 0 in Three.js's
	// per-frame setup), and we sample for unit 0 explicitly by activating
	// it briefly.
	GLint saved_active_tex = GL_TEXTURE0;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &saved_active_tex);

	// 1. UBO indexed bindings. Sample slot 0..3 (Three.js v2 typically uses
	// 2 slots — ViewData + LightingData — so 4 is comfortable headroom).
	// glGetIntegeri_v is the ES3 indexed-getter; available since GLES 3.0
	// (shared context bumped to ES3 in Phase 2.A).
	GLint ubo_base = 0;
	glGetIntegerv(GL_UNIFORM_BUFFER_BINDING, &ubo_base);
	GLint ubo[4] = {0, 0, 0, 0};
	for (int i = 0; i < 4; i++) {
		glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, i, &ubo[i]);
	}

	// 2. Sampler-unit-0. Activate unit 0 transiently to query its sampler
	// binding (the only unit Ganesh-GL is documented to touch).
	glActiveTexture(GL_TEXTURE0);
	GLint sampler0 = 0;
	glGetIntegerv(GL_SAMPLER_BINDING, &sampler0);

	// 3. READ vs DRAW framebuffer. DRAW is already in nx_gl_state_snap_t
	// (field `fbo`); the probe reads BOTH so we can observe whether they
	// diverge during Skia frames.
	GLint read_fbo = 0, draw_fbo = 0;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo);

	// 4. Transform feedback binding + rasterizer-discard enable.
	GLint tf = 0;
	glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &tf);
	GLboolean rast_disc = glIsEnabled(GL_RASTERIZER_DISCARD);

	// Restore active tex unit.
	glActiveTexture((GLenum)saved_active_tex);

	fprintf(stderr,
	        "[webgl-bridge:probe] tag=%s frame=%llu "
	        "ubo_base=%d ubo0=%d ubo1=%d ubo2=%d ubo3=%d "
	        "sampler0=%d read_fbo=%d draw_fbo=%d "
	        "tf=%d rast_disc=%d\n",
	        tag ? tag : "(null)",
	        (unsigned long long)s_probe_compose_n,
	        ubo_base, ubo[0], ubo[1], ubo[2], ubo[3],
	        sampler0, read_fbo, draw_fbo,
	        tf, (int)rast_disc);
	fflush(stderr);
}

// ---------------------------------------------------------------------------
// #16-ACTIVE state-leak probe (one-shot per launch)
// ---------------------------------------------------------------------------

void nx_webgl_state_probe_active_enable(bool on) {
	s_state_probe_active_on = on;
	if (on) {
		fprintf(stderr,
		        "[webgl-bridge:probe-active] enabled (one-shot #16-ACTIVE — "
		        "SET on compose #1, READ_BACK on compose #2, then latched)\n");
		fflush(stderr);
	}
}

// Verdict per NXJS_PATCHES_NEEDED.md #16-ACTIVE:
//   post == set                  → Skia didn't touch     → "moot"
//   post == pre  && pre != set   → Skia mutated+restored → "NEEDS_SNAP_MUTATE_RESTORE"
//   post != set && post != pre   → Skia mutated + left   → "NEEDS_SNAP_LEAVE"
//   post == set && pre == set    → couldn't distinguish (pre matched our set) → "moot-preset-matches"
static const char *ap_verdict(GLint pre, GLint set, GLint post) {
	if (post == set) {
		return (pre == set) ? "moot-preset-matches" : "moot";
	}
	if (post == pre) return "NEEDS_SNAP_MUTATE_RESTORE";
	return "NEEDS_SNAP_LEAVE";
}

static void nx_active_probe_set(void) {
	(void)glGetError(); // drain any prior residual

	// ── (a) UBO slot 3 SET + slot 0 pre-snapshot ─────────────────────────
	glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 3, &s_ap_state.ubo3_pre);
	glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 0, &s_ap_state.ubo0_pre);
	glGenBuffers(1, &s_ap_state.ubo_probe_obj);
	glBindBuffer(GL_UNIFORM_BUFFER, s_ap_state.ubo_probe_obj);
	glBufferData(GL_UNIFORM_BUFFER, 256, nullptr, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 3, s_ap_state.ubo_probe_obj, 0, 256);
	glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 3, &s_ap_state.ubo3_set);

	// ── (b) Sampler unit 0 SET ───────────────────────────────────────────
	GLint saved_active = GL_TEXTURE0;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &saved_active);
	glActiveTexture(GL_TEXTURE0);
	glGetIntegerv(GL_SAMPLER_BINDING, &s_ap_state.samp0_pre);
	glGenSamplers(1, &s_ap_state.samp_probe_obj);
	glSamplerParameteri(s_ap_state.samp_probe_obj, GL_TEXTURE_MIN_FILTER,
	                    GL_NEAREST);
	glBindSampler(0, s_ap_state.samp_probe_obj);
	glGetIntegerv(GL_SAMPLER_BINDING, &s_ap_state.samp0_set);
	glActiveTexture((GLenum)saved_active);

	// ── (c) READ_FRAMEBUFFER SET ─────────────────────────────────────────
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s_ap_state.read_fbo_pre);
	glGenFramebuffers(1, &s_ap_state.rfb_probe_obj);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, s_ap_state.rfb_probe_obj);
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s_ap_state.read_fbo_set);

	// ── (d) Transform feedback + rasterizer-discard SET ──────────────────
	glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &s_ap_state.tf_pre);
	s_ap_state.rd_pre = glIsEnabled(GL_RASTERIZER_DISCARD);
	glGenTransformFeedbacks(1, &s_ap_state.tf_probe_obj);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, s_ap_state.tf_probe_obj);
	glEnable(GL_RASTERIZER_DISCARD);
	glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &s_ap_state.tf_set);
	s_ap_state.rd_set = glIsEnabled(GL_RASTERIZER_DISCARD);

	GLenum set_err = glGetError();
	fprintf(stderr,
	        "[webgl-bridge:probe-active] SET complete: "
	        "ubo3(pre=%d,set=%d) ubo0(pre=%d) samp0(pre=%d,set=%d) "
	        "read_fbo(pre=%d,set=%d) tf(pre=%d,set=%d) "
	        "rd(pre=%d,set=%d) err=0x%x\n",
	        s_ap_state.ubo3_pre, s_ap_state.ubo3_set,
	        s_ap_state.ubo0_pre,
	        s_ap_state.samp0_pre, s_ap_state.samp0_set,
	        s_ap_state.read_fbo_pre, s_ap_state.read_fbo_set,
	        s_ap_state.tf_pre, s_ap_state.tf_set,
	        (int)s_ap_state.rd_pre, (int)s_ap_state.rd_set, set_err);
	fflush(stderr);
}

static void nx_active_probe_read_back(void) {
	GLint saved_active = GL_TEXTURE0;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &saved_active);
	glActiveTexture(GL_TEXTURE0);

	GLint ubo3_post = 0, ubo0_post = 0, samp0_post = 0;
	GLint read_fbo_post = 0, tf_post = 0;
	glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 3, &ubo3_post);
	glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 0, &ubo0_post);
	glGetIntegerv(GL_SAMPLER_BINDING, &samp0_post);
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo_post);
	glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &tf_post);
	GLboolean rd_post = glIsEnabled(GL_RASTERIZER_DISCARD);

	// Per-candidate verdict lines
	fprintf(stderr,
	        "[webgl-bridge:probe-active] candidate=ubo_slot3 pre=%d set=%d "
	        "post=%d verdict=%s\n",
	        s_ap_state.ubo3_pre, s_ap_state.ubo3_set, ubo3_post,
	        ap_verdict(s_ap_state.ubo3_pre, s_ap_state.ubo3_set, ubo3_post));
	fprintf(stderr,
	        "[webgl-bridge:probe-active] candidate=ubo_slot0 pre=%d post=%d "
	        "verdict=%s (passive-mode; pre==post means Ganesh preserved)\n",
	        s_ap_state.ubo0_pre, ubo0_post,
	        (ubo0_post == s_ap_state.ubo0_pre ? "moot-passive"
	                                          : "MUTATED_BY_SKIA"));
	fprintf(stderr,
	        "[webgl-bridge:probe-active] candidate=sampler_unit0 pre=%d "
	        "set=%d post=%d verdict=%s\n",
	        s_ap_state.samp0_pre, s_ap_state.samp0_set, samp0_post,
	        ap_verdict(s_ap_state.samp0_pre, s_ap_state.samp0_set, samp0_post));
	fprintf(stderr,
	        "[webgl-bridge:probe-active] candidate=read_fbo pre=%d set=%d "
	        "post=%d verdict=%s\n",
	        s_ap_state.read_fbo_pre, s_ap_state.read_fbo_set, read_fbo_post,
	        ap_verdict(s_ap_state.read_fbo_pre, s_ap_state.read_fbo_set,
	                   read_fbo_post));
	fprintf(stderr,
	        "[webgl-bridge:probe-active] candidate=tf pre=%d set=%d post=%d "
	        "verdict=%s\n",
	        s_ap_state.tf_pre, s_ap_state.tf_set, tf_post,
	        ap_verdict(s_ap_state.tf_pre, s_ap_state.tf_set, tf_post));
	fprintf(stderr,
	        "[webgl-bridge:probe-active] candidate=rast_disc pre=%d set=%d "
	        "post=%d verdict=%s\n",
	        (int)s_ap_state.rd_pre, (int)s_ap_state.rd_set, (int)rd_post,
	        ap_verdict((GLint)s_ap_state.rd_pre, (GLint)s_ap_state.rd_set,
	                   (GLint)rd_post));

	// SUMMARY line — feeds #17 snap-extension decisions.
	bool need_ubo3 = (ubo3_post != s_ap_state.ubo3_set);
	bool need_ubo0 = (ubo0_post != s_ap_state.ubo0_pre);
	bool need_samp0 = (samp0_post != s_ap_state.samp0_set);
	bool need_read_fbo = (read_fbo_post != s_ap_state.read_fbo_set);
	bool need_tf = (tf_post != s_ap_state.tf_set);
	bool need_rd = ((int)rd_post != (int)s_ap_state.rd_set);
	bool any = need_ubo3 || need_ubo0 || need_samp0 || need_read_fbo ||
	           need_tf || need_rd;
	fprintf(stderr, "[webgl-bridge:probe-active] SUMMARY needs_snap={");
	bool first = true;
#define AP_APPEND(cond, name) \
	if (cond) { fprintf(stderr, "%s%s", first ? "" : ",", name); first = false; }
	AP_APPEND(need_ubo3, "ubo_slot3")
	AP_APPEND(need_ubo0, "ubo_slot0")
	AP_APPEND(need_samp0, "sampler_unit0")
	AP_APPEND(need_read_fbo, "read_fbo")
	AP_APPEND(need_tf, "tf")
	AP_APPEND(need_rd, "rast_disc")
#undef AP_APPEND
	fprintf(stderr, "}%s\n", any ? "" : " (all-moot)");
	fflush(stderr);

	// ── RESTORE ──────────────────────────────────────────────────────────
	// Bind back to whatever pre was. pre==0 (unbound) is handled naturally
	// by the GL binds (buffer=0 unbinds). For UBO we use bindBufferBase
	// rather than bindBufferRange to avoid the offset+size we didn't save.
	glBindBufferBase(GL_UNIFORM_BUFFER, 3, (GLuint)s_ap_state.ubo3_pre);
	glBindSampler(0, (GLuint)s_ap_state.samp0_pre);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)s_ap_state.read_fbo_pre);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,
	                        (GLuint)s_ap_state.tf_pre);
	if (s_ap_state.rd_pre)
		glEnable(GL_RASTERIZER_DISCARD);
	else
		glDisable(GL_RASTERIZER_DISCARD);
	glActiveTexture((GLenum)saved_active);

	// ── CLEANUP ──────────────────────────────────────────────────────────
	glDeleteBuffers(1, &s_ap_state.ubo_probe_obj);
	glDeleteSamplers(1, &s_ap_state.samp_probe_obj);
	glDeleteFramebuffers(1, &s_ap_state.rfb_probe_obj);
	glDeleteTransformFeedbacks(1, &s_ap_state.tf_probe_obj);

	GLenum cleanup_err = glGetError();
	if (cleanup_err) {
		fprintf(stderr,
		        "[webgl-bridge:probe-active] restore/cleanup err=0x%x "
		        "(non-fatal; probe done)\n",
		        cleanup_err);
		fflush(stderr);
	}
}

// ---------------------------------------------------------------------------
// State save/restore primitive (public — re-used by 2.C+ WebGL bridge).
// ---------------------------------------------------------------------------

void nx_gl_state_save(nx_gl_state_snap_t *s) {
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s->fbo);
	glGetIntegerv(GL_VIEWPORT, s->viewport);
	glGetIntegerv(GL_CURRENT_PROGRAM, &s->program);
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s->vao);
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &s->array_buffer);
	glGetIntegerv(GL_ACTIVE_TEXTURE, &s->active_tex);
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &s->tex2d_binding);
	s->blend        = glIsEnabled(GL_BLEND);
	s->depth_test   = glIsEnabled(GL_DEPTH_TEST);
	s->cull         = glIsEnabled(GL_CULL_FACE);
	s->scissor      = glIsEnabled(GL_SCISSOR_TEST);
	s->stencil_test = glIsEnabled(GL_STENCIL_TEST);
	glGetBooleanv(GL_COLOR_WRITEMASK, s->color_mask);
	glGetBooleanv(GL_DEPTH_WRITEMASK, &s->depth_mask);   // cut #15
	glGetIntegerv(GL_STENCIL_WRITEMASK, &s->stencil_mask); // cut #15
	glGetFloatv(GL_COLOR_CLEAR_VALUE, s->clear_color);
	glGetIntegerv(GL_BLEND_SRC_RGB, &s->blend_src_rgb);
	glGetIntegerv(GL_BLEND_DST_RGB, &s->blend_dst_rgb);
	glGetIntegerv(GL_BLEND_SRC_ALPHA, &s->blend_src_a);
	glGetIntegerv(GL_BLEND_DST_ALPHA, &s->blend_dst_a);
	// Patch #17 (from #16-ACTIVE hardware verdict). Sampler binding is per-
	// texture-unit; we save unit 0 explicitly since that's the only unit
	// Ganesh is documented to touch and the probe confirmed on hardware.
	// Save-then-active-flip pattern: capture the caller's active unit
	// (already saved above into s->active_tex), transiently switch to unit
	// 0 to read its sampler binding, then leave the active unit as it was
	// (glGetIntegerv doesn't mutate it, but glActiveTexture does — so we
	// re-flip back).
	glActiveTexture(GL_TEXTURE0);
	glGetIntegerv(GL_SAMPLER_BINDING, &s->sampler_unit0);
	glActiveTexture((GLenum)s->active_tex);
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s->read_fbo);
}

void nx_gl_state_restore(const nx_gl_state_snap_t *s) {
	// Patch #17: restore read_fbo BEFORE the draw_fbo bind. glBindFramebuffer
	// with target=GL_FRAMEBUFFER (below) binds BOTH read AND draw to the
	// draw_fbo value — which is wrong if read_fbo diverged. So bind read
	// first (via GL_READ_FRAMEBUFFER target, which touches only read), then
	// the GL_FRAMEBUFFER call binds draw to the correct value AND overwrites
	// read to draw. If read_fbo != draw_fbo, re-bind read separately AFTER
	// the FRAMEBUFFER call.
	glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)s->fbo);
	if (s->read_fbo != s->fbo) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)s->read_fbo);
	}
	glViewport(s->viewport[0], s->viewport[1], s->viewport[2], s->viewport[3]);
	glUseProgram((GLuint)s->program);
	glBindVertexArray((GLuint)s->vao);
	glBindBuffer(GL_ARRAY_BUFFER, (GLuint)s->array_buffer);
	// Patch #17: restore sampler binding on unit 0. Active-flip pattern
	// mirrored from nx_gl_state_save. Bind BEFORE the final active-tex
	// restore so the caller's active unit ends up as it was saved.
	glActiveTexture(GL_TEXTURE0);
	glBindSampler(0, (GLuint)s->sampler_unit0);
	glActiveTexture((GLenum)s->active_tex);
	glBindTexture(GL_TEXTURE_2D, (GLuint)s->tex2d_binding);
	if (s->blend)        glEnable(GL_BLEND);        else glDisable(GL_BLEND);
	if (s->depth_test)   glEnable(GL_DEPTH_TEST);   else glDisable(GL_DEPTH_TEST);
	if (s->cull)         glEnable(GL_CULL_FACE);    else glDisable(GL_CULL_FACE);
	if (s->scissor)      glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
	if (s->stencil_test) glEnable(GL_STENCIL_TEST); else glDisable(GL_STENCIL_TEST);
	glColorMask(s->color_mask[0], s->color_mask[1], s->color_mask[2],
	            s->color_mask[3]);
	glDepthMask(s->depth_mask);              // cut #15
	glStencilMask((GLuint)s->stencil_mask);  // cut #15
	glClearColor(s->clear_color[0], s->clear_color[1], s->clear_color[2],
	             s->clear_color[3]);
	glBlendFuncSeparate(s->blend_src_rgb, s->blend_dst_rgb, s->blend_src_a,
	                    s->blend_dst_a);
}

// ---------------------------------------------------------------------------
// Lifetime + 2.B test compose path.
// ---------------------------------------------------------------------------

bool nx_webgl_bridge_init(int fbo_w, int fbo_h) {
	if (s_initialized) return true;
	if (!nx_skia_gpu_egl_context() || !nx_skia_gpu_gr_context()) {
		fprintf(stderr, "[webgl-bridge] init refused: skia_gpu not ready\n");
		fflush(stderr);
		return false;
	}
	if (fbo_w <= 0 || fbo_h <= 0) {
		fprintf(stderr, "[webgl-bridge] init refused: bad fbo size %dx%d\n",
		        fbo_w, fbo_h);
		fflush(stderr);
		return false;
	}
	s_fbo_w = fbo_w;
	s_fbo_h = fbo_h;

	// We're being called from the engine init path — the shared context is
	// already current per skia_gpu's eglMakeCurrent. Save Skia's pre-init GL
	// state so our FBO creation can't leave anything mis-bound when it
	// returns to the engine's render loop.
	nx_gl_state_snap_t snap;
	nx_gl_state_save(&snap);

	if (!create_fbo(fbo_w, fbo_h)) {
		nx_gl_state_restore(&snap);
		nx_skia_gpu_gr_context()->resetContext();
		return false;
	}
	if (!create_test_program()) {
		// FBO created but program failed — clean it up so we don't leak.
		if (s_fbo) { glDeleteFramebuffers(1, &s_fbo); s_fbo = 0; }
		if (s_color_tex) { glDeleteTextures(1, &s_color_tex); s_color_tex = 0; }
		if (s_depth_rb) { glDeleteRenderbuffers(1, &s_depth_rb); s_depth_rb = 0; }
		nx_gl_state_restore(&snap);
		nx_skia_gpu_gr_context()->resetContext();
		return false;
	}

	wrap_compose_image();
	if (!s_compose_image) {
		// Composite wrapper failed — still safe to render INTO the FBO, but
		// 2.B's smoke gate would have nothing to show. Treat as init failure.
		glDeleteFramebuffers(1, &s_fbo); s_fbo = 0;
		glDeleteTextures(1, &s_color_tex); s_color_tex = 0;
		glDeleteRenderbuffers(1, &s_depth_rb); s_depth_rb = 0;
		glDeleteProgram(s_test_prog); s_test_prog = 0;
		glDeleteBuffers(1, &s_test_vbo); s_test_vbo = 0;
		glDeleteVertexArrays(1, &s_test_vao); s_test_vao = 0;
		nx_gl_state_restore(&snap);
		nx_skia_gpu_gr_context()->resetContext();
		return false;
	}

	nx_gl_state_restore(&snap);
	nx_skia_gpu_gr_context()->resetContext();

	s_t_start_ns = ts_ns();
	s_frame_count = 0;
	s_boundary_us_accum = 0.0;
	s_probe_compose_n = 0;
	s_initialized = true;

	fprintf(stderr,
	        "[webgl-bridge] init ok fbo=%dx%d color_tex=%u depth_rb=%u fbo_id=%u\n",
	        fbo_w, fbo_h, s_color_tex, s_depth_rb, s_fbo);
	fflush(stderr);

	// Phase 2.G.0 baseline probe — fires once at init end so the log has
	// the GL state immediately after Skia has initialized + before any
	// WebGL traffic. Read-only.
	if (s_state_probe_on) nx_webgl_state_probe_log("init");

	return true;
}

void nx_webgl_bridge_exit(void) {
	if (!s_initialized) return;
	// Phase 2.G.0 exit-time probe — fires once before teardown so the log
	// captures end-of-session GL state. Read-only.
	if (s_state_probe_on) nx_webgl_state_probe_log("exit");
	// Drop the Skia-side SkImage FIRST so the GrDirectContext stops
	// referencing the tenant color texture, THEN free the GL handles.
	s_compose_image.reset();
	// Free all per-canvas extra tenants (multi-WebGL-canvas independence).
	for (auto &kv : s_extra_tenants) destroy_extra_tenant(kv.second);
	s_extra_tenants.clear();
	if (s_test_vao)  { glDeleteVertexArrays(1, &s_test_vao); s_test_vao = 0; }
	if (s_test_vbo)  { glDeleteBuffers(1, &s_test_vbo); s_test_vbo = 0; }
	if (s_test_prog) { glDeleteProgram(s_test_prog); s_test_prog = 0; }
	if (s_fbo)       { glDeleteFramebuffers(1, &s_fbo); s_fbo = 0; }
	if (s_color_tex) { glDeleteTextures(1, &s_color_tex); s_color_tex = 0; }
	if (s_depth_rb)  { glDeleteRenderbuffers(1, &s_depth_rb); s_depth_rb = 0; }
	s_initialized = false;
	s_fbo_dirty = false;
	s_webgl_owned = false;
	fprintf(stderr, "[webgl-bridge] exit ok frames=%llu avg_boundary=%.2fus\n",
	        (unsigned long long)s_frame_count,
	        s_frame_count > 0 ? s_boundary_us_accum / (double)s_frame_count
	                          : 0.0);
	fflush(stderr);
}

bool nx_webgl_bridge_is_initialized(void) { return s_initialized; }

GLuint nx_webgl_bridge_fbo_id(void) {
	return s_initialized ? s_fbo : 0;
}

void nx_webgl_bridge_fbo_size(int *out_w, int *out_h) {
	if (out_w) *out_w = s_initialized ? s_fbo_w : 0;
	if (out_h) *out_h = s_initialized ? s_fbo_h : 0;
}

void nx_webgl_bridge_mark_fbo_dirty(void) {
	if (s_initialized) s_fbo_dirty = true;
}

void nx_webgl_bridge_set_webgl_owned(bool owned) {
	s_webgl_owned = owned;
	if (owned) {
		fprintf(stderr, "[webgl-bridge] owner=webgl (test driver yields)\n");
		fflush(stderr);
	}
}

bool nx_webgl_bridge_is_webgl_owned(void) { return s_webgl_owned; }

void nx_webgl_bridge_set_auto_flush(bool v) { s_auto_flush = v; }

// Phase 2.D: parameterized sub-rect blit of the tenant FBO onto a destination
// Skia surface. Lets the runtime (brewser-runtime canvas-runner) compose each
// inline <canvas>'s region of the shared FBO at its CSS layout slot — the
// engine-side half of gl.copyBridgeToCanvas. The runtime expresses src
// coordinates in canvas/top-down convention (matching the QuickJS-era
// nx_webgl_copy_bridge_to_canvas contract); we translate to the kBottomLeft
// SkImage's coord system so a request for "top-left box.w x box.h" reads
// the GL viewport's bottom-left output (where Three.js gl.viewport(0,0,w,h)
// actually writes). Does NOT touch s_fbo_dirty — the runtime drives its own
// dirty tracking on this path.
bool nx_webgl_bridge_compose_rect(SkSurface *target,
                                  int src_x, int src_y, int src_w, int src_h,
                                  int dst_x, int dst_y) {
	if (!s_initialized || !target) return false;
	if (src_w <= 0 || src_h <= 0) return false;
	SkCanvas *c = target->getCanvas();
	if (!c) return false;
	// Phase 2.G.1 cut #12 (2026-07-01) — refresh s_compose_image at every
	// call. The SkImage returned by SkImages::BorrowTextureFrom is treated
	// as immutable by Ganesh, which caches an internal representation on
	// first sample. Any subsequent draw with the 6-arg drawImageRect
	// (which uses kStrict_SrcRectConstraint) samples that cached snapshot
	// instead of the live GL texture. Auto-flush's 3-arg drawImageRect (no
	// src rect, no constraint) sampled the same cached view too — which is
	// why the working v2 demos so far (webgl2-ubo, webgl2-shaders-sky) all
	// used full-FBO canvases where "first draw" happened AFTER Three.js's
	// setup ran, capturing a mostly-valid frame; sub-rect demos like
	// instancing-dynamic captured the initial clear and never advanced.
	// Rewrapping is cheap (no GL work, just a new SkImage handle over the
	// same texture id). Also drop kStrict_SrcRectConstraint — kFast is
	// correct for pixel-aligned rect blits (no interpolation across
	// non-existent src pixels), and it lets Skia use a live texture-atlas
	// binding instead of a materialised sub-image.
	wrap_compose_image();
	if (!s_compose_image) return false;
	// Cut #13's paint strategy: 3-arg drawImageRect (whole image at an
	// offset dst) + clipRect. The 6-arg drawImageRect(src, dst)
	// overload with explicit src rect triggers a Ganesh caching path
	// that samples a frozen initial texture snapshot; the 3-arg
	// implicit-src overload samples the live texture correctly (proven
	// by the auto-flush path, which uses the same overload).
	// Y-flip: runtime passes GL coords (matching v1's
	// nx_webgl_egl_read_bridge_to_canvas_data contract). Skia's
	// drawImageRect src is in visual/top-down coords on a kBottomLeft
	// SkImage, so we convert with visual_sy = FBO_h - src_y - src_h.
	return compose_image_rect(target, s_compose_image, s_fbo_w, s_fbo_h,
	                          src_x, src_y, src_w, src_h, dst_x, dst_y);
}

// ---------------------------------------------------------------------------
// Per-canvas tenant registry — public API (multi-WebGL-canvas independence).
// tenant_id 0 = the legacy/default tenant (globals); >0 = s_extra_tenants.
// ---------------------------------------------------------------------------
bool nx_webgl_bridge_tenant_ensure(uint32_t id, int w, int h) {
	if (w <= 0 || h <= 0) return false;
	if (id == 0) {
		// Default tenant: created once by nx_webgl_bridge_init at the first
		// context's size and intentionally NOT resized here (matches the
		// pre-existing single-tenant behavior — single-canvas apps unchanged).
		if (!s_initialized) return nx_webgl_bridge_init(w, h);
		return s_fbo != 0;
	}
	if (!s_initialized) return false; // shared ES3 context/bridge must be up
	ExtraTenant &t = s_extra_tenants[id];
	if (t.fbo && t.w == w && t.h == h) return true;   // already sized
	if (t.fbo) destroy_extra_tenant(t);               // resize -> rebuild
	if (!create_fbo_resources(w, h, &t.fbo, &t.color_tex, &t.depth_rb)) {
		s_extra_tenants.erase(id);
		return false;
	}
	t.w = w;
	t.h = h;
	t.dirty = false;
	t.image = borrow_fbo_image(t.color_tex, w, h);
	fprintf(stderr, "[webgl-bridge] tenant %u created %dx%d fbo=%u\n", id, w, h,
	        t.fbo);
	fflush(stderr);
	return t.fbo != 0;
}

GLuint nx_webgl_bridge_tenant_fbo(uint32_t id) {
	if (id == 0) return s_initialized ? s_fbo : 0;
	auto it = s_extra_tenants.find(id);
	return (it != s_extra_tenants.end()) ? it->second.fbo : 0;
}

void nx_webgl_bridge_tenant_size(uint32_t id, int *out_w, int *out_h) {
	int w = 0, h = 0;
	if (id == 0) {
		if (s_initialized) { w = s_fbo_w; h = s_fbo_h; }
	} else {
		auto it = s_extra_tenants.find(id);
		if (it != s_extra_tenants.end()) { w = it->second.w; h = it->second.h; }
	}
	if (out_w) *out_w = w;
	if (out_h) *out_h = h;
}

void nx_webgl_bridge_tenant_mark_dirty(uint32_t id) {
	if (id == 0) {
		if (s_initialized) s_fbo_dirty = true;
		return;
	}
	auto it = s_extra_tenants.find(id);
	if (it != s_extra_tenants.end()) it->second.dirty = true;
}

bool nx_webgl_bridge_tenant_compose_rect(SkSurface *target, uint32_t id,
                                         int src_x, int src_y, int src_w,
                                         int src_h, int dst_x, int dst_y) {
	if (id == 0) {
		return nx_webgl_bridge_compose_rect(target, src_x, src_y, src_w, src_h,
		                                    dst_x, dst_y);
	}
	auto it = s_extra_tenants.find(id);
	if (it == s_extra_tenants.end()) return false;
	ExtraTenant &t = it->second;
	// Re-borrow each call (same Ganesh frozen-snapshot reason as cut #12).
	t.image = borrow_fbo_image(t.color_tex, t.w, t.h);
	return compose_image_rect(target, t.image, t.w, t.h, src_x, src_y, src_w,
	                          src_h, dst_x, dst_y);
}

void nx_webgl_bridge_tenant_destroy(uint32_t id) {
	if (id == 0) return; // default tenant is freed by nx_webgl_bridge_exit
	auto it = s_extra_tenants.find(id);
	if (it != s_extra_tenants.end()) {
		destroy_extra_tenant(it->second);
		s_extra_tenants.erase(it);
		fprintf(stderr, "[webgl-bridge] tenant %u destroyed\n", id);
		fflush(stderr);
	}
}

// Production WebGL → Skia compose path. Cheap no-op on idle frames (no dirty,
// no compose call). Mirrors compose_test but without the test triangle / overlay
// chrome — the WebGL context drives the FBO contents itself.
//
// Gated by s_auto_flush: when the runtime has called gl.setBridgeAutoFlush(false)
// (canvas-runner.ts:302-309 does this in getSharedScreenGL after enabling the
// bridge), the engine MUST NOT compose the FBO to Skia — the runtime drives
// per-canvas readback into each <canvas>'s OffscreenCanvas itself and Skia
// paints the offscreens at the canvases' CSS layout slots. Auto-flushing on
// top of that overwrites the correctly-placed DOM with the whole 1280×720
// FBO at screen origin — the "demos render at screen bottom-left with shell
// stomped" symptom that 2.D fixes. Gate placed BEFORE the dirty check so a
// dirty FBO under auto-flush=false stays unflushed (the runtime's readback
// path consumes it instead).
void nx_webgl_bridge_compose(SkSurface *target) {
	// Phase 2.G.0 — count every compose-path call (independent of auto_flush
	// / dirty gates) so the probe sampling tracks the real per-frame
	// boundary. The compose function IS called once per frame by main.cc's
	// present hook regardless of dirty/auto_flush gates below.
	s_probe_compose_n++;
	if (s_state_probe_on && probe_sample_this_frame(s_probe_compose_n)) {
		nx_webgl_state_probe_log("compose-pre");
	}

	// #16-ACTIVE one-shot state-leak probe. Runs before the auto_flush /
	// dirty / target checks so it fires regardless of whether the compose
	// will actually do anything this frame. SET on compose #1, READ_BACK on
	// compose #2 → the between-composes gap is one full Skia scene frame's
	// worth of GL traffic, exposing bindings Ganesh-GL leaks or restores.
	if (s_state_probe_active_on && !s_active_probe_done) {
		if (s_probe_compose_n == 1) {
			nx_active_probe_set();
		} else if (s_probe_compose_n == 2) {
			nx_active_probe_read_back();
			s_active_probe_done = true;
		}
	}

	if (!s_auto_flush) {
		if (s_state_probe_on && probe_sample_this_frame(s_probe_compose_n)) {
			nx_webgl_state_probe_log("compose-post-noflush");
		}
		return;
	}
	if (!s_initialized || !target || !s_compose_image) {
		if (s_state_probe_on && probe_sample_this_frame(s_probe_compose_n)) {
			nx_webgl_state_probe_log("compose-post-skip");
		}
		return;
	}
	if (!s_fbo_dirty) {
		if (s_state_probe_on && probe_sample_this_frame(s_probe_compose_n)) {
			nx_webgl_state_probe_log("compose-post-clean");
		}
		return;
	}

	SkCanvas *c = target->getCanvas();
	if (!c) return;
	const int target_w = c->imageInfo().width();
	const int target_h = c->imageInfo().height();
	const float dx = (float)((target_w - s_fbo_w) / 2);
	const float dy = (float)((target_h - s_fbo_h) / 2);
	SkPaint paint;
	c->drawImageRect(s_compose_image.get(),
	                 SkRect::MakeXYWH(dx, dy, (float)s_fbo_w, (float)s_fbo_h),
	                 SkSamplingOptions(), &paint);
	s_fbo_dirty = false;
	if (s_state_probe_on && probe_sample_this_frame(s_probe_compose_n)) {
		nx_webgl_state_probe_log("compose-post");
	}
}

void nx_webgl_bridge_compose_test(SkSurface *target) {
	if (!s_initialized || !target || !s_compose_image) return;

	// Phase 2.C: when a WebGL context owns the FBO, the test driver yields its
	// "render triangle into FBO" step (WebGL is responsible for contents) but
	// still allows the compose path to run via nx_webgl_bridge_compose, called
	// from main.cc's present hook. So this function is effectively a no-op when
	// WebGL-owned — the WebGL bridge has its own compose call site.
	if (s_webgl_owned) return;

	const double t = (double)(ts_ns() - s_t_start_ns) / 1.0e9;

	// ===== Bracketed raw-GL section (the load-bearing primitive 2.C wraps) =====
	const uint64_t b0 = ts_ns();
	nx_gl_state_snap_t snap;
	nx_gl_state_save(&snap);

	render_test_into_fbo((float)t);
	s_fbo_dirty = true;  // test driver touched the FBO; let compose run

	nx_gl_state_restore(&snap);
	GrDirectContext *gr = nx_skia_gpu_gr_context();
	if (gr) gr->resetContext();
	const uint64_t b1 = ts_ns();
	const double boundary_us = (double)(b1 - b0) / 1000.0;
	s_boundary_us_accum += boundary_us;
	s_frame_count++;

	// ===== Skia compose phase =====
	SkCanvas *c = target->getCanvas();
	if (!c) return;
	// Target canvas is the persistent screen canvas — DO NOT clear it; we
	// only paint additively on top of whatever the shell drew so the smoke
	// test doesn't wipe shell content (coexistence proof requires SHELL
	// drawing to remain visible alongside our composite).

	// Composite the tenant FBO color texture at a known rect. Centered-ish
	// on a 1280×720 surface (the engine's default). 320,180 + 640,360 lands
	// it in the middle with a margin so the shell underneath is visible at
	// the edges.
	const int target_w = c->imageInfo().width();
	const int target_h = c->imageInfo().height();
	const float dx = (float)((target_w - s_fbo_w) / 2);
	const float dy = (float)((target_h - s_fbo_h) / 2);

	SkPaint compose_paint;
	c->drawImageRect(s_compose_image.get(),
	                 SkRect::MakeXYWH(dx, dy, (float)s_fbo_w, (float)s_fbo_h),
	                 SkSamplingOptions(), &compose_paint);

	// Yellow border around the FBO region so its bounds are unambiguous in a
	// screenshot.
	SkPaint border;
	border.setStyle(SkPaint::kStroke_Style);
	border.setStrokeWidth(3);
	border.setColor(SK_ColorYELLOW);
	c->drawRect(SkRect::MakeXYWH(dx - 2.f, dy - 2.f,
	                             (float)s_fbo_w + 4.f, (float)s_fbo_h + 4.f),
	            border);

	// 2D overlay BANNER on top of the composite. Drawn AFTER drawImageRect
	// so it lands above the FBO contents — proves draw order survives the
	// WebGL→Skia handoff.
	SkPaint banner;
	banner.setColor(SkColorSetARGB(180, 0, 200, 100));
	c->drawRect(SkRect::MakeXYWH(40, 40, 700, 64), banner);
	SkPaint banner_border;
	banner_border.setStyle(SkPaint::kStroke_Style);
	banner_border.setStrokeWidth(2);
	banner_border.setColor(SK_ColorYELLOW);
	c->drawRect(SkRect::MakeXYWH(40, 40, 700, 64), banner_border);

	SkFont font;
	font.setSize(22);
	SkPaint text;
	text.setColor(SK_ColorWHITE);
	c->drawString("Phase 2.B coexistence: WebGL FBO + Skia overlay",
	              56, 80, font, text);

	char hud[160];
	snprintf(hud, sizeof(hud),
	         "frame %llu  boundary %.1fus avg",
	         (unsigned long long)s_frame_count,
	         s_boundary_us_accum / (double)s_frame_count);
	SkPaint hud_text;
	hud_text.setColor(SK_ColorWHITE);
	c->drawString(hud, 56, target_h - 32.f, font, hud_text);

	// Drop a `[webgl-bridge]` line periodically so the log carries proof the
	// bracket ran (and the boundary-cost number) for the smoke gate.
	if (s_frame_count == 1 || s_frame_count == 60 || s_frame_count == 600 ||
	    (s_frame_count % 3600) == 0) {
		fprintf(stderr,
		        "[webgl-bridge] frame=%llu boundary_us=%.2f avg_boundary_us=%.2f\n",
		        (unsigned long long)s_frame_count, boundary_us,
		        s_boundary_us_accum / (double)s_frame_count);
		fflush(stderr);
	}
}
