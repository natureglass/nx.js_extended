#pragma once
//
// V8 migration Step 2 — WebGL ↔ Skia coexistence primitive.
//
// Phase 2.B realizes the Phase 0 fbo-spike recipe as proper engine code on top
// of Phase 2.A's shared ES3 context (Skia owns EGL; bridge attaches via
// nx_skia_gpu_egl_* accessors). The bridge does NOT own EGL and does NOT
// create JS-visible WebGL classes — that's Phase 2.C+. What it owns:
//
//   1. A tenant offscreen FBO (color texture + depth/stencil renderbuffer)
//      created on Skia's shared ES3 context.
//   2. The load-bearing per-frame BOUNDARY discipline that any future WebGL
//      pass MUST wrap itself in to coexist with Skia:
//
//         nx_gl_state_save(&snap);
//         /* ...raw GL into the tenant FBO... */
//         nx_gl_state_restore(&snap);
//         nx_skia_gpu_gr_context()->resetContext();
//
//      Skipping the resetContext() leaves Ganesh believing its cached
//      program/VAO/etc. are still current — next Skia draw renders garbage.
//      Skipping any element of the save/restore set leaves Skia operating on
//      mis-bound state — symptom is intermittent / shape-dependent visual
//      corruption, not a clean crash. The state list mirrors the spike's
//      contract; see [[v8-migration-phase0-go]] for provenance.
//   3. A 2.B-only test compose path that drives the FBO with hand-written GL
//      and SkImages::BorrowTextureFrom-composites the FBO color texture +
//      a 2D overlay into the passed Skia surface. Gated by the
//      `[webgl] test_fbo = true` config flag so the shell renders normally
//      when off. This is the smoke + hardware checkpoint for Phase 2.B,
//      NOT the production path; Phase 2.C+ replaces it with the WebGL
//      bridge proper.
//
// Lifetime: nx_webgl_bridge_init() must be called AFTER nx_skia_gpu_screen_init
// (the bridge depends on the shared GL context + GrDirectContext being live).
// nx_webgl_bridge_exit() must be called BEFORE nx_skia_gpu_screen_exit().
//

#include "skia_gpu.h"

#include <GLES3/gl3.h>

#include "include/core/SkSurface.h"

// State snapshot captured at the WebGL→Skia handoff. The set was empirically
// validated in the fbo-spike (4,013 frames on Citron). Each field is required
// for a specific reason:
//
//   fbo            — Skia binds different FBOs all the time; restoring the
//                    DRAW_FRAMEBUFFER_BINDING is essential or Skia next draws
//                    into our tenant FBO instead of its own surface target.
//   viewport       — same: Skia sets per-surface, tenant sets per-FBO.
//   program        — Ganesh's program cache + state cache assume the bound
//                    program is the one it set; leaving ours bound silently
//                    breaks the next Ganesh draw.
//   vao            — likewise; Ganesh tracks bound VAO for vertex-attrib state.
//   array_buffer   — for the GL-pre-VAO-era path Ganesh sometimes hits.
//   active_tex     — bridges that change it (texture-unit selector) WILL
//                    corrupt Ganesh's per-unit binding state otherwise.
//   tex2d_binding  — texture-unit 0 binding, since Ganesh assumes it.
//   blend / depth_test / cull / scissor / stencil_test — Ganesh cares about
//                    every one of these; an enable left "wrong" silently
//                    drops fragments.
//   color_mask     — leaving a channel masked silently produces black/no-
//                    alpha output in subsequent Skia draws.
//   clear_color    — Ganesh sets per-clear; if we leave ours and Ganesh's
//                    next code path is "assume already correct" we'll
//                    glClear to the wrong color.
//   blend_src/dst  — full glBlendFuncSeparate state.
//
// Phase 2.G.1 cut #15 (2026-07-01) — depth_mask + stencil_mask ADDED. The
// original comment claimed "Ganesh resets these per-draw" — that IS true for
// Skia's own draws, but when Three.js runs BETWEEN Skia frames, Three.js's
// WebGLState cache assumes depth mask starts at TRUE (WebGL default) and
// short-circuits gl.depthMask(TRUE) calls. If Ganesh left GL_DEPTH_WRITEMASK
// at FALSE (Skia's 2D drawing doesn't want depth writes), Three.js's cache
// is out of sync with the actual GL state; gl.clear(DEPTH_BUFFER_BIT)
// silently becomes a no-op; the depth buffer stays at its previous frame's
// values (or uninitialized 0); LESS depth test rejects every cube fragment.
// Symptom: draws succeed with no GL error, all state introspection reports
// clean, and yet no pixels land on the color texture. Cut #14h isolation
// test confirmed the mechanism (disabling GL_DEPTH_TEST made the same
// instanced draw produce cube pixels immediately). Instancing-dynamic is
// the first v2 demo to depend on per-frame depth clear (webgl2-ubo and
// webgl2-shaders-sky used depth_test=false custom shaders).
//
// NOT in the set (deliberately):
//   - DEPTH_FUNC — Ganesh's per-draw setup resets it, AND Three.js re-emits
//     gl.depthFunc calls per-material.
//   - PIXEL_PACK/UNPACK alignment — irrelevant for the compose path.
//   - polygon_offset / line_width / point_size — Ganesh sets these per-draw.
//
// Phase 2.G.1 patch #17 (2026-07-01) — sampler_unit0 + read_fbo ADDED per
// authoritative hardware verdict from the #16-ACTIVE probe (real Switch
// verdict, per NXJS_PATCHES_NEEDED.md #16-ACTIVE spec). SUMMARY line was:
//   needs_snap={sampler_unit0, read_fbo}
// Details:
//   sampler_unit0 → NEEDS_SNAP_LEAVE: Skia mutated it (post=2 != our set=1)
//                   and left it. Ganesh binds sampler object 2 to unit 0
//                   and expects it to persist across frames. Passive probe
//                   confirmed the steady-state value at every compose-pre
//                   is sampler0=2.
//   read_fbo      → NEEDS_SNAP_MUTATE_RESTORE: Skia unbound our probe RFB
//                   back to 0 (its expected default) — meaning Ganesh
//                   assumes GL_READ_FRAMEBUFFER_BINDING is 0 at start of
//                   its frames and re-sets it. The 2.B snap saved via
//                   GL_DRAW_FRAMEBUFFER_BINDING + restored via
//                   glBindFramebuffer(GL_FRAMEBUFFER, ...) — which binds
//                   BOTH targets — so this actually WAS covered for the
//                   Skia-mutates-and-restores case; but adding an explicit
//                   read_fbo save+restore path is defense in depth for the
//                   WebGL2-only demos (MRT + gpgpu-water) that Three.js
//                   split-binds READ vs DRAW.
//   ubo_slot3/ubo_slot0/tf/rast_disc → all moot per probe. Ganesh doesn't
//     touch UBO slots on this hardware; doesn't use transform feedback;
//     doesn't toggle rasterizer discard.
//
// If 2.C/2.D surfaces "Skia renders garbage after WebGL draws" symptoms on
// hardware, the FIRST place to look is this list — a missing entry under a
// new code path is the most likely cause.
struct nx_gl_state_snap_t {
	GLint fbo;
	GLint viewport[4];
	GLint program;
	GLint vao;
	GLint array_buffer;
	GLint active_tex;
	GLint tex2d_binding;
	GLboolean blend;
	GLboolean depth_test;
	GLboolean cull;
	GLboolean scissor;
	GLboolean stencil_test;
	GLboolean color_mask[4];
	GLboolean depth_mask;       // cut #15
	GLint stencil_mask;          // cut #15
	GLfloat clear_color[4];
	GLint blend_src_rgb;
	GLint blend_dst_rgb;
	GLint blend_src_a;
	GLint blend_dst_a;
	// Patch #17 (from #16-ACTIVE hw probe SUMMARY needs_snap={sampler_unit0,read_fbo}):
	GLint sampler_unit0;         // Skia binds sampler obj 2 to unit 0 + leaves it
	GLint read_fbo;              // Ganesh assumes READ_FRAMEBUFFER = 0 at frame start
};

// Capture / restore the GL state contract. The caller owns the snap struct.
// gr->resetContext() is NOT called here — the caller does it after restore
// so the choice (per-section vs batched) is explicit at the call site.
void nx_gl_state_save(nx_gl_state_snap_t *s);
void nx_gl_state_restore(const nx_gl_state_snap_t *s);

// Initialize the tenant FBO + (optionally) the 2.B test program / VAO on
// Skia's shared ES3 context. Must be called AFTER nx_skia_gpu_screen_init.
// Returns false if FBO/program creation failed (caller treats the bridge as
// unavailable and renders normally). `fbo_w`/`fbo_h` is the offscreen size —
// 640×360 for the 2.B smoke probe and the 2.C slice demo.
bool nx_webgl_bridge_init(int fbo_w, int fbo_h);

// Tear down the FBO + program. Idempotent. Must be called BEFORE
// nx_skia_gpu_screen_exit so the GL handles are still valid when freed.
void nx_webgl_bridge_exit(void);

// True between init success and exit.
bool nx_webgl_bridge_is_initialized(void);

// ---- Phase 2.C accessors (used by webgl.cc to render INTO the tenant FBO)
// The WebGL context binds this FBO as its "default framebuffer" — every WebGL
// draw lands in this offscreen target, then the engine present hook calls
// nx_webgl_bridge_compose() to blit it onto Skia's canvas surface. Returns 0
// when bridge isn't initialized.
GLuint nx_webgl_bridge_fbo_id(void);

// FBO dimensions (the WebGL "drawing buffer" w/h). Returns 0,0 when not init.
void nx_webgl_bridge_fbo_size(int *out_w, int *out_h);

// Mark the FBO as touched this frame. Called by webgl.cc whenever a draw /
// clear lands in the tenant FBO. The compose path checks this flag and skips
// the SkImage draw on frames the bridge wasn't touched (cheap no-op when JS
// didn't call into WebGL).
void nx_webgl_bridge_mark_fbo_dirty(void);

// Compose the tenant FBO color texture as an SkImage into `target` (Skia's
// persistent canvas surface). No-op when not initialized OR when the FBO
// hasn't been marked dirty since the last compose. Clears the dirty flag
// after composing. NO overlay drawn — this is the production WebGL→Skia
// compose path, not the 2.B test driver.
void nx_webgl_bridge_compose(SkSurface *target);

// Phase 2.B smoke driver — STILL AVAILABLE behind [webgl] test_fbo = true.
// Renders a hand-written animated triangle into the tenant FBO (state-save
// bracketed + resetContext), then composites the FBO color texture as an
// SkImage into `target` at the test rect plus a 2D overlay (banner + frame
// counter + boundary microseconds) on top. No-op when the bridge isn't
// initialized OR when a WebGL context is actively driving the FBO (the test
// driver yields to real WebGL traffic to avoid clobbering tenant content).
// Idempotent against being called once per frame.
void nx_webgl_bridge_compose_test(SkSurface *target);

// Signal that a WebGL context is now the owner of the tenant FBO — disables
// the 2.B test driver's GL render-into-FBO step (compose_test still composes
// what's there). Phase 2.C call site: nx_webgl_context_new. The flag is
// process-wide because there's one bridge + one tenant FBO.
void nx_webgl_bridge_set_webgl_owned(bool owned);
bool nx_webgl_bridge_is_webgl_owned(void);

// Phase 2.D: gate `nx_webgl_bridge_compose` so the runtime can suppress the
// engine's whole-FBO auto-stomp onto Skia's canvas surface. brewser-runtime's
// canvas-runner.ts drives its own per-canvas paint by calling the parameterized
// compose helper below at the canvas's CSS layout slot, and calls
// gl.setBridgeAutoFlush(false) at getSharedScreenGL time so the engine stops
// auto-stomping the whole FBO on top of those placements. Default is TRUE
// (preserves the 2.B test_fbo smoke path + any non-runtime caller that
// doesn't set it). The brewser runtime sets FALSE on context acquisition.
void nx_webgl_bridge_set_auto_flush(bool v);

// Phase 2.D: parameterized sub-rect blit of the tenant FBO onto `target` Skia
// surface. The engine-side half of `gl.copyBridgeToCanvas(srcX, srcY, srcW,
// srcH, dst_canvas, dstX, dstY)`. The runtime expresses src coords in
// canvas/top-down convention (matching the QuickJS-era contract); this helper
// translates to the kBottomLeft-origin SkImage's coords internally so a request
// for src (0, 0, w, h) reads where Three.js gl.viewport(0, 0, w, h) actually
// writes (GL bottom-left of the FBO). dst rect is at (dst_x, dst_y) with the
// same width/height as src. Caller (webgl.cc::w_copy_bridge_to_canvas) MUST
// exit any open per-frame WebGL bracket first so Skia draws against its own
// GL state, not the WebGL pass's. Returns false if the bridge isn't initialized
// or args are degenerate.
bool nx_webgl_bridge_compose_rect(SkSurface *target,
                                  int src_x, int src_y, int src_w, int src_h,
                                  int dst_x, int dst_y);

// ---- Per-canvas tenant registry (multi-WebGL-canvas independence) ----------
// Each page WebGL canvas owns its own tenant FBO so stacked canvases render
// independently (the tetr.io #pixi + #pixi-fg case). tenant_id 0 is the
// legacy/default tenant created by nx_webgl_bridge_init — single-canvas apps +
// the shell are byte-for-byte unchanged. id > 0 lives in an internal map, one
// per additional canvas.
//   ensure       — create (or, for id>0, resize) the tenant; false on failure.
//   fbo          — the tenant's GL framebuffer name (0 if absent). Bound as the
//                  canvas's "default framebuffer".
//   size         — the tenant's w/h (drives the compose Y-flip).
//   mark_dirty   — flag the tenant touched this frame.
//   compose_rect — blit a sub-rect of the tenant onto `target` at (dst_x,dst_y).
//   destroy      — free the tenant's GL handles + SkImage (id>0 only). GL MUST
//                  be current — call from a GL-current path, never a GC
//                  finalizer (webgl.cc defers finalizer frees to enter_bracket).
#include <stdint.h>
bool   nx_webgl_bridge_tenant_ensure(uint32_t id, int w, int h);
GLuint nx_webgl_bridge_tenant_fbo(uint32_t id);
void   nx_webgl_bridge_tenant_size(uint32_t id, int *out_w, int *out_h);
void   nx_webgl_bridge_tenant_mark_dirty(uint32_t id);
bool   nx_webgl_bridge_tenant_compose_rect(SkSurface *target, uint32_t id,
                                           int src_x, int src_y, int src_w,
                                           int src_h, int dst_x, int dst_y);
void   nx_webgl_bridge_tenant_destroy(uint32_t id);

// Phase 2.G.0 — state-contract probe controls + read-only logger.
//
// The probe is GATED by `[webgl] state_probe = true` (config.h::webgl_state_probe).
// Call nx_webgl_state_probe_enable(cfg->webgl_state_probe) once at engine boot
// after config load — main.cc does this just below the bridge init site. The
// flag is process-wide; the bridge consults it via accessor at each hook point.
//
// When enabled, nx_webgl_state_probe_log(tag) issues a small batch of GL
// queries for four bindings that the QuickJS-era impl serviced but the FROZEN
// 2.B nx_gl_state_snap_t deliberately does NOT cover:
//
//   1. UBO indexed bindings — GL_UNIFORM_BUFFER_BINDING + indexed slots
//      0..3 (sample of what Three.js v2 actually exercises; the full
//      MAX_UNIFORM_BUFFER_BINDINGS sweep is overkill for a leak probe)
//   2. Sampler-unit-0 binding — GL_SAMPLER_BINDING with active unit 0
//   3. READ_FRAMEBUFFER vs DRAW_FRAMEBUFFER — already saved DRAW; probe
//      READ to see whether Skia leaves them separable or always equal
//   4. TRANSFORM_FEEDBACK_BINDING + RASTERIZER_DISCARD
//
// The probe is READ-ONLY (only glGetIntegerv / glIsEnabled), so calling it
// from inside any GL-current path is safe (it does not perturb Skia's
// cached state, but as a defensive measure the bridge still wraps probe
// reads in a tiny bracket that records + restores GL_ACTIVE_TEXTURE — the
// only state we touch indirectly via the GL_SAMPLER_BINDING probe).
//
// Hook points (all auto-fire when probe enabled):
//   - tag "init"         — end of nx_webgl_bridge_init (baseline before any
//                          WebGL or composite traffic)
//   - tag "compose-pre"  — start of nx_webgl_bridge_compose, frames 1/60/600
//   - tag "compose-post" — end   of nx_webgl_bridge_compose, frames 1/60/600
//   - tag "exit"         — start of nx_webgl_bridge_exit
//
// Output: `[webgl-bridge:probe] tag=<tag> frame=<n> ubo0=X ubo1=X ubo2=X
// ubo3=X ubo_base=X sampler0=X read_fbo=X draw_fbo=X tf=X rast_disc=X`
// All values are GL handle ids / enables (0 = unbound / disabled).
void nx_webgl_state_probe_enable(bool on);
bool nx_webgl_state_probe_enabled(void);
void nx_webgl_state_probe_log(const char *tag);

// #16-ACTIVE state-leak probe. Runs ONCE per launch: SET four candidate
// bindings to known non-default values on the FIRST compose call, yield
// one Skia frame, READ back on the SECOND compose call, interpret and
// log verdict per candidate, RESTORE + CLEANUP. Read the SUMMARY line
// from `nxjs-debug.log` to drive #17 snap-extension decisions.
//
// Opt-in via [webgl] state_probe_active = true. Requires webgl_state_probe
// also true (reuses passive probe's log wiring). Idempotent — after the
// one-shot run, subsequent compose calls no-op.
void nx_webgl_state_probe_active_enable(bool on);
