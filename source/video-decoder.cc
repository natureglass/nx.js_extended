// Switch.VideoDecoder native bindings (Phase 2.G.1 cut #22).
//
// Minimal port of the QuickJS-era Switch.VideoDecoder API onto the V8 fork's
// portable nx_media_* pipeline. Unlike the (much larger) QuickJS video.c
// which owns its own decode threads, packet rings, and audrv voice
// bookkeeping, this file is a thin V8-binding wrapper over nx_media_open /
// nx_media_present — which the fork's Video class already uses.
//
// Scope for cut #22: enough to unblock webgl-materials-video (the only v2
// demo needing Switch.VideoDecoder). Specifically:
//   * synchronous open (blocks briefly on nx_media_open — fine for local
//     sdmc files),
//   * play(),
//   * nextFrame() returning `{data: ArrayBuffer, width, height, pts, ended}`
//     with BGRA→RGBA swizzle (nx_media outputs BGRA per media-decoder.h;
//     the demo's DataTexture uses THREE.RGBAFormat / UnsignedByteType),
//   * getters the demo reads: width, height, duration, error, ended,
//     paused (stub false — nx_media doesn't expose paused state getter),
//     usedVideo, usedAudio, usedHw (stub false — nx_media hides
//     hw/sw decision), muted/volume/audioTime/audioError (stubs — audio
//     integration deferred; the demo uses `noAudio: true` so this path
//     is exercised without an audio voice).
//
// Deferred (not needed by webgl-materials-video, ports if future demos
// require them):
//   * seek, pause, close-and-reopen cycles
//   * setMuted/setVolume + audio-graph attach
//   * getAudioLevels/getFrequencyData/getWaveform (audio-visualizer surface
//     — the QuickJS impl carries ~600 lines of audrv played-sample-count
//     wave-buffer bookkeeping; only useful once an audio path lands).

#include "video-decoder.h"
#include "audio.h"
#include "audio-graph.h"
#include "error.h"
#include "media-decoder.h"
#include "types.h"
#include "wrap.h"
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <thread>

using namespace v8;

namespace {

struct nx_video_decoder_t {
	// ---- async open (2026-09-05) ----------------------------------------
	// `nx_media_open` blocks on the network for http(s) sources
	// (avformat_open_input + avformat_find_stream_info fetch the HLS playlist
	// and probe segments), which — when run on the V8 main thread — froze the
	// whole shell for the several-second transcode ramp-up. So the open runs
	// on `open_thread` and the decoder returns immediately "pending":
	// `media` is null and stays null until the thread publishes it (release)
	// with `open_done` (acquire) as the synchronising barrier. Read `media`
	// ONLY via ready_media(); never touch it before `open_done`. On teardown
	// we JOIN the thread first (bounded by the ffmpeg rw_timeout added to the
	// network open), so media/audio_node destruction stays single-threaded
	// and race-free exactly as the synchronous path was.
	std::thread open_thread;
	std::string open_url;
	// Per-open HTTP request shaping (2026-09-17). Owned here (not borrowed
	// from the V8 strings, which are gone by the time open_thread runs) and
	// read only by that thread before it publishes `media`. Empty = use the
	// engine default UA / send no Referer. See nx_media_net_t.
	std::string open_user_agent;
	std::string open_referer;
	bool open_want_loop = false;
	// When true the media ring is opened in planar I420 mode (see
	// nx_media_open want_yuv): frame_buf holds 1.5 B/px, next_frame delivers
	// the I420 planes verbatim, and the JS side uploads them as a Skia YUVA
	// image. Set at construction from the brewser <video> path.
	bool want_yuv = false;
	std::atomic<bool> open_done{false};
	// Cancel latch handed to nx_media_open. Set before EVERY join of
	// open_thread: the open can be parked in a blocking connect/read, and
	// joining it without this stalls the V8 main thread for the whole
	// rw_timeout (30s). That freeze is what made leaving a page mid-load,
	// or switching channels while one was still connecting, lock the UI.
	// Declared before `media` so it outlives it during destruction.
	std::atomic<bool> open_abort{false};
	nx_media_t *media = nullptr; // published by open_thread; read via ready_media()
	char open_err[256] = {};
	// Raw AVERROR behind `open_err`, published under the same open_done
	// barrier. Exposed to JS as `errorKind` so a page can tell a permanent
	// rejection from a transient one instead of parsing `error` text.
	int open_av_err = 0;
	// Controls issued while still pending are remembered and applied on the
	// MAIN thread the first time next_frame() observes the media ready
	// (nx_media clock fields are main-thread-only, so the open thread must
	// NOT call play/seek itself).
	std::atomic<bool> play_requested{false};
	std::atomic<bool> seek_pending{false};
	std::atomic<double> seek_target{0};
	bool ready_applied = false; // main thread only
	// ---------------------------------------------------------------------
	// Stream-source node attached to the media (cut #22b, 2026-07-02).
	// Released strictly AFTER nx_media_destroy joins the decode thread, so
	// the producer can never touch a freed node. NULL when the source has
	// no audio track OR `noAudio: true` was passed at construction.
	nx_audio_node *audio_node = nullptr;
	uint8_t *frame_buf = nullptr; // BGRA/I420 — nx_media_present output
	// Reused delivery backing (2026-09-06): handing V8 a fresh ~1.4 MB external
	// ArrayBuffer backing every frame made its external-memory accounting fire
	// a periodic major GC — a video hitch every ~N frames (N ∝ 1/framebytes:
	// 480p ~1 s, 720p ~0.5 s). Audio runs on its own thread so it was
	// unaffected. Reuse ONE backing: copy the frame into it and wrap in a small
	// ArrayBuffer sharing it, so V8's external counter doesn't grow per frame.
	std::shared_ptr<v8::BackingStore> deliver_bs;
	size_t deliver_size = 0;
	// Persistent delivery ArrayBuffer OBJECT (2026-09-06 v2). Reusing only the
	// backing (above) did NOT stop the periodic major GC: the [vft] frame-timing
	// probe proved a mark-sweep-compact fires 2-3×/s (~7.5 ms each) during 720p
	// playback, straddling the 30 Hz (33 ms) present budget → the 24↔30 fps
	// bounce. Cause: V8 charges each NEW ArrayBuffer's byte length to its
	// array-buffer/external accounting via a fresh ArrayBufferExtension — EVEN
	// when the backing store is shared — so minting one ArrayBuffer per frame
	// still added ~1.4 MB/frame (~42 MB/s at 720p30) of old-gen GC pressure.
	// Hold ONE ArrayBuffer object across frames (memcpy into its backing each
	// frame, return the same object): the extension is created once, the
	// external counter stays flat, and the mark-compact cadence stops.
	v8::Global<v8::ArrayBuffer> deliver_ab;
	int width = 0;
	int height = 0;
	double duration = 0.0;
	bool eos_sent = false; // one-shot end-of-stream sentinel latch
	bool closed = false;
	// Cut #22b: mirrors nx_media's playing state at the JS boundary so
	// the `paused` getter can report it without adding an accessor to
	// media-decoder.h. Starts paused (nx_media_open leaves the clock
	// stopped until nx_media_play).
	bool paused = true;
	// Audio state (JS-side wraps in a GainNode too — the C state is the
	// source of truth for the `muted` / `volume` getters).
	bool muted = false;
	float volume = 1.0f;
};

// Media handle once the async open has published it, else nullptr. The
// `open_done` acquire load pairs with the open thread's release store so the
// `media` write is visible before we read it.
static nx_media_t *ready_media(nx_video_decoder_t *d) {
	if (!d || !d->open_done.load(std::memory_order_acquire)) return nullptr;
	return d->media;
}

// Apply controls that were requested while the open was still in flight.
// MAIN THREAD ONLY (called from next_frame). One-shot via `ready_applied`.
static void apply_pending_on_ready(nx_video_decoder_t *d, nx_media_t *m) {
	if (d->ready_applied) return;
	d->ready_applied = true;
	if (d->seek_pending.load(std::memory_order_relaxed))
		nx_media_seek(m, d->seek_target.load(std::memory_order_relaxed));
	if (d->play_requested.load(std::memory_order_relaxed))
		nx_media_play(m);
}

void free_video_decoder(nx_video_decoder_t *d) {
	// Join the open thread FIRST so it can't publish/destroy media under us.
	// Latch the abort before joining, or a still-connecting open blocks this
	// thread until rw_timeout expires.
	d->open_abort.store(true, std::memory_order_relaxed);
	if (d->open_thread.joinable()) d->open_thread.join();
	if (d->media) {
		nx_media_destroy(d->media); // joins decode thread first
		d->media = nullptr;
	}
	if (d->audio_node) {
		// Safe to release now — the producer thread is gone.
		nx_audio_node_release(d->audio_node);
		d->audio_node = nullptr;
	}
	free(d->frame_buf);
	d->frame_buf = nullptr;
	delete d;
}

nx_video_decoder_t *get_decoder(Isolate *iso, Local<Value> val) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(val);
	if (!d) nx_throw(iso, "expected VideoDecoder handle");
	return d;
}

// videoDecoderNew(url: string, opts?: object) → handle
void nx_video_decoder_new(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Context> ctx = iso->GetCurrentContext();
	if (info.Length() < 1 || !info[0]->IsString()) {
		nx_throw(iso, "videoDecoderNew(url, opts?) — url string required");
		return;
	}
	String::Utf8Value url_val(iso, info[0]);
	const char *url = *url_val ? *url_val : "";

	bool want_loop = false;
	bool want_yuv = false;
	std::string opt_user_agent;
	std::string opt_referer;
	if (info.Length() >= 2 && info[1]->IsObject()) {
		Local<Object> opts = info[1].As<Object>();
		Local<Value> loop_val;
		if (opts->Get(ctx, nx_str(iso, "loop")).ToLocal(&loop_val) &&
		    loop_val->IsBoolean()) {
			want_loop = loop_val->BooleanValue(iso);
		}
		// `yuv: true` (2026-09-06) opens the ring in planar I420 so the
		// caller can upload YUV planes to the GPU (Skia YUVA image) — ~2.6×
		// less per-frame texture upload than BGRA. brewser's <video> path
		// sets it; the Three.js RGBA demo does not.
		Local<Value> yuv_val;
		if (opts->Get(ctx, nx_str(iso, "yuv")).ToLocal(&yuv_val) &&
		    yuv_val->IsBoolean()) {
			want_yuv = yuv_val->BooleanValue(iso);
		}
		// `userAgent` / `referer` (2026-09-17) — override the HTTP request
		// shaping for http(s) sources. Ignored for local media. Both are
		// sanitised of CR/LF down in nx_media_open, at the point of use,
		// rather than here: that keeps the guarantee attached to the code
		// that actually composes the request, so a future second caller of
		// nx_media_open cannot bypass it.
		Local<Value> ua_val;
		if (opts->Get(ctx, nx_str(iso, "userAgent")).ToLocal(&ua_val) &&
		    ua_val->IsString()) {
			String::Utf8Value s(iso, ua_val);
			if (*s) opt_user_agent = *s;
		}
		Local<Value> ref_val;
		if (opts->Get(ctx, nx_str(iso, "referer")).ToLocal(&ref_val) &&
		    ref_val->IsString()) {
			String::Utf8Value s(iso, ref_val);
			if (*s) opt_referer = *s;
		}
		// hwAccel, muted, noAudio: consumed silently. nx_media picks
		// hw/sw internally; the audio graph is wired lazily by the JS
		// wrapper once the (async) open reports a usable audio stream.
	}

	// Async open: hand the URL to a worker thread and return a "pending"
	// decoder immediately so the JS main thread never blocks on the network
	// open. width/height/duration read 0 and nextFrame() returns null until
	// open_thread publishes the media; frame_buf is allocated lazily in
	// next_frame() once the real dimensions are known.
	nx_video_decoder_t *d = new nx_video_decoder_t();
	d->open_url = url;
	d->open_want_loop = want_loop;
	d->want_yuv = want_yuv;
	d->open_user_agent = std::move(opt_user_agent);
	d->open_referer = std::move(opt_referer);
	d->open_thread = std::thread([d]() {
		char err[256] = {};
		// Empty string -> NULL so nx_media_open applies its own default UA
		// (and sends no Referer) rather than setting an empty header.
		const nx_media_net_t net = {
		    d->open_user_agent.empty() ? nullptr : d->open_user_agent.c_str(),
		    d->open_referer.empty() ? nullptr : d->open_referer.c_str(),
		};
		int av_err = 0;
		nx_media_t *m = nx_media_open(d->open_url.c_str(), nullptr, 0,
		                              nullptr, err, sizeof(err), d->want_yuv,
		                              &net, &av_err, &d->open_abort);
		if (m) {
			if (d->open_want_loop) nx_media_set_loop(m, true);
			d->media = m; // made visible by the open_done release store
		} else {
			snprintf(d->open_err, sizeof(d->open_err), "%s", err);
			d->open_av_err = av_err;
		}
		d->open_done.store(true, std::memory_order_release);
	});

	Local<Object> obj = nx::NewWrapped(iso);
	nx::Wrap<nx_video_decoder_t>(iso, obj, d, free_video_decoder);
	info.GetReturnValue().Set(obj);
}

// videoDecoderPlay(dec) → undefined
void nx_video_decoder_play(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	if (!d) return;
	// Remember the intent so a play() issued while the open is still in
	// flight takes effect (apply_pending_on_ready) once the media lands.
	d->play_requested.store(true, std::memory_order_relaxed);
	d->paused = false;
	nx_media_t *m = ready_media(d);
	if (m) nx_media_play(m);
}

// videoDecoderPause(dec) → undefined (cut #22b: was deferred in cut #22).
void nx_video_decoder_pause(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	if (!d) return;
	d->play_requested.store(false, std::memory_order_relaxed);
	d->paused = true;
	nx_media_t *m = ready_media(d);
	if (m) nx_media_pause(m);
}

// videoDecoderSeek(dec, seconds) → undefined (cut #22b).
void nx_video_decoder_seek(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	if (!d) return;
	double t = 0;
	if (!info[1]->NumberValue(iso->GetCurrentContext()).To(&t)) t = 0;
	if (!(t >= 0)) t = 0;
	nx_media_t *m = ready_media(d);
	if (m) {
		nx_media_seek(m, t);
	} else {
		// Seek requested before the open finished (e.g. resume-position
		// seek on a fresh source) — apply it once the media is ready.
		d->seek_target.store(t, std::memory_order_relaxed);
		d->seek_pending.store(true, std::memory_order_relaxed);
	}
}

// videoDecoderClose(dec) → undefined
void nx_video_decoder_close(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	if (!d || d->closed) return;
	d->closed = true;
	// Join the open thread first so it can't publish media after we've torn
	// it down — same ordering as the finalizer. The abort latch is what
	// keeps this from blocking: close() is called straight from JS (a page
	// navigating away, a user cancelling a load), so it must return promptly
	// even when the open is parked on an unreachable host.
	d->open_abort.store(true, std::memory_order_relaxed);
	if (d->open_thread.joinable()) d->open_thread.join();
	if (d->media) {
		nx_media_destroy(d->media); // joins decode thread first
		d->media = nullptr;
	}
	if (d->audio_node) {
		nx_audio_node_release(d->audio_node);
		d->audio_node = nullptr;
	}
	free(d->frame_buf);
	d->frame_buf = nullptr;
}

// videoDecoderCreateAudioNode(dec, audioCtxHandle) -> stream node handle | null
//
// Attaches a STREAM_SOURCE node to the media's audio track and hands its
// handle to the JS wrapper (mirrors nx_video_create_audio_node for the
// Video element). Returns null when the source has no audio, when audio
// was already wired, or when the media handle is torn down.
//
// The returned wrapper has NO finalizer: node ownership is the decoder's
// (released in free_video_decoder AFTER the decode thread joins). The JS
// side keeps the returned wrapper alive as long as it holds the decoder
// (via a GainNode reference stored on the JS class).
void nx_video_decoder_create_audio_node(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	if (!d) return;
	nx_audio_ctx_t *ctx = nx::Unwrap<nx_audio_ctx_t>(info[1]);
	if (!ctx) {
		nx_throw(iso, "expected AudioContext handle");
		return;
	}
	nx_media_t *m = ready_media(d);
	if (!m || !nx_media_has_audio(m) || d->audio_node) {
		info.GetReturnValue().SetNull();
		return;
	}
	nx_audio_node *node =
	    nx_audio_node_create(ctx->graph, NX_AUDIO_NODE_STREAM_SOURCE);
	d->audio_node = node;
	nx_media_set_audio_node(m, node, ctx->graph->sample_rate);
	Local<Object> obj = nx::NewWrapped(iso);
	obj->SetAlignedPointerInInternalField(0, node,
	                                      kEmbedderDataTypeTagDefault);
	info.GetReturnValue().Set(obj);
}

// videoDecoderSetVolume(dec, value: number) — clamps to [0, 1]. The JS
// wrapper also applies this through its GainNode; the C side stores it so
// that the `volume` getter reads the last-set value.
void nx_video_decoder_set_volume(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	if (!d) return;
	double v = 1.0;
	if (!info[1]->NumberValue(iso->GetCurrentContext()).To(&v)) v = 1.0;
	if (!(v >= 0)) v = 0;
	if (v > 1) v = 1;
	d->volume = (float)v;
}

// videoDecoderSetMuted(dec, muted: boolean) — JS wrapper zeroes its
// GainNode's gain when muted; the C state feeds the `muted` getter.
void nx_video_decoder_set_muted(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	if (!d) return;
	d->muted = info[1]->BooleanValue(iso);
}

// Cut #22b Stage 2 (2026-07-02): visualizer surface.
//
// Each accessor takes a Float32Array output buffer (the caller pre-sizes
// it — spectraplay uses 256 for waveform, 128 for spectrum) and returns
// true when it was filled. Returning false lets the caller fall back to
// a synthetic waveform (spectraplay does this by design).

// Extract the caller's Float32Array output pointer + element count.
// Returns NULL and no exception on non-Float32Array (caller returns
// false to the JS side, matching the try/catch pattern in
// brewser-runtime's videoGetFrequencyData/videoGetWaveform).
float *unwrap_f32(const FunctionCallbackInfo<Value> &info, uint32_t *out_len) {
	if (!info[1]->IsFloat32Array()) return nullptr;
	Local<Float32Array> ta = info[1].As<Float32Array>();
	std::shared_ptr<BackingStore> bs = ta->Buffer()->GetBackingStore();
	*out_len = (uint32_t)(ta->ByteLength() / sizeof(float));
	return reinterpret_cast<float *>(
	    static_cast<uint8_t *>(bs->Data()) + ta->ByteOffset());
}

// videoDecoderGetWaveform(dec, out: Float32Array) → boolean
void nx_video_decoder_get_waveform(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	nx_media_t *m = ready_media(d);
	if (!m) {
		info.GetReturnValue().Set(False(iso));
		return;
	}
	uint32_t n = 0;
	float *out = unwrap_f32(info, &n);
	if (!out) {
		info.GetReturnValue().Set(False(iso));
		return;
	}
	bool ok = nx_media_read_waveform(m, out, n);
	info.GetReturnValue().Set(Boolean::New(iso, ok));
}

// videoDecoderGetFrequencyData(dec, out: Float32Array) → boolean
void nx_video_decoder_get_frequency_data(
    const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	nx_media_t *m = ready_media(d);
	if (!m) {
		info.GetReturnValue().Set(False(iso));
		return;
	}
	uint32_t n = 0;
	float *out = unwrap_f32(info, &n);
	if (!out) {
		info.GetReturnValue().Set(False(iso));
		return;
	}
	bool ok = nx_media_read_spectrum(m, out, n);
	info.GetReturnValue().Set(Boolean::New(iso, ok));
}

// videoDecoderGetAudioLevels(dec) → number[] (bass, mid, high) — [] when
// audio hasn't accumulated a tap window yet.
void nx_video_decoder_get_audio_levels(
    const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Context> ctx = iso->GetCurrentContext();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	nx_media_t *m = ready_media(d);
	if (!m) {
		info.GetReturnValue().Set(Array::New(iso, 0));
		return;
	}
	float bands[3] = {0, 0, 0};
	uint32_t n = nx_media_read_audio_levels(m, bands, 3);
	Local<Array> arr = Array::New(iso, (int)n);
	for (uint32_t i = 0; i < n; i++) {
		arr->Set(ctx, i, Number::New(iso, (double)bands[i])).Check();
	}
	info.GetReturnValue().Set(arr);
}

// videoDecoderNextFrame(dec) → {data, width, height, pts, ended} | null
void nx_video_decoder_next_frame(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Context> ctx = iso->GetCurrentContext();
	nx_video_decoder_t *d = get_decoder(iso, info[0]);
	nx_media_t *m = ready_media(d);
	if (!m) return; // still opening → JS null (the wrapper reads this as buffering)

	// First tick after the async open lands: apply any play()/seek() issued
	// while pending, latch the real dimensions, and lazily allocate the
	// presentation buffer (main thread → nx_alloc is safe here). Gated on
	// d->width == 0 (a ready video always has width > 0, so this is a
	// reliable "not yet latched" sentinel; audio-only sources have no video
	// stream and keep width 0 + frame_buf null).
	apply_pending_on_ready(d, m);
	if (d->width == 0 && nx_media_has_video(m)) {
		int w = nx_media_width(m);
		int h = nx_media_height(m);
		if (w > 0 && h > 0) {
			d->width = w;
			d->height = h;
			d->duration = nx_media_duration(m);
			// I420 frames are 1.5 B/px (Y|U|V), BGRA is 4 B/px. The ring
			// (media) and this presentation buffer must agree on size — both
			// keyed off the same want_yuv flag passed to nx_media_open.
			const size_t fb_size = d->want_yuv
			                           ? nx_media_i420_size(w, h)
			                           : (size_t)w * h * 4;
			d->frame_buf = (uint8_t *)nx_alloc(iso, fb_size);
			if (!d->frame_buf) return; // nx_alloc scheduled the OOM exception
			memset(d->frame_buf, 0, fb_size);
		}
	}

	// End-of-stream: emit one sentinel, then keep returning undefined.
	if (nx_media_ended(m)) {
		if (d->eos_sent) return;
		d->eos_sent = true;
		Local<Object> result = Object::New(iso);
		result->Set(ctx, nx_str(iso, "data"), Null(iso)).Check();
		result->Set(ctx, nx_str(iso, "width"), Integer::New(iso, d->width))
		    .Check();
		result->Set(ctx, nx_str(iso, "height"), Integer::New(iso, d->height))
		    .Check();
		result->Set(ctx, nx_str(iso, "pts"), Number::New(iso, 0.0)).Check();
		result->Set(ctx, nx_str(iso, "ended"), True(iso)).Check();
		info.GetReturnValue().Set(result);
		return;
	}

	if (!d->frame_buf) return; // audio-only source; nothing to present

	if (!nx_media_present(m, &d->frame_buf)) {
		// Media clock says no new frame is due yet.
		return;
	}

	// Have a fresh frame in d->frame_buf, in BGRA order (Skia's ARGB32
	// memory layout). Copy it into a JS ArrayBuffer.
	//   - Default (RGBA): swizzle B↔R while copying — the documented
	//     nextFrame() contract, consumed by the Three.js webgl_materials_video
	//     demo's RGBAFormat DataTexture.
	//   - `bgra` arg true (brewser video path, 2026-09-05 Fix D): a straight
	//     memcpy, no swizzle. The caller feeds it to Switch.imageWriteBGRA →
	//     Skia with NO further swap, killing the old BGRA→RGBA→BGRA double
	//     swizzle (two per-pixel byte loops on the main thread per frame).
	// YUV path (2026-09-06): the frame is already planar I420 in d->frame_buf.
	// Deliver it verbatim (no swizzle) plus a `yuv` flag and the color-space
	// tag so the JS side builds a Skia YUVA image and uploads 1.5 B/px.
	if (d->want_yuv) {
		size_t byte_count = nx_media_i420_size(d->width, d->height);
		// Reuse ONE ArrayBuffer OBJECT (see struct comment) — not just its
		// backing — so V8's per-ArrayBuffer external accounting stays flat and
		// stops firing the periodic major GC that hitched the presenter. Create
		// the backing + wrapper once per dimension; thereafter just memcpy the
		// frame into the persistent backing and return the SAME object. Safe
		// because tickVideo copies this into the frame bitmap synchronously (and
		// drops its reference — st.frameBytes=null on the YUV path) before the
		// next present() overwrites the backing.
		if (d->deliver_size != byte_count || !d->deliver_bs ||
		    d->deliver_ab.IsEmpty()) {
			d->deliver_bs = ArrayBuffer::NewBackingStore(iso, byte_count);
			d->deliver_size = byte_count;
			d->deliver_ab.Reset(iso, ArrayBuffer::New(iso, d->deliver_bs));
		}
		memcpy(d->deliver_bs->Data(), d->frame_buf, byte_count);
		Local<ArrayBuffer> ab = d->deliver_ab.Get(iso);
		Local<Object> result = Object::New(iso);
		result->Set(ctx, nx_str(iso, "data"), ab).Check();
		result->Set(ctx, nx_str(iso, "width"), Integer::New(iso, d->width))
		    .Check();
		result->Set(ctx, nx_str(iso, "height"), Integer::New(iso, d->height))
		    .Check();
		result->Set(ctx, nx_str(iso, "pts"),
		            Number::New(iso, nx_media_current_time(m)))
		    .Check();
		result->Set(ctx, nx_str(iso, "yuv"), True(iso)).Check();
		result->Set(ctx, nx_str(iso, "colorSpace"),
		            Integer::New(iso, nx_media_yuv_colorspace(m)))
		    .Check();
		result->Set(ctx, nx_str(iso, "ended"), False(iso)).Check();
		info.GetReturnValue().Set(result);
		return;
	}

	const bool want_bgra =
	    info.Length() >= 2 && info[1]->BooleanValue(iso);
	size_t byte_count = (size_t)d->width * d->height * 4;
	std::unique_ptr<BackingStore> bs =
	    ArrayBuffer::NewBackingStore(iso, byte_count);
	uint8_t *dst = (uint8_t *)bs->Data();
	const uint8_t *src = d->frame_buf;
	if (want_bgra) {
		memcpy(dst, src, byte_count);
	} else {
		for (size_t i = 0; i < byte_count; i += 4) {
			dst[i] = src[i + 2];     // R ← B
			dst[i + 1] = src[i + 1]; // G
			dst[i + 2] = src[i];     // B ← R
			dst[i + 3] = src[i + 3]; // A
		}
	}
	Local<ArrayBuffer> ab = ArrayBuffer::New(iso, std::move(bs));

	Local<Object> result = Object::New(iso);
	result->Set(ctx, nx_str(iso, "data"), ab).Check();
	result->Set(ctx, nx_str(iso, "width"), Integer::New(iso, d->width)).Check();
	result->Set(ctx, nx_str(iso, "height"), Integer::New(iso, d->height))
	    .Check();
	result->Set(ctx, nx_str(iso, "pts"),
	            Number::New(iso, nx_media_current_time(m)))
	    .Check();
	result->Set(ctx, nx_str(iso, "ended"), False(iso)).Check();
	info.GetReturnValue().Set(result);
}

// Prototype getters. `this` is the JS VideoDecoder instance (wrapped).
// All media-backed getters read via ready_media() so they report 0 / false /
// null while the async open is still in flight, and true values the instant
// it lands.
void nx_vd_get_width(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	info.GetReturnValue().Set(Integer::New(info.GetIsolate(), m ? nx_media_width(m) : 0));
}
void nx_vd_get_height(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	info.GetReturnValue().Set(Integer::New(info.GetIsolate(),
	                                        m ? nx_media_height(m) : 0));
}
void nx_vd_get_duration(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	info.GetReturnValue().Set(Number::New(info.GetIsolate(),
	                                       m ? nx_media_duration(m) : 0.0));
}
// Content frame rate (0 while opening / unknown). The brewser shell reads this
// to pick the present cadence: >~32fps => keep 60 Hz (don't engage the 30 Hz
// video pacing lock, which would halve 60fps content to 30 shown fps).
void nx_vd_get_fps(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	info.GetReturnValue().Set(Number::New(info.GetIsolate(),
	                                       m ? nx_media_content_fps(m) : 0.0));
}
void nx_vd_get_error(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	if (m) {
		const char *err = nx_media_error(m);
		if (err) info.GetReturnValue().Set(nx_str_lossy(iso, err));
		else info.GetReturnValue().SetNull();
		return;
	}
	// Open finished but FAILED (bad URL, TLS handshake, "Protocol not found")
	// → surface the captured open error. Still-opening → null.
	if (d && d->open_done.load(std::memory_order_acquire) && d->open_err[0]) {
		info.GetReturnValue().Set(nx_str_lossy(iso, d->open_err));
		return;
	}
	info.GetReturnValue().SetNull();
}
// Short stable token for WHY the open failed ("forbidden", "not-found",
// "timeout", "network", ...), or null while opening / once open succeeded.
// Lets a catalogue page retire a 404 permanently but retry a timeout, rather
// than blacklisting a working stream after one blip.
void nx_vd_get_error_kind(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	if (d && d->open_done.load(std::memory_order_acquire) && d->open_err[0]) {
		info.GetReturnValue().Set(
		    nx_str_lossy(iso, nx_media_fail_kind(d->open_av_err)));
		return;
	}
	info.GetReturnValue().SetNull();
}
void nx_vd_get_ended(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	bool v = m && nx_media_ended(m);
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), v));
}
void nx_vd_get_has_video(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	bool v = m && nx_media_has_video(m);
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), v));
}
void nx_vd_get_has_audio(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	bool v = m && nx_media_has_audio(m);
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), v));
}
// Deferred (stubs): the demo reads them but doesn't act on them.
// Cut #22b: track paused state at the JS boundary (set in play/pause
// bindings). Backs the seek-bar's is-playing indicator + spectraplay's
// visualizer scale (which reads `audio.paused`). Fresh decoders report
// `true` (not yet started); after nx_media_ended is reached we don't
// flip it — an ended decoder is effectively paused for UI purposes.
void nx_vd_get_paused(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	bool paused = true;
	if (d) {
		nx_media_t *m = ready_media(d);
		paused = d->paused || (m && nx_media_ended(m));
	}
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), paused));
}
void nx_vd_get_used_hw(const FunctionCallbackInfo<Value> &info) {
	info.GetReturnValue().Set(False(info.GetIsolate()));
}
// Cut #22b (2026-07-02): read the current audio state from the decoder;
// `audioTime` slaves to the audio clock when audio is wired (else 0.0).
void nx_vd_get_muted(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	info.GetReturnValue().Set(Boolean::New(info.GetIsolate(), d && d->muted));
}
void nx_vd_get_volume(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	info.GetReturnValue().Set(
	    Number::New(info.GetIsolate(), d ? (double)d->volume : 1.0));
}
void nx_vd_get_audio_time(const FunctionCallbackInfo<Value> &info) {
	nx_video_decoder_t *d = nx::Unwrap<nx_video_decoder_t>(info.This());
	nx_media_t *m = ready_media(d);
	double t = (m && d->audio_node) ? nx_media_current_time(m) : 0.0;
	info.GetReturnValue().Set(Number::New(info.GetIsolate(), t));
}
void nx_vd_get_audio_error(const FunctionCallbackInfo<Value> &info) {
	info.GetReturnValue().SetNull();
}

// videoDecoderInit(ClassCtor) — TS wrapper calls this once at module load
// to install prototype getters onto Switch.VideoDecoder.prototype.
void nx_video_decoder_init_class(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Context> ctx = iso->GetCurrentContext();
	if (info.Length() < 1 || !info[0]->IsObject()) {
		nx_throw(iso, "videoDecoderInit(ClassCtor) — object required");
		return;
	}
	Local<Value> proto_val;
	if (!info[0].As<Object>()
	         ->Get(ctx, nx_str(iso, "prototype"))
	         .ToLocal(&proto_val)) {
		return;
	}
	if (!proto_val->IsObject()) return;
	Local<Object> proto = proto_val.As<Object>();
	NX_DEF_GET(proto, "width", nx_vd_get_width);
	NX_DEF_GET(proto, "height", nx_vd_get_height);
	NX_DEF_GET(proto, "duration", nx_vd_get_duration);
	NX_DEF_GET(proto, "fps", nx_vd_get_fps);
	NX_DEF_GET(proto, "error", nx_vd_get_error);
	NX_DEF_GET(proto, "errorKind", nx_vd_get_error_kind);
	NX_DEF_GET(proto, "ended", nx_vd_get_ended);
	NX_DEF_GET(proto, "usedVideo", nx_vd_get_has_video);
	NX_DEF_GET(proto, "usedAudio", nx_vd_get_has_audio);
	NX_DEF_GET(proto, "paused", nx_vd_get_paused);
	NX_DEF_GET(proto, "usedHw", nx_vd_get_used_hw);
	NX_DEF_GET(proto, "muted", nx_vd_get_muted);
	NX_DEF_GET(proto, "volume", nx_vd_get_volume);
	NX_DEF_GET(proto, "audioTime", nx_vd_get_audio_time);
	NX_DEF_GET(proto, "audioError", nx_vd_get_audio_error);
}

} // namespace

void nx_init_video_decoder(Isolate *iso, Local<Object> init_obj) {
	NX_SET_FUNC(init_obj, "videoDecoderInit", nx_video_decoder_init_class);
	NX_SET_FUNC(init_obj, "videoDecoderNew", nx_video_decoder_new);
	NX_SET_FUNC(init_obj, "videoDecoderPlay", nx_video_decoder_play);
	NX_SET_FUNC(init_obj, "videoDecoderPause", nx_video_decoder_pause);
	NX_SET_FUNC(init_obj, "videoDecoderSeek", nx_video_decoder_seek);
	NX_SET_FUNC(init_obj, "videoDecoderClose", nx_video_decoder_close);
	NX_SET_FUNC(init_obj, "videoDecoderNextFrame", nx_video_decoder_next_frame);
	// Cut #22b: audio-graph attach + volume/mute (spectraplay MP3 fix).
	NX_SET_FUNC(init_obj, "videoDecoderCreateAudioNode",
	            nx_video_decoder_create_audio_node);
	NX_SET_FUNC(init_obj, "videoDecoderSetVolume",
	            nx_video_decoder_set_volume);
	NX_SET_FUNC(init_obj, "videoDecoderSetMuted",
	            nx_video_decoder_set_muted);
	// Cut #22b Stage 2: visualizer surface (spectraplay's spectrum + wave).
	NX_SET_FUNC(init_obj, "videoDecoderGetWaveform",
	            nx_video_decoder_get_waveform);
	NX_SET_FUNC(init_obj, "videoDecoderGetFrequencyData",
	            nx_video_decoder_get_frequency_data);
	NX_SET_FUNC(init_obj, "videoDecoderGetAudioLevels",
	            nx_video_decoder_get_audio_levels);
}
