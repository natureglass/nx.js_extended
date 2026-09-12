// Portable ffmpeg media decode pipeline. See media-decoder.h for the
// architecture and threading contract. No V8 / libnx — compiled into both the
// device runtime and the host nxjs-test binary.
#include "media-decoder.h"
#include "audio-graph.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

// Quiet ffmpeg's default stderr chatter (codec warnings during seeks, etc.);
// real errors still surface through the nx_media error paths.
struct av_log_quiet {
	av_log_quiet() { av_log_set_level(AV_LOG_ERROR); }
} av_log_quiet_init;

// Ledger #115 (2026-07-10): bumped from 3 to 12. The 3-slot ring kept
// producer constantly blocked in enqueue_video (post-#113 diag showed
// depth=3 constantly), which starved the shared decode thread of time to
// process audio packets — audio ring drained to 0 every ~1 s, triggering
// Fix B's wall-extrapolation-then-resume-jump-back cycle. 12 slots
// (~500 ms of video buffer at 24 fps) gives the decoder headroom to keep
// audio flowing evenly.
constexpr int RING_SLOTS = 12;
// Cut #22b Stage 2: audio-tap window size for the visualizer readers.
// Must be a power of two (masked with TAP_LEN - 1 in the writer). 1024
// samples ≈ 21.3 ms at 48 kHz — matches the pre-migration (QuickJS-era)
// video.c FFT window so per-bin magnitudes land in the same range as
// spectraplay's `vizFreqGain=30` was originally tuned for.
constexpr uint32_t TAP_LEN = 1024;
constexpr uint32_t TAP_MASK = TAP_LEN - 1;
// Log2(TAP_LEN) — the FFT butterfly loop needs a compile-time bit count.
constexpr int TAP_LOG2 = 10;
// Local pi (avoid depending on <math.h>'s M_PI which is a non-portable
// GNU/BSD extension — devkitPro's newlib does provide it, but keep it
// robust against toolchain changes).
constexpr double TAP_PI = 3.14159265358979323846;
// Present a frame when its PTS is within this much of the clock (one frame of
// slack at 24 fps is ~41 ms; this is just sub-frame jitter tolerance).
constexpr double PRESENT_EPSILON = 0.001;
// ---------------------------------------------------------------------------
// Ledger #112 (2026-07-10) — diagnostic probes for the ~1s periodic video
// stutter reported on webgl_materials_video. Toggled by MEDIA_DIAG_112. Logs
// three signals to stderr (routed to nxjs-debug.log):
//   [md-diag:enq]  per-video-frame sws_scale ms + video-ring depth
//   [md-diag:snap] per audio-clock resync event with delta + interval
//   [md-diag:prs]  per 60 nx_media_present calls: min/max/avg ring depth,
//                  no-frame count (candidate<0), skip count (>1 slot behind)
// All three tag prefixes are unique so `grep '\[md-diag:' log` returns the
// full record. Rate: enq ≈ 24/s, snap on-demand (expected ≤1/s), prs ≈ 1/s.
// Total log volume: ~2 KB/s for a 480x204 24fps stream — trivial.
// ---------------------------------------------------------------------------
#define MEDIA_DIAG_112 0

struct video_slot {
	uint8_t *bgra = nullptr;
	double pts = 0;
};

} // namespace

struct nx_media {
	// ---- IO (file or memory) ----
	FILE *file = nullptr;
	int64_t file_size = 0;
	const uint8_t *mem = nullptr;
	size_t mem_size = 0;
	size_t mem_pos = 0;
	// Owns the memory buffer for the media's lifetime (the decode thread
	// streams from it until destroy).
	std::shared_ptr<void> mem_hold;

	// ---- ffmpeg ----
	AVIOContext *avio = nullptr;
	AVFormatContext *fmt = nullptr;
	AVCodecContext *vctx = nullptr;
	AVCodecContext *actx = nullptr;
	int vstream = -1;
	int astream = -1;
	SwsContext *sws = nullptr;
	SwrContext *swr = nullptr;

	// ---- metadata ----
	int width = 0;
	int height = 0;
	double duration = 0;
	double vframe_dur = 1.0 / 30; // nominal frame duration (seek slack)

	// ---- output pixel format (2026-09-06) ----
	// When true, the ring slots hold planar I420 (Y|U|V contiguous, 1.5 B/px)
	// instead of BGRA (4 B/px) — see nx_media_open's want_yuv. yuv_cs is the
	// neutral color-space tag (0=709ltd,1=601ltd,2=full,3=709full) derived
	// from the first decoded frame's colorimetry.
	bool out_yuv = false;
	int yuv_cs = 0;

	// ---- video presentation ring (SPSC: decode thread -> main thread) ----
	video_slot slots[RING_SLOTS];
	std::atomic<uint64_t> vwrite{0};
	std::atomic<uint64_t> vread{0};

	// ---- demux-ahead decouple (2026-09-06, decode thread only) ----
	// The decode thread reads packets into these per-stream queues so that a
	// full video ring can no longer BLOCK the thread (in enqueue_video) and
	// thereby stop audio production — the coupling that starved audio and
	// jittered the video clock at a 30 Hz present. Video is enqueued
	// non-blocking (a decoded frame that doesn't fit is held in `v_pending`);
	// audio is decoded only while the audio ring is under a small time target,
	// so it neither blocks nor runs so far ahead that it starves video.
	std::deque<AVPacket *> vpktq; // compressed video packets awaiting decode
	std::deque<AVPacket *> apktq; // compressed audio packets awaiting decode
	AVFrame *v_pending = nullptr; // decoded video frame the full ring rejected
	bool v_pending_valid = false;
	double v_pending_pts = 0;
	bool demux_eof = false; // av_read_frame hit EOF; queues still draining

	// ---- audio output ----
	nx_audio_node *audio_node = nullptr;
	double audio_out_rate = 48000;
	std::vector<float> audio_scratch;
	// Mapping from the stream node's consumed-frame counter to media time:
	// audio_time(consumed) = audio_pts_base + (consumed - audio_base) / rate.
	std::atomic<bool> audio_clock_valid{false};
	std::atomic<double> audio_pts_base{0};
	std::atomic<uint64_t> audio_base{0};

	// ---- clock (main thread only) ----
	double clock_base = 0;
	std::chrono::steady_clock::time_point clock_anchor;
	bool clock_running = false;
	// Ledger #115: monotonic gate on clock_now returns. `at` itself is
	// monotonic (consumed only increases), but Fix B's wall-extrapolation
	// during audio stall can outrun `at`; when audio resumes, returning
	// the fresh `at` causes a visible frame regression (~150 ms every
	// ~1 s under Citron's audio-underrun cycle). `last_clock` gates
	// clock_now to `max(computed, last_clock)` — no regression, ever.
	// Reset to seek target in do_seek so seeks work.
	double last_clock = 0;
	// Cut #22c (2026-07-10): last audio `consumed` sample count observed by
	// clock_now(). When this matches the current value on a fresh call,
	// the audio consumer has NOT advanced since — treat the audio clock as
	// stalled and let wall time free-run rather than snap wall back to a
	// frozen `at`. Prevents the video-ring-full / audio-underrun deadlock
	// that stalls webgl_materials_video after a seek (see #109). Reset to 0
	// whenever `audio_clock_valid` transitions false → true so a fresh seek
	// anchor's first main-thread poll doesn't false-positive.
	uint64_t audio_consumed_last = 0;

	// ---- presentation quality counters (main thread only) ----
	uint64_t presented_frames = 0;
	uint64_t dropped_frames = 0;

	// ---- audio tap (Cut #22b Stage 2 — visualizer surface) ----
	// Rolling mono downmix of the resampled audio. Written by the decode
	// thread inside enqueue_audio; read by the main thread through
	// nx_media_read_waveform / _read_spectrum / _read_audio_levels.
	std::mutex tap_mutex;
	float tap_ring[TAP_LEN] = {};
	uint32_t tap_write_pos = 0;
	uint64_t tap_written = 0;

	// ---- control ----
	std::mutex ctl_mutex;
	std::condition_variable ctl_cv;
	std::thread thread;
	std::atomic<bool> quit{false};
	std::atomic<bool> playing{false};
	std::atomic<bool> looping{false};
	std::atomic<bool> seek_requested{false};
	std::atomic<double> seek_target{0};
	std::atomic<bool> seeking{false};
	// After a seek, present the first decoded frame even if its PTS is
	// slightly past the (paused) clock — the user expects to see the frame
	// nearest the seek target.
	std::atomic<bool> present_force{false};
	std::atomic<bool> eof{false};
	std::atomic<bool> fatal{false};
	char error_buf[256] = {};
	// Loop playback runs in a monotonic PTS domain: each wrap adds the media
	// duration so frame/audio PTS keep increasing past the clock.
	double loop_offset = 0; // decode thread only
};

namespace {

// ---------------------------------------------------------------------------
// Custom AVIO (stdio file or memory buffer)
// ---------------------------------------------------------------------------

int avio_read_cb(void *opaque, uint8_t *buf, int n) {
	nx_media *m = static_cast<nx_media *>(opaque);
	if (m->file) {
		size_t r = fread(buf, 1, (size_t)n, m->file);
		return r > 0 ? (int)r : AVERROR_EOF;
	}
	size_t rem = m->mem_size - m->mem_pos;
	if (rem == 0)
		return AVERROR_EOF;
	size_t r = rem < (size_t)n ? rem : (size_t)n;
	memcpy(buf, m->mem + m->mem_pos, r);
	m->mem_pos += r;
	return (int)r;
}

int64_t avio_seek_cb(void *opaque, int64_t offset, int whence) {
	nx_media *m = static_cast<nx_media *>(opaque);
	if (whence & AVSEEK_SIZE)
		return m->file ? m->file_size : (int64_t)m->mem_size;
	whence &= ~AVSEEK_FORCE;
	if (m->file) {
		if (fseek(m->file, (long)offset, whence) != 0)
			return -1;
		return (int64_t)ftell(m->file);
	}
	int64_t base = whence == SEEK_CUR    ? (int64_t)m->mem_pos
	               : whence == SEEK_END ? (int64_t)m->mem_size
	                                    : 0;
	int64_t pos = base + offset;
	if (pos < 0 || pos > (int64_t)m->mem_size)
		return -1;
	m->mem_pos = (size_t)pos;
	return pos;
}

// ---------------------------------------------------------------------------
// Decode thread
// ---------------------------------------------------------------------------

void set_fatal(nx_media *m, const char *what, int averr) {
	if (m->fatal.load())
		return;
	char detail[128] = {};
	if (averr != 0)
		av_strerror(averr, detail, sizeof(detail));
	snprintf(m->error_buf, sizeof(m->error_buf), "%s%s%s", what,
	         averr ? ": " : "", detail);
	m->fatal.store(true);
}

double frame_pts_seconds(nx_media *m, AVFrame *frame, int stream_index) {
	int64_t ts = frame->best_effort_timestamp;
	if (ts == AV_NOPTS_VALUE)
		ts = frame->pts;
	if (ts == AV_NOPTS_VALUE)
		return 0;
	return ts * av_q2d(m->fmt->streams[stream_index]->time_base);
}

// Sleep briefly, returning false if the thread should abandon its current
// blocking operation (quit or a pending seek).
bool decode_wait(nx_media *m) {
	if (m->quit.load() || m->seek_requested.load())
		return false;
	std::this_thread::sleep_for(std::chrono::milliseconds(5));
	return true;
}

// Non-blocking: convert `frame` into the next ring slot IF one is free.
// Returns 1 = enqueued, 0 = ring full (caller retries / holds the frame),
// -1 = fatal. Demux-ahead (2026-09-06): the decode thread must never block on
// a full video ring — that stalls audio production — so the slow path uses
// this and stashes a rejected frame in `v_pending` instead of waiting.
int try_enqueue_video(nx_media *m, AVFrame *frame, double pts) {
	if (m->vwrite.load(std::memory_order_relaxed) -
	        m->vread.load(std::memory_order_acquire) >=
	    RING_SLOTS)
		return 0; // ring full
	uint64_t w = m->vwrite.load(std::memory_order_relaxed);
	video_slot *slot = &m->slots[w % RING_SLOTS];

	const AVPixelFormat out_fmt =
	    m->out_yuv ? AV_PIX_FMT_YUV420P : AV_PIX_FMT_BGRA;
	m->sws = sws_getCachedContext(m->sws, frame->width, frame->height,
	                              (AVPixelFormat)frame->format, m->width,
	                              m->height, out_fmt, SWS_BILINEAR,
	                              NULL, NULL, NULL);
	if (!m->sws) {
		set_fatal(m, "failed to create scaler", 0);
		return -1;
	}
	uint8_t *dst[4];
	int dst_stride[4];
	if (m->out_yuv) {
		// Planar I420 packed contiguously into the slot: Y (W*H) then U then
		// V (each (W/2)*(H/2)). Matches Skia's SkYUVAPixmaps kY_U_V/k420 tight
		// layout so the GPU upload reads it directly.
		const int cw = (m->width + 1) / 2, ch = (m->height + 1) / 2;
		uint8_t *y = slot->bgra;
		uint8_t *u = y + (size_t)m->width * m->height;
		uint8_t *v = u + (size_t)cw * ch;
		dst[0] = y; dst[1] = u; dst[2] = v; dst[3] = NULL;
		dst_stride[0] = m->width; dst_stride[1] = cw; dst_stride[2] = cw;
		dst_stride[3] = 0;
		// Derive the neutral color-space tag once from the decoded frame's
		// colorimetry (cheap; the value is stable so re-deriving is harmless).
		const bool full = frame->color_range == AVCOL_RANGE_JPEG;
		int cs;
		if (full) {
			cs = 2; // JPEG/full range
		} else if (frame->colorspace == AVCOL_SPC_BT709) {
			cs = 0; // Rec709 limited
		} else if (frame->colorspace == AVCOL_SPC_BT470BG ||
		           frame->colorspace == AVCOL_SPC_SMPTE170M) {
			cs = 1; // Rec601 limited
		} else {
			cs = (m->height >= 720) ? 0 : 1; // unspecified → guess by size
		}
		m->yuv_cs = cs;
	} else {
		dst[0] = slot->bgra; dst[1] = NULL; dst[2] = NULL; dst[3] = NULL;
		dst_stride[0] = m->width * 4; dst_stride[1] = 0; dst_stride[2] = 0;
		dst_stride[3] = 0;
	}
#if MEDIA_DIAG_112
	auto _sws_t0 = std::chrono::steady_clock::now();
#endif
	sws_scale(m->sws, frame->data, frame->linesize, 0, frame->height, dst,
	          dst_stride);
#if MEDIA_DIAG_112
	double _sws_ms = std::chrono::duration<double, std::milli>(
	                     std::chrono::steady_clock::now() - _sws_t0)
	                     .count();
	static uint64_t _diag_enq_idx = 0;
	uint64_t _diag_idx = _diag_enq_idx++;
	uint64_t _diag_depth = w + 1 - m->vread.load(std::memory_order_relaxed);
	// One line per decoded video frame: sws_ms is the H.264 → BGRA scale
	// cost (I-frames spike vs P-frames), depth is the ring occupancy after
	// this push. Correlate periodic depth==RING_SLOTS just-before-a-drop
	// with big sws_ms values to confirm the GOP-burst hypothesis.
	fprintf(stderr, "[md-diag:enq] fi=%llu pts=%.3f sws_ms=%.2f depth=%llu\n",
	        (unsigned long long)_diag_idx, pts, _sws_ms,
	        (unsigned long long)_diag_depth);
#endif
	slot->pts = pts;
	m->vwrite.store(w + 1, std::memory_order_release);
	return 1;
}

// Blocking wrapper — waits for a ring slot, then enqueues. Used only by the
// EOF drain tail (drain_decoders), where the handful of remaining frames can
// safely block on present; the steady-state loop uses try_enqueue_video.
// Returns false if interrupted (quit/seek) or on fatal.
bool enqueue_video(nx_media *m, AVFrame *frame, double pts) {
	for (;;) {
		int r = try_enqueue_video(m, frame, pts);
		if (r == 1)
			return true;
		if (r < 0)
			return false;
		if (!decode_wait(m))
			return false; // ring full: wait for the presenter to drain
	}
}

// Resample an audio frame to interleaved stereo f32 at the output rate and
// push it into the stream node (blocking on backpressure). Returns false if
// interrupted.
bool enqueue_audio(nx_media *m, AVFrame *frame, double pts) {
	if (!m->swr) {
		AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_STEREO;
		int ret = swr_alloc_set_opts2(
		    &m->swr, &out_layout, AV_SAMPLE_FMT_FLT,
		    (int)m->audio_out_rate, &frame->ch_layout,
		    (AVSampleFormat)frame->format, frame->sample_rate, 0, NULL);
		if (ret < 0 || swr_init(m->swr) < 0) {
			set_fatal(m, "failed to create resampler", ret);
			return false;
		}
	}
	int64_t max_out = av_rescale_rnd(
	    swr_get_delay(m->swr, frame->sample_rate) + frame->nb_samples,
	    (int64_t)m->audio_out_rate, frame->sample_rate, AV_ROUND_UP);
	m->audio_scratch.resize((size_t)max_out * 2);
	uint8_t *out_ptr = (uint8_t *)m->audio_scratch.data();
	int got = swr_convert(m->swr, &out_ptr, (int)max_out,
	                      (const uint8_t **)frame->extended_data,
	                      frame->nb_samples);
	if (got <= 0)
		return true;

	if (!m->audio_clock_valid.load(std::memory_order_relaxed)) {
		// First audio after open/seek: anchor the audio clock at this
		// frame's PTS and the node's current write position.
		m->audio_base.store(
		    m->audio_node->stream_write_pos.load(std::memory_order_relaxed),
		    std::memory_order_relaxed);
		m->audio_pts_base.store(pts, std::memory_order_relaxed);
		m->audio_clock_valid.store(true, std::memory_order_release);
	}

	// Cut #22b Stage 2: tap the resampled audio into the visualizer
	// ring. Interleaved stereo f32 → mono downmix (L+R)/2. Held under a
	// short mutex so the reader can snapshot a consistent window.
	{
		std::lock_guard<std::mutex> tap_lock(m->tap_mutex);
		const float *sp = m->audio_scratch.data();
		uint32_t wp = m->tap_write_pos;
		for (int i = 0; i < got; i++) {
			m->tap_ring[wp] = 0.5f * (sp[i * 2] + sp[i * 2 + 1]);
			wp = (wp + 1) & TAP_MASK;
		}
		m->tap_write_pos = wp;
		m->tap_written += (uint64_t)got;
	}

	const float *src = m->audio_scratch.data();
	uint32_t remaining = (uint32_t)got;
	while (remaining > 0) {
		uint32_t wrote =
		    nx_audio_stream_write(m->audio_node, src, remaining);
		src += (size_t)wrote * 2;
		remaining -= wrote;
		if (remaining > 0 && !decode_wait(m))
			return false;
	}
	return true;
}

// Receive all pending frames from a codec. `seek_drop_until` (media seconds,
// or <0) drops frames decoded while converging on a seek target. Returns
// false if interrupted.
bool receive_frames(nx_media *m, AVCodecContext *ctx, int stream_index,
                    AVFrame *frame, double seek_drop_until) {
	while (true) {
		int ret = avcodec_receive_frame(ctx, frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
			return true;
		if (ret < 0) {
			set_fatal(m, "decode error", ret);
			return false;
		}
		double pts = frame_pts_seconds(m, frame, stream_index) +
		             m->loop_offset;
		bool is_video = stream_index == m->vstream;
		// Seek convergence: drop everything strictly before the target —
		// with half a frame of slack for video so the frame *containing*
		// the target is kept.
		double drop_before = is_video
		                         ? seek_drop_until - m->vframe_dur * 0.5
		                         : seek_drop_until - 0.005;
		if (seek_drop_until >= 0 && pts < drop_before) {
			av_frame_unref(frame);
			continue;
		}
		bool ok = is_video ? enqueue_video(m, frame, pts)
		                   : enqueue_audio(m, frame, pts);
		if (is_video && ok && m->seeking.load(std::memory_order_relaxed) &&
		    seek_drop_until >= 0) {
			// First on-target frame after a seek: the seek is complete.
			m->seeking.store(false, std::memory_order_release);
		}
		av_frame_unref(frame);
		if (!ok)
			return false;
	}
}

// Drain both decoders at EOF. Returns false if interrupted.
bool drain_decoders(nx_media *m, AVFrame *frame, double seek_drop_until) {
	if (m->vctx) {
		avcodec_send_packet(m->vctx, NULL);
		if (!receive_frames(m, m->vctx, m->vstream, frame, seek_drop_until))
			return false;
	}
	if (m->actx && m->audio_node) {
		avcodec_send_packet(m->actx, NULL);
		if (!receive_frames(m, m->actx, m->astream, frame, seek_drop_until))
			return false;
	}
	return true;
}

// Flush codec + demuxer state and seek the container. Caller is the decode
// thread. `to_seconds` is in the un-looped media domain.
bool container_seek(nx_media *m, double to_seconds) {
	if (m->vctx)
		avcodec_flush_buffers(m->vctx);
	if (m->actx)
		avcodec_flush_buffers(m->actx);
	if (m->swr) {
		swr_free(&m->swr); // drop resampler delay state
	}
	int64_t ts = (int64_t)(to_seconds * AV_TIME_BASE);
	int ret = av_seek_frame(m->fmt, -1, ts, AVSEEK_FLAG_BACKWARD);
	if (ret < 0) {
		// Fall back to a byte-position rewind for streams without an index.
		ret = av_seek_frame(m->fmt, -1, 0,
		                    AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_BYTE);
	}
	return ret >= 0;
}

// Handle a user-requested seek (decode thread).
void do_seek(nx_media *m, double *seek_drop_until) {
	double target = m->seek_target.load();
	// Discard everything queued: the main thread is guaranteed not to touch
	// the video ring while `seeking` is true.
	m->vread.store(0, std::memory_order_relaxed);
	m->vwrite.store(0, std::memory_order_relaxed);
	if (m->audio_node) {
		nx_audio_stream_flush(m->audio_node);
		m->audio_clock_valid.store(false, std::memory_order_relaxed);
	}
	m->loop_offset = 0;
	m->eof.store(false);
	if (!container_seek(m, target)) {
		set_fatal(m, "seek failed", 0);
		m->seeking.store(false);
		return;
	}
	*seek_drop_until = target;
	m->present_force.store(true);
	// Cut #22c (2026-07-10): reset the audio-stall tracker so the first
	// post-anchor clock_now() call sees `consumed != audio_consumed_last`
	// and re-engages the audio-slave clock immediately (rather than
	// starting one tick in "stalled" mode).
	m->audio_consumed_last = 0;
	// Ledger #115: reset the monotonic gate to the seek target so the
	// first post-seek clock_now can validly return audio-anchored PTS
	// values around the target (which are lower than pre-seek `t`).
	m->last_clock = target;
	if (m->vstream < 0) {
		// No video track: nothing will flip `seeking` in receive_frames.
		m->seeking.store(false);
	}
}

// Demux-ahead queue bounds (2026-09-06). Generous so a transient stall in one
// stream (e.g. a briefly-full video ring) doesn't starve the reader of the
// other stream: while the video ring backs up, the reader keeps pulling
// packets (up to VPKTQ_MAX ~4 s of video) and routing audio into apktq so
// audio never underruns. The audio queue is larger in count than the video
// queue's time-equivalent so it never gates the reader before vpktq does.
constexpr size_t VPKTQ_MAX = 120;
constexpr size_t APKTQ_MAX = 400;

// Free all queued packets + any held video frame. Called on seek (before
// do_seek) and at thread exit so no compressed data or half-decoded frame
// survives a flush.
void flush_pktqs(nx_media *m) {
	while (!m->vpktq.empty()) {
		AVPacket *p = m->vpktq.front();
		m->vpktq.pop_front();
		av_packet_free(&p);
	}
	while (!m->apktq.empty()) {
		AVPacket *p = m->apktq.front();
		m->apktq.pop_front();
		av_packet_free(&p);
	}
	if (m->v_pending_valid) {
		av_frame_unref(m->v_pending);
		m->v_pending_valid = false;
	}
	m->demux_eof = false;
}

// One non-blocking video decode step. `scratch` is the thread's reusable
// AVFrame. Returns true if it made any progress (sent a packet, produced or
// enqueued a frame, dropped a pre-seek frame). Never blocks on the video ring:
// a decoded frame that doesn't fit is stashed in v_pending and retried later.
bool pump_video(nx_media *m, AVFrame *scratch, double *seek_drop_until) {
	if (m->vstream < 0 || !m->vctx)
		return false;
	// Retry a previously-rejected frame first.
	if (m->v_pending_valid) {
		int r = try_enqueue_video(m, m->v_pending, m->v_pending_pts);
		if (r <= 0)
			return false; // ring still full (0) or fatal (-1)
		if (m->seeking.load(std::memory_order_relaxed) &&
		    *seek_drop_until >= 0)
			m->seeking.store(false, std::memory_order_release);
		av_frame_unref(m->v_pending);
		m->v_pending_valid = false;
		return true;
	}
	// Don't decode more while the ring is full (would have nowhere to go).
	if (m->vwrite.load(std::memory_order_relaxed) -
	        m->vread.load(std::memory_order_acquire) >=
	    RING_SLOTS)
		return false;
	int ret = avcodec_receive_frame(m->vctx, scratch);
	if (ret == AVERROR(EAGAIN)) {
		if (m->vpktq.empty())
			return false; // need the reader to supply packets
		AVPacket *pkt = m->vpktq.front();
		m->vpktq.pop_front();
		avcodec_send_packet(m->vctx, pkt); // succeeds: receive just drained
		av_packet_free(&pkt);
		return true;
	}
	if (ret == AVERROR_EOF)
		return false;
	if (ret < 0) {
		set_fatal(m, "decode error", ret);
		return false;
	}
	double pts = frame_pts_seconds(m, scratch, m->vstream) + m->loop_offset;
	if (*seek_drop_until >= 0 && pts < *seek_drop_until - m->vframe_dur * 0.5) {
		av_frame_unref(scratch); // pre-target frame during seek convergence
		return true;
	}
	int r = try_enqueue_video(m, scratch, pts);
	if (r == 1) {
		if (m->seeking.load(std::memory_order_relaxed) &&
		    *seek_drop_until >= 0)
			m->seeking.store(false, std::memory_order_release);
		av_frame_unref(scratch);
	} else if (r == 0) {
		// Ring filled between the check above and now: hold the frame.
		av_frame_move_ref(m->v_pending, scratch);
		m->v_pending_valid = true;
		m->v_pending_pts = pts;
	} else {
		av_frame_unref(scratch); // fatal
	}
	return true;
}

// One audio decode step. Only decodes while the audio ring is under a small
// buffer target, so it keeps audio fed without blocking or running so far
// ahead that it stalls the reader (and thereby video). Returns true on
// progress.
bool pump_audio(nx_media *m, AVFrame *scratch, double *seek_drop_until) {
	if (m->astream < 0 || !m->actx || !m->audio_node)
		return false;
	// Keep ~0.25 s buffered in the audio ring; above that, yield to video/read.
	uint32_t target = (uint32_t)(0.25 * m->audio_out_rate);
	if (nx_audio_stream_pending(m->audio_node) >= target)
		return false;
	int ret = avcodec_receive_frame(m->actx, scratch);
	if (ret == AVERROR(EAGAIN)) {
		if (m->apktq.empty())
			return false;
		AVPacket *pkt = m->apktq.front();
		m->apktq.pop_front();
		avcodec_send_packet(m->actx, pkt);
		av_packet_free(&pkt);
		return true;
	}
	if (ret == AVERROR_EOF)
		return false;
	if (ret < 0) {
		set_fatal(m, "decode error", ret);
		return false;
	}
	double pts = frame_pts_seconds(m, scratch, m->astream) + m->loop_offset;
	if (*seek_drop_until >= 0 && pts < *seek_drop_until - 0.005) {
		av_frame_unref(scratch);
		return true;
	}
	// With ~0.25 s of headroom checked above, enqueue_audio won't block here.
	bool ok = enqueue_audio(m, scratch, pts);
	av_frame_unref(scratch);
	return ok;
}

void decode_thread_main(nx_media *m) {
	AVPacket *pkt = av_packet_alloc();
	AVFrame *frame = av_frame_alloc();
	m->v_pending = av_frame_alloc();
	if (!pkt || !frame || !m->v_pending) {
		set_fatal(m, "out of memory", 0);
		av_packet_free(&pkt);
		av_frame_free(&frame);
		av_frame_free(&m->v_pending);
		return;
	}
	double seek_drop_until = -1;

	while (!m->quit.load()) {
		if (m->seek_requested.exchange(false)) {
			flush_pktqs(m); // drop queued packets + pending frame first
			do_seek(m, &seek_drop_until);
			continue;
		}
		if (m->fatal.load() || m->eof.load()) {
			// Parked: wait for a command (seek/quit).
			std::unique_lock<std::mutex> lock(m->ctl_mutex);
			m->ctl_cv.wait_for(lock, std::chrono::milliseconds(100));
			continue;
		}

		bool progress = false;

		// 1. Refill the packet queues (until one is full or the demuxer ends).
		// Cap the burst so decoding interleaves with reading — otherwise a big
		// refill (startup / post-seek, queues empty) would block the pumps and
		// leave the rings empty longer than necessary.
		int reads = 0;
		while (!m->demux_eof && m->vpktq.size() < VPKTQ_MAX &&
		       m->apktq.size() < APKTQ_MAX && reads++ < 16) {
			int ret = av_read_frame(m->fmt, pkt);
			if (ret == AVERROR_EOF) {
				m->demux_eof = true;
				break;
			}
			if (ret < 0) {
				set_fatal(m, "demux error", ret);
				break;
			}
			if (pkt->stream_index == m->vstream && m->vctx) {
				AVPacket *q = av_packet_alloc();
				av_packet_move_ref(q, pkt);
				m->vpktq.push_back(q);
				progress = true;
			} else if (pkt->stream_index == m->astream && m->actx &&
			           m->audio_node) {
				AVPacket *q = av_packet_alloc();
				av_packet_move_ref(q, pkt);
				m->apktq.push_back(q);
				progress = true;
			} else {
				av_packet_unref(pkt); // other stream (subtitles, etc.)
			}
		}

		// 2. Decode. Audio first (never let it starve), then video.
		if (pump_audio(m, frame, &seek_drop_until))
			progress = true;
		if (pump_video(m, frame, &seek_drop_until))
			progress = true;

		// Seek converged (first on-target frame enqueued, or audio-only): stop
		// dropping.
		if (seek_drop_until >= 0 && !m->seeking.load(std::memory_order_relaxed))
			seek_drop_until = -1;

		// 3. Finalize at end of stream: reader done, queues empty, no held
		// frame — flush the decoders for their last buffered frames, then loop
		// or park.
		if (m->demux_eof && m->vpktq.empty() && m->apktq.empty() &&
		    !m->v_pending_valid) {
			if (!drain_decoders(m, frame, seek_drop_until))
				continue; // interrupted by quit/seek
			seek_drop_until = -1;
			if (m->looping.load()) {
				double advance = m->duration > 0 ? m->duration : 0;
				if (!container_seek(m, 0)) {
					set_fatal(m, "loop seek failed", 0);
					continue;
				}
				m->loop_offset += advance;
				m->demux_eof = false; // resume reading from the top
				continue;
			}
			m->eof.store(true); // top-of-loop park catches it next iteration
			continue;
		}

		// 4. Nothing advanced (rings full + queues full, waiting on the
		// presenter/audio to drain): brief sleep to avoid a busy-spin.
		if (!progress)
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
	}

	flush_pktqs(m);
	av_frame_free(&m->v_pending);
	av_packet_free(&pkt);
	av_frame_free(&frame);
}

// ---------------------------------------------------------------------------
// Clock (main thread)
// ---------------------------------------------------------------------------

double clock_now(nx_media *m) {
	// Ledger #113 (Fix B, 2026-07-10): video-follows-audio when audio is
	// advancing.
	// Ledger #115 (2026-07-10): + monotonic gate + wall-cap during stall.
	// Diagnostic runs on Citron post-#113 showed periodic audio ring
	// underruns (~1 Hz) — during each ~150 ms underrun, `consumed` was
	// frozen, wall-fallback extrapolated `t` forward, then on resume the
	// fresh `at` was ~150 ms behind the extrapolated `t` → visible
	// frame regression. Two-part cure:
	//   (a) Monotonic gate: `t` is clamped to `>= last_clock`; when audio
	//       resumes below the wall-extrapolated position, video PAUSES on
	//       the last shown frame instead of regressing. Pauses are much
	//       less perceptible than backward jumps.
	//   (b) Wall-cap during stall: extrapolation is bounded by
	//       `ring_newest_pts + vframe_dur`. Present() still drains the
	//       ring (cut #22c deadlock contract preserved), but wall can't
	//       race far past what's actually decoded — so the resume-catch-up
	//       gap stays small.
	// Companion #115 fix: RING_SLOTS 3 → 12 (bigger video buffer prevents
	// the underrun from happening as often in the first place).
	double computed;
	if (m->clock_running && m->audio_node &&
	    m->audio_clock_valid.load(std::memory_order_acquire)) {
		uint64_t consumed = nx_audio_stream_consumed(m->audio_node);
		uint64_t base = m->audio_base.load(std::memory_order_relaxed);
		bool audio_advancing = consumed != m->audio_consumed_last;
		m->audio_consumed_last = consumed;
		if (consumed > base && audio_advancing) {
			double at = m->audio_pts_base.load(std::memory_order_relaxed) +
			            (double)(consumed - base) / m->audio_out_rate;
			m->clock_base = at;
			m->clock_anchor = std::chrono::steady_clock::now();
			computed = at;
		} else {
			// Stall path: wall-extrapolate from last `at` but cap at
			// ring_newest_pts + one frame so we don't race far ahead.
			double t_wall = m->clock_base +
			                std::chrono::duration<double>(
			                    std::chrono::steady_clock::now() -
			                    m->clock_anchor)
			                    .count();
			uint64_t w = m->vwrite.load(std::memory_order_acquire);
			uint64_t r = m->vread.load(std::memory_order_relaxed);
			if (w > r) {
				double newest_pts =
				    m->slots[(w - 1) % RING_SLOTS].pts;
				double cap = newest_pts + m->vframe_dur;
				if (t_wall > cap) t_wall = cap;
			}
			computed = t_wall;
		}
	} else {
		// Pure wall clock: no audio, video-only, or pre-first-audio-frame.
		computed = m->clock_base;
		if (m->clock_running) {
			computed += std::chrono::duration<double>(
			                std::chrono::steady_clock::now() -
			                m->clock_anchor)
			                .count();
		}
	}
	// Monotonic gate: never regress. Even a small backward step causes a
	// visible frame regression; a small forward pause is much less
	// perceptible.
	if (computed < m->last_clock) computed = m->last_clock;
	m->last_clock = computed;
	return computed;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// http(s):// sources are opened by libavformat itself — its http/https
// protocols + hls demuxer fetch the playlist and the rolling media segments
// over TLS (the libnx backend baked into switch-ffmpeg as tls_libnx), so there
// is NO fopen and NO custom AVIO wrap. Every other scheme (sdmc:/, romfs:/,
// disk paths, and in-memory buffers) still goes through fopen/memory +
// avio_read_cb, because FFmpeg's protocol layer doesn't understand those.
// Restored from the QuickJS-era video.c network branch that the V8 migration
// dropped when it adopted upstream nx.js's local-only media pipeline.
static bool nx_is_network_url(const char *path) {
	if (!path) return false;
	return strncmp(path, "http://", 7) == 0 ||
	       strncmp(path, "https://", 8) == 0;
}

nx_media_t *nx_media_open(const char *path, const uint8_t *mem,
                          size_t mem_size, std::shared_ptr<void> keepalive,
                          char *errbuf, size_t errbuf_size, bool want_yuv) {
	nx_media *m = new nx_media();
	m->mem_hold = std::move(keepalive);
	m->out_yuv = want_yuv;
	int ret = 0;
	const AVCodec *vcodec = NULL;
	const AVCodec *acodec = NULL;
	unsigned char *avio_buf = NULL;
	AVDictionary *net_opts = NULL; // declared here so the fail: gotos don't
	                               // cross its initialization

	const bool is_network = nx_is_network_url(path);

	if (is_network) {
		// Network (http/https): libavformat owns the IO and, for HLS, its
		// hls demuxer fetches the rolling media segments itself. No fopen,
		// no custom AVIO — pass the URL straight to avformat_open_input.
		m->fmt = avformat_alloc_context();
		if (!m->fmt) {
			snprintf(errbuf, errbuf_size, "out of memory");
			goto fail;
		}
		// Bound blocking network reads/writes. The open now runs on a worker
		// thread (video-decoder.cc), and a close() may join it — without a
		// timeout a hung/slow server could block the join (and any read
		// during playback) indefinitely. 30 s (in microseconds) is generous
		// for a LAN transcode while still guaranteeing forward progress.
		av_dict_set(&net_opts, "rw_timeout", "30000000", 0);
		// Protocol allow-list. libavformat will otherwise open ANY compiled-in
		// protocol that a playlist names, and a playlist is attacker-shaped
		// content: an HLS/DASH manifest fetched from a permitted origin can
		// point a segment at `file:`, `concat:` or `subfile:` and have the
		// demuxer read local files on its behalf — the console's own SD card,
		// including other apps' data, all below the runtime's path sandbox
		// because none of it goes through `Switch.readFile`.
		//
		// Restricting to the network transports the HLS/DASH path actually
		// needs turns that into "Protocol not found". `crypto` is required for
		// AES-128 encrypted HLS segments; `hls`/`dash`/`http`/`https`/`tcp`/
		// `tls` are the demuxers and transports themselves.
		//
		// NOT closed by this: a permitted origin can still 30x the decoder to
		// another HOST. libavformat follows redirects inside http.c with no
		// AVOption to veto them (this build exposes neither `follow_redirects`
		// nor `max_redirects`), so a per-host check would need an FFmpeg patch.
		// The whitelist at least keeps a redirect on http/https.
		av_dict_set(&net_opts, "protocol_whitelist",
		            "http,https,tcp,tls,crypto,hls,dash", 0);
		ret = avformat_open_input(&m->fmt, path, NULL, &net_opts);
		av_dict_free(&net_opts);
		if (ret < 0) {
			// Surface the real AVERROR (DNS / TLS handshake / HTTP 4xx /
			// "Protocol not found") — network opens fail many distinct ways.
			av_strerror(ret, errbuf, errbuf_size);
			goto fail;
		}
	} else {
		if (path) {
			m->file = fopen(path, "rb");
			if (!m->file) {
				snprintf(errbuf, errbuf_size, "failed to open file");
				goto fail;
			}
			fseek(m->file, 0, SEEK_END);
			m->file_size = (int64_t)ftell(m->file);
			fseek(m->file, 0, SEEK_SET);
		} else {
			m->mem = mem;
			m->mem_size = mem_size;
		}

		avio_buf = (unsigned char *)av_malloc(65536);
		if (!avio_buf) {
			snprintf(errbuf, errbuf_size, "out of memory");
			goto fail;
		}
		m->avio = avio_alloc_context(avio_buf, 65536, 0, m, avio_read_cb, NULL,
		                             avio_seek_cb);
		if (!m->avio) {
			av_free(avio_buf);
			snprintf(errbuf, errbuf_size, "out of memory");
			goto fail;
		}
		m->fmt = avformat_alloc_context();
		if (!m->fmt) {
			snprintf(errbuf, errbuf_size, "out of memory");
			goto fail;
		}
		m->fmt->pb = m->avio;

		ret = avformat_open_input(&m->fmt, NULL, NULL, NULL);
		if (ret < 0) {
			av_strerror(ret, errbuf, errbuf_size);
			goto fail;
		}
	}
	ret = avformat_find_stream_info(m->fmt, NULL);
	if (ret < 0) {
		av_strerror(ret, errbuf, errbuf_size);
		goto fail;
	}

	m->vstream = av_find_best_stream(m->fmt, AVMEDIA_TYPE_VIDEO, -1, -1,
	                                 &vcodec, 0);
	m->astream = av_find_best_stream(m->fmt, AVMEDIA_TYPE_AUDIO, -1, -1,
	                                 &acodec, 0);
	if (m->vstream < 0 && m->astream < 0) {
		snprintf(errbuf, errbuf_size, "no playable streams");
		goto fail;
	}

	if (m->vstream >= 0) {
		AVCodecParameters *par = m->fmt->streams[m->vstream]->codecpar;
		m->vctx = avcodec_alloc_context3(vcodec);
		if (!m->vctx ||
		    avcodec_parameters_to_context(m->vctx, par) < 0) {
			snprintf(errbuf, errbuf_size, "failed to set up video decoder");
			goto fail;
		}
		// Two decode threads: leaves the main JS thread + audio render
		// thread breathing room on the Switch's three usable cores.
		m->vctx->thread_count = 2;
		ret = avcodec_open2(m->vctx, vcodec, NULL);
		if (ret < 0) {
			av_strerror(ret, errbuf, errbuf_size);
			goto fail;
		}
		m->width = par->width;
		m->height = par->height;
		AVRational fr = m->fmt->streams[m->vstream]->avg_frame_rate;
		if (fr.num > 0 && fr.den > 0)
			m->vframe_dur = (double)fr.den / fr.num;
		if (m->width <= 0 || m->height <= 0 || m->width > 4096 ||
		    m->height > 4096) {
			snprintf(errbuf, errbuf_size, "unsupported video dimensions");
			goto fail;
		}
		// Ring slot size depends on the output format: I420 is 1.5 B/px
		// (Y|U|V contiguous), BGRA is 4 B/px. The field is named `bgra` for
		// historical reasons; under out_yuv it holds planar I420.
		const size_t slot_bytes =
		    m->out_yuv ? nx_media_i420_size(m->width, m->height)
		               : (size_t)m->width * m->height * 4;
		for (int i = 0; i < RING_SLOTS; i++) {
			m->slots[i].bgra = (uint8_t *)malloc(slot_bytes);
			if (!m->slots[i].bgra) {
				snprintf(errbuf, errbuf_size, "out of memory");
				goto fail;
			}
		}
	}

	if (m->astream >= 0) {
		AVCodecParameters *par = m->fmt->streams[m->astream]->codecpar;
		m->actx = avcodec_alloc_context3(acodec);
		if (!m->actx ||
		    avcodec_parameters_to_context(m->actx, par) < 0 ||
		    avcodec_open2(m->actx, acodec, NULL) < 0) {
			// Audio is non-fatal: play the video silently.
			if (m->actx)
				avcodec_free_context(&m->actx);
			m->astream = -1;
		}
	}

	if (m->fmt->duration != AV_NOPTS_VALUE && m->fmt->duration > 0)
		m->duration = (double)m->fmt->duration / AV_TIME_BASE;

#if MEDIA_DIAG_112
	// One-time open summary. For the 60fps-feasibility probe: content_fps tells
	// us what the device receives (Source direct-play => ~60), and cross-checked
	// against [md-diag:prs] ring depth it reveals whether SW decode sustains it
	// (depth full ~11-12 + no_frame=0 => decode keeps up at the present's
	// ~60 ring-frames/s consumption; draining => decode-bound => slow motion).
	fprintf(stderr,
	        "[md-diag:open] %dx%d content_fps=%.3f vframe_dur=%.4f dur=%.2f "
	        "out_yuv=%d has_audio=%d\n",
	        m->width, m->height,
	        m->vframe_dur > 0 ? 1.0 / m->vframe_dur : -1.0, m->vframe_dur,
	        m->duration, m->out_yuv ? 1 : 0, m->astream >= 0 ? 1 : 0);
	fflush(stderr);
#endif

	m->thread = std::thread(decode_thread_main, m);
	return m;

fail:
	nx_media_destroy(m);
	return NULL;
}

int nx_media_width(nx_media_t *m) { return m->width; }
int nx_media_height(nx_media_t *m) { return m->height; }
double nx_media_duration(nx_media_t *m) { return m->duration; }
// Content frame rate (from the stream's avg_frame_rate; see vframe_dur set at
// open). Drives the shell's swap-interval decision: >~32fps content wants a
// 60 Hz present, not the 30 Hz pacing lock (which would halve it to 30 shown).
double nx_media_content_fps(nx_media_t *m) {
	return m->vframe_dur > 0 ? 1.0 / m->vframe_dur : 0.0;
}
bool nx_media_has_audio(nx_media_t *m) { return m->astream >= 0; }
bool nx_media_has_video(nx_media_t *m) { return m->vstream >= 0; }
bool nx_media_is_yuv(nx_media_t *m) { return m->out_yuv; }
int nx_media_yuv_colorspace(nx_media_t *m) { return m->yuv_cs; }

void nx_media_set_audio_node(nx_media_t *m, nx_audio_node *node,
                             double sample_rate) {
	m->audio_node = node;
	m->audio_out_rate = sample_rate;
}

void nx_media_play(nx_media_t *m) {
	if (m->playing.load())
		return;
	m->clock_anchor = std::chrono::steady_clock::now();
	m->clock_running = true;
	m->playing.store(true);
	if (m->audio_node)
		nx_audio_stream_set_playing(m->audio_node, true);
	m->ctl_cv.notify_all();
}

void nx_media_pause(nx_media_t *m) {
	if (!m->playing.load())
		return;
	m->clock_base = clock_now(m);
	m->clock_running = false;
	m->playing.store(false);
	if (m->audio_node)
		nx_audio_stream_set_playing(m->audio_node, false);
}

void nx_media_seek(nx_media_t *m, double seconds) {
	if (seconds < 0)
		seconds = 0;
	if (m->duration > 0 && seconds > m->duration)
		seconds = m->duration;
	// Ledger #115: reset monotonic gate to the seek target immediately
	// (main thread) so between seek() and do_seek() (decode thread)
	// nx_media_current_time queries return sane values. do_seek also
	// updates last_clock, redundant but idempotent.
	m->last_clock = seconds;
	// Stop presentation/clock reads of the ring first.
	m->seeking.store(true, std::memory_order_release);
	m->clock_base = seconds;
	m->clock_anchor = std::chrono::steady_clock::now();
	m->seek_target.store(seconds);
	m->seek_requested.store(true);
	m->ctl_cv.notify_all();
}

void nx_media_set_loop(nx_media_t *m, bool loop) {
	m->looping.store(loop);
	m->ctl_cv.notify_all();
}

bool nx_media_present(nx_media_t *m, uint8_t **buffer_inout) {
	if (m->seeking.load(std::memory_order_acquire) || m->fatal.load())
		return false;
	double t = clock_now(m);
	uint64_t r = m->vread.load(std::memory_order_relaxed);
	uint64_t w = m->vwrite.load(std::memory_order_acquire);
	int64_t candidate = -1;
	for (uint64_t i = r; i < w; i++) {
		if (m->slots[i % RING_SLOTS].pts <= t + PRESENT_EPSILON)
			candidate = (int64_t)i;
		else
			break;
	}
#if MEDIA_DIAG_112
	// Aggregate over 60 present() calls (~3 s at 20 fps rAF). Extended
	// (ledger #114) with audio-side snapshots to pin the residual jitter
	// after Fix B (#113):
	//   depth_min/max/avg  — video ring occupancy
	//   no_frame / skip    — as before
	//   aur_pending_min/max — audio ring depth (frames of buffered audio)
	//                         over the window
	//   aur_underruns      — count of render-quantum underrun events since
	//                         last window (audio ring emptied → consumer
	//                         emits silence → `consumed` stalls)
	//   t_min/max          — clock_now returns over the window
	//   ring_pts_min/max   — video ring PTS range at present() calls
	static uint32_t _diag_prs_calls = 0;
	static uint64_t _diag_prs_depth_sum = 0;
	static uint32_t _diag_prs_depth_min = 0xFFFFFFFFu;
	static uint32_t _diag_prs_depth_max = 0;
	static uint32_t _diag_prs_no_frame = 0;
	static uint32_t _diag_prs_skip = 0;
	static uint32_t _diag_aur_pending_min = 0xFFFFFFFFu;
	static uint32_t _diag_aur_pending_max = 0;
	static uint64_t _diag_aur_underrun_prev = 0;
	static double _diag_t_min = 1e18;
	static double _diag_t_max = -1e18;
	static double _diag_ring_oldest_min = 1e18;
	static double _diag_ring_newest_max = -1e18;
	uint32_t _diag_depth = (uint32_t)(w - r);
	_diag_prs_calls++;
	_diag_prs_depth_sum += _diag_depth;
	if (_diag_depth < _diag_prs_depth_min) _diag_prs_depth_min = _diag_depth;
	if (_diag_depth > _diag_prs_depth_max) _diag_prs_depth_max = _diag_depth;
	if (candidate < 0) _diag_prs_no_frame++;
	else if (candidate > (int64_t)r) _diag_prs_skip++;
	if (t < _diag_t_min) _diag_t_min = t;
	if (t > _diag_t_max) _diag_t_max = t;
	if (_diag_depth > 0) {
		double _rop = m->slots[r % RING_SLOTS].pts;
		double _rnp = m->slots[(w - 1) % RING_SLOTS].pts;
		if (_rop < _diag_ring_oldest_min) _diag_ring_oldest_min = _rop;
		if (_rnp > _diag_ring_newest_max) _diag_ring_newest_max = _rnp;
	}
	if (m->audio_node) {
		uint32_t _pending = nx_audio_stream_pending(m->audio_node);
		if (_pending < _diag_aur_pending_min) _diag_aur_pending_min = _pending;
		if (_pending > _diag_aur_pending_max) _diag_aur_pending_max = _pending;
	}
	if (_diag_prs_calls >= 60) {
		double _avg = (double)_diag_prs_depth_sum / _diag_prs_calls;
		uint64_t _aur_underrun_now =
		    m->audio_node ? nx_audio_stream_underrun_count(m->audio_node) : 0;
		uint64_t _aur_underrun_delta = _aur_underrun_now - _diag_aur_underrun_prev;
		_diag_aur_underrun_prev = _aur_underrun_now;
		fprintf(stderr,
		        "[md-diag:prs] n=%u depth_min=%u max=%u avg=%.2f "
		        "no_frame=%u skip=%u aur_pending_min=%u max=%u "
		        "aur_underruns=%llu t_min=%.3f t_max=%.3f "
		        "ring_oldest_min=%.3f ring_newest_max=%.3f\n",
		        _diag_prs_calls, _diag_prs_depth_min, _diag_prs_depth_max,
		        _avg, _diag_prs_no_frame, _diag_prs_skip,
		        _diag_aur_pending_min, _diag_aur_pending_max,
		        (unsigned long long)_aur_underrun_delta,
		        _diag_t_min, _diag_t_max,
		        _diag_ring_oldest_min, _diag_ring_newest_max);
		_diag_prs_calls = 0;
		_diag_prs_depth_sum = 0;
		_diag_prs_depth_min = 0xFFFFFFFFu;
		_diag_prs_depth_max = 0;
		_diag_prs_no_frame = 0;
		_diag_prs_skip = 0;
		_diag_aur_pending_min = 0xFFFFFFFFu;
		_diag_aur_pending_max = 0;
		_diag_t_min = 1e18;
		_diag_t_max = -1e18;
		_diag_ring_oldest_min = 1e18;
		_diag_ring_newest_max = -1e18;
	}
	// Per-call verbose trace for the first 100 present() calls (~5s at
	// 20fps rAF). Lets us see the per-tick relationship of t / ring PTS /
	// candidate / audio ring level during startup.
	static uint32_t _diag_vpr_left = 100;
	if (_diag_vpr_left > 0) {
		_diag_vpr_left--;
		double _rop = _diag_depth > 0 ? m->slots[r % RING_SLOTS].pts : -1.0;
		double _rnp = _diag_depth > 0 ? m->slots[(w - 1) % RING_SLOTS].pts
		                              : -1.0;
		uint32_t _apend = m->audio_node
		                      ? nx_audio_stream_pending(m->audio_node)
		                      : 0;
		fprintf(stderr,
		        "[md-diag:vpr] t=%.4f dep=%u rop=%.3f rnp=%.3f cand=%lld "
		        "apend=%u\n",
		        t, _diag_depth, _rop, _rnp, (long long)candidate, _apend);
	}
#endif
	if (candidate < 0 && w > r &&
	    m->present_force.exchange(false, std::memory_order_acq_rel)) {
		// First frame after a seek: show it even if its PTS is just past
		// the paused clock (it is the frame nearest the seek target).
		candidate = (int64_t)r;
	}
	if (candidate < 0)
		return false;
	m->present_force.store(false, std::memory_order_relaxed);
	video_slot *slot = &m->slots[candidate % RING_SLOTS];
	uint8_t *prev = *buffer_inout;
	*buffer_inout = slot->bgra;
	slot->bgra = prev;
	m->presented_frames++;
	m->dropped_frames += (uint64_t)candidate - r;
	m->vread.store((uint64_t)candidate + 1, std::memory_order_release);
	return true;
}

uint64_t nx_media_presented_frames(nx_media_t *m) {
	return m->presented_frames;
}

uint64_t nx_media_dropped_frames(nx_media_t *m) { return m->dropped_frames; }

double nx_media_current_time(nx_media_t *m) {
	if (m->seeking.load())
		return m->seek_target.load();
	double t = clock_now(m);
	if (m->duration > 0) {
		if (m->looping.load())
			t = fmod(t, m->duration);
		else if (t > m->duration)
			t = m->duration;
	}
	return t < 0 ? 0 : t;
}

uint32_t nx_media_buffered_frames(nx_media_t *m) {
	return (uint32_t)(m->vwrite.load(std::memory_order_acquire) -
	                  m->vread.load(std::memory_order_relaxed));
}

bool nx_media_ended(nx_media_t *m) {
	if (!m->eof.load() || m->looping.load())
		return false;
	if (m->vwrite.load(std::memory_order_acquire) !=
	    m->vread.load(std::memory_order_relaxed))
		return false;
	if (m->audio_node && m->audio_clock_valid.load()) {
		uint64_t written =
		    m->audio_node->stream_write_pos.load(std::memory_order_acquire);
		if (nx_audio_stream_consumed(m->audio_node) < written)
			return false;
	}
	return true;
}

bool nx_media_seeking(nx_media_t *m) { return m->seeking.load(); }

const char *nx_media_error(nx_media_t *m) {
	return m->fatal.load() ? m->error_buf : NULL;
}

// ---------------------------------------------------------------------------
// Whole-file audio decode (decodeAudioData / Audio element)
// ---------------------------------------------------------------------------

namespace {

struct mem_reader {
	const uint8_t *data;
	size_t size;
	size_t pos;
};

int mem_read_cb(void *opaque, uint8_t *buf, int n) {
	mem_reader *r = static_cast<mem_reader *>(opaque);
	size_t rem = r->size - r->pos;
	if (rem == 0)
		return AVERROR_EOF;
	size_t count = rem < (size_t)n ? rem : (size_t)n;
	memcpy(buf, r->data + r->pos, count);
	r->pos += count;
	return (int)count;
}

int64_t mem_seek_cb(void *opaque, int64_t offset, int whence) {
	mem_reader *r = static_cast<mem_reader *>(opaque);
	if (whence & AVSEEK_SIZE)
		return (int64_t)r->size;
	whence &= ~AVSEEK_FORCE;
	int64_t base = whence == SEEK_CUR    ? (int64_t)r->pos
	               : whence == SEEK_END ? (int64_t)r->size
	                                    : 0;
	int64_t pos = base + offset;
	if (pos < 0 || pos > (int64_t)r->size)
		return -1;
	r->pos = (size_t)pos;
	return pos;
}

} // namespace

bool nx_media_decode_audio(const uint8_t *data, size_t size,
                           float *channels[NX_MEDIA_MAX_CHANNELS],
                           int *num_channels, uint32_t *length,
                           uint32_t *sample_rate, char *errbuf,
                           size_t errbuf_size) {
	mem_reader reader = {data, size, 0};
	AVIOContext *avio = NULL;
	AVFormatContext *fmt = NULL;
	AVCodecContext *ctx = NULL;
	SwrContext *swr = NULL;
	AVPacket *pkt = NULL;
	AVFrame *frame = NULL;
	const AVCodec *codec = NULL;
	int stream = -1;
	int ret = 0;
	int nch = 0;
	bool ok = false;
	// Planar accumulation buffers (resized as frames arrive).
	std::vector<std::vector<float>> acc;
	std::vector<float> scratch;

	unsigned char *avio_buf = (unsigned char *)av_malloc(65536);
	if (!avio_buf) {
		snprintf(errbuf, errbuf_size, "out of memory");
		goto done;
	}
	avio = avio_alloc_context(avio_buf, 65536, 0, &reader, mem_read_cb, NULL,
	                          mem_seek_cb);
	if (!avio) {
		av_free(avio_buf);
		snprintf(errbuf, errbuf_size, "out of memory");
		goto done;
	}
	fmt = avformat_alloc_context();
	if (!fmt) {
		snprintf(errbuf, errbuf_size, "out of memory");
		goto done;
	}
	fmt->pb = avio;
	ret = avformat_open_input(&fmt, NULL, NULL, NULL);
	if (ret < 0) {
		av_strerror(ret, errbuf, errbuf_size);
		goto done;
	}
	ret = avformat_find_stream_info(fmt, NULL);
	if (ret < 0) {
		av_strerror(ret, errbuf, errbuf_size);
		goto done;
	}
	stream = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
	if (stream < 0) {
		snprintf(errbuf, errbuf_size, "no audio stream found");
		goto done;
	}
	ctx = avcodec_alloc_context3(codec);
	if (!ctx ||
	    avcodec_parameters_to_context(ctx, fmt->streams[stream]->codecpar) <
	        0 ||
	    avcodec_open2(ctx, codec, NULL) < 0) {
		snprintf(errbuf, errbuf_size, "failed to open audio decoder");
		goto done;
	}
	pkt = av_packet_alloc();
	frame = av_frame_alloc();
	if (!pkt || !frame) {
		snprintf(errbuf, errbuf_size, "out of memory");
		goto done;
	}

	while (true) {
		ret = av_read_frame(fmt, pkt);
		bool at_eof = ret == AVERROR_EOF;
		if (ret < 0 && !at_eof) {
			av_strerror(ret, errbuf, errbuf_size);
			goto done;
		}
		if (!at_eof && pkt->stream_index != stream) {
			av_packet_unref(pkt);
			continue;
		}
		ret = avcodec_send_packet(ctx, at_eof ? NULL : pkt);
		if (!at_eof)
			av_packet_unref(pkt);
		if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
			av_strerror(ret, errbuf, errbuf_size);
			goto done;
		}
		while (true) {
			ret = avcodec_receive_frame(ctx, frame);
			if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
				break;
			if (ret < 0) {
				av_strerror(ret, errbuf, errbuf_size);
				goto done;
			}
			if (nch == 0) {
				nch = frame->ch_layout.nb_channels;
				if (nch > NX_MEDIA_MAX_CHANNELS)
					nch = NX_MEDIA_MAX_CHANNELS;
				if (nch <= 0) {
					snprintf(errbuf, errbuf_size, "no audio channels");
					goto done;
				}
				*sample_rate = (uint32_t)frame->sample_rate;
				acc.resize((size_t)nch);
				// Convert to interleaved f32 at the native rate/layout —
				// deinterleave below (swr output layout = input layout).
				ret = swr_alloc_set_opts2(
				    &swr, &frame->ch_layout, AV_SAMPLE_FMT_FLT,
				    frame->sample_rate, &frame->ch_layout,
				    (AVSampleFormat)frame->format, frame->sample_rate, 0,
				    NULL);
				if (ret < 0 || swr_init(swr) < 0) {
					snprintf(errbuf, errbuf_size,
					         "failed to create resampler");
					goto done;
				}
			}
			int src_nch = frame->ch_layout.nb_channels;
			scratch.resize((size_t)frame->nb_samples * src_nch);
			uint8_t *out_ptr = (uint8_t *)scratch.data();
			int got = swr_convert(swr, &out_ptr, frame->nb_samples,
			                      (const uint8_t **)frame->extended_data,
			                      frame->nb_samples);
			if (got > 0) {
				for (int c = 0; c < nch; c++) {
					std::vector<float> &dst = acc[(size_t)c];
					size_t off = dst.size();
					dst.resize(off + (size_t)got);
					for (int i = 0; i < got; i++)
						dst[off + i] = scratch[(size_t)i * src_nch + c];
				}
			}
			av_frame_unref(frame);
		}
		if (at_eof)
			break;
	}

	if (nch == 0 || acc[0].empty()) {
		snprintf(errbuf, errbuf_size, "audio file contains no audio data");
		goto done;
	}
	*num_channels = nch;
	*length = (uint32_t)acc[0].size();
	for (int c = 0; c < nch; c++) {
		channels[c] = (float *)malloc((size_t)*length * sizeof(float));
		if (!channels[c]) {
			for (int j = 0; j < c; j++) {
				free(channels[j]);
				channels[j] = nullptr;
			}
			snprintf(errbuf, errbuf_size, "out of memory");
			goto done;
		}
		size_t have = acc[(size_t)c].size();
		size_t want = (size_t)*length;
		memcpy(channels[c], acc[(size_t)c].data(),
		       (have < want ? have : want) * sizeof(float));
	}
	ok = true;

done:
	if (pkt)
		av_packet_free(&pkt);
	if (frame)
		av_frame_free(&frame);
	if (swr)
		swr_free(&swr);
	if (ctx)
		avcodec_free_context(&ctx);
	if (fmt)
		avformat_close_input(&fmt);
	if (avio) {
		av_freep(&avio->buffer);
		avio_context_free(&avio);
	}
	return ok;
}

// ---------------------------------------------------------------------------
// Cut #22b Stage 2: visualizer readers (waveform / spectrum / audio-levels)
// ---------------------------------------------------------------------------

namespace {

// Iterative radix-2 Cooley-Tukey FFT on TAP_LEN complex samples in-place.
// Not general-purpose — hard-coded to the tap window size so the inner
// loops can constant-fold. Called at most a few times per second (per
// spectrum read), so no need for split-radix / SIMD; correctness matters.
void fft_tap(float *re, float *im) {
	// Bit-reverse permutation.
	for (uint32_t i = 0; i < TAP_LEN; i++) {
		uint32_t j = 0;
		uint32_t k = i;
		for (int b = 0; b < TAP_LOG2; b++) {
			j = (j << 1) | (k & 1u);
			k >>= 1;
		}
		if (j > i) {
			std::swap(re[i], re[j]);
			std::swap(im[i], im[j]);
		}
	}
	// Cooley-Tukey butterflies (forward FFT — negative-angle twiddles).
	for (uint32_t size = 2; size <= TAP_LEN; size <<= 1) {
		uint32_t half = size >> 1;
		double angle_step = -2.0 * TAP_PI / (double)size;
		for (uint32_t start = 0; start < TAP_LEN; start += size) {
			for (uint32_t k = 0; k < half; k++) {
				double angle = angle_step * (double)k;
				float wr = (float)cos(angle);
				float wi = (float)sin(angle);
				uint32_t i0 = start + k;
				uint32_t i1 = i0 + half;
				float tr = wr * re[i1] - wi * im[i1];
				float ti = wr * im[i1] + wi * re[i1];
				re[i1] = re[i0] - tr;
				im[i1] = im[i0] - ti;
				re[i0] = re[i0] + tr;
				im[i0] = im[i0] + ti;
			}
		}
	}
}

// Snapshot the most-recent TAP_LEN mono samples in chronological order.
// Returns false if the tap hasn't yet accumulated a full window (e.g.
// fresh decoder). Caller-owned buffer must hold TAP_LEN floats.
bool snapshot_tap(nx_media *m, float *out) {
	std::lock_guard<std::mutex> lock(m->tap_mutex);
	if (m->tap_written < TAP_LEN) return false;
	uint32_t wp = m->tap_write_pos; // one past the newest sample
	for (uint32_t i = 0; i < TAP_LEN; i++) {
		// Oldest → newest: sample at logical index i is at ring index
		// (wp + i) mod TAP_LEN — because wp is the next write, i.e.
		// exactly TAP_LEN samples old going forward.
		uint32_t idx = (wp + i) & TAP_MASK;
		out[i] = m->tap_ring[idx];
	}
	return true;
}

} // namespace

bool nx_media_read_waveform(nx_media_t *m, float *out, uint32_t out_len) {
	if (!m || !out || out_len == 0 || out_len > TAP_LEN) return false;
	std::lock_guard<std::mutex> lock(m->tap_mutex);
	if (m->tap_written < out_len) return false;
	uint32_t wp = m->tap_write_pos;
	// The newest sample is at (wp - 1) mod TAP_LEN. Copy out_len samples
	// ending at the newest, in chronological order (oldest first).
	for (uint32_t i = 0; i < out_len; i++) {
		uint32_t idx = (wp + TAP_LEN - out_len + i) & TAP_MASK;
		out[i] = m->tap_ring[idx];
	}
	return true;
}

bool nx_media_read_spectrum(nx_media_t *m, float *out, uint32_t out_len) {
	if (!m || !out || out_len == 0) return false;
	float samples[TAP_LEN];
	if (!snapshot_tap(m, samples)) return false;
	// Hann window (reduces spectral leakage from the rectangular window
	// otherwise implied by a hard buffer boundary).
	for (uint32_t i = 0; i < TAP_LEN; i++) {
		float w = 0.5f * (1.0f - (float)cos(2.0 * TAP_PI * (double)i /
		                                    (double)(TAP_LEN - 1)));
		samples[i] *= w;
	}
	float re[TAP_LEN];
	float im[TAP_LEN] = {};
	memcpy(re, samples, sizeof(re));
	fft_tap(re, im);
	// Useful bins are 0..N/2. Convert to magnitude, normalize by (N/2)
	// so a unit-amplitude tone reads ~1.0 at its bin. Matches the
	// QuickJS-era video.c formulation.
	constexpr uint32_t USABLE = TAP_LEN / 2;
	float mag[USABLE];
	const float norm = 2.0f / (float)TAP_LEN;
	for (uint32_t i = 0; i < USABLE; i++) {
		mag[i] = sqrtf(re[i] * re[i] + im[i] * im[i]) * norm;
	}
	// Linear-scale bin-average USABLE bins into out_len output bins,
	// then flatten the natural 1/f rolloff of music (bass content
	// dominates the raw magnitude spectrum, so without compensation only
	// the first few output bins ever visibly react). The sqrt(1 + bin/K)
	// weighting boosts the tail without warping the bass response —
	// tuned so mid-range bins (~1 kHz) roughly double vs the raw
	// magnitude and treble (~10 kHz) reads ~3× stronger.
	for (uint32_t i = 0; i < out_len; i++) {
		uint32_t lo = (uint32_t)((uint64_t)i * USABLE / out_len);
		uint32_t hi = (uint32_t)((uint64_t)(i + 1) * USABLE / out_len);
		if (hi <= lo) hi = lo + 1;
		if (hi > USABLE) hi = USABLE;
		float sum = 0;
		for (uint32_t k = lo; k < hi; k++) sum += mag[k];
		float avg = sum / (float)(hi - lo);
		// Weight by ~sqrt(centre_bin) — cheap 1/f flatten.
		float centre = 0.5f * (float)(lo + hi);
		float weight = sqrtf(1.0f + centre / 6.0f);
		out[i] = avg * weight;
	}
	return true;
}

uint32_t nx_media_read_audio_levels(nx_media_t *m, float *out,
                                    uint32_t out_max) {
	if (!m || !out || out_max == 0) return 0;
	// Three bands (bass / mid / high). Reuse the spectrum path so the
	// mapping matches the visualizer's frequency reading.
	constexpr uint32_t BANDS = 3;
	const uint32_t n = out_max < BANDS ? out_max : BANDS;
	// Compute a per-band RMS of the tap window in the frequency domain
	// (Parseval). Cheaper: just RMS the raw tap samples split into
	// low-pass / mid / high-pass via a coarse Butterworth-shaped IIR —
	// but for a visualizer the FFT-band approach is more legible.
	float samples[TAP_LEN];
	if (!snapshot_tap(m, samples)) return 0;
	for (uint32_t i = 0; i < TAP_LEN; i++) {
		float w = 0.5f * (1.0f - (float)cos(2.0 * TAP_PI * (double)i /
		                                    (double)(TAP_LEN - 1)));
		samples[i] *= w;
	}
	float re[TAP_LEN];
	float im[TAP_LEN] = {};
	memcpy(re, samples, sizeof(re));
	fft_tap(re, im);
	constexpr uint32_t USABLE = TAP_LEN / 2;
	const float norm = 2.0f / (float)TAP_LEN;
	// Band edges: bass 0..~250 Hz, mid 250 Hz..~4 kHz, high 4 kHz..Nyquist.
	// At 48 kHz sample rate and TAP_LEN=2048, bin size ≈ 23.44 Hz — so
	// 10 bins ≈ 235 Hz, 170 bins ≈ 4 kHz.
	uint32_t edges[BANDS + 1] = {0, 10, 170, USABLE};
	for (uint32_t b = 0; b < n; b++) {
		uint32_t lo = edges[b];
		uint32_t hi = edges[b + 1];
		float sum = 0;
		for (uint32_t k = lo; k < hi; k++) {
			float mag = sqrtf(re[k] * re[k] + im[k] * im[k]) * norm;
			sum += mag * mag;
		}
		float rms = (hi > lo) ? sqrtf(sum / (float)(hi - lo)) : 0.0f;
		out[b] = rms;
	}
	return n;
}

void nx_media_destroy(nx_media_t *m) {
	m->quit.store(true);
	m->ctl_cv.notify_all();
	if (m->thread.joinable())
		m->thread.join();
	if (m->audio_node)
		nx_audio_stream_set_playing(m->audio_node, false);
	if (m->sws)
		sws_freeContext(m->sws);
	if (m->swr)
		swr_free(&m->swr);
	if (m->vctx)
		avcodec_free_context(&m->vctx);
	if (m->actx)
		avcodec_free_context(&m->actx);
	if (m->fmt)
		avformat_close_input(&m->fmt);
	if (m->avio) {
		av_freep(&m->avio->buffer);
		avio_context_free(&m->avio);
	}
	for (int i = 0; i < RING_SLOTS; i++)
		free(m->slots[i].bgra);
	if (m->file)
		fclose(m->file);
	delete m;
}
