#pragma once
// Portable media (video + audio) decode pipeline over ffmpeg.
//
// Compiled into BOTH the device runtime (switch-ffmpeg 7.x) and the host
// nxjs-test binary (distro ffmpeg 5.x+) — no V8, no libnx. Each open media
// owns a dedicated decode thread which demuxes the container, decodes the
// video stream into a small ring of presentation-ready BGRA frames (PTS
// stamped), and decodes/resamples the audio stream into an audio-graph
// stream-source node (see NX_AUDIO_NODE_STREAM_SOURCE in audio-graph.h).
//
// Threading contract:
//   - All functions below are called from the main (JS loop) thread, except
//     nx_media_open which may run on a libuv worker.
//   - The decode thread is internal; commands (play/pause/seek/loop/quit)
//     are delivered via atomics + a condition variable.
//   - nx_media_present() swaps the newest due frame into the caller's BGRA
//     buffer (pointer swap, zero copy) based on the media clock. The clock
//     is slaved to the audio stream node's consumed-frame counter when an
//     audio track is playing, and to the monotonic wall clock otherwise.
#include <memory>
#include <stddef.h>
#include <stdint.h>

struct nx_audio_node;

typedef struct nx_media nx_media_t;

// Channel cap for whole-file audio decoding (Web Audio allows 32).
#define NX_MEDIA_MAX_CHANNELS 32

// Decode an entire audio resource (any ffmpeg-supported container/codec)
// from memory into planar f32 channel buffers. Backs `decodeAudioData()` and
// the `Audio` element.
//
// `target_rate` is the sample rate the decoded buffer should come back at —
// pass the destination AudioContext's rate so the conversion happens ONCE
// here, through libswresample's polyphase filter, instead of every frame in
// the render loop's linear interpolator. 0 keeps the file's native rate.
//
// This matters more than it looks. The Switch's audio graph runs at 48 kHz
// and most game assets ship at 44.1 kHz, so without this the buffer-source
// node reads the buffer at r = 0.91875 with linear interpolation between
// neighbouring samples — which is fine down low but costs -1.5 dB at 10 kHz,
// -3.4 dB at 15 kHz, and folds back roughly -25 dB of imaging junk at 10 kHz
// rising to -15 dB at 15 kHz. Smooth low-passed material (a music bed) hides
// it; short bright percussive one-shots (coin blips, explosions, UI clicks)
// audibly grit and dull. Browsers resample in decodeAudioData for exactly
// this reason, and `r` then lands on 1.0 so the interpolator never engages.
//
// On success fills `channels[0..num_channels)` with malloc'd buffers (caller
// frees), `length` (frames) and `sample_rate` (the rate actually produced),
// and returns true. Blocking — call on a worker thread. On failure fills
// `errbuf` and returns false.
bool nx_media_decode_audio(const uint8_t *data, size_t size,
                           float *channels[NX_MEDIA_MAX_CHANNELS],
                           int *num_channels, uint32_t *length,
                           uint32_t *sample_rate, uint32_t target_rate,
                           char *errbuf, size_t errbuf_size);

// Open a media resource and probe its streams. Exactly one of `path` or
// `mem` must be provided; for `mem`, `keepalive` must own the buffer (e.g. a
// shared_ptr<v8::BackingStore>) — the media holds it until
// nx_media_destroy(), since the decode thread streams from it for the
// media's whole lifetime. Blocking — call off the main thread. Returns NULL
// and fills `errbuf` on failure.
// `want_yuv` (2026-09-06): when true the video ring stores planar I420
// (Y + U + V, contiguous, 1.5 bytes/px) instead of BGRA (4 bytes/px). This
// lets the caller upload the frame to the GPU as YUV planes and do YUV→RGB
// in the shader (Skia's YUVA image path) — ~2.6× less per-frame texture
// upload than BGRA, which is the dominant per-frame cost on the Switch's
// Mesa-nouveau GL (see the Jellyfin video-perf work). Default false keeps the
// documented BGRA contract for the Three.js webgl_materials_video demo and the
// legacy `<video>` element (video.cc).
nx_media_t *nx_media_open(const char *path, const uint8_t *mem,
                          size_t mem_size, std::shared_ptr<void> keepalive,
                          char *errbuf, size_t errbuf_size,
                          bool want_yuv = false);

// Metadata (valid after a successful open).
int nx_media_width(nx_media_t *m);
int nx_media_height(nx_media_t *m);
double nx_media_duration(nx_media_t *m); // seconds (0 if unknown)
double nx_media_content_fps(nx_media_t *m); // stream frame rate (0 if unknown)
bool nx_media_has_audio(nx_media_t *m);
bool nx_media_has_video(nx_media_t *m);

// True if this media's video ring is planar I420 (opened with want_yuv). The
// present buffer is then W*H + 2*((W+1)/2)*((H+1)/2) bytes, not W*H*4.
bool nx_media_is_yuv(nx_media_t *m);
// Neutral color-space tag for the I420 frames, derived from the stream:
//   0 = Rec709 limited, 1 = Rec601 limited, 2 = JPEG/full range, 3 = Rec709 full.
// The GPU draw path maps this to the matching SkYUVColorSpace. Meaningful only
// when nx_media_is_yuv() is true.
int nx_media_yuv_colorspace(nx_media_t *m);
// Size in bytes of one I420 frame for the given luma dimensions.
static inline size_t nx_media_i420_size(int w, int h) {
	size_t cw = (size_t)((w + 1) / 2), ch = (size_t)((h + 1) / 2);
	return (size_t)w * (size_t)h + 2 * cw * ch;
}

// Attach the audio output. `node` must be an NX_AUDIO_NODE_STREAM_SOURCE
// whose graph runs at `sample_rate`; the decoder resamples the audio track
// to interleaved stereo f32 at that rate. The caller owns the node and must
// keep it alive until nx_media_destroy(). Call before nx_media_play().
void nx_media_set_audio_node(nx_media_t *m, nx_audio_node *node,
                             double sample_rate);

// Transport controls (non-blocking; the decode thread reacts).
void nx_media_play(nx_media_t *m);
void nx_media_pause(nx_media_t *m);
void nx_media_seek(nx_media_t *m, double seconds);
void nx_media_set_loop(nx_media_t *m, bool loop);

// Presentation: if a video frame is due at the current media clock, swap it
// into `*buffer_inout` (a caller-owned width*height*4 BGRA buffer; the
// pointer is exchanged with the ring slot's buffer). Returns true if a new
// frame was presented. Call from the main thread (e.g. once per host frame).
bool nx_media_present(nx_media_t *m, uint8_t **buffer_inout);

// Current playback position in media seconds (wraps when looping).
double nx_media_current_time(nx_media_t *m);

// Number of decoded-and-waiting video frames (readiness heuristic).
uint32_t nx_media_buffered_frames(nx_media_t *m);

// Presentation quality counters (getVideoPlaybackQuality): frames actually
// presented, and frames skipped because a newer frame was already due.
uint64_t nx_media_presented_frames(nx_media_t *m);
uint64_t nx_media_dropped_frames(nx_media_t *m);

// True once playback reached the end of the stream (never true while
// looping) and all buffered frames/audio have been presented/consumed.
bool nx_media_ended(nx_media_t *m);

// True while a seek is in flight (target not yet decoded).
bool nx_media_seeking(nx_media_t *m);

// Sticky fatal decode error message, or NULL.
const char *nx_media_error(nx_media_t *m);

// Cut #22b Stage 2 (2026-07-02): audio-visualizer surface.
//
// The decode thread taps its resampled audio (downmixed to mono) into a
// small rolling ring; these three accessors read from the ring from the
// main thread. They target Switch.VideoDecoder's `getWaveform` /
// `getFrequencyData` / `getAudioLevels` methods, which brewser-runtime
// re-exposes on `<audio>` / `<video>` DOM elements (spectraplay's
// visualizer reads from there).
//
// All three return `false` (or 0 for `read_audio_levels`) when the tap
// hasn't yet accumulated enough samples (fresh decoder / no audio track).

// Fill `out` with the last `out_len` mono samples in [-1, 1]. `out_len`
// should be <= the internal tap window (2048). Returns true iff data was
// filled.
bool nx_media_read_waveform(nx_media_t *m, float *out, uint32_t out_len);

// Compute FFT magnitude over the tap window (Hann-windowed), bin-average
// into `out_len` bins from 0..~Nyquist, normalize to ~[0, 1]. Returns
// true iff data was filled.
bool nx_media_read_spectrum(nx_media_t *m, float *out, uint32_t out_len);

// Fill up to `out_max` per-band RMS values in ~[0, 1] (bass, mid, high).
// Returns the number of bands written (0..3). Backs `getAudioLevels()`.
uint32_t nx_media_read_audio_levels(nx_media_t *m, float *out,
                                    uint32_t out_max);

// Stop the decode thread (joins it) and free everything. The audio node is
// NOT freed (caller-owned).
void nx_media_destroy(nx_media_t *m);
