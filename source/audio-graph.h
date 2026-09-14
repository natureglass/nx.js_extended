#pragma once
// Portable Web Audio graph engine — pure C++ (std only).
//
// This file is compiled into BOTH the device runtime and the host nxjs-test
// binary, so it must not reference V8, libnx, or libuv. The V8 bindings live
// in audio.cc; the platform output sinks live in audio-sink-audren.cc (device)
// and packages/runtime/test/src/audio-sink.cc (host).
//
// Threading model: a "control thread" (the JS loop thread, via audio.cc) and a
// "render thread" (the sink, or a libuv worker for OfflineAudioContext) both
// touch the graph. Every public function in this header — and the render
// entrypoints — locks the graph mutex internally. The internal bus format is
// stereo float32, processed in 128-frame render quanta (like browsers).

#include <atomic>
#include <math.h>
#include <memory>
#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <vector>

#define NX_AUDIO_RENDER_QUANTUM 128
#define NX_AUDIO_CHANNELS 2

enum nx_audio_node_type {
	NX_AUDIO_NODE_DESTINATION = 0,
	NX_AUDIO_NODE_GAIN = 1,
	NX_AUDIO_NODE_STEREO_PANNER = 2,
	NX_AUDIO_NODE_BUFFER_SOURCE = 3,
	// Internal (not exposed as a Web Audio class): a ring-buffer-fed source
	// used by media elements (Video). A single producer thread (the media
	// decode thread) pushes interleaved stereo f32 frames at the graph rate;
	// the render thread drains a quantum at a time. The consumed-frame
	// counter is the A/V sync master clock.
	NX_AUDIO_NODE_STREAM_SOURCE = 4,
	NX_AUDIO_NODE_OSCILLATOR = 5,
	// AnalyserNode — a passthrough node (output == summed input) that also
	// taps its output into a ring buffer JS can read for time-domain /
	// frequency visualisation. Unlike other sinks it isn't reachable from the
	// destination, so `render_quantum` processes it explicitly each quantum.
	NX_AUDIO_NODE_ANALYSER = 6,
	// DelayNode — a variable-length delay line. Output for the current quantum
	// is read from the line BEFORE this quantum's input is written, so it only
	// ever reads already-written (past) samples. That decoupling is what lets a
	// DelayNode legally sit in a feedback cycle (delay -> gain -> delay): the
	// feedback path reads the delayed output computed this quantum. The delay
	// is evaluated k-rate and clamped to a minimum of one render quantum (the
	// Web Audio cycle constraint), so a feedback loop is delayed by >= Q frames.
	NX_AUDIO_NODE_DELAY = 7,
	// DynamicsCompressorNode — a feed-forward peak compressor with a soft knee
	// and attack/release smoothing. No automatic makeup gain (output is only
	// ever attenuated), so it is safe to drop into a master bus without a
	// surprise loudness jump. `reduction` (read-only, dB, <= 0) is polled by JS.
	NX_AUDIO_NODE_DYNAMICS_COMPRESSOR = 8,
	// BiquadFilterNode — a second-order IIR filter (the Audio EQ Cookbook
	// shapes the Web Audio spec prescribes). Coefficients are recomputed once
	// per render quantum (k-rate) from frequency / detune / Q / gain; the two
	// channels carry independent direct-form-I state across quanta.
	NX_AUDIO_NODE_BIQUAD_FILTER = 9,
	// ConvolverNode — FFT convolution with an impulse response, used for
	// reverb and cabinet/speaker simulation. Uniformly-partitioned overlap-save
	// with a frequency-delay line, so cost is O(K) spectral multiply-accumulates
	// per block rather than O(N^2) time-domain taps. The partition size is the
	// node's latency (see NX_AUDIO_CONVOLVER_BLOCK).
	NX_AUDIO_NODE_CONVOLVER = 10,
};

// OscillatorNode wave types (matches OscillatorType wire values in JS).
enum nx_audio_oscillator_type {
	NX_AUDIO_OSCILLATOR_SINE = 0,
	NX_AUDIO_OSCILLATOR_SQUARE = 1,
	NX_AUDIO_OSCILLATOR_SAWTOOTH = 2,
	NX_AUDIO_OSCILLATOR_TRIANGLE = 3,
};

// BiquadFilterNode shapes (matches BiquadFilterType wire values in JS).
enum nx_audio_biquad_type {
	NX_AUDIO_BIQUAD_LOWPASS = 0,
	NX_AUDIO_BIQUAD_HIGHPASS = 1,
	NX_AUDIO_BIQUAD_BANDPASS = 2,
	NX_AUDIO_BIQUAD_LOWSHELF = 3,
	NX_AUDIO_BIQUAD_HIGHSHELF = 4,
	NX_AUDIO_BIQUAD_PEAKING = 5,
	NX_AUDIO_BIQUAD_NOTCH = 6,
	NX_AUDIO_BIQUAD_ALLPASS = 7,
};

// ConvolverNode partition size, in frames. The impulse response is chopped
// into blocks of this length, each transformed once at assignment time; the
// input is convolved a block at a time by overlap-save. It is the node's
// input-to-output latency (1024 frames ~= 21 ms at 48 kHz), and it trades
// against CPU: halving it doubles the number of spectral multiply-accumulates
// per sample. Must be a multiple of the render quantum.
#define NX_AUDIO_CONVOLVER_BLOCK 1024
// Longest impulse response kept, in seconds. The partitioned spectra plus the
// frequency-delay line cost ~16 bytes per impulse frame per channel, so an
// unbounded cap would let a page (pixi-sound's ReverbFilter accepts up to 50 s)
// allocate hundreds of megabytes. Longer responses are truncated.
#define NX_AUDIO_CONVOLVER_MAX_SECONDS 5.0

// AudioParam automation event types (matches the JS side's wire protocol).
enum nx_audio_param_event_type {
	NX_AUDIO_PARAM_SET_VALUE = 0,
	NX_AUDIO_PARAM_LINEAR_RAMP = 1,
	NX_AUDIO_PARAM_EXPONENTIAL_RAMP = 2,
	NX_AUDIO_PARAM_SET_TARGET = 3,
	NX_AUDIO_PARAM_SET_VALUE_CURVE = 4,
};

// AudioBufferSourceNode playback states (polled by JS for `ended` events).
enum nx_audio_source_state {
	NX_AUDIO_SOURCE_UNSCHEDULED = 0,
	NX_AUDIO_SOURCE_SCHEDULED = 1,
	NX_AUDIO_SOURCE_FINISHED = 2,
};

struct nx_audio_param_event {
	nx_audio_param_event_type type;
	double time;          // event (start) time, in context seconds
	float value;          // target value (SET_VALUE/RAMP/SET_TARGET)
	double time_constant; // SET_TARGET only
	double duration;      // SET_VALUE_CURVE only
	std::vector<float> curve; // SET_VALUE_CURVE only
};

struct nx_audio_param {
	float value = 0.f; // [[current value]] (base when no events apply)
	float min_value = -3.402823466e+38f;
	float max_value = 3.402823466e+38f;
	std::vector<nx_audio_param_event> events; // sorted by time
};

// In-place iterative radix-2 complex FFT with precomputed twiddles and
// bit-reversal permutation. Shared by every ConvolverNode using the same
// transform size (it lives on the impulse response, which nodes share).
struct nx_audio_fft {
	uint32_t size = 0;
	std::vector<uint32_t> rev;   // bit-reversal permutation
	std::vector<float> tw_cos;   // size/2 twiddles
	std::vector<float> tw_sin;
	void init(uint32_t n);
	// Decimation-in-time forward transform (unscaled).
	void forward(float *re, float *im) const;
	// Inverse via the conjugate trick, scaled by 1/size.
	void inverse(float *re, float *im) const;
};

// A ConvolverNode impulse response, pre-transformed into per-partition
// spectra. Immutable once built, and held by shared_ptr so the control thread
// can do the (expensive) transform work outside the graph mutex and then swap
// the finished object in under it, without ever tearing a response the render
// thread is mid-way through reading.
struct nx_audio_convolver_ir {
	uint32_t block = 0;      // partition size, frames
	uint32_t fft_size = 0;   // 2 * block
	uint32_t bins = 0;       // fft_size / 2 + 1 (the non-redundant half)
	uint32_t partitions = 0; // ceil(impulse frames / block)
	int channels = 1;        // 1 = the same response on both channels
	nx_audio_fft fft;
	// Per channel, `partitions * bins` complex bins as (re, im) pairs.
	std::vector<float> spectra[NX_AUDIO_CHANNELS];
};

struct nx_audio_graph;

struct nx_audio_node {
	nx_audio_graph *graph = nullptr;
	nx_audio_node_type type;

	// Graph topology. `inputs` are the nodes connected INTO this node;
	// `outputs` is the reverse mapping (for cleanup on release). Duplicate
	// connections are collapsed (per spec: multiple connect() calls between
	// the same nodes are idempotent).
	std::vector<nx_audio_node *> inputs;
	std::vector<nx_audio_node *> outputs;

	// Per-quantum processing state.
	uint64_t processed_quantum = 0;
	bool processing = false; // cycle guard
	// Permanently silent: a finished source, or a pure gain/panner all of whose
	// inputs are permanently silent. Such a node can never produce signal again,
	// so the render walk skips it entirely. Cleared when an input is connected.
	bool silent = false;
	bool silent_checking = false; // cycle guard for the silence propagation
	float bus[NX_AUDIO_CHANNELS][NX_AUDIO_RENDER_QUANTUM];
	// Channel count of the bus content: 1 = mono (L==R), 2 = true stereo.
	// Drives spec-correct mono vs stereo panning behavior.
	int bus_ch = 1;

	// AudioParams, addressed by index:
	//   GAIN:          0 = gain
	//   STEREO_PANNER: 0 = pan
	//   BUFFER_SOURCE: 0 = playbackRate, 1 = detune
	//   OSCILLATOR:    0 = frequency, 1 = detune
	//   BIQUAD_FILTER: 0 = frequency, 1 = detune, 2 = Q, 3 = gain
	//   DELAY:         0 = delayTime
	//   DYNAMICS_COMPRESSOR: 0 = threshold, 1 = knee, 2 = ratio,
	//                        3 = attack, 4 = release
	std::vector<nx_audio_param> params;

	// ---- OscillatorNode state ----
	int oscillator_type = NX_AUDIO_OSCILLATOR_SINE;

	// ---- AudioBufferSourceNode state ----
	// Channel data points into externally-owned memory (V8 BackingStores);
	// `buffer_holds` keeps that memory alive for the render thread.
	std::vector<const float *> buffer_channels;
	std::vector<std::shared_ptr<void>> buffer_holds;
	uint32_t buffer_length = 0;   // frames
	double buffer_sample_rate = 0;
	bool loop = false;
	double loop_start = 0;
	double loop_end = 0;
	int playback_state = NX_AUDIO_SOURCE_UNSCHEDULED;
	bool started = false;       // start() called
	bool playing = false;       // playhead initialized (first audible quantum)
	double start_time = 0;      // when (context seconds)
	double start_offset = 0;    // offset into buffer (seconds)
	double duration = -1;       // <0 = no duration limit (buffer seconds)
	double stop_time = -1;      // <0 = no stop scheduled
	double position = 0;        // playhead, fractional buffer frames
	double played_frames = 0;   // cumulative buffer frames consumed

	// ---- stream source state (NX_AUDIO_NODE_STREAM_SOURCE) ----
	// Lock-free SPSC ring of interleaved stereo f32 frames. The producer
	// (media decode thread) owns `stream_write_pos`; the consumer (render
	// thread, under the graph mutex) owns `stream_read_pos`. Positions are
	// absolute frame counters (never wrapped); ring index = pos % capacity.
	std::unique_ptr<float[]> stream_ring;
	uint32_t stream_capacity = 0; // frames
	std::atomic<uint64_t> stream_write_pos{0};
	std::atomic<uint64_t> stream_read_pos{0};
	// When false the node outputs silence and consumes nothing (pause).
	std::atomic<bool> stream_playing{false};
	// Ledger #114 diag — counts render quanta where avail < Q (underrun).
	std::atomic<uint64_t> stream_underrun_count{0};

	// ---- analyser state (NX_AUDIO_NODE_ANALYSER) ----
	// Ring of the most-recent downmixed (mono) output samples. Written by the
	// render thread under the graph mutex (process_analyser), read by JS
	// (getFloatTimeDomainData) under the same mutex. Capacity is a power of
	// two >= the max fftSize (32768) so JS can request any fftSize up to that.
	std::unique_ptr<float[]> analyser_ring;
	uint32_t analyser_ring_size = 0;      // capacity in samples (power of two)
	uint64_t analyser_write_pos = 0;      // absolute samples written (ring idx = pos & (size-1))

	// ---- delay state (NX_AUDIO_NODE_DELAY) ----
	// Interleaved stereo delay line. `delay_write_pos` is an absolute frame
	// counter (never wrapped; ring index = pos % capacity), like the stream
	// ring. Capacity holds maxDelayTime seconds plus one render quantum of
	// headroom so a full-length read never overlaps the frames being written
	// this quantum. `delay_bus_ch` carries the last input's channel count to
	// the next quantum's output (output is produced before input is summed).
	std::unique_ptr<float[]> delay_ring;
	uint32_t delay_capacity = 0;   // frames
	uint64_t delay_write_pos = 0;  // absolute frames written
	double delay_max_time = 0;     // seconds
	int delay_bus_ch = 1;

	// ---- dynamics compressor state (NX_AUDIO_NODE_DYNAMICS_COMPRESSOR) ----
	// Smoothed gain reduction in dB (<= 0), carried across quanta so attack /
	// release envelopes are continuous. `comp_reduction` is the value JS reads
	// back through the read-only `.reduction` property.
	double comp_env_db = 0.0;
	float comp_reduction = 0.f;

	// ---- biquad filter state (NX_AUDIO_NODE_BIQUAD_FILTER) ----
	// Direct-form-I history, per channel: x[n-1], x[n-2], y[n-1], y[n-2].
	// Kept in double so a high-Q filter's state doesn't drift on float32.
	int biquad_type = NX_AUDIO_BIQUAD_LOWPASS;
	double biquad_x1[NX_AUDIO_CHANNELS] = {0, 0};
	double biquad_x2[NX_AUDIO_CHANNELS] = {0, 0};
	double biquad_y1[NX_AUDIO_CHANNELS] = {0, 0};
	double biquad_y2[NX_AUDIO_CHANNELS] = {0, 0};

	// ---- convolver state (NX_AUDIO_NODE_CONVOLVER) ----
	// `conv_ir` is swapped wholesale when the page assigns `.buffer`; the
	// render thread copies the shared_ptr so a swap mid-quantum is safe.
	// `conv_in` holds 2 * block frames per channel — the previous block
	// followed by the one being filled — which is the overlap-save window.
	// `conv_fdl` is the frequency-delay line: the spectra of the last
	// `partitions` windows, newest at `conv_fdl_slot`.
	std::shared_ptr<nx_audio_convolver_ir> conv_ir;
	std::vector<float> conv_in[NX_AUDIO_CHANNELS];
	std::vector<float> conv_out[NX_AUDIO_CHANNELS];
	std::vector<float> conv_fdl[NX_AUDIO_CHANNELS];
	// Per delay-line slot: is that stored spectrum known to be all zeros? A
	// zero spectrum contributes nothing, so those partitions are skipped. That
	// is the difference between a convolver costing its full price whenever a
	// reverb tail is ringing out and costing it only while signal is arriving
	// — which for one-shot sound effects is the overwhelming majority case.
	std::vector<uint8_t> conv_fdl_zero;
	std::vector<float> conv_scratch_re; // fft_size
	std::vector<float> conv_scratch_im;
	std::vector<float> conv_acc_re; // bins
	std::vector<float> conv_acc_im;
	uint32_t conv_fdl_slot = 0;
	uint32_t conv_fill = 0;    // frames accumulated into the current block
	uint32_t conv_out_pos = 0; // read cursor into conv_out
	// Consecutive all-zero input blocks. Once it exceeds `partitions` the
	// whole delay line is known to be zero, so the tail has fully decayed and
	// the (expensive) transform work is skipped until signal returns.
	uint32_t conv_zero_blocks = 0;
	bool conv_prev_zero = true; // was the previous input block all zeros?
	int conv_out_ch = 1;
};

struct nx_audio_graph {
	std::mutex mutex;
	std::atomic<int> refs{1};
	double sample_rate;
	uint64_t frames_rendered = 0; // currentTime = frames_rendered / rate
	uint64_t quantum_id = 0;
	bool suspended = false;
	bool closed = false;
	nx_audio_node *destination = nullptr;
	std::vector<nx_audio_node *> nodes; // all live nodes (owned), incl. dest
};

// ---- graph lifecycle ----
nx_audio_graph *nx_audio_graph_create(double sample_rate);
void nx_audio_graph_ref(nx_audio_graph *g);
void nx_audio_graph_unref(nx_audio_graph *g); // frees at 0 (incl. all nodes)
double nx_audio_graph_current_time(nx_audio_graph *g);
void nx_audio_graph_set_suspended(nx_audio_graph *g, bool suspended);

// ---- nodes ----
// Creating a node adds a graph ref; releasing it removes the node from the
// graph (disconnecting both directions) and drops that ref. Releasing the
// destination node only drops the ref (the node itself is graph-owned).
// Safe to call from a GC finalizer (locks the mutex; no JS API).
// `aux` is a type-specific creation parameter: for NX_AUDIO_NODE_DELAY it is
// maxDelayTime in seconds (sizes the delay line); ignored by other types.
nx_audio_node *nx_audio_node_create(nx_audio_graph *g, nx_audio_node_type type,
                                    double aux = 0.0);
void nx_audio_node_release(nx_audio_node *n);
void nx_audio_node_connect(nx_audio_node *src, nx_audio_node *dst);
// dst == NULL disconnects all outputs.
void nx_audio_node_disconnect(nx_audio_node *src, nx_audio_node *dst);

// ---- params ----
// Returns NULL for an out-of-range index.
nx_audio_param *nx_audio_node_param(nx_audio_node *n, int index);
float nx_audio_param_value(nx_audio_node *n, nx_audio_param *p); // at currentTime
void nx_audio_param_set_value(nx_audio_node *n, nx_audio_param *p, float value);
void nx_audio_param_schedule(nx_audio_node *n, nx_audio_param *p,
                             nx_audio_param_event_type type, double time,
                             float value, double time_constant);
void nx_audio_param_set_value_curve(nx_audio_node *n, nx_audio_param *p,
                                    const float *curve, size_t len,
                                    double start_time, double duration);
void nx_audio_param_cancel(nx_audio_node *n, nx_audio_param *p, double time);

// ---- buffer source ----
void nx_audio_source_set_buffer(nx_audio_node *n, const float *const *channels,
                                int num_channels, uint32_t length,
                                double sample_rate,
                                std::vector<std::shared_ptr<void>> holds);
void nx_audio_source_set_loop(nx_audio_node *n, bool loop, double loop_start,
                              double loop_end);
void nx_audio_source_start(nx_audio_node *n, double when, double offset,
                           double duration);
void nx_audio_source_stop(nx_audio_node *n, double when);
int nx_audio_source_playback_state(nx_audio_node *n);

// ---- oscillator ----
void nx_audio_oscillator_set_type(nx_audio_node *n, int type);

// ---- biquad filter ----
void nx_audio_biquad_set_type(nx_audio_node *n, int type);
// Evaluates |H(e^jw)| and arg H(e^jw) at each of `count` frequencies (Hz),
// using the coefficients for the parameter values at the current time.
// Frequencies outside [0, Nyquist] yield NaN, per spec.
void nx_audio_biquad_frequency_response(nx_audio_node *n,
                                        const float *frequency_hz,
                                        float *mag_response,
                                        float *phase_response, uint32_t count);

// ---- convolver ----
// Installs an impulse response (the samples ARE copied — they are transformed,
// not read live). `channels` may be 1 (same response both sides) or 2. Passing
// num_channels <= 0 clears the response, after which the node outputs silence,
// per spec. Resamples to the graph rate if needed, truncates beyond
// NX_AUDIO_CONVOLVER_MAX_SECONDS, and applies the spec's RMS normalisation
// when `normalize` is set. The transform work happens BEFORE the graph mutex
// is taken, so a multi-second response doesn't stall the render thread.
void nx_audio_convolver_set_buffer(nx_audio_node *n,
                                   const float *const *channels,
                                   int num_channels, uint32_t length,
                                   double sample_rate, bool normalize);

// ---- dynamics compressor ----
// Current gain reduction in dB (<= 0), for the read-only `.reduction` property.
float nx_audio_compressor_reduction(nx_audio_node *n);

// ---- analyser (time-domain readback; locks the graph mutex) ----
// Fills `out` with the most-recent `count` downmixed output samples (newest
// last), each in [-1, 1]. Samples older than the ring holds (or before any
// audio was rendered) are 0. Frequency-domain data is computed in JS from
// this time-domain window.
void nx_audio_analyser_get_float_time_data(nx_audio_node *n, float *out,
                                           uint32_t count);

// ---- stream source (producer side; lock-free, single producer thread) ----
// Number of frames that can currently be written without overwriting.
uint32_t nx_audio_stream_writable(nx_audio_node *n);
// Write up to `frames` interleaved stereo frames; returns frames written.
uint32_t nx_audio_stream_write(nx_audio_node *n, const float *interleaved,
                               uint32_t frames);
// Gate consumption (false = output silence, consume nothing, clock frozen).
void nx_audio_stream_set_playing(nx_audio_node *n, bool playing);
// Total frames consumed by the render thread (the media clock).
uint64_t nx_audio_stream_consumed(nx_audio_node *n);
// Ledger #114 diag — frames currently queued in the ring (write - read).
uint32_t nx_audio_stream_pending(nx_audio_node *n);
// Ledger #114 diag — count of render quanta where avail < Q (underrun events).
// Reset on stream_flush. Read-only snapshot; monotonic during playback.
uint64_t nx_audio_stream_underrun_count(nx_audio_node *n);
// Discard all buffered frames (seek/flush). The producer thread MUST be
// parked while this is called.
void nx_audio_stream_flush(nx_audio_node *n);

// ---- rendering (called from sink threads / libuv workers; locks mutex) ----
// Renders `frames` of interleaved stereo s16. When the graph is suspended (or
// closed), fills with silence WITHOUT advancing currentTime.
void nx_audio_graph_render_s16(nx_audio_graph *g, int16_t *out,
                               uint32_t frames);
// OfflineAudioContext rendering: renders `length` frames into `num_channels`
// planar float buffers (1 = mono downmix, 2 = stereo). Ignores `suspended`.
void nx_audio_graph_render_offline(nx_audio_graph *g, float *const *channels,
                                   int num_channels, uint32_t length);
