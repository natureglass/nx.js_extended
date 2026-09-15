// Portable Web Audio graph engine implementation. See audio-graph.h for the
// architecture overview. No V8 / libnx / libuv — compiled into both the device
// runtime and the host nxjs-test binary.
#include "audio-graph.h"
#include <algorithm>
#include <string.h>

namespace {
constexpr double NX_TWO_PI = 6.283185307179586476925;
constexpr double NX_PI = 3.141592653589793238463;
} // namespace

// ---------------------------------------------------------------------------
// FFT (used by ConvolverNode)
// ---------------------------------------------------------------------------

void nx_audio_fft::init(uint32_t n) {
	size = n;
	uint32_t bits = 0;
	while ((1u << bits) < n)
		bits++;
	rev.resize(n);
	for (uint32_t i = 0; i < n; i++) {
		uint32_t r = 0;
		for (uint32_t b = 0; b < bits; b++)
			if (i & (1u << b))
				r |= 1u << (bits - 1 - b);
		rev[i] = r;
	}
	tw_cos.resize(n / 2);
	tw_sin.resize(n / 2);
	for (uint32_t i = 0; i < n / 2; i++) {
		double ang = -NX_TWO_PI * (double)i / (double)n;
		tw_cos[i] = (float)cos(ang);
		tw_sin[i] = (float)sin(ang);
	}
}

void nx_audio_fft::forward(float *re, float *im) const {
	uint32_t n = size;
	for (uint32_t i = 0; i < n; i++) {
		uint32_t j = rev[i];
		if (j > i) {
			std::swap(re[i], re[j]);
			std::swap(im[i], im[j]);
		}
	}
	for (uint32_t len = 2; len <= n; len <<= 1) {
		uint32_t half = len >> 1;
		uint32_t step = n / len;
		for (uint32_t i = 0; i < n; i += len) {
			uint32_t k = 0;
			for (uint32_t j = 0; j < half; j++, k += step) {
				float wr = tw_cos[k], wi = tw_sin[k];
				uint32_t a = i + j, b = a + half;
				float xr = re[b] * wr - im[b] * wi;
				float xi = re[b] * wi + im[b] * wr;
				re[b] = re[a] - xr;
				im[b] = im[a] - xi;
				re[a] += xr;
				im[a] += xi;
			}
		}
	}
}

void nx_audio_fft::inverse(float *re, float *im) const {
	uint32_t n = size;
	for (uint32_t i = 0; i < n; i++)
		im[i] = -im[i];
	forward(re, im);
	float scale = 1.f / (float)n;
	for (uint32_t i = 0; i < n; i++) {
		re[i] *= scale;
		im[i] = -im[i] * scale;
	}
}

namespace {

constexpr int Q = NX_AUDIO_RENDER_QUANTUM;

float clampf(float v, float lo, float hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// AudioParam timeline evaluation
// ---------------------------------------------------------------------------

// Computes the parameter value at time `t` by walking the first `count`
// events of the (time-sorted) event list. This is a pragmatic implementation
// of the Web Audio "computedValue" algorithm covering the common shapes: set,
// linear/exponential ramps, setTarget decay, and value curves. Ramps anchor at
// the previous event's (time, value); a setTarget remains in effect until the
// next event. `count` (<= events.size()) lets param_prune evaluate a prefix in
// isolation; param_value_at below passes the full list for normal reads.
float param_value_at_n(const nx_audio_param *p, double t, size_t count) {
	const auto &evs = p->events;
	double v_prev = p->value;
	double t_prev = 0;
	for (size_t i = 0; i < count; i++) {
		const nx_audio_param_event &e = evs[i];
		if (e.time > t) {
			// `t` falls before this event takes effect. Ramps interpolate
			// from the previous anchor toward this (future) event.
			if (e.type == NX_AUDIO_PARAM_LINEAR_RAMP) {
				if (e.time <= t_prev)
					return clampf((float)e.value, p->min_value, p->max_value);
				double f = (t - t_prev) / (e.time - t_prev);
				if (f < 0)
					f = 0;
				return clampf((float)(v_prev + (e.value - v_prev) * f),
				              p->min_value, p->max_value);
			}
			if (e.type == NX_AUDIO_PARAM_EXPONENTIAL_RAMP) {
				if (v_prev == 0 || (v_prev < 0) != (e.value < 0) ||
				    e.time <= t_prev)
					return clampf((float)v_prev, p->min_value, p->max_value);
				double f = (t - t_prev) / (e.time - t_prev);
				if (f < 0)
					f = 0;
				return clampf((float)(v_prev * pow(e.value / v_prev, f)),
				              p->min_value, p->max_value);
			}
			// Other event types have no effect before their start time.
			return clampf((float)v_prev, p->min_value, p->max_value);
		}
		// Event time <= t: apply it and move the anchor forward.
		switch (e.type) {
		case NX_AUDIO_PARAM_SET_VALUE:
		case NX_AUDIO_PARAM_LINEAR_RAMP:
		case NX_AUDIO_PARAM_EXPONENTIAL_RAMP:
			v_prev = e.value;
			t_prev = e.time;
			break;
		case NX_AUDIO_PARAM_SET_VALUE_CURVE: {
			size_t n = e.curve.size();
			if (n == 0)
				break;
			double tend = e.time + e.duration;
			if (t < tend && e.duration > 0) {
				double f = (t - e.time) / e.duration * (double)(n - 1);
				size_t k = (size_t)f;
				if (k >= n - 1)
					return clampf(e.curve[n - 1], p->min_value, p->max_value);
				double frac = f - (double)k;
				return clampf(
				    (float)(e.curve[k] + (e.curve[k + 1] - e.curve[k]) * frac),
				    p->min_value, p->max_value);
			}
			v_prev = e.curve[n - 1];
			t_prev = tend;
			break;
		}
		case NX_AUDIO_PARAM_SET_TARGET: {
			// In effect until the next event (or `t`, whichever is sooner).
			// An event at exactly `t` takes over (events apply at their
			// start time, inclusive), so the comparison is `<=`.
			double tstop = t;
			bool next_applies =
			    i + 1 < count && evs[i + 1].time <= t;
			if (next_applies)
				tstop = evs[i + 1].time;
			double val;
			if (e.time_constant <= 0) {
				val = e.value;
			} else {
				val = e.value + (v_prev - e.value) *
				                    exp(-(tstop - e.time) / e.time_constant);
			}
			if (!next_applies)
				return clampf((float)val, p->min_value, p->max_value);
			v_prev = val;
			t_prev = tstop;
			break;
		}
		}
	}
	return clampf((float)v_prev, p->min_value, p->max_value);
}

// Evaluate over the full event list (the common case).
float param_value_at(const nx_audio_param *p, double t) {
	return param_value_at_n(p, t, p->events.size());
}

// Fold away automation events that can no longer influence evaluation at or
// after `t_now`, keeping each param's event list O(1) instead of growing once
// per scheduled automation. Without this a control that schedules automation
// every frame (e.g. a knob dragged with setTargetAtTime per pointermove) makes
// the list grow without bound, and param_value_at re-walks it from the start
// for every one of the 128 frames per render quantum — so the render thread's
// per-quantum cost climbs linearly and eventually starves the main thread.
//
// Because context time only advances (the render walks forward and every read
// uses currentTime), every event strictly before the most recent event with
// time <= t_now is "settled": its sole remaining role is to hand an anchor
// value to that event. We compute that value from the prefix, fold it into the
// param's base `value`, and erase the prefix. Correctness rests on the
// evaluator's anchoring rules:
//   * SET_VALUE, and ramps/curves that have completed by their own event time,
//     are self-anchoring — they overwrite v_prev and ignore the base.
//   * SET_TARGET consumes only the base as its start value (it reads v_prev,
//     never t_prev), which is exactly what we fold in.
//   * A retained *future* ramp still anchors to the kept last-past event (we
//     only drop events strictly before it), so its interpolation is unchanged.
// See the "setTargetAtTime spam" / "event pruning is behavior-preserving"
// regression tests in test/fixtures/webaudio.ts.
void param_prune(nx_audio_param *p, double t_now) {
	auto &evs = p->events;
	if (evs.size() < 2)
		return;
	// L = index of the last event with time <= t_now (the active anchor).
	size_t L = 0;
	bool found = false;
	for (size_t i = 0; i < evs.size(); i++) {
		if (evs[i].time <= t_now) {
			L = i;
			found = true;
		} else {
			break; // events are time-sorted; nothing later qualifies
		}
	}
	if (!found || L == 0)
		return; // no settled events precede the anchor
	// Value the timeline holds just as events[L] takes over, computed only from
	// the events we are about to drop (the prefix [0, L)).
	double anchor = param_value_at_n(p, evs[L].time, L);
	p->value = clampf((float)anchor, p->min_value, p->max_value);
	evs.erase(evs.begin(), evs.begin() + L);
}

// Fill `out[0..n)` with a-rate values for frame times t0, t0+1/sr, ...
void param_fill(const nx_audio_param *p, double t0, double inv_sr, float *out,
                int n) {
	if (p->events.empty()) {
		float v = clampf(p->value, p->min_value, p->max_value);
		for (int i = 0; i < n; i++)
			out[i] = v;
		return;
	}
	for (int i = 0; i < n; i++)
		out[i] = param_value_at(p, t0 + i * inv_sr);
}

// Insert an event keeping the list time-sorted (stable for equal times). A
// SET_VALUE at the exact time of an existing SET_VALUE replaces it.
void param_insert_event(nx_audio_param *p, nx_audio_param_event &&e) {
	if (e.type == NX_AUDIO_PARAM_SET_VALUE) {
		for (auto &existing : p->events) {
			if (existing.time == e.time &&
			    existing.type == NX_AUDIO_PARAM_SET_VALUE) {
				existing = std::move(e);
				return;
			}
		}
	}
	auto it = std::upper_bound(
	    p->events.begin(), p->events.end(), e.time,
	    [](double t, const nx_audio_param_event &ev) { return t < ev.time; });
	p->events.insert(it, std::move(e));
}

// ---------------------------------------------------------------------------
// Node processing
// ---------------------------------------------------------------------------

void zero_bus(nx_audio_node *n) {
	memset(n->bus, 0, sizeof(n->bus));
}

void process_node(nx_audio_graph *g, nx_audio_node *n, double t0);

// `out_channels` is set to the computed channel count of the summed input.
void sum_inputs(nx_audio_graph *g, nx_audio_node *n, double t0,
                float in[NX_AUDIO_CHANNELS][Q], int *out_channels) {
	memset(in, 0, sizeof(float) * NX_AUDIO_CHANNELS * Q);
	int ch = 1;
	for (nx_audio_node *src : n->inputs) {
		// A permanently-silent input contributes exact zeros forever; skipping
		// it avoids both rendering it and accumulating a quantum of zeros.
		if (src->silent)
			continue;
		process_node(g, src, t0);
		for (int c = 0; c < NX_AUDIO_CHANNELS; c++)
			for (int i = 0; i < Q; i++)
				in[c][i] += src->bus[c][i];
		if (src->bus_ch == 2)
			ch = 2;
	}
	*out_channels = ch;
}

void process_buffer_source(nx_audio_graph *g, nx_audio_node *n, double t0) {
	zero_bus(n);
	n->bus_ch = 1;
	if (!n->started || n->playback_state == NX_AUDIO_SOURCE_FINISHED)
		return;

	bool has_buf = !n->buffer_channels.empty() && n->buffer_length > 0 &&
	               n->buffer_sample_rate > 0;
	if (has_buf && n->buffer_channels.size() > 1)
		n->bus_ch = 2;

	// k-rate playback rate, computed once per quantum.
	double rate = param_value_at(&n->params[0], t0);
	double detune = param_value_at(&n->params[1], t0);
	double r = rate * exp2(detune / 1200.0);
	if (has_buf)
		r *= n->buffer_sample_rate / g->sample_rate;

	double inv_sr = 1.0 / g->sample_rate;
	double buf_len = has_buf ? (double)n->buffer_length : 0;

	// Loop points, in buffer frames.
	double loop_s = 0, loop_e = buf_len;
	if (n->loop && has_buf) {
		loop_s = n->loop_start * n->buffer_sample_rate;
		loop_e = n->loop_end > 0 ? n->loop_end * n->buffer_sample_rate
		                         : buf_len;
		if (loop_s < 0)
			loop_s = 0;
		if (loop_e > buf_len)
			loop_e = buf_len;
		if (loop_e <= loop_s) {
			loop_s = 0;
			loop_e = buf_len;
		}
	}
	double dur_frames =
	    n->duration >= 0 && has_buf ? n->duration * n->buffer_sample_rate : -1;

	const float *ch0 = has_buf ? n->buffer_channels[0] : nullptr;
	const float *ch1 = has_buf && n->buffer_channels.size() > 1
	                       ? n->buffer_channels[1]
	                       : ch0;

	bool finished = false;
	for (int i = 0; i < Q; i++) {
		double t = t0 + i * inv_sr;
		if (n->stop_time >= 0 && t >= n->stop_time) {
			finished = true;
			break;
		}
		if (t < n->start_time)
			continue; // bus is already zero
		if (!n->playing) {
			n->playing = true;
			n->position = n->start_offset * n->buffer_sample_rate;
			if (n->position < 0)
				n->position = 0;
			if (n->loop && n->position > loop_e)
				n->position = loop_s;
			n->played_frames = 0;
		}
		if (!has_buf)
			continue; // null buffer: silence until stop()
		if (dur_frames >= 0 && n->played_frames >= dur_frames) {
			finished = true;
			break;
		}
		if (!n->loop && (n->position >= buf_len || n->position < 0)) {
			finished = true;
			break;
		}
		uint32_t i0 = (uint32_t)n->position;
		if (i0 >= n->buffer_length)
			i0 = n->buffer_length - 1;
		uint32_t i1 = i0 + 1 < n->buffer_length ? i0 + 1 : i0;
		double frac = n->position - (double)i0;
		n->bus[0][i] = (float)(ch0[i0] + (ch0[i1] - ch0[i0]) * frac);
		n->bus[1][i] = (float)(ch1[i0] + (ch1[i1] - ch1[i0]) * frac);
		n->position += r;
		n->played_frames += fabs(r);
		if (n->loop) {
			double ll = loop_e - loop_s;
			if (ll > 0) {
				while (n->position >= loop_e)
					n->position -= ll;
				while (n->position < loop_s && r < 0)
					n->position += ll;
			}
		}
	}
	if (finished) {
		n->playback_state = NX_AUDIO_SOURCE_FINISHED;
		n->playing = false;
	}
}

// Naive (non-band-limited) oscillator: adequate for UI beeps, chords, and
// simple tone tests. Higher partials on non-sine waves will alias above
// Nyquist; a proper band-limited implementation (PolyBLEP or BLIT) is a
// future upgrade. `n->position` is repurposed as the fractional-cycle phase
// in [0, 1).
void process_oscillator(nx_audio_graph *g, nx_audio_node *n, double t0) {
	zero_bus(n);
	n->bus_ch = 1;
	if (!n->started || n->playback_state == NX_AUDIO_SOURCE_FINISHED)
		return;

	double inv_sr = 1.0 / g->sample_rate;
	float freq[Q];
	float detune[Q];
	param_fill(&n->params[0], t0, inv_sr, freq, Q);
	param_fill(&n->params[1], t0, inv_sr, detune, Q);

	bool finished = false;
	for (int i = 0; i < Q; i++) {
		double t = t0 + i * inv_sr;
		if (n->stop_time >= 0 && t >= n->stop_time) {
			finished = true;
			break;
		}
		if (t < n->start_time)
			continue;
		if (!n->playing) {
			n->playing = true;
			n->position = 0; // phase in [0, 1)
		}
		double f = (double)freq[i] * pow(2.0, (double)detune[i] / 1200.0);
		double p = n->position;
		float s;
		switch (n->oscillator_type) {
		case NX_AUDIO_OSCILLATOR_SQUARE:
			s = p < 0.5 ? 1.f : -1.f;
			break;
		case NX_AUDIO_OSCILLATOR_SAWTOOTH: {
			double x = p - floor(p + 0.5);
			s = (float)(2.0 * x);
			break;
		}
		case NX_AUDIO_OSCILLATOR_TRIANGLE: {
			double q = p - 0.25;
			q -= floor(q);
			s = (float)(4.0 * fabs(q - 0.5) - 1.0);
			break;
		}
		case NX_AUDIO_OSCILLATOR_SINE:
		default:
			s = (float)sin(p * 6.283185307179586);
			break;
		}
		n->bus[0][i] = s;
		n->bus[1][i] = s;
		p += f * inv_sr;
		// Wrap phase; handle both positive and (rare) negative-frequency
		// cases without a floor call in the hot path when unnecessary.
		if (p >= 1.0)
			p -= floor(p);
		else if (p < 0.0)
			p -= floor(p);
		n->position = p;
	}
	if (finished) {
		n->playback_state = NX_AUDIO_SOURCE_FINISHED;
		n->playing = false;
	}
}

void process_stream_source(nx_audio_graph *g, nx_audio_node *n) {
	zero_bus(n);
	n->bus_ch = 2;
	if (!n->stream_ring || !n->stream_playing.load(std::memory_order_relaxed))
		return;
	uint64_t read = n->stream_read_pos.load(std::memory_order_relaxed);
	uint64_t write = n->stream_write_pos.load(std::memory_order_acquire);
	uint32_t avail = (uint32_t)(write - read);
	uint32_t frames = avail < (uint32_t)Q ? avail : (uint32_t)Q;
	const float *ring = n->stream_ring.get();
	for (uint32_t i = 0; i < frames; i++) {
		uint32_t idx = (uint32_t)((read + i) % n->stream_capacity);
		n->bus[0][i] = ring[idx * 2];
		n->bus[1][i] = ring[idx * 2 + 1];
	}
	// Underrun: the remainder of the bus stays silent and is NOT counted as
	// consumed, so the media clock only advances for real audio.
	if (avail < (uint32_t)Q)
		n->stream_underrun_count.fetch_add(1, std::memory_order_relaxed);
	n->stream_read_pos.store(read + frames, std::memory_order_release);
	(void)g;
}

void process_gain(nx_audio_graph *g, nx_audio_node *n, double t0) {
	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	sum_inputs(g, n, t0, in, &ch);
	n->bus_ch = ch;
	float gain[Q];
	param_fill(&n->params[0], t0, 1.0 / g->sample_rate, gain, Q);
	for (int i = 0; i < Q; i++) {
		n->bus[0][i] = in[0][i] * gain[i];
		n->bus[1][i] = in[1][i] * gain[i];
	}
}

void process_stereo_panner(nx_audio_graph *g, nx_audio_node *n, double t0) {
	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	sum_inputs(g, n, t0, in, &ch);
	n->bus_ch = 2; // panner output is always stereo
	float pan[Q];
	param_fill(&n->params[0], t0, 1.0 / g->sample_rate, pan, Q);
	constexpr double HALF_PI = 1.57079632679489661923;
	if (ch == 1) {
		// Mono input (L==R): x = (pan + 1) / 2.
		for (int i = 0; i < Q; i++) {
			double x = (pan[i] + 1) / 2;
			n->bus[0][i] = (float)(in[0][i] * cos(x * HALF_PI));
			n->bus[1][i] = (float)(in[0][i] * sin(x * HALF_PI));
		}
	} else {
		// Stereo input, per the spec's equal-power algorithm.
		for (int i = 0; i < Q; i++) {
			double p = pan[i];
			double x = p <= 0 ? p + 1 : p;
			float gl = (float)cos(x * HALF_PI);
			float gr = (float)sin(x * HALF_PI);
			if (p <= 0) {
				n->bus[0][i] = in[0][i] + in[1][i] * gl;
				n->bus[1][i] = in[1][i] * gr;
			} else {
				n->bus[0][i] = in[0][i] * gl;
				n->bus[1][i] = in[1][i] + in[0][i] * gr;
			}
		}
	}
}

void process_destination(nx_audio_graph *g, nx_audio_node *n, double t0) {
	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	sum_inputs(g, n, t0, in, &ch);
	n->bus_ch = ch;
	memcpy(n->bus, in, sizeof(in));
}

// AnalyserNode: passthrough (output == summed input, like the destination)
// PLUS a tap that appends the downmixed (L+R)/2 samples to the ring buffer
// so JS can visualise the signal. Runs under the graph mutex (render thread).
void process_analyser(nx_audio_graph *g, nx_audio_node *n, double t0) {
	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	sum_inputs(g, n, t0, in, &ch);
	n->bus_ch = ch;
	memcpy(n->bus, in, sizeof(in));
	if (n->analyser_ring && n->analyser_ring_size > 0) {
		uint32_t mask = n->analyser_ring_size - 1; // size is a power of two
		for (int i = 0; i < Q; i++) {
			float s = (in[0][i] + in[1][i]) * 0.5f;
			n->analyser_ring[(uint32_t)(n->analyser_write_pos & mask)] = s;
			n->analyser_write_pos++;
		}
	}
}

// DelayNode: a stereo delay line. The current quantum's OUTPUT is read from the
// line (samples written in prior quanta) BEFORE this quantum's INPUT is summed
// and appended. Because we mark the node processed before pulling inputs, a
// feedback path that loops back into this delay reads the delayed output
// computed here (via process_node's memoization short-circuit) instead of
// re-entering — which is exactly how a DelayNode legalises an audio cycle. The
// delay is k-rate and floored at one render quantum so every read index is
// strictly behind the write head (and the Web Audio cycle constraint holds).
void process_delay(nx_audio_graph *g, nx_audio_node *n, double t0) {
	double sr = g->sample_rate;
	double delay_sec = param_value_at(&n->params[0], t0);
	if (delay_sec > n->delay_max_time)
		delay_sec = n->delay_max_time;
	if (delay_sec < 0)
		delay_sec = 0;
	double delay_frames = delay_sec * sr;
	if (delay_frames < (double)Q)
		delay_frames = (double)Q;

	const float *ring = n->delay_ring.get();
	uint32_t cap = n->delay_capacity;
	n->bus_ch = n->delay_bus_ch;
	if (ring && cap > 0) {
		for (int i = 0; i < Q; i++) {
			double rp =
			    (double)(n->delay_write_pos + (uint64_t)i) - delay_frames;
			if (rp < 0) {
				n->bus[0][i] = 0.f;
				n->bus[1][i] = 0.f;
				continue;
			}
			uint64_t i0 = (uint64_t)rp;
			double frac = rp - (double)i0;
			uint32_t k0 = (uint32_t)(i0 % cap);
			uint32_t k1 = (uint32_t)((i0 + 1) % cap);
			float l0 = ring[k0 * 2], l1 = ring[k1 * 2];
			float r0 = ring[k0 * 2 + 1], r1 = ring[k1 * 2 + 1];
			n->bus[0][i] = (float)(l0 + (l1 - l0) * frac);
			n->bus[1][i] = (float)(r0 + (r1 - r0) * frac);
		}
	} else {
		zero_bus(n);
	}

	// Mark processed BEFORE summing inputs so a feedback cycle reads the output
	// computed above rather than re-rendering or being silenced by the cycle
	// guard. process_node re-affirms this after we return (idempotent).
	n->processed_quantum = g->quantum_id;

	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	sum_inputs(g, n, t0, in, &ch);
	n->delay_bus_ch = ch;

	if (n->delay_ring && cap > 0) {
		float *w = n->delay_ring.get();
		for (int i = 0; i < Q; i++) {
			uint32_t k = (uint32_t)((n->delay_write_pos + (uint64_t)i) % cap);
			w[k * 2] = in[0][i];
			w[k * 2 + 1] = in[1][i];
		}
		n->delay_write_pos += Q;
	}
}

// DynamicsCompressorNode: a feed-forward peak compressor. A soft-knee static
// curve maps the per-sample input level to a target gain reduction, which is
// smoothed with attack/release one-pole envelopes. No automatic makeup gain
// (output is only ever attenuated), so it is safe to drop onto a master bus.
// Params are evaluated k-rate (once per quantum), like the other effect nodes.
void process_dynamics_compressor(nx_audio_graph *g, nx_audio_node *n,
                                 double t0) {
	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	sum_inputs(g, n, t0, in, &ch);
	n->bus_ch = ch;

	double sr = g->sample_rate;
	double threshold = param_value_at(&n->params[0], t0); // dB
	double knee = param_value_at(&n->params[1], t0);      // dB
	double ratio = param_value_at(&n->params[2], t0);
	double attack = param_value_at(&n->params[3], t0);  // s
	double release = param_value_at(&n->params[4], t0); // s
	if (ratio < 1.0)
		ratio = 1.0;
	double att_coef = attack > 0 ? exp(-1.0 / (attack * sr)) : 0.0;
	double rel_coef = release > 0 ? exp(-1.0 / (release * sr)) : 0.0;

	double env = n->comp_env_db; // current gain reduction (dB, <= 0)
	for (int i = 0; i < Q; i++) {
		double s = fabs(in[0][i]);
		double sr2 = fabs(in[1][i]);
		if (sr2 > s)
			s = sr2;
		double in_db = s > 1e-6 ? 20.0 * log10(s) : -120.0;
		// Soft-knee gain computer -> output level (dB).
		double diff = in_db - threshold;
		double out_db;
		if (2.0 * diff < -knee) {
			out_db = in_db;
		} else if (knee > 0 && 2.0 * fabs(diff) <= knee) {
			double x = diff + knee / 2.0;
			out_db = in_db + (1.0 / ratio - 1.0) * x * x / (2.0 * knee);
		} else {
			out_db = threshold + diff / ratio;
		}
		double target = out_db - in_db; // gain reduction, <= 0
		if (target < env)
			env = target + (env - target) * att_coef; // attack (more reduction)
		else
			env = target + (env - target) * rel_coef; // release
		double gain = pow(10.0, env / 20.0);
		n->bus[0][i] = (float)(in[0][i] * gain);
		n->bus[1][i] = (float)(in[1][i] * gain);
	}
	n->comp_env_db = env;
	n->comp_reduction = (float)env;
}

// BiquadFilterNode: the Audio EQ Cookbook filters, in the exact forms the Web
// Audio spec prescribes. Writes the five normalised coefficients (already
// divided by a0) into `b` (b0, b1, b2) and `a` (a1, a2).
//
// Two spec quirks worth flagging: `f0` is expressed relative to NYQUIST (so
// w0 = pi * f0 / nyquist), and for lowpass / highpass only, `Q` is in
// DECIBELS — a page setting Q = 1 on a lowpass means 1 dB, not a linear 1.
// The degenerate cases (cutoff at or past DC / Nyquist) are spelled out by the
// spec too, and matter here: pixi-sound's TelephoneFilter leaves Q at its
// default, and a badly-behaved filter at the band edges rings or blows up.
void biquad_coeffs(int type, double sample_rate, double frequency,
                   double detune, double q, double gain_db, double *b,
                   double *a) {
	// Passthrough by default; the degenerate branches below fall back to it.
	b[0] = 1;
	b[1] = 0;
	b[2] = 0;
	a[0] = 0;
	a[1] = 0;

	double nyquist = sample_rate * 0.5;
	double f0 = frequency * pow(2.0, detune / 1200.0);
	double fn = nyquist > 0 ? f0 / nyquist : 0; // normalised to Nyquist
	double amp = pow(10.0, gain_db / 40.0);     // shelf / peaking "A"

	if (!(fn > 0)) {
		// Cutoff at (or below) DC.
		switch (type) {
		case NX_AUDIO_BIQUAD_LOWPASS:
		case NX_AUDIO_BIQUAD_BANDPASS:
			b[0] = 0; // nothing passes
			break;
		case NX_AUDIO_BIQUAD_HIGHSHELF:
			b[0] = amp * amp; // the whole band is inside the shelf
			break;
		default:
			break; // passthrough
		}
		return;
	}
	if (fn >= 1) {
		// Cutoff at (or above) Nyquist.
		switch (type) {
		case NX_AUDIO_BIQUAD_HIGHPASS:
		case NX_AUDIO_BIQUAD_BANDPASS:
			b[0] = 0;
			break;
		case NX_AUDIO_BIQUAD_LOWSHELF:
			b[0] = amp * amp;
			break;
		default:
			break;
		}
		return;
	}

	double w0 = NX_PI * fn;
	double cw = cos(w0), sw = sin(w0);
	double alpha, a0;
	bool ok = true;

	switch (type) {
	case NX_AUDIO_BIQUAD_HIGHPASS:
		alpha = sw / (2.0 * pow(10.0, q / 20.0)); // Q in dB
		a0 = 1 + alpha;
		b[0] = ((1 + cw) / 2) / a0;
		b[1] = (-(1 + cw)) / a0;
		b[2] = b[0];
		a[0] = (-2 * cw) / a0;
		a[1] = (1 - alpha) / a0;
		break;
	case NX_AUDIO_BIQUAD_BANDPASS:
		if (!(q > 0)) { // zero bandwidth -> nothing passes
			b[0] = 0;
			break;
		}
		alpha = sw / (2.0 * q);
		a0 = 1 + alpha;
		b[0] = alpha / a0;
		b[1] = 0;
		b[2] = -alpha / a0;
		a[0] = (-2 * cw) / a0;
		a[1] = (1 - alpha) / a0;
		break;
	case NX_AUDIO_BIQUAD_NOTCH:
		if (!(q > 0)) // zero bandwidth -> nothing is notched out
			break;
		alpha = sw / (2.0 * q);
		a0 = 1 + alpha;
		b[0] = 1 / a0;
		b[1] = (-2 * cw) / a0;
		b[2] = b[0];
		a[0] = b[1];
		a[1] = (1 - alpha) / a0;
		break;
	case NX_AUDIO_BIQUAD_ALLPASS:
		if (!(q > 0)) { // degenerates to a sign flip
			b[0] = -1;
			break;
		}
		alpha = sw / (2.0 * q);
		a0 = 1 + alpha;
		b[0] = (1 - alpha) / a0;
		b[1] = (-2 * cw) / a0;
		b[2] = 1; // (1 + alpha) / a0
		a[0] = b[1];
		a[1] = b[0];
		break;
	case NX_AUDIO_BIQUAD_PEAKING:
		if (!(q > 0)) { // zero bandwidth -> the peak gain applies everywhere
			b[0] = amp * amp;
			break;
		}
		alpha = sw / (2.0 * q);
		a0 = 1 + alpha / amp;
		b[0] = (1 + alpha * amp) / a0;
		b[1] = (-2 * cw) / a0;
		b[2] = (1 - alpha * amp) / a0;
		a[0] = b[1];
		a[1] = (1 - alpha / amp) / a0;
		break;
	case NX_AUDIO_BIQUAD_LOWSHELF: {
		// S = 1, so alpha = sin(w0)/2 * sqrt(2), and the cookbook's
		// 2 * sqrt(A) * alpha collapses to sqrt(2 * A) * sin(w0).
		double tsa = sqrt(2.0 * amp) * sw;
		a0 = (amp + 1) + (amp - 1) * cw + tsa;
		b[0] = (amp * ((amp + 1) - (amp - 1) * cw + tsa)) / a0;
		b[1] = (2 * amp * ((amp - 1) - (amp + 1) * cw)) / a0;
		b[2] = (amp * ((amp + 1) - (amp - 1) * cw - tsa)) / a0;
		a[0] = (-2 * ((amp - 1) + (amp + 1) * cw)) / a0;
		a[1] = ((amp + 1) + (amp - 1) * cw - tsa) / a0;
		break;
	}
	case NX_AUDIO_BIQUAD_HIGHSHELF: {
		double tsa = sqrt(2.0 * amp) * sw;
		a0 = (amp + 1) - (amp - 1) * cw + tsa;
		b[0] = (amp * ((amp + 1) + (amp - 1) * cw + tsa)) / a0;
		b[1] = (-2 * amp * ((amp - 1) + (amp + 1) * cw)) / a0;
		b[2] = (amp * ((amp + 1) + (amp - 1) * cw - tsa)) / a0;
		a[0] = (2 * ((amp - 1) - (amp + 1) * cw)) / a0;
		a[1] = ((amp + 1) - (amp - 1) * cw - tsa) / a0;
		break;
	}
	case NX_AUDIO_BIQUAD_LOWPASS:
	default:
		alpha = sw / (2.0 * pow(10.0, q / 20.0)); // Q in dB
		a0 = 1 + alpha;
		b[0] = ((1 - cw) / 2) / a0;
		b[1] = (1 - cw) / a0;
		b[2] = b[0];
		a[0] = (-2 * cw) / a0;
		a[1] = (1 - alpha) / a0;
		break;
	}
	// A NaN anywhere (a pathological parameter combination) would latch into
	// the filter state and silence the node forever — fall back to passthrough.
	for (int i = 0; i < 3; i++)
		if (!isfinite(b[i]))
			ok = false;
	for (int i = 0; i < 2; i++)
		if (!isfinite(a[i]))
			ok = false;
	if (!ok) {
		b[0] = 1;
		b[1] = b[2] = a[0] = a[1] = 0;
	}
}

// BiquadFilterNode. Coefficients are recomputed once per render quantum
// (k-rate) — a filter sweep automated at audio rate steps every 2.7 ms rather
// than every sample, which is inaudible at the usual envelope / LFO speeds.
// Each channel keeps its own direct-form-I history across quanta.
void process_biquad(nx_audio_graph *g, nx_audio_node *n, double t0) {
	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	sum_inputs(g, n, t0, in, &ch);
	n->bus_ch = ch;

	double b[3], a[2];
	biquad_coeffs(n->biquad_type, g->sample_rate,
	              param_value_at(&n->params[0], t0), // frequency
	              param_value_at(&n->params[1], t0), // detune
	              param_value_at(&n->params[2], t0), // Q
	              param_value_at(&n->params[3], t0), // gain (dB)
	              b, a);

	for (int c = 0; c < NX_AUDIO_CHANNELS; c++) {
		double x1 = n->biquad_x1[c], x2 = n->biquad_x2[c];
		double y1 = n->biquad_y1[c], y2 = n->biquad_y2[c];
		for (int i = 0; i < Q; i++) {
			double x = in[c][i];
			double y = b[0] * x + b[1] * x1 + b[2] * x2 - a[0] * y1 - a[1] * y2;
			x2 = x1;
			x1 = x;
			y2 = y1;
			y1 = y;
			n->bus[c][i] = (float)y;
		}
		// Reset a state that has gone non-finite (it would otherwise poison
		// every later sample) or decayed into denormals (which trap).
		if (!isfinite(y1) || !isfinite(y2)) {
			x1 = x2 = y1 = y2 = 0;
			for (int i = 0; i < Q; i++)
				if (!isfinite(n->bus[c][i]))
					n->bus[c][i] = 0.f;
		} else if (fabs(y1) + fabs(y2) + fabs(x1) + fabs(x2) < 1e-25) {
			x1 = x2 = y1 = y2 = 0;
		}
		n->biquad_x1[c] = x1;
		n->biquad_x2[c] = x2;
		n->biquad_y1[c] = y1;
		n->biquad_y2[c] = y2;
	}
}

// Transforms the just-completed input window and convolves it against every
// partition of the impulse response, leaving the next block of output in
// `conv_out`. Overlap-save: the window is [previous block | current block], the
// impulse partitions are zero-padded to the same length, and only the second
// half of the inverse transform — the half free of circular wrap-around — is
// true linear convolution, so only that half is kept.
void convolver_process_block(nx_audio_node *n, const nx_audio_convolver_ir *ir,
                             bool window_zero) {
	const uint32_t N = ir->fft_size, B = ir->block, bins = ir->bins,
	               K = ir->partitions;
	n->conv_fdl_slot = (n->conv_fdl_slot + 1) % K;
	n->conv_fdl_zero[n->conv_fdl_slot] = window_zero ? 1 : 0;
	float *re = n->conv_scratch_re.data();
	float *im = n->conv_scratch_im.data();
	float *ar = n->conv_acc_re.data();
	float *ai = n->conv_acc_im.data();

	for (int c = 0; c < NX_AUDIO_CHANNELS; c++) {
		float *slot =
		    n->conv_fdl[c].data() + (size_t)n->conv_fdl_slot * bins * 2;
		if (window_zero) {
			// Transforming silence just produces zeros — skip the work.
			memset(slot, 0, (size_t)bins * 2 * sizeof(float));
		} else {
			const float *win = n->conv_in[c].data();
			for (uint32_t i = 0; i < N; i++) {
				re[i] = win[i];
				im[i] = 0.f;
			}
			ir->fft.forward(re, im);
			for (uint32_t i = 0; i < bins; i++) {
				slot[i * 2] = re[i];
				slot[i * 2 + 1] = im[i];
			}
		}

		memset(ar, 0, (size_t)bins * sizeof(float));
		memset(ai, 0, (size_t)bins * sizeof(float));
		// Partition k convolves against the input window from k blocks ago.
		const float *spec = ir->spectra[ir->channels == 2 ? c : 0].data();
		for (uint32_t k = 0; k < K; k++) {
			uint32_t s = (n->conv_fdl_slot + K - k) % K;
			if (n->conv_fdl_zero[s])
				continue; // silence convolves to silence
			const float *fd = n->conv_fdl[c].data() + (size_t)s * bins * 2;
			const float *hp = spec + (size_t)k * bins * 2;
			for (uint32_t i = 0; i < bins; i++) {
				float xr = fd[i * 2], xi = fd[i * 2 + 1];
				float hr = hp[i * 2], hi = hp[i * 2 + 1];
				ar[i] += xr * hr - xi * hi;
				ai[i] += xr * hi + xi * hr;
			}
		}

		// The accumulated half-spectrum is Hermitian (both operands came from
		// real signals) — mirror it back to full length for the inverse.
		for (uint32_t i = 0; i < bins; i++) {
			re[i] = ar[i];
			im[i] = ai[i];
		}
		for (uint32_t i = bins; i < N; i++) {
			re[i] = ar[N - i];
			im[i] = -ai[N - i];
		}
		ir->fft.inverse(re, im);
		memcpy(n->conv_out[c].data(), re + B, (size_t)B * sizeof(float));
	}
}

// ConvolverNode. The node is block-based while the graph is quantum-based, so
// each quantum emits Q frames of the block computed at the last block boundary
// and appends Q frames of input; every `block / Q` quanta a new output block is
// produced. That is exactly where the node's latency comes from: one partition.
void process_convolver(nx_audio_graph *g, nx_audio_node *n, double t0) {
	std::shared_ptr<nx_audio_convolver_ir> ir = n->conv_ir;
	float in[NX_AUDIO_CHANNELS][Q];
	int ch;
	if (!ir || ir->partitions == 0 || n->conv_out[0].empty()) {
		// No impulse response set: the spec says the node outputs silence.
		// Inputs are still pulled so upstream sources advance their playheads.
		sum_inputs(g, n, t0, in, &ch);
		zero_bus(n);
		n->bus_ch = 1;
		return;
	}
	const uint32_t B = ir->block;

	for (int c = 0; c < NX_AUDIO_CHANNELS; c++)
		memcpy(n->bus[c], n->conv_out[c].data() + n->conv_out_pos,
		       (size_t)Q * sizeof(float));
	n->bus_ch = n->conv_out_ch;
	n->conv_out_pos += Q;

	sum_inputs(g, n, t0, in, &ch);
	for (int c = 0; c < NX_AUDIO_CHANNELS; c++)
		memcpy(n->conv_in[c].data() + B + n->conv_fill, in[c],
		       (size_t)Q * sizeof(float));
	n->conv_fill += Q;
	if (n->conv_fill < B)
		return;

	// Track runs of silent input. Once the run is longer than the delay line,
	// every stored spectrum is zero, so the reverb tail has fully decayed and
	// there is nothing left to compute until signal returns — which matters,
	// because a convolver left connected to a quiet bus would otherwise burn
	// its full cost forever.
	bool zero = true;
	for (int c = 0; c < NX_AUDIO_CHANNELS && zero; c++) {
		const float *cur = n->conv_in[c].data() + B;
		for (uint32_t i = 0; i < B; i++) {
			if (cur[i] != 0.f) {
				zero = false;
				break;
			}
		}
	}
	if (!zero)
		n->conv_zero_blocks = 0;
	else if (n->conv_zero_blocks <= ir->partitions)
		n->conv_zero_blocks++;

	if (n->conv_zero_blocks > ir->partitions) {
		for (int c = 0; c < NX_AUDIO_CHANNELS; c++)
			memset(n->conv_out[c].data(), 0, (size_t)B * sizeof(float));
	} else {
		// The overlap-save window spans this block and the previous one, so it
		// is only silent if both were.
		convolver_process_block(n, ir.get(), zero && n->conv_prev_zero);
	}
	n->conv_prev_zero = zero;
	n->conv_out_ch = ir->channels == 2 ? 2 : ch;
	n->conv_fill = 0;
	n->conv_out_pos = 0;
	// The block just consumed becomes the overlap for the next one.
	for (int c = 0; c < NX_AUDIO_CHANNELS; c++)
		memmove(n->conv_in[c].data(), n->conv_in[c].data() + B,
		        (size_t)B * sizeof(float));
}

void process_node(nx_audio_graph *g, nx_audio_node *n, double t0) {
	if (n->processed_quantum == g->quantum_id)
		return; // already rendered this quantum (fan-out memoization)
	if (n->processing) {
		// Cycle without a DelayNode to break it (a DelayNode marks itself
		// processed before pulling inputs, so it is short-circuited above and
		// never reaches here). Such a cycle is not legal in Web Audio — break
		// it with silence.
		zero_bus(n);
		return;
	}
	n->processing = true;
	switch (n->type) {
	case NX_AUDIO_NODE_BUFFER_SOURCE:
		process_buffer_source(g, n, t0);
		break;
	case NX_AUDIO_NODE_OSCILLATOR:
		process_oscillator(g, n, t0);
		break;
	case NX_AUDIO_NODE_STREAM_SOURCE:
		process_stream_source(g, n);
		break;
	case NX_AUDIO_NODE_GAIN:
		process_gain(g, n, t0);
		break;
	case NX_AUDIO_NODE_STEREO_PANNER:
		process_stereo_panner(g, n, t0);
		break;
	case NX_AUDIO_NODE_DESTINATION:
		process_destination(g, n, t0);
		break;
	case NX_AUDIO_NODE_ANALYSER:
		process_analyser(g, n, t0);
		break;
	case NX_AUDIO_NODE_DELAY:
		process_delay(g, n, t0);
		break;
	case NX_AUDIO_NODE_DYNAMICS_COMPRESSOR:
		process_dynamics_compressor(g, n, t0);
		break;
	case NX_AUDIO_NODE_BIQUAD_FILTER:
		process_biquad(g, n, t0);
		break;
	case NX_AUDIO_NODE_CONVOLVER:
		process_convolver(g, n, t0);
		break;
	}
	n->processing = false;
	n->processed_quantum = g->quantum_id;
}

bool mark_silent(nx_audio_node *n);

// Marks nodes that can never produce signal again. A source that has finished
// playing is silent forever (Web Audio lets the implementation drop it from the
// rendering graph at that point — Chrome disconnects it immediately rather than
// waiting for the wrapper to be collected). Silence then propagates through
// pure multiplicative nodes: a gain or panner whose every input is permanently
// silent outputs exact zeros regardless of its own parameter value.
//
// Without this, a fire-and-forget app (one oscillator+gain per tracker row, the
// standard Web Audio idiom) keeps every note it has ever played in the render
// walk, so per-quantum cost grows without bound.
// True when `n` has inputs and every one of them is permanently silent, so this
// node will never be fed signal again. An input-less node returns false: nothing
// stops the page connecting a source into it later.
bool inputs_all_silent(nx_audio_node *n) {
	if (n->inputs.empty())
		return false;
	n->silent_checking = true;
	bool all = true;
	for (nx_audio_node *src : n->inputs) {
		if (!mark_silent(src)) {
			all = false;
			break;
		}
	}
	n->silent_checking = false;
	return all;
}

// A filter still rings after its input dies, so it may only be declared silent
// once its own state has decayed. 1e-9 is ~180 dB below full scale — far below
// one LSB of the s16 output — and stops a denormal trickle from keeping a node
// alive forever.
bool biquad_state_decayed(const nx_audio_node *n) {
	constexpr double EPS = 1e-9;
	for (int c = 0; c < NX_AUDIO_CHANNELS; c++) {
		if (fabs(n->biquad_x1[c]) > EPS || fabs(n->biquad_x2[c]) > EPS ||
		    fabs(n->biquad_y1[c]) > EPS || fabs(n->biquad_y2[c]) > EPS)
			return false;
	}
	return true;
}

bool mark_silent(nx_audio_node *n) {
	if (n->silent)
		return true;
	if (n->silent_checking)
		return false; // in a cycle — assume live
	// Every node type is listed: a new one must make an explicit choice here
	// rather than silently defaulting to "never silent", which is how adding
	// BiquadFilterNode re-opened the unbounded-render-cost bug this guards
	// against. Do NOT reintroduce a `default:` label.
	switch (n->type) {
	case NX_AUDIO_NODE_OSCILLATOR:
	case NX_AUDIO_NODE_BUFFER_SOURCE:
		// A source that has finished playing can never produce signal again.
		if (!n->started || n->playback_state != NX_AUDIO_SOURCE_FINISHED)
			return false;
		break;
	case NX_AUDIO_NODE_GAIN:
	case NX_AUDIO_NODE_STEREO_PANNER:
		// Purely multiplicative: silent in, exact zeros out, whatever the
		// parameter value.
		if (!inputs_all_silent(n))
			return false;
		break;
	case NX_AUDIO_NODE_BIQUAD_FILTER:
		// IIR: wait for the ringing to die before declaring it silent.
		if (!inputs_all_silent(n) || !biquad_state_decayed(n))
			return false;
		break;
	case NX_AUDIO_NODE_CONVOLVER: {
		// Wait for the reverb tail. process_convolver already tracks a run of
		// silent input blocks and, once it exceeds the frequency-delay line,
		// zeroes its output — at that point every stored spectrum is zero and
		// nothing is left to ring out.
		std::shared_ptr<nx_audio_convolver_ir> ir = n->conv_ir;
		if (!ir || ir->partitions == 0)
			return false; // no impulse response yet; one may still be assigned
		// `conv_zero_blocks` saturates at partitions + 1, and the block that
		// pushes it past `partitions` is the one that zeroes conv_out — so
		// "> partitions" is both the engine's own tail-decayed condition and
		// the tightest test available here.
		if (!inputs_all_silent(n) || n->conv_zero_blocks <= ir->partitions)
			return false;
		break;
	}
	case NX_AUDIO_NODE_DESTINATION:
	case NX_AUDIO_NODE_STREAM_SOURCE:
	case NX_AUDIO_NODE_ANALYSER:
	case NX_AUDIO_NODE_DELAY:
	case NX_AUDIO_NODE_DYNAMICS_COMPRESSOR:
		// Never declared permanently silent: the destination and the media
		// stream source are endpoints, an analyser must keep filling its ring
		// for JS to read, and the delay (ring contents) and compressor
		// (envelope) carry state that the cheap checks above cannot settle.
		// These are all long-lived singletons, so they never accumulate.
		return false;
	}
	n->silent = true;
	zero_bus(n);
	// The edges are deliberately left in place: a per-voice gain that the app
	// reuses (connect it to the destination once, feed it a new source per
	// note) must stay wired up. `sum_inputs` skips silent inputs by flag, and
	// `nx_audio_node_connect` clears the flag when signal can flow again.
	return true;
}

// Renders one 128-frame quantum into the destination bus and advances time.
// Caller holds the graph mutex.
void render_quantum(nx_audio_graph *g) {
	g->quantum_id++;
	double t0 = (double)g->frames_rendered / g->sample_rate;
	// Fold away settled automation events before evaluating so param timelines
	// stay bounded no matter how much automation was scheduled. Events in
	// (t0, t0 + Q/sr) are in the future relative to t0 and are retained, so
	// this quantum's evaluation is unaffected.
	for (nx_audio_node *n : g->nodes) {
		if (n->silent)
			continue;
		mark_silent(n);
		if (n->silent)
			continue;
		for (nx_audio_param &p : n->params)
			param_prune(&p, t0);
	}
	process_node(g, g->destination, t0);
	// Sources not reachable from the destination still progress through their
	// schedule (so `ended` fires even for unconnected/indirect sources).
	for (nx_audio_node *n : g->nodes) {
		if (!n->silent &&
		    (n->type == NX_AUDIO_NODE_BUFFER_SOURCE ||
		     n->type == NX_AUDIO_NODE_OSCILLATOR) &&
		    n->processed_quantum != g->quantum_id)
			process_node(g, n, t0);
	}
	// AnalyserNodes are taps, not connected to the destination, so the walk
	// above never reaches them. Process each explicitly (which pulls its
	// upstream chain — already-processed nodes are memoized) so its ring keeps
	// filling with the live signal for JS visualisation.
	for (nx_audio_node *n : g->nodes) {
		if (n->type == NX_AUDIO_NODE_ANALYSER &&
		    n->processed_quantum != g->quantum_id)
			process_node(g, n, t0);
	}
	g->frames_rendered += Q;
}

void graph_destroy(nx_audio_graph *g) {
	for (nx_audio_node *n : g->nodes)
		delete n;
	delete g;
}

nx_audio_node *node_new(nx_audio_graph *g, nx_audio_node_type type,
                        double aux = 0.0) {
	nx_audio_node *n = new nx_audio_node();
	n->graph = g;
	n->type = type;
	zero_bus(n);
	switch (type) {
	case NX_AUDIO_NODE_GAIN: {
		nx_audio_param gain;
		gain.value = 1.f;
		n->params.push_back(gain);
		break;
	}
	case NX_AUDIO_NODE_STEREO_PANNER: {
		nx_audio_param pan;
		pan.value = 0.f;
		pan.min_value = -1.f;
		pan.max_value = 1.f;
		n->params.push_back(pan);
		break;
	}
	case NX_AUDIO_NODE_BUFFER_SOURCE: {
		nx_audio_param rate;
		rate.value = 1.f;
		n->params.push_back(rate);
		nx_audio_param detune;
		detune.value = 0.f;
		n->params.push_back(detune);
		break;
	}
	case NX_AUDIO_NODE_OSCILLATOR: {
		nx_audio_param frequency;
		frequency.value = 440.f;
		// Per spec: min = -Nyquist, max = +Nyquist.
		float nyq = (float)(g->sample_rate * 0.5);
		frequency.min_value = -nyq;
		frequency.max_value = nyq;
		n->params.push_back(frequency);
		nx_audio_param detune;
		detune.value = 0.f;
		detune.min_value = -153600.f;
		detune.max_value = 153600.f;
		n->params.push_back(detune);
		break;
	}
	case NX_AUDIO_NODE_STREAM_SOURCE: {
		// One second of buffering at the graph rate.
		n->stream_capacity = (uint32_t)g->sample_rate;
		n->stream_ring = std::make_unique<float[]>(
		    (size_t)n->stream_capacity * NX_AUDIO_CHANNELS);
		break;
	}
	case NX_AUDIO_NODE_ANALYSER: {
		// Ring capacity = the max supported fftSize (32768), rounded to a power
		// of two so the write index masks cleanly. Zero-initialised so an
		// analyser read before any audio renders returns silence.
		n->analyser_ring_size = 32768;
		n->analyser_ring =
		    std::make_unique<float[]>((size_t)n->analyser_ring_size);
		n->analyser_write_pos = 0;
		break;
	}
	case NX_AUDIO_NODE_DELAY: {
		nx_audio_param delay_time;
		delay_time.value = 0.f;
		delay_time.min_value = 0.f;
		// aux = maxDelayTime (seconds); default 1s, clamp to the spec's (0, 180)
		// bound. Sizes the delay line and caps delayTime.
		double max_time = aux;
		if (!(max_time > 0.0))
			max_time = 1.0;
		if (max_time > 180.0)
			max_time = 180.0;
		n->delay_max_time = max_time;
		delay_time.max_value = (float)max_time;
		n->params.push_back(delay_time);
		// maxDelay frames + one render quantum of headroom (+ a little slack for
		// the interpolation's +1 read and rounding). Zero-initialised silence.
		uint32_t frames = (uint32_t)(max_time * g->sample_rate) + Q + 4;
		n->delay_capacity = frames;
		n->delay_ring =
		    std::make_unique<float[]>((size_t)frames * NX_AUDIO_CHANNELS);
		n->delay_write_pos = 0;
		break;
	}
	case NX_AUDIO_NODE_DYNAMICS_COMPRESSOR: {
		// threshold (dB), knee (dB), ratio, attack (s), release (s) — spec
		// defaults and value ranges.
		nx_audio_param threshold;
		threshold.value = -24.f;
		threshold.min_value = -100.f;
		threshold.max_value = 0.f;
		n->params.push_back(threshold);
		nx_audio_param knee;
		knee.value = 30.f;
		knee.min_value = 0.f;
		knee.max_value = 40.f;
		n->params.push_back(knee);
		nx_audio_param ratio;
		ratio.value = 12.f;
		ratio.min_value = 1.f;
		ratio.max_value = 20.f;
		n->params.push_back(ratio);
		nx_audio_param attack;
		attack.value = 0.003f;
		attack.min_value = 0.f;
		attack.max_value = 1.f;
		n->params.push_back(attack);
		nx_audio_param release;
		release.value = 0.25f;
		release.min_value = 0.f;
		release.max_value = 1.f;
		n->params.push_back(release);
		break;
	}
	case NX_AUDIO_NODE_BIQUAD_FILTER: {
		// frequency (Hz), detune (cents), Q, gain (dB) — spec defaults.
		nx_audio_param frequency;
		frequency.value = 350.f;
		frequency.min_value = 0.f;
		frequency.max_value = (float)(g->sample_rate * 0.5);
		n->params.push_back(frequency);
		nx_audio_param detune;
		detune.value = 0.f;
		detune.min_value = -153600.f;
		detune.max_value = 153600.f;
		n->params.push_back(detune);
		nx_audio_param qp;
		qp.value = 1.f;
		n->params.push_back(qp);
		nx_audio_param gain;
		gain.value = 0.f;
		n->params.push_back(gain);
		break;
	}
	case NX_AUDIO_NODE_CONVOLVER:
		// No params, and no buffers until an impulse response is assigned —
		// until then the node renders silence.
		break;
	case NX_AUDIO_NODE_DESTINATION:
		break;
	}
	g->nodes.push_back(n);
	return n;
}

// Clears the permanently-silent flag on `n` and everything downstream of it.
// Called when a new input is connected, which can make a previously dead
// sub-graph carry signal again. Guarded against fan-in revisits and cycles by
// stopping as soon as a node is already clear.
void clear_silent_downstream(nx_audio_node *n) {
	if (!n || !n->silent)
		return;
	n->silent = false;
	for (nx_audio_node *dst : n->outputs)
		clear_silent_downstream(dst);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API (all functions lock the graph mutex)
// ---------------------------------------------------------------------------

nx_audio_graph *nx_audio_graph_create(double sample_rate) {
	nx_audio_graph *g = new nx_audio_graph();
	g->sample_rate = sample_rate;
	g->destination = node_new(g, NX_AUDIO_NODE_DESTINATION);
	return g;
}

void nx_audio_graph_ref(nx_audio_graph *g) { g->refs.fetch_add(1); }

void nx_audio_graph_unref(nx_audio_graph *g) {
	if (g->refs.fetch_sub(1) == 1)
		graph_destroy(g);
}

double nx_audio_graph_current_time(nx_audio_graph *g) {
	std::lock_guard<std::mutex> lock(g->mutex);
	return (double)g->frames_rendered / g->sample_rate;
}

void nx_audio_graph_set_suspended(nx_audio_graph *g, bool suspended) {
	std::lock_guard<std::mutex> lock(g->mutex);
	g->suspended = suspended;
}

nx_audio_node *nx_audio_node_create(nx_audio_graph *g, nx_audio_node_type type,
                                    double aux) {
	nx_audio_graph_ref(g);
	std::lock_guard<std::mutex> lock(g->mutex);
	return node_new(g, type, aux);
}

void nx_audio_node_release(nx_audio_node *n) {
	nx_audio_graph *g = n->graph;
	{
		std::lock_guard<std::mutex> lock(g->mutex);
		if (n->type != NX_AUDIO_NODE_DESTINATION) {
			for (nx_audio_node *src : n->inputs)
				src->outputs.erase(std::remove(src->outputs.begin(),
				                               src->outputs.end(), n),
				                   src->outputs.end());
			for (nx_audio_node *dst : n->outputs)
				dst->inputs.erase(
				    std::remove(dst->inputs.begin(), dst->inputs.end(), n),
				    dst->inputs.end());
			g->nodes.erase(std::remove(g->nodes.begin(), g->nodes.end(), n),
			               g->nodes.end());
			delete n;
		}
		// The destination node is graph-owned; only the ref is dropped.
	}
	nx_audio_graph_unref(g);
}

void nx_audio_node_connect(nx_audio_node *src, nx_audio_node *dst) {
	std::lock_guard<std::mutex> lock(src->graph->mutex);
	// Idempotent: multiple connections between the same nodes collapse.
	if (std::find(dst->inputs.begin(), dst->inputs.end(), src) !=
	    dst->inputs.end())
		return;
	dst->inputs.push_back(src);
	src->outputs.push_back(dst);
	// A node that was declared permanently silent can carry signal again once
	// something is connected into it (the common "reuse one voice gain per
	// note" idiom). Clear the flag over the whole downstream cone.
	clear_silent_downstream(dst);
}

void nx_audio_node_disconnect(nx_audio_node *src, nx_audio_node *dst) {
	std::lock_guard<std::mutex> lock(src->graph->mutex);
	if (dst) {
		dst->inputs.erase(
		    std::remove(dst->inputs.begin(), dst->inputs.end(), src),
		    dst->inputs.end());
		src->outputs.erase(
		    std::remove(src->outputs.begin(), src->outputs.end(), dst),
		    src->outputs.end());
	} else {
		for (nx_audio_node *d : src->outputs)
			d->inputs.erase(
			    std::remove(d->inputs.begin(), d->inputs.end(), src),
			    d->inputs.end());
		src->outputs.clear();
	}
}

nx_audio_param *nx_audio_node_param(nx_audio_node *n, int index) {
	if (index < 0 || (size_t)index >= n->params.size())
		return nullptr;
	return &n->params[index];
}

float nx_audio_param_value(nx_audio_node *n, nx_audio_param *p) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	double t = (double)n->graph->frames_rendered / n->graph->sample_rate;
	return param_value_at(p, t);
}

void nx_audio_param_set_value(nx_audio_node *n, nx_audio_param *p,
                              float value) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	p->value = value;
	if (!p->events.empty()) {
		// Per spec, setting .value with scheduled automation behaves like
		// setValueAtTime(value, currentTime).
		double t = (double)n->graph->frames_rendered / n->graph->sample_rate;
		nx_audio_param_event e = {};
		e.type = NX_AUDIO_PARAM_SET_VALUE;
		e.time = t;
		e.value = value;
		param_insert_event(p, std::move(e));
	}
}

void nx_audio_param_schedule(nx_audio_node *n, nx_audio_param *p,
                             nx_audio_param_event_type type, double time,
                             float value, double time_constant) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	nx_audio_param_event e = {};
	e.type = type;
	e.time = time;
	e.value = value;
	e.time_constant = time_constant;
	param_insert_event(p, std::move(e));
	// Collapse now-settled past events. This is the hot path for controls that
	// automate every frame (e.g. a knob emitting setTargetAtTime per move) and
	// keeps the list bounded even when the graph is suspended (no render pass
	// runs to prune it). currentTime = frames_rendered / sample_rate.
	param_prune(p, (double)n->graph->frames_rendered / n->graph->sample_rate);
}

void nx_audio_param_set_value_curve(nx_audio_node *n, nx_audio_param *p,
                                    const float *curve, size_t len,
                                    double start_time, double duration) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	nx_audio_param_event e = {};
	e.type = NX_AUDIO_PARAM_SET_VALUE_CURVE;
	e.time = start_time;
	e.duration = duration;
	e.curve.assign(curve, curve + len);
	param_insert_event(p, std::move(e));
}

void nx_audio_param_cancel(nx_audio_node *n, nx_audio_param *p, double time) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	p->events.erase(std::remove_if(p->events.begin(), p->events.end(),
	                               [time](const nx_audio_param_event &e) {
		                               return e.time >= time;
	                               }),
	                p->events.end());
}

void nx_audio_source_set_buffer(nx_audio_node *n, const float *const *channels,
                                int num_channels, uint32_t length,
                                double sample_rate,
                                std::vector<std::shared_ptr<void>> holds) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	n->buffer_channels.assign(channels, channels + num_channels);
	n->buffer_holds = std::move(holds);
	n->buffer_length = length;
	n->buffer_sample_rate = sample_rate;
}

void nx_audio_source_set_loop(nx_audio_node *n, bool loop, double loop_start,
                              double loop_end) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	n->loop = loop;
	n->loop_start = loop_start;
	n->loop_end = loop_end;
}

void nx_audio_source_start(nx_audio_node *n, double when, double offset,
                           double duration) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	if (n->started)
		return; // JS throws InvalidStateError before reaching here
	n->started = true;
	n->playback_state = NX_AUDIO_SOURCE_SCHEDULED;
	n->start_time = when;
	n->start_offset = offset;
	n->duration = duration;
}

void nx_audio_source_stop(nx_audio_node *n, double when) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	double t = (double)n->graph->frames_rendered / n->graph->sample_rate;
	if (when < t)
		when = t;
	n->stop_time = when;
}

int nx_audio_source_playback_state(nx_audio_node *n) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	return n->playback_state;
}

void nx_audio_oscillator_set_type(nx_audio_node *n, int type) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	n->oscillator_type = type;
}

// ---------------------------------------------------------------------------
// BiquadFilterNode
// ---------------------------------------------------------------------------

void nx_audio_biquad_set_type(nx_audio_node *n, int type) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	if (n->type != NX_AUDIO_NODE_BIQUAD_FILTER)
		return;
	if (type < NX_AUDIO_BIQUAD_LOWPASS || type > NX_AUDIO_BIQUAD_ALLPASS)
		return;
	if (n->biquad_type == type)
		return;
	n->biquad_type = type;
	// The history belongs to the old transfer function; carrying it into a
	// different filter shape is what makes a live `filter.type = ...` switch
	// pop or, with a high-Q filter, ring.
	for (int c = 0; c < NX_AUDIO_CHANNELS; c++) {
		n->biquad_x1[c] = n->biquad_x2[c] = 0;
		n->biquad_y1[c] = n->biquad_y2[c] = 0;
	}
}

void nx_audio_biquad_frequency_response(nx_audio_node *n,
                                        const float *frequency_hz,
                                        float *mag_response,
                                        float *phase_response, uint32_t count) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	// The param reads below are positional — refuse anything but a biquad.
	if (n->type != NX_AUDIO_NODE_BIQUAD_FILTER || n->params.size() < 4) {
		for (uint32_t i = 0; i < count; i++)
			mag_response[i] = phase_response[i] = 0.f;
		return;
	}
	double t0 = (double)n->graph->frames_rendered / n->graph->sample_rate;
	double b[3], a[2];
	biquad_coeffs(n->biquad_type, n->graph->sample_rate,
	              param_value_at(&n->params[0], t0),
	              param_value_at(&n->params[1], t0),
	              param_value_at(&n->params[2], t0),
	              param_value_at(&n->params[3], t0), b, a);
	double nyquist = n->graph->sample_rate * 0.5;
	for (uint32_t i = 0; i < count; i++) {
		double f = frequency_hz[i];
		if (!(f >= 0) || f > nyquist) {
			// Per spec, a frequency outside [0, Nyquist] yields NaN.
			mag_response[i] = (float)NAN;
			phase_response[i] = (float)NAN;
			continue;
		}
		// H(e^jw) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
		double w = NX_PI * (nyquist > 0 ? f / nyquist : 0);
		double c1 = cos(w), s1 = sin(w);
		double c2 = cos(2 * w), s2 = sin(2 * w);
		double nr = b[0] + b[1] * c1 + b[2] * c2;
		double ni = -(b[1] * s1 + b[2] * s2);
		double dr = 1 + a[0] * c1 + a[1] * c2;
		double di = -(a[0] * s1 + a[1] * s2);
		double den = dr * dr + di * di;
		if (den <= 0) {
			mag_response[i] = (float)INFINITY;
			phase_response[i] = 0.f;
			continue;
		}
		double hr = (nr * dr + ni * di) / den;
		double hi = (ni * dr - nr * di) / den;
		mag_response[i] = (float)sqrt(hr * hr + hi * hi);
		phase_response[i] = (float)atan2(hi, hr);
	}
}

// ---------------------------------------------------------------------------
// ConvolverNode
// ---------------------------------------------------------------------------

namespace {

// The spec's normalisation: scale the response so that swapping a convolver in
// does not change perceived loudness. The constants are the ones every engine
// uses (they come from the reference implementation) — an RMS normalisation
// with a fixed calibration gain, referenced to 44.1 kHz.
float convolver_normalization_scale(const std::vector<float> *channels,
                                    int num_channels, uint32_t length,
                                    double sample_rate) {
	constexpr double GAIN_CALIBRATION = 0.00125;
	constexpr double GAIN_CALIBRATION_SAMPLE_RATE = 44100;
	constexpr double MIN_POWER = 0.000125;
	if (num_channels <= 0 || length == 0)
		return 1.f;
	double power = 0;
	for (int c = 0; c < num_channels; c++) {
		const float *d = channels[c].data();
		double cp = 0;
		for (uint32_t i = 0; i < length; i++)
			cp += (double)d[i] * (double)d[i];
		power += cp;
	}
	power = sqrt(power / ((double)num_channels * (double)length));
	if (!isfinite(power) || power < MIN_POWER)
		power = MIN_POWER;
	double scale = GAIN_CALIBRATION / power;
	if (sample_rate > 0)
		scale *= GAIN_CALIBRATION_SAMPLE_RATE / sample_rate;
	return (float)scale;
}

// Everything a ConvolverNode needs for a new impulse response, built off the
// graph mutex (transforming a multi-second response is tens of milliseconds of
// work — far too long to hold up the render thread) and then moved in wholesale.
struct convolver_state {
	std::shared_ptr<nx_audio_convolver_ir> ir;
	std::vector<float> in[NX_AUDIO_CHANNELS];
	std::vector<float> out[NX_AUDIO_CHANNELS];
	std::vector<float> fdl[NX_AUDIO_CHANNELS];
	std::vector<uint8_t> fdl_zero;
	std::vector<float> scratch_re, scratch_im, acc_re, acc_im;
};

void build_convolver_state(convolver_state *st, const float *const *channels,
                           int num_channels, uint32_t length,
                           double sample_rate, double graph_rate,
                           bool normalize) {
	// A 4-channel ("true stereo") response is matrixed down to its first two
	// channels; anything beyond two is otherwise ignored.
	int nch = num_channels >= 2 ? 2 : 1;

	// Resample to the graph rate if the page handed us a buffer recorded at a
	// different one, then truncate to the memory cap.
	std::vector<float> data[NX_AUDIO_CHANNELS];
	uint32_t len = length;
	bool resample = sample_rate > 0 && graph_rate > 0 &&
	                fabs(sample_rate - graph_rate) > 1e-6;
	if (resample) {
		double ratio = graph_rate / sample_rate;
		uint64_t want = (uint64_t)((double)length * ratio);
		len = (uint32_t)(want > 0 ? want : 1);
	}
	uint32_t max_len = (uint32_t)(graph_rate * NX_AUDIO_CONVOLVER_MAX_SECONDS);
	if (max_len > 0 && len > max_len)
		len = max_len;
	for (int c = 0; c < nch; c++) {
		data[c].resize(len);
		const float *src = channels[c];
		if (!resample) {
			for (uint32_t i = 0; i < len && i < length; i++)
				data[c][i] = src[i];
		} else {
			double step = sample_rate / graph_rate;
			for (uint32_t i = 0; i < len; i++) {
				double pos = (double)i * step;
				uint32_t i0 = (uint32_t)pos;
				if (i0 >= length) {
					data[c][i] = 0.f;
					continue;
				}
				uint32_t i1 = i0 + 1 < length ? i0 + 1 : i0;
				double frac = pos - (double)i0;
				data[c][i] = (float)(src[i0] + (src[i1] - src[i0]) * frac);
			}
		}
	}

	float scale =
	    normalize ? convolver_normalization_scale(data, nch, len, sample_rate)
	              : 1.f;

	auto ir = std::make_shared<nx_audio_convolver_ir>();
	ir->block = NX_AUDIO_CONVOLVER_BLOCK;
	ir->fft_size = ir->block * 2;
	ir->bins = ir->fft_size / 2 + 1;
	ir->partitions = (len + ir->block - 1) / ir->block;
	if (ir->partitions == 0)
		ir->partitions = 1;
	ir->channels = nch;
	ir->fft.init(ir->fft_size);

	std::vector<float> re(ir->fft_size), im(ir->fft_size);
	for (int c = 0; c < nch; c++) {
		ir->spectra[c].assign((size_t)ir->partitions * ir->bins * 2, 0.f);
		for (uint32_t k = 0; k < ir->partitions; k++) {
			// Partition k, zero-padded to the full transform length: the
			// padding is what turns the circular product back into a linear
			// convolution over the kept half.
			uint32_t off = k * ir->block;
			for (uint32_t i = 0; i < ir->fft_size; i++) {
				uint32_t j = off + i;
				re[i] = (i < ir->block && j < len) ? data[c][j] * scale : 0.f;
				im[i] = 0.f;
			}
			ir->fft.forward(re.data(), im.data());
			float *dst = ir->spectra[c].data() + (size_t)k * ir->bins * 2;
			for (uint32_t i = 0; i < ir->bins; i++) {
				dst[i * 2] = re[i];
				dst[i * 2 + 1] = im[i];
			}
		}
	}

	for (int c = 0; c < NX_AUDIO_CHANNELS; c++) {
		st->in[c].assign(ir->fft_size, 0.f);
		st->out[c].assign(ir->block, 0.f);
		st->fdl[c].assign((size_t)ir->partitions * ir->bins * 2, 0.f);
	}
	// Every slot starts as a (zero) silent spectrum.
	st->fdl_zero.assign(ir->partitions, 1);
	st->scratch_re.assign(ir->fft_size, 0.f);
	st->scratch_im.assign(ir->fft_size, 0.f);
	st->acc_re.assign(ir->bins, 0.f);
	st->acc_im.assign(ir->bins, 0.f);
	st->ir = std::move(ir);
}

} // namespace

void nx_audio_convolver_set_buffer(nx_audio_node *n,
                                   const float *const *channels,
                                   int num_channels, uint32_t length,
                                   double sample_rate, bool normalize) {
	nx_audio_graph *g = n->graph;
	double graph_rate;
	{
		std::lock_guard<std::mutex> lock(g->mutex);
		if (n->type != NX_AUDIO_NODE_CONVOLVER)
			return;
		graph_rate = g->sample_rate;
	}

	convolver_state st;
	if (channels && num_channels > 0 && length > 0 && sample_rate > 0) {
		build_convolver_state(&st, channels, num_channels, length, sample_rate,
		                      graph_rate, normalize);
	}

	std::lock_guard<std::mutex> lock(g->mutex);
	n->conv_ir = std::move(st.ir);
	for (int c = 0; c < NX_AUDIO_CHANNELS; c++) {
		n->conv_in[c] = std::move(st.in[c]);
		n->conv_out[c] = std::move(st.out[c]);
		n->conv_fdl[c] = std::move(st.fdl[c]);
	}
	n->conv_fdl_zero = std::move(st.fdl_zero);
	n->conv_scratch_re = std::move(st.scratch_re);
	n->conv_scratch_im = std::move(st.scratch_im);
	n->conv_acc_re = std::move(st.acc_re);
	n->conv_acc_im = std::move(st.acc_im);
	n->conv_fdl_slot = 0;
	n->conv_fill = 0;
	n->conv_out_pos = 0;
	n->conv_zero_blocks = 0;
	n->conv_prev_zero = true;
	n->conv_out_ch = 1;
}

float nx_audio_compressor_reduction(nx_audio_node *n) {
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	return n->comp_reduction;
}

void nx_audio_analyser_get_float_time_data(nx_audio_node *n, float *out,
                                           uint32_t count) {
	if (!n || !out || count == 0)
		return;
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	uint32_t cap = n->analyser_ring ? n->analyser_ring_size : 0;
	uint64_t total = n->analyser_write_pos;
	uint32_t mask = cap ? (cap - 1) : 0;
	// Return the last `count` samples: positions [total-count, total-1].
	// Positions before 0 (not enough rendered yet) or older than the ring
	// still holds (overwritten) read as silence.
	for (uint32_t i = 0; i < count; i++) {
		int64_t pos = (int64_t)total - (int64_t)count + (int64_t)i;
		if (cap == 0 || pos < 0 ||
		    (uint64_t)((int64_t)total - pos) > (uint64_t)cap) {
			out[i] = 0.f;
		} else {
			out[i] = n->analyser_ring[(uint32_t)((uint64_t)pos & mask)];
		}
	}
}

uint32_t nx_audio_stream_writable(nx_audio_node *n) {
	if (!n->stream_ring)
		return 0;
	uint64_t read = n->stream_read_pos.load(std::memory_order_acquire);
	uint64_t write = n->stream_write_pos.load(std::memory_order_relaxed);
	return n->stream_capacity - (uint32_t)(write - read);
}

uint32_t nx_audio_stream_write(nx_audio_node *n, const float *interleaved,
                               uint32_t frames) {
	if (!n->stream_ring)
		return 0;
	uint64_t read = n->stream_read_pos.load(std::memory_order_acquire);
	uint64_t write = n->stream_write_pos.load(std::memory_order_relaxed);
	uint32_t writable = n->stream_capacity - (uint32_t)(write - read);
	if (frames > writable)
		frames = writable;
	float *ring = n->stream_ring.get();
	for (uint32_t i = 0; i < frames; i++) {
		uint32_t idx = (uint32_t)((write + i) % n->stream_capacity);
		ring[idx * 2] = interleaved[i * 2];
		ring[idx * 2 + 1] = interleaved[i * 2 + 1];
	}
	n->stream_write_pos.store(write + frames, std::memory_order_release);
	return frames;
}

void nx_audio_stream_set_playing(nx_audio_node *n, bool playing) {
	n->stream_playing.store(playing, std::memory_order_relaxed);
}

uint64_t nx_audio_stream_consumed(nx_audio_node *n) {
	return n->stream_read_pos.load(std::memory_order_acquire);
}

uint32_t nx_audio_stream_pending(nx_audio_node *n) {
	if (!n->stream_ring)
		return 0;
	uint64_t read = n->stream_read_pos.load(std::memory_order_acquire);
	uint64_t write = n->stream_write_pos.load(std::memory_order_relaxed);
	return (uint32_t)(write - read);
}

uint64_t nx_audio_stream_underrun_count(nx_audio_node *n) {
	return n->stream_underrun_count.load(std::memory_order_relaxed);
}

void nx_audio_stream_flush(nx_audio_node *n) {
	// Producer is parked (decoder contract); empty the ring by advancing the
	// read position to the write position. Take the graph mutex so this
	// cannot interleave with a render quantum's read-modify-write.
	std::lock_guard<std::mutex> lock(n->graph->mutex);
	n->stream_read_pos.store(
	    n->stream_write_pos.load(std::memory_order_relaxed),
	    std::memory_order_release);
	// Ledger #114 diag — reset underrun counter on flush so per-play stats
	// are meaningful (avoid seek events polluting the underrun window).
	n->stream_underrun_count.store(0, std::memory_order_relaxed);
}

void nx_audio_graph_render_s16(nx_audio_graph *g, int16_t *out,
                               uint32_t frames) {
	std::lock_guard<std::mutex> lock(g->mutex);
	if (g->suspended || g->closed) {
		memset(out, 0, (size_t)frames * NX_AUDIO_CHANNELS * sizeof(int16_t));
		return;
	}
	uint32_t done = 0;
	while (done < frames) {
		render_quantum(g);
		uint32_t n = frames - done < Q ? frames - done : Q;
		const float *l = g->destination->bus[0];
		const float *r = g->destination->bus[1];
		for (uint32_t i = 0; i < n; i++) {
			float fl = clampf(l[i], -1.f, 1.f);
			float fr = clampf(r[i], -1.f, 1.f);
			out[(done + i) * 2] = (int16_t)lrintf(fl * 32767.f);
			out[(done + i) * 2 + 1] = (int16_t)lrintf(fr * 32767.f);
		}
		done += n;
	}
}

void nx_audio_graph_render_offline(nx_audio_graph *g, float *const *channels,
                                   int num_channels, uint32_t length) {
	std::lock_guard<std::mutex> lock(g->mutex);
	uint32_t done = 0;
	while (done < length) {
		render_quantum(g);
		uint32_t n = length - done < Q ? length - done : Q;
		const float *l = g->destination->bus[0];
		const float *r = g->destination->bus[1];
		if (num_channels == 1) {
			// Mono destination: speakers downmix = 0.5 * (L + R).
			for (uint32_t i = 0; i < n; i++)
				channels[0][done + i] = 0.5f * (l[i] + r[i]);
		} else {
			for (uint32_t i = 0; i < n; i++) {
				channels[0][done + i] = l[i];
				channels[1][done + i] = r[i];
			}
		}
		done += n;
	}
}
