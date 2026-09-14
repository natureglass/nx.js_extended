import { test } from '../src/tap';

// Web Audio API conformance tests. All rendering goes through
// OfflineAudioContext so the output is deterministic and does not depend on
// Chrome's autoplay policy (a real-time AudioContext would start "suspended"
// in headless Chrome without a user gesture).

const RATE = 48000;

function throwsName(
	t: any,
	fn: () => unknown,
	name: string,
	msg: string,
): void {
	try {
		fn();
		t.fail(msg);
	} catch (err: any) {
		t.equal(err?.name, name, msg);
	}
}

function closeTo(a: number, b: number, eps = 1e-4): boolean {
	return Math.abs(a - b) <= eps;
}

function maxDiff(data: Float32Array, expected: (i: number) => number): number {
	let max = 0;
	for (let i = 0; i < data.length; i++) {
		const d = Math.abs(data[i] - expected(i));
		if (d > max) max = d;
	}
	return max;
}

// --- AudioBuffer ---

test('AudioBuffer basics', (t) => {
	const buf = new AudioBuffer({
		numberOfChannels: 2,
		length: 1000,
		sampleRate: RATE,
	});
	t.equal(buf.numberOfChannels, 2, 'numberOfChannels is 2');
	t.equal(buf.length, 1000, 'length is 1000');
	t.equal(buf.sampleRate, RATE, 'sampleRate is 48000');
	t.ok(closeTo(buf.duration, 1000 / RATE, 1e-9), 'duration is length/rate');

	const ch0 = buf.getChannelData(0);
	t.ok(ch0 instanceof Float32Array, 'getChannelData returns Float32Array');
	t.equal(ch0.length, 1000, 'channel data length matches');
	t.equal(ch0[0], 0, 'channel data is zero-initialized');
	t.equal(buf.getChannelData(0), ch0, 'getChannelData returns same array');

	ch0[0] = 0.5;
	const dest = new Float32Array(4);
	buf.copyFromChannel(dest, 0);
	t.equal(dest[0], 0.5, 'copyFromChannel copies data');

	const src = new Float32Array([1, 2, 3, 4]);
	buf.copyToChannel(src, 1, 2);
	t.equal(buf.getChannelData(1)[2], 1, 'copyToChannel honors bufferOffset');
	t.equal(buf.getChannelData(1)[5], 4, 'copyToChannel copies all values');

	throwsName(
		t,
		() => buf.getChannelData(2),
		'IndexSizeError',
		'getChannelData throws for invalid channel',
	);
	throwsName(
		t,
		() =>
			new AudioBuffer({ numberOfChannels: 0, length: 10, sampleRate: RATE }),
		'NotSupportedError',
		'constructor throws for 0 channels',
	);
	throwsName(
		t,
		() =>
			new AudioBuffer({ numberOfChannels: 1, length: 0, sampleRate: RATE }),
		'NotSupportedError',
		'constructor throws for 0 length',
	);
	throwsName(
		t,
		() =>
			new AudioBuffer({ numberOfChannels: 1, length: 10, sampleRate: 1 }),
		'NotSupportedError',
		'constructor throws for bad sampleRate',
	);
});

// --- OfflineAudioContext basics ---

test('OfflineAudioContext basics', (t) => {
	const ctx = new OfflineAudioContext(1, 1280, RATE);
	t.equal(ctx.length, 1280, 'length is 1280');
	t.equal(ctx.sampleRate, RATE, 'sampleRate is 48000');
	t.equal(ctx.state, 'suspended', 'initial state is suspended');
	t.equal(ctx.currentTime, 0, 'initial currentTime is 0');
	t.ok(ctx.destination instanceof AudioDestinationNode, 'destination type');
	t.equal(ctx.destination.numberOfInputs, 1, 'destination numberOfInputs');
	t.equal(ctx.destination.numberOfOutputs, 0, 'destination numberOfOutputs');
	t.equal(ctx.destination.context, ctx, 'destination context is ctx');
});

// --- Source playback through gain ---

test('buffer source through gain', async (t) => {
	const N = 1280;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	const data = buffer.getChannelData(0);
	for (let i = 0; i < N; i++) data[i] = 1;

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	t.equal(gain.gain.value, 1, 'gain defaults to 1');
	gain.gain.value = 0.25;
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const rendered = await ctx.startRendering();
	t.equal(rendered.length, N, 'rendered buffer length');
	t.equal(rendered.numberOfChannels, 1, 'rendered buffer channels');
	t.equal(rendered.sampleRate, RATE, 'rendered buffer sampleRate');
	const out = rendered.getChannelData(0);
	t.ok(maxDiff(out, () => 0.25) < 1e-6, 'all samples scaled by gain');
	t.equal(ctx.state, 'closed', 'state is closed after rendering');
});

// --- Sample-accurate playback (no processing) ---

test('buffer source passthrough is sample-accurate', async (t) => {
	const N = 512;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	const data = buffer.getChannelData(0);
	for (let i = 0; i < N; i++) data[i] = Math.sin(i / 10);

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	source.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(
		maxDiff(out, (i) => Math.sin(i / 10)) < 1e-6,
		'output matches buffer exactly',
	);
});

// --- start() offset and duration ---

test('buffer source start offset and duration', async (t) => {
	const N = 1024;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N * 2, RATE);
	const data = buffer.getChannelData(0);
	for (let i = 0; i < data.length; i++) data[i] = i / data.length;

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	source.connect(ctx.destination);
	// Start 256 frames into the buffer, play 128 frames worth.
	source.start(0, 256 / RATE, 128 / RATE);

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(closeTo(out[0], data[256], 1e-6), 'first sample honors offset');
	t.ok(closeTo(out[127], data[256 + 127], 1e-6), 'last sample before cutoff');
	t.equal(out[200], 0, 'silent after duration elapses');
	t.equal(out[N - 1], 0, 'silent at end');
});

// --- Delayed start ---

test('buffer source delayed start', async (t) => {
	const N = 512;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	source.connect(ctx.destination);
	source.start(256 / RATE);

	const out = (await ctx.startRendering()).getChannelData(0);
	t.equal(out[0], 0, 'silent before start time');
	t.equal(out[255], 0, 'silent right before start time');
	t.equal(out[256], 1, 'plays at start time');
	t.equal(out[N - 1], 1, 'still playing at end');
});

// --- Looping ---

test('buffer source looping', async (t) => {
	const N = 1000;
	const period = 100;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, period, RATE);
	const data = buffer.getChannelData(0);
	for (let i = 0; i < period; i++) data[i] = (i + 1) / period;

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	source.loop = true;
	t.equal(source.loop, true, 'loop property set');
	source.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(
		closeTo(out[0], data[0], 1e-6) &&
			closeTo(out[period], data[0], 1e-6) &&
			closeTo(out[5 * period + 50], data[50], 1e-6),
		'looped content repeats',
	);
	t.ok(closeTo(out[N - 1], data[(N - 1) % period], 1e-6), 'loops to the end');
});

// --- playbackRate ---

test('buffer source playbackRate', async (t) => {
	const N = 256;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N * 2, RATE);
	const data = buffer.getChannelData(0);
	for (let i = 0; i < data.length; i++) data[i] = i;

	const source = ctx.createBufferSource();
	t.equal(source.playbackRate.value, 1, 'playbackRate defaults to 1');
	t.equal(source.detune.value, 0, 'detune defaults to 0');
	source.buffer = buffer;
	source.playbackRate.value = 2;
	source.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(
		maxDiff(out, (i) => 2 * i) < 1e-2,
		'output advances at twice the rate',
	);
});

// --- Gain automation: setValueAtTime + linearRamp ---

test('gain linear ramp automation', async (t) => {
	const N = 1280;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	gain.gain.setValueAtTime(0, 0);
	gain.gain.linearRampToValueAtTime(1, N / RATE);
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(
		maxDiff(out, (i) => i / N) < 1e-3,
		'gain ramps linearly from 0 to 1',
	);
});

// --- Gain automation: setValueAtTime steps ---

test('gain setValueAtTime steps', async (t) => {
	const N = 1024;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	gain.gain.setValueAtTime(0.25, 0);
	gain.gain.setValueAtTime(0.75, 512 / RATE);
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(closeTo(out[0], 0.25, 1e-6), 'first step value');
	t.ok(closeTo(out[511], 0.25, 1e-6), 'holds until second step');
	t.ok(closeTo(out[512], 0.75, 1e-6), 'second step value');
	t.ok(closeTo(out[N - 1], 0.75, 1e-6), 'holds to the end');
});

// --- Exponential ramp ---

test('gain exponential ramp automation', async (t) => {
	const N = 1280;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	gain.gain.setValueAtTime(0.01, 0);
	gain.gain.exponentialRampToValueAtTime(1, N / RATE);
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(
		maxDiff(out, (i) => 0.01 * Math.pow(100, i / N)) < 1e-3,
		'gain ramps exponentially',
	);
});

// --- setTargetAtTime ---

test('gain setTargetAtTime automation', async (t) => {
	const N = 1280;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const tc = 0.005;
	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	gain.gain.setValueAtTime(1, 0);
	gain.gain.setTargetAtTime(0, 0, tc);
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(
		maxDiff(out, (i) => Math.exp(-(i / RATE) / tc)) < 1e-3,
		'gain decays toward target',
	);
});

// --- setTargetAtTime event pruning ---
// A control that automates every frame (e.g. a knob dragged with a
// setTargetAtTime per pointermove) schedules a flood of events. Re-issuing the
// SAME decay toward an unchanged target with an unchanged time constant is
// analytically identical to a single continuous decay exp(-t/tc), so the
// rendered output must match no matter how many events accumulated (and were
// folded away by param_prune). Exercises pruning across dozens of quanta.

test('gain setTargetAtTime spam stays accurate (event pruning)', async (t) => {
	const N = 6144; // 48 render quanta
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const tc = 0.01;
	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	gain.gain.setValueAtTime(1, 0);
	// ~192 setTargetAtTime events toward the same target.
	for (let f = 0; f < N; f += 32) gain.gain.setTargetAtTime(0, f / RATE, tc);
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(
		maxDiff(out, (i) => Math.exp(-(i / RATE) / tc)) < 1e-3,
		'output matches a single continuous decay despite ~192 scheduled events',
	);
});

// Rendering the same automation intent with a single event vs. a per-frame
// flood must produce (near) sample-identical output — pruning is only a
// performance optimization and must never change what is heard. Also validates
// the non-zero base-fold path against the closed form B + (A - B) e^(-t/tc).

test('event pruning is behavior-preserving', async (t) => {
	const N = 4096;
	const tc = 0.008;
	const A = 1;
	const B = 0.3;

	async function render(spam: boolean): Promise<Float32Array> {
		const ctx = new OfflineAudioContext(1, N, RATE);
		const buffer = ctx.createBuffer(1, N, RATE);
		buffer.getChannelData(0).fill(1);
		const source = ctx.createBufferSource();
		source.buffer = buffer;
		const gain = ctx.createGain();
		gain.gain.setValueAtTime(A, 0);
		if (spam) {
			for (let f = 0; f < N; f += 16)
				gain.gain.setTargetAtTime(B, f / RATE, tc);
		} else {
			gain.gain.setTargetAtTime(B, 0, tc);
		}
		source.connect(gain);
		gain.connect(ctx.destination);
		source.start();
		return (await ctx.startRendering()).getChannelData(0);
	}

	const minimal = await render(false);
	const spammed = await render(true);
	let max = 0;
	for (let i = 0; i < N; i++) {
		const d = Math.abs(minimal[i] - spammed[i]);
		if (d > max) max = d;
	}
	t.ok(
		max < 1e-5,
		'spammed (pruned) render is sample-identical to the minimal one',
	);
	t.ok(
		maxDiff(spammed, (i) => B + (A - B) * Math.exp(-(i / RATE) / tc)) < 1e-3,
		'spammed render matches the closed-form decay',
	);
});

// --- setValueCurveAtTime ---

test('gain setValueCurveAtTime automation', async (t) => {
	const N = 1024;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const curve = new Float32Array([0, 1, 0.5]);
	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	gain.gain.setValueCurveAtTime(curve, 0, 512 / RATE);
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(closeTo(out[0], 0, 1e-5), 'curve start value');
	t.ok(closeTo(out[128], 0.5, 2e-3), 'curve interpolates first segment');
	t.ok(closeTo(out[256], 1, 2e-3), 'curve midpoint value');
	t.ok(closeTo(out[384], 0.75, 2e-3), 'curve interpolates second segment');
	t.ok(closeTo(out[600], 0.5, 1e-5), 'holds final curve value');
});

// --- cancelScheduledValues ---

test('cancelScheduledValues removes future events', async (t) => {
	const N = 1024;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const gain = ctx.createGain();
	gain.gain.setValueAtTime(0.5, 0);
	gain.gain.setValueAtTime(0.9, 512 / RATE);
	gain.gain.cancelScheduledValues(256 / RATE);
	source.connect(gain);
	gain.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(closeTo(out[0], 0.5, 1e-6), 'kept event applies');
	t.ok(closeTo(out[N - 1], 0.5, 1e-6), 'canceled event does not apply');
});

// --- StereoPannerNode ---

test('stereo panner mono input', async (t) => {
	const N = 384;
	const SQRT1_2 = Math.SQRT1_2;
	for (const [pan, gl, gr, name] of [
		[0, SQRT1_2, SQRT1_2, 'center'],
		[-1, 1, 0, 'left'],
		[1, 0, 1, 'right'],
	] as [number, number, number, string][]) {
		const ctx = new OfflineAudioContext(2, N, RATE);
		const buffer = ctx.createBuffer(1, N, RATE);
		buffer.getChannelData(0).fill(1);
		const source = ctx.createBufferSource();
		source.buffer = buffer;
		const panner = ctx.createStereoPanner();
		panner.pan.value = pan;
		source.connect(panner);
		panner.connect(ctx.destination);
		source.start();
		const rendered = await ctx.startRendering();
		const l = rendered.getChannelData(0);
		const r = rendered.getChannelData(1);
		t.ok(closeTo(l[100], gl, 1e-5), `pan ${name}: left gain`);
		t.ok(closeTo(r[100], gr, 1e-5), `pan ${name}: right gain`);
	}
});

test('stereo panner stereo input', async (t) => {
	const N = 384;
	const ctx = new OfflineAudioContext(2, N, RATE);
	const buffer = ctx.createBuffer(2, N, RATE);
	buffer.getChannelData(0).fill(0.5);
	buffer.getChannelData(1).fill(0.25);
	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const panner = ctx.createStereoPanner();
	t.equal(panner.pan.value, 0, 'pan defaults to 0');
	source.connect(panner);
	panner.connect(ctx.destination);
	source.start();
	const rendered = await ctx.startRendering();
	t.ok(
		closeTo(rendered.getChannelData(0)[100], 0.5, 1e-5),
		'pan 0 stereo passthrough left',
	);
	t.ok(
		closeTo(rendered.getChannelData(1)[100], 0.25, 1e-5),
		'pan 0 stereo passthrough right',
	);
});

// --- Stereo downmix to mono destination ---

test('stereo source downmix to mono destination', async (t) => {
	const N = 384;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(2, N, RATE);
	buffer.getChannelData(0).fill(0.8);
	buffer.getChannelData(1).fill(0.4);
	const source = ctx.createBufferSource();
	source.buffer = buffer;
	source.connect(ctx.destination);
	source.start();
	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(closeTo(out[100], 0.6, 1e-5), 'mono downmix is 0.5*(L+R)');
});

// --- Fan-in summing ---

test('multiple sources sum at destination', async (t) => {
	const N = 384;
	const ctx = new OfflineAudioContext(1, N, RATE);
	for (const value of [0.25, 0.5]) {
		const buffer = ctx.createBuffer(1, N, RATE);
		buffer.getChannelData(0).fill(value);
		const source = ctx.createBufferSource();
		source.buffer = buffer;
		source.connect(ctx.destination);
		source.start();
	}
	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(closeTo(out[100], 0.75, 1e-5), 'inputs are summed');
});

// --- ended event ---

test('source ended event fires', async (t) => {
	const N = 2048;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, 256, RATE);
	buffer.getChannelData(0).fill(1);
	const source = ctx.createBufferSource();
	source.buffer = buffer;
	source.connect(ctx.destination);
	const ended = new Promise<boolean>((resolve) => {
		const timo = setTimeout(() => resolve(false), 3000);
		source.onended = () => {
			clearTimeout(timo);
			resolve(true);
		};
	});
	source.start();
	await ctx.startRendering();
	t.ok(await ended, 'ended event fired');
});

// --- oncomplete event ---

test('OfflineAudioContext complete event', async (t) => {
	const N = 384;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const complete = new Promise<any>((resolve) => {
		const timo = setTimeout(() => resolve(null), 3000);
		ctx.oncomplete = (ev) => {
			clearTimeout(timo);
			resolve(ev);
		};
	});
	const rendered = await ctx.startRendering();
	const ev = await complete;
	t.ok(ev !== null, 'complete event fired');
	t.equal(ev?.renderedBuffer?.length, N, 'event has renderedBuffer');
	t.equal(rendered.length, N, 'startRendering resolves with buffer');
});

// --- decodeAudioData (WAV built in JS) ---

function buildWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
	const dataSize = samples.length * 2;
	const ab = new ArrayBuffer(44 + dataSize);
	const view = new DataView(ab);
	const writeStr = (off: number, s: string) => {
		for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
	};
	writeStr(0, 'RIFF');
	view.setUint32(4, 36 + dataSize, true);
	writeStr(8, 'WAVE');
	writeStr(12, 'fmt ');
	view.setUint32(16, 16, true); // fmt chunk size
	view.setUint16(20, 1, true); // PCM
	view.setUint16(22, 1, true); // mono
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * 2, true); // byte rate
	view.setUint16(32, 2, true); // block align
	view.setUint16(34, 16, true); // bits per sample
	writeStr(36, 'data');
	view.setUint32(40, dataSize, true);
	for (let i = 0; i < samples.length; i++) {
		const s = Math.max(-1, Math.min(1, samples[i]));
		view.setInt16(44 + i * 2, Math.round(s * 32767), true);
	}
	return ab;
}

test('decodeAudioData decodes WAV', async (t) => {
	const N = 4800;
	const samples = new Float32Array(N);
	for (let i = 0; i < N; i++) samples[i] = Math.sin((i / RATE) * 2 * Math.PI * 440) * 0.5;
	const wav = buildWav(samples, RATE);

	const ctx = new OfflineAudioContext(1, 384, RATE);
	const buffer = await ctx.decodeAudioData(wav);
	t.equal(buffer.numberOfChannels, 1, 'decoded channel count');
	t.equal(buffer.sampleRate, RATE, 'decoded sample rate');
	t.equal(buffer.length, N, 'decoded length');
	const data = buffer.getChannelData(0);
	t.ok(
		maxDiff(data, (i) => samples[i]) < 1e-3,
		'decoded samples match (within s16 quantization)',
	);
	t.equal(wav.byteLength, 0, 'input ArrayBuffer is detached');
});

test('decodeAudioData rejects garbage', async (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const garbage = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]).buffer;
	try {
		await ctx.decodeAudioData(garbage);
		t.fail('should have rejected');
	} catch (err: any) {
		t.equal(err.name, 'EncodingError', 'rejects with EncodingError');
	}
});

// --- API errors ---

test('Web Audio API errors', async (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const source = ctx.createBufferSource();
	const buffer = ctx.createBuffer(1, 100, RATE);

	throwsName(t, () => source.stop(), 'InvalidStateError', 'stop before start throws');

	source.buffer = buffer;
	throwsName(
		t,
		() => {
			source.buffer = buffer;
		},
		'InvalidStateError',
		'setting buffer twice throws',
	);
	source.buffer = null;
	t.ok(source.buffer === null, 'buffer can be cleared');

	source.start();
	throwsName(t, () => source.start(), 'InvalidStateError', 'start twice throws');

	const gain = ctx.createGain();
	throwsName(
		t,
		() => gain.gain.setValueAtTime(1, -1),
		'RangeError',
		'negative time throws',
	);
	throwsName(
		t,
		() => gain.gain.exponentialRampToValueAtTime(0, 1),
		'RangeError',
		'exponential ramp to 0 throws',
	);
	throwsName(
		t,
		() => gain.gain.setTargetAtTime(1, 0, -1),
		'RangeError',
		'negative timeConstant throws',
	);
	throwsName(
		t,
		() => gain.gain.setValueCurveAtTime(new Float32Array([1]), 0, 0.1),
		'InvalidStateError',
		'short curve throws',
	);

	const ctx2 = new OfflineAudioContext(1, 384, RATE);
	const gain2 = ctx2.createGain();
	throwsName(
		t,
		() => gain.connect(gain2),
		'InvalidAccessError',
		'cross-context connect throws',
	);

	throwsName(
		t,
		() => new OfflineAudioContext(0, 384, RATE),
		'NotSupportedError',
		'OfflineAudioContext with 0 channels throws',
	);
});

// --- Node properties ---

test('AudioNode properties', (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const source = ctx.createBufferSource();
	t.equal(source.numberOfInputs, 0, 'source numberOfInputs');
	t.equal(source.numberOfOutputs, 1, 'source numberOfOutputs');
	const gain = ctx.createGain();
	t.equal(gain.numberOfInputs, 1, 'gain numberOfInputs');
	t.equal(gain.numberOfOutputs, 1, 'gain numberOfOutputs');
	t.equal(gain.context, ctx, 'gain context');
	t.equal(gain.connect(ctx.destination), ctx.destination, 'connect returns destination');
	gain.disconnect();
	t.ok(source instanceof AudioScheduledSourceNode, 'source instanceof AudioScheduledSourceNode');
	t.ok(source instanceof AudioNode, 'source instanceof AudioNode');
	t.ok(gain instanceof AudioNode, 'gain instanceof AudioNode');
	t.ok(gain.gain instanceof AudioParam, 'gain.gain instanceof AudioParam');
	t.equal(gain.gain.defaultValue, 1, 'gain defaultValue');
	const panner = ctx.createStereoPanner();
	t.equal(panner.pan.minValue, -1, 'pan minValue');
	t.equal(panner.pan.maxValue, 1, 'pan maxValue');
	t.ok(ctx instanceof BaseAudioContext, 'ctx instanceof BaseAudioContext');
});

// --- DelayNode ---

test('DelayNode basics', (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const delay = ctx.createDelay();
	t.ok(delay instanceof DelayNode, 'createDelay returns a DelayNode');
	t.ok(delay instanceof AudioNode, 'DelayNode instanceof AudioNode');
	t.equal(delay.numberOfInputs, 1, 'delay numberOfInputs');
	t.equal(delay.numberOfOutputs, 1, 'delay numberOfOutputs');
	t.ok(delay.delayTime instanceof AudioParam, 'delayTime is an AudioParam');
	t.equal(delay.delayTime.value, 0, 'delayTime defaults to 0');
	t.equal(delay.delayTime.maxValue, 1, 'default maxValue is 1s');
	const d2 = ctx.createDelay(2.5);
	t.equal(d2.delayTime.maxValue, 2.5, 'maxDelayTime sets delayTime maxValue');
	throwsName(
		t,
		() => ctx.createDelay(0),
		'NotSupportedError',
		'createDelay(0) throws',
	);
	throwsName(
		t,
		() => ctx.createDelay(200),
		'NotSupportedError',
		'createDelay(>=180) throws',
	);
});

// Peak absolute value within [lo, hi). Used for cross-engine assertions that
// tolerate small (sub-quantum) timing differences between nxjs and Chrome.
function peakAbs(data: Float32Array, lo: number, hi: number): number {
	let max = 0;
	for (let i = lo; i < hi && i < data.length; i++) {
		const v = Math.abs(data[i]);
		if (v > max) max = v;
	}
	return max;
}

test('DelayNode delays the signal', async (t) => {
	const N = 2048;
	const D = 256; // delay in frames (>= one render quantum)
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	const data = buffer.getChannelData(0);
	for (let i = 0; i < N; i++) data[i] = 1;

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const delay = ctx.createDelay();
	delay.delayTime.value = D / RATE;
	source.connect(delay);
	delay.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	// Well before / well after the boundary — sample-accurate in both engines
	// (the exact boundary sample can differ by interpolation, so it's skipped).
	t.ok(closeTo(out[64], 0, 1e-4), 'silent well before the delay elapses');
	t.ok(closeTo(out[D + 128], 1, 1e-4), 'signal present well after the delay');
	t.ok(closeTo(out[N - 1], 1, 1e-4), 'steady state matches the input');
});

test('DelayNode supports a feedback echo cycle', async (t) => {
	const N = 2048;
	const D = 512; // 4 render quanta — spacing >> any one-quantum cycle latency
	const ctx = new OfflineAudioContext(1, N, RATE);
	// Single-sample impulse.
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0)[0] = 1;

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const delay = ctx.createDelay();
	delay.delayTime.value = D / RATE;
	const feedback = ctx.createGain();
	feedback.gain.value = 0.5;
	source.connect(delay);
	delay.connect(ctx.destination);
	delay.connect(feedback);
	feedback.connect(delay); // <-- the cycle a DelayNode is allowed to close
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	// Windowed peaks tolerate a possible one-quantum cycle-latency difference
	// between engines; the amplitudes (feedback 0.5 => 1, 0.5, 0.25) are
	// engine-invariant for an integer-frame delay.
	const W = 160;
	t.ok(peakAbs(out, 0, D - W) < 0.02, 'silent before the first echo');
	t.ok(closeTo(peakAbs(out, D - W, D + W), 1, 0.06), 'first echo ~1.0');
	t.ok(
		closeTo(peakAbs(out, 2 * D - W, 2 * D + W), 0.5, 0.06),
		'second echo ~0.5 (one feedback pass)',
	);
	t.ok(
		closeTo(peakAbs(out, 3 * D - W, 3 * D + W), 0.25, 0.06),
		'third echo ~0.25 (two feedback passes)',
	);
});

// --- DynamicsCompressorNode ---

test('DynamicsCompressorNode basics', (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const comp = ctx.createDynamicsCompressor();
	t.ok(
		comp instanceof DynamicsCompressorNode,
		'createDynamicsCompressor returns a DynamicsCompressorNode',
	);
	t.equal(comp.threshold.value, -24, 'threshold default');
	t.equal(comp.knee.value, 30, 'knee default');
	t.equal(comp.ratio.value, 12, 'ratio default');
	t.ok(closeTo(comp.attack.value, 0.003, 1e-6), 'attack default');
	t.equal(comp.release.value, 0.25, 'release default');
	t.equal(comp.reduction, 0, 'reduction starts at 0');
});

test('DynamicsCompressorNode compresses a loud signal', async (t) => {
	// A loud 1 kHz sine (0 dBFS) well above the threshold. Exact gain values
	// differ between engines (Chrome uses lookahead + makeup), so we assert only
	// cross-engine invariants: the compressor reports gain reduction, still
	// passes audio, and does not blow up.
	const N = 8192; // >> attack (3 ms) so the envelope settles
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	const data = buffer.getChannelData(0);
	for (let i = 0; i < N; i++) data[i] = Math.sin((2 * Math.PI * 1000 * i) / RATE);

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const comp = ctx.createDynamicsCompressor();
	comp.threshold.value = -30;
	source.connect(comp);
	comp.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	const steadyPeak = peakAbs(out, N / 2, N);
	// Cross-engine invariants only — exact levels differ (Chrome adds makeup
	// gain + lookahead; nxjs does neither).
	t.ok(comp.reduction < -3, 'reports substantial gain reduction (dB, < 0)');
	t.ok(Number.isFinite(steadyPeak) && steadyPeak > 0, 'still passes audio');
	t.ok(steadyPeak < 1.5, 'output stays bounded (no runaway gain)');
});

// --- BiquadFilterNode ---

test('BiquadFilterNode basics', (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const filter = ctx.createBiquadFilter();
	t.ok(
		filter instanceof BiquadFilterNode,
		'createBiquadFilter returns a BiquadFilterNode',
	);
	t.ok(filter instanceof AudioNode, 'BiquadFilterNode instanceof AudioNode');
	t.equal(filter.type, 'lowpass', 'type defaults to lowpass');
	t.equal(filter.frequency.value, 350, 'frequency default');
	t.equal(filter.detune.value, 0, 'detune default');
	t.equal(filter.Q.value, 1, 'Q default');
	t.equal(filter.gain.value, 0, 'gain default');
	for (const type of [
		'highpass',
		'bandpass',
		'lowshelf',
		'highshelf',
		'peaking',
		'notch',
		'allpass',
		'lowpass',
	] as BiquadFilterType[]) {
		filter.type = type;
		t.equal(filter.type, type, `type round-trips: ${type}`);
	}
});

test('BiquadFilterNode getFrequencyResponse', (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const filter = ctx.createBiquadFilter();
	filter.type = 'lowpass';
	filter.frequency.value = 1000;
	const freq = new Float32Array([20, 20000, RATE]);
	const mag = new Float32Array(3);
	const phase = new Float32Array(3);
	filter.getFrequencyResponse(freq, mag, phase);
	// The transfer function is spelled out by the spec, so these are
	// engine-invariant rather than implementation detail.
	t.ok(closeTo(mag[0], 1, 0.05), 'lowpass passes 20 Hz (|H| ~ 1)');
	t.ok(mag[1] < 0.05, 'lowpass rejects 20 kHz');
	t.ok(Number.isNaN(mag[2]), 'frequency above Nyquist yields NaN');
	t.ok(Number.isFinite(phase[0]), 'phase is finite in the passband');
	let threw = false;
	try {
		filter.getFrequencyResponse(freq, new Float32Array(2), phase);
	} catch (err) {
		threw = err instanceof TypeError;
	}
	t.ok(threw, 'mismatched array lengths throw a TypeError');
});

test('BiquadFilterNode filters the signal', async (t) => {
	// A lowpass well below the tone must remove it; well above must keep it.
	const N = 8192;
	async function renderTone(hz: number, cutoff: number): Promise<number> {
		const ctx = new OfflineAudioContext(1, N, RATE);
		const buffer = ctx.createBuffer(1, N, RATE);
		const data = buffer.getChannelData(0);
		for (let i = 0; i < N; i++)
			data[i] = Math.sin((2 * Math.PI * hz * i) / RATE);
		const source = ctx.createBufferSource();
		source.buffer = buffer;
		const filter = ctx.createBiquadFilter();
		filter.type = 'lowpass';
		filter.frequency.value = cutoff;
		source.connect(filter);
		filter.connect(ctx.destination);
		source.start();
		const out = (await ctx.startRendering()).getChannelData(0);
		return peakAbs(out, N / 2, N); // measure once settled
	}
	const passed = await renderTone(200, 2000);
	const rejected = await renderTone(12000, 2000);
	t.ok(passed > 0.8, 'lowpass 2 kHz passes a 200 Hz tone');
	t.ok(rejected < 0.1, 'lowpass 2 kHz rejects a 12 kHz tone');
});

// --- ConvolverNode ---

test('ConvolverNode basics', (t) => {
	const ctx = new OfflineAudioContext(1, 384, RATE);
	const conv = ctx.createConvolver();
	t.ok(conv instanceof ConvolverNode, 'createConvolver returns a ConvolverNode');
	t.ok(conv instanceof AudioNode, 'ConvolverNode instanceof AudioNode');
	t.equal(conv.buffer, null, 'buffer starts null');
	t.equal(conv.normalize, true, 'normalize defaults to true');
	const ir = ctx.createBuffer(1, 128, RATE);
	conv.buffer = ir;
	t.equal(conv.buffer, ir, 'buffer round-trips');
	conv.buffer = null;
	t.equal(conv.buffer, null, 'buffer can be cleared');
});

test('ConvolverNode with no impulse response is silent', async (t) => {
	const N = 4096;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, N, RATE);
	buffer.getChannelData(0).fill(1);
	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const conv = ctx.createConvolver();
	source.connect(conv);
	conv.connect(ctx.destination);
	source.start();
	const out = (await ctx.startRendering()).getChannelData(0);
	t.ok(peakAbs(out, 0, N) < 1e-4, 'outputs silence, not a passthrough');
});

test('ConvolverNode convolves with an impulse response', async (t) => {
	// A single-sample impulse response is an identity convolution, so the
	// output carries the same energy as the input. Asserted as total energy
	// rather than sample-by-sample because implementations are free to add
	// latency (nx.js convolves a partition at a time, so it delays by one).
	const N = 16384;
	const ctx = new OfflineAudioContext(1, N, RATE);
	const buffer = ctx.createBuffer(1, 4096, RATE);
	const data = buffer.getChannelData(0);
	let want = 0;
	for (let i = 0; i < data.length; i++) {
		data[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / RATE);
		want += data[i] * data[i];
	}
	const ir = ctx.createBuffer(1, 64, RATE);
	ir.getChannelData(0)[0] = 1;

	const source = ctx.createBufferSource();
	source.buffer = buffer;
	const conv = ctx.createConvolver();
	conv.normalize = false; // set before assigning the buffer, per spec
	conv.buffer = ir;
	source.connect(conv);
	conv.connect(ctx.destination);
	source.start();

	const out = (await ctx.startRendering()).getChannelData(0);
	let got = 0;
	for (let i = 0; i < out.length; i++) got += out[i] * out[i];
	t.ok(got > 0, 'convolver passes audio once a response is set');
	t.ok(
		Math.abs(got - want) / want < 0.02,
		'a unit impulse response preserves the signal energy',
	);
	t.ok(peakAbs(out, 0, N) < 1.01, 'output stays bounded');
});
