import { $ } from '../$';
import { INTERNAL_SYMBOL } from '../internal';
import { createInternal, def } from '../utils';
import { AudioNode, type AudioNodeOptions } from './audio-node';
import {
	bufferInternal,
	ctxInternal,
	NODE_TYPE_CONVOLVER,
	nodeInternal,
} from './internal';
import type { AudioBuffer } from './audio-buffer';
import type { BaseAudioContext } from './base-audio-context';

export interface ConvolverOptions extends AudioNodeOptions {
	buffer?: AudioBuffer | null;
	disableNormalization?: boolean;
}

interface ConvolverInternal {
	buffer: AudioBuffer | null;
	normalize: boolean;
}

const _ = createInternal<ConvolverNode, ConvolverInternal>();

function syncBuffer(node: ConvolverNode) {
	const i = _(node);
	const handle = nodeInternal(node).handle;
	if (!i.buffer) {
		$.audioConvolverSetBuffer(handle, null, 0, 0, false);
		return;
	}
	const b = bufferInternal(i.buffer);
	$.audioConvolverSetBuffer(
		handle,
		b.channels,
		b.length,
		b.sampleRate,
		i.normalize,
	);
}

/**
 * An {@link AudioNode} that convolves its input with an impulse response —
 * the standard way to apply a recorded reverb, or to model a speaker cabinet
 * or a telephone handset.
 *
 * nx.js convolves in the frequency domain with a uniformly-partitioned
 * overlap-save scheme, so the cost stays proportional to the response length
 * rather than to its square.
 *
 * > [!NOTE]
 * > Unlike a browser's implementation, which is latency-free, this one delays
 * > its output by one partition (1024 frames, ~21 ms at 48 kHz). Impulse
 * > responses longer than 5 seconds are truncated, and a 4-channel
 * > ("true stereo") response is reduced to its first two channels.
 *
 * @see https://developer.mozilla.org/docs/Web/API/ConvolverNode
 */
export class ConvolverNode extends AudioNode implements globalThis.ConvolverNode {
	/**
	 * @see https://developer.mozilla.org/docs/Web/API/ConvolverNode/ConvolverNode
	 */
	constructor(context: BaseAudioContext, options: ConvolverOptions = {}) {
		const handle = $.audioNodeNew(
			ctxInternal(context).handle,
			NODE_TYPE_CONVOLVER,
		);
		// @ts-expect-error internal constructor
		super(INTERNAL_SYMBOL, {
			context,
			handle,
			numberOfInputs: 1,
			numberOfOutputs: 1,
			channelCount: options.channelCount ?? 2,
			channelCountMode: options.channelCountMode ?? 'clamped-max',
			channelInterpretation: options.channelInterpretation ?? 'speakers',
		});
		_.set(this, {
			buffer: null,
			normalize: !options.disableNormalization,
		});
		if (options.buffer) this.buffer = options.buffer;
	}

	/**
	 * The impulse response to convolve with. Assigning `null` (the initial
	 * value) makes the node output silence, per spec.
	 *
	 * The samples are transformed when they are assigned, so later writes to
	 * the same `AudioBuffer` are not picked up — reassign the buffer instead.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/ConvolverNode/buffer
	 */
	get buffer(): AudioBuffer | null {
		return _(this).buffer;
	}

	set buffer(v: AudioBuffer | null) {
		_(this).buffer = v ?? null;
		syncBuffer(this);
	}

	/**
	 * When `true` (the default), the impulse response is scaled so that
	 * inserting the convolver does not change perceived loudness.
	 *
	 * Per spec this must be set **before** assigning
	 * {@link ConvolverNode.buffer | `buffer`}; changing it afterwards has no
	 * effect until a buffer is assigned again.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/ConvolverNode/normalize
	 */
	get normalize(): boolean {
		return _(this).normalize;
	}

	set normalize(v: boolean) {
		_(this).normalize = !!v;
	}
}
def(ConvolverNode);
