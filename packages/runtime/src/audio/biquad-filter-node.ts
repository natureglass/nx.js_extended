import { $ } from '../$';
import { INTERNAL_SYMBOL } from '../internal';
import { createInternal, def } from '../utils';
import { AudioNode, type AudioNodeOptions } from './audio-node';
import { createAudioParam } from './audio-param';
import {
	BIQUAD_TYPES,
	ctxInternal,
	MOST_POSITIVE_SINGLE,
	NODE_TYPE_BIQUAD_FILTER,
	nodeInternal,
} from './internal';
import type { AudioParam } from './audio-param';
import type { BaseAudioContext } from './base-audio-context';

export interface BiquadFilterOptions extends AudioNodeOptions {
	type?: BiquadFilterType;
	Q?: number;
	detune?: number;
	frequency?: number;
	gain?: number;
}

interface BiquadFilterInternal {
	type: BiquadFilterType;
	frequency: AudioParam;
	detune: AudioParam;
	Q: AudioParam;
	gain: AudioParam;
}

const _ = createInternal<BiquadFilterNode, BiquadFilterInternal>();

/**
 * An {@link AudioNode} implementing a second-order IIR filter — the building
 * block of tone controls, equalizers, and effects like the "telephone" voice
 * filter (a pair of lowpass sections into a pair of highpass sections).
 *
 * > [!NOTE]
 * > nx.js recomputes the filter coefficients once per render quantum (k-rate)
 * > rather than per sample, so an audio-rate filter sweep steps every ~2.7 ms.
 *
 * > [!IMPORTANT]
 * > As the Web Audio specification requires, `Q` is interpreted in **decibels**
 * > for the `"lowpass"` and `"highpass"` types, and as a plain linear value for
 * > every other type.
 *
 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode
 */
export class BiquadFilterNode
	extends AudioNode
	implements globalThis.BiquadFilterNode
{
	/**
	 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode/BiquadFilterNode
	 */
	constructor(context: BaseAudioContext, options: BiquadFilterOptions = {}) {
		const handle = $.audioNodeNew(
			ctxInternal(context).handle,
			NODE_TYPE_BIQUAD_FILTER,
		);
		// @ts-expect-error internal constructor
		super(INTERNAL_SYMBOL, {
			context,
			handle,
			numberOfInputs: 1,
			numberOfOutputs: 1,
			channelCount: options.channelCount ?? 2,
			channelCountMode: options.channelCountMode ?? 'max',
			channelInterpretation: options.channelInterpretation ?? 'speakers',
		});
		const i: BiquadFilterInternal = {
			type: 'lowpass',
			frequency: createAudioParam(this, handle, 0, {
				defaultValue: 350,
				minValue: 0,
				maxValue: context.sampleRate / 2,
			}),
			detune: createAudioParam(this, handle, 1, {
				defaultValue: 0,
				minValue: -153600,
				maxValue: 153600,
			}),
			Q: createAudioParam(this, handle, 2, {
				defaultValue: 1,
				minValue: -MOST_POSITIVE_SINGLE,
				maxValue: MOST_POSITIVE_SINGLE,
			}),
			gain: createAudioParam(this, handle, 3, {
				defaultValue: 0,
				minValue: -MOST_POSITIVE_SINGLE,
				maxValue: MOST_POSITIVE_SINGLE,
			}),
		};
		_.set(this, i);
		if (options.type) this.type = options.type;
		if (typeof options.frequency === 'number')
			i.frequency.value = options.frequency;
		if (typeof options.detune === 'number') i.detune.value = options.detune;
		if (typeof options.Q === 'number') i.Q.value = options.Q;
		if (typeof options.gain === 'number') i.gain.value = options.gain;
	}

	/**
	 * The shape of the filter: one of `"lowpass"`, `"highpass"`,
	 * `"bandpass"`, `"lowshelf"`, `"highshelf"`, `"peaking"`, `"notch"` or
	 * `"allpass"`. Changing it clears the filter's state, so a live switch
	 * does not carry the old transfer function's ringing into the new one.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode/type
	 */
	get type(): BiquadFilterType {
		return _(this).type;
	}

	set type(v: BiquadFilterType) {
		const index = BIQUAD_TYPES.indexOf(v);
		if (index === -1) return; // per WebIDL enum handling: ignore
		_(this).type = v;
		$.audioBiquadSetType(nodeInternal(this).handle, index);
	}

	/**
	 * The filter's characteristic frequency, in Hz (an a-rate
	 * {@link AudioParam}, evaluated k-rate in nx.js).
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode/frequency
	 */
	get frequency(): AudioParam {
		return _(this).frequency;
	}

	/**
	 * Detuning of {@link BiquadFilterNode.frequency | `frequency`}, in cents.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode/detune
	 */
	get detune(): AudioParam {
		return _(this).detune;
	}

	/**
	 * The filter's quality factor — in **decibels** for `"lowpass"` and
	 * `"highpass"`, linear for the other types. Unused by the shelving filters.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode/Q
	 */
	get Q(): AudioParam {
		return _(this).Q;
	}

	/**
	 * Gain, in dB, used only by the `"lowshelf"`, `"highshelf"` and
	 * `"peaking"` types.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode/gain
	 */
	get gain(): AudioParam {
		return _(this).gain;
	}

	/**
	 * Fills `magResponse` and `phaseResponse` with the filter's magnitude
	 * (linear) and phase (radians) at each frequency in `frequencyHz`.
	 * Frequencies outside `[0, Nyquist]` yield `NaN`.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/BiquadFilterNode/getFrequencyResponse
	 */
	getFrequencyResponse(
		frequencyHz: Float32Array,
		magResponse: Float32Array,
		phaseResponse: Float32Array,
	): void {
		if (
			frequencyHz.length !== magResponse.length ||
			frequencyHz.length !== phaseResponse.length
		) {
			throw new TypeError(
				'Failed to execute "getFrequencyResponse" on "BiquadFilterNode": The three parameter arrays must have the same length',
			);
		}
		$.audioBiquadFrequencyResponse(
			nodeInternal(this).handle,
			frequencyHz,
			magResponse,
			phaseResponse,
		);
	}
}
def(BiquadFilterNode);
