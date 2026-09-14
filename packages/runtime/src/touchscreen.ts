import { $ } from './$';
import { TouchEvent } from './polyfills/event';
import type { Screen } from './screen';
import type { Touch } from './polyfills/event';

let init = false;
let previousTouches: Touch[] = [];

/**
 * Assigned identifier per HID finger slot, for the lifetime of one press.
 *
 * The native layer reports the hardware finger SLOT in `identifier`, and slots
 * are reused — a fresh press can land on the slot a just-lifted finger had.
 * The old code disambiguated by doing `touch.identifier += Math.random()`,
 * keeping the integer part as the slot and the fraction as press identity.
 * That works for matching, but it makes `Touch.identifier` a random float when
 * the spec says it is a `long`, and brewser derives `PointerEvent.pointerId`
 * from it (`identifier + 1`) — so pages saw fractional, unrepeatable pointer
 * ids where every browser gives small integers. Libraries key their pointer
 * bookkeeping on those ids.
 *
 * Instead: hand out a monotonically increasing integer per press, exactly as a
 * browser does, and remember slot -> id until the finger lifts.
 */
const slotToTouchId = new Map<number, number>();
let nextTouchId = 0;

export function initTouchscreen() {
	if (!init) {
		$.hidInitializeTouchScreen();
		init = true;
	}
}

export function dispatchTouchEvents(screen: Screen) {
	if (!init) return;
	const raw = $.hidGetTouchScreenStates();
	// Absent AND empty both mean "no fingers down"; the old code treated them
	// as different cases, and the empty-array path wiped `previousTouches`
	// before deriving the lifted list, so a release reported through an empty
	// array produced no `touchend` at all.
	const touches: Touch[] = raw && raw.length ? raw : [];

	const startTouches: Touch[] = [];
	const changedTouches: Touch[] = [];
	const endTouches: Touch[] = [];
	const liveIds = new Set<number>();

	for (const touch of touches) {
		const slot = touch.identifier | 0;
		const assigned = slotToTouchId.get(slot);
		const prev = assigned === undefined
			? undefined
			: previousTouches.find((t) => t.identifier === assigned);
		if (assigned !== undefined && prev) {
			// @ts-expect-error identifier is readonly in the public type
			touch.identifier = assigned;
			liveIds.add(assigned);
			if (!touchIsEqual(touch, prev)) changedTouches.push(touch);
		} else {
			const id = nextTouchId++;
			slotToTouchId.set(slot, id);
			// @ts-expect-error identifier is readonly in the public type
			touch.identifier = id;
			liveIds.add(id);
			startTouches.push(touch);
		}
	}

	// Anything down last frame and absent now has lifted. Derived from the
	// PREVIOUS array before it is replaced — the old code assigned
	// `previousTouches = touches` first and then iterated it, so every entry
	// was by definition in the live set and this list was ALWAYS empty. The
	// only reason releases worked at all was the separate no-touches branch,
	// which fired a blanket touchend; lifting one of several fingers never
	// produced an event.
	for (const prevTouch of previousTouches) {
		if (liveIds.has(prevTouch.identifier)) continue;
		endTouches.push(prevTouch);
		for (const [slot, id] of slotToTouchId) {
			if (id === prevTouch.identifier) { slotToTouchId.delete(slot); break; }
		}
	}

	previousTouches = touches;

	if (startTouches.length) {
		screen.dispatchEvent(
			new TouchEvent('touchstart', {
				bubbles: true,
				cancelable: true,
				touches,
				changedTouches: startTouches,
			}),
		);
	}
	if (changedTouches.length) {
		screen.dispatchEvent(
			new TouchEvent('touchmove', {
				bubbles: true,
				cancelable: true,
				touches,
				changedTouches,
			}),
		);
	}
	if (endTouches.length) {
		screen.dispatchEvent(
			new TouchEvent('touchend', {
				bubbles: true,
				cancelable: true,
				touches,
				changedTouches: endTouches,
			}),
		);
	}
}

function touchIsEqual(a: Touch, b: Touch) {
	return (
		a.screenX === b.screenX &&
		a.screenY === b.screenY &&
		a.radiusX === b.radiusX &&
		a.radiusY === b.radiusY &&
		a.rotationAngle === b.rotationAngle
	);
}
