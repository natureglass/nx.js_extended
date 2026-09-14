import parseCssFont, { type IFont } from 'parse-css-font';
import { $ } from '../$';
import { INTERNAL_SYMBOL } from '../internal';
import { EventTarget } from '../polyfills/event-target';
import type { screen } from '../screen';
import type { FontFaceSetLoadStatus } from '../types';
import { assertInternalConstructor, def } from '../utils';
import { FontFace, isFontFaceUsable, nativeFontFace } from './font-face';

/**
 * Manages the loading of font-faces and querying of their download status.
 *
 * @see https://developer.mozilla.org/docs/Web/API/FontFaceSet
 */
export class FontFaceSet extends EventTarget {
	#set = new Set<FontFace>();

	/**
	 * @ignore
	 */
	constructor() {
		assertInternalConstructor(arguments);
		super();
		this.ready = Promise.resolve(this);
		this.status = 'loaded';
	}

	onloading: ((this: FontFaceSet, ev: Event) => any) | null = null;
	onloadingdone: ((this: FontFaceSet, ev: Event) => any) | null = null;
	onloadingerror: ((this: FontFaceSet, ev: Event) => any) | null = null;
	ready: Promise<this>;
	status: FontFaceSetLoadStatus;
	/**
	 * Whether every font needed to render `text` in `font` is available.
	 *
	 * Every face in this set is fully decoded at `add()` time (there is no
	 * pending/loading state on this platform), so "available" reduces to
	 * "a face with a matching family is registered" — plus `system-ui` /
	 * `sans-serif`, which `addSystemFont()` materialises from the Switch's
	 * shared font on demand and therefore always resolve.
	 *
	 * Matching is by family name only, case-insensitively. `findFont()`
	 * additionally requires an exact weight/style/stretch match because it
	 * has to pick ONE face to rasterise with; `check()` is asking a broader
	 * question ("would text in this family render?"), and browsers answer
	 * yes there by synthesising the missing variant.
	 *
	 * Deviation from spec: an unparseable `font` shorthand returns false
	 * rather than throwing a SyntaxError. Callers reach `check()` through
	 * feature-detection paths where a throw is far more damaging than a
	 * conservative false.
	 */
	check(font: string, _text?: string | undefined): boolean {
		const families = familiesOf(font);
		if (families.length === 0) return false;
		for (const family of families) {
			if (isFamilyAvailable(this, family)) return true;
		}
		return false;
	}

	/**
	 * Resolve the faces matching `font`. Nothing is fetched here: a
	 * {@link FontFace} owns decoded bytes from the moment it is constructed,
	 * so by the time a face is in this set it is loaded. Resolves with the
	 * matching faces (empty when none match) and never rejects — same
	 * reasoning as `check()` above.
	 */
	load(font: string, _text?: string | undefined): Promise<FontFace[]> {
		const families = familiesOf(font);
		const out: FontFace[] = [];
		for (const family of families) {
			const lower = family.toLowerCase();
			for (const face of this.#set) {
				if (
					face.family.toLowerCase() === lower &&
					isFontFaceUsable(face) &&
					!out.includes(face)
				) {
					out.push(face);
				}
			}
		}
		return Promise.resolve(out);
	}

	// Set interface
	get size() {
		return this.#set.size;
	}
	add(font: FontFace) {
		this.#set.add(font);
		return this;
	}
	clear(): void {
		this.#set.clear();
	}
	delete(font: FontFace): boolean {
		return this.#set.delete(font);
	}
	has(font: FontFace): boolean {
		return this.#set.has(font);
	}
	keys(): IterableIterator<FontFace> {
		return this.#set.keys();
	}
	values(): IterableIterator<FontFace> {
		return this.#set.values();
	}
	entries(): IterableIterator<[FontFace, FontFace]> {
		return this.#set.entries();
	}
	forEach(
		callbackfn: (value: FontFace, key: FontFace, parent: FontFaceSet) => void,
		thisArg: any = this,
	): void {
		for (const font of this.#set) {
			callbackfn.call(thisArg, font, font, this);
		}
	}
	[Symbol.iterator](): IterableIterator<FontFace> {
		return this.#set[Symbol.iterator]();
	}
}
/** Families the runtime can always produce, whether or not a face for them
 * is in the set yet — `addSystemFont()` registers both from the Switch's
 * shared font the first time an unmatched family is requested. */
const ALWAYS_AVAILABLE_FAMILIES = new Set(['system-ui', 'sans-serif']);

/** Family list from a CSS `font` shorthand, or from a bare family name.
 * Returns [] when the input can't be understood. */
function familiesOf(font: string): string[] {
	if (typeof font !== 'string' || font.trim() === '') return [];
	try {
		const parsed = parseCssFont(font);
		const family = (parsed as IFont).family;
		if (family && family.length > 0) return family;
	} catch {
		// Not a full shorthand. Callers (Cocos's loader, for one) pass a bare
		// family name to load()/check(), which parse-css-font rejects since
		// the shorthand requires a size. Fall through and treat the input as
		// a comma-separated family list.
	}
	return font
		.split(',')
		.map((f) => f.trim().replace(/^['"]|['"]$/g, ''))
		.filter((f) => f.length > 0);
}

function isFamilyAvailable(set: FontFaceSet, family: string): boolean {
	const lower = family.toLowerCase();
	if (ALWAYS_AVAILABLE_FAMILIES.has(lower)) return true;
	for (const face of set) {
		// A url()-sourced face that has not finished `load()` yet has no
		// glyphs, so reporting it as available would make a page skip its
		// own wait and render in the fallback.
		if (face.family.toLowerCase() === lower && isFontFaceUsable(face)) {
			return true;
		}
	}
	return false;
}

def(FontFaceSet);

/**
 * Contains the available fonts for use on the {@link screen | `screen`} Canvas context.
 *
 * There are two built-in fonts available:
 *
 *  - `"system-ui"` is the system font provided by the Switch operating system.
 *  - `"system-icons"` contains the icons used by the Switch operating system.
 *
 * Custom fonts can be added to the set using the {@link FontFaceSet.add | `add()`} method.
 *
 * @see https://nxjs.n8.io/runtime/concepts/fonts
 */
// @ts-expect-error Internal constructor
export var fonts = new FontFaceSet(INTERNAL_SYMBOL);
def(fonts, 'fonts');

export function findFont(
	fontFaceSet: FontFaceSet,
	desired: IFont,
): FontFace | null {
	if (!desired.family) {
		throw new Error('No `font-family` was specified');
	}
	for (const family of desired.family) {
		for (const fontFace of fontFaceSet) {
			if (
				family === fontFace.family &&
				desired.stretch === fontFace.stretch &&
				desired.style === fontFace.style &&
				desired.weight === fontFace.weight
			) {
				// url()-sourced faces are a JS shell around a buffer-backed
				// FontFace; native code can only unwrap the latter. `null`
				// means "still loading" — keep looking.
				const native = nativeFontFace(fontFace);
				if (native) return native;
			}
		}
	}
	return null;
}

export function addSystemFont(fonts: FontFaceSet): FontFace {
	// Idempotent: the generic-family fallback in the 2D `font` setter now
	// calls this on every unmatched font-set (weight variants included, since
	// findFont requires an exact weight match), so re-copying the multi-MB
	// shared font and appending duplicate FontFaces each time would bloat the
	// set. Reuse the already-registered system-ui face if present.
	for (const f of fonts) {
		if (f.family === 'system-ui') return f;
	}
	const data = $.getSystemFont(0 /* PlSharedFontType_Standard */);
	const f = new FontFace('system-ui', data);
	fonts.add(f);
	fonts.add(new FontFace('sans-serif', data));
	return f;
}

export function addIconFont(fonts: FontFaceSet): FontFace {
	const data = $.getSystemFont(5 /* PlSharedFontType_NintendoExt */);
	const f = new FontFace('system-icons', data);
	fonts.add(f);
	return f;
}
