import { $ } from '../$';
import { bufferSourceToArrayBuffer, def, proto } from '../utils';
import { URL } from '../polyfills/url';
import { isWoff2, woff2ToSfnt } from './woff2';
import type {
	FontFaceLoadStatus,
	FontDisplay,
	FontFaceDescriptors,
} from '../types';

// Late-bound (call-time) `globalThis.fetch` wrapper. Same rationale as
// `image.ts`: embedders (brewser-runtime's BrowserResourceLoader, for one)
// install their own `globalThis.fetch` at app-session start to extend the
// supported schemes beyond this package's built-ins (`brewser://` is the one
// that matters for font URLs). An import-time capture would freeze the
// pre-wrapper engine fetch and every app-relative font URL would 404.
function fetch(input: string, init?: RequestInit): Promise<Response> {
	return globalThis.fetch(input, init);
}

interface FontSrcCandidate {
	url: string;
	format: string;
}

/**
 * State for a FontFace constructed from a CSS `src` string. Such a face
 * cannot be backed by a native `nx_font_face_t` at construction time (the
 * bytes have not been fetched yet), so it starts life as a plain JS object
 * and adopts a real, buffer-backed FontFace in `load()`.
 */
interface DeferredSource {
	candidates: FontSrcCandidate[];
	/** The buffer-backed FontFace built once bytes decoded. */
	native: FontFace | null;
	inflight: Promise<FontFace> | null;
	settle: (face: FontFace | null, err?: unknown) => void;
}

const deferred = new WeakMap<FontFace, DeferredSource>();

/** Every `url(...)` in a CSS `src` value, with its `format(...)` hint when
 * present, in declaration order. `local(...)` entries are skipped — there is
 * no lookup-system-font-by-name path here. */
function extractFontSrcCandidates(src: string): FontSrcCandidate[] {
	const out: FontSrcCandidate[] = [];
	const re =
		/url\s*\(\s*(['"]?)([^'")]+)\1\s*\)(?:\s*format\s*\(\s*(['"]?)([^'")]+)\3\s*\))?/gi;
	for (const m of src.matchAll(re)) {
		const url = (m[2] ?? '').trim();
		if (url) out.push({ url, format: (m[4] ?? '').trim().toLowerCase() });
	}
	// A bare URL with no `url()` wrapper is not valid CSS `src`, but loaders
	// pass one often enough that rejecting it buys nothing.
	if (out.length === 0) {
		const bare = src.trim().replace(/^['"]|['"]$/g, '');
		if (bare && !/[\s(]/.test(bare)) out.push({ url: bare, format: '' });
	}
	return out;
}

/** Order `src` candidates by how likely FreeType is to decode them, keeping
 * declaration order within a tier.
 *
 * `src` is a PRIORITISED LIST and essentially every web font now leads with
 * WOFF2 — which this FreeType cannot read: the build defines
 * `FT_CONFIG_OPTION_USE_ZLIB` (so WOFF1, TTF and OTF are fine) but devkitPro
 * portlibs ships no brotli, which WOFF2 requires. Taking literally the first
 * `url()` therefore picks the one entry guaranteed to fail and never looks at
 * the TTF sitting right behind it.
 *
 * WOFF2 is ranked last rather than dropped: if brotli is ever linked, this
 * starts preferring it again with no further change, and a `src` whose ONLY
 * entry is WOFF2 is still attempted (and reports a real reason when it
 * fails) rather than being silently skipped. */
function orderFontSrcCandidates(
	cands: FontSrcCandidate[],
): FontSrcCandidate[] {
	const rank = (c: FontSrcCandidate): number => {
		// No `format()` hint: fall back to the extension, which usually tells
		// the truth. The FontFace ctor is the real arbiter either way.
		const fmt =
			c.format ||
			(/\.(woff2|woff|ttf|otf|ttc|eot)(?:[?#]|$)/i.exec(c.url)?.[1] ?? '');
		const f = fmt.toLowerCase();
		if (f === 'woff2') return 2;
		if (
			f === '' ||
			f === 'woff' ||
			f === 'ttf' ||
			f === 'otf' ||
			f === 'truetype' ||
			f === 'opentype' ||
			f === 'embedded-opentype'
		) {
			return 0;
		}
		return 1; // svg, collection variants, anything unrecognised
	};
	return cands
		.map((c, i) => ({ c, i }))
		.sort((a, b) => rank(a.c) - rank(b.c) || a.i - b.i)
		.map((x) => x.c);
}

function descriptorsOf(face: FontFace): FontFaceDescriptors {
	return {
		ascentOverride: face.ascentOverride,
		descentOverride: face.descentOverride,
		display: face.display,
		featureSettings: face.featureSettings,
		lineGapOverride: face.lineGapOverride,
		stretch: face.stretch,
		style: face.style,
		unicodeRange: face.unicodeRange,
		weight: face.weight,
	};
}

function errText(err: unknown): string {
	return (err as { message?: string })?.message ?? String(err);
}

async function loadDeferred(
	face: FontFace,
	d: DeferredSource,
): Promise<FontFace> {
	// @ts-expect-error Readonly
	face.status = 'loading' satisfies FontFaceLoadStatus;
	// Same base-URL preference as `Image`'s `src` setter: embedders that
	// emulate per-page navigation push a fresh `location.href` while leaving
	// `document.baseURI` pinned at the outer page URL.
	const g = globalThis as {
		location?: { href?: string };
		document?: { baseURI?: string };
	};
	const base = g.location?.href ?? g.document?.baseURI ?? $.entrypoint;
	const failures: string[] = [];
	for (const cand of d.candidates) {
		let url = cand.url;
		try {
			url = new URL(cand.url, base).href;
		} catch {
			// Leave the raw value; fetch may still understand it.
		}
		try {
			const res = await fetch(url);
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			const buf = await res.arrayBuffer();
			// Buffer path of this same constructor: builds the native face.
			d.native = new FontFace(face.family, buf, descriptorsOf(face));
			// @ts-expect-error Readonly
			face.status = 'loaded' satisfies FontFaceLoadStatus;
			d.settle(face);
			return face;
		} catch (err) {
			failures.push(`${cand.url}: ${errText(err)}`);
		}
	}
	// @ts-expect-error Readonly
	face.status = 'error' satisfies FontFaceLoadStatus;
	const err = new Error(
		`Failed to load font "${face.family}" — ${failures.join('; ')}`,
	);
	d.settle(null, err);
	throw err;
}

/**
 * Defines the source of a font face, either a URL to an external resource or a
 * buffer, and font properties such as `style`, `weight`, and so on. For URL
 * font sources it allows authors to trigger when the remote font is fetched
 * and loaded, and to track loading status.
 *
 * @see https://developer.mozilla.org/docs/Web/API/FontFace
 */
export class FontFace implements globalThis.FontFace {
	declare ascentOverride: string;
	declare descentOverride: string;
	declare display: FontDisplay;
	declare family: string;
	declare featureSettings: string;
	declare lineGapOverride: string;
	declare readonly loaded: Promise<this>;
	declare readonly status: FontFaceLoadStatus;
	declare stretch: string;
	declare style: string;
	declare unicodeRange: string;
	declare weight: string;

	constructor(
		family: string,
		source: string | BufferSource,
		descriptors: FontFaceDescriptors = {},
	) {
		// CSS `src` string (`url('…') format('woff2'), url('…')`). Web font
		// loaders overwhelmingly use this form — PixiJS's `loadWebFont`
		// parser does `new FontFace(family, "url('…')")` then `await
		// face.load()` — so rejecting it outright rejected every
		// `Assets.load()` of a font. The bytes cannot be fetched
		// synchronously, so the face is a plain JS object until `load()`
		// adopts a buffer-backed one; `nativeFontFace()` is what the
		// rasteriser resolves it through.
		if (typeof source === 'string') {
			const candidates = orderFontSrcCandidates(
				extractFontSrcCandidates(source),
			);
			if (candidates.length === 0) {
				throw new SyntaxError(
					'FontFace: `source` string contains no usable url()',
				);
			}
			const f: FontFace = Object.create(FontFace.prototype);
			applyDescriptors(f, family, descriptors);
			let settle!: DeferredSource['settle'];
			const loaded = new Promise<FontFace>((resolve, reject) => {
				settle = (face, err) => (face ? resolve(face) : reject(err));
			});
			// A rejected `loaded` that nobody awaits must not surface as an
			// unhandledrejection — callers are expected to await `load()`.
			loaded.catch(() => {});
			// @ts-expect-error Readonly
			f.loaded = loaded;
			// @ts-expect-error Readonly
			f.status = 'unloaded';
			deferred.set(f, {
				candidates,
				native: null,
				inflight: null,
				settle,
			});
			return f;
		}
		let buffer = bufferSourceToArrayBuffer(source);
		// WOFF2 is what essentially every modern web font ships as, and this
		// FreeType cannot read one (no brotli in devkitPro, so
		// FT_CONFIG_OPTION_USE_BROTLI is off). Decode it to a plain SFNT here
		// — before FreeType ever sees the bytes — so `.woff2` is just another
		// font format from every caller's point of view. WOFF1, TTF and OTF
		// pass straight through.
		if (isWoff2(buffer)) {
			const sfnt = woff2ToSfnt(buffer);
			if (sfnt) buffer = sfnt;
		}
		const f = proto($.fontFaceNew(buffer), FontFace);
		applyDescriptors(f, family, descriptors);
		// @ts-expect-error Readonly
		f.loaded = Promise.resolve(f);
		// @ts-expect-error Readonly
		f.status = 'loaded';
		return f;
	}

	async load(): Promise<this> {
		const d = deferred.get(this);
		// Buffer-backed faces own decoded bytes from construction.
		if (!d || d.native) return this;
		if (!d.inflight) d.inflight = loadDeferred(this, d);
		await d.inflight;
		return this;
	}
}

function applyDescriptors(
	f: FontFace,
	family: string,
	descriptors: FontFaceDescriptors,
): void {
	f.family = family;
	f.ascentOverride = descriptors.ascentOverride ?? 'normal';
	f.descentOverride = descriptors.descentOverride ?? 'normal';
	f.display = descriptors.display ?? 'auto';
	f.featureSettings = descriptors.featureSettings ?? 'normal';
	f.lineGapOverride = descriptors.lineGapOverride ?? 'normal';
	f.stretch = descriptors.stretch ?? 'normal';
	f.style = descriptors.style ?? 'normal';
	f.unicodeRange = descriptors.unicodeRange ?? '';
	f.weight = descriptors.weight ?? 'normal';
}

/**
 * The buffer-backed FontFace that actually carries decoded font bytes.
 *
 * A face built from a buffer is its own native object and is returned as-is.
 * A face built from a CSS `src` string is a JS shell until `load()` resolves;
 * before that it has no glyphs and this returns `null`, so callers must treat
 * `null` as "not usable yet" rather than substituting a fallback silently.
 *
 * Every path that hands a FontFace down to native code (`$.canvasContext2dSetFont`)
 * must resolve through here — a shell has no internal field to unwrap.
 */
export function nativeFontFace(face: FontFace): FontFace | null {
	const d = deferred.get(face);
	if (!d) return face;
	return d.native;
}

/** Whether `face` has decoded bytes and can rasterise glyphs today. */
export function isFontFaceUsable(face: FontFace): boolean {
	return nativeFontFace(face) !== null;
}

def(FontFace);
