import { $ } from './$';
import { createInternal, def, proto } from './utils';
import { URL } from './polyfills/url';

// Late-bound (call-time) globalThis.fetch wrapper. This module DOES NOT
// import `./fetch/fetch` directly: embedders (e.g. brewser-runtime's
// BrowserResourceLoader for `brewser://` URLs) install their own
// `globalThis.fetch` wrapper at app-session start to extend the set of
// supported schemes beyond this package's built-in
// http/https/blob/data/file/sdmc/romfs handlers. Recapitulates the
// QuickJS-era patch that the V8 migration dropped — see
// nxjs-extended/MIGRATION_PLAN.md "QuickJS-era engine patches to re-apply
// on V8" catalog.
//
// CRITICAL: late-bound. The function body calls `globalThis.fetch` at
// CALL TIME, not at module init. An import-time capture
// (`const fetch = globalThis.fetch`) would freeze the pre-wrapper engine
// fetch (this module loads at engine boot, embedders install the wrapper
// at session start) and the deferral would do nothing. Any future
// re-application of this style of patch must preserve the call-time
// lookup.
function fetch(
	input: string | URL | Request,
	init?: RequestInit,
): Promise<Response> {
	return globalThis.fetch(input, init);
}
import { Event, ErrorEvent } from './polyfills/event';
import { EventTarget } from './polyfills/event-target';
import type { CanvasRenderingContext2D } from './canvas/canvas-rendering-context-2d';

/**
 * An embedder-supplied decoder for image URLs this runtime cannot decode
 * itself, tried before the normal fetch-and-decode.
 *
 * Exists for one case that no image decoder can handle on its own:
 * `data:image/svg+xml,…<foreignObject>…HTML…</foreignObject>…`, the
 * HTML-to-texture trick behind every canvas rich-text engine (PixiJS
 * `HTMLText`, html2canvas, dom-to-image, Three's SVGRenderer). Rasterising it
 * means running an HTML layout, so only an embedder that HAS one — brewser,
 * with its live-DOM layout and painter — can produce the pixels.
 *
 * Return straight (non-premultiplied) RGBA, `width * height * 4` bytes, or
 * `null`/`undefined` to decline and let the normal path run. Throwing also
 * declines. Sync or async.
 */
export type ImageRasteriser = (
	url: string,
	hintWidth: number,
	hintHeight: number,
) => ImageRasterResult | null | undefined | Promise<ImageRasterResult | null | undefined>;

export interface ImageRasterResult {
	pixels: Uint8ClampedArray | Uint8Array | ArrayBuffer;
	width: number;
	height: number;
}

let imageRasteriser: ImageRasteriser | null = null;

/**
 * Install (or clear, with `null`) the {@link ImageRasteriser}. Embedders call
 * this at app-session start, the same way they install their `globalThis.fetch`
 * wrapper.
 */
export function setImageRasteriser(fn: ImageRasteriser | null): void {
	imageRasteriser = fn;
}

interface ImageInternal {
	complete: boolean;
	src?: URL;
	/** `width` / `height` content attributes, when the page has set them.
	 * `undefined` means "unset", and the getters fall through to the
	 * decoded intrinsic size. */
	widthAttr?: number;
	heightAttr?: number;
}

const _ = createInternal<Image, ImageInternal>();

/**
 * The `Image` class is the spiritual equivalent of the [`HTMLImageElement`](https://developer.mozilla.org/docs/Web/API/HTMLImageElement)
 * class in web browsers. You can use it to load image data from the filesytem
 * or remote source over the network. Once loaded, the image may be drawn onto the screen
 * context or an offscreen canvas context using {@link CanvasRenderingContext2D.drawImage | `ctx.drawImage()`}.
 *
 * ### Supported Image Formats
 *
 *  - `jpg` - JPEG image data using [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)
 *  - `png` - PNG image data using [libpng](http://www.libpng.org/pub/png/libpng.html)
 *  - `webp` - WebP image data using [libpng](https://github.com/webmproject/libwebp)
 *
 * @example
 *
 * ```typescript
 * const ctx = screen.getContext('2d');
 *
 * const img = new Image();
 * img.addEventListener('load', () => {
 *   ctx.drawImage(img);
 * });
 * img.src = 'romfs:/logo.png';
 * ```
 */
export class Image extends EventTarget {
	declare onload: ((this: Image, ev: Event) => any) | null;
	declare onerror: ((this: Image, ev: ErrorEvent) => any) | null;
	declare decoding: 'async' | 'sync' | 'auto';
	declare isMap: boolean;
	declare loading: 'eager' | 'lazy';
	declare width: number;
	declare height: number;

	constructor() {
		super();
		const i = proto($.imageNew(), Image);
		i.onload = null;
		i.onerror = null;
		i.decoding = 'auto';
		i.isMap = false;
		i.loading = 'eager';
		_.set(i, { complete: true });
		return i;
	}

	dispatchEvent(event: Event): boolean {
		if (event.type === 'load') {
			this.onload?.(event);
		} else if (event.type === 'error') {
			this.onerror?.(event as ErrorEvent);
		}
		return super.dispatchEvent(event);
	}

	get complete() {
		return _(this).complete;
	}

	get naturalWidth() {
		return intrinsic(this, 'width');
	}

	get naturalHeight() {
		return intrinsic(this, 'height');
	}

	get src() {
		return _(this).src?.href ?? '';
	}

	set src(val: string) {
		// Ledger #80 — prefer `globalThis.location?.href` over
		// `document.baseURI`. Embedders that emulate per-page navigation by
		// pushing a fresh `location.href` (e.g. the webgl-conformance runner
		// evaluating each test HTML in-place) leave `document.baseURI` pinned
		// at the outer page URL, so relative `image.src` values were
		// resolving against the wrong base — the from_image cluster's
		// `image.src = resourcePath + "..."` fetches 404'd and `onload`
		// never fired (TIMEOUT). `location.href` is the more responsive
		// signal here: real browsers keep `baseURI ≡ location.href` unless
		// `<base href>` is set, so preferring location.href diverges only
		// for the (uncommon in nx.js) `<base href>` case. Falls back to
		// `document.baseURI` then `$.entrypoint`.
		const g = globalThis as {
			location?: { href?: string };
			document?: { baseURI?: string };
		};
		const baseUrl =
			g.location?.href ?? g.document?.baseURI ?? $.entrypoint;
		const url = new URL(val, baseUrl);
		const internal = _(this);
		internal.src = url;
		internal.complete = false;
		// Embedder decoder first (see `setImageRasteriser`). It is the only
		// way `<foreignObject>` SVG — which needs an HTML layout, not an
		// image decoder — can ever become pixels.
		if (imageRasteriser) {
			let attempt: ReturnType<ImageRasteriser>;
			try {
				attempt = imageRasteriser(url.href, this.width, this.height);
			} catch (_) {
				attempt = null;
			}
			if (attempt) {
				Promise.resolve(attempt).then(
					(raster) => {
						// A declined rasterise falls back to the normal path
						// rather than failing the load outright.
						if (!raster) {
							loadImageBytes(this, internal, url);
							return;
						}
						try {
							$.imageAlloc(this, raster.width, raster.height);
							$.imageWriteRGBA(this, toArrayBuffer(raster.pixels), true);
						} catch (error) {
							internal.complete = false;
							this.dispatchEvent(new ErrorEvent('error', { error }));
							return;
						}
						internal.complete = true;
						this.dispatchEvent(new Event('load'));
					},
					() => loadImageBytes(this, internal, url),
				);
				return;
			}
		}
		loadImageBytes(this, internal, url);
	}

	// Compat with HTML DOM interface
	className = '';

	get nodeType() {
		return 1;
	}
	get nodeName() {
		return 'IMG';
	}
	getAttribute(name: string): string | null {
		if (name === 'width') return String(this.width);
		if (name === 'height') return String(this.height);
		return null;
	}
	setAttribute(name: string, value: string | number) {
		if (name === 'width' || name === 'height') {
			this[name] = value as number;
		}
	}
	removeAttribute(name: string) {
		if (name === 'width') _(this).widthAttr = undefined;
		else if (name === 'height') _(this).heightAttr = undefined;
	}
}
/** The normal fetch-and-decode path, factored out so the `src` setter can
 * reach it from both the rasteriser-declined branches and the default one. */
function loadImageBytes(img: Image, internal: ImageInternal, url: URL): void {
	fetch(url)
		.then((res) => {
			if (!res.ok) {
				throw new Error(`Failed to load image: ${res.status}`);
			}
			return res.arrayBuffer();
		})
		.then((buf) => $.imageDecode(img, buf))
		.then(
			() => {
				internal.complete = true;
				img.dispatchEvent(new Event('load'));
			},
			(error) => {
				internal.complete = false;
				img.dispatchEvent(new ErrorEvent('error', { error }));
			},
		);
}

function toArrayBuffer(
	pixels: Uint8ClampedArray | Uint8Array | ArrayBuffer,
): ArrayBuffer {
	if (pixels instanceof ArrayBuffer) return pixels;
	// A view can be a window onto a larger buffer (getImageData's is not, but
	// a caller's might be), so slice to exactly the view's bytes rather than
	// handing over the whole backing store.
	const view = pixels as Uint8Array | Uint8ClampedArray;
	if (view.byteOffset === 0 && view.byteLength === view.buffer.byteLength) {
		return view.buffer as ArrayBuffer;
	}
	return view.buffer.slice(
		view.byteOffset,
		view.byteOffset + view.byteLength,
	) as ArrayBuffer;
}

$.imageInit(Image);

// `width` / `height` are getter-only accessors on the native prototype
// (`nx_image_init_class`), but on `HTMLImageElement` they REFLECT the content
// attributes and are settable — and code in the wild relies on writing them
// and reading back what it wrote before any bytes exist. PixiJS's HTMLText
// rasteriser is the case that surfaced this: it sizes one reused `Image`
// (`img.width = w; img.height = h`) and then reads those values back to size
// the SVG root and the destination texture, so a getter-only property threw
// `Cannot set property width of #<Image> which has only a getter` and killed
// the whole text pipeline.
//
// Reflection semantics, per HTML: the getter returns the content attribute
// when set, otherwise the intrinsic (decoded) size, otherwise 0.
// `naturalWidth` / `naturalHeight` are NOT reflected and always report the
// decoded size, and native drawing/upload paths read the decoded dimensions
// straight off `nx_image_t` — so an override changes what JS sees, never what
// gets rasterised.
const nativeDimension: { width?: () => number; height?: () => number } = {};
for (const name of ['width', 'height'] as const) {
	const desc = Object.getOwnPropertyDescriptor(Image.prototype, name);
	const nativeGet = desc?.get;
	if (!nativeGet) continue;
	nativeDimension[name] = nativeGet as () => number;
	const attr = name === 'width' ? 'widthAttr' : 'heightAttr';
	Object.defineProperty(Image.prototype, name, {
		configurable: true,
		enumerable: desc.enumerable ?? true,
		get(this: Image) {
			const v = _(this)[attr];
			return v !== undefined ? v : nativeGet.call(this);
		},
		set(this: Image, value: unknown) {
			// HTML parses these as non-negative integers and ignores
			// anything else; a NaN here would propagate into texture sizes.
			const n = Math.trunc(Number(value));
			_(this)[attr] = Number.isFinite(n) && n > 0 ? n : 0;
		},
	});
}

/** The decoded size, ignoring any `width` / `height` content attribute. */
function intrinsic(img: Image, name: 'width' | 'height'): number {
	const get = nativeDimension[name];
	return get ? get.call(img) : 0;
}

def(Image);
