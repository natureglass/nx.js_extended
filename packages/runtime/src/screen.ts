import { $ } from './$';
import {
	assertInternalConstructor,
	createInternal,
	def,
	normalizeImageMime,
	proto,
	stub,
} from './utils';
import { Blob } from './polyfills/blob';
import { EventTarget } from './polyfills/event-target';
import { INTERNAL_SYMBOL } from './internal';
import { CanvasRenderingContext2D } from './canvas/canvas-rendering-context-2d';
import {
	createWebGL2Context,
	type WebGL2RenderingContext,
} from './canvas/webgl2-rendering-context';
import {
	createWebGLContext,
	type WebGLRenderingContext,
} from './canvas/webgl-rendering-context';
import { initTouchscreen } from './touchscreen';
import type { TouchEvent } from './polyfills/event';
import {
	isAppOwningScreen,
	markAppOwnsScreen,
	registerConsoleScreen,
} from './console-screen';

interface ScreenInternal {
	context2d?: CanvasRenderingContext2D;
	contextWebGL?: WebGLRenderingContext;
	contextWebGL2?: WebGL2RenderingContext;
}

const _ = createInternal<Screen, ScreenInternal>();

// Internal hook so the console present can get the screen's 2D context without
// the public getContext()'s "app owns screen" side effect.
const CONSOLE_SCREEN_CTX = Symbol('consoleScreenCtx');

// Acquire the screen's 2D context + put the screen into canvas (framebuffer)
// mode. Shared by the public `getContext('2d')` and the console present.
//
// NOTE: this MUST be a free function (keyed off the `_` WeakMap), NOT a
// `#private` method. The `Screen` constructor returns a substitute native
// canvas object (`proto($.canvasNew(...), Screen)`), so private class members
// are never installed on the actual `screen` instance — calling a `#method` on
// it throws "Receiver must be an instance of class Screen".
function ensureContext(self: Screen): CanvasRenderingContext2D {
	const i = _(self);
	if (!i.context2d) {
		i.context2d = new CanvasRenderingContext2D(
			// @ts-expect-error Internal constructor
			INTERNAL_SYMBOL,
			self,
		);
		$.framebufferInit(self);
	}
	return i.context2d;
}

export class Screen extends EventTarget implements globalThis.Screen {
	/**
	 * @ignore
	 */
	constructor() {
		assertInternalConstructor(arguments);
		super();
		const c = proto($.canvasNew(1280, 720), Screen);
		_.set(c, {});
		return c;
	}

	/** [MDN Reference](https://developer.mozilla.org/docs/Web/API/Screen/availWidth) */
	get availWidth() {
		return this.width;
	}

	/** [MDN Reference](https://developer.mozilla.org/docs/Web/API/Screen/availHeight) */
	get availHeight() {
		return this.height;
	}

	/** [MDN Reference](https://developer.mozilla.org/docs/Web/API/Screen/colorDepth) */
	get colorDepth() {
		return 24;
	}

	/** [MDN Reference](https://developer.mozilla.org/docs/Web/API/Screen/orientation) */
	get orientation(): ScreenOrientation {
		throw new Error('Method not implemented.');
	}

	/** [MDN Reference](https://developer.mozilla.org/docs/Web/API/Screen/pixelDepth) */
	get pixelDepth() {
		return 24;
	}

	/**
	 * The width of the screen in CSS pixels.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/Screen/width
	 */
	declare readonly width: number;

	/**
	 * The height of the screen in CSS pixels.
	 *
	 * @see https://developer.mozilla.org/docs/Web/API/Screen/height
	 */
	declare readonly height: number;

	getContext(contextId: '2d'): CanvasRenderingContext2D;
	getContext(contextId: 'webgl' | 'experimental-webgl', options?: unknown): WebGLRenderingContext | null;
	getContext(contextId: 'webgl2', options?: unknown): WebGL2RenderingContext | null;
	getContext(contextId: string, options?: unknown): null;
	getContext(
		contextId: string,
		options?: unknown,
	): CanvasRenderingContext2D | WebGLRenderingContext | WebGL2RenderingContext | null {
		const i = _(this);
		// brewser multi-WebGL-canvas support: with `{ __brewserNewContext: true }`
		// the WebGL branches mint a FRESH engine context (its own per-canvas
		// tenant FBO) instead of returning the cached screen context, so the
		// runtime can back each page <canvas> with an independent WebGL context.
		// The cached-context behavior (all other callers) is unchanged.
		const forceNewWebGL = !!(options && typeof options === 'object' &&
			(options as { __brewserNewContext?: unknown }).__brewserNewContext);
		if (contextId === '2d') {
			// The screen may only have one context kind (per the HTML canvas
			// spec): once a WebGL context exists, '2d' returns null.
			if (i.contextWebGL || i.contextWebGL2) return null;
			// User code acquiring the screen context takes over rendering
			// from the default console present (see console-screen.ts).
			markAppOwnsScreen();
			return ensureContext(this);
		}
		// Phase 2.C/2.G: WebGL 1 + 2 paths. Both render into the SAME bridge
		// tenant offscreen FBO on the SAME shared ES3 context (Phase 2.A); the
		// engine-side WebGLState is process-wide. The two context objects
		// differ ONLY in their JS prototype (WebGLRenderingContext vs
		// WebGL2RenderingContext), which is what Three.js's
		// `gl.constructor.name === 'WebGL2RenderingContext'` detection
		// requires.
		//
		// IMPORTANT: cross-family co-existence is ALLOWED.
		//
		// Standard browsers enforce "one context kind per canvas" because a
		// browser canvas is single-app and one-shot. The brewser screen is
		// SHARED across multiple demos sequentially within a single brewser
		// session — the shell's `canvas-runner.ts::getSharedScreenGL`
		// acquires v1 the moment ANY page-canvas asks for it, and a later v2
		// demo on a different page then asks for v2 on the same screen. The
		// mutual-exclusion rule blocked the second acquisition (symptom:
		// "WebGL 2 not supported" after any v1 demo had run earlier in the
		// session). Both contexts share underlying engine state, so caching
		// both is functionally fine: each returns a view of the same shared
		// tenant FBO with the appropriate JS class.
		//
		// 2D also coexists with WebGL (the shell's 2D context carries the
		// HTML overlay; the WebGL contexts draw into the tenant FBO; the
		// engine composites both into the screen surface before present).
		// So NO cross-family exclusions at all in the WebGL branches.
		if (contextId === 'webgl' || contextId === 'experimental-webgl') {
			if (forceNewWebGL) {
				// Fresh context with its own per-canvas tenant FBO; NOT cached
				// (the caller owns it). Enables independent stacked WebGL canvases.
				return createWebGLContext(this);
			}
			if (!i.contextWebGL) {
				const ctx = createWebGLContext(this);
				if (!ctx) return null;
				i.contextWebGL = ctx;
			}
			return i.contextWebGL;
		}
		if (contextId === 'webgl2') {
			// Phase 2.G.0 — 'webgl2' returns a non-null context backed by the
			// SEPARATE engine v2 factory ($.webgl2ContextNew + $.webgl2InitClass)
			// sharing the SAME tenant FBO + shared ES3 context as v1. 2.G.0
			// binds ZERO v2 methods: an empty-but-correctly-shaped v2 context
			// for JIT install-shape verification. A v2 instance has all 387
			// v2 constants on its prototype (instanceof / constructor.name
			// detection works) but every method call throws `TypeError: X
			// is not a function` until 2.G.1 lands the webgl2-ubo slice's
			// method allowlist. See MIGRATION_PLAN.md Phase 2.G.0.
			if (forceNewWebGL) {
				// Fresh v2 context with its own per-canvas tenant FBO; NOT cached.
				return createWebGL2Context(this);
			}
			if (!i.contextWebGL2) {
				const ctx = createWebGL2Context(this);
				if (!ctx) return null;
				i.contextWebGL2 = ctx;
			}
			return i.contextWebGL2;
		}
		return null;
	}

	/**
	 * @ignore
	 * Internal: used by the console present to draw onto the screen WITHOUT
	 * marking the app as owning the screen.
	 */
	[CONSOLE_SCREEN_CTX](): CanvasRenderingContext2D {
		return ensureContext(this);
	}

	// @ts-expect-error
	addEventListener(
		type: 'touchstart' | 'touchmove' | 'touchend',
		listener: (ev: TouchEvent) => any,
		options?: boolean | AddEventListenerOptions,
	): void;
	addEventListener(
		type: string,
		listener: EventListenerOrEventListenerObject,
		options?: boolean | AddEventListenerOptions,
	): void;
	addEventListener(
		type: string,
		callback: EventListenerOrEventListenerObject | null,
		options?: boolean | AddEventListenerOptions,
	): void {
		if (type === 'touchstart' || type === 'touchmove' || type === 'touchend') {
			initTouchscreen();
		}
		super.addEventListener(type, callback, options);
	}

	/**
	 * Creates a {@link Blob} object representing the image contained on the screen.
	 *
	 * @example
	 *
	 * ```typescript
	 * screen.toBlob(blob => {
	 *   blob.arrayBuffer().then(buffer => {
	 *     Switch.writeFileSync('out.png', buffer);
	 *   });
	 * });
	 * ```
	 *
	 * @param callback A callback function with the resulting {@link Blob} object as a single argument. `null` may be passed if the image cannot be created for any reason.
	 * @param type A string indicating the image format. The default type is `image/png`. This image format will be also used if the specified type is not supported.
	 * @param quality A number between `0` and `1` indicating the image quality to be used when creating images using file formats that support lossy compression (such as `image/jpeg`). A user agent will use its default quality value if this option is not specified, or if the number is outside the allowed range.
	 */
	toBlob(
		callback: (blob: Blob | null) => void,
		type = 'image/png',
		quality = 0.92,
	) {
		$.canvasToBuffer(this, type, quality).then((buf: ArrayBuffer) => {
			callback(new Blob([buf], { type: normalizeImageMime(type) }));
		});
	}

	/**
	 * Returns a `data:` URL containing a representation of the image in the format specified by the type parameter.
	 *
	 * @example
	 *
	 * ```typescript
	 * const url = screen.toDataURL();
	 * fetch(url)
	 *   .then(res => res.arrayBuffer())
	 *   .then(buffer => {
	 *     Switch.writeFileSync('out.png', buffer);
	 *   });
	 * ```
	 *
	 * @param type A string indicating the image format. The default type is `image/png`. This image format will be also used if the specified type is not supported.
	 * @param quality A number between `0` and `1` indicating the image quality to be used when creating images using file formats that support lossy compression (such as `image/jpeg`). The default quality value will be used if this option is not specified, or if the number is outside the allowed range.
	 * @see https://developer.mozilla.org/docs/Web/API/HTMLCanvasElement/toDataURL
	 */
	toDataURL(type = 'image/png', quality = 0.92): string {
		stub();
	}

	// Compat with HTML DOM interface
	className = '';
	get nodeType() {
		return 1;
	}
	get nodeName() {
		return 'CANVAS';
	}
	get offsetWidth() {
		return this.width;
	}
	get offsetHeight() {
		return this.height;
	}
	get offsetTop() {
		return 0;
	}
	get offsetLeft() {
		return 0;
	}
	getAttribute(name: string): string | null {
		if (name === 'width') return String(this.width);
		if (name === 'height') return String(this.height);
		return null;
	}
	setAttribute(name: string, value: string | number) {}
}
$.canvasInitClass(Screen);
def(Screen);

// @ts-expect-error Internal constructor
export var screen = new Screen(INTERNAL_SYMBOL);

// Let the console present blit onto the screen (without claiming app ownership)
// and wire touch-drag scrollback.
registerConsoleScreen(() => (screen as any)[CONSOLE_SCREEN_CTX](), screen);
def(screen, 'screen');
