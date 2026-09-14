import type { PromiseState } from '@nx.js/inspect';
import type { CanvasRenderingContext2D } from './canvas/canvas-rendering-context-2d';
import type { ImageBitmap } from './canvas/image-bitmap';
import type { WebGL2RenderingContext } from './canvas/webgl2-rendering-context';
import type { WebGLRenderingContext } from './canvas/webgl-rendering-context';
import type { OffscreenCanvas } from './canvas/offscreen-canvas';
import type { OffscreenCanvasRenderingContext2D } from './canvas/offscreen-canvas-rendering-context-2d';
import type { Crypto, CryptoKey } from './crypto';
import type { DOMMatrix, DOMMatrixInit, DOMMatrixReadOnly } from './dommatrix';
import type { DOMPoint, DOMPointInit } from './dompoint';
import type { FontFace } from './font/font-face';
import type { Image } from './image';
import type {
	Callback,
	Keys,
	Opaque,
	RGBA,
	VibrationValues,
} from './internal';
import type { BatteryManager } from './navigator/battery';
import type { Gamepad, GamepadButton } from './navigator/gamepad';
import type { VirtualKeyboard } from './navigator/virtual-keyboard';
import type { Touch } from './polyfills/event';
import type { URL, URLSearchParams } from './polyfills/url';
import type { Screen } from './screen';
import type {
	Album,
	AlbumFile,
	Application,
	FileSystem,
	IRSensor,
	MemoryUsage,
	NetworkInfo,
	Profile,
	ProfileUid,
	DirEntry,
	ReadFileOptions,
	SaveData,
	SaveDataCreationInfo,
	Service,
	Stats,
	Versions,
} from './switch';
import type { Server, TlsContextOpaque } from './tcp';
import type { Algorithm, BufferSource } from './types';
import type { DatagramSocket } from './udp';
import type { Window } from './window';

type ClassOf<T> = {
	new (...args: any[]): T;
};

export type AudioContextHandle = Opaque<'AudioContextHandle'>;
export type AudioNodeHandle = Opaque<'AudioNodeHandle'>;

export interface BtleScanResult {
	address: string;
	name?: string;
	/** Raw BtdrvBleScanResult bytes (only when requested; diagnostics). */
	raw?: ArrayBuffer;
}

export interface BtleConnection {
	handle: number;
	address: string;
}

export interface BtleService {
	uuid: string;
	handle: number;
	instanceId: number;
	primary: boolean;
}

export interface BtleCharacteristic {
	uuid: string;
	handle: number;
	instanceId: number;
	properties: number;
}

export interface BtleDescriptor {
	uuid: string;
	handle: number;
}

export type BtleEvent =
	| { type: 'scan' | 'connection' | 'discovery' | 'mtu' }
	| {
			type: 'gatt';
			/** BtdrvBleEventType (8 = ClientNotify) */
			event: number;
			status: number;
			connId: number;
			op: number;
			serviceUuid: string;
			characteristicUuid: string;
			descriptorUuid: string;
			data: ArrayBuffer;
	  };

export type VideoHandle = Opaque<'VideoHandle'>;

export interface VideoMetadata {
	width: number;
	height: number;
	duration: number;
	hasAudio: boolean;
	hasVideo: boolean;
}

export interface VideoPlaybackState {
	currentTime: number;
	ended: boolean;
	seeking: boolean;
	/** Number of decoded video frames waiting for presentation. */
	buffered: number;
	/** Total video frames presented so far. */
	presentedFrames: number;
	/** Frames skipped because a newer frame was already due. */
	droppedFrames: number;
	/** Sticky fatal decode error message (if any). */
	error?: string;
}
type FileHandle = Opaque<'FileHandle'>;
type CanvasGradientOpaque = Opaque<'CanvasGradientOpaque'>;
type CompressHandle = Opaque<'CompressHandle'>;
type DecompressHandle = Opaque<'DecompressHandle'>;
type DecompressFileHandle = Opaque<'DecompressFileHandle'>;
type SaveDataIterator = Opaque<'SaveDataIterator'>;
type URLSearchParamsIterator = Opaque<'URLSearchParamsIterator'>;
export type USBNativeDevice = Opaque<'USBNativeDevice'>;

/** Effective socket (libnx SocketInitConfig) values, after nxjs.ini overrides. */
export interface NxSocketConfig {
	tcpTxBufSize: number;
	tcpRxBufSize: number;
	tcpTxBufMaxSize: number;
	tcpRxBufMaxSize: number;
	udpTxBufSize: number;
	udpRxBufSize: number;
	sbEfficiency: number;
	numBsdSessions: number;
	/** libnx BsdServiceType: 1=user, 2=system, 3=auto. */
	serviceType: number;
}

/**
 * Effective libuv worker thread pool settings (after `[threadpool]` overrides
 * from `nxjs.ini` are clamped). The pool services every async native operation
 * (fs, crypto, compression, image decode, dns, …). Defaults: 4 workers with
 * 1 MiB stacks (Switch-appropriate; upstream libuv's 8 MiB stacks cannot be
 * committed in applet mode).
 */
export interface NxThreadpoolConfig {
	/** Number of worker threads. */
	size: number;
	/** Stack size per worker thread, in bytes. */
	stackSize: number;
}

/**
 * On-screen console styling from the `[console]` section of `nxjs.ini`. Only the
 * keys present in the file are set. The global `console` seeds its options from
 * this at startup; an explicit `console.options =` assignment overrides it. The
 * shape matches the runtime's `TerminalOptions` (theme, fontSize, …).
 */
export interface NxConsoleConfig {
	fontSize?: number;
	lineHeight?: number;
	scrollback?: number;
	cursorStyle?: 'block' | 'underline' | 'bar';
	cursorOpacity?: number;
	/** Theme colors: `background`/`foreground`/`cursor` + the ANSI palette. */
	theme?: Record<string, string>;
}

/** Effective application config (from `nxjs.ini`); values reflect post-clamp reality. */
export interface NxConfig {
	/** Whether V8 JIT is enabled (vs jitless interpreter). */
	jit: boolean;
	/**
	 * Effective extra JIT code-arena headroom (MiB) reserved for WebAssembly
	 * beyond V8's 64 MiB code-range floor. 0 means WASM is effectively
	 * unavailable (no room for its code space) — opt in via `[v8]
	 * code_headroom_mb` / `wasm = on`. Always 0 when `jit` is false.
	 */
	codeHeadroomMb: number;
	/** Effective V8 max heap size in bytes (post-clamp; the value actually passed to V8). */
	heapLimit: number;
	/** Requested renderer mode. */
	renderer: 'auto' | 'cpu' | 'gpu';
	/** App-provided V8 flag string applied after the runtime defaults (empty if none). */
	v8Flags: string;
	/** Effective libnx socket configuration. */
	socket: NxSocketConfig;
	/** Effective libuv worker thread pool configuration. */
	threadpool: NxThreadpoolConfig;
	/** On-screen console styling from the `[console]` section (empty if none). */
	console: NxConsoleConfig;
	/** Whether an `nxjs.ini` file was found and parsed. */
	loaded: boolean;
}

export interface Init {
	// account.c
	accountInitialize(): () => void;
	accountProfileInit(c: ClassOf<Profile>): void;
	accountCurrentProfile(): Profile | null;
	accountSelectProfile(): Profile | null;
	accountProfileNew(uid: ProfileUid): Profile;
	accountProfiles(): Profile[];

	// album.c
	capsaInitialize(): () => void;
	albumInit(c: ClassOf<Album>): void;
	albumFileInit(c: ClassOf<AlbumFile>): void;
	albumFileList(album: Album): AlbumFile[];

	// applet.c
	appletIlluminance(): number;
	appletGetAppletType(): number;
	appletGetOperationMode(): number;
	appletSetMediaPlaybackState(state: boolean): void;

	// battery.c
	batteryInit(): void;
	batteryInitClass(c: ClassOf<BatteryManager>): void;
	batteryExit(): void;

	// canvas.c
	canvasNew(width: number, height: number): Screen | OffscreenCanvas;
	canvasToBuffer(
		canvas: Screen | OffscreenCanvas,
		type?: string,
		quality?: number,
	): Promise<ArrayBuffer>;
	canvasInitClass(c: ClassOf<Screen | OffscreenCanvas>): void;
	canvasContext2dNew(c: Screen): CanvasRenderingContext2D;
	canvasContext2dNew(c: OffscreenCanvas): OffscreenCanvasRenderingContext2D;
	canvasContext2dInitClass(
		c: ClassOf<CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D>,
	): void;
	canvasContext2dGetImageData(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		sx: number,
		sy: number,
		sw: number,
		sh: number,
	): ArrayBuffer;
	canvasContext2dGetTransform(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
	): number[];
	canvasContext2dGetFont(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
	): string;
	canvasContext2dSetFont(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		font: FontFace,
		size: number,
		fontString: string,
	): number[];
	canvasContext2dGetFillStyle(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
	): RGBA;
	canvasContext2dSetFillStyle(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		...rgba: RGBA
	): number[];
	canvasContext2dGetStrokeStyle(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
	): RGBA;
	canvasContext2dSetStrokeStyle(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		...rgba: RGBA
	): number[];
	canvasContext2dSetFillStyleGradient(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		gradient: CanvasGradientOpaque,
	): void;
	canvasContext2dSetStrokeStyleGradient(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		gradient: CanvasGradientOpaque,
	): void;
	canvasContext2dGetShadowColor(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
	): RGBA;
	canvasContext2dSetShadowColor(
		ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
		...rgba: RGBA
	): void;
	canvasGradientNewLinear(
		x0: number,
		y0: number,
		x1: number,
		y1: number,
	): CanvasGradientOpaque;
	canvasGradientNewRadial(
		x0: number,
		y0: number,
		r0: number,
		x1: number,
		y1: number,
		r1: number,
	): CanvasGradientOpaque;
	canvasGradientInitClass(c: any): void;
	canvasGradientAddColorStop(
		gradient: any,
		offset: number,
		r: number,
		g: number,
		b: number,
		a: number,
	): void;

	// compression.c
	compressNew(format: string): CompressHandle;
	compressWrite(
		handle: CompressHandle,
		buf: BufferSource,
	): Promise<ArrayBuffer>;
	compressFlush(handle: CompressHandle): Promise<ArrayBuffer | null>;
	decompressNew(format: string): DecompressHandle;
	decompressWrite(
		handle: DecompressHandle,
		buf: BufferSource,
	): Promise<ArrayBuffer>;
	decompressFlush(handle: DecompressHandle): Promise<ArrayBuffer | null>;
	/**
	 * Fused file decompression: open `path` at `[start, end)` and decompress it
	 * with `format`, reading + decompressing in one thread-pool dispatch per
	 * {@link decompressFilePull | `decompressFilePull()`} call. Powers the
	 * transparent fast path for `file.stream().pipeThrough(DecompressionStream)`.
	 */
	decompressFileNew(
		format: string,
		path: string,
		start?: number,
		end?: number,
		/** Per-pull output capacity in bytes (clamped 256 KiB..8 MiB). Larger
		 * means fewer thread-pool dispatches per MB of output. */
		outCap?: number,
	): DecompressFileHandle;
	/** Pull the next decompressed chunk, or `null` at end of stream. */
	decompressFilePull(
		handle: DecompressFileHandle,
	): Promise<ArrayBuffer | null>;

	// text.cc — native TextDecoder.decode(). Decodes the whole buffer into a
	// single string in C++, avoiding the JS polyfill's ~1-V8-handle-per-byte
	// `String.fromCharCode.apply` path (which leaks V8 HandleScope block reserve
	// on large inputs). `encoding` is one of "utf-8" | "utf-16le" | "utf-16be"
	// (the TextDecoder constructor pre-normalizes the WHATWG label). Throws a
	// TypeError on invalid input when `fatal`.
	textDecode(
		bytes: BufferSource,
		encoding: string,
		fatal: boolean,
		ignoreBOM: boolean,
	): string;

	// crypto.c
	cryptoKeyNew(
		algorithm: Algorithm,
		key: ArrayBuffer,
		extractable: boolean,
		keyUsages: KeyUsage[],
	): CryptoKey<any>;
	cryptoInit(c: ClassOf<Crypto>): void;
	cryptoKeyInit(c: ClassOf<CryptoKey<any>>): void;
	cryptoEncrypt(
		algorithm: Algorithm,
		key: CryptoKey<any>,
		data: BufferSource,
	): Promise<ArrayBuffer>;
	cryptoDecrypt(
		algorithm: Algorithm,
		key: CryptoKey<any>,
		data: BufferSource,
	): Promise<ArrayBuffer>;
	cryptoSign(
		algorithm: Algorithm,
		key: CryptoKey<any>,
		data: BufferSource,
	): Promise<ArrayBuffer>;
	cryptoVerify(
		algorithm: Algorithm,
		key: CryptoKey<any>,
		signature: BufferSource,
		data: BufferSource,
	): Promise<boolean>;
	cryptoExportKey(format: string, key: CryptoKey<any>): ArrayBuffer;
	cryptoGenerateKeyEc(namedCurve: string): [ArrayBuffer, ArrayBuffer];
	cryptoKeyNewEcPrivate(
		algorithm: any,
		privateKey: ArrayBuffer,
		publicKey: ArrayBuffer,
		extractable: boolean,
		usages: string[],
	): any;
	cryptoDeriveBits(
		algorithm: any,
		baseKey: CryptoKey<any>,
		length: number,
	): Promise<ArrayBuffer>;
	cryptoDigest(algorithm: string, buf: BufferSource): Promise<ArrayBuffer>;
	cryptoGenerateKeyRsa(
		modulusLength: number,
		publicExponent: number,
	): Promise<ArrayBuffer[]>;
	cryptoKeyNewRsa(
		algoName: string,
		hashName: string,
		type: string,
		n: ArrayBuffer,
		e: ArrayBuffer,
		d: ArrayBuffer | null,
		p: ArrayBuffer | null,
		q: ArrayBuffer | null,
		extractable: boolean,
		usages: string[],
	): CryptoKey<any>;
	cryptoRsaExportComponents(key: CryptoKey<any>): ArrayBuffer[];
	cryptoExportKeyPkcs8(key: CryptoKey<any>): ArrayBuffer;
	cryptoExportKeySpki(key: CryptoKey<any>): ArrayBuffer;
	cryptoImportKeyPkcs8Spki(
		format: string,
		data: BufferSource,
		algoName: string,
		paramName: string,
		extractable: boolean,
		usages: string[],
	): CryptoKey<any>;
	cryptoEcExportPublicRaw(key: CryptoKey<any>): ArrayBuffer;
	sha256Hex(str: string): string;

	// dommatrix.c
	dommatrixNew(values?: number[]): DOMMatrix | DOMMatrixReadOnly;
	dommatrixFromMatrix(init?: DOMMatrixInit): DOMMatrix | DOMMatrixReadOnly;
	dommatrixROInitClass(c: ClassOf<DOMMatrixReadOnly>): void;
	dommatrixInitClass(c: ClassOf<DOMMatrix>): void;
	dommatrixTransformPoint(m: DOMMatrixReadOnly, p: DOMPointInit): DOMPoint;

	// dns.c
	dnsResolve(hostname: string): Promise<string[]>;

	// error.c
	onError(fn: (err: any) => number): void;
	onUnhandledRejection(
		fn: (promise: Promise<unknown>, reason: any) => number,
	): void;

	// font.c
	fontFaceNew(data: ArrayBuffer): FontFace;
	getSystemFont(type: number): ArrayBuffer;

	// fs.c
	fclose(f: FileHandle): Promise<void>;
	fopen(path: string, mode: string, startOffset?: number): Promise<FileHandle>;
	fread(f: FileHandle, buf: ArrayBuffer): Promise<number | null>;
	fwrite(f: FileHandle, data: ArrayBuffer): Promise<void>;
	fsCreateBigFile(path: string): void;
	mkdir(path: string, mode: number): Promise<number>;
	mkdirSync(path: string, mode: number): number;
	openDir(path: string): Promise<object>;
	readDirNext(handle: object): Promise<DirEntry | null>;
	closeDir(handle: object): Promise<void>;
	readDirSync(path: string): string[] | null;
	readFile(path: string, opts?: ReadFileOptions): Promise<ArrayBuffer | null>;
	readFileSync(path: string, opts?: ReadFileOptions): ArrayBuffer | null;
	remove(path: string): Promise<void>;
	removeSync(path: string): void;
	rename(path: string, dest: string): Promise<void>;
	renameSync(path: string, dest: string): void;
	stat(path: string): Promise<Stats | null>;
	statSync(path: string): Stats | null;
	writeFile(path: string, data: ArrayBuffer): Promise<void>;
	writeFileSync(path: string, data: ArrayBuffer): void;
	appendFileSync(path: string, data: ArrayBuffer): void;

	// fsdev.c
	fsInit(c: ClassOf<FileSystem>): void;
	fsMount(fs: FileSystem, name: string): void;
	fsOpenBis(id: number): FileSystem;
	fsOpenSdmc(): FileSystem;
	fsOpenWithId(
		titleId: bigint,
		type: number,
		path: string,
		attributes: number,
	): FileSystem;
	saveDataInit(c: ClassOf<SaveData>): void;
	saveDataCreateSync(info: SaveDataCreationInfo, nacp?: ArrayBuffer): void;
	saveDataMount(saveData: SaveData, name: string): void;
	fsOpenSaveDataInfoReader(saveDataSpaceId: number): SaveDataIterator | null;
	fsSaveDataInfoReaderNext(iterator: SaveDataIterator): SaveData | null;

	// gamepad.c
	gamepadInit(c: ClassOf<Gamepad>): void;
	gamepadNew(index: number): Gamepad;
	gamepadButtonInit(c: ClassOf<GamepadButton>): void;
	gamepadButtonNew(gamepad: Gamepad, index: number): void;

	// hidsys.c
	/**
	 * Non-blocking check of the OS controller connect/disconnect event.
	 * Returns `true` when a controller was connected or disconnected since the
	 * last call (and invalidates the cached `Gamepad.id` values), `false`
	 * otherwise. Called once per frame to drive `gamepadconnected` /
	 * `gamepaddisconnected` dispatch.
	 */
	gamepadConnectionChanged(): boolean;

	// image.c
	imageInit(c: ClassOf<Image | ImageBitmap>): void;
	imageNew(width?: number, height?: number): Image | ImageBitmap;
	imageDecode(img: Image | ImageBitmap, data: ArrayBuffer): Promise<void>;
	imageClose(img: ImageBitmap): void;
	/** Give an existing Image a fresh w*h BGRA backing buffer. */
	/** Brotli-decompress (vendored google/brotli decoder). The primitive
	 * WOFF2 decoding is built on; also what a `Content-Encoding: br` or
	 * `DecompressionStream('br')` path would use. */
	brotliDecompress(data: ArrayBuffer | ArrayBufferView): ArrayBuffer;
	imageAlloc(image: Image, width: number, height: number): void;
	imageWriteRGBA(
		img: Image | ImageBitmap,
		bytes: ArrayBuffer | Uint8Array | Uint8ClampedArray,
		premultiply?: boolean,
	): void;
	// Fix D (2026-09-05) — zero-swizzle BGRA write for decoded video frames
	// (bytes already in Skia's ARGB32/BGRA order + opaque). Straight memcpy.
	imageWriteBGRA(
		img: Image | ImageBitmap,
		bytes: ArrayBuffer | Uint8Array | Uint8ClampedArray,
	): void;
	// 2026-09-06 — planar-I420 video-frame write. `bytes` is contiguous
	// Y|U|V (1.5 B/px). Marks the image YUV so drawImage builds a GPU YUVA
	// SkImage (Skia does YUV→RGB), uploading ~2.6× less per frame than BGRA.
	// `colorSpace`: 0=Rec709 ltd, 1=Rec601 ltd, 2=full/JPEG, 3=Rec709 full.
	imageWriteYUV(
		img: Image | ImageBitmap,
		bytes: ArrayBuffer | Uint8Array | Uint8ClampedArray,
		width: number,
		height: number,
		colorSpace?: number,
	): void;
	// 2026-09-06 — EGL swap interval (vsync divisor): 1 = 60 Hz, 2 = 30 Hz.
	// The shell pins fullscreen 30 fps video to 30 Hz for a judder-free 1:1
	// cadence. No-op on the raster fallback.
	gfxSetSwapInterval(interval: number): void;
	// Ledger #78 — engine-side BGRA byte copy with premul-state conversion
	// + optional Y-flip. Reads src.unpremultiplied and dst premul target
	// to decide whether to do a same-state memcpy, premultiply, or
	// un-premultiply during the row-by-row copy.
	imageCopyPixels(
		dst: Image | ImageBitmap,
		src: Image | ImageBitmap,
		dstPremultiply: boolean,
		flipY?: boolean,
	): void;

	// irs.c
	irsInit(): () => void;
	irsSensorNew(image: ImageBitmap, color: RGBA): IRSensor;
	irsSensorStart(s: IRSensor): void;
	irsSensorStop(s: IRSensor): void;
	irsSensorUpdate(s: IRSensor): boolean;

	// memory.c
	memoryUsage(): MemoryUsage;
	lowMemoryNotification(): void;

	// main.c
	argv: string[];
	entrypoint: string;
	/**
	 * Source for `Application.self`: a `.nro` path for standalone/slim NRO apps,
	 * or `null` for installed titles (fat/slim NSP) — in which case
	 * `nsAppNew(null)` resolves via the running process's ProgramId. Identifies
	 * the launched app, not the shared runtime NRO.
	 */
	selfNroPath: string | null;
	version: Versions;
	/** Configured bsdsocket TCP receive buffer size (bytes) for this memory regime. */
	tcpRxBufSize: number;
	/** Effective application config parsed from `nxjs.ini` (next to the entrypoint). */
	config: NxConfig;
	exit(): never;
	queueMicrotask(callback: () => void): void;
	cwd(): string;
	chdir(dir: string): void;
	print(v: string): void;
	printErr(v: string): void;
	getInternalPromiseState(p: Promise<unknown>): [PromiseState, unknown];
	getenv(name: string): string | undefined;
	setenv(name: string, value: string): void;
	unsetenv(name: string): void;
	envToObject(): Record<string, string>;
	onFrame(fn: (plusDown: boolean) => void): void;
	onExit(fn: () => void): void;
	framebufferInit(screen: Screen): void;
	hidInitializeTouchScreen(): void;
	hidGetTouchScreenStates(): Touch[] | undefined;
	hidInitializeKeyboard(): void;
	hidInitializeVibrationDevices(): void;
	hidGetKeyboardStates(): Keys;
	hidSendVibrationValues(v: VibrationValues): void;

	// webgl.c
	/**
	 * Initializes EGL + an OpenGL ES 3 context on the screen. Returns the
	 * native context carrier object, or `undefined` when GL init fails.
	 */
	webglContextNew(screen: Screen): WebGLRenderingContext | undefined;
	// Installs the v1 method table (Phase 2.C) on whichever class is passed
	// (typically WebGLRenderingContext). The v2 path uses the separate
	// webgl2InitClass binding (Phase 2.G.0) — see NXJS_PATCHES_NEEDED.md #15
	// for the table-split rationale.
	webglInitClass(
		c: ClassOf<WebGLRenderingContext>,
		classes: Record<string, unknown>,
	): void;

	// Phase 2.G.0 — SEPARATE v2 context factory + class init binding pair.
	// Engine impls share WebGLState with v1 but the install paths are wholly
	// distinct so v1's hardware-verified JIT install shape stays byte-
	// identical and v2's empty-but-correctly-shaped install can be hardware-
	// verified independently. See NXJS_PATCHES_NEEDED.md #14 + #15.
	webgl2ContextNew(screen: Screen): WebGL2RenderingContext | undefined;
	webgl2InitClass(
		c: ClassOf<WebGL2RenderingContext>,
		classes: Record<string, unknown>,
	): void;

	// nifm.c
	nifmInitialize(): () => void;
	networkInfo(): NetworkInfo;

	// sensors.cc
	sensorsSixAxisStart(): boolean;
	sensorsSixAxisRead(): {
		acceleration: { x: number; y: number; z: number };
		angularVelocity: { x: number; y: number; z: number };
		angle: { x: number; y: number; z: number };
		/** Fused 3x3 orientation matrix, row-major flat [m00..m22]. */
		direction: number[];
		samplingNumber: bigint;
		deltaTime: bigint;
	} | null;
	sensorsSixAxisStop(): void;

	// ns.c
	nsInitialize(): () => void;
	nsAppInit(c: ClassOf<Application>): void;
	nsAppNew(id: string | bigint | ArrayBuffer | null): Application;
	nsAppNext(index: number): bigint | null;

	// service.c
	serviceInit(c: ClassOf<Service>): () => void;
	serviceNew(name?: string): Service;

	// software-keyboard.c
	swkbdCreate(fns: {
		onCancel: (this: VirtualKeyboard) => void;
		onChange: (
			this: VirtualKeyboard,
			str: string,
			cursorPos: number,
			dicStartCursorPos: number,
			dicEndCursorPos: number,
		) => void;
		onSubmit: (this: VirtualKeyboard, str: string) => void;
		onCursorMove: (
			this: VirtualKeyboard,
			str: string,
			cursorPos: number,
		) => void;
	}): VirtualKeyboard;
	swkbdSetCursorPos(s: VirtualKeyboard, cursorPos: number): void;
	swkbdSetInputText(s: VirtualKeyboard, value: string): void;
	swkbdShow(s: VirtualKeyboard): [number, number, number, number];
	swkbdHide(s: VirtualKeyboard): void;
	swkbdUpdate(this: VirtualKeyboard): void;

	// web.c
	webAppletNew(): any;
	webAppletStart(applet: any, url: string, options: Record<string, any>): void;
	webAppletAppear(applet: any): boolean;
	webAppletSendMessage(applet: any, msg: string): boolean;
	webAppletPollMessages(applet: any): string[];
	webAppletRequestExit(applet: any): void;
	webAppletClose(applet: any): void;
	webAppletIsRunning(applet: any): boolean;
	webAppletGetMode(applet: any): string;

	// tcp.c
	connect(cb: Callback<number>, ip: string, port: number): void;
	write(cb: Callback<number>, fd: number, data: ArrayBuffer): void;
	read(cb: Callback<number>, fd: number, buffer: ArrayBuffer): void;
	close(fd: number): void;
	tcpServerInit(c: any): void;
	tcpServerNew(
		ip: string,
		port: number,
		onAccept: (fd: number) => void,
	): Server;

	// udp.c
	udpInit(c: any): void;
	udpNew(
		ip: string,
		port: number,
		onRecv: (
			err: Error | null,
			data?: ArrayBuffer,
			remoteIp?: string,
			remotePort?: number,
		) => void,
	): DatagramSocket;
	udpSend(
		cb: Callback<number>,
		fd: number,
		data: ArrayBuffer,
		ip: string,
		port: number,
	): void;

	// tls.c
	tlsHandshake(
		cb: Callback<TlsContextOpaque>,
		fd: number,
		hostname: string,
		rejectUnauthorized: boolean,
	): void;
	tlsWrite(
		cb: Callback<number>,
		ctx: TlsContextOpaque,
		data: ArrayBuffer,
	): void;
	tlsRead(
		cb: Callback<number>,
		ctx: TlsContextOpaque,
		buffer: ArrayBuffer,
	): void;
	tlsClose(ctx: TlsContextOpaque): void;

	// url.c
	urlInit(c: ClassOf<URL>): void;
	urlNew(url: string | URL, base?: string | URL): URL;
	urlSearchInit(c: ClassOf<URLSearchParams>): void;
	urlSearchNew(input: string, url?: URL): URLSearchParams;
	urlSearchIterator(
		params: URLSearchParams,
		type: number,
	): URLSearchParamsIterator;
	urlSearchIteratorNext(it: URLSearchParamsIterator): any;

	// audio.cc — Web Audio API
	audioContextNew(sampleRate: number, offline: boolean): AudioContextHandle;
	audioContextClose(ctx: AudioContextHandle): void;
	audioContextSuspend(ctx: AudioContextHandle): void;
	audioContextResume(ctx: AudioContextHandle): void;
	audioContextCurrentTime(ctx: AudioContextHandle): number;
	audioContextDestination(ctx: AudioContextHandle): AudioNodeHandle;
	/** `aux` is a type-specific creation parameter (DelayNode: maxDelayTime in
	 * seconds); ignored by other node types. */
	audioNodeNew(
		ctx: AudioContextHandle,
		type: number,
		aux?: number,
	): AudioNodeHandle;
	audioNodeConnect(src: AudioNodeHandle, dst: AudioNodeHandle): void;
	audioNodeDisconnect(src: AudioNodeHandle, dst?: AudioNodeHandle): void;
	audioParamValue(node: AudioNodeHandle, index: number): number;
	audioParamSetValue(node: AudioNodeHandle, index: number, value: number): void;
	audioParamSchedule(
		node: AudioNodeHandle,
		index: number,
		type: number,
		time: number,
		value: number,
		timeConstant: number,
	): void;
	audioParamSetValueCurve(
		node: AudioNodeHandle,
		index: number,
		curve: Float32Array,
		startTime: number,
		duration: number,
	): void;
	audioParamCancel(node: AudioNodeHandle, index: number, time: number): void;
	audioSourceSetBuffer(
		node: AudioNodeHandle,
		channels: Float32Array[],
		length: number,
		sampleRate: number,
	): void;
	audioSourceStart(
		node: AudioNodeHandle,
		when: number,
		offset: number,
		duration: number,
	): void;
	audioSourceStop(node: AudioNodeHandle, when: number): void;
	audioSourceSetLoop(
		node: AudioNodeHandle,
		loop: boolean,
		loopStart: number,
		loopEnd: number,
	): void;
	audioSourceState(node: AudioNodeHandle): number;
	audioOscillatorSetType(node: AudioNodeHandle, type: number): void;
	/** Select the filter shape (an `nx_audio_biquad_type`). Resets the
	 * filter state so a live switch does not ring. */
	audioBiquadSetType(node: AudioNodeHandle, type: number): void;
	/** Fill `mag` and `phase` with the filter response at each frequency in
	 * `freqHz`. All three arrays must be the same length. */
	audioBiquadFrequencyResponse(
		node: AudioNodeHandle,
		freqHz: Float32Array,
		mag: Float32Array,
		phase: Float32Array,
	): void;
	/** Install a ConvolverNode impulse response. The samples ARE copied (they
	 * are transformed into partitioned spectra), unlike `audioSourceSetBuffer`.
	 * `channels: null` clears the response, after which the node is silent. */
	audioConvolverSetBuffer(
		node: AudioNodeHandle,
		channels: Float32Array[] | null,
		length: number,
		sampleRate: number,
		normalize: boolean,
	): void;
	/** Current gain reduction in dB (<= 0) for DynamicsCompressorNode.reduction. */
	audioCompressorReduction(node: AudioNodeHandle): number;
	/** Fill `out` with the analyser's most-recent time-domain samples
	 * (newest last, each in [-1, 1]). `out.length` samples are returned. */
	audioAnalyserFloatTimeData(node: AudioNodeHandle, out: Float32Array): void;
	/** Decode a whole audio file to planar f32. `targetSampleRate` (the
	 * destination context's rate) makes swresample do the rate conversion
	 * once, here, rather than leaving it to the buffer-source node's
	 * per-sample linear interpolation; omit or pass 0 to keep the file's
	 * native rate. The resolved `sampleRate` is what was actually produced. */
	audioDecode(
		buffer: ArrayBuffer,
		targetSampleRate?: number,
	): Promise<{
		channelData: ArrayBuffer[];
		length: number;
		sampleRate: number;
	}>;
	audioOfflineRender(
		ctx: AudioContextHandle,
		numberOfChannels: number,
		length: number,
	): Promise<ArrayBuffer[]>;

	// bluetooth.cc — Web Bluetooth (BLE GATT client over btm.u + bt)
	btleInit(): void;
	btleExit(): void;
	/**
	 * With `serviceUuid`: a "smart device" scan matching devices that
	 * advertise that service UUID. With `companyId` (+ up to 6 pattern
	 * bytes): a "general" scan with a custom manufacturer-data filter.
	 * Without either: the "general" scan (Nintendo accessory filter).
	 */
	btleScanStart(
		serviceUuid?: string | null,
		companyId?: number | null,
		pattern?: ArrayBuffer | null,
	): void;
	btleScanStop(): void;
	btleScanResults(includeRaw?: boolean): BtleScanResult[];
	btleConnect(address: string): void;
	btleDisconnect(handle: number): void;
	btleConnections(): BtleConnection[];
	btleGetServices(conn: number): BtleService[];
	btleGetCharacteristics(
		conn: number,
		serviceHandle: number,
	): BtleCharacteristic[];
	btleGetDescriptors(conn: number, charHandle: number): BtleDescriptor[];
	btleRead(
		conn: number,
		primary: boolean,
		serviceUuid: string,
		serviceInstance: number,
		charUuid: string,
		charInstance: number,
	): void;
	btleWrite(
		conn: number,
		primary: boolean,
		serviceUuid: string,
		serviceInstance: number,
		charUuid: string,
		charInstance: number,
		data: ArrayBuffer,
		withResponse: boolean,
	): void;
	btleWriteDescriptor(
		conn: number,
		primary: boolean,
		serviceUuid: string,
		serviceInstance: number,
		charUuid: string,
		charInstance: number,
		descUuid: string,
		descInstance: number,
		data: ArrayBuffer,
	): void;
	btleNotify(
		conn: number,
		primary: boolean,
		serviceUuid: string,
		serviceInstance: number,
		charUuid: string,
		charInstance: number,
		enable: boolean,
	): void;
	btlePollEvents(): BtleEvent[];
	/**
	 * Unfiltered btdrv-level scan (no result delivery) used to prime the
	 * Bluetooth stack's device cache before connecting by explicit address.
	 */
	btleRawScanStart(): void;
	btleRawScanStop(): void;
	/** Register/unregister a BLE GATT data path for a service UUID. */
	btleRegisterDataPath(uuid: string, register: boolean): void;
	/** Request an ATT MTU for the connection (browsers negotiate ~517). */
	btleConfigureMtu(conn: number, mtu: number): void;
	btleGetMtu(conn: number): number;

	// usb.cc — WebUSB over libnx usb:hs (USB host mode)
	usbInit(): void;
	usbExit(): void;
	usbGetDevices(filter?: {
		vendorId?: number;
		productId?: number;
		classCode?: number;
		subclassCode?: number;
		protocolCode?: number;
		interfaceClass?: number;
		interfaceSubclass?: number;
		interfaceProtocol?: number;
	}): USBNativeDevice[];
	/** Returns `true` if a USB hotplug (attach/detach) event fired since the last check. */
	usbHotplugCheck(): boolean;
	usbDeviceOpen(device: USBNativeDevice): void;
	usbDeviceClose(device: USBNativeDevice): void;
	usbClaimInterface(device: USBNativeDevice, interfaceNumber: number): void;
	usbReleaseInterface(device: USBNativeDevice, interfaceNumber: number): void;
	usbSelectAlternateInterface(
		device: USBNativeDevice,
		interfaceNumber: number,
		alternateSetting: number,
	): void;
	/** Clear a halted endpoint (CLEAR_FEATURE + host toggle reset). */
	usbClearHalt(
		device: USBNativeDevice,
		directionIn: boolean,
		endpointNumber: number,
	): void;
	usbTransferIn(
		device: USBNativeDevice,
		endpointNumber: number,
		length: number,
	): ArrayBuffer;
	/** Post a non-blocking bulk-IN transfer (idempotent per endpoint). */
	usbReadStart(
		device: USBNativeDevice,
		endpointNumber: number,
		length: number,
	): void;
	/** Poll a posted bulk-IN transfer: `ArrayBuffer` when done, `undefined` while pending. */
	usbReadPoll(
		device: USBNativeDevice,
		endpointNumber: number,
	): ArrayBuffer | undefined;
	usbTransferOut(
		device: USBNativeDevice,
		endpointNumber: number,
		data: BufferSource,
	): number;
	/** Post a non-blocking bulk-OUT transfer (one in flight per endpoint). */
	usbWriteStart(
		device: USBNativeDevice,
		endpointNumber: number,
		data: BufferSource,
	): void;
	/** Poll a posted bulk-OUT transfer: `bytesWritten` when done, `undefined` while pending. */
	usbWritePoll(
		device: USBNativeDevice,
		endpointNumber: number,
	): number | undefined;
	usbControlTransferIn(
		device: USBNativeDevice,
		setup: {
			requestType: string;
			recipient: string;
			request: number;
			value: number;
			index: number;
		},
		length: number,
	): ArrayBuffer;
	usbControlTransferOut(
		device: USBNativeDevice,
		setup: {
			requestType: string;
			recipient: string;
			request: number;
			value: number;
			index: number;
		},
		data?: BufferSource,
	): number;
	usbResetDevice(device: USBNativeDevice): void;

	// nfc.cc — Web NFC over libnx nfc:user (NTAG / Type-2 passthrough)
	/** Initialize nfc:user + enumerate the reader device. Returns `false`
	 * (without throwing) when NFC is unsupported, e.g. under Citron. */
	nfcInit(): boolean;
	nfcExit(): void;
	/** True when NFC is enabled in settings AND a reader controller is present. */
	nfcIsAvailable(): boolean;
	nfcStartDetection(): void;
	nfcStopDetection(): void;
	/** Current NfcDeviceState (0..4; 2=TagFound, 4=TagMounted), -1 if unavailable. */
	nfcGetState(): number;
	/** Detected tag's UID + protocol/type, or `undefined` when no tag. */
	nfcGetTagInfo(): { uid: ArrayBuffer; protocol: number; tagType: number } | undefined;
	nfcKeepSession(): void;
	nfcReleaseSession(): void;
	/** Raw ISO14443-3A command passthrough (e.g. Type-2 READ [0x30,page]). */
	nfcTransceive(command: BufferSource): ArrayBuffer;

	// video.cc — Video element (ffmpeg media pipeline)
	videoNew(): VideoHandle;
	videoLoad(
		video: VideoHandle,
		path: string | null,
		buffer: ArrayBuffer | null,
	): Promise<VideoMetadata>;
	videoPlay(video: VideoHandle): void;
	videoPause(video: VideoHandle): void;
	videoSeek(video: VideoHandle, seconds: number): void;
	videoSetLoop(video: VideoHandle, loop: boolean): void;
	videoTick(video: VideoHandle): boolean;
	videoState(video: VideoHandle): VideoPlaybackState;
	videoCreateAudioNode(
		video: VideoHandle,
		ctx: AudioContextHandle,
	): AudioNodeHandle | null;
	videoClose(video: VideoHandle): void;

	// video-decoder.cc — Switch.VideoDecoder (raw-pixel decode API).
	// Cut #22 (2026-07-01): thin V8 binding over nx_media_*, distinct from
	// videoNew/videoLoad above (which power the drawImage-integrated Video
	// element). See packages/runtime/src/switch/video-decoder.ts.
	videoDecoderInit(ctor: Function): void;
	videoDecoderNew(url: string, opts?: unknown): unknown;
	videoDecoderPlay(dec: unknown): void;
	videoDecoderPause(dec: unknown): void;
	videoDecoderSeek(dec: unknown, seconds: number): void;
	videoDecoderClose(dec: unknown): void;
	videoDecoderNextFrame(dec: unknown, bgra?: boolean): {
		data: ArrayBuffer | null;
		width: number;
		height: number;
		pts: number;
		ended: boolean;
		// 2026-09-06 — set when the decoder was opened with `yuv: true`: `data`
		// is planar I420 (1.5 B/px), `colorSpace` is the neutral tag for the
		// GPU YUVA path (0=709 ltd,1=601 ltd,2=full,3=709 full).
		yuv?: boolean;
		colorSpace?: number;
	} | null;
	// Cut #22b (2026-07-02): audio-graph attach + volume/mute for
	// Switch.VideoDecoder. Restores playback for audio-bearing sources
	// (spectraplay MP3 flow + audio-bearing <video> tracks).
	videoDecoderCreateAudioNode(
		dec: unknown,
		ctx: AudioContextHandle,
	): AudioNodeHandle | null;
	videoDecoderSetVolume(dec: unknown, value: number): void;
	videoDecoderSetMuted(dec: unknown, muted: boolean): void;
	// Cut #22b Stage 2 (2026-07-02): visualizer surface — spectraplay's
	// `audio.getFrequencyData(specData)` / `audio.getWaveform(waveData)`.
	videoDecoderGetWaveform(dec: unknown, out: Float32Array): boolean;
	videoDecoderGetFrequencyData(dec: unknown, out: Float32Array): boolean;
	videoDecoderGetAudioLevels(dec: unknown): number[];

	// (Uint8Array base64/hex methods are provided natively by V8 — no binding.)

	// window.c
	windowInit(c: Window): void;

	// path2d.c — Path2D backed by a native SkPath (user space). The methods
	// are installed on the prototype by path2dInitClass; only the constructor
	// backing + class installer are exposed on `$`.
	path2dNew(path?: unknown): unknown;
	path2dInitClass(Path2D: Function): void;
}

export const $: Init = (globalThis as any).$;
delete (globalThis as any).$;
