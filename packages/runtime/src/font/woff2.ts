/**
 * WOFF2 → SFNT (TTF/OTF) decoding.
 *
 * WOFF2 is what essentially every modern web font ships as — Google Fonts
 * serves it, and any Vite/webpack build emits it — and nothing in this stack
 * could read one: devkitPro ships no brotli, so FreeType here is built without
 * `FT_CONFIG_OPTION_USE_BROTLI` and the FontFace constructor could only report
 * "failed to load font face". The brotli half is a vendored native decoder
 * (`$.brotliDecompress`); this module does the container work on top of it.
 *
 * Two things make WOFF2 more than "gunzip the tables":
 *
 *  1. The table directory is a compact encoding — known tags are a 6-bit
 *     index into a fixed 63-entry table, lengths are UIntBase128 — and table
 *     data is one single brotli stream with every table concatenated, in the
 *     directory's order, with NO padding between them.
 *  2. `glyf` and `loca` are usually stored TRANSFORMED: the glyph outlines are
 *     split into parallel streams (contour counts, point counts, flags,
 *     triplet-encoded coordinate deltas, composites, bboxes, instructions) and
 *     `loca` is dropped entirely, to be rebuilt from the reconstructed `glyf`.
 *     That reverse transform is most of this file.
 *
 * @see https://www.w3.org/TR/WOFF2/
 */

import { $ } from '../$';

/** The 63 tags that get a 6-bit index in the table directory; index 63 means
 * "an explicit 4-byte tag follows". Order is normative — do not sort. */
const KNOWN_TAGS = [
	'cmap', 'head', 'hhea', 'hmtx', 'maxp', 'name', 'OS/2', 'post',
	'cvt ', 'fpgm', 'glyf', 'loca', 'prep', 'CFF ', 'VORG', 'EBDT',
	'EBLC', 'gasp', 'hdmx', 'kern', 'LTSH', 'PCLT', 'VDMX', 'vhea',
	'vmtx', 'BASE', 'GDEF', 'GPOS', 'GSUB', 'EBSC', 'JSTF', 'MATH',
	'CBDT', 'CBLC', 'COLR', 'CPAL', 'SVG ', 'sbix', 'acnt', 'avar',
	'bdat', 'bloc', 'bsln', 'cvar', 'fdsc', 'feat', 'fmtx', 'fvar',
	'gvar', 'hsty', 'just', 'lcar', 'mort', 'morx', 'opbd', 'prop',
	'trak', 'Zapf', 'Silf', 'Glat', 'Gloc', 'Feat', 'Sill',
];

const WOFF2_SIGNATURE = 0x774f4632; // 'wOF2'

/** True when `buffer` looks like a WOFF2 file. */
export function isWoff2(buffer: ArrayBuffer): boolean {
	if (buffer.byteLength < 48) return false;
	return new DataView(buffer).getUint32(0) === WOFF2_SIGNATURE;
}

class Reader {
	private view: DataView;
	pos = 0;
	constructor(public bytes: Uint8Array) {
		this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	}
	get remaining(): number {
		return this.bytes.length - this.pos;
	}
	u8(): number {
		if (this.pos >= this.bytes.length) throw new RangeError('woff2: truncated');
		return this.bytes[this.pos++];
	}
	u16(): number {
		const v = this.view.getUint16(this.pos);
		this.pos += 2;
		return v;
	}
	i16(): number {
		const v = this.view.getInt16(this.pos);
		this.pos += 2;
		return v;
	}
	u32(): number {
		const v = this.view.getUint32(this.pos);
		this.pos += 4;
		return v;
	}
	/** UIntBase128: up to 5 bytes, 7 bits each, big-endian, high bit = more. */
	base128(): number {
		let value = 0;
		for (let i = 0; i < 5; i++) {
			const byte = this.u8();
			// Spec: no leading zeros, and the value must fit in 32 bits.
			if (i === 0 && byte === 0x80) throw new RangeError('woff2: base128 leading zero');
			if (value & 0xfe000000) throw new RangeError('woff2: base128 overflow');
			value = (value << 7) | (byte & 0x7f);
			if ((byte & 0x80) === 0) return value >>> 0;
		}
		throw new RangeError('woff2: base128 too long');
	}
	/** 255UInt16: a compact 1-or-2-byte unsigned short. */
	short255(): number {
		const code = this.u8();
		if (code === 253) return this.u16();
		if (code === 254) return this.u8() + 253 * 2;
		if (code === 255) return this.u8() + 253;
		return code;
	}
}

class Writer {
	bytes: Uint8Array;
	private view: DataView;
	pos = 0;
	constructor(capacity: number) {
		this.bytes = new Uint8Array(capacity);
		this.view = new DataView(this.bytes.buffer);
	}
	private need(n: number): void {
		if (this.pos + n <= this.bytes.length) return;
		let cap = this.bytes.length * 2 || 1024;
		while (cap < this.pos + n) cap *= 2;
		const grown = new Uint8Array(cap);
		grown.set(this.bytes);
		this.bytes = grown;
		this.view = new DataView(grown.buffer);
	}
	u8(v: number): void {
		this.need(1);
		this.bytes[this.pos++] = v & 0xff;
	}
	u16(v: number): void {
		this.need(2);
		this.view.setUint16(this.pos, v & 0xffff);
		this.pos += 2;
	}
	i16(v: number): void {
		this.need(2);
		this.view.setInt16(this.pos, v);
		this.pos += 2;
	}
	u32(v: number): void {
		this.need(4);
		this.view.setUint32(this.pos, v >>> 0);
		this.pos += 4;
	}
	raw(src: Uint8Array): void {
		this.need(src.length);
		this.bytes.set(src, this.pos);
		this.pos += src.length;
	}
	padTo4(): void {
		while (this.pos & 3) this.u8(0);
	}
	result(): Uint8Array {
		return this.bytes.subarray(0, this.pos);
	}
}

interface TableEntry {
	tag: string;
	flags: number;
	origLength: number;
	transformLength: number;
	transformed: boolean;
	data?: Uint8Array;
}

/**
 * Decode a WOFF2 buffer into an SFNT (TTF/OTF) buffer.
 *
 * Returns `null` when the input is not WOFF2. Throws when it IS WOFF2 but
 * malformed, so callers can tell "not my format" from "broken file".
 */
export function woff2ToSfnt(buffer: ArrayBuffer): ArrayBuffer | null {
	if (!isWoff2(buffer)) return null;
	const r = new Reader(new Uint8Array(buffer));
	r.u32(); // signature, already checked
	const flavor = r.u32();
	r.u32(); // length
	const numTables = r.u16();
	r.u16(); // reserved
	const totalSfntSize = r.u32();
	r.u32(); // totalCompressedSize
	r.u16(); // majorVersion
	r.u16(); // minorVersion
	r.u32(); // metaOffset
	r.u32(); // metaLength
	r.u32(); // metaOrigLength
	r.u32(); // privOffset
	r.u32(); // privLength

	if (numTables === 0) throw new RangeError('woff2: no tables');

	const tables: TableEntry[] = [];
	for (let i = 0; i < numTables; i++) {
		const flags = r.u8();
		const index = flags & 0x3f;
		const tag = index === 0x3f ? readTag(r) : KNOWN_TAGS[index];
		if (!tag) throw new RangeError(`woff2: bad table index ${index}`);
		const origLength = r.base128();
		// Transform version lives in bits 6-7. For glyf/loca, version 0 means
		// TRANSFORMED and 3 means untransformed — the reverse of every other
		// table, where 0 means untransformed. This inversion is the single
		// easiest thing to get backwards in a WOFF2 reader.
		const version = (flags >> 6) & 0x3;
		const isGlyfOrLoca = tag === 'glyf' || tag === 'loca';
		const transformed = isGlyfOrLoca ? version === 0 : version !== 0;
		const transformLength = transformed ? r.base128() : origLength;
		tables.push({ tag, flags, origLength, transformLength, transformed });
	}

	// `flavor === 'ttcf'` would put a collection directory here. Unsupported:
	// no web font is delivered as a collection, and the caller falls back.
	if (flavor === 0x74746366) throw new RangeError('woff2: font collections unsupported');

	const compressed = r.bytes.subarray(r.pos);
	const decompressed = new Uint8Array(brotliDecompress(compressed));

	// One stream, tables back to back in directory order, no padding.
	let offset = 0;
	for (const t of tables) {
		const end = offset + t.transformLength;
		if (end > decompressed.length) {
			throw new RangeError(`woff2: table ${t.tag} runs past the decompressed data`);
		}
		t.data = decompressed.subarray(offset, end);
		offset = end;
	}

	const byTag = new Map<string, TableEntry>();
	for (const t of tables) byTag.set(t.tag, t);

	const glyf = byTag.get('glyf');
	if (glyf?.transformed) {
		const loca = byTag.get('loca');
		const head = byTag.get('head');
		if (!loca || !head?.data) throw new RangeError('woff2: transformed glyf without loca/head');
		const rebuilt = reconstructGlyf(glyf.data as Uint8Array);
		glyf.data = rebuilt.glyf;
		// indexToLocFormat (head + 50) says which loca width to write.
		const indexToLocFormat = new DataView(
			head.data.buffer, head.data.byteOffset, head.data.byteLength,
		).getInt16(50);
		loca.data = buildLoca(rebuilt.offsets, indexToLocFormat);
		loca.origLength = loca.data.length;
		glyf.origLength = glyf.data.length;
	}

	// hmtx transform 1 drops the leading lsb array, to be recomputed from the
	// glyph bounding boxes. Not attempted: it is rare in practice (fonttools
	// and woff2_compress do not emit it by default) and getting it silently
	// wrong would shift every glyph. Fail loudly instead.
	const hmtx = byTag.get('hmtx');
	if (hmtx?.transformed) throw new RangeError('woff2: transformed hmtx unsupported');

	return buildSfnt(flavor, tables, totalSfntSize);
}

function readTag(r: Reader): string {
	let s = '';
	for (let i = 0; i < 4; i++) s += String.fromCharCode(r.u8());
	return s;
}

function brotliDecompress(input: Uint8Array): ArrayBuffer {
	const fn = ($ as unknown as {
		brotliDecompress?: (b: ArrayBuffer | Uint8Array) => ArrayBuffer;
	}).brotliDecompress;
	if (typeof fn !== 'function') {
		throw new Error('woff2: brotli decompression is not available in this build');
	}
	// Pass a tight copy: the native side reads `byteLength` from the view, but
	// a subarray of a larger buffer is the common case here and copying once
	// is cheaper than reasoning about every caller's offsets.
	const copy = new Uint8Array(input.length);
	copy.set(input);
	return fn(copy);
}

/* ------------------------------------------------------------------ *
 * glyf / loca reverse transform
 * ------------------------------------------------------------------ */

/** Reverse the WOFF2 `glyf` transform, returning the rebuilt table and the
 * per-glyph offsets that `loca` must describe. */
function reconstructGlyf(data: Uint8Array): { glyf: Uint8Array; offsets: number[] } {
	const head = new Reader(data);
	head.u16(); // reserved
	const optionFlags = head.u16();
	const numGlyphs = head.u16();
	const indexFormat = head.u16();
	const nContourStreamSize = head.u32();
	const nPointsStreamSize = head.u32();
	const flagStreamSize = head.u32();
	const glyphStreamSize = head.u32();
	const compositeStreamSize = head.u32();
	const bboxStreamSize = head.u32();
	const instructionStreamSize = head.u32();

	let p = head.pos;
	const slice = (n: number): Uint8Array => {
		const s = data.subarray(p, p + n);
		if (s.length !== n) throw new RangeError('woff2: glyf sub-stream truncated');
		p += n;
		return s;
	};
	const nContour = new Reader(slice(nContourStreamSize));
	const nPoints = new Reader(slice(nPointsStreamSize));
	const flagStream = new Reader(slice(flagStreamSize));
	const glyphStream = new Reader(slice(glyphStreamSize));
	const compositeStream = new Reader(slice(compositeStreamSize));
	const bboxStream = slice(bboxStreamSize);
	const instructionStream = new Reader(slice(instructionStreamSize));
	void optionFlags;
	void indexFormat;

	// The bbox stream opens with a bitmap: one bit per glyph, set when that
	// glyph has an explicit bbox rather than one derived from its points.
	const bitmapSize = ((numGlyphs + 31) >> 5) << 2;
	const bboxBitmap = bboxStream.subarray(0, bitmapSize);
	const bboxValues = new Reader(bboxStream.subarray(bitmapSize));
	const hasBbox = (i: number): boolean =>
		((bboxBitmap[i >> 3] >> (7 - (i & 7))) & 1) === 1;

	const out = new Writer(Math.max(1024, data.length * 2));
	const offsets: number[] = [0];

	for (let i = 0; i < numGlyphs; i++) {
		const nContours = nContour.i16();
		const glyphStart = out.pos;

		if (nContours === 0) {
			// Empty glyph — no outline at all, and loca records a zero-length
			// entry. Nothing is written.
			offsets.push(out.pos);
			continue;
		}

		if (nContours < 0) {
			// Composite: the component data is copied verbatim from the
			// composite stream, then instructions if the flag says so.
			out.i16(nContours);
			if (!hasBbox(i)) throw new RangeError(`woff2: composite glyph ${i} without bbox`);
			out.i16(bboxValues.i16());
			out.i16(bboxValues.i16());
			out.i16(bboxValues.i16());
			out.i16(bboxValues.i16());
			const start = compositeStream.pos;
			let haveInstructions = false;
			// Walk components to find where they end.
			for (;;) {
				const flags = compositeStream.u16();
				compositeStream.u16(); // glyphIndex
				const ARG_1_AND_2_ARE_WORDS = 0x0001;
				const WE_HAVE_A_SCALE = 0x0008;
				const MORE_COMPONENTS = 0x0020;
				const WE_HAVE_AN_X_AND_Y_SCALE = 0x0040;
				const WE_HAVE_A_TWO_BY_TWO = 0x0080;
				const WE_HAVE_INSTRUCTIONS = 0x0100;
				compositeStream.pos += flags & ARG_1_AND_2_ARE_WORDS ? 4 : 2;
				if (flags & WE_HAVE_A_SCALE) compositeStream.pos += 2;
				else if (flags & WE_HAVE_AN_X_AND_Y_SCALE) compositeStream.pos += 4;
				else if (flags & WE_HAVE_A_TWO_BY_TWO) compositeStream.pos += 8;
				if (flags & WE_HAVE_INSTRUCTIONS) haveInstructions = true;
				if (!(flags & MORE_COMPONENTS)) break;
			}
			out.raw(compositeStream.bytes.subarray(start, compositeStream.pos));
			if (haveInstructions) {
				const n = glyphStream.short255();
				out.u16(n);
				out.raw(instructionStream.bytes.subarray(
					instructionStream.pos, instructionStream.pos + n,
				));
				instructionStream.pos += n;
			}
			out.padTo4();
			offsets.push(out.pos);
			continue;
		}

		// Simple glyph.
		const endPts: number[] = [];
		let total = 0;
		for (let c = 0; c < nContours; c++) {
			total += nPoints.short255();
			endPts.push(total - 1);
		}
		const points = readTriplets(flagStream, glyphStream, total);

		out.i16(nContours);
		if (hasBbox(i)) {
			out.i16(bboxValues.i16());
			out.i16(bboxValues.i16());
			out.i16(bboxValues.i16());
			out.i16(bboxValues.i16());
		} else {
			// Derived from the points, per spec.
			let xMin = 0, yMin = 0, xMax = 0, yMax = 0;
			for (let k = 0; k < points.length; k++) {
				const { x, y } = points[k];
				if (k === 0) { xMin = xMax = x; yMin = yMax = y; continue; }
				if (x < xMin) xMin = x;
				if (x > xMax) xMax = x;
				if (y < yMin) yMin = y;
				if (y > yMax) yMax = y;
			}
			out.i16(xMin);
			out.i16(yMin);
			out.i16(xMax);
			out.i16(yMax);
		}
		for (const e of endPts) out.u16(e);

		const instrLen = glyphStream.short255();
		out.u16(instrLen);
		out.raw(instructionStream.bytes.subarray(
			instructionStream.pos, instructionStream.pos + instrLen,
		));
		instructionStream.pos += instrLen;

		writeSimpleGlyphPoints(out, points);
		out.padTo4();
		offsets.push(out.pos);
		void glyphStart;
	}

	return { glyf: out.result(), offsets };
}

interface Point {
	x: number;
	y: number;
	onCurve: boolean;
}

/**
 * Decode the triplet-encoded coordinate deltas.
 *
 * Each point is one flag byte (high bit = on-curve) plus 1-4 bytes of
 * coordinate data, where the flag's low 7 bits select one of 128 encodings
 * that fix how many bits each of dx/dy gets and what sign they carry. The
 * table below is the spec's, expressed arithmetically rather than as 128
 * literal rows.
 */
function readTriplets(flags: Reader, glyphs: Reader, count: number): Point[] {
	const points: Point[] = [];
	let x = 0;
	let y = 0;
	for (let i = 0; i < count; i++) {
		const flag = flags.u8();
		const onCurve = (flag & 0x80) === 0;
		const code = flag & 0x7f;
		let dx = 0;
		let dy = 0;
		if (code < 10) {
			// dx is 0; dy is 8 bits, sign from bit 0 of the code.
			dy = withSign(code & 1, ((code >> 1) << 8) + glyphs.u8());
		} else if (code < 20) {
			dx = withSign(code & 1, (((code - 10) >> 1) << 8) + glyphs.u8());
		} else if (code < 84) {
			// 64 codes = 4 dx bases x 4 dy bases x 2 signs x 2 signs, dx base
			// varying slowest. Bases are 1, 17, 33, 49 (1 + 16*index); one byte
			// carries dx in its high nibble and dy in its low nibble.
			const b0 = code - 20;
			const b1 = glyphs.u8();
			dx = withSign(xSignOf(b0), 1 + (((b0 >> 4) & 0x3) << 4) + (b1 >> 4));
			dy = withSign(ySignOf(b0), 1 + (((b0 >> 2) & 0x3) << 4) + (b1 & 0x0f));
		} else if (code < 120) {
			// 36 codes = 3 dx bases x 3 dy bases x 2 x 2, dx base slowest.
			// Bases are 1, 257, 513 (1 + 256*index), one full byte each.
			const b0 = code - 84;
			dx = withSign(xSignOf(b0), 1 + (Math.floor(b0 / 12) << 8) + glyphs.u8());
			dy = withSign(ySignOf(b0), 1 + ((Math.floor(b0 / 4) % 3) << 8) + glyphs.u8());
		} else if (code < 124) {
			// 12 bits each, packed across three bytes.
			const b0 = code - 120;
			const b1 = glyphs.u8();
			const b2 = glyphs.u8();
			dx = withSign(xSignOf(b0), (b1 << 4) + (b2 >> 4));
			dy = withSign(ySignOf(b0), ((b2 & 0x0f) << 8) + glyphs.u8());
		} else {
			const b0 = code - 124;
			dx = withSign(xSignOf(b0), glyphs.u16());
			dy = withSign(ySignOf(b0), glyphs.u16());
		}
		x += dx;
		y += dy;
		points.push({ x, y, onCurve });
	}
	return points;
}

/** WOFF2 stores sign as a bit where 1 means POSITIVE. */
function withSign(signBit: number, value: number): number {
	return signBit ? value : -value;
}

// In every code range that carries BOTH signs, the four consecutive codes for
// a given (dx base, dy base) pair run (x-,y-), (x+,y-), (x-,y+), (x+,y+) — so
// the X sign is the LOW bit and the Y sign the high one. Getting these the
// other way round decodes correct magnitudes with mirrored deltas, which is
// exactly the kind of bug that still produces plausible-looking outlines.
function xSignOf(n: number): number {
	return n & 1;
}
function ySignOf(n: number): number {
	return (n >> 1) & 1;
}

/** Write the flags + x/y delta arrays of a simple glyph, using the short and
 * same/positive encodings the SFNT format expects. Repeat-packing the flag
 * array is optional, so it is left unpacked for clarity. */
function writeSimpleGlyphPoints(out: Writer, points: Point[]): void {
	const X_SHORT = 0x02;
	const Y_SHORT = 0x04;
	const X_SAME_OR_POS = 0x10;
	const Y_SAME_OR_POS = 0x20;

	let prevX = 0;
	let prevY = 0;
	const flags: number[] = [];
	const xs: number[] = [];
	const ys: number[] = [];
	for (const pt of points) {
		let flag = pt.onCurve ? 0x01 : 0x00;
		const dx = pt.x - prevX;
		const dy = pt.y - prevY;
		if (dx === 0) {
			flag |= X_SAME_OR_POS;
		} else if (dx > -256 && dx < 256) {
			flag |= X_SHORT;
			if (dx > 0) flag |= X_SAME_OR_POS;
			xs.push(Math.abs(dx));
		} else {
			xs.push(dx);
		}
		if (dy === 0) {
			flag |= Y_SAME_OR_POS;
		} else if (dy > -256 && dy < 256) {
			flag |= Y_SHORT;
			if (dy > 0) flag |= Y_SAME_OR_POS;
			ys.push(Math.abs(dy));
		} else {
			ys.push(dy);
		}
		flags.push(flag);
		prevX = pt.x;
		prevY = pt.y;
	}
	for (const f of flags) out.u8(f);
	// The two arrays are written in the order their flags said, so re-walk
	// rather than trusting the push order alone.
	let xi = 0;
	for (let i = 0; i < points.length; i++) {
		const f = flags[i];
		if (f & X_SHORT) out.u8(xs[xi++]);
		else if (!(f & X_SAME_OR_POS)) out.i16(xs[xi++]);
	}
	let yi = 0;
	for (let i = 0; i < points.length; i++) {
		const f = flags[i];
		if (f & Y_SHORT) out.u8(ys[yi++]);
		else if (!(f & Y_SAME_OR_POS)) out.i16(ys[yi++]);
	}
}

/** Build a `loca` table from glyph offsets. Short format stores offset/2. */
function buildLoca(offsets: number[], indexToLocFormat: number): Uint8Array {
	const short = indexToLocFormat === 0;
	const w = new Writer(offsets.length * (short ? 2 : 4));
	for (const off of offsets) {
		if (short) w.u16(off >> 1);
		else w.u32(off);
	}
	return w.result();
}

/* ------------------------------------------------------------------ *
 * SFNT assembly
 * ------------------------------------------------------------------ */

function buildSfnt(
	flavor: number,
	tables: TableEntry[],
	totalSfntSize: number,
): ArrayBuffer {
	// The SFNT directory must be sorted by tag; WOFF2's is in its own order.
	const sorted = tables.slice().sort((a, b) => (a.tag < b.tag ? -1 : a.tag > b.tag ? 1 : 0));
	const numTables = sorted.length;
	const headerSize = 12 + numTables * 16;
	let bodySize = 0;
	for (const t of sorted) bodySize += (t.data as Uint8Array).length + 3 & ~3;

	const out = new Writer(Math.max(totalSfntSize, headerSize + bodySize));
	out.u32(flavor);
	out.u16(numTables);
	// searchRange / entrySelector / rangeShift: derived, and FreeType ignores
	// them, but a wrong value trips stricter validators.
	const maxPow2 = Math.floor(Math.log2(numTables));
	const searchRange = Math.pow(2, maxPow2) * 16;
	out.u16(searchRange);
	out.u16(maxPow2);
	out.u16(numTables * 16 - searchRange);

	let offset = headerSize;
	for (const t of sorted) {
		const data = t.data as Uint8Array;
		out.u32(tagToUint32(t.tag));
		out.u32(checksum(data));
		out.u32(offset);
		out.u32(data.length);
		offset += (data.length + 3) & ~3;
	}
	for (const t of sorted) {
		out.raw(t.data as Uint8Array);
		out.padTo4();
	}
	const bytes = out.result();
	return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length) as ArrayBuffer;
}

function tagToUint32(tag: string): number {
	return (
		((tag.charCodeAt(0) & 0xff) << 24) |
		((tag.charCodeAt(1) & 0xff) << 16) |
		((tag.charCodeAt(2) & 0xff) << 8) |
		(tag.charCodeAt(3) & 0xff)
	) >>> 0;
}

/** SFNT table checksum: sum of big-endian uint32s, zero-padded. */
function checksum(data: Uint8Array): number {
	let sum = 0;
	const full = data.length & ~3;
	for (let i = 0; i < full; i += 4) {
		sum = (sum + ((data[i] << 24) | (data[i + 1] << 16) | (data[i + 2] << 8) | data[i + 3])) >>> 0;
	}
	if (full < data.length) {
		let tail = 0;
		for (let i = 0; i < 4; i++) {
			tail = (tail << 8) | (full + i < data.length ? data[full + i] : 0);
		}
		sum = (sum + tail) >>> 0;
	}
	return sum >>> 0;
}
