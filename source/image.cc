#include "image.h"
#include "async.h"
#include "error.h"
#include "util.h"
#include "wrap.h"
#include <png.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <turbojpeg.h>
#include <webp/decode.h>

using namespace v8;

namespace {

struct buffer_state {
	uint8_t *ptr;
	size_t size;
};

void close_image(nx_image_t *image) {
	// Drop the memoized SkImage (built from a copy of the pixels). Ordering
	// vs. freeing `data` below doesn't strictly matter since the cache owns its
	// own copy, but release it up front to keep teardown tidy.
	nx_image_release_cache(image);
	if (image->data) {
		if (image->format == FORMAT_JPEG) {
			tjFree(image->data);
		} else {
			free(image->data);
		}
		image->data = NULL;
	}
	image->width = image->height = 0;
	image->is_yuv = false;
}

void user_read_data(png_structp png_ptr, png_bytep data, png_size_t length) {
	struct buffer_state *state =
	    (struct buffer_state *)png_get_io_ptr(png_ptr);
	// Bounds-check every read against the bytes remaining. A truncated or
	// malformed PNG makes libpng request more than the buffer holds; the old
	// unconditional memcpy read past the end of the input (OOB → segfault).
	// `state->size` tracks REMAINING bytes and is decremented as we consume.
	// On overrun, raise a libpng error, which longjmps to the setjmp in
	// decode_png (turning the crash into a clean decode failure / .onerror).
	if (!state || length > state->size) {
		png_error(png_ptr, "read past end of image buffer");
		return; // not reached — png_error longjmps
	}
	memcpy(data, state->ptr, length);
	state->ptr += length;
	state->size -= length;
}

enum ImageFormat identify_image_format(uint8_t *data, size_t size) {
	if (size >= 8 && !memcmp(data, "\211PNG\r\n\032\n", 8))
		return FORMAT_PNG;
	if (size >= 2 && !memcmp(data, "\377\330", 2))
		return FORMAT_JPEG;
	if (size >= 12 && !memcmp(data, "RIFF", 4) && !memcmp(data + 8, "WEBP", 4))
		return FORMAT_WEBP;
	// Ledger #89 — minimal SVG detection. Skip UTF-8 BOM, optional whitespace,
	// then look for `<?xml` (XML preamble) or `<svg` (bare root). Only handles
	// SVGs whose <rect> children paint rectangles with solid-color fills — the
	// exact pattern used by Khronos's red-green.svg. See decode_svg below for
	// scope and fallback behavior on complex SVGs.
	{
		size_t o = 0;
		if (size >= 3 && !memcmp(data, "\xEF\xBB\xBF", 3)) o = 3;
		while (o < size && (data[o] == ' ' || data[o] == '\t' ||
		                    data[o] == '\r' || data[o] == '\n')) o++;
		if (o + 5 <= size && !memcmp(data + o, "<?xml", 5)) return FORMAT_SVG;
		if (o + 4 <= size && !memcmp(data + o, "<svg", 4)) return FORMAT_SVG;
	}
	return FORMAT_UNKNOWN;
}

// Ledger #89 — targeted SVG parser for Khronos conformance's red-green.svg
// (WebGL 1 `textures-svg_image-tex-2d-*` cluster). Handles the exact pattern:
//   <svg xmlns=... width="N" height="M">
//     <rect [x="N"] [y="N"] width="M" height="M" fill="#RGB|#RRGGBB"/>
//     ...
//   </svg>
// Anything more complex (transforms, gradients, paths, opacity, non-hex
// colors, css) falls back to NULL, which the async decode path surfaces
// as "Image decode was not initialized" → `.onerror`. Since Khronos's
// red-green.svg is the only SVG asset the WebGL 1 corpus references, a
// full SVG rasterizer (SkSVG) would be over-engineering. If future assets
// need broader support, extend the parser incrementally or link SkSVG.
static const char *svg_find_char(const char *s, const char *end, char c) {
	while (s < end && *s != c) s++;
	return s;
}
static bool svg_find_attr(const char *tag_start, const char *tag_end,
                          const char *name, const char **out_val,
                          const char **out_val_end) {
	size_t nlen = strlen(name);
	const char *s = tag_start;
	while (s + nlen + 2 < tag_end) {
		// Match `<space>name=` at a whitespace boundary.
		if ((s[0] == ' ' || s[0] == '\t' || s[0] == '\r' || s[0] == '\n') &&
		    !memcmp(s + 1, name, nlen) && s[1 + nlen] == '=') {
			const char *q = s + 2 + nlen;
			if (q >= tag_end) return false;
			char qc = *q;
			if (qc != '"' && qc != '\'') return false;
			const char *e = svg_find_char(q + 1, tag_end, qc);
			if (e >= tag_end) return false;
			*out_val = q + 1;
			*out_val_end = e;
			return true;
		}
		s++;
	}
	return false;
}
static int svg_parse_int(const char *s, const char *end, int def) {
	int v = 0;
	bool any = false;
	while (s < end && *s >= '0' && *s <= '9') {
		v = v * 10 + (*s - '0');
		any = true;
		s++;
	}
	return any ? v : def;
}
static bool svg_parse_hex_color(const char *s, const char *end,
                                uint8_t *r, uint8_t *g, uint8_t *b) {
	if (s >= end || *s != '#') return false;
	s++;
	size_t len = (size_t)(end - s);
	auto hex = [](char c, uint8_t *out) {
		if (c >= '0' && c <= '9') { *out = c - '0'; return true; }
		if (c >= 'a' && c <= 'f') { *out = c - 'a' + 10; return true; }
		if (c >= 'A' && c <= 'F') { *out = c - 'A' + 10; return true; }
		return false;
	};
	if (len == 3) {
		uint8_t hr, hg, hb;
		if (!hex(s[0], &hr) || !hex(s[1], &hg) || !hex(s[2], &hb)) return false;
		*r = (hr << 4) | hr;
		*g = (hg << 4) | hg;
		*b = (hb << 4) | hb;
		return true;
	}
	if (len == 6) {
		uint8_t h0, h1, h2, h3, h4, h5;
		if (!hex(s[0], &h0) || !hex(s[1], &h1) || !hex(s[2], &h2) ||
		    !hex(s[3], &h3) || !hex(s[4], &h4) || !hex(s[5], &h5)) return false;
		*r = (h0 << 4) | h1;
		*g = (h2 << 4) | h3;
		*b = (h4 << 4) | h5;
		return true;
	}
	return false;
}
uint8_t *decode_svg(uint8_t *input, size_t input_size, uint32_t *width,
                    uint32_t *height) {
	const char *src = (const char *)input;
	const char *src_end = src + input_size;
	// Find `<svg` opening tag (skip past optional <?xml?> preamble +
	// <!DOCTYPE ...> preamble).
	const char *svg_open = nullptr;
	for (const char *s = src; s + 4 <= src_end; s++) {
		if (s[0] == '<' && s[1] == 's' && s[2] == 'v' && s[3] == 'g' &&
		    (s + 4 == src_end || s[4] == ' ' || s[4] == '\t' ||
		     s[4] == '\r' || s[4] == '\n' || s[4] == '>')) {
			svg_open = s;
			break;
		}
	}
	if (!svg_open) return nullptr;
	const char *svg_tag_end = svg_find_char(svg_open, src_end, '>');
	if (svg_tag_end >= src_end) return nullptr;
	const char *wv, *wve, *hv, *hve;
	if (!svg_find_attr(svg_open, svg_tag_end, "width", &wv, &wve)) return nullptr;
	if (!svg_find_attr(svg_open, svg_tag_end, "height", &hv, &hve)) return nullptr;
	int w = svg_parse_int(wv, wve, 0);
	int h = svg_parse_int(hv, hve, 0);
	if (w <= 0 || h <= 0 || w > 16384 || h > 16384) return nullptr;
	*width = (uint32_t)w;
	*height = (uint32_t)h;
	size_t buf_size = (size_t)w * (size_t)h * 4;
	uint8_t *out = (uint8_t *)calloc(1, buf_size);
	if (!out) return nullptr;
	// Iterate <rect ...> children. Each fills a solid color into the
	// specified rectangle. Storage is premultiplied BGRA (Skia's kPremul).
	const char *cur = svg_tag_end + 1;
	while (cur + 5 < src_end) {
		const char *rect_open = nullptr;
		for (const char *s = cur; s + 5 <= src_end; s++) {
			if (s[0] == '<' && s[1] == 'r' && s[2] == 'e' && s[3] == 'c' &&
			    s[4] == 't' && (s + 5 == src_end || s[5] == ' ' ||
			                    s[5] == '\t' || s[5] == '\r' ||
			                    s[5] == '\n')) {
				rect_open = s;
				break;
			}
		}
		if (!rect_open) break;
		const char *rect_end = svg_find_char(rect_open, src_end, '>');
		if (rect_end >= src_end) break;
		const char *xv = nullptr, *xve = nullptr;
		const char *yv = nullptr, *yve = nullptr;
		const char *rwv = nullptr, *rwve = nullptr;
		const char *rhv = nullptr, *rhve = nullptr;
		const char *fv = nullptr, *fve = nullptr;
		bool has_x = svg_find_attr(rect_open, rect_end, "x", &xv, &xve);
		bool has_y = svg_find_attr(rect_open, rect_end, "y", &yv, &yve);
		bool has_w = svg_find_attr(rect_open, rect_end, "width", &rwv, &rwve);
		bool has_h = svg_find_attr(rect_open, rect_end, "height", &rhv, &rhve);
		bool has_fill =
		    svg_find_attr(rect_open, rect_end, "fill", &fv, &fve);
		int rx = has_x ? svg_parse_int(xv, xve, 0) : 0;
		int ry = has_y ? svg_parse_int(yv, yve, 0) : 0;
		int rw = has_w ? svg_parse_int(rwv, rwve, 0) : w;
		int rh = has_h ? svg_parse_int(rhv, rhve, 0) : h;
		uint8_t rr = 0, rg = 0, rb = 0;
		if (has_fill) {
			if (!svg_parse_hex_color(fv, fve, &rr, &rg, &rb)) {
				// Unrecognized fill (color name, none, url(...)) — skip this
				// rect rather than aborting; may still produce a usable
				// image if other rects paint the target pixels.
				cur = rect_end + 1;
				continue;
			}
		}
		if (rx < 0) { rw += rx; rx = 0; }
		if (ry < 0) { rh += ry; ry = 0; }
		if (rx + rw > w) rw = w - rx;
		if (ry + rh > h) rh = h - ry;
		if (rw <= 0 || rh <= 0) {
			cur = rect_end + 1;
			continue;
		}
		for (int y = ry; y < ry + rh; y++) {
			uint8_t *row = out + ((size_t)y * (size_t)w + (size_t)rx) * 4;
			for (int x = 0; x < rw; x++) {
				// Alpha = 255 (opaque). Storage is premul BGRA — for
				// alpha=255 premul == non-premul.
				row[x * 4 + 0] = rb;
				row[x * 4 + 1] = rg;
				row[x * 4 + 2] = rr;
				row[x * 4 + 3] = 0xFF;
			}
		}
		cur = rect_end + 1;
	}
	return out;
}

void premultiply_alpha(uint8_t *image_data, int width, int height) {
	for (int i = 0; i < width * height; ++i) {
		uint8_t *pixel = &image_data[i * 4];
		uint8_t alpha = pixel[3];
		pixel[0] = (pixel[0] * alpha) / 255;
		pixel[1] = (pixel[1] * alpha) / 255;
		pixel[2] = (pixel[2] * alpha) / 255;
	}
}

uint8_t *decode_png(uint8_t *input, size_t input_size, u32 *width,
                    u32 *height) {
	png_structp png_ptr =
	    png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr)
		return NULL;
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_read_struct(&png_ptr, NULL, NULL);
		return NULL;
	}
	// libpng reports EVERY decode failure (bad header, truncated IDAT, CRC
	// mismatch, unsupported feature, ...) by longjmp-ing back to a setjmp
	// point. Without one, its default handler longjmps through an
	// uninitialized jmp_buf and segfaults this worker thread with no catchable
	// JS error. Establish the target and treat any error as a clean decode
	// failure: return NULL, which the caller maps to an `error` event on the
	// Image (nx_decode_image_do -> "Image decode was not initialized").
	// `image_data` / `rows` are read in the longjmp handler, so they are
	// `volatile` (their post-longjmp value must be their last stored value).
	uint8_t *volatile image_data = NULL;
	png_bytep *volatile rows = NULL;
	if (setjmp(png_jmpbuf(png_ptr))) {
		free((void *)image_data);
		free((void *)rows);
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return NULL;
	}
	struct buffer_state state = {input, input_size};
	png_set_read_fn(png_ptr, &state, user_read_data);
	png_read_info(png_ptr, info_ptr);
	*width = png_get_image_width(png_ptr, info_ptr);
	*height = png_get_image_height(png_ptr, info_ptr);
	// Capture what the FILE says BEFORE any transform: this is what decides
	// whether any pixel can have alpha < 255. After the transforms below every
	// image is nominally RGBA, so asking afterwards answers a different
	// question.
	//
	// THE BUG (2026-09-12, fixed here): this used to be
	//     has_alpha = png_get_color_type(...) == PNG_COLOR_TYPE_RGBA;
	// which is false for three cases that DO carry transparency - PALETTE with
	// tRNS, RGB with tRNS, and GREY_ALPHA - so `premultiply_alpha()` below was
	// skipped for them. Everything downstream treats nx_image_t as
	// premultiplied BGRA (w_tex_image_2d hardcodes un_premultiply = false), so
	// those images reached the GPU as STRAIGHT alpha while being blended as
	// premultiplied.
	//
	// How it surfaced: the pixi-filters demo's `overlay.png` is a palette PNG
	// whose transparent texels decode to (255,255,255,0). PIXI draws it as a
	// full-screen TilingSprite with src=ONE, dst=ONE_MINUS_SRC_ALPHA, so every
	// such texel computes 1 + dst*(1-0) - it ADDS full white and subtracts
	// nothing, and the whole frame saturated to white. A premultiplied texel
	// can never have a channel above its alpha, which is precisely the
	// invariant those samples violated (rgb=255 > a=0).
	const png_byte file_color_type = png_get_color_type(png_ptr, info_ptr);
	const bool file_has_trns =
	    png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0;
	const bool has_alpha =
	    (file_color_type & PNG_COLOR_MASK_ALPHA) != 0 || file_has_trns;

	// Normalise every input to 8-bit BGRA before reading rows: expand
	// palette / tRNS / sub-8-bit, drop 16-bit channels (the buffer below is 4
	// bytes per pixel - a 16-bit PNG would otherwise need 8 and overflow it),
	// promote grey to RGB, and guarantee an alpha channel.
	png_set_bgr(png_ptr);
	png_set_expand(png_ptr);
	png_set_strip_16(png_ptr);
	png_set_gray_to_rgb(png_ptr);
	png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
	// Make info_ptr describe the POST-transform rows so the stride check below
	// is meaningful.
	png_read_update_info(png_ptr, info_ptr);
	if (*width == 0 || *height == 0 || *width > 16384 || *height > 16384 ||
	    (size_t)(*width) > SIZE_MAX / 4 / (*height)) {
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return NULL;
	}
	// The row pointers below hand libpng a 4-bytes-per-pixel buffer; refuse
	// anything whose post-transform stride disagrees rather than overflowing it.
	if (png_get_rowbytes(png_ptr, info_ptr) != (png_size_t)(4 * (size_t)(*width))) {
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return NULL;
	}
	image_data = (uint8_t *)malloc(4 * (size_t)(*width) * (*height));
	rows = (png_bytep *)malloc(sizeof(png_bytep) * (*height));
	if (!image_data || !rows) {
		free((void *)image_data);
		free((void *)rows);
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return NULL;
	}
	for (u32 i = 0; i < *height; ++i)
		rows[i] = image_data + i * 4 * (*width);
	png_read_image(png_ptr, (png_bytepp)rows);
	free((void *)rows);
	rows = NULL;
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	if (has_alpha)
		premultiply_alpha((uint8_t *)image_data, *width, *height);
	return (uint8_t *)image_data;
}

uint8_t *decode_webp(uint8_t *webp_data, size_t data_size, int *width,
                     int *height) {
	uint8_t *bgra_data = WebPDecodeBGRA(webp_data, data_size, width, height);
	if (bgra_data == NULL)
		return NULL;
	premultiply_alpha(bgra_data, *width, *height);
	return bgra_data;
}

// ---- async decode ----
typedef struct {
	int err;
	const char *err_str;
	nx_image_t *image;
	Global<Value> image_val;
	Global<Value> buffer_val;
	uint8_t *input;
	size_t input_size;
} decode_image_t;

void nx_decode_image_do(nx_work_t *req) {
	decode_image_t *data = (decode_image_t *)req->data;
	data->image->format = identify_image_format(data->input, data->input_size);
	if (data->image->format == FORMAT_PNG) {
		data->image->data = decode_png(data->input, data->input_size,
		                               &data->image->width,
		                               &data->image->height);
	} else if (data->image->format == FORMAT_JPEG) {
		if (decode_jpeg(data->input, data->input_size, &data->image->data,
		                (int *)&data->image->width,
		                (int *)&data->image->height)) {
			data->err_str = tjGetErrorStr();
			return;
		}
	} else if (data->image->format == FORMAT_WEBP) {
		data->image->data = decode_webp(data->input, data->input_size,
		                                (int *)&data->image->width,
		                                (int *)&data->image->height);
	} else if (data->image->format == FORMAT_SVG) {
		data->image->data = decode_svg(data->input, data->input_size,
		                               &data->image->width,
		                               &data->image->height);
	} else {
		data->err_str = "Unsupported image format";
		return;
	}
	if (data->image->data == NULL) {
		data->err_str = "Image decode was not initialized";
		return;
	}
	// `data->image->data` holds premultiplied BGRA; canvas.cc wraps it in an
	// SkImage at draw time. No persistent surface object needed.
}

MaybeLocal<Value> nx_decode_image_cb(Isolate *iso, nx_work_t *req) {
	decode_image_t *data = (decode_image_t *)req->data;
	data->image_val.Reset();
	data->buffer_val.Reset();
	if (data->err_str) {
		iso->ThrowException(Exception::Error(nx_str(iso, data->err_str)));
		return MaybeLocal<Value>();
	}
	return Undefined(iso).As<Value>();
}

void nx_image_decode(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_image_t *image = nx_get_image(iso, info[0]);
	if (!image)
		return;
	size_t size = 0;
	uint8_t *buf = NX_GetBufferSource(iso, &size, info[1]);
	if (!buf) {
		nx_throw(iso, "expected ArrayBuffer");
		return;
	}
	nx_work_t *req = new nx_work_t();
	decode_image_t *data = new decode_image_t();
	req->data = data;
	req->data_dtor = [](void *p) { delete static_cast<decode_image_t *>(p); };
	data->image = image;
	data->image_val.Reset(iso, info[0]);
	data->buffer_val.Reset(iso, info[1]);
	data->input = buf;
	data->input_size = size;
	info.GetReturnValue().Set(
	    nx_queue_async(iso, req, nx_decode_image_do, nx_decode_image_cb));
}

void free_image(nx_image_t *image) {
	close_image(image);
	free(image);
}

void nx_image_new(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Context> context = iso->GetCurrentContext();
	Local<Object> img = nx::NewWrapped(iso);
	nx_image_t *data = (nx_image_t *)calloc(1, sizeof(nx_image_t));
	data->magic = NX_IMAGE_MAGIC;
	nx::Wrap<nx_image_t>(iso, img, data, free_image);
	if (info.Length() == 2) {
		uint32_t w = 0, h = 0;
		if (!info[0]->Uint32Value(context).To(&w) ||
		    !info[1]->Uint32Value(context).To(&h))
			return;
		data->width = w;
		data->height = h;
		if (w == 0 || h == 0 || w > SIZE_MAX / 4 ||
		    (size_t)h > SIZE_MAX / ((size_t)w * 4)) {
			iso->ThrowException(
			    Exception::RangeError(nx_str(iso, "Image dimensions too large")));
			return;
		}
		data->data = (uint8_t *)calloc(1, (size_t)w * h * 4);
	}
	info.GetReturnValue().Set(img);
}

void nx_image_close(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_image_t *image = nx_get_image(iso, info[0]);
	if (image)
		close_image(image);
}

// Copy `width*height*4` RGBA bytes from `argv[1]` into the image's backing
// buffer with the RGBA→BGRA-premultiplied swizzle Skia expects. Used by
// brewser's `<video>` frame delivery to feed decoded FFmpeg RGBA frames into
// an ImageBitmap that can be drawn via the standard drawImage(img, ...)
// path — which scales via Skia (unlike `putImageData` on an OffscreenCanvas
// where a raw data-buffer write can't invalidate the Ganesh texture cache
// and every frame after the first would paint from the stale upload).
// The image must have been constructed via `imageNew(w, h)` so it has a
// backing buffer allocated up-front.
void nx_image_write_rgba(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	if (info.Length() < 2) {
		nx_throw(iso, "imageWriteRGBA: expected (image, bytes, [premultiply])");
		return;
	}
	nx_image_t *image = nx_get_image(iso, info[0]);
	if (!image) {
		nx_throw(iso, "imageWriteRGBA: first arg must be an Image");
		return;
	}
	if (!image->data || image->width == 0 || image->height == 0) {
		nx_throw(iso,
		         "imageWriteRGBA: image has no backing buffer — construct via "
		         "imageNew(width, height)");
		return;
	}
	size_t src_size = 0;
	uint8_t *src = NX_GetBufferSource(iso, &src_size, info[1]);
	if (!src) {
		nx_throw(iso, "imageWriteRGBA: second arg must be ArrayBuffer or "
		              "TypedArray");
		return;
	}
	size_t expected = (size_t)image->width * (size_t)image->height * 4;
	if (src_size < expected) {
		nx_throw(iso, "imageWriteRGBA: buffer smaller than expected");
		return;
	}
	// Ledger #73 — optional 3rd arg `premultiply` (default true) controls
	// whether the RGB channels are multiplied by alpha before storage.
	// createImageBitmap({premultiplyAlpha: "none"}) needs the bitmap
	// stored with UN-premultiplied pixels so texImage2D can hand them
	// straight through to the driver (matches the Chrome behavior our
	// #72 hardcoded "no un-multiply" bakes into the WebGL upload path).
	// Default true preserves the existing video-frame delivery contract
	// used by Switch.VideoDecoder (bitmap.ts::imageWriteRGBA).
	bool premultiply = true;
	if (info.Length() >= 3 && info[2]->IsBoolean()) {
		premultiply = info[2]->BooleanValue(iso);
	}
	// Ledger #78 — record premul state on the image so downstream
	// createImageBitmap(imageBitmap, opts) can convert correctly instead
	// of losing alpha=0 pixels' RGB channels through a canvas round-trip.
	image->unpremultiplied = !premultiply;
	uint8_t *dst = image->data;
	size_t pixels = (size_t)image->width * (size_t)image->height;
	for (size_t i = 0; i < pixels; i++) {
		uint8_t r = src[i * 4 + 0];
		uint8_t g = src[i * 4 + 1];
		uint8_t b = src[i * 4 + 2];
		uint8_t a = src[i * 4 + 3];
		if (!premultiply) {
			// Raw store: keep RGBA values as-is, just swap R↔B for BGRA order.
			dst[i * 4 + 0] = b;
			dst[i * 4 + 1] = g;
			dst[i * 4 + 2] = r;
			dst[i * 4 + 3] = a;
		} else if (a == 0) {
			dst[i * 4 + 0] = 0;
			dst[i * 4 + 1] = 0;
			dst[i * 4 + 2] = 0;
			dst[i * 4 + 3] = 0;
		} else if (a == 255) {
			dst[i * 4 + 0] = b;
			dst[i * 4 + 1] = g;
			dst[i * 4 + 2] = r;
			dst[i * 4 + 3] = a;
		} else {
			dst[i * 4 + 0] = (uint8_t)((b * a) / 255);
			dst[i * 4 + 1] = (uint8_t)((g * a) / 255);
			dst[i * 4 + 2] = (uint8_t)((r * a) / 255);
			dst[i * 4 + 3] = a;
		}
	}
	// Invalidate the memoized SkImage so canvas.cc's drawImage path rebuilds
	// it (RasterFromPixmapCopy) with the fresh pixels. Without this, Ganesh's
	// texture cache keys on the SkImage identity and every subsequent draw
	// re-uses the first frame's texture upload.
	nx_image_release_cache(image);
}

// imageWriteBGRA(image, bytes) — zero-swizzle counterpart to imageWriteRGBA
// for decoded video frames (2026-09-05 Fix D). FFmpeg's sws_scale already
// produces BGRA (Skia's ARGB32 memory order) and video frames are fully
// opaque (alpha 255, so premultiplied == straight), so this is a plain
// memcpy — no R↔B swap and no premultiply pass. Feeding Switch.VideoDecoder's
// BGRA frames through here instead of imageWriteRGBA removes the
// BGRA→RGBA→BGRA double swizzle (two per-pixel byte loops per frame on the
// main thread) that the RGBA path incurred. The image must have been
// constructed via imageNew(w, h).
void nx_image_write_bgra(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	if (info.Length() < 2) {
		nx_throw(iso, "imageWriteBGRA: expected (image, bytes)");
		return;
	}
	nx_image_t *image = nx_get_image(iso, info[0]);
	if (!image) {
		nx_throw(iso, "imageWriteBGRA: first arg must be an Image");
		return;
	}
	if (!image->data || image->width == 0 || image->height == 0) {
		nx_throw(iso,
		         "imageWriteBGRA: image has no backing buffer — construct via "
		         "imageNew(width, height)");
		return;
	}
	size_t src_size = 0;
	uint8_t *src = NX_GetBufferSource(iso, &src_size, info[1]);
	if (!src) {
		nx_throw(iso, "imageWriteBGRA: second arg must be ArrayBuffer or "
		              "TypedArray");
		return;
	}
	size_t expected = (size_t)image->width * (size_t)image->height * 4;
	if (src_size < expected) {
		nx_throw(iso, "imageWriteBGRA: buffer smaller than expected");
		return;
	}
	image->unpremultiplied = false; // opaque BGRA == premultiplied BGRA
	memcpy(image->data, src, expected);
	// Same cache invalidation rationale as imageWriteRGBA above.
	nx_image_release_cache(image);
}

// imageWriteYUV(image, bytes, width, height, [colorSpace]) — store a planar
// I420 video frame (Y|U|V contiguous, 1.5 B/px) so canvas.cc's drawImage
// builds a GPU YUVA SkImage (Skia does YUV→RGB in the shader), uploading ~2.6×
// less per frame than the BGRA path. The image must have been constructed via
// imageNew(width, height) — its width*height*4 alloc comfortably holds the
// 1.5*width*height I420, so no reallocation is needed even on reuse. Marks the
// image `is_yuv`; the next imageWriteBGRA/RGBA would clear it back to RGBA.
void nx_image_write_yuv(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	if (info.Length() < 4) {
		nx_throw(iso, "imageWriteYUV: expected (image, bytes, width, height, "
		              "[colorSpace])");
		return;
	}
	nx_image_t *image = nx_get_image(iso, info[0]);
	if (!image) {
		nx_throw(iso, "imageWriteYUV: first arg must be an Image");
		return;
	}
	Local<Context> c = iso->GetCurrentContext();
	uint32_t w = 0, h = 0;
	if (!info[2]->Uint32Value(c).To(&w) || !info[3]->Uint32Value(c).To(&h))
		return;
	if (!image->data || image->width != w || image->height != h) {
		nx_throw(iso, "imageWriteYUV: image must be imageNew(width, height) at "
		              "these dimensions");
		return;
	}
	size_t src_size = 0;
	uint8_t *src = NX_GetBufferSource(iso, &src_size, info[1]);
	if (!src) {
		nx_throw(iso,
		         "imageWriteYUV: second arg must be ArrayBuffer or TypedArray");
		return;
	}
	const size_t cw = (w + 1) / 2, ch = (h + 1) / 2;
	const size_t expected = (size_t)w * h + 2 * cw * ch;
	if (src_size < expected) {
		nx_throw(iso, "imageWriteYUV: buffer smaller than expected");
		return;
	}
	memcpy(image->data, src, expected);
	image->is_yuv = true;
	image->unpremultiplied = false;
	if (info.Length() >= 5) {
		int32_t cs = 0;
		if (info[4]->Int32Value(c).To(&cs))
			image->yuv_colorspace = cs;
	}
	nx_image_release_cache(image);
}

// Ledger #78 — imageCopyPixels(dst, src, dstPremultiply, flipY). Copies
// src.data (BGRA) to dst.data (BGRA) with premul-state conversion and
// optional Y-flip. Bypasses the canvas 2D round-trip that
// createImageBitmap's ImageBitmap / HTMLImageElement source branches
// previously used — that round-trip zeroed the RGB channels of any
// alpha=0 pixel because canvas 2D storage is premul (and premul
// (r, g, b, 0) = (0, 0, 0, 0)). This helper reads src.unpremultiplied
// to decide the conversion:
//   - same premul state: byte copy (fast path, preserves alpha=0 RGB).
//   - src unpremul → dst premul: R,G,B *= alpha/255 (rounding).
//   - src premul → dst unpremul: R,G,B *= 255/alpha (undefined at
//     alpha=0; spec-per-Canvas-2D returns 0 in that case, matching the
//     existing WebGL upload path).
// dst.unpremultiplied is set to !dstPremultiply; dst dims must match
// src dims (caller allocates via imageNew(w, h) beforehand).
void nx_image_copy_pixels(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	if (info.Length() < 3) {
		nx_throw(iso, "imageCopyPixels: expected (dst, src, dstPremultiply, "
		              "[flipY])");
		return;
	}
	nx_image_t *dst = nx_get_image(iso, info[0]);
	nx_image_t *src = nx_get_image(iso, info[1]);
	if (!dst || !src) {
		nx_throw(iso, "imageCopyPixels: both args must be Images");
		return;
	}
	if (!dst->data || !src->data ||
	    dst->width != src->width || dst->height != src->height) {
		nx_throw(iso,
		         "imageCopyPixels: dst must be allocated at src dimensions");
		return;
	}
	const bool dst_premultiply = info[2]->BooleanValue(iso);
	const bool flip_y = info.Length() >= 4 && info[3]->BooleanValue(iso);
	const bool src_unpremul = src->unpremultiplied;
	const bool same_state = src_unpremul == !dst_premultiply;
	const uint32_t W = src->width;
	const uint32_t H = src->height;
	for (uint32_t y = 0; y < H; y++) {
		const uint32_t src_y = flip_y ? (H - 1 - y) : y;
		const uint8_t *src_row = src->data + (size_t)src_y * (size_t)W * 4;
		uint8_t *dst_row = dst->data + (size_t)y * (size_t)W * 4;
		if (same_state) {
			memcpy(dst_row, src_row, (size_t)W * 4);
			continue;
		}
		for (uint32_t x = 0; x < W; x++) {
			uint8_t b = src_row[x * 4 + 0];
			uint8_t g = src_row[x * 4 + 1];
			uint8_t r = src_row[x * 4 + 2];
			uint8_t a = src_row[x * 4 + 3];
			if (src_unpremul && !dst_premultiply) {
				// Same state — handled by same_state above; unreachable.
			} else if (src_unpremul && dst_premultiply) {
				// unpremul → premul.
				b = (uint8_t)(((int)b * (int)a + 127) / 255);
				g = (uint8_t)(((int)g * (int)a + 127) / 255);
				r = (uint8_t)(((int)r * (int)a + 127) / 255);
			} else if (!src_unpremul && !dst_premultiply) {
				// premul → unpremul. Divide by alpha; alpha=0 leaves
				// channels at 0 (matches canvas 2D getImageData behavior).
				if (a > 0 && a < 255) {
					int nb = ((int)b * 255 + a / 2) / a;
					int ng = ((int)g * 255 + a / 2) / a;
					int nr = ((int)r * 255 + a / 2) / a;
					b = (uint8_t)(nb > 255 ? 255 : nb);
					g = (uint8_t)(ng > 255 ? 255 : ng);
					r = (uint8_t)(nr > 255 ? 255 : nr);
				}
			}
			dst_row[x * 4 + 0] = b;
			dst_row[x * 4 + 1] = g;
			dst_row[x * 4 + 2] = r;
			dst_row[x * 4 + 3] = a;
		}
	}
	dst->unpremultiplied = !dst_premultiply;
	nx_image_release_cache(dst);
}

void nx_image_get_width(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_image_t *image = nx_get_image(iso, info.This());
	if (image)
		info.GetReturnValue().Set(Integer::NewFromUnsigned(iso, image->width));
}

void nx_image_get_height(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	nx_image_t *image = nx_get_image(iso, info.This());
	if (image)
		info.GetReturnValue().Set(Integer::NewFromUnsigned(iso, image->height));
}

void nx_image_init_class(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	Local<Context> context = iso->GetCurrentContext();
	Local<Object> proto = info[0]
	                          .As<Object>()
	                          ->Get(context, nx_str(iso, "prototype"))
	                          .ToLocalChecked()
	                          .As<Object>();
	NX_DEF_GET(proto, "width", nx_image_get_width);
	NX_DEF_GET(proto, "height", nx_image_get_height);
}

} // namespace

// ---- shared, non-namespaced symbols ----

nx_image_t *nx_get_image(Isolate *iso, Local<Value> obj) {
	(void)iso;
	nx_image_t *image = nx::Unwrap<nx_image_t>(obj);
	// Validate the type discriminator: a wrapped Canvas unwrapped as an
	// image would alias `surface_dirty`/`gpu` with `cached_sk_image`,
	// turning a bool into a dereferenced pointer (crash). See image.h.
	if (image && image->magic != NX_IMAGE_MAGIC)
		return nullptr;
	return image;
}

int decode_jpeg(uint8_t *jpegBuf, size_t jpegSize, uint8_t **output, int *width,
                int *height) {
	tjhandle handle = NULL;
	int subsamp, colorspace;
	int ret = -1;
	handle = tjInitDecompress();
	if (handle == NULL)
		goto cleanup;
	if (tjDecompressHeader3(handle, jpegBuf, jpegSize, width, height, &subsamp,
	                        &colorspace) == -1)
		goto cleanup;
	*output = tjAlloc((*width) * (*height) * tjPixelSize[TJPF_BGRA]);
	if (tjDecompress2(handle, jpegBuf, jpegSize, *output, *width, 0, *height,
	                  TJPF_BGRA, TJFLAG_FASTDCT) == -1) {
		tjFree(*output);
		*output = NULL;
		goto cleanup;
	}
	ret = 0;
cleanup:
	if (handle != NULL)
		tjDestroy(handle);
	return ret;
}

void nx_init_image(Isolate *iso, Local<Object> init_obj) {
	NX_SET_FUNC(init_obj, "imageInit", nx_image_init_class);
	NX_SET_FUNC(init_obj, "imageNew", nx_image_new);
	NX_SET_FUNC(init_obj, "imageDecode", nx_image_decode);
	NX_SET_FUNC(init_obj, "imageClose", nx_image_close);
	NX_SET_FUNC(init_obj, "imageWriteRGBA", nx_image_write_rgba);
	NX_SET_FUNC(init_obj, "imageWriteBGRA", nx_image_write_bgra);
	NX_SET_FUNC(init_obj, "imageWriteYUV", nx_image_write_yuv);
	NX_SET_FUNC(init_obj, "imageCopyPixels", nx_image_copy_pixels);
}
