// Brotli decompression, exposed to the TS runtime as `$.brotliDecompress`.
//
// Motivation: WOFF2 — which is what essentially every modern web font ships as
// (Google Fonts, any Vite/webpack build) — is a brotli-compressed SFNT.
// devkitPro ships no brotli portlib, and this FreeType is built WITHOUT
// `FT_CONFIG_OPTION_USE_BROTLI`, so a `.woff2` could not be decoded by any
// path: the FontFace constructor just reported "failed to load font face".
// `source/brotli/` is a vendored copy of the upstream decoder; the WOFF2
// container/table reconstruction on top of it lives in TS
// (`packages/runtime/src/font/woff2.ts`).
//
// Also the primitive an HTTP `Content-Encoding: br` path or a
// `DecompressionStream('br')` would need later.

#include "brotli_bind.h"
#include "error.h"
#include "types.h"
#include "util.h"
#include <brotli/decode.h>
#include <stdlib.h>
#include <string.h>

using namespace v8;

namespace {

// Brotli gives no cheap way to know the decompressed size up front, so grow a
// buffer around the streaming API. Fonts are the target here (tens to a few
// hundred KB), and an over-eager initial guess wastes more than it saves.
constexpr size_t kInitialFactor = 4;   // decompressed >= 4x compressed, usually
constexpr size_t kMinCapacity = 64 * 1024;
constexpr size_t kMaxCapacity = 64u * 1024u * 1024u;

void nx_brotli_decompress(const FunctionCallbackInfo<Value> &info) {
	Isolate *iso = info.GetIsolate();
	size_t in_size = 0;
	uint8_t *in = NX_GetBufferSource(iso, &in_size, info[0]);
	if (!in) {
		nx_throw(iso, "brotliDecompress: expected ArrayBuffer or TypedArray");
		return;
	}
	if (in_size == 0) {
		info.GetReturnValue().Set(ArrayBuffer::New(iso, 0));
		return;
	}

	size_t capacity = in_size * kInitialFactor;
	if (capacity < kMinCapacity)
		capacity = kMinCapacity;
	if (capacity > kMaxCapacity)
		capacity = kMaxCapacity;
	uint8_t *out = (uint8_t *)malloc(capacity);
	if (!out) {
		nx_throw_oom(iso, capacity);
		return;
	}

	BrotliDecoderState *state =
	    BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
	if (!state) {
		free(out);
		nx_throw(iso, "brotliDecompress: could not create decoder");
		return;
	}

	const uint8_t *next_in = in;
	size_t avail_in = in_size;
	size_t total_out = 0;
	BrotliDecoderResult result = BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT;

	while (true) {
		uint8_t *next_out = out + total_out;
		size_t avail_out = capacity - total_out;
		result = BrotliDecoderDecompressStream(state, &avail_in, &next_in,
		                                       &avail_out, &next_out, nullptr);
		total_out = capacity - avail_out;
		if (result == BROTLI_DECODER_RESULT_SUCCESS)
			break;
		if (result == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
			if (capacity >= kMaxCapacity) {
				BrotliDecoderDestroyInstance(state);
				free(out);
				nx_throw(iso, "brotliDecompress: output exceeds 64 MiB limit");
				return;
			}
			size_t next_capacity = capacity * 2;
			if (next_capacity > kMaxCapacity)
				next_capacity = kMaxCapacity;
			uint8_t *grown = (uint8_t *)realloc(out, next_capacity);
			if (!grown) {
				BrotliDecoderDestroyInstance(state);
				free(out);
				nx_throw_oom(iso, next_capacity);
				return;
			}
			out = grown;
			capacity = next_capacity;
			continue;
		}
		// NEEDS_MORE_INPUT with nothing left to give means a truncated
		// stream; ERROR means a corrupt one. Both are "not decodable".
		{
			const char *why = BrotliDecoderErrorString(
			    BrotliDecoderGetErrorCode(state));
			BrotliDecoderDestroyInstance(state);
			free(out);
			char msg[160];
			snprintf(msg, sizeof(msg), "brotliDecompress: %s",
			         why ? why : "malformed stream");
			nx_throw(iso, msg);
			return;
		}
	}
	BrotliDecoderDestroyInstance(state);

	// Hand the exact-size buffer to V8, which takes ownership via the
	// backing store's deleter. `realloc` down is a no-op when it already
	// fits; a failure to shrink is harmless (keep the larger block).
	uint8_t *shrunk = (uint8_t *)realloc(out, total_out ? total_out : 1);
	if (shrunk)
		out = shrunk;
	std::unique_ptr<BackingStore> bs = ArrayBuffer::NewBackingStore(
	    out, total_out, [](void *p, size_t, void *) { free(p); }, nullptr);
	info.GetReturnValue().Set(ArrayBuffer::New(iso, std::move(bs)));
}

} // namespace

void nx_init_brotli(Isolate *iso, Local<Object> init_obj) {
	NX_SET_FUNC(init_obj, "brotliDecompress", nx_brotli_decompress);
}
