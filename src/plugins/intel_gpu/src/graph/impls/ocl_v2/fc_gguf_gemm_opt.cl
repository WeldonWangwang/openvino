// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF prefill GEMM (OCL dpas) for Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and Q8_0 (ported from the CM gemm_q*_full.cm kernels).
//
// One source for all six formats: the host defines exactly one GG_FMT_* macro per kernel and gives each format its
// own entry point (fc_gguf_gemm_<fmt>). Each section below is the per-format tile codec (gg_fetch / gg_decode, see
// fc_gguf_gemm_body.cl) that decodes the SG-shuffled weight layout; the shared pipeline lives in the body.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "fc_gguf_gemm_common.cl"

#if !GG_SHUFFLED

#define GG_RAW_T uint16

inline uchar FUNC(gg_native_quant)(const __global uchar* block, uint index) {
#if GG_FMT_Q4_0
    return (block[2 + index % 16] >> (index / 16 * 4)) & 15;
#elif GG_FMT_Q4_1
    return (block[4 + index % 16] >> (index / 16 * 4)) & 15;
#elif GG_FMT_Q8_0
    return block[2 + index];
#elif GG_FMT_Q4_K || GG_FMT_Q5_K
#if GG_FMT_Q5_K
    const uint quant_offset = 48;
#else
    const uint quant_offset = 16;
#endif
    const uint group = index / 32;
    uint quant = (block[quant_offset + group / 2 * 32 + index % 32] >> (group % 2 * 4)) & 15;
#if GG_FMT_Q5_K
    quant |= ((block[16 + index % 32] >> group) & 1) << 4;
#endif
    return (uchar)quant;
#elif GG_FMT_Q6_K
    const uint half_block = index / 128;
    const uint group = index % 128 / 32;
    const uint low = (block[half_block * 64 + group % 2 * 32 + index % 32] >> (group / 2 * 4)) & 15;
    const uint high = (block[128 + half_block * 32 + index % 32] >> (group * 2)) & 3;
    return (uchar)(low | (high << 4));
#endif
}

inline uint16 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint blocks_per_row = GG_K / 256;
    const uint row = blk / blocks_per_row * 16 + get_sub_group_local_id();
    const uint column_block = blk % blocks_per_row;
#if GG_FMT_Q4_0 || GG_FMT_Q4_1 || GG_FMT_Q8_0
#if GG_FMT_Q4_0
    const uint block_bytes = 18;
#elif GG_FMT_Q4_1
    const uint block_bytes = 20;
#else
    const uint block_bytes = 34;
#endif
    const __global uchar* block = W + ((ulong)row * (GG_K / 32) + column_block * 8 + j) * block_bytes;
    const uint start = 0;
#else
#if GG_FMT_Q4_K
    const uint block_bytes = 144;
#elif GG_FMT_Q5_K
    const uint block_bytes = 176;
#else
    const uint block_bytes = 210;
#endif
    const __global uchar* block = W + ((ulong)row * blocks_per_row + column_block) * block_bytes;
    const uint start = j * 32;
#endif
    float scale_low = convert_float(vload_half(0, (const __global half*)block));
    float scale_high = scale_low;
    float offset_low = 0.0f;
    float offset_high = 0.0f;
#if GG_FMT_Q4_0
    offset_low = offset_high = -8.0f * scale_low;
#elif GG_FMT_Q4_1
    offset_low = offset_high = convert_float(vload_half(0, (const __global half*)(block + 2)));
#elif GG_FMT_Q4_K || GG_FMT_Q5_K
    const __global uchar* scales = block + 4;
    const uint scale = j < 4 ? scales[j] & 63 : (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4);
    const uint minimum = j < 4 ? scales[j + 4] & 63 : (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    scale_low = scale_high = (float)scale * scale_low;
    offset_low = offset_high = -(float)minimum * convert_float(vload_half(0, (const __global half*)(block + 2)));
#elif GG_FMT_Q6_K
    const float delta = convert_float(vload_half(0, (const __global half*)(block + 208)));
    scale_low = delta * (float)(char)block[192 + j * 2];
    scale_high = delta * (float)(char)block[193 + j * 2];
    offset_low = -32.0f * scale_low;
    offset_high = -32.0f * scale_high;
#endif
    uint8 low;
    uint8 high;
    __attribute__((opencl_unroll_hint))
    for (uint chunk = 0; chunk < 4; ++chunk) {
        uint quant_low = 0;
        uint quant_high = 0;
        __attribute__((opencl_unroll_hint))
        for (uint byte_index = 0; byte_index < 4; ++byte_index) {
            const uint index = start + chunk * 4 + byte_index;
            quant_low |= (uint)FUNC_CALL(gg_native_quant)(block, index) << (byte_index * 8);
            quant_high |= (uint)FUNC_CALL(gg_native_quant)(block, index + 16) << (byte_index * 8);
        }
#if GG_FMT_Q8_0
        const uint2 decoded_low = FUNC_CALL(gg_cvt4s)(quant_low, scale_low);
        const uint2 decoded_high = FUNC_CALL(gg_cvt4s)(quant_high, scale_high);
#else
        const uint2 decoded_low = FUNC_CALL(gg_cvt4u)(quant_low, scale_low, offset_low);
        const uint2 decoded_high = FUNC_CALL(gg_cvt4u)(quant_high, scale_high, offset_high);
#endif
        low[chunk * 2] = decoded_low.s0;
        low[chunk * 2 + 1] = decoded_low.s1;
        high[chunk * 2] = decoded_high.s0;
        high[chunk * 2 + 1] = decoded_high.s1;
    }
    return (uint16)(low, high);
}

inline void FUNC(gg_decode)(uint16 raw, uint j, uint8* low, uint8* high) {
    *low = raw.s01234567;
    *high = raw.s89abcdef;
}

#elif GG_FMT_Q4_0
// GGUF Q4_0 prefill GEMM (port of cm/gguf/gemm_q40_full.cm). w = (q - 8) * d.
// Weights (shuffle layout): pqs_T(128B/block) || pd_T(16B/block).

#define GG_RAW_T uint8

inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 128 + (ulong)blk * 256 + j * 32), 0);
    return (uint8)(q, d, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s4));
    FUNC_CALL(gg_decode_nib4)(r.s0123, d, d * 8.0f, blo, bhi);
}

#elif GG_FMT_Q4_1
// GGUF Q4_1 prefill GEMM (port of cm/gguf/gemm_q41_full.cm). w = q * d + m.
// Weights (shuffle layout): pqs_T(128B/block) || pd_T(16B/block) || pm_T(16B/block).

#define GG_RAW_T uint8

inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const ulong meta = (ulong)blk * 256 + j * 32;
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 128 + meta), 0);
    const uint m = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 144 + meta), 0);
    return (uint8)(q, d, m, 0, 0);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s4));
    const float m = convert_float(as_half((ushort)r.s5));
    FUNC_CALL(gg_decode_nib4)(r.s0123, d, -m, blo, bhi);
}

#elif GG_FMT_Q4_K
// GGUF Q4_K prefill GEMM (port of cm/gguf/gemm_q4k_full.cm). w = q * (sq * d) - (mq * dmin).
// Weights (shuffle layout): pqs_T(128B/block) || psl_T(16B/block, SoA fields sl,ml,sh,mh,d,dmin).

#define GG_RAW_T uint8

// Raw tile: q(4) | sl | ml | sh,mh | d,dmin.
inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const __global uchar* psl = W + (ulong)num_blocks * 128 + (ulong)blk * 256;
    const uint2 slml = BLOCK_READN(uint, 2, psl, 0);
    const uint2 meta = as_uint2(BLOCK_READN(ushort, 4, (psl + 128), 0));
    return (uint8)(q, slml, meta);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const uint sh = r.s6 & 0xFFFFu;
    const uint mh = r.s6 >> 16;
    const float d = convert_float(as_half((ushort)(r.s7 & 0xFFFFu)));
    const float dmin = convert_float(as_half((ushort)(r.s7 >> 16)));
    const uint sq = ((r.s4 >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
    const uint mq = ((r.s5 >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
    FUNC_CALL(gg_decode_nib4)(r.s0123, (float)sq * d, (float)mq * dmin, blo, bhi);
}

#elif GG_FMT_Q5_K
// GGUF Q5_K prefill GEMM (port of cm/gguf/gemm_q5k_full.cm). w = (nib | bit5 << 4) * (sq * d) - (mq * dmin).
// Weights (shuffle layout): pqs_T(128B/block) || pqh_T(32B/block) || psl_T(16B/block).

#define GG_RAW_T uint16

// Raw tile: q(4) | H | sl | ml | sh,mh | d,dmin.
inline uint16 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const uint H = BLOCK_READN(uint, 1, (W + (ulong)num_blocks * 128 + (ulong)blk * 512 + j * 64), 0);
    const __global uchar* psl = W + (ulong)num_blocks * 160 + (ulong)blk * 256;
    const uint2 slml = BLOCK_READN(uint, 2, psl, 0);
    const uint2 meta = as_uint2(BLOCK_READN(ushort, 4, (psl + 128), 0));
    return (uint16)(q, H, slml, meta, 0, 0, 0, 0, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint16 r, uint j, uint8* blo, uint8* bhi) {
    const uint H = r.s4;
    const uint sh = r.s7 & 0xFFFFu;
    const uint mh = r.s7 >> 16;
    const float d = convert_float(as_half((ushort)(r.s8 & 0xFFFFu)));
    const float dmin = convert_float(as_half((ushort)(r.s8 >> 16)));
    const uint sq = ((r.s5 >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
    const uint mq = ((r.s6 >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
    const float scale = (float)sq * d;
    const float minv = (float)mq * dmin;
    // Byte b of chunk c: low nibble K=4c+b with 5th bit at H bit 8b+c, high nibble K=16+4c+b with H bit 8b+4+c.
    uint2 lo[4];
    uint2 hi[4];
    __attribute__((opencl_unroll_hint))
    for (uint c = 0; c < 4; ++c) {
        const uint u = c == 0 ? r.s0 : c == 1 ? r.s1 : c == 2 ? r.s2 : r.s3;
        lo[c] = FUNC_CALL(gg_cvt4u)((u & 0x0F0F0F0Fu) | (((H >> c) & 0x01010101u) << 4), scale, -minv);
        hi[c] = FUNC_CALL(gg_cvt4u)(((u >> 4) & 0x0F0F0F0Fu) | (((H >> (c + 4)) & 0x01010101u) << 4), scale, -minv);
    }
    *blo = (uint8)(lo[0], lo[1], lo[2], lo[3]);
    *bhi = (uint8)(hi[0], hi[1], hi[2], hi[3]);
}

#elif GG_FMT_Q6_K
// GGUF Q6_K prefill GEMM (port of cm/gguf/gemm_q6k_full.cm). w = ((nib | h2 << 4) - 32) * (d * sc).
// Weights (shuffle layout): pql_T(128B/block) || pqh_T(64B/block) || ps_T(16B/block) || pd_T(2B/block).

#define GG_RAW_T uint8

// Raw tile: q(4) | HL | HH | d | scales(lo, hi bytes).
inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const uint2 h = BLOCK_READN(uint, 2, (W + (ulong)num_blocks * 128 + (ulong)blk * 1024 + (2 * j) * 64), 0);
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 208 + (ulong)blk * 32), 0);
    const uchar2 sc = BLOCK_READN(uchar, 2, W, (ulong)num_blocks * 192 + (ulong)blk * 256 + (2 * j) * 16);
    return (uint8)(q, h, d, (uint)as_ushort(sc));
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const uint HL = r.s4;
    const uint HH = r.s5;
    const float d = convert_float(as_half((ushort)r.s6));
    const float clo = d * (float)(char)(r.s7 & 0xFFu);
    const float chi = d * (float)(char)(r.s7 >> 8);
    // Byte b of chunk c: low nibble K=4c+b with upper bits at HL bits 8b+2c, high nibble K=16+4c+b with HH bits 8b+2c.
    uint2 lo[4];
    uint2 hi[4];
    __attribute__((opencl_unroll_hint))
    for (uint c = 0; c < 4; ++c) {
        const uint u = c == 0 ? r.s0 : c == 1 ? r.s1 : c == 2 ? r.s2 : r.s3;
        lo[c] = FUNC_CALL(gg_cvt4u)((u & 0x0F0F0F0Fu) | (((HL >> (2 * c)) & 0x03030303u) << 4), clo, -32.0f * clo);
        hi[c] = FUNC_CALL(gg_cvt4u)(((u >> 4) & 0x0F0F0F0Fu) | (((HH >> (2 * c)) & 0x03030303u) << 4), chi, -32.0f * chi);
    }
    *blo = (uint8)(lo[0], lo[1], lo[2], lo[3]);
    *bhi = (uint8)(hi[0], hi[1], hi[2], hi[3]);
}

#elif GG_FMT_Q8_0
// GGUF Q8_0 prefill GEMM (port of cm/gguf/gemm_q80_full.cm). w = q * d.
// Weights (shuffle layout): pqs_T(256B/block) || pd_T(16B/block). Sub-block payload: 512B = lo(256B) || hi(256B).

#define GG_RAW_T uint16

// Raw tile: q lo(4) | q hi(4) | d.
inline uint16 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint8 q = BLOCK_READN(uint, 8, (W + (ulong)blk * 4096 + j * 512), 0);
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 256 + (ulong)blk * 256 + j * 32), 0);
    return (uint16)(q, d, 0, 0, 0, 0, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint16 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s8));
    *blo = (uint8)(FUNC_CALL(gg_cvt4s)(r.s0, d), FUNC_CALL(gg_cvt4s)(r.s1, d), FUNC_CALL(gg_cvt4s)(r.s2, d), FUNC_CALL(gg_cvt4s)(r.s3, d));
    *bhi = (uint8)(FUNC_CALL(gg_cvt4s)(r.s4, d), FUNC_CALL(gg_cvt4s)(r.s5, d), FUNC_CALL(gg_cvt4s)(r.s6, d), FUNC_CALL(gg_cvt4s)(r.s7, d));
}

#else
#error "fc_gguf_gemm_opt.cl: define exactly one GG_FMT_* format macro"
#endif

#include "fc_gguf_gemm_body.cl"
