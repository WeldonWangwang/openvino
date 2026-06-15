// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF weight transcode kernel (compute-bound / large-M prefill path).
//
// Converts a raw GGUF block-quantised weight matrix W[N, K] into a OneDNN-WOQ-native low-bit layout:
//   - a packed weight scratchpad: i4 (TRANSCODE_TO_I4=1) or i8 (signed), in [N, K] physical order
//     (matches dnnl wei_md [K,N] with format_tag::ba), and
//   - a parallel f16 per-group scale scratchpad [K/REQUANT_GROUP, N] = dnnl scale md [K/group, N]
//     with element (g, n) at g*N + n (per-K-group x per-N mask).
//
// The block bytes are decoded to half in registers with the SAME per-format decoders used by the
// native GEMV kernel (so numerics track exactly), then symmetrically re-quantized per REQUANT_GROUP
// elements to the target low-bit domain. dequant NEVER lands in an f16/f32 weight buffer (constraint
// C2): the only persisted weight is the low-bit scratchpad; the f16 values live only in registers.
//
// One work-item owns one (n, GGUF block): global = [N_SIZE, K_SIZE / GGUF_BLOCK_ELEM, 1], local = [SG, 1, 1].
// The block is decoded once and every REQUANT group inside it is requantized from the shared decoded
// window, so the heavy bit-unpacking runs a single time per block instead of
// (GGUF_BLOCK_ELEM / REQUANT_GROUP)x (8x for K-quants with a 256-elem block and a 32-elem group).
// REQUANT_GROUP must divide GGUF_BLOCK_ELEM (so a group never straddles two GGUF blocks).

#include "include/batch_headers/common.cl"

inline half FUNC(tq_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

// ---- per-format block decoders (identical math to fc_gguf_opt.cl) ----

#if defined(GGUF_IS_Q4_0)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        out[j]      = (half)(((int)(qs[j] & 0x0F) - 8) * (float)d);
        out[j + 16] = (half)(((int)(qs[j] >> 4)   - 8) * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q8_0)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(tq_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    for (int j = 0; j < 32; ++j) {
        out[j] = (half)((float)qs[j] * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
inline void FUNC(tq_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
#endif

#if defined(GGUF_IS_Q4_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(tq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(tq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qs     = blk + 16;
    int o = 0, is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(tq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(tq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d1 * (float)(qs[l] & 0x0F) - m1);
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d2 * (float)(qs[l] >> 4) - m2);
        qs += 32; is += 2;
    }
}
#endif

#if defined(GGUF_IS_Q5_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(tq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(tq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qh     = blk + 16;
    const __global uchar* ql     = blk + 48;
    int o = 0, is = 0; uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(tq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(tq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0); out[o++] = (half)(d1 * (float)q - m1); }
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0); out[o++] = (half)(d2 * (float)q - m2); }
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
#endif

#if defined(GGUF_IS_Q6_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const __global uchar* ql = blk;
    const __global uchar* qh = blk + 128;
    const __global char*  sc = (const __global char*)(blk + 192);
    const float d = (float)FUNC_CALL(tq_load_f16)(blk + 208);
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[o + l + 0]  = (half)(d * (float)sc[is + 0] * q1);
            out[o + l + 32] = (half)(d * (float)sc[is + 2] * q2);
            out[o + l + 64] = (half)(d * (float)sc[is + 4] * q3);
            out[o + l + 96] = (half)(d * (float)sc[is + 6] * q4);
        }
        o += 128; ql += 64; qh += 32; sc += 8;
    }
}
#endif

#if defined(GGUF_IS_IQ3_XXS)
CONST_ARRAY_DECL(tq_iq3xxs_grid) = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

CONST_ARRAY_DECL(tq_ksigns_iq2xs) = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    const __global uchar* scales_signs = blk + 2 + 64;
    int ai = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const __global uchar* p4 = scales_signs + 4 * ib32;
        const uint aux32 = (uint)p4[0] | ((uint)p4[1] << 8) | ((uint)p4[2] << 16) | ((uint)p4[3] << 24);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uchar signs = CONST_ARRAY_REF(tq_ksigns_iq2xs)[(aux32 >> (7 * l)) & 127u];
            const uint g1 = CONST_ARRAY_REF(tq_iq3xxs_grid)[qs[2 * l + 0]];
            const uint g2 = CONST_ARRAY_REF(tq_iq3xxs_grid)[qs[2 * l + 1]];
            out[ai + 0] = (half)(db * (float)(uchar)(g1 & 0xFFu)         * ((signs & 1u) ? -1.0f : 1.0f));
            out[ai + 1] = (half)(db * (float)(uchar)((g1 >> 8) & 0xFFu)  * ((signs & 2u) ? -1.0f : 1.0f));
            out[ai + 2] = (half)(db * (float)(uchar)((g1 >> 16) & 0xFFu) * ((signs & 4u) ? -1.0f : 1.0f));
            out[ai + 3] = (half)(db * (float)(uchar)((g1 >> 24) & 0xFFu) * ((signs & 8u) ? -1.0f : 1.0f));
            out[ai + 4] = (half)(db * (float)(uchar)(g2 & 0xFFu)         * ((signs & 16u) ? -1.0f : 1.0f));
            out[ai + 5] = (half)(db * (float)(uchar)((g2 >> 8) & 0xFFu)  * ((signs & 32u) ? -1.0f : 1.0f));
            out[ai + 6] = (half)(db * (float)(uchar)((g2 >> 16) & 0xFFu) * ((signs & 64u) ? -1.0f : 1.0f));
            out[ai + 7] = (half)(db * (float)(uchar)((g2 >> 24) & 0xFFu) * ((signs & 128u) ? -1.0f : 1.0f));
            ai += 8;
        }
        qs += 8;
    }
}
#endif

// ---- main transcode kernel ----
// TRANSCODE_TO_I4 : 1 -> pack two i4 nibbles per output byte; 0 -> one i8 per byte.
// QMAX            : 7 (i4 symmetric) or 127 (i8 symmetric).
// REQUANT_GROUP   : elements sharing one f16 scale (divides GGUF_BLOCK_ELEM).
KERNEL(fc_gguf_transcode)(
    const __global uchar* W,        // GGUF block weights [N, K] (opaque bytes)
          __global uchar* WQ,       // out: packed low-bit weight [N, K] (i4 packed / i8)
          __global half*  SC        // out: per-group f16 scale [N, K/REQUANT_GROUP]
)
{
    const int n   = (int)get_global_id(0);          // output row (subgroup lane axis, padded to SG)
    const int blk = (int)get_global_id(1);          // GGUF block index along K
    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    if (n >= N_SIZE || blk >= blocks_per_row)
        return;

    const __global uchar* w_row = W + (uint)n * (uint)blocks_per_row * GGUF_BLOCK_BYTES;

    // Decode the whole GGUF block ONCE. Every REQUANT group inside it reuses this decoded window, so
    // the expensive bit-unpacking runs a single time per block instead of once per group.
    half blk_vals[GGUF_BLOCK_ELEM];
    FUNC_CALL(tq_decode_block)(w_row + (uint)blk * GGUF_BLOCK_BYTES, blk_vals);

    const int groups_per_block = GGUF_BLOCK_ELEM / REQUANT_GROUP;
    const uint row_base = (uint)n * (uint)K_SIZE;
#if !TRANSCODE_TO_I4
    __global char* wq_i8 = (__global char*)WQ;
#endif

    // Symmetric per-group requantization for each REQUANT group within the decoded block.
    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int off_in_blk = gi * REQUANT_GROUP;        // group offset within the decoded block
        const int g  = blk * groups_per_block + gi;       // global group index along K
        const int k0 = g * REQUANT_GROUP;                 // first K element of this group

        float amax = 0.0f;
        for (int i = 0; i < REQUANT_GROUP; ++i) {
            float v = fabs((float)blk_vals[off_in_blk + i]);
            amax = fmax(amax, v);
        }
        const float scale     = (amax > 0.0f) ? (amax / (float)QMAX) : 1.0f;
        const float inv_scale = (amax > 0.0f) ? ((float)QMAX / amax) : 0.0f;

        // Scale md is [K/group, N] (per-K-group x per-N): element (g, n) at g*N + n.
        SC[(uint)g * (uint)N_SIZE + (uint)n] = (half)scale;

#if TRANSCODE_TO_I4
        // i4 packed two-per-byte; weight byte index = (n*K + k)/2. REQUANT_GROUP is even.
        for (int i = 0; i < REQUANT_GROUP; i += 2) {
            const int k = k0 + i;
            int q0 = (int)round((float)blk_vals[off_in_blk + i]     * inv_scale);
            int q1 = (int)round((float)blk_vals[off_in_blk + i + 1] * inv_scale);
            q0 = clamp(q0, -8, 7);
            q1 = clamp(q1, -8, 7);
            const uint byte_idx = (row_base + (uint)k) >> 1; // two consecutive k share one byte
            WQ[byte_idx] = (uchar)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
#else
        for (int i = 0; i < REQUANT_GROUP; ++i) {
            int q = (int)round((float)blk_vals[off_in_blk + i] * inv_scale);
            q = clamp(q, -128, 127);
            wq_i8[row_base + (uint)(k0 + i)] = (char)q;
        }
#endif
    }
}
