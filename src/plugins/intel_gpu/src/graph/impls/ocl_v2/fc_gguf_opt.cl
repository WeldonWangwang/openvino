// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF FullyConnected kernel.
//
// Computes  C[bm, n] = sum_k  A[bm, k] * W[n, k]
// where W is a raw GGUF block-quantised weight matrix [N, K] consumed directly from HBM and
// dequantised in registers (never materialised to f16/f32 in memory). The activation A is f16/f32
// [BM, K] and the output C is f16/f32 [BM, N]. BM is the flattened batch*sequence dimension, so the
// same kernel serves both the M==1 decode (GEMV) and M>1 prefill (GEMM) phases.
//
// One subgroup (SG_SIZE lanes) cooperatively owns one (n, bm) output: the blocks of W's row `n` are
// striped across the lanes (lane L decodes blocks L, L+SG_SIZE, ...), each lane streams its block's
// dot product against the matching A slice WITHOUT materialising the dequantised block in private
// memory, and a single sub_group_reduce_add collapses the partials. Striping keeps the SG_SIZE
// lanes sweeping a contiguous block window each step (coalesced) and keeps all SIMD lanes busy --
// the previous 1-work-item-per-output layout left 15/16 lanes idle (LWS=1) and was memory-starved.
// The streaming dot mirrors the canonical ggml reference (ggml-quants.c) and the CPU reference in
// the GGUF frontend (src/frontends/gguf/src/builders/dequantize.cpp).
//
// The packed GGUF source format is selected at JIT time by exactly one GGUF_IS_<TYPE> flag, together
// with GGUF_BLOCK_ELEM (logical elements per block) and GGUF_BLOCK_BYTES (bytes per block).
//
// Helper functions are wrapped in FUNC()/FUNC_CALL() so their names are decorated with the kernel
// entry point — multiple GGUF FC kernels (different shapes/formats) are batch-compiled into a single
// OpenCL program, and undecorated names would collide ("redefinition of ...").

#include "include/batch_headers/common.cl"

// Reconstruct a half from two little-endian bytes (GGUF is little-endian, as is every OV host/target).
inline half FUNC(gguf_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

// ============================================================================
// Per-format streaming block dot. Each returns sum_{j in [0, GGUF_BLOCK_ELEM)} a[j] * dequant(blk[j])
// for the block starting at `blk` against the activation slice `a`, accumulating in float without
// materialising the dequantised block (keeps register pressure low so SG_SIZE lanes stay resident).
// ============================================================================

#if defined(GGUF_IS_Q4_0)
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const int lo = (int)(qs[j] & 0x0F) - 8;
        const int hi = (int)(qs[j] >> 4) - 8;
        acc += (float)a[j]      * ((float)lo * d);
        acc += (float)a[j + 16] * ((float)hi * d);
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q8_0)
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    float acc = 0.0f;
    for (int j = 0; j < 32; ++j) {
        acc += (float)a[j] * ((float)qs[j] * d);
    }
    return acc;
}
#endif

// 6-bit packed sub-block scale/min extraction shared by Q4_K / Q5_K (ggml get_scale_min_k4).
#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
inline void FUNC(gguf_get_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
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
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const float d    = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;    // 12 bytes
    const __global uchar* qs     = blk + 16;   // 128 bytes
    float acc = 0.0f;
    int ai = 0;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        // Factor the per-element (d*q - m) into d*sum(a*q) - m*sum(a): one fma + one add per element.
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float av = (float)a[ai + l];
            sq1 += av * (float)(qs[l] & 0x0F);
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = (float)a[ai + 32 + l];
            sq2 += av * (float)(qs[l] >> 4);
            sa2 += av;
        }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        qs += 32;
        is += 2;
        ai += 64;
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q5_K)
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const float d    = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;    // 12 bytes
    const __global uchar* qh     = blk + 16;   // 32 bytes (high bit-plane)
    const __global uchar* ql     = blk + 48;   // 128 bytes (low 4 bits)
    float acc = 0.0f;
    int ai = 0;
    int is = 0;
    uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        // Factor the per-element (d*q - m) into d*sum(a*q) - m*sum(a): one fma + one add per element.
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float av = (float)a[ai + l];
            const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            sq1 += av * (float)q;
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = (float)a[ai + 32 + l];
            const int q = (int)(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            sq2 += av * (float)q;
            sa2 += av;
        }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        ai += 64;
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q6_K)
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const __global uchar* ql = blk;            // 128 bytes (low 4 bits)
    const __global uchar* qh = blk + 128;      // 64 bytes (high 2 bits)
    const __global char*  sc = (const __global char*)(blk + 192);  // 16 signed scales
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk + 208);
    // Four independent accumulators + inner unroll: Q6_K decode is latency-bound on the four
    // length-32 dependent FMA chains, so unrolling lets the (independent) per-element unpacks
    // pipeline and the four chains overlap. Measured +27% (24.9% -> 31.7% of B580 BW roofline) vs
    // the single-accumulator form. Q5_K showed the opposite (register/occupancy-bound) so only Q6_K
    // uses this form -- the split is intentionally format-local.
    float acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        __attribute__((opencl_unroll_hint(8)))
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc1 += (float)a[o + l + 0]  * (d * (float)sc[is + 0] * q1);
            acc2 += (float)a[o + l + 32] * (d * (float)sc[is + 2] * q2);
            acc3 += (float)a[o + l + 64] * (d * (float)sc[is + 4] * q3);
            acc4 += (float)a[o + l + 96] * (d * (float)sc[is + 6] * q4);
        }
        o  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc1 + acc2 + acc3 + acc4;
}
#endif

#if defined(GGUF_IS_IQ3_XXS)
CONST_ARRAY_DECL(iq3xxs_grid) = {
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

CONST_ARRAY_DECL(ksigns_iq2xs) = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    const __global uchar* scales_signs = blk + 2 + 64;
    float acc = 0.0f;
    int ai = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const __global uchar* p4 = scales_signs + 4 * ib32;
        const uint aux32 = (uint)p4[0] | ((uint)p4[1] << 8) | ((uint)p4[2] << 16) | ((uint)p4[3] << 24);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uchar signs = CONST_ARRAY_REF(ksigns_iq2xs)[(aux32 >> (7 * l)) & 127u];
            const uint g1 = CONST_ARRAY_REF(iq3xxs_grid)[qs[2*l + 0]];
            const uint g2 = CONST_ARRAY_REF(iq3xxs_grid)[qs[2*l + 1]];
            const uchar g1b0 = (uchar)(g1 & 0xFFu);
            const uchar g1b1 = (uchar)((g1 >> 8) & 0xFFu);
            const uchar g1b2 = (uchar)((g1 >> 16) & 0xFFu);
            const uchar g1b3 = (uchar)((g1 >> 24) & 0xFFu);
            const uchar g2b0 = (uchar)(g2 & 0xFFu);
            const uchar g2b1 = (uchar)((g2 >> 8) & 0xFFu);
            const uchar g2b2 = (uchar)((g2 >> 16) & 0xFFu);
            const uchar g2b3 = (uchar)((g2 >> 24) & 0xFFu);
            acc += (float)a[ai + 0] * (db * (float)g1b0 * ((signs & 1u) ? -1.0f : 1.0f));
            acc += (float)a[ai + 1] * (db * (float)g1b1 * ((signs & 2u) ? -1.0f : 1.0f));
            acc += (float)a[ai + 2] * (db * (float)g1b2 * ((signs & 4u) ? -1.0f : 1.0f));
            acc += (float)a[ai + 3] * (db * (float)g1b3 * ((signs & 8u) ? -1.0f : 1.0f));
            acc += (float)a[ai + 4] * (db * (float)g2b0 * ((signs & 16u) ? -1.0f : 1.0f));
            acc += (float)a[ai + 5] * (db * (float)g2b1 * ((signs & 32u) ? -1.0f : 1.0f));
            acc += (float)a[ai + 6] * (db * (float)g2b2 * ((signs & 64u) ? -1.0f : 1.0f));
            acc += (float)a[ai + 7] * (db * (float)g2b3 * ((signs & 128u) ? -1.0f : 1.0f));
            ai += 8;
        }
        qs += 8;
    }
    return acc;
}
#endif

// ============================================================================
// Main kernel: one subgroup (SG_SIZE lanes) per (n, bm) output element.
//   global = [N_SIZE * SG_SIZE, BM, 1]   (BM = flattened batch*seq rows of the activation)
//   local  = [SG_SIZE, 1, 1]             (one subgroup per work-group)
// K_SIZE and N_SIZE are static (the reduction and output-channel dims are fixed by the GGUF weight);
// only BM (activation rows) may be dynamic, and the dispatch sets global[1] == BM exactly, so the row
// index is taken straight from get_global_id(1) and needs no BM_SIZE bound (works for static & dynamic).
// n = get_global_id(0)/SG_SIZE is uniform across a subgroup, so the early-out and the
// sub_group_reduce_add are reached by all lanes together (no collective divergence).
// ============================================================================
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(fc_gguf_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,   // activations [BM, K]
    const __global uchar*       W,   // GGUF block weights [N, K] (opaque bytes)
          __global OUTPUT_TYPE* C    // output      [BM, N]
)
{
    const int n    = (int)(get_global_id(0) / SG_SIZE);
    const int lane = (int)get_sub_group_local_id();
    const int bm   = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    const __global uchar* w_row = W + (uint)n * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
    const __global INPUT0_TYPE* a_row = A + (uint)bm * (uint)K_SIZE;

    // Stripe row `n`'s blocks across the subgroup lanes; each lane streams its blocks' dot product.
    float partial = 0.0f;
    for (int kb = lane; kb < blocks_per_row; kb += SG_SIZE) {
        partial += FUNC_CALL(gguf_block_dot)(w_row + (uint)kb * GGUF_BLOCK_BYTES,
                                             a_row + (uint)kb * GGUF_BLOCK_ELEM);
    }

    const float total = sub_group_reduce_add(partial);
    if (lane == 0)
        C[(uint)bm * (uint)N_SIZE + n] = TO_OUTPUT_TYPE(total);
}
