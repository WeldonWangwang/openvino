// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// IQ3_XXS decode + fused dot product. See iq3_xxs_kernels.hpp for the API.
//
// Constant tables `kIq3xxsGrid[256]` and `kSignsIq2xs[128]` are copied verbatim
// from ggml-common.h (llama.cpp). They are the canonical IQ3_XXS codebook;
// every reference implementation in this repo agrees on these values.

#include "iq3_xxs_kernels.hpp"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

namespace ov::intel_cpu::kernels::iq3_xxs {

namespace {

// ---------------------------------------------------------------------------
// IQ3_XXS codebook tables (verbatim from ggml-common.h).
// ---------------------------------------------------------------------------
const uint32_t kIq3xxsGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e,
    0x04041404, 0x04041414, 0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c,
    0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14, 0x040c140c, 0x040c142c,
    0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c,
    0x04141c1c, 0x04141c3e, 0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c,
    0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c,
    0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04,
    0x043e0c24, 0x043e0c34, 0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c,
    0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c, 0x0c041c04, 0x0c041c14,
    0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14,
    0x0c14140c, 0x0c141c04, 0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404,
    0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04,
    0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404,
    0x14041414, 0x14041434, 0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c,
    0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c, 0x140c1c04, 0x140c341c,
    0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c,
    0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c,
    0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24, 0x143e040c, 0x143e041c,
    0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414,
    0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c,
    0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c, 0x1c1c1c1c,
    0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404,
    0x24040424, 0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e,
    0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434,
    0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04,
    0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434,
    0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14, 0x2c1c0414, 0x2c1c2c1c,
    0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434,
    0x34043424, 0x340c140c, 0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04,
    0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c,
    0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14,
    0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,
    0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

const uint8_t kSignsIq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

const uint8_t kMaskIq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// ---------------------------------------------------------------------------
// Scalar IEEE 754 fp16->fp32 conversion. Independent of the host CPU's f16
// support: we want bit-exact output across builds.
// ---------------------------------------------------------------------------
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal
            exp = 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// decode_iq3_xxs_row: byte-for-byte translation of
// ggml::dequantize_row_iq3_xxs(). See ggml-quants.c for the original.
// ---------------------------------------------------------------------------
void decode_iq3_xxs_row(const uint8_t* w_row, float* out, size_t blocks_per_row) {
    for (size_t blk = 0; blk < blocks_per_row; ++blk) {
        const uint8_t* block_data = w_row + blk * BLOCK_BYTES;

        uint16_t d_fp16;
        std::memcpy(&d_fp16, block_data, sizeof(d_fp16));
        const float d = fp16_to_f32(d_fp16);

        const uint8_t* qs = block_data + 2;                  // 64 bytes grid indices
        const uint8_t* scales_and_signs = qs + QK_K / 4;     // 32 bytes (8x uint32_t)

        float* o = out + blk * QK_K;
        size_t oi = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = kSignsIq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 =
                    reinterpret_cast<const uint8_t*>(&kIq3xxsGrid[qs[2 * l + 0]]);
                const uint8_t* grid2 =
                    reinterpret_cast<const uint8_t*>(&kIq3xxsGrid[qs[2 * l + 1]]);

                for (int j = 0; j < 4; ++j) {
                    o[oi++] = db * static_cast<float>(grid1[j]) *
                              ((signs & kMaskIq2xs[j + 0]) ? -1.0f : 1.0f);
                }
                for (int j = 0; j < 4; ++j) {
                    o[oi++] = db * static_cast<float>(grid2[j]) *
                              ((signs & kMaskIq2xs[j + 4]) ? -1.0f : 1.0f);
                }
            }
            qs += 8;
        }
    }
}

// ---------------------------------------------------------------------------
// iq3_xxs_fc: parallel decode + dense MatMul.
// ---------------------------------------------------------------------------
void iq3_xxs_fc(const float* act,
                const uint8_t* weight_blob,
                float* out,
                size_t rows,
                size_t K,
                size_t N,
                size_t nthreads_hint) {
    const size_t blocks_per_row = K / QK_K;
    const size_t bytes_per_row = blocks_per_row * BLOCK_BYTES;

    // Worker: for output channels [n_lo, n_hi), decode each weight row once
    // and accumulate the dot product against every activation row.
    auto worker = [&](size_t n_lo, size_t n_hi) {
        std::vector<float> wbuf(K);
        for (size_t n = n_lo; n < n_hi; ++n) {
            decode_iq3_xxs_row(weight_blob + n * bytes_per_row, wbuf.data(), blocks_per_row);
            const float* w = wbuf.data();
            for (size_t r = 0; r < rows; ++r) {
                const float* a = act + r * K;
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += a[k] * w[k];
                }
                out[r * N + n] = acc;
            }
        }
    };

    size_t nthreads = nthreads_hint;
    if (nthreads == 0) {
        unsigned hw = std::thread::hardware_concurrency();
        nthreads = (hw == 0) ? 4u : static_cast<size_t>(hw);
    }

    // Cheap-shape escape: don't pay thread-launch cost on small problems.
    constexpr size_t PARALLEL_THRESHOLD = 1ull << 20;
    const size_t total_work = N * rows * K;
    if (total_work < PARALLEL_THRESHOLD || N < nthreads || nthreads <= 1) {
        worker(0, N);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    const size_t chunk = (N + nthreads - 1) / nthreads;
    for (size_t t = 1; t < nthreads; ++t) {
        const size_t b = t * chunk;
        if (b >= N) {
            break;
        }
        pool.emplace_back(worker, b, std::min(N, b + chunk));
    }
    worker(0, std::min(N, chunk));
    for (auto& th : pool) {
        th.join();
    }
}

}  // namespace ov::intel_cpu::kernels::iq3_xxs
