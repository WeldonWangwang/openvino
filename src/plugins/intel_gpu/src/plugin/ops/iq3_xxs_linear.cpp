// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GPU plugin op factory for the internal IQ3XXSLinear op.
//
// M4 step 1 (correctness closure): the compressed IQ3_XXS weight blob is
// decoded on the host into a dense weights Constant and lowered to the existing,
// highly-optimized cldnn `fully_connected` primitive. This makes the *native*
// IQ3XXSLinear graph (OV_GENAI_IQ3XXS_NATIVE=1) loadable and correct on GPU,
// reusing the proven GPU MatMul kernels.
//
// NOTE: this materializes the full FP16/F32 weight tensor in device memory at
// compile time, so it does NOT yet preserve the runtime compression benefit on
// GPU. The on-the-fly compressed OpenCL kernel (decode-in-kernel) is the planned
// follow-up (M4 step 2) and will replace this lowering.

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/iq3_xxs_linear.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/type/float16.hpp"

#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/data.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// IQ3_XXS codebook tables (from ggml-common.h; identical to the CPU/core paths
// so GPU results are bit-compatible with the reference dequantizer).
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

constexpr size_t QK_K = 256;        // weights per super-block
constexpr size_t BLOCK_BYTES = 98;  // bytes per super-block

inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

// Decode one IQ3_XXS weight row (K values) into a contiguous f32 buffer.
inline void decode_iq3_xxs_row(const uint8_t* w_row, float* out, size_t blocks_per_row) {
    for (size_t blk = 0; blk < blocks_per_row; blk++) {
        const uint8_t* block_data = w_row + blk * BLOCK_BYTES;

        uint16_t d_fp16;
        std::memcpy(&d_fp16, block_data, 2);
        const float d = fp16_to_f32(d_fp16);

        const uint8_t* qs = block_data + 2;
        const uint8_t* scales_and_signs = qs + QK_K / 4;  // +64

        float* o = out + blk * QK_K;
        size_t oi = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = kSignsIq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&kIq3xxsGrid[qs[2 * l + 0]]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&kIq3xxsGrid[qs[2 * l + 1]]);

                for (int j = 0; j < 4; ++j) {
                    o[oi++] = db * grid1[j] * ((signs & kMaskIq2xs[j + 0]) ? -1.f : 1.f);
                }
                for (int j = 0; j < 4; ++j) {
                    o[oi++] = db * grid2[j] * ((signs & kMaskIq2xs[j + 4]) ? -1.f : 1.f);
                }
            }
            qs += 8;
        }
    }
}

}  // namespace

namespace ov::intel_gpu {

static void CreateIQ3XXSLinearOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::IQ3XXSLinear>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    const auto& weight_shape = op->get_weight_shape();  // [N, K]
    OPENVINO_ASSERT(weight_shape.size() == 2,
                    "[GPU] IQ3XXSLinear weight_shape must be 2D [N, K], got rank ", weight_shape.size());
    const size_t N = weight_shape[0];
    const size_t K = weight_shape[1];
    OPENVINO_ASSERT(K % QK_K == 0,
                    "[GPU] IQ3XXSLinear K (", K, ") must be a multiple of ", QK_K);

    // Fetch the opaque compressed weight blob from the u8 Constant input.
    auto weights_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(weights_const != nullptr,
                    "[GPU] IQ3XXSLinear expects a Constant compressed-weights input");
    const uint8_t* compressed = weights_const->get_data_ptr<uint8_t>();

    const size_t blocks_per_row = K / QK_K;
    const size_t bytes_per_row = blocks_per_row * BLOCK_BYTES;

    // Host-side full dequantization into a dense [N, K] weights buffer.
    const auto out_et = op->get_output_element_type(0);
    const auto w_dtype = cldnn::element_type_to_data_type(out_et);

    cldnn::layout weights_layout(ov::PartialShape({static_cast<int64_t>(N), static_cast<int64_t>(K)}),
                                 w_dtype,
                                 cldnn::format::get_default_format(2));
    auto mem = p.get_engine().allocate_memory(weights_layout, false);

    {
        auto& stream = p.get_engine().get_service_stream();
        cldnn::mem_lock<char> lock{mem, stream};
        char* dst = lock.data();

        std::vector<float> row(K);
        for (size_t n = 0; n < N; ++n) {
            decode_iq3_xxs_row(compressed + n * bytes_per_row, row.data(), blocks_per_row);
            if (out_et == ov::element::f16) {
                auto* hdst = reinterpret_cast<ov::float16*>(dst) + n * K;
                for (size_t k = 0; k < K; ++k) {
                    hdst[k] = ov::float16(row[k]);
                }
            } else {  // f32
                auto* fdst = reinterpret_cast<float*>(dst) + n * K;
                std::memcpy(fdst, row.data(), K * sizeof(float));
            }
        }
    }

    cldnn::primitive_id weightsName = layerName + "_decoded_weights";
    p.add_primitive(*op, cldnn::data(weightsName, mem));

    const size_t rank_a = op->get_input_partial_shape(0).size();
    auto fcPrim = cldnn::fully_connected(layerName,
                                         inputs[0],
                                         weightsName,
                                         "",  // no bias
                                         w_dtype,
                                         rank_a,
                                         2 /* weights_rank */);

    p.add_primitive(*op, fcPrim);
}

REGISTER_FACTORY_IMPL(internal, IQ3XXSLinear);

}  // namespace ov::intel_gpu
