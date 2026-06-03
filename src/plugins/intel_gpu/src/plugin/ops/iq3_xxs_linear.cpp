// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GPU plugin op factory for the internal IQ3XXSLinear op.
//
// Two lowerings are provided:
//
//  * Default (M4 step 2): a custom OpenCL kernel that keeps the IQ3_XXS weight
//    blob *compressed* (u8) in device memory and decodes grid/sign/scale
//    on-the-fly inside the kernel while accumulating the dot product. The full
//    FP16/F32 weight tensor is never materialized in device memory, preserving
//    the IQ3_XXS runtime compression benefit on GPU.
//
//  * Fallback (M4 step 1, env OV_GPU_IQ3XXS_HOST_DEQUANT=1): host-side full
//    dequantization into a dense weights Constant lowered to the optimized
//    cldnn fully_connected primitive. Kept for A/B comparison and as a safety
//    net; it does materialize the dense weights.

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/iq3_xxs_linear.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/type/float16.hpp"

#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/data.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
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

    const size_t blocks_per_row = K / QK_K;
    const size_t bytes_per_row = blocks_per_row * BLOCK_BYTES;

    const auto out_et = op->get_output_element_type(0);
    const auto in_et = op->get_input_element_type(0);
    const auto w_dtype = cldnn::element_type_to_data_type(out_et);

    const bool host_dequant = std::getenv("OV_GPU_IQ3XXS_HOST_DEQUANT") != nullptr;

    // ----------------------------------------------------------------------
    // Default path (step 2): on-the-fly compressed-weight decode OCL kernel.
    // The u8 compressed blob (inputs[1]) stays compressed in device memory.
    // ----------------------------------------------------------------------
    if (!host_dequant) {
        const std::string act_type = (in_et == ov::element::f16) ? "half" : "float";
        const std::string out_type = (out_et == ov::element::f16) ? "half" : "float";

        std::ostringstream src;
        src << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        // IQ3_XXS codebook tables (inline __constant; identical to reference).
        src << "__constant uint IQ3_GRID[256] = {\n";
        for (int i = 0; i < 256; ++i) {
            src << "0x" << std::hex << kIq3xxsGrid[i] << std::dec << "u,";
            if ((i & 7) == 7) src << "\n";
        }
        src << "};\n";
        src << "__constant uchar IQ3_SIGNS[128] = {\n";
        for (int i = 0; i < 128; ++i) {
            src << static_cast<int>(kSignsIq2xs[i]) << ",";
            if ((i & 15) == 15) src << "\n";
        }
        src << "};\n";
        src << "__constant uchar IQ3_MASK[8] = {1,2,4,8,16,32,64,128};\n";

        // One work-item per output element (m, n). gws = M*N (via "b*f*y*x").
        src << "__kernel void iq3xxs_fc(\n"
               "    const __global " << act_type << "* act,\n"
               "    const __global uchar* weights,\n"
               "    __global " << out_type << "* output) {\n"
               "    const uint gid = (uint)get_global_id(0);\n"
               "    if (gid >= (uint)GLOBAL_WORKSIZE[0]) return;\n"
               "    const uint n = gid % (uint)(IQ_N);\n"
               "    const uint m = gid / (uint)(IQ_N);\n"
               "    const __global uchar* wrow = weights + (ulong)n * IQ_BYTES_PER_ROW;\n"
               "    const __global " << act_type << "* arow = act + (ulong)m * IQ_K;\n"
               "    float acc = 0.0f;\n"
               "    for (uint blk = 0; blk < IQ_BLOCKS_PER_ROW; ++blk) {\n"
               "        const __global uchar* bd = wrow + (ulong)blk * 98u;\n"
               "        const ushort dh = (ushort)bd[0] | ((ushort)bd[1] << 8);\n"
               "        const float d = (float)as_half(dh);\n"
               "        const __global uchar* qs = bd + 2;\n"
               "        const __global uchar* ss = qs + 64;\n"  // QK_K/4
               "        const uint kbase = blk * 256u;\n"
               "        for (uint ib32 = 0; ib32 < 8u; ++ib32) {\n"
               "            const __global uchar* s4 = ss + 4u * ib32;\n"
               "            const uint aux = (uint)s4[0] | ((uint)s4[1] << 8) | ((uint)s4[2] << 16) | ((uint)s4[3] << 24);\n"
               "            const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
               "            const __global uchar* qsb = qs + ib32 * 8u;\n"
               "            uint oi = ib32 * 32u;\n"
               "            for (uint l = 0; l < 4u; ++l) {\n"
               "                const uchar signs = IQ3_SIGNS[(aux >> (7u * l)) & 127u];\n"
               "                const uint g1 = IQ3_GRID[qsb[2u*l + 0u]];\n"
               "                const uint g2 = IQ3_GRID[qsb[2u*l + 1u]];\n"
               "                for (uint j = 0; j < 4u; ++j) {\n"
               "                    const float gv = (float)((g1 >> (8u*j)) & 0xffu);\n"
               "                    const float sgn = (signs & IQ3_MASK[j]) ? -1.0f : 1.0f;\n"
               "                    acc += (db * gv * sgn) * (float)arow[kbase + oi]; ++oi;\n"
               "                }\n"
               "                for (uint j = 0; j < 4u; ++j) {\n"
               "                    const float gv = (float)((g2 >> (8u*j)) & 0xffu);\n"
               "                    const float sgn = (signs & IQ3_MASK[j + 4u]) ? -1.0f : 1.0f;\n"
               "                    acc += (db * gv * sgn) * (float)arow[kbase + oi]; ++oi;\n"
               "                }\n"
               "            }\n"
               "        }\n"
               "    }\n"
               "    output[gid] = (" << out_type << ")acc;\n"
               "}\n";

        std::ostringstream opts;
        opts << "-DIQ_N=" << N << " -DIQ_K=" << K
             << " -DIQ_BLOCKS_PER_ROW=" << blocks_per_row
             << " -DIQ_BYTES_PER_ROW=" << bytes_per_row;

        // Reorder activation to dense bfyx to guarantee contiguous [M, K] layout.
        auto act_reorder_name = layerName + "_act_bfyx";
        auto act_reorder = cldnn::reorder(act_reorder_name,
                                          inputs[0],
                                          cldnn::format::bfyx,
                                          cldnn::element_type_to_data_type(in_et));
        p.add_primitive(*op, act_reorder);

        std::vector<cldnn::custom_gpu_primitive::arg_desc> args(3);
        args[0].type = cldnn::custom_gpu_primitive::arg_input;
        args[0].index = 0;  // activation
        args[1].type = cldnn::custom_gpu_primitive::arg_input;
        args[1].index = 1;  // compressed weights (u8)
        args[2].type = cldnn::custom_gpu_primitive::arg_output;
        args[2].index = 0;

        cldnn::layout out_layout(op->get_output_partial_shape(0),
                                 w_dtype,
                                 cldnn::format::get_default_format(op->get_output_partial_shape(0).size()));

        // Clone the op (with Parameter inputs) so the custom primitive can run
        // the op's shape inference for dynamic output shapes (M is dynamic in
        // LLM decode). A null op here would crash update_output_shape().
        ov::OutputVector clone_inputs;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            clone_inputs.emplace_back(
                std::make_shared<ov::op::v0::Parameter>(op->get_input_element_type(i),
                                                        op->get_input_partial_shape(i)));
        }
        std::shared_ptr<ov::Node> op_clone = op->clone_with_new_inputs(clone_inputs);

        auto custom = cldnn::custom_gpu_primitive(
            layerName,
            {cldnn::input_info(act_reorder_name), inputs[1]},
            {src.str()},
            "iq3xxs_fc",
            args,
            opts.str(),
            {out_layout},
            /*gws*/ {1},
            /*lws*/ {},
            /*op*/ op_clone,
            /*calcWgDimInputIdx*/ -1,
            /*globalSizeRules*/ {"b*f*y*x"},
            /*localSizeRules*/ {});

        p.add_primitive(*op, custom);
        return;
    }

    // ----------------------------------------------------------------------
    // Fallback path (step 1): host dequant -> dense weights -> fully_connected.
    // ----------------------------------------------------------------------

    // Fetch the opaque compressed weight blob from the u8 Constant input.
    auto weights_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(weights_const != nullptr,
                    "[GPU] IQ3XXSLinear expects a Constant compressed-weights input");
    const uint8_t* compressed = weights_const->get_data_ptr<uint8_t>();

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
