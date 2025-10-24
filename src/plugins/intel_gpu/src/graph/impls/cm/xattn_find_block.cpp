// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xattn_find_block.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {

constexpr auto get_find_block_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

constexpr uint32_t SG_M = 4;
constexpr uint32_t SG_N = 8;
constexpr uint32_t BLOCK_SG_M = 64;
constexpr uint32_t BLOCK_SG_N = 32;
constexpr uint32_t BLOCK_SHARE_MAX = BLOCK_SG_N * SG_N;
constexpr uint32_t KV_BLOCK_SIZE = 256;

class XAttnFindBlockGenerator : public KernelGenerator {
public:
    XAttnFindBlockGenerator() : KernelGenerator("xattn_find_block") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_find_block_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<xattn_find_block>();

        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(desc->head_size)) / static_cast<float>(desc->stride);
        int scale_factor_i;
        std::memcpy(static_cast<void*>(&scale_factor_i), &scale_factor, sizeof(scale_factor));

        jit.make("STRIDE", desc->stride);
        jit.make("HQ", desc->heads_num);
        jit.make("HK", desc->heads_num);
        jit.make("HEAD_SIZE", desc->head_size);
        jit.make("SG_M", SG_M);
        jit.make("SG_N", SG_N);
        jit.make("BLOCK_SG_M", BLOCK_SG_M);
        jit.make("BLOCK_SG_N", BLOCK_SG_N);
        jit.make("BLOCK_SIZE", desc->block_size);
        jit.make("KV_BLOCK_SIZE", KV_BLOCK_SIZE);
        jit.add(make_jit_constant("INV_S", scale_factor_i));
        jit.make("BLOCK_SHARE_MAX", BLOCK_SHARE_MAX);
        jit.make("WALK_HQ", 1);
        jit.make("IS_CAUSAL", desc->is_causal ? 1 : 0);
        jit.make("USE_INT8", desc->use_int8 ? 1 : 0);
        const uint32_t head_size_key = desc->use_int8 ? desc->head_size + 4 : desc->head_size;
        jit.make("HEAD_SIZE_KEY", head_size_key);
        jit.make("SOFTMAX_TYPE", "float");

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 3});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 4});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 5});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 6});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            auto desc = params.typed_desc<xattn_find_block>();

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->q_block_pad, desc->heads_num, 1};
            wgs.local = {1, 1, 1};

            const uint32_t tokens_per_block = desc->block_size / desc->stride;
            const uint32_t q_block = cldnn::ceil_div(desc->q_stride, tokens_per_block);
            const uint32_t k_block = cldnn::ceil_div(desc->k_stride, tokens_per_block);

            auto& scalars = kd.params.scalars;
            std::vector<uint32_t> scalar_values = {desc->q_len,
                                                   desc->q_stride,
                                                   desc->q_stride_pad,
                                                   desc->q_block_pad,
                                                   desc->k_block_pad,
                                                   (k_block >= q_block) ? k_block - q_block : 0u};
            scalars.resize(scalar_values.size() + 1);

            for (size_t i = 0; i < scalar_values.size(); ++i) {
                scalars[i].t = ScalarDescriptor::Types::UINT32;
                scalars[i].v.u32 = scalar_values[i];
            }

            scalars[scalar_values.size()].t = ScalarDescriptor::Types::FLOAT32;
            scalars[scalar_values.size()].v.f32 = desc->thresh;
        }};
    }
};

class XAttnFindBlockCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::XAttnFindBlockCmImpl)

    Stage::Ptr find_block = make_stage<XAttnFindBlockGenerator>();

    XAttnFindBlockCmImpl() : PrimitiveImplCM(XAttnFindBlockImplementationManager::get_type_info_static()) {}
    XAttnFindBlockCmImpl(const program_node& node, const RuntimeParams& params) : XAttnFindBlockCmImpl() {
        add_stage(find_block, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<XAttnFindBlockCmImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> XAttnFindBlockImplementationManager::create_impl(const program_node& node,
                                                                                 const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<xattn_find_block>());
    try {
        return std::make_unique<XAttnFindBlockCmImpl>(node, params);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to create XAttnFindBlockCmImpl: ", e.what());
    }
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::XAttnFindBlockCmImpl)
