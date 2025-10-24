// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/runtime/layout.hpp"
#include "registry/implementation_manager.hpp"
#include "xattn_find_block_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

struct XAttnFindBlockImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::xattn_find_block")
    explicit XAttnFindBlockImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<xattn_find_block>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<xattn_find_block>());

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();

        if (!check_cm_jit_support(engine, config) || !config.get_use_cm()) {
            return false;
        }

        if (node.is_dynamic()) {
            return false;
        }

        const auto& input0 = node.get_input_layout(0);
        const auto& input1 = node.get_input_layout(1);
        const auto& output = node.get_output_layout(0);

        if (input0.data_type != ov::element::f32 || input1.data_type != ov::element::f32) {
            return false;
        }

        if (output.data_type != ov::element::u8 && output.data_type != ov::element::boolean) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
