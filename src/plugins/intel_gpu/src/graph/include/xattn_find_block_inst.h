// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/xattn_find_block.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<xattn_find_block> : public typed_program_node_base<xattn_find_block> {
    using parent = typed_program_node_base<xattn_find_block>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using xattn_find_block_node = typed_program_node<xattn_find_block>;

template <>
class typed_primitive_inst<xattn_find_block> : public typed_primitive_inst_base<xattn_find_block> {
    using parent = typed_primitive_inst_base<xattn_find_block>;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const xattn_find_block_node&, const kernel_impl_params& impl_params) {
        return {impl_params.get_output_layout(0)};
    }

    static layout calc_output_layout(const xattn_find_block_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }

    static std::string to_string(const xattn_find_block_node& node) {
        auto node_info = node.desc_to_json();
        auto& desc = node.get_primitive();
        node_info->add("heads_num", desc->heads_num);
        node_info->add("head_size", desc->head_size);
        node_info->add("q_len", desc->q_len);
        node_info->add("q_stride", desc->q_stride);
        node_info->add("k_stride", desc->k_stride);
        node_info->add("q_stride_pad", desc->q_stride_pad);
        node_info->add("q_block_pad", desc->q_block_pad);
        node_info->add("k_block_pad", desc->k_block_pad);
        node_info->add("causal_start_index", desc->causal_start_index);
        node_info->add("thresh", desc->thresh);
        node_info->add("stride", desc->stride);
        node_info->add("block_size", desc->block_size);
        node_info->add("is_causal", desc->is_causal);
        node_info->add("use_int8", desc->use_int8);
        return node_info->to_string();
    }

    typed_primitive_inst(network& network, const xattn_find_block_node& node)
        : parent(network, node) {}

    typed_primitive_inst(network& network) : parent(network) {}
};

using xattn_find_block_inst = typed_primitive_inst<xattn_find_block>;

}  // namespace cldnn
