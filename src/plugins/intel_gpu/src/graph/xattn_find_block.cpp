// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/xattn_find_block.hpp"

#include "json_object.h"
#include "primitive_inst.h"
#include "xattn_find_block_inst.h"

#include <sstream>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(xattn_find_block);

std::string typed_primitive_inst<xattn_find_block>::to_string(const xattn_find_block_node& node) {
    const auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    json_composite info;
    info.add("heads_num", desc->heads_num);
    info.add("head_size", desc->head_size);
    info.add("q_len", desc->q_len);
    info.add("q_stride", desc->q_stride);
    info.add("k_stride", desc->k_stride);
    info.add("q_stride_pad", desc->q_stride_pad);
    info.add("q_block_pad", desc->q_block_pad);
    info.add("k_block_pad", desc->k_block_pad);
    info.add("causal_start_index", desc->causal_start_index);
    info.add("thresh", desc->thresh);
    info.add("stride", desc->stride);
    info.add("block_size", desc->block_size);
    info.add("is_causal", desc->is_causal);
    info.add("use_int8", desc->use_int8);
    node_info->add("xattn_find_block_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
