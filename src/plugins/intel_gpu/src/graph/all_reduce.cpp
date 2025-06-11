// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/op/all_reduce.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include <all_reduce_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(all_reduce)

all_reduce_inst::typed_primitive_inst(network& network, const all_reduce_node& node) :
    parent(network, node, !node.can_be_optimized() && (node.get_output_layout().is_static() || node.get_output_layout().has_upper_bound())) {
}

layout all_reduce_inst::calc_output_layout(const all_reduce_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> all_reduce_inst::calc_output_layouts(all_reduce_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<all_reduce>();
    ov::intel_gpu::op::AllReduce op(impl_param.w_size, impl_param.w_rank);
    op.set_output_size(desc->num_outputs);

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>()};

    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < desc->num_outputs; i++) {
        auto out_type = impl_param.get_input_layout(0).data_type;
        out_layouts.push_back(layout(output_shapes[i], out_type, impl_param.get_output_layout(i).format));
    }

    return out_layouts;
}

template std::vector<layout> all_reduce_inst::calc_output_layouts<ov::PartialShape>(all_reduce_node const& node, const kernel_impl_params& impl_param);
std::string all_reduce_inst::to_string(const all_reduce_node& node) {
    auto node_info = node.desc_to_json();
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void all_reduce_inst::on_execute() {
    update_output_memory();
}

void all_reduce_inst::update_output_memory() {
    if (!can_be_optimized()) {
        if (_node->get_preferred_impl_type() == impl_types::ocl) {
            if (_outputs.size() == 2) {
                // All gather need new shape output for concat
                _outputs[1] = input_memory_ptr();
            } else {
                // All reduce will use input as addition's output
                _outputs[0] = input_memory_ptr();
            }
        } else {
            auto my_rank = get_impl_params()->w_rank;
            _outputs[my_rank] = input_memory_ptr();
        }
        return;
    }
    // do nothing for now
}
} // namespace cldnn