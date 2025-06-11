// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/all_reduce.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace ov {
namespace intel_gpu {
namespace op {
AllReduce::AllReduce(const size_t& world_size, const size_t& rank)
    : ov::op::Op(),
      m_world_size(world_size),
      m_world_rank(rank) {
    //validate_and_infer_types();
}

AllReduce::AllReduce(const Output<Node>& input, const size_t& world_size, const size_t& rank)
    : ov::op::Op({input}),
      m_world_size(world_size),
      m_world_rank(rank) {
    set_output_size(1);
    m_output_type = input.get_element_type(); // all-reduce does not change output types
    validate_and_infer_types();
}

bool AllReduce::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void AllReduce::validate_and_infer_types_fallback() {
    set_output_size(m_world_size);
    auto original_fc_out = get_input_source_output(0).get_partial_shape();
    std::vector<ov::PartialShape> p_shapes(m_world_size, original_fc_out);
    for (size_t i = 0; i < p_shapes.size(); i++)
        set_output_type(i, m_output_type, p_shapes[i]);
}

void AllReduce::validate_and_infer_types() {
    auto original_fc_out = get_input_source_output(0).get_partial_shape();
    std::vector<ov::PartialShape> p_shapes(1, original_fc_out);
    for (size_t i = 0; i < p_shapes.size(); i++)
        set_output_type(i, m_output_type, p_shapes[i]);
}

std::shared_ptr<Node> AllReduce::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    OPENVINO_ASSERT(new_args.size() == 1,
                    "Unable to clone AllReduce with name ",
                    this->get_friendly_name(),
                    ", which should only has 1 input!");
    return std::make_shared<AllReduce>(new_args[0], m_world_size, m_world_rank);
}

std::vector<ov::PartialShape> shape_infer(const AllReduce* op, std::vector<ov::PartialShape> input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    for (size_t i = 0; i < op->get_output_size(); i++) {
        out_shapes.push_back(input_shapes[0]);
    }
    return out_shapes;
}
}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
