// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"

#include "util.hpp"

namespace ov {
namespace intel_gpu {
namespace op {
class AllReduce : public ov::op::Op {
public:
    OPENVINO_OP("AllReduce", "gpu_opset");
    AllReduce() = default;
    AllReduce(const size_t& world_size, const size_t& rank);
    AllReduce(const Output<Node>& input, const size_t& world_size, const size_t& world_rank);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    const size_t& get_world_size() const {
        return m_world_size;
    }
    const size_t& get_world_rank() const {
        return m_world_size;
    }
    void validate_and_infer_types_fallback();

protected:
    size_t m_world_size;
    size_t m_world_rank;
    ov::element::Type m_output_type;
};

std::vector<ov::PartialShape> shape_infer(const AllReduce* op, std::vector<ov::PartialShape> input_shapes);
}   // namespace op
}   // namespace intel_gpu
}   // namespace ov