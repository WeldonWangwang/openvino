// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/pinned_compressed_constant.hpp"

#include "openvino/core/node.hpp"
#include "openvino/op/util/compressed_constant.hpp"

namespace ov::intel_cpu {

PinnedCompressedConstant::PinnedCompressedConstant(
    std::shared_ptr<ov::op::util::CompressedConstant> cc)
    : ov::op::Op()
    , m_wrapped_cc(std::move(cc)) {
    // No graph inputs — this is a source node (like Constant).
    // Set the output type based on the CC's logical face.
    constructor_validate_and_infer_types();
}

PinnedCompressedConstant::~PinnedCompressedConstant() = default;

void PinnedCompressedConstant::validate_and_infer_types() {
    // Always report the logical face (f32 [N, K]) so that:
    //   1. Shape inference during transforms works correctly
    //   2. FC's port negotiation sees the expected 2-D weight shape
    //   3. Edge descriptors are consistent throughout the pipeline
    //
    // The actual compressed blob is accessed by the executor via
    // attrs.compressedDataPtr, completely bypassing edge memory.
    // Input::cloneCompressedBlob() creates a zero-copy Memory wrapper
    // that points to the compressed buffer with a logical-face descriptor.
    if (m_wrapped_cc) {
        set_output_type(0,
                        m_wrapped_cc->get_logical_element_type(),
                        m_wrapped_cc->get_logical_shape());
    }
}

std::shared_ptr<ov::Node> PinnedCompressedConstant::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<PinnedCompressedConstant>(m_wrapped_cc);
}

bool PinnedCompressedConstant::visit_attributes(ov::AttributeVisitor& visitor) {
    // Serialize enough to reconstruct: quant_type, logical_shape, logical_type.
    // The actual compressed blob lives in the wrapped CC and would be serialized
    // separately if needed (IR export path). For now, this is plugin-internal only.
    if (m_wrapped_cc) {
        std::string qt_str = ov::op::util::CompressedConstant::quant_type_to_string(
            m_wrapped_cc->get_quant_type());
        visitor.on_attribute("quant_type", qt_str);

        auto logical_shape = m_wrapped_cc->get_logical_shape();
        visitor.on_attribute("logical_shape", logical_shape);

        auto logical_type_str = m_wrapped_cc->get_logical_element_type().to_string();
        visitor.on_attribute("logical_element_type", logical_type_str);

        auto compressed_bytes = static_cast<int64_t>(m_wrapped_cc->get_compressed_byte_size());
        visitor.on_attribute("compressed_bytes", compressed_bytes);
    }
    return true;
}

}  // namespace ov::intel_cpu
