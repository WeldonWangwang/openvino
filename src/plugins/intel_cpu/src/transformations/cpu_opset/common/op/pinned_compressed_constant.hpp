// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/compressed_constant.hpp"

namespace ov::intel_cpu {

/// \brief A plugin-private "wrapper" op that shields a CompressedConstant from
///        standard OV transformation passes.
///
/// \details
/// The problem: many OV graph transformations use `pattern::wrap_type<v0::Constant>()`
/// to match Constants, and then call APIs like `cast_vector<float>()` or
/// `get_data_ptr<float>()`. When a `CompressedConstant` (which inherits from
/// `v0::Constant`) is matched, these calls interpret the compressed blob as f32
/// data, causing buffer over-reads and crashes.
///
/// The solution: at the very beginning of the CPU plugin transformation pipeline,
/// a `CompressedConstantPin` pass replaces every `CompressedConstant` node with a
/// `PinnedCompressedConstant`. Because this class inherits from `ov::op::Op` (NOT
/// from `v0::Constant`), it is invisible to `wrap_type<v0::Constant>()` patterns.
///
/// It has zero inputs and one output. The output reports the CC's logical face
/// (e.g. f32 [N, K]) so that downstream MatMul / FullyConnected shape inference
/// continues to work correctly.
///
/// The wrapped CompressedConstant is held as a shared_ptr, so the compressed blob
/// remains alive (zero-copy) throughout the compilation pipeline.
///
/// The `FullyConnected` ctor and `Input` node both know how to unwrap this op to
/// retrieve the underlying CC and its compressed storage.
class PinnedCompressedConstant : public ov::op::Op {
public:
    OPENVINO_OP("PinnedCompressedConstant", "cpu_plugin_opset");

    PinnedCompressedConstant() = default;

    /// \brief Construct a wrapper around a CompressedConstant.
    ///
    /// \param cc  The underlying CompressedConstant to shield from pattern passes.
    explicit PinnedCompressedConstant(std::shared_ptr<ov::op::util::CompressedConstant> cc);

    ~PinnedCompressedConstant() override;

    // ----- Node interface -----

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    /// \brief Never fold this node.
    bool can_constant_fold(const ov::OutputVector& inputs) const override {
        return false;
    }

    /// \brief No evaluate support.
    bool has_evaluate() const override {
        return false;
    }

    /// \brief No evaluate support.
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        return false;
    }

    // ----- CompressedConstant access -----

    /// \brief Returns the underlying CompressedConstant.
    std::shared_ptr<ov::op::util::CompressedConstant> get_compressed_constant() const {
        return m_wrapped_cc;
    }

    /// \brief Shortcut: quant type.
    ov::op::util::CompressedConstant::QuantType get_quant_type() const {
        return m_wrapped_cc->get_quant_type();
    }

    /// \brief Shortcut: logical shape.
    const ov::Shape& get_logical_shape() const {
        return m_wrapped_cc->get_logical_shape();
    }

    /// \brief Shortcut: logical element type.
    const ov::element::Type& get_logical_element_type() const {
        return m_wrapped_cc->get_logical_element_type();
    }

    /// \brief Shortcut: compressed blob pointer.
    const void* get_compressed_data_ptr() const {
        return m_wrapped_cc->get_compressed_data_ptr();
    }

    /// \brief Shortcut: compressed byte count.
    size_t get_compressed_byte_size() const {
        return m_wrapped_cc->get_compressed_byte_size();
    }

    /// \brief Switch output from logical face (f32 [N,K]) to storage view (u8 [N_bytes]).
    ///
    /// Called after all transformations are done (ConvertMatMulToFC etc.) and before
    /// the CPU graph builder runs. After this call, validate_and_infer_types will
    /// report u8 [N_bytes] which matches the actual memory allocation.
    void switch_to_storage_view() {
        m_use_storage_view = true;
        validate_and_infer_types();
    }

private:
    std::shared_ptr<ov::op::util::CompressedConstant> m_wrapped_cc;
    bool m_use_storage_view = false;
};

}  // namespace ov::intel_cpu
