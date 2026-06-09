// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace op {
namespace util {

/// \brief A Constant subclass that holds block-structured compressed weight data
///        (e.g. GGUF IQ3_XXS, IQ2_S, IQ4_XS, IQ3_S, IQ2_XS, Q3_K, Q5_K).
///
/// \details
/// `CompressedConstant` is the cornerstone of the unified GGUF weight pipeline.
/// It presents a *logical* face to the graph (`f32` element type, `[N, K]` shape)
/// so that downstream standard ops (`MatMul`, `FullyConnected`, fusions, etc.) see
/// a normal weight tensor and run their shape inference / pattern matching as usual.
///
/// Internally, the storage is the raw compressed blob (typically `u8 [total_bytes]`),
/// and the actual dequantization happens inside plugin kernels at execution time.
///
/// Because we inherit publicly from `ov::op::v0::Constant`, every existing pass that
/// does `is_type<Constant>(node)` / `as_type_ptr<Constant>(node)` / `is_on_path<Constant>(...)`
/// will transparently accept us via RTTI parent chain. Passes that try to *read element data*
/// from us (e.g. `cast_vector<float>()`, `get_data_ptr<float>()`) must be detected and
/// either skipped (rt_info markers) or have an explicit early-out.
///
/// \par Defense strategy
/// 1. The caller is expected to apply `ov::disable_fp16_compression(node)` and
///    `ov::enable_keep_const_precision(node)` immediately after construction. Those
///    helpers live in the `transformations` library, which `core` cannot depend on,
///    so the rt_info setup is done by callers (see `apply_compressed_constant_defenses()`
///    helper in `transformations/rt_info/compressed_constant_defenses.hpp`).
///    → Protects against `compress_float_constants` (C-2) and `ConvertPrecision` (C-3)
/// 2. `get_byte_size()` is non-virtual but the base implementation uses Constant's private
///    `m_shape` (== storage shape `[N_bytes]`), so it returns the correct compressed size
///    → SharedOpsOptimization (M-1) memcmp works correctly
/// 3. `evaluate()` / `has_evaluate()` overridden to return `false`
///    → ConstantFolding cannot fold us into a real f32 buffer
/// 4. Plugin entry points (CPU `Input::Input` ctor, GPU `create_data`) need an early-out
///    that uses the storage view instead of the logical view
/// 5. `FlushFP32SubnormalsToZero` (C-4) needs an explicit `is_type<CompressedConstant>` skip
///    because it has no rt_info-based opt-out mechanism
///
/// \par Output dual-faced API
/// | API                              | Returns                | Source                  |
/// |----------------------------------|------------------------|-------------------------|
/// | `get_output_element_type(0)`     | logical (`f32`)        | `set_output_type` in vai|
/// | `get_output_partial_shape(0)`    | logical (`[N,K]`)      | `set_output_type` in vai|
/// | `get_element_type()`             | logical (`f32`)        | Node base               |
/// | `get_shape()`                    | logical (`[N,K]`)      | Node base               |
/// | `get_byte_size()`                | storage (`N_bytes`)    | Constant base, private m_shape |
/// | `get_data_ptr()` (non-templated) | storage blob start     | Constant base           |
/// | `get_logical_shape()`            | logical (`[N,K]`)      | this class              |
/// | `get_logical_element_type()`     | logical (`f32`)        | this class              |
/// | `get_quant_type()`               | quantization scheme    | this class              |
/// | `get_compressed_byte_size()`     | storage (`N_bytes`)    | this class (alias)      |
/// | `get_compressed_data_ptr()`      | storage blob start     | this class (alias)      |
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API CompressedConstant : public ov::op::v0::Constant {
public:
    OPENVINO_OP("CompressedConstant", "util", ov::op::v0::Constant);

    /// \brief Identifier for the underlying compressed weight format.
    ///
    /// Each entry corresponds to one block-structured GGUF quantization type.
    /// Block sizes and bytes/block are documented per entry.
    enum class QuantType : uint8_t {
        IQ3_XXS,  ///< 256 elements per block, 98 bytes/block  (~3.0625 bpw)
        IQ2_S,    ///< 256 elements per block, 96 bytes/block  (~3.0 bpw including 16B half-block)
        IQ4_XS,   ///< 256 elements per block, 136 bytes/block (~4.25 bpw)
        IQ3_S,    ///< 256 elements per block, 110 bytes/block (~3.4375 bpw)
        IQ2_XS,   ///< 256 elements per block, 74 bytes/block  (~2.3125 bpw)
        Q3_K,     ///< 256 elements per block, 110 bytes/block (~3.4375 bpw)
        Q5_K,     ///< 256 elements per block, 176 bytes/block (~5.5 bpw)
    };

    CompressedConstant() = default;

    /// \brief Construct a CompressedConstant from a raw compressed blob.
    ///
    /// \param raw_data       Pointer to the compressed blob. The bytes will be copied into
    ///                       an internal AlignedBuffer owned by the underlying Constant.
    /// \param compressed_bytes  Total number of bytes in the compressed blob.
    /// \param logical_shape  The shape this constant logically represents (e.g. `[N, K]`).
    /// \param logical_type   The element type this constant logically represents (e.g. `f32`).
    /// \param quant_type     Identifier for the compression scheme.
    CompressedConstant(const void* raw_data,
                       size_t compressed_bytes,
                       const ov::Shape& logical_shape,
                       const ov::element::Type& logical_type,
                       QuantType quant_type);

    /// \brief Construct a CompressedConstant from an existing ov::Tensor holding the blob.
    ///
    /// \param compressed_blob  Tensor containing the compressed bytes. Its element_type/shape
    ///                         are ignored; only the raw byte buffer and its size are used.
    /// \param logical_shape    The shape this constant logically represents.
    /// \param logical_type     The element type this constant logically represents.
    /// \param quant_type       Identifier for the compression scheme.
    CompressedConstant(const ov::Tensor& compressed_blob,
                       const ov::Shape& logical_shape,
                       const ov::element::Type& logical_type,
                       QuantType quant_type);

    /// \brief Construct from a shared AlignedBuffer (zero-copy take-ownership path).
    ///
    /// \param compressed_buffer  Buffer holding the raw compressed bytes. Lifetime is managed.
    /// \param logical_shape      The shape this constant logically represents.
    /// \param logical_type       The element type this constant logically represents.
    /// \param quant_type         Identifier for the compression scheme.
    CompressedConstant(const std::shared_ptr<ov::AlignedBuffer>& compressed_buffer,
                       const ov::Shape& logical_shape,
                       const ov::element::Type& logical_type,
                       QuantType quant_type);

    ~CompressedConstant() override;

    // ---------------------------------------------------------------------------------------
    // Required Node / Op overrides
    // ---------------------------------------------------------------------------------------

    /// \brief Reports the *logical* element_type / shape, not the storage view.
    ///
    /// This is what makes downstream MatMul / FullyConnected shape inference work.
    void validate_and_infer_types() override;

    /// \brief Standard shared_ptr clone with new inputs (Constant subclass takes no inputs,
    ///        so `new_args` must be empty).
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    /// \brief Serialize logical_shape, logical_element_type, quant_type as attributes.
    ///
    /// The raw compressed bytes are persisted by the base `Constant::visit_attributes`
    /// (which writes the storage view: `u8 [N_bytes]`).
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    /// \brief Always returns false. We must never be folded into a real f32 tensor.
    bool can_constant_fold(const ov::OutputVector& input_values) const override {
        return false;
    }

    /// \brief Always returns false. We do not implement element-wise evaluation.
    bool has_evaluate() const override {
        return false;
    }

    /// \brief Always returns false. Refuse to materialize the logical tensor.
    bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const override {
        return false;
    }

    // ---------------------------------------------------------------------------------------
    // CompressedConstant-specific API
    // ---------------------------------------------------------------------------------------

    /// \brief Returns the quantization scheme identifier.
    QuantType get_quant_type() const {
        return m_quant_type;
    }

    /// \brief Returns the logical shape that downstream ops see (e.g. `[N, K]`).
    const ov::Shape& get_logical_shape() const {
        return m_logical_shape;
    }

    /// \brief Returns the logical element type that downstream ops see (e.g. `f32`).
    const ov::element::Type& get_logical_element_type() const {
        return m_logical_type;
    }

    /// \brief Returns the size of the compressed blob in bytes.
    ///
    /// Equivalent to `Constant::get_byte_size()` because the base class stores the data
    /// as `u8 [N_bytes]`. Provided as a name-clearer alias for plugin code.
    size_t get_compressed_byte_size() const {
        return Constant::get_byte_size();
    }

    /// \brief Returns a pointer to the start of the compressed blob.
    ///
    /// Equivalent to `Constant::get_data_ptr()` (non-templated). Provided as a name-clearer
    /// alias for plugin code that consumes the compressed bytes directly.
    const void* get_compressed_data_ptr() const {
        return Constant::get_data_ptr();
    }

    // ---------------------------------------------------------------------------------------
    // Conversion helpers (string<->QuantType) for serialization and diagnostics
    // ---------------------------------------------------------------------------------------

    static std::string quant_type_to_string(QuantType qt);
    static QuantType quant_type_from_string(const std::string& s);

private:
    /// Once the base Constant ctor has been called with the storage view,
    /// remember the logical shape/type/quant tag so we can answer queries
    /// and re-establish the output descriptor in validate_and_infer_types().
    ov::Shape m_logical_shape{};
    ov::element::Type m_logical_type{};
    QuantType m_quant_type{QuantType::IQ3_XXS};
};

/// \brief Convenience predicate. Equivalent to `ov::is_type<CompressedConstant>(node)`.
inline bool is_compressed_constant(const std::shared_ptr<const ov::Node>& node) {
    return ov::is_type<CompressedConstant>(node);
}

/// \brief Convenience predicate. Equivalent to `ov::is_type<CompressedConstant>(node)`.
inline bool is_compressed_constant(const ov::Node* node) {
    return ov::is_type<CompressedConstant>(node);
}

}  // namespace util
}  // namespace op
}  // namespace ov
