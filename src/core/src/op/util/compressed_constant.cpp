// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/compressed_constant.hpp"

#include <stdexcept>
#include <unordered_map>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
namespace op {
namespace util {

namespace {

// ---------------------------------------------------------------------------------------
// QuantType <-> string mapping (also used by visit_attributes for IR serialization)
// ---------------------------------------------------------------------------------------

const std::unordered_map<CompressedConstant::QuantType, std::string>& qt_to_str_map() {
    static const std::unordered_map<CompressedConstant::QuantType, std::string> m{
        {CompressedConstant::QuantType::IQ3_XXS, "IQ3_XXS"},
        {CompressedConstant::QuantType::IQ2_S, "IQ2_S"},
        {CompressedConstant::QuantType::IQ4_XS, "IQ4_XS"},
        {CompressedConstant::QuantType::IQ3_S, "IQ3_S"},
        {CompressedConstant::QuantType::IQ2_XS, "IQ2_XS"},
        {CompressedConstant::QuantType::Q3_K, "Q3_K"},
        {CompressedConstant::QuantType::Q5_K, "Q5_K"},
    };
    return m;
}

const std::unordered_map<std::string, CompressedConstant::QuantType>& str_to_qt_map() {
    static const std::unordered_map<std::string, CompressedConstant::QuantType> m{
        {"IQ3_XXS", CompressedConstant::QuantType::IQ3_XXS},
        {"IQ2_S", CompressedConstant::QuantType::IQ2_S},
        {"IQ4_XS", CompressedConstant::QuantType::IQ4_XS},
        {"IQ3_S", CompressedConstant::QuantType::IQ3_S},
        {"IQ2_XS", CompressedConstant::QuantType::IQ2_XS},
        {"Q3_K", CompressedConstant::QuantType::Q3_K},
        {"Q5_K", CompressedConstant::QuantType::Q5_K},
    };
    return m;
}

// ---------------------------------------------------------------------------------------
// Storage helpers: every public ctor must place the raw blob into the base Constant as
// u8[N_bytes]. These small helpers keep the construction discipline in one place.
// ---------------------------------------------------------------------------------------

constexpr const char* k_storage_type_mismatch_msg =
    "CompressedConstant: internal invariant violated. The base Constant must be initialized "
    "with element::u8 and shape == [N_bytes].";

void verify_storage_invariant(const ov::op::v0::Constant& base) {
    OPENVINO_ASSERT(base.Constant::get_byte_size() != 0, "CompressedConstant: storage blob is empty.");
}

}  // namespace

// ---------------------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------------------

CompressedConstant::CompressedConstant(const void* raw_data,
                                       size_t compressed_bytes,
                                       const ov::Shape& logical_shape,
                                       const ov::element::Type& logical_type,
                                       QuantType quant_type)
    : ov::op::v0::Constant(ov::element::u8, ov::Shape{compressed_bytes}, raw_data),
      m_logical_shape(logical_shape),
      m_logical_type(logical_type),
      m_quant_type(quant_type) {
    OPENVINO_ASSERT(raw_data != nullptr || compressed_bytes == 0,
                    "CompressedConstant: raw_data is null but compressed_bytes > 0.");
    OPENVINO_ASSERT(!m_logical_shape.empty(),
                    "CompressedConstant: logical_shape must not be empty.");
    OPENVINO_ASSERT(m_logical_type != ov::element::dynamic,
                    "CompressedConstant: logical_type must be a concrete element type.");
    verify_storage_invariant(*this);
    constructor_validate_and_infer_types();
}

CompressedConstant::CompressedConstant(const ov::Tensor& compressed_blob,
                                       const ov::Shape& logical_shape,
                                       const ov::element::Type& logical_type,
                                       QuantType quant_type)
    : ov::op::v0::Constant(ov::element::u8,
                           ov::Shape{compressed_blob.get_byte_size()},
                           compressed_blob.data()),
      m_logical_shape(logical_shape),
      m_logical_type(logical_type),
      m_quant_type(quant_type) {
    OPENVINO_ASSERT(compressed_blob && compressed_blob.get_byte_size() > 0,
                    "CompressedConstant: compressed_blob is empty.");
    OPENVINO_ASSERT(!m_logical_shape.empty(),
                    "CompressedConstant: logical_shape must not be empty.");
    OPENVINO_ASSERT(m_logical_type != ov::element::dynamic,
                    "CompressedConstant: logical_type must be a concrete element type.");
    verify_storage_invariant(*this);
    constructor_validate_and_infer_types();
}

CompressedConstant::CompressedConstant(const std::shared_ptr<ov::AlignedBuffer>& compressed_buffer,
                                       const ov::Shape& logical_shape,
                                       const ov::element::Type& logical_type,
                                       QuantType quant_type)
    : ov::op::v0::Constant(ov::element::u8,
                           ov::Shape{compressed_buffer ? compressed_buffer->size() : 0u},
                           compressed_buffer),
      m_logical_shape(logical_shape),
      m_logical_type(logical_type),
      m_quant_type(quant_type) {
    OPENVINO_ASSERT(compressed_buffer && compressed_buffer->size() > 0,
                    "CompressedConstant: compressed_buffer is null or empty.");
    OPENVINO_ASSERT(!m_logical_shape.empty(),
                    "CompressedConstant: logical_shape must not be empty.");
    OPENVINO_ASSERT(m_logical_type != ov::element::dynamic,
                    "CompressedConstant: logical_type must be a concrete element type.");
    verify_storage_invariant(*this);
    constructor_validate_and_infer_types();
}

CompressedConstant::~CompressedConstant() = default;

// ---------------------------------------------------------------------------------------
// Overrides
// ---------------------------------------------------------------------------------------

void CompressedConstant::validate_and_infer_types() {
    // Crucially: present the *logical* element type and shape to the graph. This is what
    // makes downstream MatMul / FullyConnected shape inference and pattern matching work.
    // The base Constant's storage view (u8 [N_bytes]) is invisible to the graph at this
    // level; it is only consulted by plugin code that does `as_type_ptr<CompressedConstant>`
    // and reads the storage via `get_compressed_data_ptr()` / `get_compressed_byte_size()`.
    set_output_type(0, m_logical_type, m_logical_shape);
}

std::shared_ptr<ov::Node> CompressedConstant::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.empty(),
                    "CompressedConstant: clone_with_new_inputs expects no inputs, got ",
                    new_args.size());
    // Re-use the base Constant's shared AlignedBuffer to avoid blob duplication on clone.
    // We need a thin handle to the same storage. We construct via the raw-pointer ctor
    // and rely on Constant's copy-into-AlignedBuffer behaviour. For PoC simplicity we
    // copy the bytes; if blob sharing becomes important we can extend the API to expose
    // the underlying AlignedBuffer.
    return std::make_shared<CompressedConstant>(get_compressed_data_ptr(),
                                                get_compressed_byte_size(),
                                                m_logical_shape,
                                                m_logical_type,
                                                m_quant_type);
}

bool CompressedConstant::visit_attributes(ov::AttributeVisitor& visitor) {
    // First delegate to base Constant so the raw storage bytes (u8 [N_bytes]) are persisted.
    // The base also handles element_type/shape attributes for the storage view.
    Constant::visit_attributes(visitor);

    // Then add our logical-side metadata so deserialization can reconstruct the same node.
    // AttributeVisitor is bidirectional (reader OR writer), so we serialize into a local,
    // pass the local to the visitor, then sync back from the local. This covers both
    // write-out (where the local is unchanged) and read-in (where the visitor overwrites
    // the local with deserialized data) without branching.

    std::string quant_type_str = quant_type_to_string(m_quant_type);
    visitor.on_attribute("quant_type", quant_type_str);
    m_quant_type = quant_type_from_string(quant_type_str);

    std::vector<int64_t> logical_shape_vec;
    logical_shape_vec.reserve(m_logical_shape.size());
    for (size_t d : m_logical_shape) {
        logical_shape_vec.push_back(static_cast<int64_t>(d));
    }
    visitor.on_attribute("logical_shape", logical_shape_vec);
    m_logical_shape.clear();
    m_logical_shape.reserve(logical_shape_vec.size());
    for (int64_t d : logical_shape_vec) {
        OPENVINO_ASSERT(d >= 0,
                        "CompressedConstant: deserialized negative dimension in logical_shape.");
        m_logical_shape.push_back(static_cast<size_t>(d));
    }

    std::string logical_type_str = m_logical_type.to_string();
    visitor.on_attribute("logical_element_type", logical_type_str);
    m_logical_type = ov::element::Type(logical_type_str);

    return true;
}

// ---------------------------------------------------------------------------------------
// QuantType <-> string
// ---------------------------------------------------------------------------------------

std::string CompressedConstant::quant_type_to_string(QuantType qt) {
    const auto& m = qt_to_str_map();
    auto it = m.find(qt);
    OPENVINO_ASSERT(it != m.end(),
                    "CompressedConstant: unknown QuantType value ",
                    static_cast<int>(qt));
    return it->second;
}

CompressedConstant::QuantType CompressedConstant::quant_type_from_string(const std::string& s) {
    const auto& m = str_to_qt_map();
    auto it = m.find(s);
    OPENVINO_ASSERT(it != m.end(),
                    "CompressedConstant: unknown QuantType string '",
                    s,
                    "'");
    return it->second;
}

}  // namespace util
}  // namespace op
}  // namespace ov
