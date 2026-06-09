// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/compressed_constant.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

using ov::op::util::CompressedConstant;

namespace {

// Build a deterministic synthetic IQ3_XXS-shaped blob. The block size for IQ3_XXS is
// 256 elements / 98 bytes. We construct a `rows` × `cols` weight where cols is a
// multiple of 256, so total_bytes == (rows*cols/256) * 98.
std::vector<uint8_t> make_iq3_xxs_blob(size_t rows, size_t cols, uint8_t seed = 0x5A) {
    OPENVINO_ASSERT(cols % 256 == 0, "Test setup error: cols must be a multiple of 256 for IQ3_XXS");
    const size_t blocks = (rows * cols) / 256;
    const size_t total_bytes = blocks * 98;
    std::vector<uint8_t> blob(total_bytes);
    for (size_t i = 0; i < total_bytes; ++i) {
        blob[i] = static_cast<uint8_t>((i + seed) & 0xFF);
    }
    return blob;
}

}  // namespace

// ---------------------------------------------------------------------------------------
// Construction & basic dual-faced API
// ---------------------------------------------------------------------------------------

TEST(type_prop, compressed_constant_iq3_xxs_basic_construction) {
    const ov::Shape logical_shape{4096, 4096};
    auto blob = make_iq3_xxs_blob(logical_shape[0], logical_shape[1]);

    auto cc = std::make_shared<CompressedConstant>(blob.data(),
                                                   blob.size(),
                                                   logical_shape,
                                                   ov::element::f32,
                                                   CompressedConstant::QuantType::IQ3_XXS);

    // Logical view (what the graph sees):
    EXPECT_EQ(cc->get_element_type(), ov::element::f32);
    EXPECT_EQ(cc->get_shape(), logical_shape);
    EXPECT_EQ(cc->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(cc->get_output_partial_shape(0), ov::PartialShape(logical_shape));

    // Logical accessors:
    EXPECT_EQ(cc->get_logical_element_type(), ov::element::f32);
    EXPECT_EQ(cc->get_logical_shape(), logical_shape);
    EXPECT_EQ(cc->get_quant_type(), CompressedConstant::QuantType::IQ3_XXS);

    // Storage view (what plugin kernels see):
    EXPECT_EQ(cc->get_compressed_byte_size(), blob.size());
    ASSERT_NE(cc->get_compressed_data_ptr(), nullptr);

    // The base Constant::get_byte_size() returns the storage view, NOT the logical view.
    // This is the SharedOpsOptimization (M-1) memcmp invariant.
    EXPECT_EQ(cc->ov::op::v0::Constant::get_byte_size(), blob.size());
}

TEST(type_prop, compressed_constant_construction_from_tensor) {
    const ov::Shape logical_shape{256, 256};
    auto blob = make_iq3_xxs_blob(logical_shape[0], logical_shape[1]);

    ov::Tensor blob_tensor(ov::element::u8, ov::Shape{blob.size()}, blob.data());

    auto cc = std::make_shared<CompressedConstant>(blob_tensor,
                                                   logical_shape,
                                                   ov::element::f32,
                                                   CompressedConstant::QuantType::IQ3_XXS);

    EXPECT_EQ(cc->get_shape(), logical_shape);
    EXPECT_EQ(cc->get_compressed_byte_size(), blob.size());
}

TEST(type_prop, compressed_constant_rejects_null_data_with_nonzero_bytes) {
    EXPECT_THROW(std::make_shared<CompressedConstant>(nullptr,
                                                      /*compressed_bytes=*/16,
                                                      ov::Shape{16, 16},
                                                      ov::element::f32,
                                                      CompressedConstant::QuantType::IQ3_XXS),
                 ov::AssertFailure);
}

TEST(type_prop, compressed_constant_rejects_empty_logical_shape) {
    auto blob = make_iq3_xxs_blob(256, 256);
    EXPECT_THROW(std::make_shared<CompressedConstant>(blob.data(),
                                                      blob.size(),
                                                      ov::Shape{},  // empty
                                                      ov::element::f32,
                                                      CompressedConstant::QuantType::IQ3_XXS),
                 ov::AssertFailure);
}

TEST(type_prop, compressed_constant_rejects_dynamic_logical_type) {
    auto blob = make_iq3_xxs_blob(256, 256);
    EXPECT_THROW(std::make_shared<CompressedConstant>(blob.data(),
                                                      blob.size(),
                                                      ov::Shape{256, 256},
                                                      ov::element::dynamic,
                                                      CompressedConstant::QuantType::IQ3_XXS),
                 ov::AssertFailure);
}

// ---------------------------------------------------------------------------------------
// RTTI / dispatch
// ---------------------------------------------------------------------------------------

TEST(type_prop, compressed_constant_is_recognized_as_constant_via_rtti) {
    auto blob = make_iq3_xxs_blob(256, 256);
    auto cc = std::make_shared<CompressedConstant>(blob.data(),
                                                   blob.size(),
                                                   ov::Shape{256, 256},
                                                   ov::element::f32,
                                                   CompressedConstant::QuantType::IQ3_XXS);

    // Cast to Constant (parent) must succeed.
    auto as_constant = ov::as_type_ptr<ov::op::v0::Constant>(cc);
    ASSERT_NE(as_constant, nullptr);

    // Cast back to CompressedConstant must also succeed.
    auto as_cc = ov::as_type_ptr<CompressedConstant>(as_constant);
    ASSERT_NE(as_cc, nullptr);
    EXPECT_EQ(as_cc.get(), cc.get());

    // is_type<Constant> must report true for our subclass (this is what every pattern
    // matcher uses internally — confirms ConvertMatMulToFC etc. will accept us).
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(cc));
    EXPECT_TRUE(ov::is_type<CompressedConstant>(cc));

    // is_compressed_constant convenience helper.
    EXPECT_TRUE(ov::op::util::is_compressed_constant(cc));
    EXPECT_TRUE(ov::op::util::is_compressed_constant(cc.get()));
}

TEST(type_prop, compressed_constant_plain_constant_is_not_recognized) {
    auto plain = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4, 4}, 1.0f);
    EXPECT_FALSE(ov::op::util::is_compressed_constant(plain));
    EXPECT_FALSE(ov::is_type<CompressedConstant>(plain));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(plain));
}

// ---------------------------------------------------------------------------------------
// Graph integration: downstream MatMul shape inference must work with logical view
// ---------------------------------------------------------------------------------------

TEST(type_prop, compressed_constant_matmul_shape_inference_uses_logical_view) {
    const ov::Shape logical_weight_shape{4096, 4096};  // [N, K]
    auto blob = make_iq3_xxs_blob(logical_weight_shape[0], logical_weight_shape[1]);
    auto cc = std::make_shared<CompressedConstant>(blob.data(),
                                                   blob.size(),
                                                   logical_weight_shape,
                                                   ov::element::f32,
                                                   CompressedConstant::QuantType::IQ3_XXS);

    // input is [batch, seq, K]
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 4096});

    // MatMul with transpose_b=true: output is [batch, seq, N]
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, cc, /*transpose_a=*/false, /*transpose_b=*/true);

    EXPECT_EQ(matmul->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(matmul->get_output_partial_shape(0), (ov::PartialShape{-1, -1, 4096}));
}

// ---------------------------------------------------------------------------------------
// ConstantFolding must NOT touch CompressedConstant
// ---------------------------------------------------------------------------------------

TEST(type_prop, compressed_constant_is_not_constant_foldable) {
    auto blob = make_iq3_xxs_blob(256, 256);
    auto cc = std::make_shared<CompressedConstant>(blob.data(),
                                                   blob.size(),
                                                   ov::Shape{256, 256},
                                                   ov::element::f32,
                                                   CompressedConstant::QuantType::IQ3_XXS);

    EXPECT_FALSE(cc->can_constant_fold(ov::OutputVector{}));
    EXPECT_FALSE(cc->has_evaluate());

    ov::TensorVector outputs(1);
    ov::TensorVector inputs;
    EXPECT_FALSE(cc->evaluate(outputs, inputs));
}

// ---------------------------------------------------------------------------------------
// QuantType string conversion (used by visit_attributes)
// ---------------------------------------------------------------------------------------

TEST(type_prop, compressed_constant_quant_type_string_roundtrip) {
    for (auto qt : {CompressedConstant::QuantType::IQ3_XXS,
                    CompressedConstant::QuantType::IQ2_S,
                    CompressedConstant::QuantType::IQ4_XS,
                    CompressedConstant::QuantType::IQ3_S,
                    CompressedConstant::QuantType::IQ2_XS,
                    CompressedConstant::QuantType::Q3_K,
                    CompressedConstant::QuantType::Q5_K}) {
        const auto s = CompressedConstant::quant_type_to_string(qt);
        EXPECT_FALSE(s.empty());
        EXPECT_EQ(CompressedConstant::quant_type_from_string(s), qt);
    }
}

TEST(type_prop, compressed_constant_quant_type_from_unknown_string_throws) {
    EXPECT_THROW(CompressedConstant::quant_type_from_string("totally_made_up_type"), ov::AssertFailure);
}

// ---------------------------------------------------------------------------------------
// clone_with_new_inputs must preserve all logical/storage information
// ---------------------------------------------------------------------------------------

TEST(type_prop, compressed_constant_clone_preserves_state) {
    auto blob = make_iq3_xxs_blob(256, 256);
    auto cc = std::make_shared<CompressedConstant>(blob.data(),
                                                   blob.size(),
                                                   ov::Shape{256, 256},
                                                   ov::element::f32,
                                                   CompressedConstant::QuantType::IQ3_XXS);

    auto cloned = cc->clone_with_new_inputs(ov::OutputVector{});
    ASSERT_NE(cloned, nullptr);

    auto cloned_cc = ov::as_type_ptr<CompressedConstant>(cloned);
    ASSERT_NE(cloned_cc, nullptr);

    EXPECT_EQ(cloned_cc->get_logical_shape(), cc->get_logical_shape());
    EXPECT_EQ(cloned_cc->get_logical_element_type(), cc->get_logical_element_type());
    EXPECT_EQ(cloned_cc->get_quant_type(), cc->get_quant_type());
    EXPECT_EQ(cloned_cc->get_compressed_byte_size(), cc->get_compressed_byte_size());

    // Memory contents must match (storage was actually copied through).
    const auto* src = static_cast<const uint8_t*>(cc->get_compressed_data_ptr());
    const auto* dst = static_cast<const uint8_t*>(cloned_cc->get_compressed_data_ptr());
    for (size_t i = 0; i < blob.size(); ++i) {
        ASSERT_EQ(src[i], dst[i]) << "Byte " << i << " differs after clone.";
    }
}

TEST(type_prop, compressed_constant_clone_with_nonempty_inputs_throws) {
    auto blob = make_iq3_xxs_blob(256, 256);
    auto cc = std::make_shared<CompressedConstant>(blob.data(),
                                                   blob.size(),
                                                   ov::Shape{256, 256},
                                                   ov::element::f32,
                                                   CompressedConstant::QuantType::IQ3_XXS);

    auto dummy_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    EXPECT_THROW(cc->clone_with_new_inputs(ov::OutputVector{dummy_input->output(0)}), ov::AssertFailure);
}
