// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "config.h"
#include "executor_config.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/util/compressed_constant.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

// @todo require explicit initialization of all the attributes?
struct FCAttrs {
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    uint64_t dynamicQuantizationGroupSize = 0;
    bool constantWeights = true;

    // ---------------------------------------------------------------------
    // CompressedConstant weight metadata.
    //
    // Set by FullyConnected ctor when WEIGHTS input is an
    // ov::op::util::CompressedConstant (e.g. an IQ3_XXS gguf weight).
    // Consumed by the FullyConnected executor dispatcher:
    // CompressedConstantFCExecutor::supports() returns true iff this flag
    // is set, so existing impls (DNNL / MLAS / ACL / KAI) remain untouched
    // for non-CC weights.
    //
    // - isCompressedConstantWeight: true iff WEIGHTS is a CompressedConstant.
    // - compressedQuantType:        which GGUF quant scheme (IQ3_XXS, IQ4_XS, ...).
    // - compressedLogicalWeightShape: logical [N, K] shape of the weight
    //                                 (independent of the u8 [N_bytes] storage).
    // ---------------------------------------------------------------------
    bool isCompressedConstantWeight = false;
    ov::op::util::CompressedConstant::QuantType compressedQuantType =
        ov::op::util::CompressedConstant::QuantType::IQ3_XXS;
    ov::Shape compressedLogicalWeightShape;
    // Direct pointer to the compressed blob. The executor reads from this
    // pointer instead of from ARG_WEI memory, bypassing edge negotiation
    // entirely. Lifetime is guaranteed: the CompressedConstant (held by
    // PinnedCompressedConstant) owns the buffer for the model's lifetime.
    const void* compressedDataPtr = nullptr;
    size_t compressedByteSize = 0;

    ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::Config::ModelType::Unknown;

    PostOps postOps;
};

using FCConfig = executor::Config<FCAttrs>;
}  // namespace ov::intel_cpu
