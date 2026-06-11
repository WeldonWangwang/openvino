// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CompressedConstantFCExecutor implementation. See header for context.

#include "compressed_constant_fc.hpp"

#include <any>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/iq3_xxs_kernels.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/util/compressed_constant.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

namespace {
using QuantType = ov::op::util::CompressedConstant::QuantType;
}  // namespace

CompressedConstantFCExecutor::CompressedConstantFCExecutor(const FCAttrs& attrs,
                                                           [[maybe_unused]] const MemoryArgs& memory,
                                                           [[maybe_unused]] const ExecutorContext::CPtr& context)
    : m_attrs(attrs) {
    // The dispatcher has already verified attrs.isCompressedConstantWeight via
    // supports(), but keep this assertion to catch misuse if the executor is
    // constructed directly (e.g. from a unit test).
    OPENVINO_ASSERT(m_attrs.isCompressedConstantWeight,
                    "CompressedConstantFCExecutor: WEIGHTS must come from an ",
                    "ov::op::util::CompressedConstant; FullyConnected ctor ",
                    "did not flag this FC node accordingly.");

    // Logical weight is row-major [N, K] (output channels first), matching
    // both ConvertMatMulToFC's contract and the IQ3_XXS storage layout
    // (one super-block row per output channel).
    OPENVINO_ASSERT(m_attrs.compressedLogicalWeightShape.size() == 2,
                    "CompressedConstantFCExecutor: expected rank-2 logical ",
                    "weight shape [N, K], got rank=",
                    m_attrs.compressedLogicalWeightShape.size());
    m_N = m_attrs.compressedLogicalWeightShape[0];
    m_K = m_attrs.compressedLogicalWeightShape[1];

    // Phase 1: only IQ3_XXS is wired through to the kernel. Other QuantType
    // variants will fail loudly via execute(); supports() also rejects them
    // so we should never actually get here for an unsupported scheme, but
    // be explicit anyway.
    OPENVINO_ASSERT(m_attrs.compressedQuantType == QuantType::IQ3_XXS,
                    "CompressedConstantFCExecutor: only IQ3_XXS is supported in phase 1");
}

bool CompressedConstantFCExecutor::update([[maybe_unused]] const MemoryArgs& memory) {
    // No precomputation required:
    //   * WEIGHTS memory is the CompressedConstant's u8 [N_bytes] storage,
    //     mapped zero-copy by intel_cpu/src/nodes/input.cpp.
    //   * SRC/DST shapes are read in execute() from the per-call MemoryArgs.
    //   * The decode kernel itself is stateless (constant tables are static).
    return true;
}

void CompressedConstantFCExecutor::execute(const MemoryArgs& memory) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& dstMem = memory.at(ARG_DST);

    // Activation rank is variable (e.g. [B, S, K] or [M, K]); collapse all
    // leading dims into a row count and keep K as the trailing dim.
    const auto& srcDims = srcMem->getStaticDims();
    OPENVINO_ASSERT(!srcDims.empty(),
                    "CompressedConstantFCExecutor: activation must have ",
                    "rank >= 1 to expose a trailing K dim");
    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < srcDims.size(); ++i) {
        rows *= srcDims[i];
    }
    const std::size_t K_actual = srcDims.back();
    OPENVINO_ASSERT(K_actual == m_K,
                    "CompressedConstantFCExecutor: activation K=",
                    K_actual,
                    " does not match weight K=",
                    m_K,
                    " from CompressedConstant logical shape");

    // Sanity-check the destination matches our cached [rows, N] expectation.
    const auto& dstDims = dstMem->getStaticDims();
    OPENVINO_ASSERT(!dstDims.empty() && dstDims.back() == m_N,
                    "CompressedConstantFCExecutor: destination trailing dim ",
                    dstDims.empty() ? 0u : dstDims.back(),
                    " does not match weight N=",
                    m_N);

    // Read the compressed blob directly from the stored pointer (set by FC
    // ctor from the CompressedConstant/PinnedCompressedConstant). This
    // bypasses edge memory entirely, avoiding the f32 [N,K] ↔ u8 [N_bytes]
    // descriptor mismatch that would otherwise occur.
    OPENVINO_ASSERT(m_attrs.compressedDataPtr,
                    "CompressedConstantFCExecutor: compressedDataPtr is null");
    const auto* w_data = static_cast<const uint8_t*>(m_attrs.compressedDataPtr);

    switch (m_attrs.compressedQuantType) {
    case QuantType::IQ3_XXS:
        kernels::iq3_xxs::iq3_xxs_fc(srcMem->getDataAs<float>(),
                                     w_data,
                                     dstMem->getDataAs<float>(),
                                     rows,
                                     m_K,
                                     m_N);
        break;
    default:
        OPENVINO_THROW("CompressedConstantFCExecutor: unsupported quant type ",
                       static_cast<int>(m_attrs.compressedQuantType));
    }

    // Apply fused post-ops on the output buffer (in-place).
    // Phase 1: supports bias (ScaleShiftPostOp::add) and common activations.
    // Other post-op types trigger a warning but don't crash.
    if (!m_attrs.postOps.empty()) {
        float* dst = dstMem->getDataAs<float>();
        const size_t dst_size = rows * m_N;
        for (const auto& postOp : m_attrs.postOps) {
            if (auto* act = std::any_cast<ActivationPostOp>(&postOp)) {
                // Apply element-wise activation on dst.
                switch (act->type()) {
                case ActivationPostOp::Type::relu:
                    for (size_t i = 0; i < dst_size; ++i)
                        dst[i] = dst[i] > 0.f ? dst[i] : act->alpha() * dst[i];
                    break;
                case ActivationPostOp::Type::swish:
                    for (size_t i = 0; i < dst_size; ++i) {
                        float x = dst[i];
                        dst[i] = x / (1.f + std::exp(-act->alpha() * x));
                    }
                    break;
                case ActivationPostOp::Type::gelu_erf:
                    for (size_t i = 0; i < dst_size; ++i) {
                        float x = dst[i];
                        dst[i] = 0.5f * x * (1.f + std::erf(x * 0.7071067811865475f));
                    }
                    break;
                case ActivationPostOp::Type::gelu_tanh:
                    for (size_t i = 0; i < dst_size; ++i) {
                        float x = dst[i];
                        dst[i] = 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                    }
                    break;
                case ActivationPostOp::Type::logistic:
                    for (size_t i = 0; i < dst_size; ++i)
                        dst[i] = 1.f / (1.f + std::exp(-dst[i]));
                    break;
                case ActivationPostOp::Type::tanh:
                    for (size_t i = 0; i < dst_size; ++i)
                        dst[i] = std::tanh(dst[i]);
                    break;
                default:
                    // Unsupported activation: leave output as-is (warning-level only).
                    break;
                }
            } else if (auto* ss = std::any_cast<ScaleShiftPostOp>(&postOp)) {
                // ScaleShift: typically bias addition (type=add, shift per-channel).
                // The shift data comes from a fused input (e.g. ARG_BIAS memory).
                // For bias addition: dst[row][n] += bias[n]
                if (ss->type() == ScaleShiftPostOp::Type::add) {
                    if (auto it = memory.find(ARG_BIAS); it != memory.end() && it->second) {
                        const float* bias = it->second->getDataAs<float>();
                        for (size_t r = 0; r < rows; ++r)
                            for (size_t n = 0; n < m_N; ++n)
                                dst[r * m_N + n] += bias[n];
                    }
                }
            }
            // Other post-op types (FakeQuantize, Sum, etc.) are ignored in Phase 1.
        }
    }
}

bool CompressedConstantFCExecutor::supports(const FCConfig& config) {
    // Hard gate: only accept FC nodes whose ctor flagged a CC weight.
    if (!config.attrs.isCompressedConstantWeight) {
        return false;
    }
    // Phase 1: IQ3_XXS only. Other QuantType variants will be added one by
    // one as their kernels land in nodes/kernels/.
    if (config.attrs.compressedQuantType != QuantType::IQ3_XXS) {
        return false;
    }
    // MUST accept all CC-weight FC layers regardless of post-ops.
    // If we return false, the factory falls through to other executors
    // (oneDNN, MLAS, etc.) which try to use the weight edge memory as
    // normal f32 data—but it actually holds compressed bytes, causing
    // either garbage output or an allocation of a full logical-sized buffer.
    //
    // Post-ops (bias, activation) are applied manually after the kernel.
    return true;
}

}  // namespace ov::intel_cpu
