// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CompressedConstantFCExecutor: FullyConnected executor that consumes a weight
// blob from an ov::op::util::CompressedConstant directly (no oneDNN packing,
// no f32 materialization). Decode-on-the-fly happens inside the shared
// kernels::iq3_xxs::iq3_xxs_fc() helper.
//
// Dispatch path (registered in fullyconnected_implementations.cpp):
//   1. FullyConnected ctor inspects WEIGHTS input. If it's a CompressedConstant,
//      attrs.isCompressedConstantWeight is set and quant-type / logical [N,K]
//      shape are recorded.
//   2. CompressedConstantFCExecutor::supports() returns true iff that flag is set.
//      It is registered *before* the other FC impls (MLAS / DNNL / ACL / KAI),
//      whose supports() predicates already reject CC weights via
//      noWeightsDecompression(config), so dispatch is unambiguous.
//   3. update() is a no-op (weight memory is the CC u8 [N_bytes] storage,
//      zero-copy, no precomputation required).
//   4. execute() dispatches by attrs.compressedQuantType. Phase 1 supports
//      IQ3_XXS; remaining QuantType variants throw OPENVINO_THROW.
//
// This executor intentionally does NOT support post-ops, tensor parallel
// (FC's split paths), or non-f32 src/dst in its first iteration. Those will
// be added incrementally once the bit-exact baseline is locked down.

#pragma once

#include <cstddef>
#include <memory>

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class CompressedConstantFCExecutor : public Executor {
public:
    CompressedConstantFCExecutor(const FCAttrs& attrs,
                                 const MemoryArgs& memory,
                                 const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

    /// supports() is called by ExecutorFactory against a FCConfig snapshot
    /// taken at ctor / prepareParams time. We only accept the FC node iff
    /// the FullyConnected ctor flagged the WEIGHTS input as a
    /// CompressedConstant. Other rejections (post-ops, unsupported quant
    /// type) are conservative until we extend coverage.
    static bool supports(const FCConfig& config);

private:
    FCAttrs m_attrs;
    // Captured from attrs.compressedLogicalWeightShape in ctor so execute()
    // doesn't need to re-introspect the FCAttrs each call.
    std::size_t m_K = 0;  ///< reduction dim (input channels)
    std::size_t m_N = 0;  ///< output channels
};

using CompressedConstantFCExecutorPtr = std::shared_ptr<CompressedConstantFCExecutor>;

}  // namespace ov::intel_cpu
