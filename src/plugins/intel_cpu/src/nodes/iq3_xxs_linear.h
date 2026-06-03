// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

// CPU plugin native kernel for the IQ3XXSLinear op.
//
// Computes Y = X @ W^T where W is stored as an opaque IQ3_XXS compressed blob
// (256-weight super-blocks, 98 bytes each). The compressed weights stay
// compressed in the model; this node decodes weight tiles on the fly during
// execute() and never materializes the full dequantized weight matrix.
//
// All geometry is derived from the I/O shapes and the fixed IQ3_XXS constants
// (block_size = 256, bytes_per_block = 98), so the node does not depend on the
// op's C++ type identity (the op header exists in two builds). isSupportedOperation
// matches by type name string for robustness.
class IQ3XXSLinear : public Node {
public:
    IQ3XXSLinear(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::IQ3XXSLinear;
    }
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {}
    void execute(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
};

}  // namespace ov::intel_cpu::node
