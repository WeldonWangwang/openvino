// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CPU plugin native op wrapper for IQ3XXSLinear. The compute itself lives in
// nodes/kernels/iq3_xxs_kernels.{hpp,cpp}, which is shared with the new
// CompressedConstantFCExecutor path. This file only adapts the OV node
// shape/Memory metadata into the kernel's plain pointer/size API.

#include "iq3_xxs_linear.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/iq3_xxs_kernels.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

IQ3XXSLinear::IQ3XXSLinear(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool IQ3XXSLinear::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                        std::string& errorMessage) noexcept {
    try {
        // Match by type name to stay independent of the op's C++ type identity
        // (the IQ3XXSLinear op header exists in both the core and GenAI builds).
        if (std::string(op->get_type_name()) != "IQ3XXSLinear") {
            errorMessage = "Not an IQ3XXSLinear operation.";
            return false;
        }
        if (op->get_input_size() != 2) {
            errorMessage = "IQ3XXSLinear expects 2 inputs.";
            return false;
        }
        if (op->get_input_element_type(1) != ov::element::u8) {
            errorMessage = "IQ3XXSLinear compressed weights must be u8.";
            return false;
        }
        const auto& act_pshape = op->get_input_partial_shape(0);
        if (act_pshape.rank().is_dynamic()) {
            errorMessage = "IQ3XXSLinear activation rank must be static.";
            return false;
        }
        if (act_pshape[act_pshape.rank().get_length() - 1].is_dynamic()) {
            errorMessage = "IQ3XXSLinear activation K dim must be static.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void IQ3XXSLinear::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Activation and output in f32 (the model builder converts activations to
    // f32 before this op); compressed weights are an opaque u8 blob.
    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::u8}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref);
}

void IQ3XXSLinear::execute([[maybe_unused]] const dnnl::stream& strm) {
    // One-time diagnostic: confirms this CPU-plugin native kernel is actually
    // the path executing the IQ3XXSLinear op (vs. a generic Reference fallback).
    // Gated behind an env var so it adds no noise unless explicitly requested.
    static const bool s_trace = (std::getenv("OV_CPU_IQ3XXS_TRACE") != nullptr);
    if (s_trace) {
        static std::once_flag s_once;
        std::call_once(s_once, [] {
            std::fprintf(stderr, "[intel_cpu] IQ3XXSLinear native plugin kernel executing\n");
            std::fflush(stderr);
        });
    }

    auto srcMem = getSrcMemoryAtPort(0);
    auto wMem = getSrcMemoryAtPort(1);
    auto dstMem = getDstMemoryAtPort(0);

    const auto& actDims = srcMem->getStaticDims();
    const auto& dstDims = dstMem->getStaticDims();
    const size_t rank = actDims.size();

    const size_t K = actDims[rank - 1];
    const size_t N = dstDims[dstDims.size() - 1];
    size_t rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        rows *= actDims[i];
    }

    kernels::iq3_xxs::iq3_xxs_fc(srcMem->getDataAs<float>(),
                                 wMem->getDataAs<uint8_t>(),
                                 dstMem->getDataAs<float>(),
                                 rows,
                                 K,
                                 N);
}

}  // namespace ov::intel_cpu::node
