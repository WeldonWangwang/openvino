// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/compressed_constant_pin.hpp"

#include <cstdio>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/compressed_constant.hpp"
#include "transformations/cpu_opset/common/op/pinned_compressed_constant.hpp"

namespace ov::intel_cpu {

bool CompressedConstantPin::run_on_model(const std::shared_ptr<ov::Model>& model) {
    size_t n_pinned = 0;
    size_t n_skipped = 0;

    // Iterate over a snapshot of the ordered ops (avoid invalidating iterators).
    const auto ops = model->get_ordered_ops();
    for (const auto& node : ops) {
        auto cc = ov::as_type_ptr<ov::op::util::CompressedConstant>(node);
        if (!cc) {
            continue;
        }

        // Only wrap CC nodes whose quant type we can handle end-to-end.
        // Currently only IQ3_XXS is wired through CompressedConstantFCExecutor.
        // Other quant types (IQ2_S, IQ4_XS, Q3_K, etc.) must be left as raw
        // CompressedConstants — the standard graph pipeline will decompress them
        // via the generic Constant path or they'll be handled by other passes.
        using QT = ov::op::util::CompressedConstant::QuantType;
        if (cc->get_quant_type() != QT::IQ3_XXS) {
            ++n_skipped;
            continue;
        }

        // Create the wrapper. It holds a shared_ptr to the CC (zero-copy blob).
        auto pinned = std::make_shared<PinnedCompressedConstant>(cc);
        pinned->set_friendly_name(cc->get_friendly_name() + "_pinned");

        // Replace all downstream consumers: every output port of the CC that was
        // connected to a consumer is now reconnected to the wrapper's output.
        // The CC itself becomes dangling and will be cleaned up by the framework.
        ov::replace_node(cc, pinned);

        ++n_pinned;
    }

    if (n_pinned > 0 || n_skipped > 0) {
        std::fprintf(stderr,
                     "[intel_cpu] CompressedConstantPin: wrapped %zu IQ3_XXS node(s), "
                     "skipped %zu other-quant node(s)\n",
                     n_pinned, n_skipped);
    }

    // Verify: no IQ3_XXS CompressedConstant nodes should remain in the graph.
    // Other quant types are intentionally left as raw CC for standard handling.
    if (n_pinned > 0) {
        size_t n_cc_remaining = 0;
        for (const auto& node : model->get_ordered_ops()) {
            auto cc_check = ov::as_type_ptr<ov::op::util::CompressedConstant>(node);
            if (cc_check && cc_check->get_quant_type() ==
                    ov::op::util::CompressedConstant::QuantType::IQ3_XXS) {
                ++n_cc_remaining;
            }
        }
        if (n_cc_remaining > 0) {
            std::fprintf(stderr,
                         "[intel_cpu] CompressedConstantPin WARNING: %zu CompressedConstant "
                         "nodes survived (some earlier pass may have duplicated them)\n",
                         n_cc_remaining);
        }
    }

    return n_pinned > 0;
}

}  // namespace ov::intel_cpu
