// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

/// \brief A model pass that wraps all `ov::op::util::CompressedConstant` nodes in
///        a plugin-private `PinnedCompressedConstant` wrapper.
///
/// \details
/// This pass must be registered at the very front of the CPU plugin transformation
/// pipeline — before any pass that pattern-matches `v0::Constant` or reads Constant
/// data (e.g. `cast_vector`, `get_data_ptr<float>`).
///
/// After this pass runs, no `CompressedConstant` node remains in the graph (they are
/// each replaced by a `PinnedCompressedConstant` wrapper). Since the wrapper inherits
/// from `ov::op::Op` (not from `v0::Constant`), it is invisible to
/// `pattern::wrap_type<v0::Constant>()` patterns, preventing buffer-over-read crashes.
///
/// The wrapper is zero-cost: it holds a `shared_ptr` to the original CC (no blob copy),
/// and reports the same logical output type/shape, so downstream ops still see a valid
/// tensor.
///
/// Later in the pipeline, the `FullyConnected` ctor and `Input` node know how to
/// recognize `PinnedCompressedConstant` and extract the underlying compressed blob.
class CompressedConstantPin : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("CompressedConstantPin");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
