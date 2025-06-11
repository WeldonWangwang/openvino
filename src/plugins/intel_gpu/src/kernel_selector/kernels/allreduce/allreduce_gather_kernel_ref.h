// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct all_reduce_gather_params : public base_params {
    all_reduce_gather_params() : base_params(KernelType::ALLREDUCE_GATHER) {}
    std::vector<uint16_t> input_order;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GatherKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class AllReduceGatherKernelRef : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    AllReduceGatherKernelRef() : KernelBaseOpenCL{"all_reduce_gather_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const all_reduce_gather_params& params) const;
    static DispatchData SetDefault(const all_reduce_gather_params& params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    bool Validate(const Params& params) const override;
};
}  // namespace kernel_selector