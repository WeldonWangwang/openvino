// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// broadcast_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct all_reduce_broadcast_params : public base_params {
    all_reduce_broadcast_params() : base_params(KernelType::ALLREDUCE_BROADCAST) {}
    size_t world_size = 1;  // Number of devices in the all-reduce operation
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BroadcastKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class AllReduceBroadcastKernelRef : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    AllReduceBroadcastKernelRef() : KernelBaseOpenCL{"all_reduce_broadcast_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const all_reduce_broadcast_params& params) const;
    static DispatchData SetDefault(const all_reduce_broadcast_params& params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    bool Validate(const Params& params) const override;
};
}  // namespace kernel_selector