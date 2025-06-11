// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "allreduce_kernel_selector.h"
#include "allreduce_gather_kernel_ref.h"
#include "allreduce_broadcast_kernel_ref.h"

namespace kernel_selector {

allreduce_broadcast_kernel_selector::allreduce_broadcast_kernel_selector() {
    Attach<AllReduceBroadcastKernelRef>();
}

KernelsData allreduce_broadcast_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ALLREDUCE_BROADCAST);
}

allreduce_gather_kernel_selector::allreduce_gather_kernel_selector() {
    Attach<AllReduceGatherKernelRef>();
}

KernelsData allreduce_gather_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ALLREDUCE_GATHER);
}

}  // namespace kernel_selector
