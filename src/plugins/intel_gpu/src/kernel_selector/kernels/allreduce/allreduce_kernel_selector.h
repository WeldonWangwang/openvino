// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class allreduce_broadcast_kernel_selector : public kernel_selector_base {
public:
    static allreduce_broadcast_kernel_selector& Instance() {
        static allreduce_broadcast_kernel_selector instance_;
        return instance_;
    }

    allreduce_broadcast_kernel_selector();

    virtual ~allreduce_broadcast_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};

class allreduce_gather_kernel_selector : public kernel_selector_base {
public:
    static allreduce_gather_kernel_selector& Instance() {
        static allreduce_gather_kernel_selector instance_;
        return instance_;
    }

    allreduce_gather_kernel_selector();

    virtual ~allreduce_gather_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector
