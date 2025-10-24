// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/xattn_find_block.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_CM
    #include "impls/cm/xattn_find_block.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<xattn_find_block>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_CM(cm::XAttnFindBlockImplementationManager, shape_types::static_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
