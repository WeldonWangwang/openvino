// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "allreduce_gather_kernel_ref.h"
#include "kernel_selector_utils.h"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace kernel_selector {
ParamsKey AllReduceGatherKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDynamicShapesSupport();

    return k;
}
JitConstants AllReduceGatherKernelRef::GetJitConstants(const all_reduce_gather_params& params) const {
    auto jit = MakeBaseParamsJitConstants(params);

    return jit;
}
CommonDispatchData AllReduceGatherKernelRef::SetDefault(const all_reduce_gather_params& params) {
    CommonDispatchData dispatch_data;

    const auto& out = params.outputs[0];
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    dispatch_data.gws = {out.X().v, out.Y().v * out.Z().v * out.W().v * out.U().v * out.V().v, out.Feature().v * out.Batch().v};
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{Tensor::DataChannelName::X},
                                                                        {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                        Tensor::DataChannelName::U, Tensor::DataChannelName::V},
                                                                        {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    return dispatch_data;
}

KernelsData AllReduceGatherKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<all_reduce_gather_params>(params);
    all_reduce_gather_params& newParams = *static_cast<all_reduce_gather_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1,
                     GetFusedPrimitiveInputsCount(params), 1, newParams.is_shape_agnostic);

    return {kd};
}

bool AllReduceGatherKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::ALLREDUCE_GATHER) {
        return false;
    }

    const all_reduce_gather_params& params = static_cast<const all_reduce_gather_params&>(p);

    for (size_t i = 0; i < params.inputs.size(); i++) {
        if (params.inputs[i].Dimentions() != 4)
            return false;
    }

    if (params.outputs[0].Dimentions() != 4)
        return false;

    return true;
}
}  // namespace kernel_selector
