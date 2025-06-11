// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "allreduce_broadcast_kernel_ref.h"
#include "kernel_selector_utils.h"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace kernel_selector {
ParamsKey AllReduceBroadcastKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    return k;
}

JitConstants AllReduceBroadcastKernelRef::GetJitConstants(const all_reduce_broadcast_params& params) const {
    auto jit = MakeBaseParamsJitConstants(params);
    return jit;
}
CommonDispatchData AllReduceBroadcastKernelRef::SetDefault(const all_reduce_broadcast_params& params) {
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

KernelsData AllReduceBroadcastKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<all_reduce_broadcast_params>(params);
    all_reduce_broadcast_params& newParams = *static_cast<all_reduce_broadcast_params*>(kd.params.get());

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

bool AllReduceBroadcastKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::ALLREDUCE_BROADCAST) {
        return false;
    }

    const all_reduce_broadcast_params& params = static_cast<const all_reduce_broadcast_params&>(p);

    for (size_t i = 0; i < params.inputs.size(); i++) {
        if (params.inputs[i].Dimentions() != 4)
            return false;
    }

    if (params.outputs[0].Dimentions() != 4)
        return false;

    return true;
}
void AllReduceBroadcastKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const all_reduce_broadcast_params&>(params);

        auto dispatch_data = SetDefault(prim_params);

        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatch_data.gws;
        kd.kernels[0].params.workGroups.local = dispatch_data.lws;
        kd.kernels[0].skip_execution = false;

        const auto inter_dt = prim_params.inputs[0].GetDType();
        const auto indexes_buf_size = prim_params.inputs[0].PhysicalSizeInBytes() / prim_params.world_size;

        const bool lockable = false;
        kd.internalBuffers.clear();
        kd.internalBuffers.emplace_back(indexes_buf_size, lockable); // local buffer for remote mapping
        //kd.internalBuffers.emplace_back(indexes_buf_size, lockable); // remote buffer for local mapping
        kd.internalBufferDataType = inter_dt;
    };
}
}  // namespace kernel_selector
