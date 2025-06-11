// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#define CL_VERSION_3_0 1
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>
#include "multi_stage_primitive.hpp"
#include "eltwise/eltwise_kernel_base.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "allreduce/allreduce_kernel_selector.h"
#include "allreduce/allreduce_gather_kernel_ref.h"
#include "allreduce/allreduce_broadcast_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "register.hpp"
#include "registry/implementation_map.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_ext.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "all_reduce_inst.h"
#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050
namespace cldnn {
namespace ocl {

struct all_reduce_impl : public multi_stage_primitive<all_reduce> {
    using parent = multi_stage_primitive<all_reduce>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::eltwise_kernel_selector;
    using kernel_params_t = kernel_selector::eltwise_params;
    using broadcast_kernel_selector_t = kernel_selector::allreduce_broadcast_kernel_selector;
    using broadcast_kernel_params_t = kernel_selector::all_reduce_broadcast_params;
    using gather_kernel_selector_t = kernel_selector::allreduce_gather_kernel_selector;
    using gather_kernel_params_t = kernel_selector::all_reduce_gather_params;
    cldnn::memory::ptr p2p_memory = nullptr;

    enum Stage {
        BROADCAST = 0,
        GATHER
    };

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::all_reduce_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<all_reduce_impl>(*this);
    }

    all_reduce_impl() : parent() {}

    all_reduce_impl(const std::vector<kernel_selector::kernel_data>& kd) : parent(kd) {
        this->can_reuse_memory = true;
    }

    ~all_reduce_impl() = default;

    explicit all_reduce_impl(const all_reduce_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<all_reduce>());
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    kernel_arguments_data get_arguments(const all_reduce_inst& instance, size_t stage) const override {
        const auto desc = instance.get_node().as<all_reduce>().get_primitive();

        kernel_arguments_data args;
        if (stage == Stage::BROADCAST) {
            args.inputs = {  instance.input_memory_ptr(0)};

            args.outputs = { instance.output_memory_ptr(0) };
        }
        return args;
    }

    void set_arguments_impl(all_reduce_inst& instance) override {}

    static kernel_params_t get_eltwise_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto params = get_default_params<kernel_selector::eltwise_params>(impl_param, is_shape_agnostic);

        const auto inputs_count = 2;
        params.inputs.resize(inputs_count);
        // need to update second input to intermediate buffer, how?
        for (size_t i = 0; i < inputs_count; ++i) {
            params.inputs[i] = convert_data_tensor(impl_param.input_layouts[i]);
        }

        /*const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset; // [kv_past, kv_new_token, [beam_idx, beam_table_past]
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset; // [kv_present, beam_table_present]
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(1)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);*/

        return params;
    }
    static broadcast_kernel_params_t get_allreduce_broadcast_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<all_reduce>();

        auto params = get_default_params<kernel_selector::all_reduce_broadcast_params>(impl_param, is_shape_agnostic);

        params.inputs.resize(1);
        params.inputs[0] = convert_data_tensor(impl_param.input_layouts[0]);
        params.outputs.resize(1);
        params.outputs[0] = convert_data_tensor(impl_param.output_layouts[0]);
        params.world_size = impl_param.w_size;

        return params;
    }
    static gather_kernel_params_t get_allreduce_gather_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<all_reduce>();
        auto inputs_count = primitive->input.size();

        auto params = get_default_params<kernel_selector::all_reduce_gather_params>(impl_param, is_shape_agnostic);
        // TBD
        for (size_t i = 1; i < inputs_count; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }

        return params;
    }
    void execute_stage(const std::vector<event::ptr>& events,
                       all_reduce_inst& instance,
                       std::vector<event::ptr>& all_events,
                       Stage stage,
                       size_t send_chunk) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        size_t kernel_offset = 0;
        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto args = get_arguments(instance, stage);
            //args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }
    }
    event::ptr execute_impl(const std::vector<event::ptr>& events, all_reduce_inst& instance) override {
        auto& local_stream = instance.get_network().get_stream();
        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto& engine = instance.get_network().get_engine();
        auto& ocl_engine = downcast<ocl::ocl_engine>(engine);
        auto& ocl_stream = downcast<ocl::ocl_stream>(local_stream);
        //auto local_queue = ocl_stream.get_cl_queue().get();                        // raw cl queue
        auto local_context = ocl_stream.get_engine().get_cl_context().get();       // raw cl context
        auto local_device_handle = ocl_stream.get_engine().get_cl_device().get();  // raw cl device
        auto dst_idx = (w_rank + 1) % w_size;                                      // write peer for p2p of current rank in ring all_reduce
        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);
        std::vector<event::ptr> res_events;
        // sync point between ranks
        while (true) {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->updated_flag = false;
            sub_mem_mgr->step1_copy_done.store(0);
            sub_mem_mgr->step2_add_done.store(0);
            sub_mem_mgr->step3_concat_copy_done.store(0);
            sub_mem_mgr->step4_concat_copy_done.store(0);
            if (sub_mem_mgr->_use_count[id] == w_size) {
                sub_mem_mgr->_use_count[id] = 0;
                for (size_t i = 0; i < w_size; i++) {
                    sub_mem_mgr->_memorys_table[id][i].flag = false;
                    for (size_t j = 0; j < w_size; j++) {
                        sub_mem_mgr->_memorys_table[id][i].events[j] = nullptr;
                        sub_mem_mgr->_memorys_table[id][i].recv_flag[j] = false;
                    }
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
        }
        // register memory of local rank to sub memory manager
        sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0] = instance.get_intermediates_memories()[0];
        // update flag for local rank memory ready
        sub_mem_mgr->_memorys_table[id][w_rank].recv_flag[0] = true;
        // wait for peer memory ready
        while (true) {
            size_t wait_all_ouput_ready = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (sub_mem_mgr->_memorys_table[id][idx].recv_flag[0] == true)
                    wait_all_ouput_ready++;
            }
            if (wait_all_ouput_ready == w_size) {
                break;
            }
        }
        // map remote memory for write
        cl_int err;
        uint64_t fd;
        auto dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].recv_bufs[0];
        auto size = dst_mem->get_layout().bytes_count();
        auto clbuf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
        err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(fd), &fd, NULL);
        if (err < 0) {
            GPU_DEBUG_TRACE_DETAIL << "FAILED to get handle from mem obj " << clbuf << std::endl;
        }
        cl_mem_properties_intel extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                CL_MEM_DEVICE_ID_INTEL,
                                                (cl_mem_properties_intel)local_device_handle,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(local_context, extMemProperties, 0, size, NULL, &err);
        if (err < 0) {
            GPU_DEBUG_TRACE_DETAIL << "FAILED to import buffer from mem " << clbuf << ", fd as " << fd << std::endl;
            OPENVINO_ASSERT(false,
                            "clCreateBufferWithProperties failed, clbuf = %p, fd = %ld, size = %ld, new_cl_mem = %p\n",
                            clbuf,
                            fd,
                            size,
                            extMemBuffer);
        }
        p2p_memory = std::make_shared<ocl::gpu_buffer>(&ocl_engine, dst_mem->get_layout(), cl::Buffer(extMemBuffer, true), nullptr);

        for (size_t step = 0; step < w_size -1; step++) {
            int32_t send_chunk_idx = (w_rank - step + w_size) % w_size;
            //int32_t target_chunk_idx = (w_rank - step - 1 + w_size) % w_size;
            execute_stage(events, instance, res_events, Stage::BROADCAST, send_chunk_idx);
        }
        return nullptr;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const typed_program_node<all_reduce>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto all_reduce_broadcast_params = get_allreduce_broadcast_params(impl_param, impl_param.is_dynamic());
        kernels_data.push_back(broadcast_kernel_selector_t::Instance().get_best_kernel(all_reduce_broadcast_params));
        //auto eltwise_kernel_params = get_eltwise_kernel_params(impl_param, impl_param.is_dynamic());
        //auto& eltwise_kernel_selector = kernel_selector_t::Instance();
        //kernels_data.push_back(eltwise_kernel_selector.get_best_kernel(eltwise_kernel_params));
        //auto all_reduce_gather_params = get_allreduce_gather_params(impl_param, impl_param.is_dynamic());
        //kernels_data.push_back(broadcast_kernel_selector_t::Instance().get_best_kernel(all_reduce_gather_params));
        return std::make_unique<all_reduce_impl>(kernels_data);
    }
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& impl_param) const override {
        /*
        * Description:
        * sub-buffers used for remote mapping and peer-to-peer copy
        */

        auto add_internal_buffers = [&](std::vector<BufferDescriptor>& internal_buffers,
                                       const kernel_selector::KernelData& kd) {
            for (const auto& buffer_desc : kd.internalBuffers) {
                internal_buffers.emplace_back(buffer_desc.byte_count, ov::element::u8, buffer_desc.lockable);
            }
        };

        std::vector<BufferDescriptor> internal_buffers;
        add_internal_buffers(internal_buffers, _kernels_data[Stage::BROADCAST]);

        return internal_buffers;
    }
    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto& params = static_cast<kernel_params_t&>(*_kernels_data[Stage::BROADCAST].params);
        const auto inputs_count = 1;
        for (size_t i = 0; i < inputs_count; ++i) {
            params.inputs[i] = convert_data_tensor(impl_param.input_layouts[i]);
        }
        params.outputs[0] = convert_data_tensor(impl_param.output_layouts[0]);

        (_kernels_data[Stage::BROADCAST].update_dispatch_data_func)(params, _kernels_data[Stage::BROADCAST]);
        _kernels_data[Stage::BROADCAST].kernels[0].skip_execution = false;
    }
};

namespace detail {

attach_all_reduce_impl::attach_all_reduce_impl() {
    auto types = { data_types::i8, data_types::f16, data_types::f32 };
    auto formats = { format::bfyx };
    implementation_map<all_reduce>::add(impl_types::ocl, shape_types::dynamic_shape, all_reduce_impl::create, types, formats);
    implementation_map<all_reduce>::add(impl_types::ocl, shape_types::static_shape, all_reduce_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::all_reduce_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::all_reduce)
