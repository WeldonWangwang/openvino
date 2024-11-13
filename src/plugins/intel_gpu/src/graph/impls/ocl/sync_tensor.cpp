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

#include "impls/registry/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "register.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "sync_tensor_inst.h"

namespace cldnn {
namespace ocl {

#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050
static std::map<int, std::string> oclErrorCode = {
    {0, "CL_SUCCESS"},
    {-1, "CL_DEVICE_NOT_FOUND"},
    {-2, "CL_DEVICE_NOT_AVAILABLE"},
    {-3, "CL_COMPILER_NOT_AVAILABLE"},
    {-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {-5, "CL_OUT_OF_RESOURCES"},
    {-6, "CL_OUT_OF_HOST_MEMORY"},
    {-7, "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {-8, "CL_MEM_COPY_OVERLAP"},
    {-9, "CL_IMAGE_FORMAT_MISMATCH"},
    {-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {-11, "CL_BUILD_PROGRAM_FAILURE"},
    {-12, "CL_MAP_FAILURE"},
    {-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {-15, "CL_COMPILE_PROGRAM_FAILURE"},
    {-16, "CL_LINKER_NOT_AVAILABLE"},
    {-17, "CL_LINK_PROGRAM_FAILURE"},
    {-18, "CL_DEVICE_PARTITION_FAILED"},
    {-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {-30, "CL_INVALID_VALUE"},
    {-31, "CL_INVALID_DEVICE_TYPE"},
    {-32, "CL_INVALID_PLATFORM"},
    {-33, "CL_INVALID_DEVICE"},
    {-34, "CL_INVALID_CONTEXT"},
    {-35, "CL_INVALID_QUEUE_PROPERTIES"},
    {-36, "CL_INVALID_COMMAND_QUEUE"},
    {-37, "CL_INVALID_HOST_PTR"},
    {-38, "CL_INVALID_MEM_OBJECT"},
    {-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {-40, "CL_INVALID_IMAGE_SIZE"},
    {-41, "CL_INVALID_SAMPLER"},
    {-42, "CL_INVALID_BINARY"},
    {-43, "CL_INVALID_BUILD_OPTIONS"},
    {-44, "CL_INVALID_PROGRAM"},
    {-45, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {-46, "CL_INVALID_KERNEL_NAME"},
    {-47, "CL_INVALID_KERNEL_DEFINITION"},
    {-48, "CL_INVALID_KERNEL"},
    {-49, "CL_INVALID_ARG_INDEX"},
    {-50, "CL_INVALID_ARG_VALUE"},
    {-51, "CL_INVALID_ARG_SIZE"},
    {-52, "CL_INVALID_KERNEL_ARGS"},
    {-53, "CL_INVALID_WORK_DIMENSION"},
    {-54, "CL_INVALID_WORK_GROUP_SIZE"},
    {-55, "CL_INVALID_WORK_ITEM_SIZE"},
    {-56, "CL_INVALID_GLOBAL_OFFSET"},
    {-57, "CL_INVALID_EVENT_WAIT_LIST"},
    {-58, "CL_INVALID_EVENT"},
    {-59, "CL_INVALID_OPERATION"},
    {-60, "CL_INVALID_GL_OBJECT"},
    {-61, "CL_INVALID_BUFFER_SIZE"},
    {-62, "CL_INVALID_MIP_LEVEL"},
    {-63, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {-64, "CL_INVALID_PROPERTY"},
    {-65, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {-66, "CL_INVALID_COMPILER_OPTIONS"},
    {-67, "CL_INVALID_LINKER_OPTIONS"},
    {-68, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {-69, "CL_INVALID_PIPE_SIZE"},
    {-70, "CL_INVALID_DEVICE_QUEUE"},
    {-71, "CL_INVALID_SPEC_ID"},
    {-72, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"},
};
#define CHECK_OCL_ERROR(err, msg)                                                                            \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
    }

class gpu_p2p_helper {
public:
    gpu_p2p_helper() {}
    ~gpu_p2p_helper() {}

    uint64_t derive_handle(cl_mem clbuf) {
        cl_int err;
        uint64_t fd;
        err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(fd), &fd, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_ALLOCATION_HANDLE_INTEL failed");
        return fd;
    }

    cl_mem map_remote_mem(cl_context context, cl_mem clbuf, size_t size) {
        cl_int err;
        uint64_t fd = derive_handle(clbuf);
        // Create extMemBuffer of type cl_mem from fd.
        cl_mem_properties extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(context, extMemProperties, 0, size, NULL, &err);
        if (err < 0) {
            OPENVINO_ASSERT(false,
                            "clCreateBufferWithProperties failed, clbuf = %p, fd = %ld, size = %ld, new_cl_mem = %p\n",
                            clbuf,
                            fd,
                            size,
                            extMemBuffer);
        }

        return extMemBuffer;
    }

    cl_mem map_remote_mem(cl_context context, uint64_t fd, size_t size) {
        cl_int err;
        // Create extMemBuffer of type cl_mem from fd.
        cl_mem_properties extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(context, extMemProperties, 0, size, NULL, &err);
        CHECK_OCL_ERROR(err, "clCreateBufferWithProperties - CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR failed");

        return extMemBuffer;
    }

    void destory_remote_mem(cl_mem clbuf) {
        clReleaseMemObject(clbuf);
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

    void remote_copy(cldnn::stream& stream, cl_mem src, cl_mem dst, size_t size) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, size, 0, NULL, &ret);
        clWaitForEvents(1, &ret);  // blocked copy
        clReleaseEvent(ret);
        return;
    }
};

class simple_tensor_add {
public:
    simple_tensor_add() {}
    ~simple_tensor_add() {
        for (auto& item : kernels) {
            if (item.second)
                clReleaseKernel(item.second);
        }
        kernels.clear();
        if (program)
            clReleaseProgram(program);
    }

    typedef enum _kernel_data_type {
        e_type_fp16 = 0,
        e_type_int8 = 1,
        e_type_fp32 = 2,
    } kernel_data_type;

    kernel_data_type element_type_to_kernel_data_type(ov::element::Type_t element_type) {
        switch (element_type) {
        case ov::element::f16:
            return kernel_data_type::e_type_fp16;
        case ov::element::i8:
            return kernel_data_type::e_type_int8;
        case ov::element::f32:
            return kernel_data_type::e_type_fp32;
        default:
            OPENVINO_THROW("Error: unsupported element type for kernel adder - ",
                           ov::element::Type(element_type).to_string().c_str());
            break;
        }
        return kernel_data_type::e_type_int8;
    }

    cl_kernel create_kernel(cldnn::stream& stream, const char* kernel_code, const char* kernelName) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        std::cout << "create_kernel: name = " << kernelName << std::endl;

        cl_uint knlcount = 1;
        const char* knlstrList[] = {kernel_code};
        size_t knlsizeList[] = {strlen(kernel_code)};

        cl_context context = ocl_stream.get_engine().get_cl_context().get();
        program = clCreateProgramWithSource(context, knlcount, knlstrList, knlsizeList, &err);
        CHECK_OCL_ERROR(err, "clCreateProgramWithSource failed");

        std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
        err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
        if (err < 0) {
            size_t logsize = 0;
            auto device = ocl_stream.get_engine().get_cl_device().get();
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
            CHECK_OCL_ERROR(err, "clGetProgramBuildInfo failed");

            std::vector<char> logbuf(logsize + 1, 0);
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
            std::cout << "clGetProgramBuildInfo failed: " << logbuf.data() << std::endl;
            // OPENVINO_ASSERT(err >= 0, "clGetProgramBuildInfo: ", logbuf.data());
        }
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        CHECK_OCL_ERROR(err, "clCreateKernel failed");
        return kernel;
    }

    cl_kernel get_or_create_kernel_if_possible_sub(cldnn::stream& stream,
                                                   kernel_data_type type,
                                                   size_t offset) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = kernels.find(type);
        if (it != kernels.end()) {
            return it->second;
        }

#define ADD_OP_KERNEL_SOURCE_CODE(DATA_TYPE)                                                            \
    "kernel void tensor_add_kernel_" #DATA_TYPE "(const global " #DATA_TYPE " *src, global " #DATA_TYPE \
    " *dst, int offset) {"                                                                              \
    "const int id = get_global_id(0);"                                                                  \
    "const int idx = id + offset;"                                                                      \
    "dst[idx] += src[id];"                                                                              \
    "}"
        if (type == kernel_data_type::e_type_fp16) {
            const char tensor_add_kernel_fp16[] = ADD_OP_KERNEL_SOURCE_CODE(half);
            const char kernel_name[] = "tensor_add_kernel_half";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp16, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_int8) {
            const char tensor_add_kernel_int8[] = ADD_OP_KERNEL_SOURCE_CODE(char);
            const char kernel_name[] = "tensor_add_kernel_char";
            kernels[type] = create_kernel(stream, tensor_add_kernel_int8, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_fp32) {
            const char tensor_add_kernel_fp32[] = ADD_OP_KERNEL_SOURCE_CODE(float);
            const char kernel_name[] = "tensor_add_kernel_float";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp32, kernel_name);
            return kernels[type];
        } else {
            OPENVINO_THROW("error: unsupported adder kernel data type ", static_cast<int>(type));
        }
#undef ADD_OP_KERNEL_SOURCE_CODE
        return kernels[type];
    }

    event::ptr tensor_add_sub(cldnn::stream& stream,
                              cl_mem src,
                              cl_mem dst,
                              size_t element_count,
                              kernel_data_type data_type,
                              size_t offset) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        if (src == nullptr || dst == nullptr) {
            std::cout << "tensor_add: invalid arguments!" << std::endl;
        }
        OPENVINO_ASSERT(src != nullptr && dst != nullptr, "tensor_add: invalid arguments!");

        cl_kernel kernel = get_or_create_kernel_if_possible_sub(stream,
                                                                data_type,
                                                                static_cast<int>(offset));

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        CHECK_OCL_ERROR(err, "clSetKernelArg src failed");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        err = clSetKernelArg(kernel, 2, sizeof(int), &offset);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        size_t global_size[] = {element_count};
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, &ret);
        CHECK_OCL_ERROR(err, "clEnqueueNDRangeKernel failed");

        return ocl_stream.create_event(cl::Event(ret));
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

private:
    cl_program program;
    std::mutex mutex;
    std::map<kernel_data_type, cl_kernel> kernels;
};

static gpu_p2p_helper& get_p2p_instance() {
    static gpu_p2p_helper gpu_p2p_instance;
    return gpu_p2p_instance;
}

static simple_tensor_add& get_adder_instance(size_t idx) {
    static simple_tensor_add adder_instance[4];
    return adder_instance[idx];
}

struct sync_tensor_impl : public typed_primitive_impl<sync_tensor> {
    using parent = typed_primitive_impl<sync_tensor>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::sync_tensor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<sync_tensor_impl>(*this);
    }

    sync_tensor_impl() : parent() {}

    ~sync_tensor_impl() {}

    explicit sync_tensor_impl(const sync_tensor_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<sync_tensor>());
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    bool update_internal_buffer(sync_tensor_inst& instance,
                                std::vector<cldnn::memory::ptr>& bufs,
                                cldnn::layout& last_layout,
                                cldnn::layout& layout,
                                size_t w_size,
                                size_t w_rank) {
        auto& engine = instance.get_network().get_engine();
        size_t required_size = layout.bytes_count();
        bool allocated = false;

        auto need_realloc = [&](size_t idx) {
            if (idx != 1)
                return false;

            if (bufs[idx] == nullptr || last_layout.bytes_count() == 0)
                return true;

            if (bufs[idx]->size() * w_size < required_size)
                return true;

            // Batch has been changed to smaller, need reallocate to decrease memory.
            auto last_batch = last_layout.batch() * last_layout.feature();
            auto batch = layout.batch() * layout.feature();
            if (last_batch > batch)
                return true;
            return false;
        };
        for (size_t i = 0; i < w_size; i++) {
            if (!need_realloc(i)) {
                continue;
            }
            // size_t origin_size = bufs[i] != nullptr ? bufs[i]->size() : 0;
            bufs[i] = nullptr;
            auto width = layout.get_shape()[-1];
            auto sub_width = width / w_size;
            if (sub_width * w_size != width)
                std::cout << "[Warning] the shape of FC output has ODD number!!!" << std::endl;
            auto sub_layout = layout;
            auto sub_shape = layout.get_shape();
            sub_shape[-1] = sub_width;
            sub_layout.set_partial_shape(sub_shape);
            bufs[i] = engine.allocate_memory(sub_layout, cldnn::allocation_type::cl_mem, false);
            allocated = true;
        }
        return allocated;
    }

    void release_remote_mems(cl_mem remote_mems) {
        if (remote_mems) {
            auto _cl_mem = static_cast<cl_mem>(remote_mems);
            auto ret = clReleaseMemObject(_cl_mem);
            CHECK_OCL_ERROR(ret, "[ERROR] release_remote_mems failed!!!");
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();

        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto is_all_reduce = instance.get_impl_params()->need_add == true;

        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto id = 0;
        // auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);
        while (true) {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->updated_flag = false;
            if (sub_mem_mgr->_use_count[id] == w_size) {
                sub_mem_mgr->_use_count[id] = 0;
                for (size_t i = 0; i < w_size; i++) {
                    sub_mem_mgr->_memorys_table[id][i].flag = false;
                    sub_mem_mgr->_memorys_table[id][i].all_gather_read_flag = false;
                    sub_mem_mgr->_memorys_table[id][i].all_gather_copy_flag = false;
                    for (size_t j = 0; j < w_size; j++) {
                        sub_mem_mgr->_memorys_table[id][i].events[j] = nullptr;
                        sub_mem_mgr->_memorys_table[id][i].all_reduce_copy_flag[j] = false;
                        sub_mem_mgr->_memorys_table[id][i].all_reduce_concat_flag[j] = false;
                        sub_mem_mgr->_memorys_table[id][i].all_reduce_concat_flag2[j] = false;
                        sub_mem_mgr->_memorys_table[id][i].all_reduce_add_flag[j] = false;
                    }
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
        }

        gpu_p2p_helper& gpu_p2p_instance = get_p2p_instance();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        auto local_context = ocl_stream.get_engine().get_cl_context().get();
        auto p2p_src_layout = instance.get_output_layout(0);
        bool need_update_remote_mems = false;
        if (is_all_reduce) {
            OPENVINO_ASSERT(1 == instance.get_output_memorys().size(), "All reduce only has one output!");
            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0] = instance.get_output_memorys()[0];
            // Allocate or reuse buffer for P2P target, same shape with output[0]
            p2p_src_layout = instance.get_output_layout(0);
            need_update_remote_mems = update_internal_buffer(instance,
                                                             sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs,
                                                             sub_mem_mgr->_memorys_table[id][w_rank].layout,
                                                             p2p_src_layout,
                                                             w_size,
                                                             w_rank);
        }

        if (is_all_reduce) {
            sub_mem_mgr->_memorys_table[id][w_rank].all_reduce_copy_flag[0] = true;
            sub_mem_mgr->_memorys_table[id][w_rank].all_reduce_concat_flag[0] = true;
            sub_mem_mgr->_memorys_table[id][w_rank].all_reduce_concat_flag2[0] = true;
        } else {
            sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        }

        auto split_parts = [](int len, int n) {
            int average = len / n;
            std::vector<int> parts(n, average);
            parts.back() = len - average * (n - 1);
            return parts;
        };
        auto output_layout = instance.get_output_layout(0);
        ov::element::Type output_element_type = output_layout.data_type;
        auto output_element_size = output_element_type.size();
        auto output_shape = output_layout.get_shape();

        if (is_all_reduce) {
            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].all_reduce_copy_flag[0] == true)
                        wait_all_ouput_ready++;
                }
                if (wait_all_ouput_ready == w_size) {
                    break;
                }
            }
            auto sub_out_dim_vec = split_parts(output_shape[-1], w_size);
            auto output_height = ov::shape_size(output_shape) / output_shape[-1];
            auto dst_idx = (w_rank + 1) % w_size;
            // Prepare CL memory mapping for P2P copying next
            {
                cl_mem dst_cl_buf = nullptr;
                dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0]);
                if (need_update_remote_mems || dst_cl_buf == nullptr) {
                    if (dst_cl_buf) {
                        release_remote_mems(dst_cl_buf);
                    }
                    cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].recv_bufs[1];
                    auto dst_cl_buf_remote =
                        std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                    dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, dst_mem->size());
                    sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0] = dst_cl_buf;
                }
            }
            // Start loop for sub buff copy & add
            auto mem_recv_1 = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[1];
            auto cl_buf_recv_1 = std::dynamic_pointer_cast<const ocl::gpu_buffer>(mem_recv_1)->get_buffer().get();
            auto mem_recv_0 =
                std::dynamic_pointer_cast<const ocl::gpu_buffer>(sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
            auto cl_buf_recv_0 = mem_recv_0->get_buffer().get();
            for (int32_t i = 0; i < static_cast<int>(w_size) - 1; i++) {
                int32_t sub_part = (w_rank - i) >= 0 ? (w_rank - i) : ((w_rank - i) + w_size);
                int32_t rec_sub_part = (sub_part - 1) >= 0 ? (sub_part - 1) : ((sub_part - 1) + w_size);

                auto dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0]);

                int32_t off_set = 0;
                for (int32_t j = 0; j < sub_part; j++) {
                    off_set = off_set + sub_out_dim_vec[j];
                }
                size_t src_rec[3] = {off_set * output_element_size * output_height, 0, 0};
                size_t dst_rec[3] = {0, 0, 0};
                size_t rect[3] = {sub_out_dim_vec[sub_part] * output_element_size, output_height, 1};
                auto ret = clEnqueueCopyBufferRect(queue,
                                                   cl_buf_recv_0,
                                                   dst_cl_buf,
                                                   src_rec,
                                                   dst_rec,
                                                   rect,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   nullptr);
                ret = clFinish(queue);
                CHECK_OCL_ERROR(ret, "clEnqueueCopyBufferRect failed");

                sub_mem_mgr->_memorys_table[id][dst_idx].all_reduce_copy_flag[i + 1] = true;
                while (true) {
                    if (sub_mem_mgr->_memorys_table[id][w_rank].all_reduce_copy_flag[i + 1] == true)
                        break;
                }

                int32_t off_set_add = 0;
                for (int32_t j = 0; j < rec_sub_part; j++)
                    off_set_add = off_set_add + sub_out_dim_vec[j];

                cldnn::event::ptr sync_add_event;
                auto& adder_instance = get_adder_instance(w_rank);
                sync_add_event =
                    adder_instance.tensor_add_sub(stream,
                                                  cl_buf_recv_1,
                                                  cl_buf_recv_0,
                                                  output_height * sub_out_dim_vec[rec_sub_part],
                                                  adder_instance.element_type_to_kernel_data_type(output_element_type),
                                                  output_height * off_set_add);
                sync_add_event->wait();

                sub_mem_mgr->_memorys_table[id][w_rank].all_reduce_add_flag[i + 1] = true;
                while (true) {
                    size_t wait_all_ouput_ready = 0;
                    for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                        if (sub_mem_mgr->_memorys_table[id][idx].all_reduce_add_flag[i + 1] == true)
                            wait_all_ouput_ready++;
                    }
                    if (wait_all_ouput_ready == w_size)
                        break;
                }
            }

            for (int32_t i = 0; i < static_cast<int>(w_size) - 1; i++) {
                int32_t sub_part = (w_rank - i + 1) % w_size;
                int32_t rec_sub_part = (sub_part - 1) < 0 ? (w_size - 1) : (sub_part - 1) % w_size;
                {
                    cl_mem dst_cl_buf = nullptr;
                    dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0]);

                    int32_t off_set = 0;
                    for (int32_t j = 0; j < sub_part; j++) {
                        off_set = off_set + sub_out_dim_vec[j];
                    }
                    size_t src_rec[3] = {off_set * output_element_size * output_height, 0, 0};
                    size_t dst_rec[3] = {0, 0, 0};
                    size_t rect[3] = {sub_out_dim_vec[sub_part] * output_element_size, output_height, 1};
                    auto ret = clEnqueueCopyBufferRect(queue,
                                                       cl_buf_recv_0,
                                                       dst_cl_buf,
                                                       src_rec,
                                                       dst_rec,
                                                       rect,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       nullptr);
                    ret = clFinish(queue);
                    CHECK_OCL_ERROR(ret, "clEnqueueCopyBufferRect failed");

                    sub_mem_mgr->_memorys_table[id][dst_idx].all_reduce_concat_flag[i + 1] = true;
                    while (true) {
                        size_t wait_all_ouput_ready = 0;
                        for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                            if (sub_mem_mgr->_memorys_table[id][idx].all_reduce_concat_flag[i + 1] == true) {
                                wait_all_ouput_ready++;
                            }
                        }
                        if (wait_all_ouput_ready == w_size)
                            break;
                    }
                }

                {
                    int32_t off_set = 0;
                    for (int32_t j = 0; j < rec_sub_part; j++) {
                        off_set = off_set + sub_out_dim_vec[j];
                    }
                    size_t dst_rec[3] = {off_set * output_element_size * output_height, 0, 0};
                    size_t src_rec[3] = {0, 0, 0};
                    size_t rect[3] = {sub_out_dim_vec[rec_sub_part] * output_element_size, output_height, 1};
                    auto ret = clEnqueueCopyBufferRect(queue,
                                                       cl_buf_recv_1,
                                                       cl_buf_recv_0,
                                                       src_rec,
                                                       dst_rec,
                                                       rect,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       nullptr);
                    ret = clFinish(queue);
                    CHECK_OCL_ERROR(ret, "clEnqueueCopyBufferRect failed");

                    sub_mem_mgr->_memorys_table[id][w_rank].all_reduce_concat_flag2[i + 1] = true;
                    while (true) {
                        size_t wait_all_ouput_ready = 0;
                        for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                            if (sub_mem_mgr->_memorys_table[id][idx].all_reduce_concat_flag2[i + 1] == true) {
                                wait_all_ouput_ready++;
                            }
                        }
                        if (wait_all_ouput_ready == w_size)
                            break;
                    }
                }
            }
        } else {
            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].flag) {
                        wait_all_ouput_ready++;
                    }
                }
                if (wait_all_ouput_ready == w_size)
                    break;
            }

            auto src_p2p_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.get_output_memorys()[1]);
            auto src_cl_buf = src_p2p_buf->get_buffer().get();

            auto sub_out_dim_vec = split_parts(output_shape[-1], w_size);
            int32_t off_set = 0;
            for (int32_t j = 0; j < w_rank; j++) {
                off_set = off_set + sub_out_dim_vec[j];
            }
            size_t src_rec[3] = {0, 0, 0};
            size_t dst_rec[3] = {off_set * output_element_size, 0, 0};
            size_t rect[3] = {sub_out_dim_vec[w_rank] * output_element_size, 1, output_shape[0]};

            if (w_rank == 0) {
                std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                if (sub_mem_mgr->result != nullptr) {
                    free(sub_mem_mgr->result);
                    sub_mem_mgr->result = nullptr;
                }
                sub_mem_mgr->result = malloc(instance.get_output_memorys()[0]->size());
                sub_mem_mgr->updated_flag = true;
            } else {
                while (true) {
                    std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                    if (sub_mem_mgr->updated_flag)
                        break;
                }
            }

            auto ret = clEnqueueReadBufferRect(queue,
                                               src_cl_buf,
                                               CL_TRUE,
                                               src_rec,
                                               dst_rec,
                                               rect,
                                               sub_out_dim_vec[w_rank] * output_element_size,
                                               sub_out_dim_vec[w_rank] * output_element_size,
                                               output_shape[-1] * output_element_size,
                                               output_shape[-1] * output_element_size,
                                               sub_mem_mgr->result,
                                               0,
                                               nullptr,
                                               nullptr);
            CHECK_OCL_ERROR(ret, "clEnqueueReadBufferRect failed");
            clFinish(queue);

            sub_mem_mgr->_memorys_table[id][w_rank].all_gather_read_flag = true;

            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].all_gather_read_flag == true) {
                        wait_all_ouput_ready++;
                    }
                }
                if (wait_all_ouput_ready == w_size)
                    break;
            }

            instance.get_output_memorys()[0]
                ->copy_from(ocl_stream, sub_mem_mgr->result, 0, 0, instance.get_output_memorys()[0]->size(), true);

            sub_mem_mgr->_memorys_table[id][w_rank].all_gather_copy_flag = true;

            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].all_gather_copy_flag == true) {
                        wait_all_ouput_ready++;
                    }
                }
                if (wait_all_ouput_ready == w_size)
                    break;
            }
        }

        // This block MUST be put exactly at the end of this method.
        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_memorys_table[id][w_rank].layout = p2p_src_layout;
            sub_mem_mgr->_use_count[id]++;
        }
        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const sync_tensor_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<sync_tensor_impl>();
    }
};

namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::dynamic_shape, sync_tensor_impl::create, {});
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::static_shape, sync_tensor_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)
