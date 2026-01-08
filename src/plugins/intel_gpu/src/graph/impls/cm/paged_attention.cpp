// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <utility>

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/util/file_util.hpp"
#include "paged_attention_gen.hpp"
#include "paged_attention_inst.h"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "to_string_utils.h"

namespace ov::intel_gpu::cm {

#ifdef GPU_DEBUG_CONFIG
namespace {

float convert_element(int64_t i) { return static_cast<float>(i); }
float convert_element(int32_t i) { return static_cast<float>(i); }
float convert_element(float f) { return f; }
float convert_element(ov::float16 h) { return static_cast<float>(h); }

size_t get_x_pitch(const cldnn::layout& layout) {
    try {
        auto tensor_x0 = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(0, 0, 0, 0));
        auto tensor_x1 = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    } catch (...) {
        // When spatial size of x=0, x_pitch is meaningless.
        return 0;
    }
}

template <class T>
void dump(cldnn::memory::ptr mem, cldnn::stream& stream, std::ofstream& file_stream, bool dump_raw) {
    auto&& size = mem->get_layout().get_tensor();

    auto batch_size =
        std::max<ov::Dimension::value_type>(std::min<ov::Dimension::value_type>(cldnn::ExecutionConfig::get_dump_batch_limit(), size.batch[0]), 1);
    cldnn::tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count() << ", addr: " << mem->buffer_ptr()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count() << ", addr: " << mem->buffer_ptr()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        file_stream << "Empty buffer" << std::endl;
        return;
    }

    cldnn::mem_lock<T, cldnn::mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(mem->get_layout());
    std::stringstream buffer;

    if (!dump_raw) {
        for (ov::Dimension::value_type g = 0; g < size.group[0]; ++g) {
            for (ov::Dimension::value_type b = 0; b < batch_size; ++b) {
                for (ov::Dimension::value_type f = 0; f < size.feature[0]; ++f) {
                    for (ov::Dimension::value_type w = 0; w < size.spatial[3]; ++w) {
                        for (ov::Dimension::value_type z = 0; z < size.spatial[2]; ++z) {
                            for (ov::Dimension::value_type y = 0; y < size.spatial[1]; ++y) {
                                cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                                size_t input_it = mem->get_layout().get_linear_offset(t);

                                for (ov::Dimension::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                    buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[input_it]) << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < lock.size(); ++i) {
            buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[i]) << std::endl;
        }
    }
    file_stream << buffer.str();
}

void unpack(cldnn::data_types type, uint8_t input, int8_t& v0, int8_t& v1) {
    if (type == cldnn::data_types::i4) {
        char s_bit = (input & 0x08);
        char mask = s_bit > 0 ? 0xF0 : 0x00;
        v0 = (input & 0x0F) | mask;

        input >>= 4;
        s_bit = (input & 0x08);
        mask = s_bit > 0 ? 0xF0 : 0x00;
        v1 = (input & 0x0F) | mask;
    } else if (type == cldnn::data_types::u4) {
        v0 = input & 0x0F;
        v1 = input >> 4;
    } else {
        OPENVINO_ASSERT(false, "not supported unpacking");
    }
}

void dump_i4u4(cldnn::data_types type, cldnn::memory::ptr mem, cldnn::stream& stream, std::ofstream& file_stream, bool dump_raw) {
    auto&& size = mem->get_layout().get_tensor();

    auto batch_size =
        std::max<ov::Dimension::value_type>(std::min<ov::Dimension::value_type>(cldnn::ExecutionConfig::get_dump_batch_limit(), size.batch[0]), 1);
    cldnn::tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count() << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count() << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format)
                    << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        file_stream << "Empty buffer" << std::endl;
        return;
    }

    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    std::stringstream buffer;

    if (dump_raw) {
        for (size_t i = 0; i < lock.size(); ++i) {
            int8_t v0, v1;
            unpack(type, mem_ptr[i], v0, v1);
            buffer << std::fixed << std::setprecision(6) << static_cast<int>(v0) << std::endl;
            buffer << std::fixed << std::setprecision(6) << static_cast<int>(v1) << std::endl;
        }
    } else {
        GPU_DEBUG_COUT << " supports raw dump only" << std::endl;
    }
    file_stream << buffer.str();
}

std::string get_name_for_dump(const std::string& file_name) {
    std::string filename = file_name;
    std::replace(filename.begin(), filename.end(), '\\', '_');
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), ':', '_');
    return filename;
}

void log_memory_to_file(cldnn::memory::ptr mem, cldnn::layout data_layout, cldnn::stream& stream, const std::string& filename, bool dump_raw) {
    std::ofstream file_stream(filename);
    if (!mem) {
        file_stream << "Empty" << std::endl;
        return;
    }

    auto actual_mem = mem->get_engine()->reinterpret_buffer(*mem, data_layout);

    auto mem_dt = actual_mem->get_layout().data_type;
    if (mem_dt == cldnn::data_types::f32)
        dump<float>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::f16)
        dump<ov::float16>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i64)
        dump<int64_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i32)
        dump<int32_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i8)
        dump<int8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::u8)
        dump<uint8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::boolean)
        dump<uint8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i4 || mem_dt == cldnn::data_types::u4)
        dump_i4u4(mem_dt, actual_mem, stream, file_stream, dump_raw);
    else
        GPU_DEBUG_COUT << "Dump for this data type is not supported: " << dt_to_str(mem_dt) << std::endl;
}

std::string get_file_path_for_binary_dump(cldnn::layout layout, const std::string& name, const std::string& dump_layers_path) {
    std::string filename;
    std::string data_type = ov::element::Type(layout.data_type).get_type_name();
    std::string format = layout.format.to_string();
    std::string tensor;
    auto dims = layout.get_dims();
    for (size_t r = 0; r < layout.get_rank(); r++) {
        tensor += ("_" + to_string(dims[r]));
    }

    std::string layer_name = get_name_for_dump(name);
    filename = dump_layers_path + layer_name + "__" + data_type + "_" + tensor + "__" + format + ".bin";
    return filename;
}

bool is_target_iteration(int64_t iteration, const std::set<int64_t>& dump_iteration) {
    if (iteration < 0)
        return true;

    if (dump_iteration.empty())
        return true;

    if (dump_iteration.find(iteration) == std::end(dump_iteration))
        return false;

    return true;
}

bool is_layer_name_matched(const std::string& layer_name, const std::string& pattern) {
    auto upper_layer_name = std::string(layer_name.length(), '\0');
    std::transform(layer_name.begin(), layer_name.end(), upper_layer_name.begin(), ::toupper);
    auto upper_pattern = std::string(pattern.length(), '\0');
    std::transform(pattern.begin(), pattern.end(), upper_pattern.begin(), ::toupper);

    size_t pos = upper_layer_name.find(':');
    auto upper_exec_graph_name = upper_layer_name.substr(pos + 1, upper_layer_name.size());
    if (upper_exec_graph_name.compare(upper_pattern) == 0) {
        return true;
    }

    std::regex re(upper_pattern);
    return std::regex_match(upper_layer_name, re);
}

bool is_layer_for_dumping(const cldnn::ExecutionConfig& config, const std::string& layer_name) {
    const auto& dump_layers = config.get_dump_layer_names();
    if (dump_layers.empty())
        return true;

    auto iter = std::find_if(dump_layers.begin(), dump_layers.end(), [&](const std::string& dl) {
        return is_layer_name_matched(layer_name, dl);
    });
    return (iter != dump_layers.end());
}

std::string get_file_prefix(const cldnn::primitive_inst& instance) {
    auto prog = instance.get_network().get_program().get();
    auto prog_id = ((prog != nullptr) ? prog->get_id() : 0);
    auto net_id = instance.get_network().get_id();
    auto iter = instance.get_network().get_current_iteration_num();
    auto iteration_prefix = iter < 0 ? std::string("") : std::to_string(iter) + "_";

    return "program" + std::to_string(prog_id) + "_network" + std::to_string(net_id) + "_" + iteration_prefix + instance.id();
}

void dump_updated_src_after_exec(const cldnn::primitive_inst& instance) {
    const auto& config = instance.get_config();
    if (config.get_dump_tensors_path().empty() || !config.get_dump_src_after_exec())
        return;

    if (!is_target_iteration(instance.get_network().get_current_iteration_num(), config.get_dump_iterations()))
        return;

    const std::string layer_name = instance.id();
    if (config.get_dump_tensors() == ov::intel_gpu::DumpTensors::in || !is_layer_for_dumping(config, layer_name))
        return;

    auto& stream = instance.get_network().get_stream();
    stream.finish();
    for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
        std::string name = get_file_prefix(instance) + "_updated_src_" + std::to_string(i);
        auto output_mem = instance.input_memory_ptr(i);
        if (output_mem == nullptr) {
            GPU_DEBUG_COUT << " updated_input_mem is nullptr. Nothing to dump." << std::endl;
            continue;
        }

        auto& output_layout = instance.get_input_layout(i);
        if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary) {
            auto filename = get_file_path_for_binary_dump(output_layout, name, config.get_dump_tensors_path());

            cldnn::mem_lock<char, cldnn::mem_lock_type::read> lock(output_mem, stream);
            ov::util::save_binary(filename, lock.data(), output_mem->size());
            GPU_DEBUG_COUT << " Dump layer dst : " << layer_name << " to " << filename << std::endl;
        } else {
            const bool dump_raw = config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::text_raw;
            GPU_DEBUG_COUT << " Dump " << (dump_raw ? "raw " : "") << name << std::endl;
            auto filename = config.get_dump_tensors_path() + get_name_for_dump(name) + ".txt";
            log_memory_to_file(output_mem, output_layout, stream, filename, dump_raw);
        }
    }
}
}  // namespace
#endif  // GPU_DEBUG_CONFIG

class PagedAttentionCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedAttentionCmImpl)

    Stage::Ptr kv_cache_update = make_stage<PagedAttentionGeneratorKVCacheUpdate>();
    Stage::Ptr pa_single_token = make_stage<PagedAttentionGeneratorSingleToken>();
    Stage::Ptr pa_single_token_finalization = make_stage<PagedAttentionGeneratorSingleTokenFinalization>();
    Stage::Ptr pa_multi_token = make_stage<PagedAttentionGeneratorMultiToken>();
    Stage::Ptr xattn_estimate_gemmqk = make_stage<XAttentionEstimateGEMMQK>();
    Stage::Ptr xattn_estimate_find_block = make_stage<XAttentionEstimateFindBlock>();
    Stage::Ptr xattn_estimate_post_proc = make_stage<XAttentionEstimatePostProc>();

    PagedAttentionCmImpl() : PrimitiveImplCM(PagedAttentionImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
    }
    explicit PagedAttentionCmImpl(const kernel_impl_params& params) : PagedAttentionCmImpl() {
        const auto desc = params.typed_desc<paged_attention>();

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::PagedAttentionCmImpl()" << std::endl;
        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_multi_token, params);
        if (desc->has_xattention) {
            add_stage(xattn_estimate_gemmqk, params);
            add_stage(xattn_estimate_find_block, params);
            add_stage(xattn_estimate_post_proc, params);
        }
    }

    void update_xattn_rt_params(const kernel_impl_params& params) {
        const auto desc = params.typed_desc<paged_attention>();

        // XAttention estimate is following afer kvcache_update.
        auto out_shape = params.output_layouts[0].get_shape();
        const size_t block_size = get_xattn_block_size(params);
        const size_t kv_len = get_max_context_len(params);
        const size_t q_len = out_shape[0];
        const size_t N = kv_len / STRIDE;
        const size_t N_kq_groups = ceil_div(N, BLOCK_WG_N);

        const auto q_block_pad = ceil_div(q_len, block_size);
        const auto sum_per_token_in_block = block_size / STRIDE;
        const auto k_block_in_group = BLOCK_WG_N / sum_per_token_in_block;
        const auto k_block_pad = k_block_in_group * N_kq_groups;

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        rt_params->q_block_pad = q_block_pad;
        rt_params->k_block_pad = k_block_pad;
        rt_params->q_block_pad_merged = ceil_div(q_block_pad, MERGED_Q_NUM);

        const size_t head_size = desc->k_head_size;

        const auto M = q_len / STRIDE;  //# will slient drop the tails which is less than `stride`
        const auto K = STRIDE * head_size;

        const size_t q_stride_pad = round_up_to(M, BLOCK_WG_M);

        rt_params->N_kq_groups = N_kq_groups;
        rt_params->M = M;
        rt_params->N = N;
        rt_params->K = K;
        rt_params->q_stride_pad = q_stride_pad;
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
        }

        const auto& params = *instance.get_impl_params();
        OPENVINO_ASSERT(!params.is_dynamic());
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        const auto& desc = params.typed_desc<paged_attention>();

        rt_params->stage = get_paged_attention_stage(params);
        const auto max_context_len = get_max_context_len(params);
        rt_params->max_context_len = max_context_len;
        GPU_DEBUG_TRACE_DETAIL << "update_rt_params for stage: " << static_cast<size_t>(rt_params->stage) << "  max_context_len: " << rt_params->max_context_len
                               << std::endl;

        if (rt_params->stage == PagedAttentionStage::GENERATE) {
            auto partition_size = get_partition_size(desc->has_xattention);
            rt_params->num_of_partitions = ceil_div(max_context_len, partition_size);

            GPU_DEBUG_TRACE_DETAIL << "  partition_size: " << partition_size << "  num_of_partitions: " << rt_params->num_of_partitions << std::endl;
        } else {
            if (desc->has_xattention) {
                update_xattn_rt_params(params);
            }
        }
    }

    // update impl_parameter and rt_parameter
    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplCM::update(inst, impl_params);
        update_rt_params(inst);
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_attention>();

        update_stages_flags(instance);
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::execute():  stage = " << static_cast<int>(rt_params->stage) << std::endl;
        std::vector<event::ptr> res_event = events;
        res_event = {execute_stage(res_event, instance, kv_cache_update)};

        if (rt_params->stage == PagedAttentionStage::PREFILL || rt_params->stage == PagedAttentionStage::MIXED) {
            if (has_stage(xattn_estimate_gemmqk) && !bypass_xattn(params)) {
                res_event = {execute_stage(res_event, instance, xattn_estimate_gemmqk)};
                res_event = {execute_stage(res_event, instance, xattn_estimate_find_block)};
                res_event = {execute_stage(res_event, instance, xattn_estimate_post_proc)};
            }
            res_event = {execute_stage(res_event, instance, pa_multi_token)};
        } else if (rt_params->stage == PagedAttentionStage::GENERATE) {
            res_event = {execute_stage(res_event, instance, pa_single_token)};
            res_event = {execute_stage(res_event, instance, pa_single_token_finalization)};
        }
#ifdef GPU_DEBUG_CONFIG
        dump_updated_src_after_exec(instance);
#endif
        return res_event[0];
    }

    bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const override {
        const auto stage = get_paged_attention_stage(impl_params);

        // In case of MIXED mode execution Paged Attention may require dispatch data update and internal
        // buffers reallocation even if the input shapes haven't been changed. Therefore, check the current execution
        // mode and update parameters if needed
        return stage == PagedAttentionStage::MIXED;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        std::vector<BufferDescriptor> internal_buffers;

        const auto desc = params.typed_desc<paged_attention>();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        // Assume rt_params are updated, because get_internal_buffer_descs surely occurs after update_rt_params.
        OPENVINO_ASSERT(rt_params != nullptr);

        const auto stage = rt_params->stage;
        GPU_DEBUG_TRACE_DETAIL << " stage = " << static_cast<int>(stage) << std::endl;
        if (stage == PagedAttentionStage::GENERATE) {
            OPENVINO_ASSERT(rt_params->num_of_partitions != 0);
            size_t num_of_partitions = rt_params->num_of_partitions;

            const auto& input = params.input_layouts[0];
            const int64_t total_tokens = input.get_partial_shape()[0].get_length();
            auto buf_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * num_of_partitions);
            auto tmp_out_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * desc->v_head_size * num_of_partitions);

            internal_buffers.emplace_back(tmp_out_elements_count, ov::element::f32);  // 0: intermediate partition output
            internal_buffers.emplace_back(buf_elements_count, ov::element::f32);      // 1: softmax exp_sums

            GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: tmp_out=" << tmp_out_elements_count * 4 << "  exp_sums=" << buf_elements_count * 4 << std::endl;
        } else {
            internal_buffers.emplace_back(16, ov::element::f32);  // 0: intermediate partition output
            internal_buffers.emplace_back(16, ov::element::f32);  // 1: softmax exp_sums

            // internal buffer for XAttention
            if (desc->has_xattention) {
                auto count_kq_max_wg = static_cast<int64_t>(desc->heads_num * rt_params->N_kq_groups * rt_params->q_stride_pad);
                internal_buffers.emplace_back(count_kq_max_wg, ov::element::f32);  // 2: kq_max_wg

                auto count_kq_exp_partial_sum = static_cast<int64_t>(desc->heads_num * rt_params->q_stride_pad * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_kq_exp_partial_sum, ov::element::f32);  // 3: kq_exp_partial_sum

                auto count_elements_mask = static_cast<int64_t>(desc->heads_num * rt_params->q_block_pad * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_mask, ov::element::boolean);  // 4: sparse_block_mask

                auto count_elements_mask_merged = static_cast<int64_t>(desc->heads_num * rt_params->q_block_pad_merged * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_mask_merged, ov::element::boolean);  // 5: sparse_block_mask_wg

                GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: count_kq_max_wg=" << count_kq_max_wg * 4
                                       << "  count_kq_exp_partial_sum=" << count_kq_exp_partial_sum * 4 << "  count_elements_mask=" << count_elements_mask * 1
                                       << "  count_elements_mask_merged=" << count_elements_mask_merged * 1 << std::endl;
            }
        }

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedAttentionCmImpl>(this);
    }
};

std::unique_ptr<primitive_impl> PagedAttentionImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<paged_attention>());
    try {
        return std::make_unique<PagedAttentionCmImpl>(params);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to create PagedAttentionCmImpl: ", e.what());
    }
}

}  // namespace ov::intel_gpu::cm
// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::PagedAttentionCmImpl)
