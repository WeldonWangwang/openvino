// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/xattn_find_block.hpp>

#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {
constexpr uint32_t kBlockShareMax = 256;
}

TEST(xattn_find_block_gpu, smoke_basic_mask) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::use_cm(true));

    const uint32_t stride = 16;
    const uint32_t block_size = 128;
    const uint32_t tokens_per_block = block_size / stride;
    const uint32_t token_share_max = kBlockShareMax / tokens_per_block;

    const uint32_t heads_num = 1;
    const uint32_t head_size = 128;
    const uint32_t q_stride = 8;
    const uint32_t k_stride = 32;
    const uint32_t q_stride_pad = 8;
    const uint32_t q_block_pad = 1;
    const uint32_t k_block_pad = 32;
    const uint32_t q_len = q_stride * stride;
    const int32_t causal_start_index = 0;
    const float thresh = 0.0f;

    const uint32_t max_elements = q_stride_pad * (k_block_pad / token_share_max);
    const uint32_t exp_elements = q_stride_pad * k_block_pad;
    const uint32_t mask_elements = k_block_pad;

    auto kq_max_layout = layout{ data_types::f32, format::bfyx, { 1, 1, 1, static_cast<int>(max_elements) } };
    auto kq_exp_layout = layout{ data_types::f32, format::bfyx, { 1, 1, 1, static_cast<int>(exp_elements) } };

    auto kq_max_mem = engine.allocate_memory(kq_max_layout);
    auto kq_exp_mem = engine.allocate_memory(kq_exp_layout);

    set_values(kq_max_mem, std::vector<float>(max_elements, 0.0f));
    set_values(kq_exp_mem, std::vector<float>(exp_elements, 1.0f));

    topology topo;
    topo.add(input_layout("kq_max", kq_max_mem->get_layout()));
    topo.add(input_layout("kq_exp", kq_exp_mem->get_layout()));
    topo.add(xattn_find_block("find",
                              {input_info("kq_max"), input_info("kq_exp")},
                              heads_num,
                              head_size,
                              q_len,
                              q_stride,
                              k_stride,
                              q_stride_pad,
                              q_block_pad,
                              k_block_pad,
                              causal_start_index,
                              thresh,
                              stride,
                              block_size,
                              true,
                              false));

    network network(engine, topo, config);
    network.set_input_data("kq_max", kq_max_mem);
    network.set_input_data("kq_exp", kq_exp_mem);

    auto outputs = network.execute();
    ASSERT_TRUE(outputs.count("find"));
    auto output_mem = outputs.at("find").get_memory();

    cldnn::mem_lock<uint8_t> output_ptr(output_mem, network.get_stream());

    std::vector<uint8_t> expected(mask_elements, 0);
    expected[0] = 1;

    for (size_t i = 0; i < mask_elements; ++i) {
        ASSERT_EQ(expected[i], output_ptr[i]);
    }
}
