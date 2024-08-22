// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_pagedattention.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/op/sync_tensor.hpp"
#include "intel_gpu/op/util.hpp"
#include "openvino/op/slice.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/rank_constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/concat.hpp"
#include <cstdlib>

namespace ov {
namespace intel_gpu {

std::shared_ptr<ov::Node> PagedAttentionSplitInput::bfs_fc(std::shared_ptr<ov::Node> root_node) {
    const auto& users = root_node->get_users();
    if (users.size() != 1)
        return nullptr;
    auto cur_node = users[0];

    if (ov::is_type<ov::op::v0::Result>(cur_node)) {
        return nullptr;
    }

    if (ov::is_type<ov::op::PagedAttentionExtension>(cur_node)) {
        return nullptr;
    }

    if (ov::is_type<ov::intel_gpu::op::FullyConnected>(cur_node)) {
        // std::cout << "FullyConnected: " << cur_node->get_name() << ", " << cur_node->get_input_partial_shape(0)
        //           << std::endl;
        return cur_node;
    }
    if (ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(cur_node)) {
        // std::cout << "FullyConnectedCompressed: " << cur_node->get_name() << ", "
        //           << cur_node->get_input_partial_shape(0) << std::endl;
        return cur_node;
    }
    // std::cout << cur_node->get_name() << ", " << cur_node->get_input_partial_shape(0) << std::endl;
    return bfs_fc(cur_node);
}

PagedAttentionSplitInput::PagedAttentionSplitInput(size_t world_size, size_t rank_size) {
    using namespace ov::pass::pattern;

    auto in0 = any_input();
    auto in1 = any_input();
    auto in2 = any_input();
    auto in3 = any_input();
    auto in4 = any_input();
    auto in5 = any_input();
    auto in6 = any_input();
    auto in7 = any_input();
    auto in8 = any_input();
    auto in9 = any_input();
    auto in10 = any_input();
    auto in11 = any_input();
    auto in12 = any_input();
    auto fully_connected = wrap_type<ov::op::PagedAttentionExtension>({in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12});
    // auto fully_connected_compressed = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input()});
    // auto fully_connected_compressed_with_zp = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input(), any_input()});
    // auto paged_attention_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected,
    //                                                                                 fully_connected_compressed,
    //                                                                                 fully_connected_compressed_with_zp});
    auto paged_attention_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected});
    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected));

        std::shared_ptr<ov::Node> first_fc_after_pa = nullptr;
        {
            // std::cout << "---------------------Find FC after PA--------------------" << std::endl;
            auto root = m.get_match_root();
            if (root) {
                first_fc_after_pa = bfs_fc(root);
            }
        }

        const auto& m_data_in0 = pattern_map.at(in0).get_node_shared_ptr();
        const auto& m_data_in1 = pattern_map.at(in1).get_node_shared_ptr();
        const auto& m_data_in2 = pattern_map.at(in2).get_node_shared_ptr();
        const auto& m_data_in3 = pattern_map.at(in3).get_node_shared_ptr();
        const auto& m_data_in4 = pattern_map.at(in4).get_node_shared_ptr();
        const auto& m_data_in5 = pattern_map.at(in5).get_node_shared_ptr();
        const auto& m_data_in6 = pattern_map.at(in6).get_node_shared_ptr();
        const auto& m_data_in7 = pattern_map.at(in7).get_node_shared_ptr();
        const auto& m_data_in8 = pattern_map.at(in8).get_node_shared_ptr();
        const auto& m_data_in9 = pattern_map.at(in9).get_node_shared_ptr();
        const auto& m_data_in10 = pattern_map.at(in10).get_node_shared_ptr();
        const auto& m_data_in11 = pattern_map.at(in11).get_node_shared_ptr();
        const auto& m_data_in12 = pattern_map.at(in12).get_node_shared_ptr();

        auto print_shape = [&](const std::shared_ptr<ov::Node>& m_data) {
            // std::cout << m_data->get_friendly_name() << ": '";
            // for (size_t shape_id = 0; shape_id < m_data->get_output_partial_shape(0).size(); shape_id++) {
            //     if (!m_data->get_output_partial_shape(0)[shape_id].is_dynamic()) {
            //         int64_t len = m_data->get_output_partial_shape(0)[shape_id].get_length();
            //         std::cout << len << ", ";
            //     } else {
            //         std::cout << "?" << ", ";
            //     }
            // }
            // std::cout << "'\n";
        };

        std::shared_ptr<Node> m_pa = nullptr;
        if (pattern_map.find(fully_connected) != pattern_map.end())
            m_pa = pattern_map.at(fully_connected).get_node_shared_ptr();
        print_shape(m_data_in0);
        print_shape(m_data_in1);
        print_shape(m_data_in2);
        print_shape(m_data_in3);
        print_shape(m_data_in4);
        print_shape(m_data_in5);
        print_shape(m_data_in6);
        print_shape(m_data_in7);
        print_shape(m_data_in8);
        print_shape(m_data_in9);
        print_shape(m_data_in10);
        print_shape(m_data_in11);
        print_shape(m_data_in12);
        int w_rank = rank_size;
        int w_size = world_size;
        // std::cout << "w-size: " << w_size << std::endl;
        if (w_size != 1) {
            int slice_axis_length = m_data_in0->get_output_partial_shape(0)[-1].get_length();
            // std::cout << "slice_axis_length: " << slice_axis_length << std::endl;
            auto scop = std::div(slice_axis_length, w_size).quot;
            auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop});
            auto stop = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop});
            auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            int64_t input_axis_value = m_data_in0->get_output_partial_shape(0).size() - 1;
            // std::cout << "input_axis_value: " << input_axis_value << std::endl;
            auto input_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value});
            auto new_in0 = std::make_shared<ov::op::v8::Slice>(m_data_in0, start, stop, step, input_axis);
            print_shape(new_in0);

            int slice_axis_length1 = m_data_in1->get_output_partial_shape(0)[-1].get_length();
            auto scop1 = std::div(slice_axis_length1, w_size).quot;
            auto start1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop1});
            auto stop1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop1});
            auto step1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            int64_t input_axis_value1 = m_data_in1->get_output_partial_shape(0).size() - 1;
            auto input_axis1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value1});
            auto new_in1 = std::make_shared<ov::op::v8::Slice>(m_data_in1, start1, stop1, step1, input_axis1);

            int slice_axis_length2 = m_data_in2->get_output_partial_shape(0)[-1].get_length();
            auto scop2 = std::div(slice_axis_length2, w_size).quot;
            auto start2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop2});
            auto stop2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop2});
            auto step2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            int64_t input_axis_value2 = m_data_in2->get_output_partial_shape(0).size() - 1;
            auto input_axis2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value2});
            auto new_in2 = std::make_shared<ov::op::v8::Slice>(m_data_in2, start2, stop2, step2, input_axis2);

            OutputVector params = {new_in0,
                                   new_in1,
                                   new_in2,
                                   m_data_in3,
                                   m_data_in4,
                                   m_data_in5,
                                   m_data_in6,
                                   m_data_in7,
                                   m_data_in8,
                                   m_data_in9,
                                   m_data_in10,
                                   m_data_in11,
                                   m_data_in12};
            std::shared_ptr<Node> new_pa = nullptr;
            new_pa =
                std::make_shared<ov::op::PagedAttentionExtension>(params);

            // std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
            // sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(new_pa, w_size, 4096, new_pa->get_element_type());
            // sync_node->set_friendly_name(new_pa->get_friendly_name()+ "_TP");

            // auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
            // concat_node->set_friendly_name(new_pa->get_friendly_name()+ "_ALLGATHER");
            // copy_runtime_info(new_fc, concat_node);
            // for (auto& iter : org_users) {
            //     iter.second->input(iter.first).replace_source_output(concat_node->output(0));
            // }
            // new_fc->clear_control_dependencies();


            // copy_runtime_info(m_pa, concat_node);
            // replace_node(m_pa, concat_node);
            copy_runtime_info(m_pa, new_pa);
            replace_node(m_pa, new_pa);
            m_pa->clear_control_dependencies();

            if (first_fc_after_pa) {
                // std::cout << "---------------------Start split FC weights--------------------" << std::endl;
                // std::cout << first_fc_after_pa->get_name() << ": " << std::endl;
                auto compressed_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(first_fc_after_pa);
                // if (compressed_fc) {
                //     std::cout << "input[0]: " << compressed_fc->get_input_node_shared_ptr(0)->get_output_partial_shape(0) << std::endl;
                //     std::cout << "input[1]: " << compressed_fc->get_input_node_shared_ptr(1)->get_shape() << std::endl;
                //     std::cout << "input[2]: " << compressed_fc->get_input_node_shared_ptr(2)->get_shape() << std::endl;
                //     std::cout << "input[3]: " << compressed_fc->get_input_node_shared_ptr(3)->get_shape() << std::endl;
                //     if (compressed_fc->inputs().size() > 4)
                //         std::cout << "input[4]: " << compressed_fc->get_input_node_shared_ptr(4)->get_shape()
                //                   << std::endl;
                // }

                std::map<int, std::shared_ptr<ov::Node>> org_users;
                for (auto u : first_fc_after_pa->get_users()) {
                    for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                        if (u->get_input_node_shared_ptr(idx) == first_fc_after_pa) {
                            org_users.insert({idx, u});
                        }
                    }
                }

                auto ranked_weight =
                    std::make_shared<ov::intel_gpu::op::RankConstant>(compressed_fc->get_input_node_shared_ptr(1),
                                                                      world_size,
                                                                      rank_size);

                std::shared_ptr<ov::Node> ranked_bias, ranked_scale, ranked_zp;

                if (!std::dynamic_pointer_cast<op::Placeholder>(compressed_fc->get_input_node_shared_ptr(2))) {
                    ranked_bias =
                        std::make_shared<ov::intel_gpu::op::RankConstant>(compressed_fc->get_input_node_shared_ptr(2),
                                                                          world_size,
                                                                          rank_size);
                }

                std::shared_ptr<Node> new_fc = nullptr;
                if (compressed_fc) {
                    auto scale_node = compressed_fc->get_input_node_shared_ptr(3);
                    if (scale_node->get_shape()[1] > 1)
                        ranked_scale =
                            std::make_shared<ov::intel_gpu::op::RankConstant>(scale_node, world_size, rank_size);
                    else
                        ranked_scale = compressed_fc->get_input_node_shared_ptr(3);
                    if (compressed_fc->inputs().size() > 4) {
                        auto zp_node = compressed_fc->get_input_node_shared_ptr(4);
                        if (zp_node->get_shape()[1] > 1)
                            ranked_zp =
                                std::make_shared<ov::intel_gpu::op::RankConstant>(zp_node, world_size, rank_size);
                        else
                            ranked_zp = compressed_fc->get_input_node_shared_ptr(4);
                        new_fc = std::make_shared<op::FullyConnectedCompressed>(first_fc_after_pa->get_input_node_shared_ptr(0),
                                                                                ranked_weight,
                                                                                ranked_bias ? ranked_bias : first_fc_after_pa->get_input_node_shared_ptr(2),
                                                                                ranked_scale,
                                                                                ranked_zp,
                                                                                first_fc_after_pa->get_element_type());
                    } else {
                        new_fc = std::make_shared<op::FullyConnectedCompressed>(first_fc_after_pa->get_input_node_shared_ptr(0),
                                                                                ranked_weight,
                                                                                ranked_bias ? ranked_bias : first_fc_after_pa->get_input_node_shared_ptr(2),
                                                                                ranked_scale,
                                                                                first_fc_after_pa->get_element_type());
                    }
                } else {
                    new_fc = std::make_shared<op::FullyConnected>(first_fc_after_pa->get_input_node_shared_ptr(0),
                                                                  ranked_weight,
                                                                  first_fc_after_pa->get_input_node_shared_ptr(2),
                                                                  first_fc_after_pa->get_element_type());
                }

                std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
                sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(new_fc,
                                                                            w_size,
                                                                            first_fc_after_pa->get_input_node_shared_ptr(1)->get_shape()[-1],
                                                                            first_fc_after_pa->get_element_type(),
                                                                            ov::intel_gpu::op::TP_MODE::ALL_REDUCE);
                sync_node->set_friendly_name(first_fc_after_pa->get_friendly_name() + "_TP");
                copy_runtime_info(first_fc_after_pa, new_fc);
                for (auto& iter : org_users) {
                    iter.second->input(iter.first).replace_source_output(sync_node->output(0));
                }
                first_fc_after_pa->clear_control_dependencies();
            }
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(paged_attention_m, "PagedAttentionSplitInput");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov