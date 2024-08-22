// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class PagedAttentionSplitInput: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PagedAttentionSplitInput", "0");
    PagedAttentionSplitInput(size_t world_size, size_t rank_size);

private:
    std::shared_ptr<ov::Node> bfs_fc(std::shared_ptr<ov::Node> input);
    // std::unordered_set<std::shared_ptr<ov::Node>> visited;
    // std::unordered_set<std::shared_ptr<ov::Node>> has_visited;
};

}   // namespace intel_gpu
}   // namespace ov