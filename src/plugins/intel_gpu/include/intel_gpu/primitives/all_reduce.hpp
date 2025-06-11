// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/op/util.hpp"
#include "primitive.hpp"
namespace cldnn {

/// @brief
/// @details
struct all_reduce : public primitive_base<all_reduce> {
    CLDNN_DECLARE_PRIMITIVE(all_reduce)

    all_reduce() : primitive_base("", {}) {}

    /// @brief Constructs all_reduce primitive.
    /// @param id This primitive id.
    /// @param inputs of all_reduce.
    all_reduce(const primitive_id& id, const input_info& input, const size_t& world_rank, const size_t& world_size)
        : primitive_base(id, {input}) {
            m_rank = world_rank;
            m_size = world_size;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    /* We don't have any argument to serialize at this moment. */
    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<all_reduce>::save(ob);
        ob << m_rank;
        ob << m_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<all_reduce>::load(ib);
        ib >> m_rank;
        ib >> m_size;
    }
    layout output_layout;
    size_t m_rank;
    size_t m_size;
};
}  // namespace cldnn