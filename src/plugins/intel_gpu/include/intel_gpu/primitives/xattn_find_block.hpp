// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

#include <cstdint>
#include <vector>

namespace cldnn {

/// @brief Primitive that wraps XAttention find_block CM kernel.
struct xattn_find_block : public primitive_base<xattn_find_block> {
    CLDNN_DECLARE_PRIMITIVE(xattn_find_block)

    xattn_find_block()
        : primitive_base("", {}) {}

    xattn_find_block(const primitive_id& id,
                     const std::vector<input_info>& inputs,
                     uint32_t heads_num,
                     uint32_t head_size,
                     uint32_t q_len,
                     uint32_t q_stride,
                     uint32_t k_stride,
                     uint32_t q_stride_pad,
                     uint32_t q_block_pad,
                     uint32_t k_block_pad,
                     int32_t causal_start_index,
                     float thresh,
                     uint32_t stride = 16,
                     uint32_t block_size = 128,
                     bool is_causal = true,
                     bool use_int8 = false)
        : primitive_base(id, inputs, 1, {optional_data_type(data_types::u8)})
        , heads_num(heads_num)
        , head_size(head_size)
        , q_len(q_len)
        , q_stride(q_stride)
        , k_stride(k_stride)
        , q_stride_pad(q_stride_pad)
        , q_block_pad(q_block_pad)
        , k_block_pad(k_block_pad)
        , causal_start_index(causal_start_index)
        , thresh(thresh)
        , stride(stride)
        , block_size(block_size)
        , is_causal(is_causal)
        , use_int8(use_int8) {}

    uint32_t heads_num = 0;
    uint32_t head_size = 0;
    uint32_t q_len = 0;
    uint32_t q_stride = 0;
    uint32_t k_stride = 0;
    uint32_t q_stride_pad = 0;
    uint32_t q_block_pad = 0;
    uint32_t k_block_pad = 0;
    int32_t causal_start_index = 0;
    float thresh = 0.f;
    uint32_t stride = 16;
    uint32_t block_size = 128;
    bool is_causal = true;
    bool use_int8 = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, heads_num);
        seed = hash_combine(seed, head_size);
        seed = hash_combine(seed, q_len);
        seed = hash_combine(seed, q_stride);
        seed = hash_combine(seed, k_stride);
        seed = hash_combine(seed, q_stride_pad);
        seed = hash_combine(seed, q_block_pad);
        seed = hash_combine(seed, k_block_pad);
        seed = hash_combine(seed, causal_start_index);
        seed = hash_combine(seed, thresh);
        seed = hash_combine(seed, stride);
        seed = hash_combine(seed, block_size);
        seed = hash_combine(seed, is_causal);
        seed = hash_combine(seed, use_int8);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        const auto& typed_rhs = static_cast<const xattn_find_block&>(rhs);
        return heads_num == typed_rhs.heads_num && head_size == typed_rhs.head_size && q_len == typed_rhs.q_len &&
               q_stride == typed_rhs.q_stride && k_stride == typed_rhs.k_stride && q_stride_pad == typed_rhs.q_stride_pad &&
               q_block_pad == typed_rhs.q_block_pad && k_block_pad == typed_rhs.k_block_pad &&
               causal_start_index == typed_rhs.causal_start_index && thresh == typed_rhs.thresh &&
               stride == typed_rhs.stride && block_size == typed_rhs.block_size && is_causal == typed_rhs.is_causal &&
               use_int8 == typed_rhs.use_int8;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<xattn_find_block>::save(ob);
        ob << heads_num;
        ob << head_size;
        ob << q_len;
        ob << q_stride;
        ob << k_stride;
        ob << q_stride_pad;
        ob << q_block_pad;
        ob << k_block_pad;
        ob << causal_start_index;
        ob << thresh;
        ob << stride;
        ob << block_size;
        ob << is_causal;
        ob << use_int8;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<xattn_find_block>::load(ib);
        ib >> heads_num;
        ib >> head_size;
        ib >> q_len;
        ib >> q_stride;
        ib >> k_stride;
        ib >> q_stride_pad;
        ib >> q_block_pad;
        ib >> k_block_pad;
        ib >> causal_start_index;
        ib >> thresh;
        ib >> stride;
        ib >> block_size;
        ib >> is_causal;
        ib >> use_int8;
    }
};

}  // namespace cldnn
