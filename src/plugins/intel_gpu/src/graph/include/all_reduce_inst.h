// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/all_reduce.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<all_reduce> : public typed_program_node_base<all_reduce> {
private:
    using parent = typed_program_node_base<all_reduce>;

public:
    using parent::parent;
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog) {
        }
    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    using parent::get_kernel_impl_params;
};

using all_reduce_node = typed_program_node<all_reduce>;

template<>
class typed_primitive_inst<all_reduce> : public typed_primitive_inst_base<all_reduce> {
    using parent = typed_primitive_inst_base<all_reduce>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(all_reduce_node const& /*node*/, const kernel_impl_params& impl_param);

    static layout calc_output_layout(const all_reduce_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const all_reduce_node& node);

    typed_primitive_inst(network& network, const all_reduce_node& desc);
    typed_primitive_inst(network& network) : parent(network) {}
    void update_output_memory() override;

protected:
    void on_execute() override;
};

using all_reduce_inst = typed_primitive_inst<all_reduce>;

} // namespace cldnn