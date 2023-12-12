// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include <openvino/core/graph_util.hpp>

#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/shape_of_base.hpp"

#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/fused_names_cleanup.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace {

std::unordered_set<std::string> get_removed_nodes(const std::shared_ptr<const ov::Model>& original_model,
                                                  const std::shared_ptr<const ov::Model>& transformed_model) {
    std::unordered_set<std::string> result = {};
    std::unordered_set<std::string> transformed_node_names = {};

    for (auto&& node : transformed_model->get_ops()) {
        transformed_node_names.emplace(node->get_friendly_name());
        for (auto&& fused_layer_name : ov::getFusedNamesVector(node)) {
            transformed_node_names.emplace(fused_layer_name);
        }
    }

    for (auto&& original_node : original_model->get_ops()) {
        if (!transformed_node_names.count(original_node->get_friendly_name()))
            result.emplace(original_node->get_friendly_name());
    }

    return result;
}

}  // namespace

ov::IPlugin::IPlugin() : m_executor_manager(ov::threading::executor_manager()), m_is_new_api(true) {}

void ov::IPlugin::set_version(const ov::Version& version) {
    m_version = version;
}

const ov::Version& ov::IPlugin::get_version() const {
    return m_version;
}

void ov::IPlugin::set_device_name(const std::string& name) {
    m_plugin_name = name;
}

const std::string& ov::IPlugin::get_device_name() const {
    return m_plugin_name;
}

void ov::IPlugin::add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::set_core(const std::weak_ptr<ov::ICore>& core) {
    OPENVINO_ASSERT(!core.expired());
    m_core = core;
    auto locked_core = m_core.lock();
    if (locked_core)
        m_is_new_api = locked_core->is_new_api();
}

std::shared_ptr<ov::ICore> ov::IPlugin::get_core() const {
    return m_core.lock();
}

bool ov::IPlugin::is_new_api() const {
    return m_is_new_api;
}

const std::shared_ptr<ov::threading::ExecutorManager>& ov::IPlugin::get_executor_manager() const {
    return m_executor_manager;
}

std::shared_ptr<ov::ICompiledModel> ov::IPlugin::compile_model(const std::string& model_path,
                                                               const ov::AnyMap& properties) const {
    auto core = get_core();
    OPENVINO_ASSERT(core);
    auto model = core->read_model(model_path, std::string());
    return compile_model(model, properties);
}

std::unordered_set<std::string> ov::get_supported_nodes(
    const std::shared_ptr<const ov::Model>& model,
    std::function<void(std::shared_ptr<ov::Model>&)> transform,
    std::function<bool(const std::shared_ptr<ov::Node>)> is_node_supported,
    std::function<bool(const std::shared_ptr<ov::Node>)> is_node_under_memory_control,
    uint64_t memory_size_in_bytes) {
    // Collect original operation names
    std::unordered_set<std::string> original_ops;
    for (auto&& node : model->get_ops()) {
        original_ops.emplace(node->get_friendly_name());
    }

    auto transformed_model = model->clone();
    // for (auto &item_1 : transformed_model->get_ordered_ops()) {
    //     if (item_1->get_friendly_name() == "self.model.model.layers.0.self_attn.q_proj.weight_compressed") {
    //         std::cout << item_1->get_friendly_name() << "3 compile model constandfolding end: " << item_1->get_element_type() << std::endl;
    //     }
    // }
    // Cleanup fused names if there are present in original model
    ov::pass::Manager m;
    m.register_pass<ov::pass::FusedNamesCleanup>();
    m.run_passes(transformed_model);

    transform(transformed_model);
    auto ops = transformed_model->get_ordered_ops();

    std::unordered_set<std::string> new_ops;
    for (auto&& node : transformed_model->get_ops()) {
        new_ops.emplace(node->get_friendly_name());
    }

    using NameSet = std::unordered_set<std::string>;
    using NodePtr = std::shared_ptr<ov::Node>;

    NameSet supported;
    NameSet unsupported;

    // for (auto&& name : original_ops) {
    //     if (!new_ops.count(name)) {
    //         supported.insert(name);
    //     }
    // }

    auto get_names_set = [](const NodePtr& op) -> NameSet {
        auto fused_names = ov::getFusedNamesVector(op);
        NameSet names(fused_names.begin(), fused_names.end());
        names.insert(op->get_friendly_name());
        return names;
    };

    // Collect all operation names even there are no such names in original model
    for (auto&& op : ops) {
        auto names = get_names_set(op);
        if (is_node_supported(op)) {
            supported.insert(names.begin(), names.end());
        } else {
            unsupported.insert(names.begin(), names.end());
        }
    }

    for (auto&& name : supported) {
        if (name == "Constant_207503") {
            std::cout << "!@" << std::endl;
        }
    }

    // If operation was fused into several operations where one is supported
    // but another one is not supported remove it from supported
    for (auto&& name : unsupported) {
        supported.erase(name);
    }

    auto get_output_node = [](const ov::Output<ov::Node>& output) -> NodePtr {
        return output.get_node_shared_ptr();
    };

    auto get_input_node = [&get_output_node](const ov::Input<ov::Node>& input) -> NodePtr {
        return get_output_node(input.get_source_output());
    };

    auto has_all_consumers_unsupported = [&](const NameSet& supported, const NodePtr& node) -> bool {
        bool has_consumers = false;
        for (auto&& output : node->outputs()) {
            for (auto&& input : output.get_target_inputs()) {
                has_consumers = true;
                if (supported.count(input.get_node()->get_friendly_name())) {
                    return false;
                }
            }
        }
        return has_consumers;
    };

    auto has_users_supported = [&](const NameSet& supported, const NodePtr& node) -> bool {
        auto users_ = node->get_users();
        for (auto &itt : users_) {
            // std::cout << itt->get_friendly_name() << std::endl;
            if (supported.count(itt->get_friendly_name())) {
                return true;
            }
        }
        return false;
    };


    auto has_all_consumers_unsupported1 = [&](std::unordered_set<std::string> original_ops, const NodePtr& node) -> bool {
        for (auto&& output : node->outputs()) {
            for (auto&& input : output.get_target_inputs()) {
                if (!original_ops.count(input.get_node()->get_friendly_name())) {
                    return true;
                }
            }
        }
        return false;
    };


    auto has_all_sources_supported =
        [&get_input_node](const NameSet& supported, const NodePtr& op, bool skip_const = true) -> bool {
        for (auto& input : op->inputs()) {
            const auto& node = get_input_node(input);
            if ((skip_const && (ov::op::util::is_constant(node))) || (ov::op::util::is_parameter(node)))
                continue;
            if (!supported.count(node->get_friendly_name()))
                return false;
        }
        return true;
    };

    auto has_unsupported_source =
        [&get_input_node](const NameSet& supported, const NodePtr& op, bool const_only = false) -> bool {
        for (auto& input : op->inputs()) {
            const auto& node = get_input_node(input);
            if (const_only && !ov::op::util::is_constant(node))
                continue;
            if (!supported.count(node->get_friendly_name())) {
                return true;
            }
        }
        return false;
    };

    auto has_unsupported_source1 =
        [&get_input_node](const NameSet& supported, const NameSet& unsupported, const NodePtr& op) -> bool {
        size_t arg_count = op->get_input_size();
        for (size_t i = 0; i < arg_count; ++i) {
            Node* dep = op->get_input_node_ptr(arg_count - i - 1);
            if (!supported.count(dep->get_friendly_name()) && !unsupported.count(dep->get_friendly_name())) {
                return true;
            }
            // std::cout << dep->get_friendly_name() << " hetero type end: " << dep->get_element_type() << std::endl;
        }
        return false;
    };


    auto remove_op_from_supported = [&](const NodePtr& node) {
        auto names = get_names_set(node);
        for (auto& name : get_names_set(node)) {
            supported.erase(name);
        }
    };

    bool breaks_control = false;
    // Walk over transformed model for special handing of Parameters/Constants/Results
    bool memory_control = memory_size_in_bytes > 0;
    auto available_memory_size = memory_size_in_bytes;
    auto is_node_can_breaks_graph = [&](const NodePtr& node) -> bool {
        return !is_node_under_memory_control || is_node_under_memory_control(node);
    };
    unsigned long total_size = 0;
    bool filter = false;
    NameSet removed_nodes = get_removed_nodes(model, transformed_model);
    // std::cout << "-----------------------------------------------------------" << std::endl;
    int i = 0;
    bool start_split = false;
    bool start_split_1 = false;
    unsigned long total_ops_size = 0;
    for (auto&& op : ops) {
        if (ov::op::util::is_constant(op)) {
            const auto const_byte_size = op->get_element_type().size() * shape_size(op->get_shape());
            total_ops_size += const_byte_size;
        }
    }
    std::cout << "total_ops_size/2 = " << total_ops_size/2 << std::endl;
    std::cout << "available_memory_size = " << available_memory_size << std::endl;
    for (auto&& op : ops) {
        i++;
        // std::cout << i << std::endl;
        // if (op->get_friendly_name() == "__module.model.model.layers.14.self_attn/aten::add_/Add") {
        //     std::cout << "__module.model.model.layers.14.self_attn/aten::add_/Add" << std::endl;
        // }
        if (op->get_friendly_name() == "Constant_207503") {
            std::cout << "Constant_207503" << std::endl;
        }
        // if (!has_unsupported_source(supported, op, false)) {
        //     continue;
        // }
        // std::cout << "name: " << op->get_friendly_name() << " " << op->get_element_type() << std::endl;
        auto are_all_consumers_can_break_graph = [&]() {
            // If break control is turned off or callback is not given
            // than it doesn't matter who are consumers of this node
            if (!breaks_control) {
                return true;
            }
            // Otherwise check all consumers
            for (auto&& output : op->outputs()) {
                for (auto&& input : output.get_target_inputs()) {
                    if (!is_node_can_breaks_graph(input.get_node()->shared_from_this())) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (memory_control) {
            if (ov::op::util::is_constant(op)) {
                const auto const_byte_size = op->get_element_type().size() * shape_size(op->get_shape());
                total_size += const_byte_size;                
                if (total_size >= total_ops_size/2) {
                    if (!start_split) {
                        // std::cout << op->get_friendly_name() << " " << " start_split" << std::endl;
                        start_split = true;                
                    }
                }
            }
            // else if (start_split_1) {
            //     // if (ov::is_type<ov::op::v0::Convert>(op)) {
            //     if (ov::is_type<ov::op::v>(op)) {
            //         std::cout << op->get_friendly_name() << " " << " start_split" << std::endl;
            //         start_split = true;
            //     }
            // }
            if (start_split) {
                if (!ov::op::util::is_constant(op)) {
                    if (!has_unsupported_source(supported, op, false)) {
                            continue;
                    }
                    remove_op_from_supported(op);
                    for (auto& input : op->inputs()) {
                        const auto& node = get_input_node(input);
                        // if (ov::op::util::is_constant(node) || dynamic_cast<const ov::op::v0::Convert*>(node.get()) != nullptr || ov::is_type<ov::op::v0::Convert>(node)) {
                        if (ov::op::util::is_constant(node)) {
                            remove_op_from_supported(node);
                        }
                    }
                } else {
                    remove_op_from_supported(op);
                }
            }
        }

        // if (ov::op::util::is_constant(op)) {
        //     // Mark Constants and all fused names as unsupported if they are have no
        //     // supported consumers/sources
        //     if (has_all_consumers_unsupported(supported, op)) {
        //         // remove_op_from_supported(op);
        //         std::cout << "@@" << std::endl;
        //         // If memory control is used, check available device memory
        //     } else if (memory_control) {
        //         const auto const_byte_size = op->get_element_type().size() * shape_size(op->get_shape());
        //         total_size += const_byte_size;
        //         if (total_size >= total_ops_size/2) {
        //         // if (i >= 2001) {
        //             // if (has_all_consumers_unsupported1(original_ops, op)) {
        //             //     continue;
        //             // }
        //             if (!start_split)
        //                 std::cout << op->get_friendly_name() << " " << const_byte_size << " start_split" << std::endl;
        //             start_split = true;
        //             // std::cout << "start split" << std::endl;
        //         // if (total_size > available_memory_size) {
        //             // Memory limit is exceeded
        //             remove_op_from_supported(op);
        //             // Starting from this position we start to check node types
        //             // breaks_control = true;
        //             // filter = true;;
        //         } else {
        //             // available_memory_size -= const_byte_size;
        //             // breaks_control = false;
        //         }
        //     }
        // } else if ((memory_control) || (start_split)) {
        // // } else if ((memory_control) || (total_size >= total_ops_size/2)) {
        // // } else if ((memory_control) || (total_size > available_memory_size)) {
        // // } else if ((memory_control && has_unsupported_source(supported, op, is_node_can_breaks_graph(op)))) {
        //     // Operation has unsupported input constants in memory control mode when
        //     // break control is turned off or any unsupported input if break control is turned on
        //                                     //   __module.model.model.layers.11.mlp.up_proj/aten::linear/MatMul_2610
        //     // if (op->get_friendly_name().find("MatMul") == std::string::npos) {
        //     // if (op->get_friendly_name() == "__module.model.model.layers.1.self_attn/aten::cat/Concat_503") {
        //     //     std::cout << has_unsupported_source(supported, op, is_node_can_breaks_graph(op)) << std::endl;
        //     // }
        //     // if (has_unsupported_source1(supported, unsupported, op)) {
        //     //     std::cout << op->get_friendly_name() << std::endl;
        //     //     std::cout << "22" << std::endl;
        //     // }
        //     if (!has_unsupported_source(supported, op, false)) {
        //         continue;
        //     }
        //     remove_op_from_supported(op);

        // //     std::cout << "!!" << std::endl;
        // // }
        // // std::cout << "removed 1" << std::endl;
        //     for (auto& input : op->inputs()) {
        //         const auto& node = get_input_node(input);
        //         // if (ov::op::util::is_constant(node) || dynamic_cast<const ov::op::v0::Convert*>(node.get()) != nullptr || ov::is_type<ov::op::v0::Convert>(node)) {
        //         if (ov::op::util::is_constant(node)) {
        //             remove_op_from_supported(node);
        //         }
        //     }
        //     // }
        // }
    }

    // Get removed nodes
    // NameSet removed_nodes = get_removed_nodes(model, transformed_model);

    // Filter ShapeOfs
    for (auto& op : model->get_ordered_ops()) {
        const auto& name = op->get_friendly_name();
        if (ov::is_type<ov::op::util::ShapeOfBase>(op) && (supported.count(name) || removed_nodes.count(name))) {
            // Don't allow cut on ShapeOf
            if (has_all_consumers_unsupported(supported, op) && has_all_consumers_unsupported(removed_nodes, op)) {
                remove_op_from_supported(op);
                removed_nodes.erase(name);
            }
        }
    }
    for (auto&& name : supported) {
        if (name == "Constant_207503") {
            std::cout << "!@" << std::endl;
        }
    }
    bool changed = true;
    if (memory_control) {
        for (auto& op : model->get_ordered_ops()) {
            if (op->get_friendly_name() == "Constant_207503") {
                std::cout << "" << std::endl;
            }
            if ((removed_nodes.count(op->get_friendly_name()) && has_all_sources_supported(supported, op)) && !ov::is_type<ov::op::v0::Convert>(op)) {
                supported.insert(op->get_friendly_name());
            }
        }
    } else {
        // If memory control is off
        // mark all removed nodes as supported
        supported.insert(removed_nodes.begin(), removed_nodes.end());
    }
    // for (auto&& name : supported) {
    //     if (name == "Constant_207503") {
    //         std::cout << "!@" << std::endl;
    //     }
    // }
    if (memory_control) {
        while (changed) {
            changed = false;
            for (auto& op : model->get_ordered_ops()) {
                if (!supported.count(op->get_friendly_name()) && has_users_supported(supported, op)) {
                    // if (op->get_friendly_name() == "__module.model.model.layers.29.self_attn.q_proj/aten::linear/MatMul_6106") {
                    //     std::cout << "UUUUU: " << op->get_friendly_name() << std::endl;
                    //     std::cout << !has_all_consumers_unsupported(supported, op) << std::endl;
                    //     std::cout << supported.count(op->get_friendly_name()) << std::endl;
                    // }
                    supported.insert(op->get_friendly_name());
                    changed = true;
                }
            }
        }
    }

    // Finally get intersection of all supported operation names
    // and operation names from original model
    NameSet res;
    for (auto&& name : supported) {
        // if (name == "Constant_207503") {
        //     std::cout << "!@" << std::endl;
        // }
        if (original_ops.count(name)) {
            res.insert(name);
        }
    }

    // Remove parameters (or parameter + convert) which has no supported consumers
    // and results (or result + convert) which has no supported source node
    for (auto& op : model->get_ordered_ops()) {
        // if (op->get_friendly_name() == "Constant_207503") {
        //     std::cout << "!@" << std::endl;
        // }
        if (ov::is_type<ov::op::v0::Convert>(op)) {
            if (ov::op::util::is_parameter(get_input_node(op->input(0))) && has_all_consumers_unsupported(res, op)) {
                res.erase(op->get_friendly_name());
            }
        } else {
            auto outputs = op->outputs();
            auto all_consumers_are_results =
                std::all_of(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& output) -> bool {
                    return ov::op::util::is_output(get_output_node(output));
                });
            if (all_consumers_are_results && has_unsupported_source(res, op, true)) {
                res.erase(op->get_friendly_name());
            }
        }
    }

    for (auto& param : model->get_parameters()) {
        // if (param->get_friendly_name() == "Constant_207503") {
        //     std::cout << "!@" << std::endl;
        // }
        if (has_all_consumers_unsupported(res, param)) {
            res.erase(param->get_friendly_name());
        }
    }

    for (auto& result : model->get_results()) {
        if (has_unsupported_source(res, result)) {
            res.erase(result->get_friendly_name());
        }
    }
    // for (auto item = res.begin(); item != res.end(); item++) {
    //     if (*item == "Constant_207503") {
    //         std::cout << "!" << std::endl;
    //     }
    // }

    return res;
}
