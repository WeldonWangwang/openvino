// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include <openvino/core/graph_util.hpp>

#include "openvino/op/convert.hpp"
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

    // Cleanup fused names if there are present in original model
    ov::pass::Manager m;
    m.register_pass<ov::pass::FusedNamesCleanup>();
    m.run_passes(transformed_model);

    transform(transformed_model);
    auto ops = transformed_model->get_ordered_ops();

    using NameSet = std::unordered_set<std::string>;
    using NodePtr = std::shared_ptr<ov::Node>;

    NameSet supported;
    NameSet unsupported;

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

    auto has_all_sources_supported =
        [&get_input_node](const NameSet& supported, const NodePtr& op, bool skip_const = true) -> bool {
        for (auto& input : op->inputs()) {
            const auto& node = get_input_node(input);
            if ((skip_const && ov::op::util::is_constant(node)) || (ov::op::util::is_parameter(node)))
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
    for (auto&& op : ops) {
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
        if (ov::op::util::is_constant(op)) {
            // Mark Constants and all fused names as unsupported if they are have no
            // supported consumers/sources
            if (has_all_consumers_unsupported(supported, op)) {
                remove_op_from_supported(op);
                // If memory control is used, check available device memory
            } else if (memory_control) {
                const auto const_byte_size = op->get_element_type().size() * shape_size(op->get_shape());
                if (const_byte_size > available_memory_size || !are_all_consumers_can_break_graph()) {
                    // Memory limit is exceeded
                    remove_op_from_supported(op);
                    // Starting from this position we start to check node types
                    breaks_control = true;
                } else {
                    available_memory_size -= const_byte_size;
                    breaks_control = false;
                }
            }
        } else if (memory_control && has_unsupported_source(supported, op, is_node_can_breaks_graph(op))) {
            // Operation has unsupported input constants in memory control mode when
            // break control is turned off or any unsupported input if break control is turned on
            remove_op_from_supported(op);
            for (auto& input : op->inputs()) {
                const auto& node = get_input_node(input);
                if (ov::op::util::is_constant(node)) {
                    remove_op_from_supported(node);
                }
            }
        }
    }

    // Get removed nodes
    NameSet removed_nodes = get_removed_nodes(model, transformed_model);

    // Filter ShapeOfs
    for (auto& op : model->get_ordered_ops()) {
        const auto& name = op->get_friendly_name();
        if (ov::is_type<ov::op::util::ShapeOfBase>(op) && (supported.count(name) || removed_nodes.count(name))) {
            // Don't allow cut on ShapeOf
            if (has_all_consumers_unsupported(supported, op) && has_all_consumers_unsupported(removed_nodes, op) && memory_size_in_bytes) {
                remove_op_from_supported(op);
                removed_nodes.erase(name);
            }
        }
    }

    if (memory_control) {
        for (auto& op : model->get_ordered_ops()) {
            if (removed_nodes.count(op->get_friendly_name()) && has_all_sources_supported(supported, op)) {
                supported.insert(op->get_friendly_name());
            }
        }
    } else {
        // If memory control is off
        // mark all removed nodes as supported
        supported.insert(removed_nodes.begin(), removed_nodes.end());
    }

    // Finally get intersection of all supported operation names
    // and operation names from original model
    NameSet res;
    for (auto&& name : supported) {
        if (original_ops.count(name)) {
            res.insert(name);
        }
    }

    // Remove parameters (or parameter + convert) which has no supported consumers
    // and results (or result + convert) which has no supported source node
    for (auto& op : model->get_ordered_ops()) {
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
        if (has_all_consumers_unsupported(res, param)) {
            res.erase(param->get_friendly_name());
        }
    }

    for (auto& result : model->get_results()) {
        if (has_unsupported_source(res, result)) {
            res.erase(result->get_friendly_name());
        }
    }

    return res;
}
