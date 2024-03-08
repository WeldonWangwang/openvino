// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiled_model.hpp"
#include "itt.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "properties.hpp"

ov::hetero::Plugin::Plugin() {
    set_device_name("HETERO");
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                      const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::compile_model");

    auto config = Configuration{properties, m_cfg};
    auto compiled_model = std::make_shared<CompiledModel>(model->clone(), shared_from_this(), config);
    return compiled_model;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
                                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                                     const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
                                                                     const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::import_model");

    // check ov::loaded_from_cache property and erase it due to not needed any more.
    auto _properties = properties;
    const auto& it = _properties.find(ov::loaded_from_cache.name());
    bool loaded_from_cache = false;
    if (it != _properties.end()) {
        loaded_from_cache = it->second.as<bool>();
        _properties.erase(it);
    }

    auto config = Configuration{_properties, m_cfg};
    auto compiled_model = std::make_shared<CompiledModel>(model, shared_from_this(), config, loaded_from_cache);
    return compiled_model;
}

ov::hetero::Plugin::DeviceProperties ov::hetero::Plugin::get_properties_per_device(const std::string& device_priorities,
                                                                                   const ov::AnyMap& properties) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    DeviceProperties device_properties;
    for (const auto& device_name : device_names) {
        auto properties_it = device_properties.find(device_name);
        if (device_properties.end() == properties_it)
            device_properties[device_name] = get_core()->get_supported_property(device_name, properties, false);
    }
    return device_properties;
}

std::pair<ov::SupportedOpsMap, ov::hetero::SubgraphsMappingInfo> ov::hetero::Plugin::query_model_update(
    std::shared_ptr<ov::Model>& model,
    const ov::AnyMap& properties,
    bool allow_exception) const {
    Configuration full_config{properties, m_cfg};
    DeviceProperties properties_per_device =
        get_properties_per_device(full_config.device_priorities, full_config.get_device_properties());

    //  WARNING: Here is devices with user set priority
    auto device_names = ov::DeviceIDParser::get_hetero_devices(full_config.device_priorities);
    std::map<std::string, size_t> device_mem_map;
    for (const auto& device_name : device_names) {
        if (device_name.find("GPU") != std::string::npos) {
            try {
                device_mem_map[device_name] = get_core()->get_property(device_name, ov::intel_gpu::device_total_mem_size);
            } catch (const ov::Exception&) {
            }
        }
    }

    auto update_supported_ops = [](ov::SupportedOpsMap& final_results, const ov::SupportedOpsMap& device_results) {
        for (const auto& layer_query_result : device_results)
            final_results.emplace(layer_query_result);
    };

    auto has_subgraph_ops = [](std::shared_ptr<ov::Model>& model) {
        for (auto& op : model->get_ordered_ops()) {
            if (ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
                return true;
            }
        }
        return false;
    };

    auto update_config = [&](ov::AnyMap& device_config,
                            const std::shared_ptr<const ov::Model>& model,
                            std::string device_name,
                            // std::map<std::string, size_t>& device_mem_map,
                            bool fallback_device) {
        bool config_update_flag = false;
        auto supported_properties = get_core()->get_property(device_name, ov::supported_properties);
        if (ov::util::contains(supported_properties, ov::query_model_uses_device_mem)) {
            if (fallback_device && !config_update_flag) {
                device_config[ov::query_model_uses_device_mem.name()] = 1.0f;
                config_update_flag = true;
            } else {
                double total_ops_size = 0;
                for (auto&& op : model->get_ordered_ops()) {
                    if (ov::op::util::is_constant(op)) {
                        total_ops_size += op->get_element_type().size() * shape_size(op->get_shape());
                    }
                }
                // Check if there is a device that can take the entire model
                size_t all_device_mem = 0;
                for (auto& devic_mem : device_mem_map) {
                    if (devic_mem.second >= 1.2 * total_ops_size && !config_update_flag) {
                        std::cout << device_name << " " << devic_mem.second << " " << 1.2 * total_ops_size << std::endl;
                        if (device_name == devic_mem.first) {
                            device_config[ov::query_model_uses_device_mem.name()] = 1.0f;
                        } else {
                            device_config[ov::query_model_uses_device_mem.name()] = 0.0f;
                        }
                        config_update_flag = true;
                    }
                    all_device_mem += devic_mem.second;
                }
                if (!config_update_flag) {
                    float model_ratio = device_mem_map[device_name] * 1.0 / all_device_mem;
                    device_config[ov::query_model_uses_device_mem.name()] = model_ratio;
                }
                auto device_item = device_mem_map.find(device_name);
                device_mem_map.erase(device_item);
            }
            std::cout << device_name << " " << device_config[ov::query_model_uses_device_mem.name()].as<float>() << std::endl;
        } else {
            std::cout << device_name << " do not changed query_model_uses_device_mem\n";
        }
    };

    ov::SupportedOpsMap supported_ops_temp;
    ov::SupportedOpsMap supported_ops_temp_1;
    ov::SupportedOpsMap supported_ops_final;
    std::map<std::string, ov::SupportedOpsMap> query_results;
    ov::hetero::SubgraphsMappingInfo mapping_info;
    ResultVector new_outputs;
    for (auto& param : model->get_parameters()) {
        if (param->get_users().size() == 0) {
            auto result = std::make_shared<ov::op::v0::Result>(param);
            ov::copy_runtime_info(param->shared_from_this(), result);
            new_outputs.push_back(result);
        }
    }
    model->add_results(new_outputs);
    for (const auto& device_name : device_names) {
        // If there are some unsupported operations and it is a last device
        // exception should be raised when allowed
        bool fallback_device = (device_name == device_names.back());
        const auto& default_device = (!allow_exception || !fallback_device) ? get_device_name() : "";
        auto& device_config = properties_per_device.at(device_name);
        if (!has_subgraph_ops(model)) {
            update_config(device_config, model, device_name, fallback_device);
            query_results[device_name] = get_core()->query_model(model, device_name, device_config);
            update_supported_ops(supported_ops_temp, query_results[device_name]);
            update_supported_ops(supported_ops_final, query_results[device_name]);

            mapping_info = ov::hetero::mask_model_subgraphs_by_ops(model,
                                                                   supported_ops_temp,
                                                                   m_cfg.dump_dot_files(),
                                                                   default_device);
        } else {
            auto temp_model = model->clone();
            update_supported_ops(supported_ops_temp_1, supported_ops_temp);
            for (auto&& node : temp_model->get_ops()) {
                supported_ops_temp_1.emplace(node->get_friendly_name(), "HETERO-TEMP");
            }
            auto mapping_info_temp =
                ov::hetero::mask_model_subgraphs_by_ops(temp_model, supported_ops_temp_1, false, default_device);
            for (const auto& op : temp_model->get_ordered_ops()) {
                if (const auto& subgraph = ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
                    if (subgraph->get_affinity() == "HETERO-TEMP") {
                        update_config(device_config, subgraph->get_function(), device_name, fallback_device);
                        query_results[device_name] =
                            get_core()->query_model(subgraph->get_function(), device_name, device_config);
                        update_supported_ops(supported_ops_temp, query_results[device_name]);
                        update_supported_ops(supported_ops_final, query_results[device_name]);
                    }
                }
            }
            mapping_info = ov::hetero::mask_model_subgraphs_by_ops(model,
                                                                   supported_ops_temp,
                                                                   m_cfg.dump_dot_files(),
                                                                   default_device);
        }
    }
    return {supported_ops_final, mapping_info};
}

ov::SupportedOpsMap ov::hetero::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::query_model");

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    std::shared_ptr<ov::Model> query_model = model->clone();

    return query_model_update(query_model, properties).first;
}

void ov::hetero::Plugin::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg, true};
}

ov::Any ov::hetero::Plugin::get_property(const std::string& name, const ov::AnyMap& properties) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::capabilities};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities};
        return rw_properties;
    };

    Configuration full_config{properties, m_cfg};
    if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}};
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{get_device_name()};
    } else if (ov::internal::caching_properties == name) {
        return decltype(ov::internal::caching_properties)::value_type{ov::hetero::caching_device_properties.name()};
    } else if (ov::hetero::caching_device_properties == name) {
        return caching_device_properties(full_config.device_priorities);
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else {
        return full_config.get(name);
    }
}

ov::Any ov::hetero::Plugin::caching_device_properties(const std::string& device_priorities) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    // Vector of caching properties per device
    std::vector<ov::AnyMap> result = {};
    for (const auto& device_name : device_names) {
        ov::AnyMap properties = {};
        auto supported_properties = get_core()->get_property(device_name, ov::supported_properties);
        auto supported_internal_properties = get_core()->get_property(device_name, ov::internal::supported_properties);
        if (ov::util::contains(supported_internal_properties, ov::internal::caching_properties)) {
            auto caching_properties = get_core()->get_property(device_name, ov::internal::caching_properties);
            for (const auto& property_name : caching_properties) {
                properties[property_name] = get_core()->get_property(device_name, std::string(property_name), {});
            }
        } else if (ov::util::contains(supported_properties, ov::device::architecture)) {
            // If caching properties are not supported by device, try to add at least device architecture
            auto device_architecture = get_core()->get_property(device_name, ov::device::architecture);
            properties = ov::AnyMap{{ov::device::architecture.name(), device_architecture}};
        } else {
            // Device architecture is not supported, add device name w/o id as achitecture
            ov::DeviceIDParser parser(device_name);
            properties = ov::AnyMap{{ov::device::architecture.name(), parser.get_device_name()}};
        }
        result.emplace_back(properties);
    }
    return ov::Any(result);
}

ov::SoPtr<ov::IRemoteContext> ov::hetero::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> ov::hetero::Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}
