/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "morpheus_llm/llm/input_map.hpp"
#include "morpheus_llm/llm/llm_node_runner.hpp"

#include "morpheus/utilities/json_types.hpp"

#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/utils.hpp>

#include <memory>

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace PYBIND11_NAMESPACE {
namespace detail {

template <>
struct type_caster<morpheus::llm::UserInputMapping>
{
  public:
    using input_map_caster_t   = make_caster<morpheus::llm::InputMap>;
    using node_runner_caster_t = make_caster<std::shared_ptr<morpheus::llm::LLMNodeRunner>>;
    using string_caster_t      = make_caster<std::string>;
    using tuple_caster_t       = make_caster<std::tuple<std::string, std::string>>;
    /**
     * This macro establishes the name 'inty' in
     * function signatures and declares a local variable
     * 'value' of type inty
     */
    PYBIND11_TYPE_CASTER(morpheus::llm::UserInputMapping,
                         _("Union[") + input_map_caster_t::name + _(", ") + string_caster_t::name + _(", ") +
                             tuple_caster_t::name + _(", ") + node_runner_caster_t::name + _("]"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a inty
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src || src.is_none())
        {
            return false;
        }

        input_map_caster_t input_map_caster;

        if (input_map_caster.load(src, convert))
        {
            auto input_map      = cast_op<morpheus::llm::InputMap&&>(std::move(input_map_caster));
            value.external_name = input_map.external_name;
            value.internal_name = input_map.internal_name;

            return true;
        }

        node_runner_caster_t node_runner_caster;

        if (node_runner_caster.load(src, convert))
        {
            auto& node_runner = cast_op<std::shared_ptr<morpheus::llm::LLMNodeRunner>&>(node_runner_caster);
            value             = morpheus::llm::UserInputMapping{node_runner};
            return true;
        }

        tuple_caster_t tuple_caster;

        if (tuple_caster.load(src, convert))
        {
            value = morpheus::llm::UserInputMapping{
                cast_op<std::tuple<std::string, std::string>&&>(std::move(tuple_caster))};
            return true;
        }

        string_caster_t string_caster;

        if (string_caster.load(src, convert))
        {
            value = morpheus::llm::UserInputMapping{cast_op<std::string&&>(std::move(string_caster))};
            return true;
        }

        return false;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(morpheus::llm::UserInputMapping src, return_value_policy policy, handle parent)
    {
        pybind11::pybind11_fail("Cannot convert UserInputMapping to Python object");
    }
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
