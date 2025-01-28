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

#include "morpheus/utilities/json_types.hpp"

#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/utils.hpp>

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace PYBIND11_NAMESPACE {
namespace detail {

template <>
struct type_caster<nlohmann::json>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type nlohmann::json
     */
    PYBIND11_TYPE_CASTER(nlohmann::json, _("dict[str, typing.Any] | list | str | bool | int | float | None"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into an nlohmann::json
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = nlohmann::json(nullptr);
        }
        else
        {
            value = mrc::pymrc::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an nlohmann::json instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(nlohmann::json src, return_value_policy policy, handle parent)
    {
        return mrc::pymrc::cast_from_json(src).release();
    }
};

template <>
struct type_caster<nlohmann::json_dict>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type nlohmann::json_dict
     */
    PYBIND11_TYPE_CASTER(nlohmann::json_dict, _("dict[str, typing.Any] | None"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into an nlohmann::json_dict
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = nlohmann::json_dict(nullptr);
        }
        else
        {
            if (!PyDict_Check(src.ptr()))
            {
                return false;
            }

            value = static_cast<const nlohmann::json_dict>(
                mrc::pymrc::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src)));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an nlohmann::json_dict instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(nlohmann::json_dict src, return_value_policy policy, handle parent)
    {
        return mrc::pymrc::cast_from_json(src).release();
    }
};

template <>
struct type_caster<nlohmann::json_list>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type nlohmann::json_list
     */
    PYBIND11_TYPE_CASTER(nlohmann::json_list, _("list"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into an nlohmann::json_list
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = nlohmann::json_list(nullptr);
        }
        else
        {
            if (!PyList_Check(src.ptr()))
            {
                return false;
            }

            value = static_cast<const nlohmann::json_list>(
                mrc::pymrc::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src)));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an nlohmann::json_list instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(nlohmann::json_list src, return_value_policy policy, handle parent)
    {
        return mrc::pymrc::cast_from_json(src).release();
    }
};

template <>
struct type_caster<morpheus::utilities::json_t>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type morpheus::utilities::json_t
     */
    PYBIND11_TYPE_CASTER(morpheus::utilities::json_t, _("object | None"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into an morpheus::utilities::json_t
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = morpheus::utilities::json_t(nullptr);
        }
        else
        {
            value = morpheus::utilities::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an morpheus::utilities::json_t instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(morpheus::utilities::json_t src, return_value_policy policy, handle parent)
    {
        return morpheus::utilities::cast_from_json(src).release();
    }
};

template <>
struct type_caster<morpheus::utilities::json_dict_t>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type morpheus::utilities::json_t_dict
     */
    PYBIND11_TYPE_CASTER(morpheus::utilities::json_dict_t, _("dict[str, typing.Any] | None"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into an morpheus::utilities::json_t_dict
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = morpheus::utilities::json_dict_t(nullptr);
        }
        else
        {
            if (!PyDict_Check(src.ptr()))
            {
                return false;
            }

            value = static_cast<const morpheus::utilities::json_dict_t>(
                morpheus::utilities::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src)));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an morpheus::utilities::json_t_dict instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(morpheus::utilities::json_dict_t src, return_value_policy policy, handle parent)
    {
        return morpheus::utilities::cast_from_json(src).release();
    }
};

template <>
struct type_caster<morpheus::utilities::json_list_t>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type morpheus::utilities::json_t_list
     */
    PYBIND11_TYPE_CASTER(morpheus::utilities::json_list_t, _("list | None"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into an morpheus::utilities::json_t_list
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = morpheus::utilities::json_list_t(nullptr);
        }
        else
        {
            if (!PyList_Check(src.ptr()))
            {
                return false;
            }

            value = static_cast<const morpheus::utilities::json_list_t>(
                morpheus::utilities::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src)));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an morpheus::utilities::json_t_list instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(morpheus::utilities::json_list_t src, return_value_policy policy, handle parent)
    {
        return morpheus::utilities::cast_from_json(src).release();
    }
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
