/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"                   // for exporting symbols
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/types.hpp"                  // for TensorMap, TensorIndex
#include "morpheus/utilities/cupy_util.hpp"    // for CupyUtil

#include <pybind11/pytypes.h>  // for object

#include <memory>  // for shared_ptr
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorMemory****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief Container for holding a collection of named `TensorObject`s in a `std::map` keyed by name.
 * Base class for `InferenceMemory` & `ResponseMemory`
 *
 */
class MORPHEUS_EXPORT TensorMemory
{
  public:
    /**
     * @brief Construct a new Tensor Memory object
     *
     * @param count
     */
    TensorMemory(TensorIndex count);

    /**
     * @brief Construct a new Tensor Memory object
     *
     * @param count
     * @param tensors
     */
    TensorMemory(TensorIndex count, TensorMap&& tensors);
    virtual ~TensorMemory() = default;

    TensorIndex count{0};

    /**
     * @brief Verify whether the specified tensor name is present in the tensor memory
     *
     * @param name
     * @return true
     * @return false
     */
    bool has_tensor(const std::string& name) const;

    /**
     * @brief Get the tensor object identified by `name`
     *
     * @param name
     * @return TensorObject&
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    TensorObject& get_tensor(const std::string& name);

    /**
     * @brief Get the tensor object identified by `name`
     *
     * @param name
     * @return const TensorObject&
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    const TensorObject& get_tensor(const std::string& name) const;

    /**
     * @brief Set the tensor object identified by `name`
     *
     * @param name
     * @param tensor
     * @throws std::length_error If the number of rows in `tensor` does not match `count`.
     */
    void set_tensor(const std::string& name, TensorObject&& tensor);

    /**
     * @brief Get a reference to the internal tensors map
     *
     * @return const TensorMap&
     */
    const TensorMap& get_tensors() const;

    /**
     * @brief Set the tensors object
     *
     * @param tensors
     * @throws std::length_error If the number of rows in the `tensors` do not match `count`.
     */
    void set_tensors(TensorMap&& tensors);

    /**
     * @brief Copy tensor ranges
     *
     * @param ranges
     * @param num_selected_rows
     * @return TensorMap
     */
    TensorMap copy_tensor_ranges(const std::vector<RangeType>& ranges, TensorIndex num_selected_rows) const;

  protected:
    /**
     * @brief Checks if the number of rows in `tensor` matches `count`
     *
     * @param tensor
     * @throws std::length_error If the number of rows in `tensor` do not match `count`.
     */
    void check_tensor_length(const TensorObject& tensor);

    /**
     * @brief Checks each tesnor in `tensors` verifying that the number of rows matches count
     *
     * @param tensors
     * @throws std::length_error If the number of rows in the `tensors` do not match `count`.
     */
    void check_tensors_length(const TensorMap& tensors);

    /**
     * @brief Verify that a tensor identified by `name` exists, throws a `runtime_error` othwerwise.
     *
     * @param name
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    void verify_tensor_exists(const std::string& name) const;

  private:
    TensorMap m_tensors;
};

/****** TensorMemoryInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT TensorMemoryInterfaceProxy
{
    /**
     * @brief Create and initialize a TensorMemory object, and return a shared pointer to the result. Each array in
     * `tensors` should be of length `count`.
     *
     * @param count : Lenght of each array in `tensors`
     * @param tensors : Map of string on to cupy arrays
     * @return std::shared_ptr<TensorMemory>
     */
    static std::shared_ptr<TensorMemory> init(TensorIndex count, pybind11::object& tensors);

    /**
     * @brief Get the count object
     *
     * @param self
     * @return TensorIndex
     */
    static TensorIndex get_count(TensorMemory& self);

    /**
     * @brief Returns a list of the current tensor names
     *
     * @param self
     * @return std::vector<std::string>
     */
    static std::vector<std::string> tensor_names_getter(TensorMemory& self);

    /**
     * @brief Returns true if a tensor with the requested name exists in the tensors object
     *
     * @param name Tensor name to lookup
     */
    static bool has_tensor(TensorMemory& self, std::string name);

    /**
     * @brief Get the tensors converted to CuPy arrays. Pybind11 will convert the std::map to a Python dict.
     *
     * @param self
     * @return py_tensor_map_t
     */
    static CupyUtil::py_tensor_map_t get_tensors(TensorMemory& self);

    /**
     * @brief Set the tensors object converting a map of CuPy arrays to Tensors
     *
     * @param self
     * @param tensors
     */
    static void set_tensors(TensorMemory& self, CupyUtil::py_tensor_map_t tensors);

    /**
     * @brief Get the tensor object identified by `name`
     *
     * @param self
     * @param name
     * @return pybind11::object
     * @throws pybind11::key_error When no matching tensor exists.
     */
    static pybind11::object get_tensor(TensorMemory& self, const std::string name);

    /**
     * @brief Same as `get_tensor` but used when the method is being bound to a python property
     *
     * @param self
     * @param name
     * @return pybind11::object
     * @throws pybind11::attribute_error When no matching tensor exists.
     */
    static pybind11::object get_tensor_property(TensorMemory& self, const std::string name);

    /**
     * @brief Set the tensor object identified by `name`
     *
     * @param self
     * @param cupy_tensor
     */
    static void set_tensor(TensorMemory& self, const std::string name, const pybind11::object& cupy_tensor);
};
/** @} */  // end of group
}  // namespace morpheus
