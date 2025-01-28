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

#include "morpheus/objects/tensor_object.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <map>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** CupyUtil****************************************/

/**
 * @addtogroup utilities
 * @{
 * @file
 */

/**
 * @brief Structure that encapsulates cupy utilities.
 */
struct CupyUtil
{
    using tensor_map_t    = std::map<std::string, TensorObject>;
    using py_tensor_map_t = std::map<std::string, pybind11::object>;

    /**
     * @brief Import and return the cupy module. Requires GIL to have already been aqcuired.
     *
     * @return pybind11::module_
     */
    static pybind11::module_ get_cp();

    /**
     * @brief Tests whether or not an object is an instance of `cupy.ndarray`
     *
     * @param test_obj Python object to test
     * @return true The object is a cupy array
     * @return false The object is not a cupy array
     */
    static bool is_cupy_array(pybind11::object test_obj);

    /**
     * @brief Convert a TensorObject to a CuPy array. Requires GIL to have already been aqcuired.
     *
     * @param tensor
     * @return pybind11::object
     */
    static pybind11::object tensor_to_cupy(const TensorObject& tensor);

    /**
     * @brief Convert a CuPy array into a TensorObject. Requires GIL to have already been aqcuired.
     *
     * @param cupy_array
     * @return TensorObject
     */
    static TensorObject cupy_to_tensor(pybind11::object cupy_array);

    /**
     * @brief Convert a map of CuPy arrays into a map of TensorObjects. Requires GIL to have already been aqcuired.
     *
     * @param cupy_tensors
     * @return tensor_map_t
     */
    static tensor_map_t cupy_to_tensors(const py_tensor_map_t& cupy_tensors);

    /**
     * @brief Convert a map of TensorObjects into a map of CuPy arrays. Requires GIL to have already been aqcuired.
     *
     * @param tensors
     * @return py_tensor_map_t
     */
    static py_tensor_map_t tensors_to_cupy(const tensor_map_t& tensors);
};
/** @} */  // end of group
}  // namespace morpheus
