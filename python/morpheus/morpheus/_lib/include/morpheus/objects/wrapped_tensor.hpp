
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

#include "morpheus/export.h"
#include "morpheus/objects/tensor_object.hpp"

#include <pybind11/pytypes.h>

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorObject****************************************/
/****** <NAME>InterfaceProxy *************************/

/**
 * @addtogroup objects
 * @{
 * @file
 */

/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT TensorObjectInterfaceProxy
{
    static pybind11::dict cuda_array_interface(TensorObject& self);
    static pybind11::object to_cupy(TensorObject& self);
    static TensorObject from_cupy(pybind11::object cupy_array);
};
/** @} */  // end of group
}  // namespace morpheus
