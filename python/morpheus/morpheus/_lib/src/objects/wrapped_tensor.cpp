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

#include "morpheus/objects/wrapped_tensor.hpp"

#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/utilities/cupy_util.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstdint>  // for uintptr_t
#include <utility>
#include <vector>  // get_shape & get_stride return vectors

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorObjectInterfaceProxy *************************/
pybind11::dict TensorObjectInterfaceProxy::cuda_array_interface(TensorObject& self)
{
    // See https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    pybind11::dict array_interface;

    pybind11::list shape_list;

    for (auto& idx : self.get_shape())
    {
        shape_list.append(idx);
    }

    pybind11::list stride_list;

    for (auto& idx : self.get_stride())
    {
        stride_list.append(idx * self.dtype_size());
    }

    pybind11::int_ stream_val = self.stream();

    array_interface["shape"]   = pybind11::cast<pybind11::tuple>(shape_list);
    array_interface["typestr"] = self.get_numpy_typestr();
    array_interface["stream"]  = stream_val;
    array_interface["version"] = 3;

    if (self.is_compact() || self.get_stride().empty())
    {
        array_interface["strides"] = pybind11::none();
    }
    else
    {
        array_interface["strides"] = pybind11::cast<pybind11::tuple>(stride_list);
    }
    array_interface["data"] = pybind11::make_tuple((uintptr_t)self.data(), false);

    return array_interface;
}

pybind11::object TensorObjectInterfaceProxy::to_cupy(TensorObject& self)
{
    return CupyUtil::tensor_to_cupy(self);
}

TensorObject TensorObjectInterfaceProxy::from_cupy(pybind11::object cupy_array)
{
    return CupyUtil::cupy_to_tensor(std::move(cupy_array));
}

}  // namespace morpheus
