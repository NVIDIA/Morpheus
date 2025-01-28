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

#include "morpheus/utilities/cupy_util.hpp"

#include "morpheus/objects/dtype.hpp"   // for DType
#include "morpheus/objects/tensor.hpp"  // for Tensor
#include "morpheus/types.hpp"           // for TensorIndex
#include "morpheus/utilities/tensor_util.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>         // for COMPACT_GOOGLE_LOG_FATAL, DCHECK, LogMessageFatal
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/gil.h>         // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>            // IWYU pragma: keep
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <cstdint>  // for uintptr_t
#include <memory>   // for make_shared
#include <optional>
#include <ostream>
#include <string>   // for string
#include <utility>  // for move
#include <vector>   // for vector

namespace morpheus {

namespace py = pybind11;

pybind11::module_ CupyUtil::get_cp()
{
    DCHECK(PyGILState_Check() != 0);
    return pybind11::cast<pybind11::module_>(pybind11::module_::import("cupy"));
}

bool CupyUtil::is_cupy_array(pybind11::object test_obj)
{
    return py::isinstance(test_obj, CupyUtil::get_cp().attr("ndarray"));
}

pybind11::object CupyUtil::tensor_to_cupy(const TensorObject& tensor)
{
    // These steps follow the cupy._convert_object_with_cuda_array_interface function shown here:
    // https://github.com/cupy/cupy/blob/a5b24f91d4d77fa03e6a4dd2ac954ff9a04e21f4/cupy/core/core.pyx#L2478-L2514
    auto cp      = CupyUtil::get_cp();
    auto cuda    = cp.attr("cuda");
    auto ndarray = cp.attr("ndarray");

    auto py_tensor = pybind11::cast(tensor);

    auto ptr    = (uintptr_t)tensor.data();
    auto nbytes = tensor.bytes();
    auto owner  = py_tensor;
    int dev_id  = -1;

    pybind11::list shape_list;
    pybind11::list stride_list;

    for (auto& idx : tensor.get_shape())
    {
        shape_list.append(idx);
    }

    for (auto& idx : tensor.get_stride())
    {
        stride_list.append(idx * tensor.dtype_size());
    }

    pybind11::object mem    = cuda.attr("UnownedMemory")(ptr, nbytes, owner, dev_id);
    pybind11::object dtype  = cp.attr("dtype")(tensor.get_numpy_typestr());
    pybind11::object memptr = cuda.attr("MemoryPointer")(mem, 0);

    // TODO(MDD): Sync on stream
    return ndarray(
        pybind11::cast<pybind11::tuple>(shape_list), dtype, memptr, pybind11::cast<pybind11::tuple>(stride_list));
}

TensorObject CupyUtil::cupy_to_tensor(pybind11::object cupy_array)
{
    // Convert inputs from cupy to Tensor
    pybind11::dict arr_interface = cupy_array.attr("__cuda_array_interface__");

    pybind11::tuple shape_tup = arr_interface["shape"];

    auto shape = shape_tup.cast<ShapeType>();

    auto typestr = arr_interface["typestr"].cast<std::string>();

    pybind11::tuple data_tup = arr_interface["data"];

    auto data_ptr = data_tup[0].cast<uintptr_t>();

    ShapeType strides{};

    if (arr_interface.contains("strides") && !arr_interface["strides"].is_none())
    {
        pybind11::tuple strides_tup = arr_interface["strides"];

        strides = strides_tup.cast<ShapeType>();
    }

    auto dtype = DType::from_numpy(typestr);

    //  Get the size from the shape and dtype
    auto size = TensorUtils::get_elem_count(shape) * dtype.item_size();

    // Finally, handle the stream
    auto stream_value = arr_interface["stream"].cast<std::optional<intptr_t>>();

    // Always create with stream per thread. Only need to check the stream for synchronization purposes
    // See https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html#synchronization
    if (stream_value.has_value())
    {
        DCHECK_NE(*stream_value, 0) << "Invalid for stream to be 0";

        auto stream_view = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(*stream_value));

        // Make sure to sync on this
        stream_view.synchronize();
    }

    auto tensor =
        Tensor::create(std::make_shared<rmm::device_buffer>((void const*)data_ptr, size, rmm::cuda_stream_per_thread),
                       DType::from_numpy(typestr),
                       shape,
                       strides,
                       0);

    return tensor;
}

TensorMap CupyUtil::cupy_to_tensors(const py_tensor_map_t& cupy_tensors)
{
    tensor_map_t tensors;
    for (const auto& tensor : cupy_tensors)
    {
        tensors[tensor.first].swap(std::move(cupy_to_tensor(tensor.second)));
    }

    return tensors;
}

CupyUtil::py_tensor_map_t CupyUtil::tensors_to_cupy(const tensor_map_t& tensors)
{
    py_tensor_map_t cupy_tensors;
    for (const auto& tensor : tensors)
    {
        cupy_tensors[tensor.first] = std::move(tensor_to_cupy(tensor.second));
    }

    return cupy_tensors;
}

}  // namespace morpheus
