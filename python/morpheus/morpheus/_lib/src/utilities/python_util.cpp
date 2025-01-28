/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/utilities/python_util.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pyerrors.h>
#include <warnings.h>

namespace morpheus::utilities {

void show_warning_message(const std::string& deprecation_message, PyObject* category, ssize_t stack_level)
{
    pybind11::gil_scoped_acquire gil;

    // Default to standard warnings
    if (category == nullptr)
    {
        category = PyExc_Warning;
    }

    PyErr_WarnEx(category, deprecation_message.c_str(), stack_level);
}
}  // namespace morpheus::utilities
