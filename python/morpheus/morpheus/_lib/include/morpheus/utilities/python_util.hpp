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

#include "morpheus/export.h"

#include <sys/types.h>

#include <string>

struct _object;

using PyObject = _object;  // NOLINT(readability-identifier-naming)

namespace morpheus::utilities {

/**
 * @brief Shows a python warning using the `warnings.warn` module. These warnings can be suppressed and work different
 * than `logger.warn()`
 *
 * @param deprecation_message The message to show
 * @param category A Python warning message type such as `PyExc_DeprecationWarning`
 * @param stack_level If the warning should appear earlier up in the stack, set this to >1
 */
MORPHEUS_EXPORT void show_warning_message(const std::string& deprecation_message,
                                          PyObject* category  = nullptr,
                                          ssize_t stack_level = 1);

}  // namespace morpheus::utilities
