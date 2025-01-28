/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "morpheus/stages/deserialize.hpp"

#include "morpheus/messages/control.hpp"       // for ControlMessage
#include "morpheus/messages/meta.hpp"          // for MessageMeta, SlicedMessageMeta
#include "morpheus/objects/table_info.hpp"     // for TableInfo
#include "morpheus/types.hpp"                  // for TensorIndex
#include "morpheus/utilities/cudf_util.hpp"    // for CudfHelper
#include "morpheus/utilities/json_types.hpp"   // for PythonByteContainer
#include "morpheus/utilities/python_util.hpp"  // for show_warning_message
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR

#include <glog/logging.h>       // for COMPACT_GOOGLE_LOG_WARNING, LOG, LogMessage
#include <pybind11/pybind11.h>  // for cast
#include <pyerrors.h>           // for PyExc_RuntimeWarning
#include <pymrc/utils.hpp>      // for cast_from_pyobject

#include <algorithm>  // for min
#include <exception>  // for exception_ptr
#include <optional>   // for optional
#include <sstream>    // for operator<<, basic_ostringstream
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

namespace morpheus {

DeserializeStage::subscribe_fn_t DeserializeStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t incoming_message) {
                if (!incoming_message->has_sliceable_index())
                {
                    if (m_ensure_sliceable_index)
                    {
                        auto old_index_name = incoming_message->ensure_sliceable_index();

                        if (old_index_name.has_value())
                        {
                            // Generate a warning
                            LOG(WARNING) << MORPHEUS_CONCAT_STR(
                                "Incoming MessageMeta does not have a unique and monotonic "
                                "index. Updating index "
                                "to be unique. Existing index will be retained in column '"
                                << *old_index_name << "'");
                        }
                    }
                    else
                    {
                        utilities::show_warning_message(
                            "Detected a non-sliceable index on an incoming MessageMeta. "
                            "Performance when taking slices "
                            "of messages may be degraded. Consider setting "
                            "`ensure_sliceable_index==True`",
                            PyExc_RuntimeWarning);
                    }
                }
                // Loop over the MessageMeta and create sub-batches
                for (TensorIndex i = 0; i < incoming_message->count(); i += this->m_batch_size)
                {
                    std::shared_ptr<ControlMessage> windowed_message = std::make_shared<ControlMessage>();

                    auto sliced_meta = SlicedMessageMeta(
                        incoming_message, i, std::min(i + this->m_batch_size, incoming_message->count()));
                    auto sliced_info = sliced_meta.get_info();

                    // This unforuntately requires grabbing the GIL and is a work-around for issue #2018
                    auto new_meta = MessageMeta::create_from_python(CudfHelper::table_from_table_info(sliced_info));
                    windowed_message->payload(new_meta);

                    auto task = m_task.get();
                    if (task)
                    {
                        windowed_message->add_task(task->first, task->second);
                    }

                    output.on_next(std::move(windowed_message));
                }
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

std::shared_ptr<mrc::segment::Object<DeserializeStage>> DeserializeStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    TensorIndex batch_size,
    bool ensure_sliceable_index,
    const pybind11::object& task_type,
    const pybind11::object& task_payload)
{
    std::unique_ptr<control_message_task_t> task{nullptr};

    if (!task_type.is_none() && !task_payload.is_none())
    {
        task = std::make_unique<control_message_task_t>(pybind11::cast<std::string>(task_type),
                                                        mrc::pymrc::cast_from_pyobject(task_payload));
    }

    auto stage = builder.construct_object<DeserializeStage>(name, batch_size, ensure_sliceable_index, std::move(task));

    return stage;
}

}  // namespace morpheus
