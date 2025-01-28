# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from morpheus.messages import ControlMessage
from morpheus.utils.control_message_utils import CMDefaultFailureContextManager
from morpheus.utils.control_message_utils import cm_set_failure
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed


def test_cm_set_failure():
    control_message = ControlMessage()
    reason = "Test Failure"

    assert cm_set_failure(control_message, reason) == control_message

    assert control_message.get_metadata("cm_failed") is True
    assert control_message.get_metadata("cm_failed_reason") == reason


def test_skip_forward_on_cm_failed():
    control_message = ControlMessage()
    reason = "Test Failure"
    cm_set_failure(control_message, reason)

    # pylint: disable=unused-argument
    @cm_skip_processing_if_failed
    def dummy_func(cm, *args, **kwargs):
        return "Function Executed"

    assert dummy_func(control_message) == control_message

    control_message2 = ControlMessage()
    assert dummy_func(control_message2) == "Function Executed"


def test_cm_default_failure_context_manager_no_exception():
    control_message = ControlMessage()
    with CMDefaultFailureContextManager(control_message):
        pass

        assert control_message.get_metadata("cm_failed") is None


def test_cm_default_failure_context_manager_with_exception():
    control_message = ControlMessage()
    with CMDefaultFailureContextManager(control_message):
        raise RuntimeError("Test Exception")

    assert control_message.get_metadata("cm_failed") is True
    assert control_message.get_metadata("cm_failed_reason") == "Test Exception"


if (__name__ == "__main__"):
    test_cm_set_failure()
    test_skip_forward_on_cm_failed()
    test_cm_default_failure_context_manager_no_exception()
    test_cm_default_failure_context_manager_with_exception()
