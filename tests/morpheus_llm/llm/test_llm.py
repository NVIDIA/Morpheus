# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import asyncio.mixins
import collections

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus_llm.llm import LLMContext
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm import LLMLambdaNode
from morpheus_llm.llm import LLMNode
from morpheus_llm.llm import LLMNodeBase
from morpheus_llm.llm import LLMTaskHandler


async def yield_to_loop():
    # Imitate what would happen with asyncio.sleep(0.001)
    # This yields to the loop waiting for any ready and scheduled tasks to complete before returning
    # Different from asyncio.sleep(0) which yields to the loop but get scheduled again immediately

    def _set_result_unless_cancelled(fut, result):
        """Helper setting the result only if the future was not cancelled."""
        if fut.cancelled():
            return
        fut.set_result(result)

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    handle = loop.call_later(0, _set_result_unless_cancelled, future, None)
    try:
        return await future
    finally:
        handle.cancel()


class AutoResetEvent(asyncio.mixins._LoopBoundMixin):
    """Asynchronous equivalent to threading.Event.

    Class implementing event objects. An event manages a flag that can be set
    to true with the set() method and reset to false with the clear() method.
    The wait() method blocks until the flag is true. The flag is initially
    false.
    """

    def __init__(self, *, loop=asyncio.mixins._marker):
        super().__init__(loop=loop)
        self._waiters = collections.deque()
        self._value = False

    def __repr__(self):
        res = super().__repr__()
        extra = 'set' if self._value else 'unset'
        if self._waiters:
            extra = f'{extra}, waiters:{len(self._waiters)}'
        return f'<{res[1:-1]} [{extra}]>'

    def is_set(self):
        """Return True if and only if the internal flag is true."""
        return self._value

    def set(self):
        """Set the internal flag to true. All coroutines waiting for it to
        become true are awakened. Coroutine that call wait() once the flag is
        true will not block at all.
        """
        if not self._value:
            self._value = True

            for fut in self._waiters:
                if not fut.done():
                    fut.set_result(True)

                    # Exit because we only want to
                    break

    def clear(self):
        """Reset the internal flag to false. Subsequently, coroutines calling
        wait() will block until set() is called to set the internal flag
        to true again."""
        self._value = False

    async def wait(self):
        """Block until the internal flag is true.

        If the internal flag is true on entry, return True
        immediately.  Otherwise, block until another coroutine calls
        set() to set the flag to true, then return True.
        """
        if self._value:
            # Was already set, reset it now
            self.clear()
            return True

        fut = self._get_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
            return True
        finally:
            # Automatically clear the flag to reset the event
            self.clear()
            self._waiters.remove(fut)


async def test_simple_engine():

    event = AutoResetEvent()
    has_started = False

    dataset = {
        "questions": ["Question A", "Question B", "Question C"],
        "indices": [1, 2, 0],
        "values": [4, 5, 6],
        "answers": ["Question A - 5", "Question B - 6", "Question C - 4"],
    }

    async def source_node():
        nonlocal has_started
        has_started = True

        await event.wait()

        return dataset

    class NestedNode(LLMNode):

        def __init__(self) -> None:
            super().__init__()

            # Create some dummy nodes
            async def node1(questions: list[str]):

                return [f"{q} - " for q in questions]

            self.add_node("node1", inputs=[("questions")], node=LLMLambdaNode(node1))

            async def node2(node1: list[str]):

                return node1

            # Add a pass through node with input the same name as the output of node1
            self.add_node("node2", inputs=["/node1"], node=LLMLambdaNode(node2))

            async def node3(questions: list[str], indices: list[int], values: list[int]):
                # Reorder based on the indices
                return [f"{q}{values[indices[i]]}" for i, q in enumerate(questions)]

            self.add_node("node3",
                          inputs=[("/node2", "questions"), "indices", "values"],
                          node=LLMLambdaNode(node3),
                          is_output=True)

    class SinkNode(LLMNodeBase):

        def get_input_names(self):
            return ["nested_answers", "answers"]

        async def execute(self, context: LLMContext):  # pylint: disable=invalid-overridden-method

            nested_answers = context.get_input("nested_answers")
            answers = context.get_input("answers")

            assert nested_answers == answers

            context.set_output(answers)

            return context

    class SimpleTaskHandler(LLMTaskHandler):

        def get_input_names(self):
            return ["response"]

        async def try_handle(self, context: LLMContext):  # pylint: disable=invalid-overridden-method

            with context.message().payload().mutable_dataframe() as df:
                df["response"] = context.get_input()

            return [context.message()]

    engine = LLMEngine()

    engine.add_node("source", node=LLMLambdaNode(source_node))
    engine.add_node("nested", inputs=[("/source/*", "*")], node=NestedNode())
    engine.add_node("sink", inputs=[("/nested", "nested_answers"), ("/source/answers", "answers")], node=SinkNode())

    # # Add our task handler
    engine.add_task_handler(inputs=["/sink"], handler=SimpleTaskHandler())

    # Create a control message with a single task which uses the LangChain agent executor
    message = ControlMessage()

    message.add_task("llm_engine", {
        "task_type": "template", "task_dict": {
            "task_type": "dictionary",
            "model_name": "test",
        }
    })

    payload = cudf.DataFrame()
    message.payload(MessageMeta(payload))

    # Finally, run the engine
    run_task = asyncio.ensure_future(engine.run(message))

    # Yield to the engine
    await yield_to_loop()

    # Should have started but waiting on the event
    assert has_started
    assert not run_task.done()

    event.set()

    result = await run_task

    assert len(result) == 1
    assert DatasetManager.df_equal(result[0].payload().df["response"], dataset["answers"])
