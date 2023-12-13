from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.messages import MessageMeta, MultiMessage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.utils.logger import configure_logging

from morpheus.pipeline import LinearPipeline
from morpheus.config import Config

from examples.llm.nemo_client import NemoLLMClient
from examples.llm.mock_client import MockLLMClient
from examples.llm.openai_client import OpenAICompletionLLMClient
from examples.llm.llm_client import LLMClient

import logging
import cudf
import mrc
import typing
import asyncio
import os

class PromptCompletionStage(SinglePortStage):

    _client: LLMClient
    _prompt_key: str
    _response_key: str

    def __init__(
            self,
            config: Config,
            client: LLMClient,
            prompt_key:str="prompt",
            response_key:str="response"
    ):
        super().__init__(config)
        self._client = client
        self._prompt_key = prompt_key
        self._response_key = response_key

    @property
    def name(self) -> str:
        return "prompt-completion"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    async def on_data_async(self, message: MultiMessage):
        prompts = [prompt for prompt in message.get_meta(self._prompt_key).to_pandas()]
        responses = await self._client.generate_multiple_async(prompts)
        message.set_meta(self._response_key, responses)
        return message

    def _build_single(self, builder: mrc.Builder, input: StreamPair) -> StreamPair:
        [input_node, input_type] = input
        node = builder.make_node(self.unique_name, mrc.operators.map_async(self.on_data_async))
        builder.make_edge(input_node, node)
        return node, input_type

def test_nemo_pipeline():

    # configure_logging(log_level=logging.DEBUG)

    input_df_a = cudf.DataFrame({"prompt": ["3+5=", "4-2=", "7*1="]})
    input_df_b = cudf.DataFrame({"prompt": ["the man wore a red"]})
    input_df_c = cudf.DataFrame({"prompt": ["blue+red=", "yellow+blue="]})

    config = Config()
    pipeline = LinearPipeline(config)

    # ngc_api_key=os.environ.get("NGC_API_KEY")
    # llm_client=NemoLLMClient(ngc_api_key, model="gpt-43b-002")

    # openai_api_key=os.environ.get("OPENAI_API_KEY")
    # llm_client=OpenAICompletionLLMClient(openai_api_key)

    llm_client=MockLLMClient(response="cool story")

    pipeline.set_source(InMemorySourceStage(config, [input_df_a, input_df_b, input_df_c]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(PromptCompletionStage(config, llm_client))
    pipeline.add_stage(SerializeStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    messages: list[MessageMeta] = sink.get_messages()

    for message in messages:
        print(message.copy_dataframe())

async def run_client():
    # openai_api_key=os.environ.get("OPENAI_API_KEY")
    # client=OpenAICompletionLLMClient(openai_api_key)

    # ngc_api_key=os.environ.get("NGC_API_KEY")
    # client=NemoLLMClient(ngc_api_key)

    client=MockLLMClient(response="cool story")

    responses = await client.generate_multiple_async([
        "the numbers 5 and 6 sum to",
        "blue and red combine to form"
    ])

    print(responses)


def main():
    asyncio.run(run_client())

if __name__ == '__main__':
    test_nemo_pipeline()
    # main()
