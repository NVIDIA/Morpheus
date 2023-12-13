from examples.llm.llm_client import LLMClient

import nemollm
import asyncio

class NemoLLMClient(LLMClient):

    _connection: nemollm.NemoLLM
    _model: str

    def __init__(
            self,
            api_key: str | None = None,
            org_id: str | None = None,
            api_host: str | None = None,
            model: str | None = None
    ):
        self._connection = nemollm.NemoLLM(api_key, org_id, api_host)
        self._model = model or "gpt5b"

    async def generate_async(self, prompt: str) -> str:
        future = self._connection.generate(self._model, prompt, return_type="async")
        result = await asyncio.wrap_future(future)
        response = nemollm.NemoLLM.post_process_generate_response(result, return_text_completion_only=True)
        return response

    async def generate_multiple_async(self, prompts: [str]) -> [str]:
      tasks = [asyncio.ensure_future(self.generate_async(prompt)) for prompt in prompts]
      await asyncio.wait(tasks)
      return [task.result() for task in tasks]