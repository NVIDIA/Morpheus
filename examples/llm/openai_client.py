from examples.llm.llm_client import LLMClient
import asyncio
import openai

class OpenAICompletionLLMClient(LLMClient):

    _api_key: str
    _model: str

    def __init__(
            self,
            api_key: str | None = None,
            model: str | None = None
    ):
        self._api_key = api_key
        self._model = model or "ada"

    async def generate_async(self, prompt: str) -> str:
        result = await openai.Completion.acreate(model=self._model, prompt=prompt)
        return result.choices[0].text

    async def generate_multiple_async(self, prompts: [str]) -> [str]:
      tasks = [asyncio.ensure_future(self.generate_async(prompt)) for prompt in prompts]
      await asyncio.wait(tasks)
      return [task.result() for task in tasks]
