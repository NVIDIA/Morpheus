from examples.llm.llm_client import LLMClient

class MockLLMClient(LLMClient):

    _response: str

    def __init__(self, response: str):
        self._response = response

    def _query(self, prompt: str) -> str:
        return self._response

    async def generate_async(self, prompt: str) -> str:
        return self._query(prompt)

    async def generate_multiple_async(self, prompts: [str]) -> [str]:
        return [self._query(prompt) for prompt in prompts]