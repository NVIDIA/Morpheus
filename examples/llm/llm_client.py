from abc import ABC, abstractmethod

class LLMClient(ABC):

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_multiple_async(self, prompts: [str]) -> [str]:
        pass