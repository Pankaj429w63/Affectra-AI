from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates an explanation given a prompt."""
        pass

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        """
        Mock implementation for testing without an API key.
        Returns a deterministic pseudo-explanation.
        """
        # A real implementation would call an API. We just return a standard message.
        # We can extract the labels from the prompt or just return a generic text for mock purposes.
        return (
            "This is a mock explanation generated locally for testing purposes. "
            "The Affectra AI model has analyzed the input and determined its emotion and sentiment based "
            "on the combined features. In a production environment, a real LLM would provide a tailored, "
            "human-friendly explanation detailing why this specific emotion and sentiment were detected."
        )
