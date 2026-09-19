from backend.app.core.config import settings
from backend.app.llm.provider import LLMProvider, MockLLMProvider

def get_llm_provider() -> LLMProvider:
    """
    Factory function to get the configured LLM provider.
    """
    provider_name = settings.LLM_PROVIDER.lower()
    
    if provider_name == "mock":
        return MockLLMProvider()
    else:
        # Future real providers (e.g., OpenAIProvider, AnthropicProvider) go here
        # For now, fallback to mock and print a warning
        print(f"Warning: Provider '{provider_name}' not implemented. Falling back to MockLLMProvider.")
        return MockLLMProvider()
