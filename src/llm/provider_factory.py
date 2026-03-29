from __future__ import annotations

from typing import Dict, Any

from .base import BaseLLMClient
from .mock_client import MockLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient


class PlaceholderAPIClient(MockLLMClient):
    pass


def build_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    provider = config.get('provider', 'mock')
    mode = config.get('mode', 'mock')
    if mode == 'mock' or provider == 'mock':
        return MockLLMClient(config)
    if provider == 'openai':
        return OpenAIClient(config)
    if provider == 'anthropic':
        return AnthropicClient(config)
    return PlaceholderAPIClient(config)
