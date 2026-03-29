from __future__ import annotations

from typing import Any, Dict

from .provider_factory import build_llm_client
from .schemas import LLMRequest, LLMResponse


class LLMService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = build_llm_client(config)

    def call(self, *, system_prompt: str, user_prompt: str, agent_role: str, response_format: str = "json", metadata: Dict[str, Any] | None = None) -> LLMResponse:
        req = LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            metadata={"agent_role": agent_role, **(metadata or {})},
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 2000),
        )
        return self.client.generate(req)
