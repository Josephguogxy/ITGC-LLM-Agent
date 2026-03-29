from __future__ import annotations

import json
import os
import time
from typing import Dict, Any

from .base import BaseLLMClient
from .schemas import LLMRequest, LLMResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('model', 'claude-3-5-sonnet-latest')
        self.api_key = os.getenv(config.get('api_key_env', 'ANTHROPIC_API_KEY'))

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                raw_text='',
                parsed={},
                model_name=self.model,
                provider='anthropic',
                success=False,
                error='Missing ANTHROPIC_API_KEY (or configured env var).',
            )
        start = time.time()
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt,
                messages=[{'role': 'user', 'content': request.user_prompt}],
            )
            text = ''.join(block.text for block in msg.content if getattr(block, 'type', '') == 'text') or '{}'
            parsed = json.loads(text) if request.response_format == 'json' else {'text': text}
            return LLMResponse(
                raw_text=text,
                parsed=parsed,
                model_name=self.model,
                provider='anthropic',
                latency_ms=(time.time() - start) * 1000,
                success=True,
            )
        except Exception as exc:
            return LLMResponse(
                raw_text='',
                parsed={},
                model_name=self.model,
                provider='anthropic',
                latency_ms=(time.time() - start) * 1000,
                success=False,
                error=str(exc),
            )
