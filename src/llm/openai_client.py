from __future__ import annotations

import json
import os
import time
from typing import Dict, Any

from .base import BaseLLMClient
from .schemas import LLMRequest, LLMResponse


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('model', 'gpt-4o-mini')
        self.api_key = os.getenv(config.get('api_key_env', 'OPENAI_API_KEY'))

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                raw_text='',
                parsed={},
                model_name=self.model,
                provider='openai',
                success=False,
                error='Missing OPENAI_API_KEY (or configured env var).',
            )
        start = time.time()
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                messages=[
                    {'role': 'system', 'content': request.system_prompt},
                    {'role': 'user', 'content': request.user_prompt},
                ],
                response_format={'type': 'json_object'} if request.response_format == 'json' else None,
            )
            text = completion.choices[0].message.content or '{}'
            parsed = json.loads(text) if request.response_format == 'json' else {'text': text}
            return LLMResponse(
                raw_text=text,
                parsed=parsed,
                model_name=self.model,
                provider='openai',
                latency_ms=(time.time() - start) * 1000,
                success=True,
            )
        except Exception as exc:
            return LLMResponse(
                raw_text='',
                parsed={},
                model_name=self.model,
                provider='openai',
                latency_ms=(time.time() - start) * 1000,
                success=False,
                error=str(exc),
            )
