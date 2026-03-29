from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    response_format: str = "json"
    temperature: float = 0.1
    max_tokens: int = 2000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    raw_text: str
    parsed: Dict[str, Any]
    model_name: str
    provider: str
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
