from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceInfo:
    source: str
    page: int | None = None


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[SourceInfo] = field(default_factory=list)
    retrieved_documents: list[Any] = field(default_factory=list)
    latency_seconds: float = 0.0
