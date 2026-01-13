"""
核心模块

包含RAG流水线和答案生成器
"""

from .generator import (
    Generator,
    GenerationResult,
    SourceRef,
    PromptBuilder,
    SourceTracer,
)

from .pipeline import (
    RAGPipeline,
    QueryResult,
    IndexResult,
    ConversationManager,
    get_pipeline,
    reset_pipeline,
)

__all__ = [
    # 生成器
    "Generator",
    "GenerationResult",
    "SourceRef",
    "PromptBuilder",
    "SourceTracer",
    # 流水线
    "RAGPipeline",
    "QueryResult",
    "IndexResult",
    "ConversationManager",
    "get_pipeline",
    "reset_pipeline",
]
