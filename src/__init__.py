"""
RAG增强智能问答系统
=====================

基于检索增强生成(RAG)技术的中英双语智能问答系统

模块结构:
- config: 配置管理
- processing: 文档处理与分块
- retrieval: 嵌入与检索
- core: 核心流水线与生成
- api: HTTP API服务
"""

__version__ = "1.0.0"
__author__ = "NLP Course Project"

# 配置
from .config import Config, get_config, update_config, update_llm_config

# 文档处理
from .processing import (
    Document,
    Chunk,
    DocumentProcessor,
    DocumentLoader,
    DocumentParseError,
    TextChunker,
    SemanticChunker,
    RecursiveChunker,
)

# 检索
from .retrieval import (
    Embedder,
    VectorStore,
    SearchResult,
    BaseRetriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    Reranker,
    RetrieverFactory,
)

# 核心
from .core import (
    Generator,
    GenerationResult,
    SourceRef,
    PromptBuilder,
    SourceTracer,
    RAGPipeline,
    QueryResult,
    IndexResult,
    ConversationManager,
    get_pipeline,
    reset_pipeline,
)

__all__ = [
    # 配置
    "Config",
    "get_config",
    "update_config",
    "update_llm_config",
    # 文档处理
    "Document",
    "Chunk",
    "DocumentProcessor",
    "DocumentLoader",
    "DocumentParseError",
    "TextChunker",
    "SemanticChunker",
    "RecursiveChunker",
    # 检索
    "Embedder",
    "VectorStore",
    "SearchResult",
    "BaseRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "Reranker",
    "RetrieverFactory",
    # 核心
    "Generator",
    "GenerationResult",
    "SourceRef",
    "PromptBuilder",
    "SourceTracer",
    "RAGPipeline",
    "QueryResult",
    "IndexResult",
    "ConversationManager",
    "get_pipeline",
    "reset_pipeline",
]
