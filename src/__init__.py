"""
RAG增强智能问答系统
=====================

基于检索增强生成(RAG)技术的中英双语智能问答系统

模块说明:
- config: 配置管理
- document_processor: 文档处理与分块
- embedder: 文本嵌入与向量存储
- retriever: 文档检索
- generator: 答案生成
- rag_pipeline: RAG流水线编排
- api: HTTP API服务
"""

__version__ = "1.0.0"
__author__ = "NLP Course Project"

from .config import Config
from .document_processor import DocumentProcessor, Chunk, Document
from .embedder import Embedder, VectorStore
from .retriever import HybridRetriever, Reranker
from .generator import Generator
from .rag_pipeline import RAGPipeline

__all__ = [
    "Config",
    "DocumentProcessor",
    "Chunk",
    "Document",
    "Embedder",
    "VectorStore",
    "HybridRetriever",
    "Reranker",
    "Generator",
    "RAGPipeline",
]
