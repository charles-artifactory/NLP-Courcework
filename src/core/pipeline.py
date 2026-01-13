"""
RAG流水线模块

编排完整的RAG流程，管理对话历史
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..config import Config, get_config
from ..processing.document_processor import DocumentProcessor, Document, Chunk
from ..retrieval.embedder import Embedder, VectorStore, SearchResult
from ..retrieval.retriever import HybridRetriever, DenseRetriever, SparseRetriever, Reranker
from .generator import Generator, GenerationResult, SourceRef

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """索引结果数据类"""
    success: bool
    document_id: Optional[str]
    chunk_count: int
    message: str


@dataclass
class QueryResult:
    """查询结果数据类"""
    answer: str
    sources: List[Dict]
    confidence: float
    search_results: List[SearchResult] = field(default_factory=list)


class ConversationManager:
    """
    对话管理器
    
    管理多轮对话历史
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到对话历史"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.conversations.get(session_id, [])
    
    def clear(self, session_id: str) -> None:
        """清空对话历史"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def clear_all(self) -> None:
        """清空所有对话历史"""
        self.conversations.clear()
    
    def format_history(self, session_id: str) -> str:
        """格式化对话历史为字符串"""
        history = self.get_history(session_id)
        if not history:
            return ""
        
        lines = []
        for msg in history[-6:]:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(lines)


class RAGPipeline:
    """
    RAG流水线
    
    集成文档处理、检索和生成的完整流程
    """
    
    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self._initialized = False
        
        self._document_processor: Optional[DocumentProcessor] = None
        self._embedder: Optional[Embedder] = None
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[HybridRetriever] = None
        self._generator: Optional[Generator] = None
        self._conversation_manager: Optional[ConversationManager] = None
        self._sparse_retriever: Optional[SparseRetriever] = None
        
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, List[Chunk]] = {}
    
    def initialize(self) -> None:
        """初始化所有组件"""
        if self._initialized:
            return
        
        logger.info("正在初始化RAG流水线...")
        
        # 1. 初始化文档处理器
        self._document_processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            supported_formats=self.config.SUPPORTED_FORMATS
        )
        
        # 2. 初始化嵌入器
        try:
            self._embedder = Embedder(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE
            )
        except Exception as e:
            logger.warning(f"主嵌入模型加载失败，尝试备选: {e}")
            self._embedder = Embedder(
                model_name=self.config.EMBEDDING_MODEL_FALLBACK,
                batch_size=self.config.EMBEDDING_BATCH_SIZE
            )
        
        # 3. 初始化向量存储
        self._vector_store = VectorStore(
            persist_directory=str(self.config.VECTOR_DB_PATH),
            collection_name=self.config.COLLECTION_NAME,
            embedder=self._embedder
        )
        
        # 4. 初始化检索器
        self._sparse_retriever = SparseRetriever()
        dense_retriever = DenseRetriever(self._embedder, self._vector_store)
        
        reranker = None
        if self.config.USE_RERANKER:
            try:
                reranker = Reranker(self.config.RERANKER_MODEL)
            except Exception as e:
                logger.warning(f"重排序器初始化失败: {e}")
        
        self._retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=self._sparse_retriever,
            alpha=self.config.HYBRID_ALPHA,
            use_reranker=self.config.USE_RERANKER,
            reranker=reranker,
            similarity_threshold=self.config.SIMILARITY_THRESHOLD
        )
        
        # 5. 初始化生成器
        self._generator = Generator(
            provider=self.config.LLM_PROVIDER,
            model=self.config.LLM_MODEL,
            base_url=self.config.LLM_BASE_URL,
            api_key=self.config.OPENAI_API_KEY,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_NEW_TOKENS
        )
        
        # 6. 初始化对话管理器
        self._conversation_manager = ConversationManager(
            max_history=self.config.MAX_HISTORY_LENGTH
        )
        
        # 7. 恢复已有索引的BM25
        self._rebuild_sparse_index()
        
        self._initialized = True
        logger.info("RAG流水线初始化完成")
    
    def _rebuild_sparse_index(self) -> None:
        """重建稀疏索引"""
        all_docs = self._vector_store.get_all_documents()
        if not all_docs:
            return
        
        try:
            results = self._vector_store.collection.get(include=["documents", "metadatas"])
            if results["documents"]:
                chunks = []
                for i, doc in enumerate(results["documents"]):
                    chunk = Chunk(
                        id=results["ids"][i],
                        document_id=results["metadatas"][i].get("document_id", ""),
                        content=doc,
                        metadata=results["metadatas"][i]
                    )
                    chunks.append(chunk)
                
                self._sparse_retriever.build_index(chunks)
                logger.info(f"重建BM25索引，共 {len(chunks)} 个文本块")
        except Exception as e:
            logger.warning(f"重建稀疏索引失败: {e}")
    
    def _ensure_initialized(self) -> None:
        """确保已初始化"""
        if not self._initialized:
            self.initialize()
    
    def index_document(self, file_path: str) -> IndexResult:
        """索引文档"""
        self._ensure_initialized()
        
        try:
            logger.info(f"正在处理文档: {file_path}")
            document = self._document_processor.load_document(file_path)
            
            chunks = self._document_processor.process(document)
            logger.info(f"文档分块完成，共 {len(chunks)} 个块")
            
            self._vector_store.add_chunks(chunks)
            self._sparse_retriever.add_chunks(chunks)
            
            self._documents[document.id] = document
            self._chunks[document.id] = chunks
            
            return IndexResult(
                success=True,
                document_id=document.id,
                chunk_count=len(chunks),
                message=f"成功索引文档 '{document.filename}'，共 {len(chunks)} 个文本块"
            )
            
        except Exception as e:
            logger.error(f"索引文档失败: {e}")
            return IndexResult(
                success=False,
                document_id=None,
                chunk_count=0,
                message=f"索引失败: {str(e)}"
            )
    
    def index_documents(self, file_paths: List[str]) -> List[IndexResult]:
        """批量索引文档"""
        results = []
        for file_path in file_paths:
            result = self.index_document(file_path)
            results.append(result)
        return results
    
    def query(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = None
    ) -> QueryResult:
        """处理用户问题"""
        self._ensure_initialized()
        
        if top_k is None:
            top_k = self.config.TOP_K
        
        history = self._conversation_manager.get_history(session_id)
        
        search_results = self._retriever.retrieve(
            question,
            top_k=top_k,
            rerank_top_k=self.config.RERANK_TOP_K
        )
        
        gen_result = self._generator.generate(
            query=question,
            search_results=search_results,
            history=history
        )
        
        self._conversation_manager.add_message(session_id, "user", question)
        self._conversation_manager.add_message(session_id, "assistant", gen_result.answer)
        
        sources = self._generator.source_tracer.format_sources(search_results) if search_results else []
        
        return QueryResult(
            answer=gen_result.answer,
            sources=sources,
            confidence=gen_result.confidence if search_results else 1.0,
            search_results=search_results
        )
    
    def query_stream(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = None
    ) -> Iterator[Tuple[str, List[Dict]]]:
        """流式处理用户问题"""
        self._ensure_initialized()
        
        if top_k is None:
            top_k = self.config.TOP_K
        
        history = self._conversation_manager.get_history(session_id)
        search_results = self._retriever.retrieve(
            question,
            top_k=top_k,
            rerank_top_k=self.config.RERANK_TOP_K
        )
        
        sources = self._generator.source_tracer.format_sources(search_results) if search_results else []
        
        full_answer = ""
        for chunk in self._generator.generate_stream(question, search_results, history):
            full_answer += chunk
            yield chunk, sources
        
        self._conversation_manager.add_message(session_id, "user", question)
        self._conversation_manager.add_message(session_id, "assistant", full_answer)
    
    def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        self._ensure_initialized()
        
        try:
            deleted_count = self._vector_store.delete_by_document(document_id)
            self._sparse_retriever.remove_document(document_id)
            
            if document_id in self._documents:
                del self._documents[document_id]
            if document_id in self._chunks:
                del self._chunks[document_id]
            
            logger.info(f"删除文档 {document_id}，共 {deleted_count} 个块")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def get_documents(self) -> List[Dict]:
        """获取已索引的文档列表"""
        self._ensure_initialized()
        return self._vector_store.get_all_documents()
    
    def clear_conversation(self, session_id: str = "default") -> None:
        """清空对话历史"""
        self._ensure_initialized()
        self._conversation_manager.clear(session_id)
    
    def clear_all_data(self) -> None:
        """清空所有数据"""
        self._ensure_initialized()
        
        self._vector_store.clear()
        self._sparse_retriever = SparseRetriever()
        self._retriever._sparse_retriever = self._sparse_retriever
        self._conversation_manager.clear_all()
        self._documents.clear()
        self._chunks.clear()
        
        logger.info("已清空所有数据")
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        self._ensure_initialized()
        
        documents = self.get_documents()
        total_chunks = sum(d.get("chunk_count", 0) for d in documents)
        
        return {
            "document_count": len(documents),
            "chunk_count": total_chunks,
            "vector_store_count": self._vector_store.count,
            "embedding_model": self._embedder.model_name,
            "llm_provider": self.config.LLM_PROVIDER,
            "llm_model": self.config.LLM_MODEL
        }
    
    def update_generator(
        self,
        provider: str = None,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> bool:
        """动态更新生成器配置"""
        self._ensure_initialized()
        
        try:
            if provider is not None:
                self.config.LLM_PROVIDER = provider
            if base_url is not None:
                self.config.LLM_BASE_URL = base_url
            if api_key is not None:
                self.config.OPENAI_API_KEY = api_key
            if temperature is not None:
                self.config.TEMPERATURE = temperature
            if max_tokens is not None:
                self.config.MAX_NEW_TOKENS = max_tokens
            
            if model is not None:
                if self.config.LLM_PROVIDER == "openai":
                    self.config.OPENAI_MODEL = model
                self.config.LLM_MODEL = model
            
            self._generator = Generator(
                provider=self.config.LLM_PROVIDER,
                model=self.config.LLM_MODEL,
                base_url=self.config.LLM_BASE_URL,
                api_key=self.config.OPENAI_API_KEY,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_NEW_TOKENS
            )
            
            logger.info(f"生成器配置已更新: provider={self.config.LLM_PROVIDER}, model={self.config.LLM_MODEL}")
            return True
            
        except Exception as e:
            logger.error(f"更新生成器失败: {e}")
            return False
    
    def get_llm_config(self) -> Dict:
        """获取当前LLM配置"""
        return {
            "provider": self.config.LLM_PROVIDER,
            "model": self.config.LLM_MODEL,
            "base_url": self.config.LLM_BASE_URL,
            "api_key": self.config.OPENAI_API_KEY[:8] + "..." if self.config.OPENAI_API_KEY else "",
            "temperature": self.config.TEMPERATURE,
            "max_tokens": self.config.MAX_NEW_TOKENS
        }


# 创建全局实例
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """获取RAG流水线单例"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def reset_pipeline() -> None:
    """重置RAG流水线"""
    global _pipeline
    _pipeline = None
