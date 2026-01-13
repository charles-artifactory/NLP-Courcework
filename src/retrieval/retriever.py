"""
检索模块

实现稠密检索、稀疏检索、混合检索和重排序
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

import numpy as np

from .embedder import Embedder, VectorStore, SearchResult
from ..processing.document_processor import Chunk

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """检索器基类"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """检索相关文档"""
        pass


class DenseRetriever(BaseRetriever):
    """
    稠密检索器
    
    使用向量相似度进行检索
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[SearchResult]:
        """执行稠密检索"""
        if not query.strip():
            return []
        
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results


class SparseRetriever(BaseRetriever):
    """
    稀疏检索器 - 创新点
    
    使用BM25算法进行检索
    """
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.chunk_ids = []
        self.metadatas = []
        self._initialized = False
    
    def build_index(self, chunks: List[Chunk]) -> None:
        """构建BM25索引"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25未安装，稀疏检索不可用")
            return
        
        if not chunks:
            return
        
        self.documents = []
        self.chunk_ids = []
        self.metadatas = []
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self.documents.append(tokens)
            self.chunk_ids.append(chunk.id)
            self.metadatas.append({
                "document_id": chunk.document_id,
                **chunk.metadata
            })
        
        self.bm25 = BM25Okapi(self.documents)
        self._initialized = True
        
        logger.info(f"BM25索引构建完成，共 {len(chunks)} 个文档")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """增量添加文本块到索引"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self.documents.append(tokens)
            self.chunk_ids.append(chunk.id)
            self.metadatas.append({
                "document_id": chunk.document_id,
                **chunk.metadata
            })
        
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
            self._initialized = True
    
    def remove_document(self, document_id: str) -> None:
        """从索引中移除指定文档"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        
        indices_to_remove = [
            i for i, meta in enumerate(self.metadatas)
            if meta.get("document_id") == document_id
        ]
        
        for i in sorted(indices_to_remove, reverse=True):
            del self.documents[i]
            del self.chunk_ids[i]
            del self.metadatas[i]
        
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
        else:
            self.bm25 = None
            self._initialized = False
    
    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词（支持中英文）"""
        import re
        
        tokens = []
        parts = text.lower().split()
        
        for part in parts:
            if re.search(r'[\u4e00-\u9fff]', part):
                for char in part:
                    if '\u4e00' <= char <= '\u9fff':
                        tokens.append(char)
                    elif char.isalnum():
                        tokens.append(char)
            else:
                clean = re.sub(r'[^\w]', '', part)
                if clean:
                    tokens.append(clean)
        
        return tokens
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """执行BM25检索"""
        if not self._initialized or not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    chunk_id=self.chunk_ids[idx],
                    content=" ".join(self.documents[idx]),
                    score=float(scores[idx]),
                    metadata=self.metadatas[idx]
                ))
        
        return results


class HybridRetriever(BaseRetriever):
    """
    混合检索器 - 创新点
    
    结合稠密检索和稀疏检索的优势
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        alpha: float = 0.7,
        use_reranker: bool = True,
        reranker: "Reranker" = None,
        similarity_threshold: float = 0.3
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha
        self.use_reranker = use_reranker
        self.reranker = reranker
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int = None
    ) -> List[SearchResult]:
        """执行混合检索"""
        if rerank_top_k is None:
            rerank_top_k = top_k * 2
        
        dense_results = self.dense_retriever.retrieve(query, rerank_top_k)
        sparse_results = self.sparse_retriever.retrieve(query, rerank_top_k)
        
        # 先用原始分数过滤稠密检索结果（关键！避免归一化后阈值失效）
        dense_results = self._pre_filter_dense(dense_results)
        
        if not dense_results and not sparse_results:
            return []
        
        if not sparse_results:
            results = dense_results[:top_k]
        elif not dense_results:
            results = sparse_results[:top_k]
        else:
            dense_results = self._normalize_scores(dense_results)
            sparse_results = self._normalize_scores(sparse_results)
            results = self._fuse_results(dense_results, sparse_results)
        
        if self.use_reranker and self.reranker and self.reranker.is_available and results:
            results = self.reranker.rerank(query, results, top_k)
            # 重排序后用重排序分数过滤
            results = self._filter_by_rerank_score(results)
        else:
            results = results[:top_k]
        
        # 最终过滤：确保返回的结果分数足够高
        results = self._final_filter(results)
        
        return results
    
    def _pre_filter_dense(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        用原始余弦相似度预过滤稠密检索结果
        
        这在归一化之前进行，避免低相关结果因归一化而通过阈值
        """
        if not results:
            return results

        filtered = [r for r in results if r.score >= self.similarity_threshold]
        
        if len(filtered) < len(results):
            logger.debug(f"稠密预过滤: {len(results)} -> {len(filtered)} 个结果 (阈值={self.similarity_threshold})")
        
        return filtered
    
    def _filter_by_rerank_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        用重排序分数过滤结果
        
        BGE-reranker的分数范围大约是 [-10, 10]，> 0 表示相关
        使用更严格的阈值来过滤不相关结果
        """
        if not results:
            return results
        
        # 使用更严格的阈值：重排序分数需要 > -2 才保留
        # 这比之前的 > 0 更宽松一些，但能过滤明显不相关的
        rerank_threshold = -2.0
        
        filtered = []
        for r in results:
            if hasattr(r, 'rerank_score') and r.rerank_score is not None:
                if r.rerank_score > rerank_threshold:
                    filtered.append(r)
            else:
                filtered.append(r)
        
        if len(filtered) < len(results):
            logger.debug(f"重排序过滤: {len(results)} -> {len(filtered)} 个结果")
        
        return filtered
    
    def _final_filter(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        最终分数过滤
        
        确保返回的结果具有足够高的相关性分数
        同时检查重排序分数和融合分数，两者都要满足阈值
        """
        if not results:
            return results
        
        # 融合分数阈值（归一化后范围 [0, 1]）
        score_threshold = 0.25
        # 重排序分数阈值（BGE-reranker 范围约 [-10, 10]，> 0 表示相关）
        rerank_threshold = 0.0
        
        filtered = []
        for r in results:
            has_rerank = hasattr(r, 'rerank_score') and r.rerank_score is not None
            
            if has_rerank:
                # 有重排序分数：两个条件都要满足
                # 1. 重排序分数 > 0（表示语义相关）
                # 2. 融合分数 >= 阈值（确保基础相似度足够）
                if r.rerank_score > rerank_threshold and r.score >= score_threshold:
                    filtered.append(r)
            else:
                # 无重排序分数：只检查融合分数
                if r.score >= score_threshold:
                    filtered.append(r)
        
        if len(filtered) < len(results):
            logger.info(f"最终过滤: {len(results)} -> {len(filtered)} 个结果")
        
        return filtered
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """归一化分数到 [0, 1]"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                r.score = 1.0
        else:
            for r in results:
                r.score = (r.score - min_score) / (max_score - min_score)
        
        return results
    
    def _fuse_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult]
    ) -> List[SearchResult]:
        """融合稠密和稀疏检索结果"""
        score_map: Dict[str, Dict[str, float]] = {}
        result_map: Dict[str, SearchResult] = {}
        
        for r in dense_results:
            score_map[r.chunk_id] = {"dense": r.score, "sparse": 0}
            result_map[r.chunk_id] = r
        
        for r in sparse_results:
            if r.chunk_id in score_map:
                score_map[r.chunk_id]["sparse"] = r.score
            else:
                score_map[r.chunk_id] = {"dense": 0, "sparse": r.score}
                result_map[r.chunk_id] = r
        
        fused = []
        for chunk_id, scores in score_map.items():
            final_score = (
                self.alpha * scores["dense"] +
                (1 - self.alpha) * scores["sparse"]
            )
            result = result_map[chunk_id]
            result.score = final_score
            fused.append(result)
        
        fused.sort(key=lambda x: x.score, reverse=True)
        
        return fused


class Reranker:
    """
    重排序器 - 创新点
    
    使用Cross-Encoder对检索结果进行重排序
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"正在加载重排序模型: {self.model_name}")
            
            try:
                self.model = CrossEncoder(self.model_name)
                self._initialized = True
                logger.info("重排序模型加载成功")
            except Exception as e:
                logger.warning(f"重排序模型加载失败: {e}")
                self._initialized = False
                
        except ImportError:
            logger.warning("sentence-transformers未安装，重排序功能不可用")
            self._initialized = False
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """对检索结果重排序"""
        if not self._initialized or not results:
            return results[:top_k]
        
        pairs = [(query, r.content) for r in results]
        
        try:
            scores = self.model.predict(pairs)
            
            for i, result in enumerate(results):
                result.rerank_score = float(scores[i])
            
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
        
        return results[:top_k]
    
    @property
    def is_available(self) -> bool:
        """检查重排序器是否可用"""
        return self._initialized


class RetrieverFactory:
    """检索器工厂"""
    
    @staticmethod
    def create_hybrid_retriever(
        embedder: Embedder,
        vector_store: VectorStore,
        chunks: List[Chunk] = None,
        alpha: float = 0.7,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base"
    ) -> HybridRetriever:
        """创建混合检索器"""
        dense_retriever = DenseRetriever(embedder, vector_store)
        
        sparse_retriever = SparseRetriever()
        if chunks:
            sparse_retriever.build_index(chunks)
        
        reranker = None
        if use_reranker:
            reranker = Reranker(reranker_model)
        
        return HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            alpha=alpha,
            use_reranker=use_reranker,
            reranker=reranker
        )
