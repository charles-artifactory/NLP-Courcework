"""
检索模块

实现稠密检索、稀疏检索、混合检索和重排序
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .embedder import Embedder, VectorStore, SearchResult
from .document_processor import Chunk

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """检索器基类"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
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
        """
        初始化稠密检索器
        
        Args:
            embedder: 嵌入器
            vector_store: 向量存储
        """
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[SearchResult]:
        """
        执行稠密检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        if not query.strip():
            return []
        
        # 计算查询向量
        query_embedding = self.embedder.embed_query(query)
        
        # 向量搜索
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
        """初始化稀疏检索器"""
        self.bm25 = None
        self.documents = []
        self.chunk_ids = []
        self.metadatas = []
        self._initialized = False
    
    def build_index(self, chunks: List[Chunk]) -> None:
        """
        构建BM25索引
        
        Args:
            chunks: 文本块列表
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25未安装，稀疏检索不可用")
            return
        
        if not chunks:
            return
        
        # 提取文本并分词
        self.documents = []
        self.chunk_ids = []
        self.metadatas = []
        
        for chunk in chunks:
            # 简单分词（支持中英文）
            tokens = self._tokenize(chunk.content)
            self.documents.append(tokens)
            self.chunk_ids.append(chunk.id)
            self.metadatas.append({
                "document_id": chunk.document_id,
                **chunk.metadata
            })
        
        # 构建BM25索引
        self.bm25 = BM25Okapi(self.documents)
        self._initialized = True
        
        logger.info(f"BM25索引构建完成，共 {len(chunks)} 个文档")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        增量添加文本块到索引
        
        Args:
            chunks: 新的文本块列表
        """
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
        
        # 重建索引
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
            self._initialized = True
    
    def remove_document(self, document_id: str) -> None:
        """
        从索引中移除指定文档
        
        Args:
            document_id: 文档ID
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        
        # 找出要移除的索引
        indices_to_remove = [
            i for i, meta in enumerate(self.metadatas)
            if meta.get("document_id") == document_id
        ]
        
        # 反向删除以避免索引偏移
        for i in sorted(indices_to_remove, reverse=True):
            del self.documents[i]
            del self.chunk_ids[i]
            del self.metadatas[i]
        
        # 重建索引
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
        else:
            self.bm25 = None
            self._initialized = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        支持中英文混合分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 词元列表
        """
        import re
        
        # 分离中英文
        # 英文按空格分词，中文按字符分词
        tokens = []
        
        # 先按空白分割
        parts = text.lower().split()
        
        for part in parts:
            # 检查是否包含中文
            if re.search(r'[\u4e00-\u9fff]', part):
                # 中文逐字分词
                for char in part:
                    if '\u4e00' <= char <= '\u9fff':
                        tokens.append(char)
                    elif char.isalnum():
                        tokens.append(char)
            else:
                # 英文保持整词
                clean = re.sub(r'[^\w]', '', part)
                if clean:
                    tokens.append(clean)
        
        return tokens
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        执行BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        if not self._initialized or not self.bm25:
            return []
        
        # 查询分词
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # BM25评分
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取top-k索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有相关性的结果
                results.append(SearchResult(
                    chunk_id=self.chunk_ids[idx],
                    content=" ".join(self.documents[idx]),  # 重建文本
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
        reranker: "Reranker" = None
    ):
        """
        初始化混合检索器
        
        Args:
            dense_retriever: 稠密检索器
            sparse_retriever: 稀疏检索器
            alpha: 混合权重（稠密检索权重）
            use_reranker: 是否使用重排序
            reranker: 重排序器实例
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha
        self.use_reranker = use_reranker
        self.reranker = reranker
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int = None
    ) -> List[SearchResult]:
        """
        执行混合检索
        
        算法：
        1. 分别执行稠密检索和稀疏检索
        2. 归一化分数
        3. 加权融合
        4. 可选重排序
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            rerank_top_k: 重排序前的候选数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        if rerank_top_k is None:
            rerank_top_k = top_k * 2
        
        # 稠密检索
        dense_results = self.dense_retriever.retrieve(query, rerank_top_k)
        
        # 稀疏检索
        sparse_results = self.sparse_retriever.retrieve(query, rerank_top_k)
        
        # 如果只有稠密结果
        if not sparse_results:
            results = dense_results[:top_k]
        # 如果只有稀疏结果
        elif not dense_results:
            results = sparse_results[:top_k]
        else:
            # 归一化分数
            dense_results = self._normalize_scores(dense_results)
            sparse_results = self._normalize_scores(sparse_results)
            
            # 融合结果
            results = self._fuse_results(dense_results, sparse_results)
        
        # 重排序
        if self.use_reranker and self.reranker and results:
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def _normalize_scores(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        归一化分数到 [0, 1]
        
        使用Min-Max归一化
        """
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
        """
        融合稠密和稀疏检索结果
        
        公式: final_score = α × dense_score + (1-α) × sparse_score
        """
        score_map: Dict[str, Dict[str, float]] = {}
        result_map: Dict[str, SearchResult] = {}
        
        # 收集稠密检索分数
        for r in dense_results:
            score_map[r.chunk_id] = {"dense": r.score, "sparse": 0}
            result_map[r.chunk_id] = r
        
        # 收集稀疏检索分数
        for r in sparse_results:
            if r.chunk_id in score_map:
                score_map[r.chunk_id]["sparse"] = r.score
            else:
                score_map[r.chunk_id] = {"dense": 0, "sparse": r.score}
                result_map[r.chunk_id] = r
        
        # 计算融合分数
        fused = []
        for chunk_id, scores in score_map.items():
            final_score = (
                self.alpha * scores["dense"] +
                (1 - self.alpha) * scores["sparse"]
            )
            result = result_map[chunk_id]
            result.score = final_score
            fused.append(result)
        
        # 按分数降序排序
        fused.sort(key=lambda x: x.score, reverse=True)
        
        return fused


class Reranker:
    """
    重排序器 - 创新点
    
    使用Cross-Encoder对检索结果进行重排序
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
        """
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
        """
        对检索结果重排序
        
        原理：
        Cross-Encoder同时接收query和document作为输入，
        能够建模更精细的交互关系，排序效果优于Bi-Encoder
        
        Args:
            query: 查询文本
            results: 初步检索结果
            top_k: 返回数量
            
        Returns:
            List[SearchResult]: 重排序后的结果
        """
        if not self._initialized or not results:
            return results[:top_k]
        
        # 构建输入对
        pairs = [(query, r.content) for r in results]
        
        # 获取重排序分数
        try:
            scores = self.model.predict(pairs)
            
            # 更新分数
            for i, result in enumerate(results):
                result.rerank_score = float(scores[i])
            
            # 按重排序分数排序
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 失败时使用原始分数
        
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
        """
        创建混合检索器
        
        Args:
            embedder: 嵌入器
            vector_store: 向量存储
            chunks: 初始文本块（用于BM25索引）
            alpha: 混合权重
            use_reranker: 是否使用重排序
            reranker_model: 重排序模型
            
        Returns:
            HybridRetriever: 混合检索器实例
        """
        # 创建稠密检索器
        dense_retriever = DenseRetriever(embedder, vector_store)
        
        # 创建稀疏检索器
        sparse_retriever = SparseRetriever()
        if chunks:
            sparse_retriever.build_index(chunks)
        
        # 创建重排序器
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
