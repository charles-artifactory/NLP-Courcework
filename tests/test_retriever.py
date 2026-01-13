"""
检索模块测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.document_processor import Chunk
from src.retrieval.embedder import SearchResult
from src.retrieval.retriever import (
    SparseRetriever,
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RetrieverFactory
)


class TestSparseRetriever:
    """稀疏检索器测试类"""
    
    @pytest.fixture
    def retriever(self):
        """创建稀疏检索器"""
        return SparseRetriever()
    
    @pytest.fixture
    def sample_chunks(self):
        """创建示例文本块"""
        return [
            Chunk(
                id="chunk_1",
                document_id="doc_1",
                content="机器学习是人工智能的一个重要分支，它使用算法让计算机从数据中学习。",
                metadata={"filename": "ml.txt"}
            ),
            Chunk(
                id="chunk_2",
                document_id="doc_1",
                content="深度学习是机器学习的子领域，使用神经网络进行特征学习。",
                metadata={"filename": "ml.txt"}
            ),
            Chunk(
                id="chunk_3",
                document_id="doc_2",
                content="自然语言处理技术可以让计算机理解和生成人类语言。",
                metadata={"filename": "nlp.txt"}
            ),
            Chunk(
                id="chunk_4",
                document_id="doc_2",
                content="检索增强生成(RAG)结合了信息检索和文本生成技术。",
                metadata={"filename": "rag.txt"}
            ),
        ]
    
    def test_build_index(self, retriever, sample_chunks):
        """测试构建BM25索引"""
        retriever.build_index(sample_chunks)
        
        assert retriever._initialized
        assert len(retriever.documents) == len(sample_chunks)
        assert len(retriever.chunk_ids) == len(sample_chunks)
    
    def test_retrieve_relevant(self, retriever, sample_chunks):
        """测试检索相关文档"""
        retriever.build_index(sample_chunks)
        
        results = retriever.retrieve("什么是机器学习", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        
        # 机器学习相关的块应该排在前面
        if results:
            assert results[0].score > 0
    
    def test_retrieve_chinese_query(self, retriever, sample_chunks):
        """测试中文查询"""
        retriever.build_index(sample_chunks)
        
        results = retriever.retrieve("深度学习和神经网络", top_k=3)
        
        assert len(results) > 0
        # 深度学习相关的块应该有较高分数
    
    def test_retrieve_empty_query(self, retriever, sample_chunks):
        """测试空查询"""
        retriever.build_index(sample_chunks)
        
        results = retriever.retrieve("", top_k=3)
        
        assert results == []
    
    def test_retrieve_no_index(self, retriever):
        """测试未建索引时检索"""
        results = retriever.retrieve("test query", top_k=3)
        
        assert results == []
    
    def test_add_chunks_incremental(self, retriever, sample_chunks):
        """测试增量添加文本块"""
        # 先添加前两个
        retriever.build_index(sample_chunks[:2])
        assert len(retriever.documents) == 2
        
        # 再添加后两个
        retriever.add_chunks(sample_chunks[2:])
        assert len(retriever.documents) == 4
    
    def test_remove_document(self, retriever, sample_chunks):
        """测试删除文档"""
        retriever.build_index(sample_chunks)
        initial_count = len(retriever.documents)
        
        retriever.remove_document("doc_1")
        
        # doc_1有2个块，删除后应该减少2个
        assert len(retriever.documents) == initial_count - 2
    
    def test_tokenize_mixed_language(self, retriever):
        """测试中英文混合分词"""
        text = "NLP是Natural Language Processing的缩写"
        tokens = retriever._tokenize(text)
        
        assert len(tokens) > 0
        # 应该包含英文词和中文字
        assert any(t.isascii() for t in tokens)
        assert any('\u4e00' <= t <= '\u9fff' for t in tokens)


class TestHybridRetriever:
    """混合检索器测试类"""
    
    @pytest.fixture
    def mock_dense_retriever(self):
        """创建模拟的稠密检索器"""
        mock = Mock()
        mock.retrieve.return_value = [
            SearchResult(chunk_id="chunk_1", content="机器学习内容", score=0.9, metadata={}),
            SearchResult(chunk_id="chunk_2", content="深度学习内容", score=0.7, metadata={}),
        ]
        return mock
    
    @pytest.fixture
    def mock_sparse_retriever(self):
        """创建模拟的稀疏检索器"""
        mock = Mock()
        mock.retrieve.return_value = [
            SearchResult(chunk_id="chunk_1", content="机器学习内容", score=5.0, metadata={}),
            SearchResult(chunk_id="chunk_3", content="NLP内容", score=3.0, metadata={}),
        ]
        return mock
    
    def test_hybrid_retrieve(self, mock_dense_retriever, mock_sparse_retriever):
        """测试混合检索"""
        hybrid = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            alpha=0.7,
            use_reranker=False
        )
        
        results = hybrid.retrieve("测试查询", top_k=3)
        
        assert len(results) <= 3
        # chunk_1应该在结果中（两个检索器都返回了它）
        chunk_ids = [r.chunk_id for r in results]
        assert "chunk_1" in chunk_ids
    
    def test_normalize_scores(self, mock_dense_retriever, mock_sparse_retriever):
        """测试分数归一化"""
        hybrid = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            alpha=0.5,
            use_reranker=False
        )
        
        results = [
            SearchResult(chunk_id="1", content="", score=10.0, metadata={}),
            SearchResult(chunk_id="2", content="", score=5.0, metadata={}),
            SearchResult(chunk_id="3", content="", score=0.0, metadata={}),
        ]
        
        normalized = hybrid._normalize_scores(results)
        
        # 分数应该在 [0, 1] 范围内
        for r in normalized:
            assert 0 <= r.score <= 1
        
        # 最高分应该是1
        assert normalized[0].score == 1.0
        # 最低分应该是0
        assert normalized[2].score == 0.0
    
    def test_fuse_results(self, mock_dense_retriever, mock_sparse_retriever):
        """测试结果融合"""
        hybrid = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            alpha=0.5,
            use_reranker=False
        )
        
        dense_results = [
            SearchResult(chunk_id="1", content="A", score=0.8, metadata={}),
            SearchResult(chunk_id="2", content="B", score=0.6, metadata={}),
        ]
        sparse_results = [
            SearchResult(chunk_id="1", content="A", score=0.9, metadata={}),
            SearchResult(chunk_id="3", content="C", score=0.7, metadata={}),
        ]
        
        fused = hybrid._fuse_results(dense_results, sparse_results)
        
        # 应该有3个唯一的chunk
        assert len(fused) == 3
        
        # chunk_1应该有最高分（因为两边都有高分）
        chunk_ids = [r.chunk_id for r in fused]
        assert chunk_ids[0] == "1"  # 第一个应该是融合分数最高的


class TestReranker:
    """重排序器测试类"""
    
    def test_reranker_not_available(self):
        """测试重排序器不可用时的行为"""
        reranker = Reranker(model_name="nonexistent/model")
        
        results = [
            SearchResult(chunk_id="1", content="测试", score=0.5, metadata={}),
        ]
        
        # 不可用时应返回原始结果
        reranked = reranker.rerank("查询", results, top_k=1)
        assert len(reranked) == 1
    
    def test_rerank_empty_results(self):
        """测试空结果重排序"""
        reranker = Reranker()
        reranked = reranker.rerank("查询", [], top_k=5)
        assert reranked == []


class TestRetrieverFactory:
    """检索器工厂测试类"""
    
    def test_create_hybrid_retriever(self):
        """测试创建混合检索器"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = np.zeros(768)
        
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = []
        
        retriever = RetrieverFactory.create_hybrid_retriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            chunks=None,
            alpha=0.6,
            use_reranker=False
        )
        
        assert isinstance(retriever, HybridRetriever)
        assert retriever.alpha == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
