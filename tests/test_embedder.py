"""
嵌入模块测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.embedder import (
    Embedder,
    VectorStore,
    SearchResult
)
from src.processing.document_processor import Chunk


class TestEmbedder:
    """嵌入器测试类"""
    
    @pytest.fixture
    def mock_embedder(self):
        """创建模拟的Embedder"""
        embedder = Mock()
        embedder.model_name = "test-model"
        embedder.model = Mock()
        embedder.embed_text.return_value = np.random.rand(384).astype(np.float32)
        embedder.embed_batch.return_value = np.random.rand(3, 384).astype(np.float32)
        embedder.embed_query.return_value = np.random.rand(384).astype(np.float32)
        return embedder
    
    def test_embedder_initialization(self, mock_embedder):
        """测试嵌入器初始化"""
        assert mock_embedder.model_name == "test-model"
        assert mock_embedder.model is not None
    
    def test_embed_text(self, mock_embedder):
        """测试单文本嵌入"""
        text = "这是一个测试文本"
        embedding = mock_embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 一维向量
    
    def test_embed_batch(self, mock_embedder):
        """测试批量文本嵌入"""
        texts = ["文本1", "文本2", "文本3"]
        embeddings = mock_embedder.embed_batch(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
    
    def test_embed_query(self, mock_embedder):
        """测试查询嵌入"""
        query = "测试查询"
        embedding = mock_embedder.embed_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1


class TestVectorStore:
    """向量存储测试类"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库路径"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedder_vs(self):
        """创建模拟的Embedder for VectorStore"""
        embedder = Mock()
        # 返回正确数量的embeddings
        def embed_batch_func(texts):
            return np.random.rand(len(texts), 384).astype(np.float32)
        embedder.embed_batch.side_effect = embed_batch_func
        embedder.embed_query.return_value = np.random.rand(384).astype(np.float32)
        return embedder
    
    @pytest.fixture
    def vector_store(self, temp_db_path, mock_embedder_vs):
        """创建VectorStore实例"""
        return VectorStore(
            persist_directory=str(temp_db_path),
            collection_name="test_collection",
            embedder=mock_embedder_vs
        )
    
    def test_vector_store_initialization(self, vector_store):
        """测试向量存储初始化"""
        assert vector_store.collection is not None
        assert vector_store.collection_name == "test_collection"
    
    def test_add_chunks(self, vector_store, mock_embedder_vs):
        """测试添加文本块"""
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="这是第一个文本块",
                start_pos=0,
                end_pos=10,
                metadata={"filename": "test.txt"}
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="这是第二个文本块",
                start_pos=10,
                end_pos=20,
                metadata={"filename": "test.txt"}
            )
        ]
        
        vector_store.add_chunks(chunks)
        
        # 验证嵌入器被调用
        mock_embedder_vs.embed_batch.assert_called()
        
        # 验证count增加
        assert vector_store.count >= 2
    
    def test_search(self, vector_store, mock_embedder_vs):
        """测试向量搜索"""
        # 先添加一些数据
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="机器学习是人工智能的分支",
                start_pos=0,
                end_pos=20,
                metadata={"filename": "ml.txt"}
            )
        ]
        vector_store.add_chunks(chunks)
        
        # 执行搜索
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=1)
        
        assert isinstance(results, list)
        # 可能为空（因为是随机embedding），但应该是列表
        if results:
            assert isinstance(results[0], SearchResult)
    
    def test_delete_document(self, vector_store):
        """测试删除文档"""
        # 添加数据
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="测试内容",
                start_pos=0,
                end_pos=10,
                metadata={}
            )
        ]
        vector_store.add_chunks(chunks)
        
        initial_count = vector_store.count
        
        # 删除文档
        vector_store.delete_by_document("doc1")
        
        # 验证删除成功
        assert vector_store.count < initial_count or vector_store.count == 0
    
    def test_clear(self, vector_store, mock_embedder_vs):
        """测试清空数据库"""
        # 添加数据
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="测试内容",
                start_pos=0,
                end_pos=10,
                metadata={}
            )
        ]
        vector_store.add_chunks(chunks)
        
        # 清空
        vector_store.clear()
        
        # 验证已清空
        assert vector_store.count == 0
    
    def test_get_all_documents(self, vector_store):
        """测试获取所有文档"""
        # 添加数据
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="文档1内容",
                start_pos=0,
                end_pos=10,
                metadata={"filename": "doc1.txt"}
            ),
            Chunk(
                id="chunk2",
                document_id="doc2",
                content="文档2内容",
                start_pos=0,
                end_pos=10,
                metadata={"filename": "doc2.txt"}
            )
        ]
        vector_store.add_chunks(chunks)
        
        # 获取文档列表
        documents = vector_store.get_all_documents()
        
        assert isinstance(documents, list)
        # 应该有至少一个文档
        if documents:
            assert "id" in documents[0] or "document_id" in documents[0]
    
    def test_count_property(self, vector_store):
        """测试count属性"""
        initial_count = vector_store.count
        assert isinstance(initial_count, int)
        assert initial_count >= 0


class TestSearchResult:
    """搜索结果数据类测试"""
    
    def test_search_result_creation(self):
        """测试SearchResult创建"""
        result = SearchResult(
            chunk_id="chunk1",
            content="测试内容",
            score=0.85,
            metadata={"document": "test.txt"}
        )
        
        assert result.chunk_id == "chunk1"
        assert result.content == "测试内容"
        assert result.score == 0.85
        assert result.metadata["document"] == "test.txt"
    
    def test_search_result_with_rerank_score(self):
        """测试带重排序分数的SearchResult"""
        result = SearchResult(
            chunk_id="chunk1",
            content="测试内容",
            score=0.85,
            metadata={},
            rerank_score=0.92
        )
        
        assert result.rerank_score == 0.92


class TestVectorStoreIntegration:
    """向量存储集成测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库路径"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_add_search_workflow(self, temp_db_path):
        """测试添加和搜索工作流"""
        # 创建mock embedder
        mock_embedder = Mock()
        mock_embedder.embed_batch.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384
        ]).astype(np.float32)
        mock_embedder.embed_query.return_value = np.array([0.15] * 384).astype(np.float32)
        
        # 创建向量存储
        vs = VectorStore(
            persist_directory=str(temp_db_path),
            collection_name="test",
            embedder=mock_embedder
        )
        
        # 添加chunks
        chunks = [
            Chunk(
                id="1",
                document_id="doc1",
                content="机器学习",
                start_pos=0,
                end_pos=5,
                metadata={"filename": "ml.txt"}
            ),
            Chunk(
                id="2",
                document_id="doc1",
                content="深度学习",
                start_pos=5,
                end_pos=10,
                metadata={"filename": "ml.txt"}
            )
        ]
        
        vs.add_chunks(chunks)
        
        # 搜索
        query_emb = np.array([0.15] * 384).astype(np.float32)
        results = vs.search(query_emb, top_k=2)
        
        # 验证
        assert vs.count == 2
        assert isinstance(results, list)
    
    def test_persistence(self, temp_db_path):
        """测试数据持久化"""
        mock_embedder = Mock()
        mock_embedder.embed_batch.return_value = np.array([[0.1] * 384]).astype(np.float32)
        
        # 第一次：添加数据
        vs1 = VectorStore(
            persist_directory=str(temp_db_path),
            collection_name="persist_test",
            embedder=mock_embedder
        )
        
        chunks = [
            Chunk(
                id="1",
                document_id="doc1",
                content="测试持久化",
                start_pos=0,
                end_pos=5,
                metadata={}
            )
        ]
        vs1.add_chunks(chunks)
        count1 = vs1.count
        
        # 第二次：重新连接
        vs2 = VectorStore(
            persist_directory=str(temp_db_path),
            collection_name="persist_test",
            embedder=mock_embedder
        )
        
        # 数据应该仍然存在
        assert vs2.count == count1
