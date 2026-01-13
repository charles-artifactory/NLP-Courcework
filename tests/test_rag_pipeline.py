"""
RAG流水线集成测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.core.pipeline import (
    RAGPipeline,
    ConversationManager,
    IndexResult,
    QueryResult,
    get_pipeline,
    reset_pipeline
)


class TestConversationManager:
    """对话管理器测试类"""
    
    @pytest.fixture
    def manager(self):
        """创建对话管理器"""
        return ConversationManager(max_history=5)
    
    def test_add_message(self, manager):
        """测试添加消息"""
        manager.add_message("session_1", "user", "你好")
        manager.add_message("session_1", "assistant", "你好！有什么可以帮助你的？")
        
        history = manager.get_history("session_1")
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_max_history_limit(self, manager):
        """测试历史长度限制"""
        # 添加超过限制的消息
        for i in range(20):
            manager.add_message("session_1", "user", f"消息{i}")
        
        history = manager.get_history("session_1")
        
        # 应该只保留 max_history * 2 = 10 条
        assert len(history) <= manager.max_history * 2
    
    def test_multiple_sessions(self, manager):
        """测试多会话隔离"""
        manager.add_message("session_1", "user", "会话1的消息")
        manager.add_message("session_2", "user", "会话2的消息")
        
        history_1 = manager.get_history("session_1")
        history_2 = manager.get_history("session_2")
        
        assert len(history_1) == 1
        assert len(history_2) == 1
        assert "会话1" in history_1[0]["content"]
        assert "会话2" in history_2[0]["content"]
    
    def test_clear_session(self, manager):
        """测试清空会话"""
        manager.add_message("session_1", "user", "消息")
        manager.clear("session_1")
        
        history = manager.get_history("session_1")
        
        assert len(history) == 0
    
    def test_clear_all(self, manager):
        """测试清空所有会话"""
        manager.add_message("session_1", "user", "消息1")
        manager.add_message("session_2", "user", "消息2")
        manager.clear_all()
        
        assert len(manager.get_history("session_1")) == 0
        assert len(manager.get_history("session_2")) == 0
    
    def test_format_history(self, manager):
        """测试格式化历史"""
        manager.add_message("session_1", "user", "什么是机器学习？")
        manager.add_message("session_1", "assistant", "机器学习是...")
        
        formatted = manager.format_history("session_1")
        
        assert "用户:" in formatted or "用户：" in formatted
        assert "什么是机器学习" in formatted


class TestRAGPipelineUnit:
    """RAG流水线单元测试"""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """创建测试配置"""
        config = Config()
        config.DATA_DIR = tmp_path / "data"
        config.UPLOAD_DIR = tmp_path / "uploads"
        config.VECTOR_DB_PATH = tmp_path / "chroma_db"
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        config.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        return config
    
    def test_pipeline_initialization(self, mock_config):
        """测试流水线初始化"""
        # 使用模拟来避免加载实际模型
        with patch('src.core.pipeline.Embedder') as mock_embedder, \
             patch('src.core.pipeline.VectorStore') as mock_vector_store, \
             patch('src.core.pipeline.Generator') as mock_generator:
            
            mock_embedder.return_value.model_name = "test-model"
            mock_vector_store.return_value.count = 0
            mock_vector_store.return_value.get_all_documents.return_value = []
            mock_vector_store.return_value.collection.get.return_value = {"documents": [], "metadatas": [], "ids": []}
            
            pipeline = RAGPipeline(config=mock_config)
            pipeline.initialize()
            
            assert pipeline._initialized
    
    def test_get_stats(self, mock_config):
        """测试获取统计信息"""
        with patch('src.core.pipeline.Embedder') as mock_embedder, \
             patch('src.core.pipeline.VectorStore') as mock_vector_store, \
             patch('src.core.pipeline.Generator') as mock_generator:
            
            mock_embedder.return_value.model_name = "test-model"
            mock_vector_store.return_value.count = 10
            mock_vector_store.return_value.get_all_documents.return_value = [
                {"id": "1", "chunk_count": 5},
                {"id": "2", "chunk_count": 5}
            ]
            mock_vector_store.return_value.collection.get.return_value = {"documents": [], "metadatas": [], "ids": []}
            
            pipeline = RAGPipeline(config=mock_config)
            pipeline.initialize()
            
            stats = pipeline.get_stats()
            
            assert "document_count" in stats
            assert "chunk_count" in stats
            assert stats["document_count"] == 2


class TestRAGPipelineIntegration:
    """RAG流水线集成测试（需要完整环境）"""
    
    @pytest.fixture
    def sample_document(self, tmp_path):
        """创建示例文档"""
        content = """自然语言处理基础知识

自然语言处理(NLP)是人工智能的一个重要分支。它研究如何让计算机理解、解释和生成人类语言。

主要技术包括：
1. 文本分类 - 将文本归类到预定义的类别
2. 命名实体识别 - 识别文本中的实体如人名、地名等
3. 情感分析 - 判断文本表达的情感倾向
4. 机器翻译 - 将一种语言翻译成另一种语言

检索增强生成(RAG)是一种新兴技术，结合了信息检索和文本生成。
它首先检索相关文档，然后基于检索结果生成答案。
这种方法可以提高问答系统的准确性和可靠性。
"""
        doc_path = tmp_path / "nlp_basics.txt"
        doc_path.write_text(content, encoding="utf-8")
        return str(doc_path)
    
    @pytest.mark.skip(reason="需要完整环境和模型")
    def test_full_pipeline_flow(self, sample_document, tmp_path):
        """测试完整流程（需要模型）"""
        # 创建配置
        config = Config()
        config.DATA_DIR = tmp_path / "data"
        config.UPLOAD_DIR = tmp_path / "uploads"
        config.VECTOR_DB_PATH = tmp_path / "chroma_db"
        
        # 创建流水线
        pipeline = RAGPipeline(config=config)
        pipeline.initialize()
        
        # 索引文档
        result = pipeline.index_document(sample_document)
        assert result.success
        assert result.chunk_count > 0
        
        # 查询
        query_result = pipeline.query("什么是自然语言处理？")
        assert len(query_result.answer) > 0
        
        # 获取文档列表
        docs = pipeline.get_documents()
        assert len(docs) > 0
        
        # 删除文档
        success = pipeline.delete_document(result.document_id)
        assert success


class TestPipelineSingleton:
    """流水线单例测试"""
    
    def test_get_pipeline_singleton(self):
        """测试获取单例"""
        reset_pipeline()
        
        pipeline1 = get_pipeline()
        pipeline2 = get_pipeline()
        
        assert pipeline1 is pipeline2
    
    def test_reset_pipeline(self):
        """测试重置单例"""
        pipeline1 = get_pipeline()
        reset_pipeline()
        pipeline2 = get_pipeline()
        
        assert pipeline1 is not pipeline2


class TestIndexResult:
    """索引结果测试"""
    
    def test_success_result(self):
        """测试成功结果"""
        result = IndexResult(
            success=True,
            document_id="doc_123",
            chunk_count=10,
            message="成功"
        )
        
        assert result.success
        assert result.document_id == "doc_123"
        assert result.chunk_count == 10
    
    def test_failure_result(self):
        """测试失败结果"""
        result = IndexResult(
            success=False,
            document_id=None,
            chunk_count=0,
            message="文件解析失败"
        )
        
        assert not result.success
        assert result.document_id is None


class TestQueryResult:
    """查询结果测试"""
    
    def test_query_result(self):
        """测试查询结果"""
        result = QueryResult(
            answer="这是答案",
            sources=[{"doc": "test.txt", "content": "相关内容"}],
            confidence=0.85
        )
        
        assert result.answer == "这是答案"
        assert len(result.sources) == 1
        assert result.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
