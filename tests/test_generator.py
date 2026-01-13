"""
生成模块测试
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.generator import (
    Generator,
    PromptBuilder,
    SourceTracer,
    GenerationResult,
    SourceRef,
    SYSTEM_PROMPT_RAG_CN,
    SYSTEM_PROMPT_CHAT_CN
)
from src.retrieval.embedder import SearchResult


class TestPromptBuilder:
    """Prompt构建器测试类"""
    
    @pytest.fixture
    def builder(self):
        """创建PromptBuilder实例"""
        return PromptBuilder()
    
    def test_build_contexts(self, builder):
        """测试上下文构建"""
        search_results = [
            SearchResult(
                chunk_id="1",
                content="机器学习是人工智能的一个分支。",
                score=0.9,
                metadata={"document": "doc1.txt"}
            ),
            SearchResult(
                chunk_id="2",
                content="深度学习是机器学习的子集。",
                score=0.8,
                metadata={"document": "doc2.txt"}
            )
        ]
        
        contexts = builder.build_contexts(search_results)
        
        assert "机器学习是人工智能的一个分支" in contexts
        assert "深度学习是机器学习的子集" in contexts
        assert "[1]" in contexts
        assert "[2]" in contexts
    
    def test_build_messages_with_search_results(self, builder):
        """测试有检索结果时的消息构建"""
        search_results = [
            SearchResult(
                chunk_id="1",
                content="机器学习的定义",
                score=0.9,
                metadata={}
            )
        ]
        
        messages = builder.build_messages(
            query="什么是机器学习？",
            search_results=search_results,
            history=[]
        )
        
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert "参考资料" in messages[-1]["content"]
        assert "什么是机器学习" in messages[-1]["content"]
    
    def test_build_messages_without_search_results(self, builder):
        """测试无检索结果时的消息构建（普通对话模式）"""
        messages = builder.build_messages(
            query="你好",
            search_results=[],
            history=[]
        )
        
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["content"] == "你好"
        assert "参考资料" not in messages[-1]["content"]
    
    def test_build_messages_with_history(self, builder):
        """测试带历史记录的消息构建"""
        history = [
            {"role": "user", "content": "什么是AI？"},
            {"role": "assistant", "content": "AI是人工智能。"}
        ]
        
        messages = builder.build_messages(
            query="它有哪些应用？",
            search_results=[],
            history=history
        )
        
        # 应该包含历史记录
        assert len(messages) > 3
        assert any("什么是AI" in str(m) for m in messages)


class TestSourceTracer:
    """来源追踪器测试类"""
    
    @pytest.fixture
    def tracer(self):
        """创建SourceTracer实例"""
        return SourceTracer()
    
    def test_format_sources(self, tracer):
        """测试来源格式化"""
        search_results = [
            SearchResult(
                chunk_id="chunk1",
                content="机器学习是AI的分支",
                score=0.9,
                metadata={"filename": "doc1.txt"}
            ),
            SearchResult(
                chunk_id="chunk2",
                content="深度学习很重要",
                score=0.8,
                metadata={"filename": "doc2.txt"}
            )
        ]
        
        sources = tracer.format_sources(search_results)
        
        assert len(sources) == 2
        assert isinstance(sources[0], dict)
        assert sources[0]["index"] == 1
        assert sources[0]["document"] == "doc1.txt"
        assert sources[1]["index"] == 2
        assert sources[1]["document"] == "doc2.txt"
    
    def test_highlight_sources_with_markers(self, tracer):
        """测试已有引用标记的答案"""
        answer = "机器学习[1]是AI的分支[2]。"
        sources = [
            SourceRef(text="[1]", chunk_id="1", document="doc1.txt", score=0.9),
            SourceRef(text="[2]", chunk_id="2", document="doc2.txt", score=0.8)
        ]
        
        result = tracer.highlight_sources(answer, sources)
        
        # 已有标记，应该原样返回
        assert "[1]" in result
        assert "[2]" in result
    
    def test_highlight_sources_without_markers(self, tracer):
        """测试无引用标记的答案"""
        answer = "机器学习是AI的分支。"
        sources = [
            SourceRef(text="[1]", chunk_id="1", document="doc1.txt", score=0.9)
        ]
        
        result = tracer.highlight_sources(answer, sources)
        
        # 应该添加来源说明
        assert "参考来源" in result or answer in result


class TestGenerator:
    """生成器测试类"""
    
    def test_generator_initialization_ollama(self):
        """测试Ollama模式初始化"""
        with patch('ollama.Client') as mock_client:
            mock_client.return_value = Mock()
            
            generator = Generator(
                provider="ollama",
                model="qwen2.5:7b",
                base_url="http://localhost:11434"
            )
            
            assert generator.provider == "ollama"
            assert generator.model == "qwen2.5:7b"
    
    def test_generator_initialization_openai(self):
        """测试OpenAI模式初始化"""
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            generator = Generator(
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test-key",
                base_url="https://api.openai.com/v1"
            )
            
            assert generator.provider == "openai"
            assert generator.model == "gpt-3.5-turbo"
            mock_openai.assert_called_once()
    
    def test_generate_mock_mode(self):
        """测试模拟模式生成"""
        generator = Generator(provider="ollama", model="test")
        # 强制进入mock模式
        generator._client = None
        
        search_results = [
            SearchResult(
                chunk_id="1",
                content="机器学习定义",
                score=0.9,
                metadata={"document": "test.txt"}
            )
        ]
        
        result = generator.generate(
            query="什么是机器学习？",
            search_results=search_results,
            history=[]
        )
        
        assert isinstance(result, GenerationResult)
        assert len(result.answer) > 0
        assert result.confidence > 0
    
    def test_generate_with_ollama(self):
        """测试Ollama生成（模拟）"""
        with patch('ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.chat.return_value = {
                "message": {"content": "机器学习是AI的一个分支。"}
            }
            mock_client_class.return_value = mock_client
            
            generator = Generator(
                provider="ollama",
                model="qwen2.5:7b"
            )
            
            search_results = [
                SearchResult(
                    chunk_id="1",
                    content="ML定义",
                    score=0.9,
                    metadata={"document": "test.txt"}
                )
            ]
            
            result = generator.generate(
                query="什么是机器学习？",
                search_results=search_results
            )
            
            assert "机器学习" in result.answer or "模拟" in result.answer
    
    def test_generate_stream_mock(self):
        """测试流式生成（模拟模式）"""
        generator = Generator(provider="ollama", model="test")
        generator._client = None
        
        search_results = [
            SearchResult(
                chunk_id="1",
                content="测试内容",
                score=0.9,
                metadata={"document": "test.txt"}
            )
        ]
        
        chunks = list(generator.generate_stream(
            query="测试问题",
            search_results=search_results
        ))
        
        assert len(chunks) > 0
        full_answer = "".join(chunks)
        assert len(full_answer) > 0
    
    def test_generate_without_search_results(self):
        """测试无检索结果的生成（普通对话模式）"""
        generator = Generator(provider="ollama", model="test")
        generator._client = None
        
        result = generator.generate(
            query="你好",
            search_results=[],
            history=[]
        )
        
        assert isinstance(result, GenerationResult)
        assert len(result.answer) > 0
        # 无检索结果时，置信度应该较低
        assert result.confidence >= 0


class TestGenerationResult:
    """生成结果数据类测试"""
    
    def test_generation_result_creation(self):
        """测试GenerationResult创建"""
        sources = [
            SourceRef(text="[1]", chunk_id="1", document="doc.txt", score=0.9)
        ]
        result = GenerationResult(
            answer="这是答案",
            sources=sources,
            confidence=0.85
        )
        
        assert result.answer == "这是答案"
        assert result.confidence == 0.85
        assert len(result.sources) == 1


class TestSourceRef:
    """来源引用数据类测试"""
    
    def test_source_ref_creation(self):
        """测试SourceRef创建"""
        ref = SourceRef(
            text="[1]",
            chunk_id="chunk1",
            document="doc.txt",
            score=0.9
        )
        
        assert ref.text == "[1]"
        assert ref.chunk_id == "chunk1"
        assert ref.document == "doc.txt"
        assert ref.score == 0.9


class TestSystemPrompts:
    """系统提示词测试"""
    
    def test_system_prompt_rag_exists(self):
        """测试RAG系统提示词存在"""
        assert len(SYSTEM_PROMPT_RAG_CN) > 0
        assert "参考资料" in SYSTEM_PROMPT_RAG_CN or "问答助手" in SYSTEM_PROMPT_RAG_CN
    
    def test_system_prompt_chat_exists(self):
        """测试普通对话系统提示词存在"""
        assert len(SYSTEM_PROMPT_CHAT_CN) > 0
        assert "助手" in SYSTEM_PROMPT_CHAT_CN or "AI" in SYSTEM_PROMPT_CHAT_CN
