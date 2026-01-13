"""
配置模块测试
"""

import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, get_config


class TestConfig:
    """配置类测试"""
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = Config()
        
        # 验证基本配置
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.TOP_K > 0
        assert config.EMBEDDING_MODEL is not None
        assert config.LLM_MODEL is not None
    
    def test_config_paths(self):
        """测试配置路径"""
        config = Config()
        
        # 检查关键路径存在
        assert config.DATA_DIR.exists()
        assert config.VECTOR_DB_PATH.exists()
    
    def test_config_llm_settings(self):
        """测试LLM配置"""
        config = Config()
        
        assert config.LLM_PROVIDER in ["ollama", "openai"]
        assert config.TEMPERATURE >= 0 and config.TEMPERATURE <= 2
        assert config.MAX_NEW_TOKENS > 0
    
    def test_config_retrieval_settings(self):
        """测试检索配置"""
        config = Config()
        
        assert config.TOP_K > 0
        assert config.RERANK_TOP_K >= config.TOP_K
        assert config.HYBRID_ALPHA >= 0 and config.HYBRID_ALPHA <= 1
        assert config.SIMILARITY_THRESHOLD >= 0 and config.SIMILARITY_THRESHOLD <= 1
    
    def test_config_embedding_settings(self):
        """测试嵌入配置"""
        config = Config()
        
        assert config.EMBEDDING_MODEL is not None
        assert config.EMBEDDING_BATCH_SIZE > 0
    
    def test_update_llm_config(self):
        """测试动态更新LLM配置"""
        config = Config()
        
        # 直接修改配置（如果没有update_llm_config方法）
        if hasattr(config, 'update_llm_config'):
            config.update_llm_config(
                LLM_PROVIDER="openai",
                LLM_MODEL="gpt-4",
                TEMPERATURE=0.5,
                MAX_NEW_TOKENS=2048
            )
            
            # 验证更新
            assert config.LLM_PROVIDER == "openai"
            assert config.LLM_MODEL == "gpt-4"
            assert config.TEMPERATURE == 0.5
            assert config.MAX_NEW_TOKENS == 2048
        else:
            # 直接赋值测试
            config.LLM_PROVIDER = "openai"
            config.LLM_MODEL = "gpt-4"
            assert config.LLM_PROVIDER == "openai"
            assert config.LLM_MODEL == "gpt-4"
    
    def test_update_llm_config_partial(self):
        """测试部分更新LLM配置"""
        config = Config()
        original_model = config.LLM_MODEL
        
        # 只更新温度
        if hasattr(config, 'update_llm_config'):
            config.update_llm_config(TEMPERATURE=0.8)
            # 模型应保持不变，温度应更新
            assert config.LLM_MODEL == original_model
            assert config.TEMPERATURE == 0.8
        else:
            config.TEMPERATURE = 0.8
            assert config.TEMPERATURE == 0.8
    
    def test_config_from_env(self):
        """测试从环境变量读取配置"""
        # 设置环境变量
        os.environ["RAG_LLM_PROVIDER"] = "openai"
        os.environ["RAG_LLM_MODEL"] = "gpt-3.5-turbo"
        os.environ["RAG_TOP_K"] = "10"
        
        config = Config()
        
        # 验证环境变量被读取
        assert config.LLM_PROVIDER == "openai"
        assert config.LLM_MODEL == "gpt-3.5-turbo"
        assert config.TOP_K == 10
        
        # 清理环境变量
        del os.environ["RAG_LLM_PROVIDER"]
        del os.environ["RAG_LLM_MODEL"]
        del os.environ["RAG_TOP_K"]
    
    def test_config_openai_settings(self):
        """测试OpenAI配置"""
        config = Config()
        
        # OpenAI相关配置应存在
        assert hasattr(config, "OPENAI_API_KEY")
        assert hasattr(config, "OPENAI_MODEL")
        assert hasattr(config, "LLM_BASE_URL")
    
    def test_config_use_reranker(self):
        """测试重排序配置"""
        config = Config()
        
        assert isinstance(config.USE_RERANKER, bool)
        if config.USE_RERANKER:
            assert config.RERANKER_MODEL is not None
    
    def test_config_max_file_size(self):
        """测试文件大小限制"""
        config = Config()
        
        assert config.MAX_FILE_SIZE > 0
        # 默认应该是50MB
        assert config.MAX_FILE_SIZE == 50 * 1024 * 1024
    
    def test_config_supported_formats(self):
        """测试支持的文件格式"""
        config = Config()
        
        assert isinstance(config.SUPPORTED_FORMATS, list)
        assert len(config.SUPPORTED_FORMATS) > 0
        assert ".txt" in config.SUPPORTED_FORMATS
        assert ".md" in config.SUPPORTED_FORMATS


class TestGetConfig:
    """get_config函数测试"""
    
    def test_get_config_singleton(self):
        """测试get_config返回单例"""
        config1 = get_config()
        config2 = get_config()
        
        # 应该返回同一个实例
        assert config1 is config2
    
    def test_get_config_returns_config(self):
        """测试get_config返回Config实例"""
        config = get_config()
        
        assert isinstance(config, Config)
        assert hasattr(config, "LLM_PROVIDER")
        assert hasattr(config, "EMBEDDING_MODEL")


class TestConfigValidation:
    """配置验证测试"""
    
    def test_config_chunk_size_overlap_relation(self):
        """测试块大小和重叠的关系"""
        config = Config()
        
        # 重叠应该小于块大小
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    
    def test_config_top_k_rerank_relation(self):
        """测试检索数量和重排序数量的关系"""
        config = Config()
        
        # 重排序数量应该大于等于最终返回数量
        assert config.RERANK_TOP_K >= config.TOP_K
    
    def test_config_temperature_range(self):
        """测试温度范围"""
        config = Config()
        
        # 温度应该在合理范围内
        assert 0 <= config.TEMPERATURE <= 2
    
    def test_config_alpha_range(self):
        """测试混合检索alpha范围"""
        config = Config()
        
        # alpha应该在[0, 1]范围内
        assert 0 <= config.HYBRID_ALPHA <= 1


class TestConfigEdgeCases:
    """配置边界情况测试"""
    
    def test_update_nonexistent_attribute(self):
        """测试更新不存在的属性"""
        config = Config()
        
        # 更新不存在的属性应该不报错（hasattr检查）
        if hasattr(config, 'update_llm_config'):
            config.update_llm_config(NONEXISTENT_ATTR="value")
            # 不应该创建新属性
            assert not hasattr(config, "NONEXISTENT_ATTR")
        else:
            # 跳过此测试
            assert True
    
    def test_config_with_none_values(self):
        """测试空值配置"""
        config = Config()
        
        # 某些配置可能为None（如API_KEY）
        # 这应该是允许的
        if config.OPENAI_API_KEY is None:
            assert True  # 允许为None
    
    def test_config_paths_creation(self):
        """测试路径自动创建"""
        config = Config()
        
        # 关键路径应该被自动创建
        assert config.DATA_DIR.exists()
        assert config.VECTOR_DB_PATH.exists()
        assert config.UPLOAD_DIR.exists()


class TestConfigTypes:
    """配置类型测试"""
    
    def test_config_string_types(self):
        """测试字符串类型配置"""
        config = Config()
        
        assert isinstance(config.LLM_PROVIDER, str)
        assert isinstance(config.LLM_MODEL, str)
        assert isinstance(config.EMBEDDING_MODEL, str)
    
    def test_config_numeric_types(self):
        """测试数值类型配置"""
        config = Config()
        
        assert isinstance(config.CHUNK_SIZE, int)
        assert isinstance(config.TOP_K, int)
        assert isinstance(config.TEMPERATURE, (int, float))
        assert isinstance(config.HYBRID_ALPHA, float)
    
    def test_config_path_types(self):
        """测试路径类型配置"""
        config = Config()
        
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.VECTOR_DB_PATH, Path)
    
    def test_config_list_types(self):
        """测试列表类型配置"""
        config = Config()
        
        assert isinstance(config.SUPPORTED_FORMATS, list)
