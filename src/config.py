"""
配置管理模块

管理系统所有配置参数，支持环境变量覆盖
"""

import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class Config:
    """系统配置类"""
    
    # ==================== 路径配置 ====================
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = field(default=None)
    UPLOAD_DIR: Path = field(default=None)
    VECTOR_DB_PATH: Path = field(default=None)
    
    # ==================== 文档处理配置 ====================
    CHUNK_SIZE: int = 512  # 分块大小(字符数)
    CHUNK_OVERLAP: int = 64  # 分块重叠
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 最大文件大小(50MB)
    SUPPORTED_FORMATS: List[str] = field(
        default_factory=lambda: [".pdf", ".txt", ".docx", ".md"]
    )
    
    # ==================== 嵌入模型配置 ====================
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    # 备选轻量级模型（如果BGE-M3下载慢）
    EMBEDDING_MODEL_FALLBACK: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM_FALLBACK: int = 384
    
    # ==================== 向量数据库配置 ====================
    COLLECTION_NAME: str = "documents"
    
    # ==================== 检索配置 ====================
    TOP_K: int = 5  # 返回结果数量
    RERANK_TOP_K: int = 10  # 重排序前的候选数量
    HYBRID_ALPHA: float = 0.7  # 混合检索权重 (稠密检索权重)
    SIMILARITY_THRESHOLD: float = 0.3  # 相似度阈值
    
    # ==================== 重排序配置 ====================
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    USE_RERANKER: bool = True
    
    # ==================== 生成配置 ====================
    LLM_PROVIDER: str = "ollama"  # ollama 或 openai
    LLM_MODEL: str = "qwen2.5:7b"
    LLM_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    
    # ==================== 对话配置 ====================
    MAX_HISTORY_LENGTH: int = 10  # 最大对话历史轮数
    
    # ==================== 服务配置 ====================
    HOST: str = "0.0.0.0"
    PORT: int = 7860
    
    def __post_init__(self):
        """初始化后处理，设置默认路径"""
        if self.DATA_DIR is None:
            self.DATA_DIR = self.BASE_DIR / "data"
        if self.UPLOAD_DIR is None:
            self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        if self.VECTOR_DB_PATH is None:
            self.VECTOR_DB_PATH = self.DATA_DIR / "chroma_db"
        
        # 确保目录存在
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # 从环境变量加载配置
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        env_mappings = {
            "EMBEDDING_MODEL": str,
            "LLM_MODEL": str,
            "LLM_PROVIDER": str,
            "LLM_BASE_URL": str,
            "OPENAI_API_KEY": str,
            "OPENAI_MODEL": str,
            "TOP_K": int,
            "CHUNK_SIZE": int,
            "CHUNK_OVERLAP": int,
            "HYBRID_ALPHA": float,
            "TEMPERATURE": float,
            "PORT": int,
        }
        
        for key, converter in env_mappings.items():
            env_value = os.getenv(f"RAG_{key}")
            if env_value:
                try:
                    setattr(self, key, converter(env_value))
                except ValueError:
                    pass  # 忽略转换失败
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取全局配置"""
    return config


def update_config(**kwargs) -> Config:
    """更新全局配置"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
