"""
嵌入模块

负责文本向量化和向量数据库管理
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import numpy as np

from ..processing.document_processor import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """检索结果数据类"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    rerank_score: Optional[float] = None


class Embedder:
    """
    文本嵌入器
    
    使用预训练模型将文本转换为向量
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = None,
        batch_size: int = 32
    ):
        """
        初始化嵌入器
        
        Args:
            model_name: 模型名称或路径
            device: 计算设备 (cuda/cpu/mps)
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"正在加载嵌入模型: {self.model_name}")
            
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
                logger.info(f"成功加载模型: {self.model_name}")
            except Exception as e:
                fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                logger.warning(f"加载 {self.model_name} 失败，使用备选模型: {fallback_model}")
                self.model = SentenceTransformer(
                    fallback_model,
                    device=self.device
                )
                self.model_name = fallback_model
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"嵌入维度: {self.embedding_dim}")
            
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
    
    def embed_text(self, text: str) -> np.ndarray:
        """计算单个文本的嵌入向量"""
        if not text.strip():
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量计算文本嵌入"""
        if not texts:
            return np.array([])
        
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return np.zeros((len(texts), self.embedding_dim))
        
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(valid_texts) > 100
        )
        
        result = np.zeros((len(texts), self.embedding_dim))
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = embeddings[idx]
        
        return result
    
    def embed_query(self, query: str) -> np.ndarray:
        """计算查询文本的嵌入"""
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"
        
        return self.embed_text(query)


class VectorStore:
    """
    向量存储管理器
    
    使用ChromaDB存储和检索向量
    """
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "documents",
        embedder: Embedder = None
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedder = embedder
        
        self._init_chromadb()
    
    def _init_chromadb(self):
        """初始化ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB初始化完成，集合: {self.collection_name}")
            logger.info(f"当前文档数: {self.collection.count()}")
            
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """添加文本块到向量数据库"""
        if not chunks:
            return
        
        if self.embedder is None:
            raise ValueError("需要嵌入器来添加文本块")
        
        texts = [chunk.content for chunk in chunks]
        
        logger.info(f"正在计算 {len(texts)} 个文本块的嵌入...")
        embeddings = self.embedder.embed_batch(texts)
        
        ids = [chunk.id for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                **{k: str(v) for k, v in chunk.metadata.items()}
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"成功添加 {len(chunks)} 个文本块到向量数据库")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[SearchResult]:
        """搜索相似文本块"""
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                score = 1 - distance
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i]
                ))
        
        return search_results
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[SearchResult]:
        """通过文本搜索"""
        if self.embedder is None:
            raise ValueError("需要嵌入器来执行文本搜索")
        
        query_embedding = self.embedder.embed_query(query)
        return self.search(query_embedding, top_k, filter_dict)
    
    def delete_by_document(self, document_id: str) -> int:
        """删除指定文档的所有文本块"""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
        
        if not results["ids"]:
            return 0
        
        self.collection.delete(ids=results["ids"])
        
        logger.info(f"删除文档 {document_id} 的 {len(results['ids'])} 个文本块")
        return len(results["ids"])
    
    def get_all_documents(self) -> List[Dict]:
        """获取所有已索引的文档信息"""
        if self.collection.count() == 0:
            return []
        
        results = self.collection.get(include=["metadatas"])
        
        documents = {}
        for metadata in results["metadatas"]:
            doc_id = metadata.get("document_id", "unknown")
            if doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "filename": metadata.get("filename", "unknown"),
                    "format": metadata.get("format", ""),
                    "chunk_count": 0,
                    "created_at": metadata.get("document_created_at", "")
                }
            documents[doc_id]["chunk_count"] += 1
        
        return list(documents.values())
    
    def clear(self) -> None:
        """清空向量数据库"""
        try:
            # 删除集合
            self.client.delete_collection(self.collection_name)
            logger.info(f"已删除集合: {self.collection_name}")
        except Exception as e:
            logger.warning(f"删除集合时出错: {e}")
        
        # 重新创建空集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"已重新创建集合: {self.collection_name}")
        
        # 多次确保集合为空（三重保险）
        for attempt in range(3):
            count = self.collection.count()
            if count == 0:
                break
                
            # 如果还有数据，强制删除所有
            try:
                all_data = self.collection.get()
                all_ids = all_data.get("ids", [])
                if all_ids:
                    self.collection.delete(ids=all_ids)
                    logger.warning(f"第{attempt+1}次尝试：强制删除了 {len(all_ids)} 个残留文档")
            except Exception as e:
                logger.error(f"强制删除失败: {e}")
        
        final_count = self.collection.count()
        if final_count > 0:
            logger.error(f"警告：向量数据库清空后仍有 {final_count} 个文档残留！")
        else:
            logger.info("向量数据库已完全清空，验证通过")
    
    @property
    def count(self) -> int:
        """获取文本块总数"""
        return self.collection.count()
