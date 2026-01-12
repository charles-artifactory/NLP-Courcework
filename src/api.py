"""
API服务模块

提供RESTful API接口
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_config
from .rag_pipeline import get_pipeline, RAGPipeline

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class QueryRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(default="default", description="会话ID")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="检索数量")


class QueryResponse(BaseModel):
    """问答响应"""
    answer: str = Field(..., description="生成的答案")
    sources: List[dict] = Field(default_factory=list, description="来源引用")
    confidence: float = Field(..., description="置信度")


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    chunk_count: int
    created_at: Optional[str] = None
    format: Optional[str] = None


class UploadResponse(BaseModel):
    """上传响应"""
    status: str
    document_id: Optional[str] = None
    chunk_count: int = 0
    message: str


class StatusResponse(BaseModel):
    """状态响应"""
    status: str
    message: str


class StatsResponse(BaseModel):
    """统计响应"""
    document_count: int
    chunk_count: int
    vector_store_count: int
    embedding_model: str
    llm_provider: str
    llm_model: str


# ==================== 应用初始化 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("正在初始化RAG系统...")
    pipeline = get_pipeline()
    pipeline.initialize()
    logger.info("RAG系统初始化完成")
    
    yield
    
    # 关闭时清理
    logger.info("RAG系统关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="RAG智能问答系统 API",
        description="基于检索增强生成(RAG)技术的中英双语智能问答系统",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ==================== API端点 ====================

@app.get("/", tags=["健康检查"])
async def root():
    """根路径健康检查"""
    return {"status": "ok", "message": "RAG问答系统API正常运行"}


@app.get("/api/health", tags=["健康检查"])
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/api/stats", response_model=StatsResponse, tags=["系统信息"])
async def get_stats():
    """获取系统统计信息"""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    return StatsResponse(**stats)


# ==================== 文档管理 ====================

@app.post("/api/documents/upload", response_model=UploadResponse, tags=["文档管理"])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    上传并索引文档
    
    支持格式: PDF, TXT, DOCX, Markdown
    最大文件大小: 50MB
    """
    config = get_config()
    pipeline = get_pipeline()
    
    # 验证文件格式
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    
    if suffix not in config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}。支持的格式: {config.SUPPORTED_FORMATS}"
        )
    
    # 验证文件大小
    file.file.seek(0, 2)  # 移到文件末尾
    file_size = file.file.tell()
    file.file.seek(0)  # 移回开头
    
    if file_size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大。最大允许: {config.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # 保存文件到临时目录
    upload_path = config.UPLOAD_DIR / filename
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")
    
    # 索引文档
    try:
        result = pipeline.index_document(str(upload_path))
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)
        
        return UploadResponse(
            status="success",
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            message=result.message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"索引文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引失败: {e}")


@app.get("/api/documents", response_model=List[DocumentInfo], tags=["文档管理"])
async def list_documents():
    """获取已索引的文档列表"""
    pipeline = get_pipeline()
    documents = pipeline.get_documents()
    
    return [
        DocumentInfo(
            id=doc["id"],
            filename=doc.get("filename", "unknown"),
            chunk_count=doc.get("chunk_count", 0),
            created_at=doc.get("created_at"),
            format=doc.get("format")
        )
        for doc in documents
    ]


@app.delete("/api/documents/{doc_id}", response_model=StatusResponse, tags=["文档管理"])
async def delete_document(doc_id: str):
    """删除指定文档"""
    pipeline = get_pipeline()
    
    success = pipeline.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    return StatusResponse(status="success", message="文档已删除")


@app.delete("/api/documents", response_model=StatusResponse, tags=["文档管理"])
async def clear_all_documents():
    """清空所有文档"""
    pipeline = get_pipeline()
    pipeline.clear_all_data()
    
    return StatusResponse(status="success", message="已清空所有文档")


# ==================== 问答接口 ====================

@app.post("/api/qa/query", response_model=QueryResponse, tags=["问答"])
async def query(request: QueryRequest):
    """
    提交问题并获取答案
    
    支持多轮对话（通过session_id关联）
    返回答案及来源引用
    """
    pipeline = get_pipeline()
    
    try:
        result = pipeline.query(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence
        )
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答处理失败: {e}")


@app.post("/api/conversation/clear", response_model=StatusResponse, tags=["对话管理"])
async def clear_conversation(session_id: str = "default"):
    """清空对话历史"""
    pipeline = get_pipeline()
    pipeline.clear_conversation(session_id)
    
    return StatusResponse(status="success", message="对话历史已清空")


# ==================== 启动函数 ====================

def run_api(host: str = None, port: int = None):
    """运行API服务"""
    import uvicorn
    
    config = get_config()
    host = host or config.HOST
    port = port or config.PORT
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api()
