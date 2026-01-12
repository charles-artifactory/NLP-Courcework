"""
Gradio前端应用

提供友好的Web交互界面
"""

from src.rag_pipeline import get_pipeline, RAGPipeline
from src.config import get_config
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Generator, Optional

import gradio as gr

# 添加父目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


# ==================== 事件处理函数 ====================

def handle_upload(files: List) -> Tuple[str, List[List]]:
    """
    处理文件上传

    Args:
        files: 上传的文件列表

    Returns:
        Tuple[str, List[List]]: (状态消息, 文档列表)
    """
    if not files:
        return "请选择要上传的文件", get_document_list()

    pipeline = get_pipeline()
    pipeline.initialize()

    results = []
    for file in files:
        try:
            # file 可能是 tempfile 路径
            file_path = file.name if hasattr(file, 'name') else str(file)
            result = pipeline.index_document(file_path)

            if result.success:
                results.append(f"✅ {Path(file_path).name}: {result.chunk_count}个块")
            else:
                results.append(f"❌ {Path(file_path).name}: {result.message}")
        except Exception as e:
            results.append(f"❌ 处理失败: {str(e)}")

    status = "\n".join(results)
    doc_list = get_document_list()

    return status, doc_list


def get_document_list() -> List[List]:
    """
    获取文档列表

    Returns:
        List[List]: 文档列表数据
    """
    try:
        pipeline = get_pipeline()
        pipeline.initialize()
        documents = pipeline.get_documents()

        return [
            [doc.get("filename", "未知"), doc.get("chunk_count", 0), doc.get("id", "")]
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return []


def handle_query(
    question: str,
    history: List[dict],
    top_k: int
) -> Tuple[List[dict], dict, str]:
    """
    处理用户问题

    Args:
        question: 用户问题
        history: 对话历史
        top_k: 检索数量

    Returns:
        Tuple[List[dict], dict, str]: (更新后的历史, 来源信息, 清空的输入框)
    """
    if not question.strip():
        return history, {}, ""

    pipeline = get_pipeline()
    pipeline.initialize()

    try:
        result = pipeline.query(
            question=question,
            session_id="gradio_session",
            top_k=top_k
        )

        # 更新历史 - 使用Gradio 6.x的新消息格式
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})

        # 格式化来源
        sources_display = {
            "answer_confidence": f"{result.confidence:.2%}",
            "sources": result.sources
        }

        return history, sources_display, ""

    except Exception as e:
        logger.error(f"问答失败: {e}")
        error_msg = f"处理失败: {str(e)}"
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return history, {"error": str(e)}, ""


def handle_query_stream(
    question: str,
    history: List[dict],
    top_k: int
) -> Generator[Tuple[List[dict], dict, str], None, None]:
    """
    流式处理用户问题

    Args:
        question: 用户问题
        history: 对话历史
        top_k: 检索数量

    Yields:
        Tuple[List[dict], dict, str]: (更新后的历史, 来源信息, 清空的输入框)
    """
    if not question.strip():
        yield history, {}, ""
        return

    pipeline = get_pipeline()
    pipeline.initialize()

    # 使用Gradio 6.x的新消息格式
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})

    try:
        full_answer = ""
        sources = []

        for chunk, src in pipeline.query_stream(
            question=question,
            session_id="gradio_session",
            top_k=top_k
        ):
            full_answer += chunk
            sources = src
            history[-1]["content"] = full_answer

            sources_display = {
                "sources": sources
            }

            yield history, sources_display, ""

    except Exception as e:
        logger.error(f"流式问答失败: {e}")
        history[-1]["content"] = f"处理失败: {str(e)}"
        yield history, {"error": str(e)}, ""


def handle_clear() -> Tuple[List[dict], dict]:
    """
    清空对话

    Returns:
        Tuple[List[dict], dict]: (空历史, 空来源)
    """
    pipeline = get_pipeline()
    pipeline.clear_conversation("gradio_session")
    return [], {}


def handle_delete_doc(doc_id: str) -> Tuple[str, List[List]]:
    """
    删除文档

    Args:
        doc_id: 文档ID

    Returns:
        Tuple[str, List[List]]: (状态消息, 更新后的文档列表)
    """
    if not doc_id:
        return "请选择要删除的文档", get_document_list()

    pipeline = get_pipeline()
    success = pipeline.delete_document(doc_id)

    if success:
        return f"✅ 文档已删除", get_document_list()
    else:
        return f"❌ 删除失败", get_document_list()


def handle_clear_all() -> Tuple[str, List[List]]:
    """
    清空所有文档

    Returns:
        Tuple[str, List[List]]: (状态消息, 空文档列表)
    """
    pipeline = get_pipeline()
    pipeline.clear_all_data()
    return "✅ 已清空所有文档", []


# ==================== 创建应用 ====================

def create_app() -> gr.Blocks:
    """
    创建Gradio应用

    Returns:
        gr.Blocks: Gradio应用实例
    """
    with gr.Blocks(
        title="RAG智能问答系统"
    ) as app:

        # 标题
        gr.Markdown("""
        # 🤖 RAG增强智能问答系统
        
        基于检索增强生成(RAG)技术的中英双语智能问答系统。上传文档，然后基于文档内容进行问答。
        
        **特色功能**: 混合检索 | 智能分块 | 答案溯源 | 结果重排序 | 多轮对话
        """)

        with gr.Row():
            # ==================== 左侧面板 ====================
            with gr.Column(scale=1):
                gr.Markdown("### 📁 文档管理")

                # 文件上传
                file_upload = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".txt", ".docx", ".md"],
                    file_count="multiple"
                )

                upload_btn = gr.Button("📤 上传并索引", variant="primary")
                upload_status = gr.Textbox(
                    label="上传状态",
                    interactive=False,
                    lines=3
                )

                gr.Markdown("### 📋 已索引文档")

                doc_table = gr.Dataframe(
                    headers=["文档名", "块数", "ID"],
                    datatype=["str", "number", "str"],
                    label="文档列表",
                    interactive=False,
                    value=get_document_list
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新", size="sm")
                    clear_all_btn = gr.Button("🗑️ 清空全部", size="sm", variant="stop")

                # 删除功能
                with gr.Row():
                    delete_id = gr.Textbox(
                        label="要删除的文档ID",
                        placeholder="从上表复制ID",
                        scale=2
                    )
                    delete_btn = gr.Button("删除", size="sm", scale=1)

                gr.Markdown("### ⚙️ 设置")

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="检索数量 (Top-K)"
                )

                stream_mode = gr.Checkbox(
                    label="流式输出",
                    value=True
                )

            # ==================== 右侧面板 ====================
            with gr.Column(scale=2):
                gr.Markdown("### 💬 问答对话")

                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=450,
                    show_label=False
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="输入问题",
                        placeholder="请输入您的问题...",
                        lines=2,
                        scale=5
                    )
                    send_btn = gr.Button("🚀 发送", variant="primary", scale=1)

                clear_chat_btn = gr.Button("🗑️ 清空对话", size="sm")

                gr.Markdown("### 📚 来源引用")

                sources_json = gr.JSON(
                    label="答案来源"
                )

        # ==================== 页脚 ====================
        gr.Markdown("""
        ---
        
        <div style="text-align: center; color: #666;">
        
        **技术栈**: BGE-M3 嵌入 | ChromaDB 向量库 | Qwen2.5/Ollama LLM | FastAPI | Gradio
        
        </div>
        """)

        # ==================== 事件绑定 ====================

        # 上传事件
        upload_btn.click(
            fn=handle_upload,
            inputs=[file_upload],
            outputs=[upload_status, doc_table]
        )

        # 刷新文档列表
        refresh_btn.click(
            fn=lambda: get_document_list(),
            inputs=[],
            outputs=[doc_table]
        )

        # 删除文档
        delete_btn.click(
            fn=handle_delete_doc,
            inputs=[delete_id],
            outputs=[upload_status, doc_table]
        )

        # 清空所有文档
        clear_all_btn.click(
            fn=handle_clear_all,
            inputs=[],
            outputs=[upload_status, doc_table]
        )

        # 发送问题 - 根据流式模式选择处理函数
        def get_query_handler(stream: bool):
            return handle_query_stream if stream else handle_query

        send_btn.click(
            fn=handle_query_stream,
            inputs=[question_input, chatbot, top_k_slider],
            outputs=[chatbot, sources_json, question_input]
        )

        # 回车发送
        question_input.submit(
            fn=handle_query_stream,
            inputs=[question_input, chatbot, top_k_slider],
            outputs=[chatbot, sources_json, question_input]
        )

        # 清空对话
        clear_chat_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[chatbot, sources_json]
        )

    return app


def launch_app(
    host: str = None,
    port: int = None,
    share: bool = False
) -> None:
    """
    启动Gradio应用

    Args:
        host: 主机地址
        port: 端口号
        share: 是否创建公共链接
    """
    config = get_config()
    host = host or config.HOST
    port = port or config.PORT

    # 预初始化
    logger.info("正在初始化RAG系统...")
    pipeline = get_pipeline()
    pipeline.initialize()
    logger.info("RAG系统初始化完成")

    # 创建并启动应用
    app = create_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True
    )


# ==================== 主入口 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG智能问答系统")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="创建公共链接")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    launch_app(host=args.host, port=args.port, share=args.share)
