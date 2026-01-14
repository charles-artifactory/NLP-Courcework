"""
Gradio前端应用

提供友好的Web交互界面
"""

from src.core.pipeline import get_pipeline, RAGPipeline
from src.config import get_config
import logging
from pathlib import Path
from typing import List, Tuple, Generator, Dict

import gradio as gr

# 添加父目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


# ==================== 事件处理函数 ====================

def handle_load_example(progress=gr.Progress()) -> Tuple[str, List[List], str]:
    """
    加载示例文档并提供示例问题
    
    Args:
        progress: Gradio进度条对象
    
    Returns:
        Tuple[str, List[List], str]: (上传状态, 文档列表, 示例问题提示)
    """
    # 示例文档路径
    example_file = Path(__file__).parent.parent / "data" / "examples" / "sample_document.md"
    
    if not example_file.exists():
        return "❌ 示例文件不存在", get_document_list(), ""
    
    progress(0, desc="📦 初始化系统...")
    pipeline = get_pipeline()
    pipeline.initialize()
    
    try:
        progress(0.3, desc="📄 加载示例文档...")
        result = pipeline.index_document(str(example_file))
        
        progress(0.8, desc="🔢 生成向量索引...")
        
        if result.success:
            progress(1.0, desc="✅ 加载完成！")
            
            sample_questions = """📝 **示例问题建议**（复制粘贴到下方输入框）：

🔹 什么是人工智能？它有哪些主要特征？
🔹 What are the main types of machine learning?
🔹 请解释RAG技术的工作原理
🔹 深度学习和机器学习有什么区别？
🔹 What are the advantages of using RAG technology?
🔹 学习AI需要什么基础知识？"""
            
            status = f"✅ 示例文档已加载: {result.chunk_count}个文本块\n\n{sample_questions}"
            return status, get_document_list(), ""
        else:
            return f"❌ 加载失败: {result.message}", get_document_list(), ""
    except Exception as e:
        logger.error(f"加载示例失败: {e}")
        return f"❌ 加载示例失败: {str(e)}", get_document_list(), ""


def handle_upload(files: List, progress=gr.Progress()) -> Tuple[str, List[List]]:
    """
    处理文件上传

    Args:
        files: 上传的文件列表
        progress: Gradio进度条对象

    Returns:
        Tuple[str, List[List]]: (状态消息, 文档列表)
    """
    if not files:
        return "请选择要上传的文件", get_document_list()

    pipeline = get_pipeline()
    
    # 显示初始化进度
    progress(0, desc="📦 初始化系统...")
    pipeline.initialize()

    results = []
    total_files = len(files)
    
    for idx, file in enumerate(files):
        try:
            # file 可能是 tempfile 路径
            file_path = file.name if hasattr(file, 'name') else str(file)
            filename = Path(file_path).name
            
            # 更新进度：显示当前处理的文件
            progress((idx / total_files), desc=f"📄 处理文件 ({idx+1}/{total_files}): {filename}")
            
            # 索引过程的子进度
            progress((idx + 0.3) / total_files, desc=f"📝 切片文档: {filename}")
            result = pipeline.index_document(file_path)
            
            progress((idx + 0.7) / total_files, desc=f"🔢 生成向量: {filename}")

            if result.success:
                results.append(f"✅ {filename}: {result.chunk_count}个文本块")
            else:
                results.append(f"❌ {filename}: {result.message}")
                
        except Exception as e:
            results.append(f"❌ {filename if 'filename' in locals() else '未知文件'}: {str(e)}")
    
    # 完成
    progress(1.0, desc="✅ 索引完成！")
    
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
) -> Generator[Tuple[List[dict], dict, str], None, None]:
    """
    处理用户问题（非流式也用生成器模式以支持即时显示）

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

    # 立即显示用户消息和loading状态
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "🤔 思考中..."})
    yield history, {}, ""
    
    pipeline = get_pipeline()
    pipeline.initialize()

    try:
        result = pipeline.query(
            question=question,
            session_id="gradio_session",
            top_k=top_k
        )

        # 更新助手回答
        history[-1]["content"] = result.answer

        # 格式化来源
        sources_display = {
            "answer_confidence": f"{result.confidence:.2%}",
            "sources": result.sources
        }

        yield history, sources_display, ""

    except ConnectionError as e:
        # 专门捕获连接错误
        logger.error(f"LLM连接失败: {e}")
        error_msg = f"""❌ **LLM服务连接失败**

**可能原因**：
1. 🔴 Ollama服务未启动
2. ⚠️ Ollama地址配置错误
3. 🌐 网络连接问题

**解决方案**：

💡 **方案1：启动Ollama服务**（推荐本地使用）
```bash
# 在新终端执行
ollama serve
```

💡 **方案2：切换到OpenAI模式**（无需本地服务）
1. 在左侧找到 **"🤖 LLM 配置"** 区域
2. **LLM 提供商** 选择 `openai`
3. 填写配置：
   - **API Base URL**: `https://api.deepseek.com/v1` 或 `https://api.openai.com/v1`
   - **API Key**: 你的API密钥
   - **模型名称**: `deepseek-chat` 或 `gpt-3.5-turbo`
4. 点击 **"💾 保存LLM配置"**
5. 重新提问即可

📝 **详细错误信息**: {str(e)}"""
        
        history[-1]["content"] = error_msg
        yield history, {"error": "连接失败"}, ""
        
    except Exception as e:
        logger.error(f"问答失败: {e}")
        
        # 检查是否是网络相关错误
        error_str = str(e).lower()
        network_keywords = [
            'connection', 'connect', 'refused', 'timeout', 
            'errno', 'address', 'network', 'unreachable',
            'socket', 'host', 'port', 'ollama'
        ]
        
        if any(keyword in error_str for keyword in network_keywords):
            error_msg = f"""❌ **LLM服务连接失败**

**可能原因**：
1. 🔴 Ollama服务未启动
2. ⚠️ Ollama地址配置错误
3. 🌐 网络连接问题

**解决方案**：

💡 **方案1：启动Ollama服务**
```bash
ollama serve
```

💡 **方案2：切换到OpenAI模式**
在左侧"🤖 LLM配置"区域：
- 选择 `openai` 提供商
- 填写API Key和模型名称
- 点击"保存LLM配置"

📝 **错误详情**: {str(e)}"""
        else:
            error_msg = f"❌ 处理失败: {str(e)}"
        
        history[-1]["content"] = error_msg
        yield history, {"error": str(e)}, ""


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

    # 立即显示用户消息和loading状态
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "🔍 检索中..."})
    yield history, {}, ""
    
    pipeline = get_pipeline()
    pipeline.initialize()

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

    except ConnectionError as e:
        # 专门捕获连接错误
        logger.error(f"LLM连接失败: {e}")
        error_msg = f"""❌ **LLM服务连接失败**

**可能原因**：
1. 🔴 Ollama服务未启动
2. ⚠️ Ollama地址配置错误
3. 🌐 网络连接问题

**解决方案**：

💡 **方案1：启动Ollama服务**（推荐本地使用）
```bash
# 在新终端执行
ollama serve
```

💡 **方案2：切换到OpenAI模式**（无需本地服务）
1. 在左侧找到 **"🤖 LLM 配置"** 区域
2. **LLM 提供商** 选择 `openai`
3. 填写配置：
   - **API Base URL**: `https://api.deepseek.com/v1` 或 `https://api.openai.com/v1`
   - **API Key**: 你的API密钥
   - **模型名称**: `deepseek-chat` 或 `gpt-3.5-turbo`
4. 点击 **"💾 保存LLM配置"**
5. 重新提问即可

📝 **详细错误信息**: {str(e)}"""
        
        history[-1]["content"] = error_msg
        yield history, {"error": "连接失败"}, ""
        
    except Exception as e:
        logger.error(f"流式问答失败: {e}")
        
        # 检查是否是网络相关错误
        error_str = str(e).lower()
        network_keywords = [
            'connection', 'connect', 'refused', 'timeout',
            'errno', 'address', 'network', 'unreachable',
            'socket', 'host', 'port', 'ollama'
        ]
        
        if any(keyword in error_str for keyword in network_keywords):
            error_msg = f"""❌ **LLM服务连接失败**

**可能原因**：
1. 🔴 Ollama服务未启动
2. ⚠️ Ollama地址配置错误
3. 🌐 网络连接问题

**解决方案**：

💡 **方案1：启动Ollama服务**
```bash
ollama serve
```

💡 **方案2：切换到OpenAI模式**
在左侧"🤖 LLM配置"区域：
- 选择 `openai` 提供商
- 填写API Key和模型名称
- 点击"保存LLM配置"

📝 **错误详情**: {str(e)}"""
        else:
            error_msg = f"❌ 处理失败: {str(e)}"
        
        history[-1]["content"] = error_msg
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


def handle_clear_all() -> Tuple[str, List[List], List[dict], dict]:
    """
    清空所有文档

    Returns:
        Tuple: (状态消息, 空文档列表, 空对话历史, 空来源)
    """
    pipeline = get_pipeline()
    pipeline.clear_all_data()
    pipeline.clear_conversation("gradio_session")
    return "✅ 已清空所有文档", [], [], {}


def get_current_llm_config() -> Dict:
    """
    获取当前LLM配置
    
    Returns:
        Dict: 包含provider, ollama_model, ollama_url, openai_model, openai_key, openai_url
    """
    config = get_config()
    return {
        "provider": config.LLM_PROVIDER,
        "ollama_model": config.LLM_MODEL if config.LLM_PROVIDER == "ollama" else "qwen2.5:7b",
        "ollama_url": config.LLM_BASE_URL,
        "openai_model": config.OPENAI_MODEL,
        "openai_key": config.OPENAI_API_KEY,
        "openai_url": config.LLM_BASE_URL if config.LLM_PROVIDER == "openai" else "https://api.openai.com/v1"
    }


def handle_llm_config_update(
    provider: str,
    ollama_model: str,
    ollama_url: str,
    openai_model: str,
    openai_key: str,
    openai_url: str
) -> str:
    """
    更新LLM配置
    
    Args:
        provider: LLM提供商
        ollama_model: Ollama模型名称
        ollama_url: Ollama API地址
        openai_model: OpenAI模型名称
        openai_key: OpenAI API密钥
        openai_url: OpenAI API地址
        
    Returns:
        str: 状态消息
    """
    pipeline = get_pipeline()
    pipeline.initialize()
    
    try:
        if provider == "ollama":
            success = pipeline.update_generator(
                provider=provider,
                model=ollama_model,
                base_url=ollama_url
            )
        else:  # openai
            if not openai_key:
                return "❌ 请输入OpenAI API Key"
            success = pipeline.update_generator(
                provider=provider,
                model=openai_model,
                base_url=openai_url,
                api_key=openai_key
            )
        
        if success:
            return f"✅ LLM配置已更新: {provider} / {ollama_model if provider == 'ollama' else openai_model}"
        else:
            return "❌ 配置更新失败"
    except Exception as e:
        return f"❌ 配置更新失败: {str(e)}"


def handle_provider_change(provider: str):
    """
    处理Provider切换
    
    Args:
        provider: 选择的Provider
        
    Returns:
        Tuple: 控制各配置区域的可见性
    """
    if provider == "ollama":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


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
        
        💡 点击左侧 "🎯 快速开始：加载示例文档" 按钮立即体验！
        """)

        with gr.Row():
            # ==================== 左侧面板 ====================
            with gr.Column(scale=1):
                gr.Markdown("### 📁 文档管理")
                
                # 快速开始：示例文档按钮
                with gr.Row():
                    load_example_btn = gr.Button(
                        "🎯 快速开始：加载示例文档", 
                        variant="primary",
                        size="sm"
                    )

                # 文件上传
                file_upload = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".txt", ".docx", ".md"],
                    file_count="multiple"
                )

                upload_btn = gr.Button("📤 上传并索引", variant="secondary")
                upload_status = gr.Textbox(
                    label="上传状态",
                    interactive=False,
                    lines=6
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

                gr.Markdown("### ⚙️ 检索设置")

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
                
                gr.Markdown("### 🤖 LLM 配置")
                
                # 获取当前配置
                current_config = get_config()
                
                llm_provider = gr.Dropdown(
                    choices=["ollama", "openai"],
                    value=current_config.LLM_PROVIDER,
                    label="LLM 提供商",
                    info="选择使用Ollama本地模型或OpenAI API"
                )
                
                # Ollama配置区
                with gr.Group(visible=(current_config.LLM_PROVIDER == "ollama")) as ollama_config:
                    ollama_model = gr.Textbox(
                        label="Ollama 模型",
                        value=current_config.LLM_MODEL,
                        placeholder="例如: qwen2.5:7b, llama3:8b"
                    )
                    ollama_url = gr.Textbox(
                        label="Ollama 地址",
                        value=current_config.LLM_BASE_URL,
                        placeholder="http://localhost:11434"
                    )
                
                # OpenAI配置区
                with gr.Group(visible=(current_config.LLM_PROVIDER == "openai")) as openai_config:
                    openai_url = gr.Textbox(
                        label="API Base URL",
                        value="https://api.openai.com/v1",
                        placeholder="https://api.openai.com/v1 或自定义地址"
                    )
                    openai_key = gr.Textbox(
                        label="API Key",
                        value=current_config.OPENAI_API_KEY,
                        placeholder="sk-...",
                        type="password"
                    )
                    openai_model = gr.Textbox(
                        label="模型名称",
                        value=current_config.OPENAI_MODEL,
                        placeholder="例如: gpt-3.5-turbo, gpt-4"
                    )
                
                llm_save_btn = gr.Button("💾 保存LLM配置", variant="secondary", size="sm")
                llm_status = gr.Textbox(
                    label="配置状态",
                    interactive=False,
                    lines=1
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
                        placeholder="请输入您的问题，按回车发送...",
                        lines=1,
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

        # 加载示例文档
        load_example_btn.click(
            fn=handle_load_example,
            inputs=[],
            outputs=[upload_status, doc_table, question_input]
        )

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

        # 清空所有文档（同时清空对话和来源）
        clear_all_btn.click(
            fn=handle_clear_all,
            inputs=[],
            outputs=[upload_status, doc_table, chatbot, sources_json]
        )

        # 发送问题 - 根据流式模式选择处理函数
        def query_with_mode(question, history, top_k, use_stream):
            """根据流式模式选择处理方式"""
            if use_stream:
                # 流式模式：使用 yield from 传递生成器
                yield from handle_query_stream(question, history, top_k)
            else:
                # 非流式模式：也使用生成器以支持即时显示
                yield from handle_query(question, history, top_k)

        send_btn.click(
            fn=query_with_mode,
            inputs=[question_input, chatbot, top_k_slider, stream_mode],
            outputs=[chatbot, sources_json, question_input]
        )

        # 回车发送
        question_input.submit(
            fn=query_with_mode,
            inputs=[question_input, chatbot, top_k_slider, stream_mode],
            outputs=[chatbot, sources_json, question_input]
        )

        # 清空对话
        clear_chat_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[chatbot, sources_json]
        )
        
        # LLM Provider切换事件
        llm_provider.change(
            fn=handle_provider_change,
            inputs=[llm_provider],
            outputs=[ollama_config, openai_config]
        )
        
        # 保存LLM配置事件
        llm_save_btn.click(
            fn=handle_llm_config_update,
            inputs=[llm_provider, ollama_model, ollama_url, openai_model, openai_key, openai_url],
            outputs=[llm_status]
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
