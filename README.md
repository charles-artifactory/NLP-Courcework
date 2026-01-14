# RAG增强智能问答系统

基于检索增强生成(RAG)技术的中英双语智能问答系统。

## 🌟 特色功能

### 核心功能
- **混合检索**: 结合稀疏检索(BM25)和稠密检索(向量)的混合检索策略
- **智能分块**: 基于语义边界的自适应文档分块
- **答案溯源**: 在答案中标注来源，支持查看原文
- **结果重排序**: 使用Cross-Encoder对检索结果进行重排序
- **多轮对话**: 支持上下文感知的多轮问答
- **中英双语**: 同时支持中文和英文文档及问答

### 创新功能
- **LLM动态切换**: 运行时切换Ollama/OpenAI等LLM提供商，支持DeepSeek等兼容API
- **无知识库对话**: 无相关文档时自动切换为普通对话模式
- **多层相似度过滤**: 预过滤+最终过滤，确保只返回高相关结果
- **流式输出**: 支持答案流式输出，提升用户体验

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      表现层 (Gradio)                         │
├─────────────────────────────────────────────────────────────┤
│                    业务逻辑层 (FastAPI)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 文档处理 │ │ 嵌入模块 │ │ 检索模块 │ │ 生成模块 │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
├─────────────────────────────────────────────────────────────┤
│                    数据访问层 (ChromaDB)                     │
├─────────────────────────────────────────────────────────────┤
│                基础设施层 (BGE-M3 + Qwen2.5)                 │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
NLP-Courcework/
├── docs/                           # 项目文档
│   ├── 需求分析.md                  # 功能与非功能需求
│   ├── 概要设计.md                  # 系统架构设计
│   ├── 详细设计.md                  # 模块详细设计
│   └── 测试报告.md                  # 测试用例与结果
├── src/                            # 源代码（模块化结构）
│   ├── __init__.py                 # 包初始化与公共导出
│   ├── config.py                   # 配置管理
│   ├── processing/                 # 文档处理模块
│   │   ├── __init__.py
│   │   └── document_processor.py   # 文档解析与分块
│   ├── retrieval/                  # 检索模块
│   │   ├── __init__.py
│   │   ├── embedder.py             # 文本嵌入与向量存储
│   │   └── retriever.py            # 稀疏/稠密/混合检索
│   ├── core/                       # 核心模块
│   │   ├── __init__.py
│   │   ├── generator.py            # LLM答案生成
│   │   └── pipeline.py             # RAG流水线编排
│   └── api/                        # API服务模块
│       ├── __init__.py
│       └── server.py               # FastAPI服务
├── app/                            # 前端应用
│   ├── __init__.py
│   └── gradio_app.py               # Gradio Web界面
├── tests/                          # 单元测试
│   ├── __init__.py
│   ├── test_document_processor.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
├── data/                           # 数据目录
│   ├── uploads/                    # 上传文件存储
│   ├── chroma_db/                  # ChromaDB向量数据库
│   ├── examples/                   # 示例文档（快速开始）
│   │   └── sample_document.md      # AI/ML中英文示例文档
│   └── sample_docs/                # 用户示例文档
├── requirements.txt                # Python依赖
├── pytest.ini                      # pytest配置
├── main.py                         # 程序入口
└── README.md
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 8GB+ 内存
- (可选) NVIDIA GPU + CUDA

### 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd NLP-Courcework

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 配置LLM

系统支持多种LLM提供商，可在Web界面中动态切换。

#### 方式1: 使用Ollama (推荐本地部署)

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull qwen2.5:7b

# 启动服务
ollama serve
```

#### 方式2: 使用OpenAI API

```bash
# 设置环境变量
export RAG_LLM_PROVIDER=openai
export RAG_OPENAI_API_KEY=your_api_key
export RAG_OPENAI_MODEL=gpt-3.5-turbo
```

#### 方式3: 使用OpenAI兼容API (DeepSeek、Azure等)

在Web界面中选择"openai"提供商，然后配置：
- **API Base URL**: 自定义API地址（如 `https://api.deepseek.com/v1`）
- **API Key**: 您的API密钥
- **模型名称**: 目标模型（如 `deepseek-chat`）

```bash
# 或通过环境变量配置
export RAG_LLM_PROVIDER=openai
export RAG_LLM_BASE_URL=https://api.deepseek.com/v1
export RAG_OPENAI_API_KEY=your_deepseek_api_key
export RAG_OPENAI_MODEL=deepseek-chat
```

### 启动服务

```bash
# 启动Web界面 (默认)
python main.py

# 或者启动API服务
python main.py --api

# 指定端口
python main.py --port 8080

# 创建公共分享链接
python main.py --share
```

访问 http://localhost:7860 即可使用。

## 📖 使用说明

### 🎯 快速开始

系统提供了一键体验功能，无需配置即可快速上手：

1. **点击"快速开始"按钮**：在左侧文档管理区域，点击 "🎯 快速开始：加载示例文档" 按钮
2. **自动加载示例文档**：系统会自动加载包含AI/机器学习知识的中英文混合示例文档
3. **查看示例问题**：上传状态区域会显示多个示例问题，直接复制使用
4. **立即体验**：复制任意示例问题到输入框，按回车即可体验RAG问答

**示例问题包括**：
- 什么是人工智能？它有哪些主要特征？
- What are the main types of machine learning?
- 请解释RAG技术的工作原理
- 深度学习和机器学习有什么区别？

💡 **注意**：如果遇到"LLM连接失败"提示，请参考下方"配置LLM"或"故障排除"部分。

---

### 1. 配置LLM（可选）

在左侧面板的"LLM配置"区域：
- 选择提供商（Ollama/OpenAI）
- 填写相应配置（模型、地址、API Key等）
- 点击"保存LLM配置"

### 2. 上传文档

支持以下格式：
- PDF (.pdf)
- 纯文本 (.txt)
- Word文档 (.docx)
- Markdown (.md)

### 3. 提问

在输入框中输入问题，按回车或点击发送：
- **有相关文档时**：系统检索相关片段 → 使用RAG生成答案 → 显示来源引用
- **无相关文档时**：自动切换普通对话模式 → LLM直接回答

### 4. 查看来源

答案下方会显示引用来源（仅RAG模式）：
- 来源文档名
- 相关片段内容
- 相似度分数

### 5. 其他操作

- **清空对话**: 清空当前对话历史
- **清空全部**: 清空所有文档和对话
- **流式模式**: 勾选后实时显示生成过程

### 🚨 故障排除

#### Ollama连接失败

如果出现"❌ LLM连接失败"错误，系统会自动提供友好的解决方案提示：

**可能原因**：
1. Ollama服务未启动
2. Ollama地址配置错误
3. 网络连接问题

**解决方案**：

**方案1：启动Ollama服务**
```bash
# 终端执行
ollama serve
```

**方案2：切换到OpenAI模式**（无需本地服务）
1. 在左侧"LLM配置"区域选择 `openai`
2. 填写API Key（可使用OpenAI、DeepSeek、智谱等兼容API）
3. 填写模型名称（如：`gpt-3.5-turbo`、`deepseek-chat`）
4. （可选）修改API Base URL为自定义地址
5. 点击"💾 保存LLM配置"

**方案3：使用兼容API（推荐国内用户）**

DeepSeek配置示例：
- LLM提供商：`openai`
- API Base URL：`https://api.deepseek.com/v1`
- API Key：你的DeepSeek API Key
- 模型名称：`deepseek-chat`

## 🔧 配置说明

可通过环境变量配置系统：

| 变量名 | 说明 | 默认值 |
|-------|------|--------|
| RAG_EMBEDDING_MODEL | 嵌入模型 | BAAI/bge-m3 |
| RAG_LLM_PROVIDER | LLM提供商 | ollama |
| RAG_LLM_MODEL | LLM模型 | qwen2.5:7b |
| RAG_TOP_K | 检索数量 | 5 |
| RAG_CHUNK_SIZE | 分块大小 | 512 |
| RAG_HYBRID_ALPHA | 混合检索权重 | 0.7 |
| RAG_TEMPERATURE | 生成温度 | 0.7 |

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行并生成覆盖率报告
pytest tests/ -v --cov=src --cov-report=html
```

## 📊 API文档

启动API服务后，访问 http://localhost:7860/docs 查看Swagger文档。

### 主要接口

| 接口 | 方法 | 描述 |
|-----|------|------|
| /api/documents/upload | POST | 上传文档 |
| /api/documents | GET | 获取文档列表 |
| /api/documents/{id} | DELETE | 删除文档 |
| /api/qa/query | POST | 提交问题 |
| /api/conversation/clear | POST | 清空对话 |

## 🎯 创新点

### 检索增强
1. **混合检索策略**: 结合BM25稀疏检索和向量稠密检索，提高检索准确率
2. **检索结果重排序**: 使用Cross-Encoder对候选结果重排序，提升相关性
3. **多层相似度过滤**: 预过滤+最终过滤，有效过滤不相关结果，避免误导性答案

### 文档处理
4. **自适应分块**: 基于语义边界进行智能分块，保持语义完整性
5. **答案溯源高亮**: 在答案中标注来源，增强可解释性

### 用户体验
6. **LLM动态切换**: 运行时切换Ollama/OpenAI，支持DeepSeek等兼容API
7. **无知识库对话**: 无相关文档时自动切换普通对话模式
8. **多轮对话记忆**: 支持上下文感知的多轮问答
9. **流式输出**: 实时显示生成过程，提升交互体验
10. **一键快速体验**: 内置中英文示例文档，新用户无需准备即可体验
11. **智能错误提示**: Ollama连接失败时提供详细解决方案，引导配置切换

## 🔧 技术栈

| 类别 | 技术 | 说明 |
|-----|------|------|
| 语言 | Python 3.10+ | 主开发语言 |
| 后端 | FastAPI | 高性能API服务 |
| 前端 | Gradio | 机器学习Web界面 |
| 向量库 | ChromaDB | 轻量级向量数据库 |
| 嵌入模型 | BGE-M3 | 中英双语1024维 |
| 重排序 | BGE-reranker | Cross-Encoder |
| LLM | Ollama/OpenAI | 多后端支持 |

## 👥 作者

Zheyun Zhao<2022213670@bupt.cn>

---

**技术栈**: Python | FastAPI | Gradio | ChromaDB | Sentence-Transformers | Ollama | OpenAI
