# RAG增强智能问答系统

基于检索增强生成(RAG)技术的中英双语智能问答系统。

## 🌟 特色功能

- **混合检索**: 结合稀疏检索(BM25)和稠密检索(向量)的混合检索策略
- **智能分块**: 基于语义边界的自适应文档分块
- **答案溯源**: 在答案中标注来源，支持查看原文
- **结果重排序**: 使用Cross-Encoder对检索结果进行重排序
- **多轮对话**: 支持上下文感知的多轮问答
- **中英双语**: 同时支持中文和英文文档及问答

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
nlp-rag-qa/
├── docs/                     # 文档
│   ├── 需求分析.md
│   ├── 概要设计.md
│   ├── 详细设计.md
│   └── 测试报告.md
├── src/                      # 源代码
│   ├── __init__.py
│   ├── config.py             # 配置管理
│   ├── document_processor.py # 文档处理
│   ├── embedder.py           # 文本嵌入
│   ├── retriever.py          # 文档检索
│   ├── generator.py          # 答案生成
│   ├── rag_pipeline.py       # RAG流水线
│   └── api.py                # API服务
├── app/                      # 前端应用
│   ├── __init__.py
│   └── gradio_app.py
├── tests/                    # 测试代码
│   ├── test_document_processor.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
├── data/                     # 数据目录
│   ├── uploads/              # 上传文件
│   ├── chroma_db/            # 向量数据库
│   └── sample_docs/          # 示例文档
├── requirements.txt          # 依赖列表
├── main.py                   # 主入口
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

#### 方式1: 使用Ollama (推荐)

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

### 1. 上传文档

支持以下格式：
- PDF (.pdf)
- 纯文本 (.txt)
- Word文档 (.docx)
- Markdown (.md)

### 2. 提问

在输入框中输入问题，系统会：
1. 检索相关文档片段
2. 使用LLM生成答案
3. 标注答案来源

### 3. 查看来源

答案下方会显示引用来源，包含：
- 来源文档名
- 相关片段内容
- 相似度分数

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

1. **混合检索策略**: 结合BM25稀疏检索和向量稠密检索，提高检索准确率
2. **自适应分块**: 基于语义边界进行智能分块，保持语义完整性
3. **答案溯源高亮**: 在答案中标注来源，增强可解释性
4. **检索结果重排序**: 使用Cross-Encoder对候选结果重排序，提升相关性
5. **多轮对话记忆**: 支持上下文感知的多轮问答

## 📄 许可证

MIT License

## 👥 作者

NLP课程项目

---

**技术栈**: Python | FastAPI | Gradio | ChromaDB | Sentence-Transformers | Ollama
