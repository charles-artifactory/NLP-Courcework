"""
生成模块

负责构建Prompt和调用LLM生成答案
"""

import logging
import re
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass

from .embedder import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SourceRef:
    """来源引用数据类"""
    text: str
    chunk_id: str
    document: str
    score: float
    position: Optional[Tuple[int, int]] = None


@dataclass 
class GenerationResult:
    """生成结果数据类"""
    answer: str
    sources: List[SourceRef]
    confidence: float


# ==================== Prompt模板 ====================

SYSTEM_PROMPT_CN = """你是一个专业的问答助手。请根据提供的参考资料回答用户问题。

规则：
1. 只根据参考资料回答，不要编造信息
2. 如果参考资料中没有相关内容，请明确说明"根据现有资料无法回答这个问题"
3. 回答时引用来源，使用[1], [2]等标记
4. 回答要准确、简洁、有条理
5. 使用与用户问题相同的语言回答"""

SYSTEM_PROMPT_EN = """You are a professional Q&A assistant. Please answer user questions based on the provided reference materials.

Rules:
1. Only answer based on reference materials, do not make up information
2. If there is no relevant content in the reference materials, clearly state "Unable to answer this question based on available information"
3. Cite sources using [1], [2] markers when answering
4. Answers should be accurate, concise, and well-organized
5. Answer in the same language as the user's question"""

RAG_TEMPLATE = """参考资料：
{contexts}

用户问题：{query}

请根据参考资料回答上述问题："""

RAG_TEMPLATE_WITH_HISTORY = """参考资料：
{contexts}

对话历史：
{history}

用户问题：{query}

请根据参考资料和对话上下文回答上述问题："""


class PromptBuilder:
    """Prompt构建器"""
    
    def __init__(
        self,
        system_prompt: str = None,
        rag_template: str = None,
        rag_template_with_history: str = None
    ):
        """
        初始化Prompt构建器
        
        Args:
            system_prompt: 系统提示词
            rag_template: RAG模板
            rag_template_with_history: 带历史的RAG模板
        """
        self.system_prompt = system_prompt or SYSTEM_PROMPT_CN
        self.rag_template = rag_template or RAG_TEMPLATE
        self.rag_template_with_history = rag_template_with_history or RAG_TEMPLATE_WITH_HISTORY
    
    def build_contexts(
        self,
        search_results: List[SearchResult],
        max_contexts: int = 5
    ) -> str:
        """
        构建上下文文本
        
        Args:
            search_results: 检索结果
            max_contexts: 最大上下文数量
            
        Returns:
            str: 格式化的上下文
        """
        if not search_results:
            return "（无相关参考资料）"
        
        contexts = []
        for i, result in enumerate(search_results[:max_contexts], 1):
            source = result.metadata.get("filename", "未知来源")
            contexts.append(f"[{i}] 来源：{source}\n{result.content}")
        
        return "\n\n".join(contexts)
    
    def build_history(self, history: List[Dict]) -> str:
        """
        构建对话历史文本
        
        Args:
            history: 对话历史列表
            
        Returns:
            str: 格式化的对话历史
        """
        if not history:
            return ""
        
        lines = []
        for msg in history[-6:]:  # 最近3轮对话
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(lines)
    
    def build_prompt(
        self,
        query: str,
        search_results: List[SearchResult],
        history: List[Dict] = None
    ) -> str:
        """
        构建完整的Prompt
        
        Args:
            query: 用户问题
            search_results: 检索结果
            history: 对话历史
            
        Returns:
            str: 完整的prompt
        """
        contexts = self.build_contexts(search_results)
        
        if history:
            history_text = self.build_history(history)
            prompt = self.rag_template_with_history.format(
                contexts=contexts,
                history=history_text,
                query=query
            )
        else:
            prompt = self.rag_template.format(
                contexts=contexts,
                query=query
            )
        
        return prompt
    
    def build_messages(
        self,
        query: str,
        search_results: List[SearchResult],
        history: List[Dict] = None
    ) -> List[Dict]:
        """
        构建消息列表格式（用于Chat API）
        
        Args:
            query: 用户问题
            search_results: 检索结果
            history: 对话历史
            
        Returns:
            List[Dict]: 消息列表
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加历史消息
        if history:
            for msg in history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # 添加当前问题（带上下文）
        contexts = self.build_contexts(search_results)
        user_message = f"参考资料：\n{contexts}\n\n问题：{query}"
        messages.append({"role": "user", "content": user_message})
        
        return messages


class SourceTracer:
    """
    来源追踪器 - 创新点
    
    追踪答案中的内容来源
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        初始化来源追踪器
        
        Args:
            similarity_threshold: 相似度阈值
        """
        self.similarity_threshold = similarity_threshold
    
    def trace_sources(
        self,
        answer: str,
        search_results: List[SearchResult]
    ) -> List[SourceRef]:
        """
        追踪答案中的内容来源
        
        Args:
            answer: 生成的答案
            search_results: 检索结果
            
        Returns:
            List[SourceRef]: 来源引用列表
        """
        sources = []
        
        # 从答案中提取已有的引用标记
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        for citation in set(citations):
            idx = int(citation) - 1
            if 0 <= idx < len(search_results):
                result = search_results[idx]
                sources.append(SourceRef(
                    text=f"[{citation}]",
                    chunk_id=result.chunk_id,
                    document=result.metadata.get("filename", "unknown"),
                    score=result.score
                ))
        
        return sources
    
    def highlight_sources(
        self,
        answer: str,
        sources: List[SourceRef]
    ) -> str:
        """
        在答案中高亮显示来源
        
        对于已经有引用标记的答案，直接返回
        
        Args:
            answer: 答案文本
            sources: 来源列表
            
        Returns:
            str: 带高亮的答案
        """
        # 如果已经有引用标记，直接返回
        if re.search(r'\[\d+\]', answer):
            return answer
        
        return answer
    
    def format_sources(
        self,
        search_results: List[SearchResult]
    ) -> List[Dict]:
        """
        格式化来源信息
        
        Args:
            search_results: 检索结果
            
        Returns:
            List[Dict]: 格式化的来源列表
        """
        return [
            {
                "index": i + 1,
                "document": result.metadata.get("filename", "unknown"),
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "score": round(result.score, 3)
            }
            for i, result in enumerate(search_results)
        ]


class Generator:
    """
    答案生成器
    
    支持Ollama和OpenAI两种后端
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        初始化生成器
        
        Args:
            provider: 提供商 (ollama/openai)
            model: 模型名称
            base_url: API地址
            api_key: API密钥（OpenAI需要）
            temperature: 温度参数
            max_tokens: 最大生成token数
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.prompt_builder = PromptBuilder()
        self.source_tracer = SourceTracer()
        
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化LLM客户端"""
        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "openai":
            self._init_openai()
        else:
            logger.warning(f"未知的提供商: {self.provider}，使用模拟模式")
    
    def _init_ollama(self):
        """初始化Ollama客户端"""
        try:
            import ollama
            self._client = ollama.Client(host=self.base_url)
            # 测试连接
            try:
                self._client.list()
                logger.info(f"Ollama连接成功: {self.base_url}")
            except Exception as e:
                logger.warning(f"Ollama连接失败: {e}，将使用模拟模式")
                self._client = None
        except ImportError:
            logger.warning("ollama库未安装，使用模拟模式")
            self._client = None
    
    def _init_openai(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            if self.api_key:
                self._client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI客户端初始化成功")
            else:
                logger.warning("未提供OpenAI API Key")
                self._client = None
        except ImportError:
            logger.warning("openai库未安装")
            self._client = None
    
    def generate(
        self,
        query: str,
        search_results: List[SearchResult],
        history: List[Dict] = None
    ) -> GenerationResult:
        """
        生成答案
        
        Args:
            query: 用户问题
            search_results: 检索结果
            history: 对话历史
            
        Returns:
            GenerationResult: 生成结果
        """
        # 构建消息
        messages = self.prompt_builder.build_messages(
            query, search_results, history
        )
        
        # 调用LLM
        if self._client is None:
            # 模拟模式
            answer = self._generate_mock(query, search_results)
        elif self.provider == "ollama":
            answer = self._generate_ollama(messages)
        elif self.provider == "openai":
            answer = self._generate_openai(messages)
        else:
            answer = self._generate_mock(query, search_results)
        
        # 追踪来源
        sources = self.source_tracer.trace_sources(answer, search_results)
        
        # 计算置信度
        if search_results:
            confidence = sum(r.score for r in search_results[:3]) / min(3, len(search_results))
        else:
            confidence = 0.0
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    
    def _generate_ollama(self, messages: List[Dict]) -> str:
        """使用Ollama生成"""
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama生成失败: {e}")
            return f"生成失败：{str(e)}"
    
    def _generate_openai(self, messages: List[Dict]) -> str:
        """使用OpenAI生成"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI生成失败: {e}")
            return f"生成失败：{str(e)}"
    
    def _generate_mock(
        self,
        query: str,
        search_results: List[SearchResult]
    ) -> str:
        """模拟生成（用于测试或无LLM时）"""
        if not search_results:
            return "根据现有资料无法回答这个问题。请上传相关文档后再试。"
        
        # 基于检索结果构建简单回答
        top_result = search_results[0]
        source = top_result.metadata.get("filename", "未知来源")
        
        answer = f"""根据参考资料，我找到了以下相关信息：

{top_result.content[:500]}

[1] 来源：{source}

注意：当前使用模拟模式。如需更准确的回答，请配置Ollama或OpenAI。"""
        
        return answer
    
    def generate_stream(
        self,
        query: str,
        search_results: List[SearchResult],
        history: List[Dict] = None
    ) -> Iterator[str]:
        """
        流式生成答案
        
        Args:
            query: 用户问题
            search_results: 检索结果
            history: 对话历史
            
        Yields:
            str: 生成的文本片段
        """
        messages = self.prompt_builder.build_messages(
            query, search_results, history
        )
        
        if self._client is None or self.provider not in ["ollama", "openai"]:
            # 模拟流式输出
            mock_response = self._generate_mock(query, search_results)
            for char in mock_response:
                yield char
            return
        
        try:
            if self.provider == "ollama":
                for chunk in self._stream_ollama(messages):
                    yield chunk
            elif self.provider == "openai":
                for chunk in self._stream_openai(messages):
                    yield chunk
        except Exception as e:
            yield f"生成失败：{str(e)}"
    
    def _stream_ollama(self, messages: List[Dict]) -> Iterator[str]:
        """Ollama流式生成"""
        response = self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    
    def _stream_openai(self, messages: List[Dict]) -> Iterator[str]:
        """OpenAI流式生成"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
