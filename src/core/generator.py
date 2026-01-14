"""
生成模块

负责构建Prompt和调用LLM生成答案
"""

import logging
import re
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass

from ..retrieval.embedder import SearchResult

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

# RAG模式的系统提示词（有知识库时使用）
SYSTEM_PROMPT_RAG_CN = """你是一个专业的问答助手。请根据提供的参考资料回答用户问题。

规则：
1. 优先根据参考资料回答，不要编造信息
2. 如果参考资料中没有相关内容，请明确说明"根据现有资料无法回答这个问题"
3. 回答时引用来源，使用[1], [2]等标记
4. 回答要准确、简洁、有条理
5. 使用与用户问题相同的语言回答"""

# 通用对话模式的系统提示词（无知识库时使用）
SYSTEM_PROMPT_CHAT_CN = """你是一个友好、专业的AI助手。请用准确、简洁、有条理的方式回答用户问题。

规则：
1. 回答要准确、简洁、有条理
2. 使用与用户问题相同的语言回答
3. 如果不确定答案，请诚实说明
4. 保持友好和专业的态度"""

# 兼容旧版本的别名
SYSTEM_PROMPT_CN = SYSTEM_PROMPT_RAG_CN

SYSTEM_PROMPT_EN = """You are a professional Q&A assistant. Please answer user questions based on the provided reference materials.

Rules:
1. Prioritize answering based on reference materials, do not make up information
2. If there is no relevant content in the reference materials, clearly state "Unable to answer this question based on available information"
3. Cite sources using [1], [2] markers when answering
4. Answers should be accurate, concise, and well-organized
5. Answer in the same language as the user's question"""

SYSTEM_PROMPT_CHAT_EN = """You are a friendly and professional AI assistant. Please answer user questions accurately, concisely, and in an organized manner.

Rules:
1. Answers should be accurate, concise, and well-organized
2. Answer in the same language as the user's question
3. If uncertain about an answer, be honest about it
4. Maintain a friendly and professional attitude"""

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
        self.system_prompt = system_prompt or SYSTEM_PROMPT_CN
        self.rag_template = rag_template or RAG_TEMPLATE
        self.rag_template_with_history = rag_template_with_history or RAG_TEMPLATE_WITH_HISTORY
    
    def build_contexts(
        self,
        search_results: List[SearchResult],
        max_contexts: int = 5
    ) -> str:
        """构建上下文文本"""
        if not search_results:
            return "（无相关参考资料）"
        
        contexts = []
        for i, result in enumerate(search_results[:max_contexts], 1):
            source = result.metadata.get("filename", "未知来源")
            contexts.append(f"[{i}] 来源：{source}\n{result.content}")
        
        return "\n\n".join(contexts)
    
    def build_history(self, history: List[Dict]) -> str:
        """构建对话历史文本"""
        if not history:
            return ""
        
        lines = []
        for msg in history[-6:]:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(lines)
    
    def build_prompt(
        self,
        query: str,
        search_results: List[SearchResult],
        history: List[Dict] = None
    ) -> str:
        """构建完整的Prompt"""
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
        """构建消息列表格式（用于Chat API）"""
        if search_results:
            system_prompt = self.system_prompt
        else:
            system_prompt = SYSTEM_PROMPT_CHAT_CN
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            for msg in history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        if search_results:
            contexts = self.build_contexts(search_results)
            user_message = f"参考资料：\n{contexts}\n\n问题：{query}"
        else:
            user_message = query
        messages.append({"role": "user", "content": user_message})
        
        return messages


class SourceTracer:
    """
    来源追踪器 - 创新点
    
    追踪答案中的内容来源
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
    
    def trace_sources(
        self,
        answer: str,
        search_results: List[SearchResult]
    ) -> List[SourceRef]:
        """追踪答案中的内容来源"""
        sources = []
        
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
        
        注意：当前实现仅验证引用标记是否存在，
        未来可扩展为HTML/Markdown高亮格式
        """
        # 验证答案中已有引用标记
        if re.search(r'\[\d+\]', answer):
            return answer
        
        # 如果没有引用标记但有来源，可以在末尾添加来源说明
        if sources:
            source_notes = "\n\n**参考来源：**\n"
            for src in sources:
                source_notes += f"- {src.text}: {src.document}\n"
            return answer + source_notes
        
        return answer
    
    def format_sources(
        self,
        search_results: List[SearchResult]
    ) -> List[Dict]:
        """格式化来源信息"""
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
            logger.info(f"Ollama客户端已创建: {self.base_url}")
            # 不在初始化时检查连接，避免服务暂时不可用导致永久进入mock模式
            # 连接问题将在实际调用时处理
        except ImportError:
            logger.warning("ollama库未安装，使用模拟模式")
            self._client = None
    
    def _init_openai(self):
        """初始化OpenAI客户端（支持OpenAI兼容API，如DeepSeek等）"""
        try:
            from openai import OpenAI
            if self.api_key:
                client_kwargs = {"api_key": self.api_key}
                if self.base_url and self.base_url.strip():
                    client_kwargs["base_url"] = self.base_url
                    logger.info(f"使用自定义API地址: {self.base_url}")
                
                self._client = OpenAI(**client_kwargs)
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
        """生成答案"""
        messages = self.prompt_builder.build_messages(
            query, search_results, history
        )
        
        if self._client is None:
            answer = self._generate_mock(query, search_results)
        elif self.provider == "ollama":
            answer = self._generate_ollama(messages)
        elif self.provider == "openai":
            answer = self._generate_openai(messages)
        else:
            answer = self._generate_mock(query, search_results)
        
        sources = self.source_tracer.trace_sources(answer, search_results)
        
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
        except (ConnectionError, OSError) as e:
            # 捕获连接错误和网络错误
            logger.error(f"Ollama连接失败: {e}")
            raise ConnectionError(f"Ollama服务连接失败 ({self.base_url}): {str(e)}")
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Ollama生成失败: {e}")
            
            # 检查是否是网络/连接相关错误
            network_error_keywords = [
                'connection', 'connect', 'refused', 'timeout',
                'errno', 'address', 'network', 'unreachable',
                'socket', 'host', 'port'
            ]
            
            if any(keyword in error_msg for keyword in network_error_keywords):
                raise ConnectionError(f"Ollama服务连接失败 ({self.base_url}): {str(e)}")
            elif "model" in error_msg and "not found" in error_msg:
                return f"⚠️ 模型 '{self.model}' 未找到。请先运行：`ollama pull {self.model}`"
            
            # 其他未知错误
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
            return f"""您好！您的问题是："{query}"

当前系统处于模拟模式，无法提供真实回答。

请在左侧"LLM配置"区域配置Ollama或OpenAI后即可正常使用问答功能。

💡 提示：
- 使用Ollama：确保Ollama服务已启动，并填写正确的模型名称和地址
- 使用OpenAI：填写您的API Key和模型名称"""
        
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
        """流式生成答案"""
        messages = self.prompt_builder.build_messages(
            query, search_results, history
        )
        
        if self._client is None or self.provider not in ["ollama", "openai"]:
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
        except ConnectionError as e:
            # 连接错误 - 重新抛出，让上层处理友好提示
            raise
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"流式生成失败: {e}")
            
            # 检查是否是网络相关错误
            network_keywords = [
                'connection', 'connect', 'refused', 'timeout',
                'errno', 'address', 'network', 'unreachable',
                'socket', 'host', 'port'
            ]
            
            if any(keyword in error_msg for keyword in network_keywords):
                # 网络错误 - 抛出ConnectionError让上层处理
                raise ConnectionError(f"服务连接失败: {str(e)}")
            elif "model" in error_msg and "not found" in error_msg:
                yield f"⚠️ 模型 '{self.model}' 未找到。请先运行：`ollama pull {self.model}`"
            else:
                yield f"生成失败：{str(e)}"
    
    def _stream_ollama(self, messages: List[Dict]) -> Iterator[str]:
        """Ollama流式生成"""
        try:
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
        except (ConnectionError, OSError) as e:
            # 捕获连接错误和网络错误
            logger.error(f"Ollama流式连接失败: {e}")
            raise ConnectionError(f"Ollama服务连接失败 ({self.base_url}): {str(e)}")
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Ollama流式生成失败: {e}")
            
            # 检查是否是网络/连接相关错误
            network_error_keywords = [
                'connection', 'connect', 'refused', 'timeout',
                'errno', 'address', 'network', 'unreachable',
                'socket', 'host', 'port'
            ]
            
            if any(keyword in error_msg for keyword in network_error_keywords):
                raise ConnectionError(f"Ollama服务连接失败 ({self.base_url}): {str(e)}")
            
            # 其他错误继续抛出
            raise
    
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
