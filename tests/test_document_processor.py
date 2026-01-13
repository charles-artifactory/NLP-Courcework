"""
文档处理模块测试
"""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.document_processor import (
    DocumentProcessor,
    SemanticChunker,
    RecursiveChunker,
    Document,
    Chunk,
    DocumentParseError
)


class TestDocumentProcessor:
    """文档处理器测试类"""
    
    @pytest.fixture
    def processor(self):
        """创建文档处理器实例"""
        return DocumentProcessor(chunk_size=200, chunk_overlap=20)
    
    @pytest.fixture
    def sample_txt_file(self):
        """创建示例TXT文件"""
        content = """这是一个测试文档。
        
它包含多个段落，用于测试文档处理功能。

第一段讨论了自然语言处理的基本概念。自然语言处理是人工智能的一个重要分支。

第二段介绍了机器学习的应用。机器学习在文本分析中有广泛应用。

第三段说明了检索增强生成技术。RAG技术可以提高问答系统的准确性。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            return f.name
    
    @pytest.fixture
    def sample_md_file(self):
        """创建示例Markdown文件"""
        content = """# 测试文档

## 第一章

这是第一章的内容。包含一些**重要**的信息。

## 第二章

这是第二章的内容。

- 列表项1
- 列表项2

[链接](http://example.com)

```python
print("Hello World")
```
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            return f.name
    
    def test_load_txt_document(self, processor, sample_txt_file):
        """测试加载TXT文档"""
        doc = processor.load_document(sample_txt_file)
        
        assert isinstance(doc, Document)
        assert doc.filename.endswith('.txt')
        assert doc.format == '.txt'
        assert len(doc.content) > 0
        assert "自然语言处理" in doc.content
        
        # 清理
        Path(sample_txt_file).unlink()
    
    def test_load_md_document(self, processor, sample_md_file):
        """测试加载Markdown文档"""
        doc = processor.load_document(sample_md_file)
        
        assert isinstance(doc, Document)
        assert doc.format == '.md'
        assert len(doc.content) > 0
        
        # 清理
        Path(sample_md_file).unlink()
    
    def test_unsupported_format(self, processor):
        """测试不支持的文件格式"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"test")
            filepath = f.name
        
        with pytest.raises(DocumentParseError):
            processor.load_document(filepath)
        
        Path(filepath).unlink()
    
    def test_file_not_found(self, processor):
        """测试文件不存在"""
        with pytest.raises(FileNotFoundError):
            processor.load_document("/nonexistent/file.txt")
    
    def test_process_document(self, processor, sample_txt_file):
        """测试文档分块处理"""
        doc = processor.load_document(sample_txt_file)
        chunks = processor.process(doc)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        
        # 检查块元数据
        for chunk in chunks:
            assert chunk.document_id == doc.id
            assert len(chunk.content) > 0
            assert "filename" in chunk.metadata
        
        # 清理
        Path(sample_txt_file).unlink()
    
    def test_process_file_convenience(self, processor, sample_txt_file):
        """测试便捷方法process_file"""
        doc, chunks = processor.process_file(sample_txt_file)
        
        assert isinstance(doc, Document)
        assert isinstance(chunks, list)
        assert doc.chunk_count == len(chunks)
        
        # 清理
        Path(sample_txt_file).unlink()


class TestSemanticChunker:
    """语义分块器测试类"""
    
    @pytest.fixture
    def chunker(self):
        """创建语义分块器"""
        return SemanticChunker(chunk_size=200, chunk_overlap=20)
    
    def test_chunk_empty_text(self, chunker):
        """测试空文本分块"""
        chunks = chunker.chunk("", "doc_1")
        assert chunks == []
    
    def test_chunk_short_text(self, chunker):
        """测试短文本分块"""
        text = "这是一段短文本。"
        chunks = chunker.chunk(text, "doc_1")
        
        assert len(chunks) == 1
        assert chunks[0].content.strip() == text.strip()
    
    def test_chunk_long_text(self, chunker):
        """测试长文本分块"""
        text = "这是第一句话。" * 50 + "这是第二部分。" * 50
        chunks = chunker.chunk(text, "doc_1")
        
        assert len(chunks) > 1
        
        # 检查每个块的大小合理
        for chunk in chunks:
            # 允许一定的误差
            assert len(chunk.content) <= chunker.chunk_size * 2
    
    def test_chunk_preserves_sentences(self, chunker):
        """测试分块保持句子完整性"""
        text = "这是完整的第一句。这是完整的第二句。这是完整的第三句。"
        chunks = chunker.chunk(text, "doc_1")
        
        # 合并所有块内容应包含原始句子
        combined = " ".join(c.content for c in chunks)
        assert "第一句" in combined
        assert "第二句" in combined
    
    def test_chunk_metadata(self, chunker):
        """测试块包含正确的元数据"""
        text = "这是测试文本。用于检查元数据。"
        chunks = chunker.chunk(text, "doc_123")
        
        for chunk in chunks:
            assert chunk.document_id == "doc_123"
            assert chunk.start_pos >= 0
            assert chunk.end_pos > chunk.start_pos


class TestRecursiveChunker:
    """递归分块器测试类"""
    
    @pytest.fixture
    def chunker(self):
        """创建递归分块器"""
        return RecursiveChunker(chunk_size=100, chunk_overlap=10)
    
    def test_recursive_chunk_long_text(self, chunker):
        """测试递归分块长文本"""
        text = "这是一段很长的文本。" * 100
        chunks = chunker.chunk(text, "doc_1")
        
        assert len(chunks) > 1
        
        # 验证内容被完整保留
        combined = "".join(c.content for c in chunks)
        # 由于重叠，可能有重复内容
        assert "这是一段很长的文本" in combined


class TestChineseEnglishMixed:
    """中英文混合测试"""
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=300, chunk_overlap=30)
    
    def test_mixed_language_document(self, processor):
        """测试中英文混合文档"""
        content = """This is an English paragraph about NLP.
        
这是一段中文内容，介绍自然语言处理技术。

Natural Language Processing (NLP) is a field of AI.
自然语言处理是人工智能的一个分支。

The RAG technique combines retrieval and generation.
RAG技术结合了检索和生成。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            filepath = f.name
        
        doc, chunks = processor.process_file(filepath)
        
        # 应该成功处理
        assert len(chunks) > 0
        
        # 合并后应包含中英文内容
        combined = " ".join(c.content for c in chunks)
        assert "NLP" in combined
        assert "自然语言处理" in combined
        
        # 清理
        Path(filepath).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
