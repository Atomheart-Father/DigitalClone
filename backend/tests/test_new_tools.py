"""
Tests for the new Sprint 0.5 tools.

Tests web_search, web_read, python_exec, file_read, tabular_qa, markdown_writer tools.
"""

import pytest
import sys
import os
import tempfile
import pandas as pd
from pathlib import Path

# Add the backend directory to Python path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)

from tool_registry import registry


class TestWebSearchTool:
    """Test web_search tool."""

    def test_web_search_parameter_validation(self):
        """Test parameter validation for web_search."""
        # Valid parameters - may fail due to network/API, but should not fail validation
        result = registry.execute('web_search', query='AI tools', k=3)
        # In test environment, this may fail due to no network/API key, but parameters should be valid
        assert result.ok == True or "未找到搜索结果" in result.error or "SerpAPI" in result.error

        # Missing required parameter
        result = registry.execute('web_search', k=3)
        assert result.ok == False
        assert "参数验证失败" in result.error and "query" in result.error

        # Invalid k parameter
        result = registry.execute('web_search', query='test', k=0)
        assert result.ok == False
        assert "参数验证失败" in result.error or "minimum" in result.error


class TestWebReadTool:
    """Test web_read tool."""

    def test_web_read_parameter_validation(self):
        """Test parameter validation for web_read."""
        # Valid parameters - example.com should work
        result = registry.execute('web_read', url='https://example.com', max_length=1000)
        assert result.ok == True  # example.com should work in tests
        assert "Example Domain" in result.value.get('title', '')

        # Missing required parameter
        result = registry.execute('web_read', max_length=1000)
        assert result.ok == False
        assert "参数验证失败" in result.error and "url" in result.error

        # Invalid URL
        result = registry.execute('web_read', url='not-a-url')
        assert result.ok == False
        assert "无效的URL格式" in result.error


class TestPythonExecTool:
    """Test python_exec tool."""

    def test_python_exec_simple_calculation(self):
        """Test simple Python execution."""
        result = registry.execute('python_exec', code='print(2 + 3)', timeout=5)
        assert result.ok == True
        assert '5' in result.value['result']

    def test_python_exec_with_pandas(self):
        """Test Python execution with pandas."""
        code = '''
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df.sum().sum())
'''
        result = registry.execute('python_exec', code=code, timeout=5)
        assert result.ok == True
        assert '21' in result.value['result']

    def test_python_exec_parameter_validation(self):
        """Test parameter validation for python_exec."""
        # Missing required parameter
        result = registry.execute('python_exec', timeout=5)
        assert result.ok == False
        assert "参数验证失败" in result.error and "code" in result.error

    def test_python_exec_unsafe_code(self):
        """Test that unsafe code is rejected."""
        result = registry.execute('python_exec', code='import os; os.system("echo test")', timeout=5)
        assert result.ok == False
        assert "不安全的操作" in result.error


class TestFileReadTool:
    """Test file_read tool."""

    def test_file_read_text_file(self):
        """Test reading a text file."""
        # Create a test file in the data directory (allowed)
        test_file_path = os.path.join(backend_dir, '..', 'data', 'test_file.txt')
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

        try:
            with open(test_file_path, 'w') as f:
                f.write("Hello, World!\nThis is a test file.")

            result = registry.execute('file_read', file_path=test_file_path, max_length=1000)
            assert result.ok == True
            assert "Hello, World!" in result.value['content']
            assert result.value['file_info']['extension'] == '.txt'
        finally:
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)

    def test_file_read_parameter_validation(self):
        """Test parameter validation for file_read."""
        # Missing required parameter
        result = registry.execute('file_read', max_length=1000)
        assert result.ok == False
        assert "参数验证失败" in result.error and "file_path" in result.error

        # Non-existent file
        result = registry.execute('file_read', file_path=os.path.join(backend_dir, '..', 'data', 'non_existent.txt'))
        assert result.ok == False
        assert "文件不存在" in result.error


class TestTabularQATool:
    """Test tabular_qa tool."""

    def test_tabular_qa_csv_file(self):
        """Test querying a CSV file."""
        # Create a test CSV file in the data directory (allowed)
        test_file_path = os.path.join(backend_dir, '..', 'data', 'test_data.csv')
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

        data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'city': ['NYC', 'LA', 'Chicago']}
        df = pd.DataFrame(data)

        try:
            df.to_csv(test_file_path, index=False)
            result = registry.execute('tabular_qa', file_path=test_file_path, query='显示前2行')
            assert result.ok == True
            # The query might return more rows depending on implementation
            assert len(result.value['data']) >= 2
            assert result.value['shape'][0] >= 2
        finally:
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)

    def test_tabular_qa_parameter_validation(self):
        """Test parameter validation for tabular_qa."""
        # Missing required parameters
        result = registry.execute('tabular_qa', file_path='test.csv')
        assert result.ok == False
        assert "参数验证失败" in result.error and "query" in result.error

        result = registry.execute('tabular_qa', query='test query')
        assert result.ok == False
        assert "参数验证失败" in result.error and "file_path" in result.error


class TestMarkdownWriterTool:
    """Test markdown_writer tool."""

    def test_markdown_writer_basic(self):
        """Test basic markdown writing."""
        result = registry.execute('markdown_writer',
                                content='# Hello\n\nThis is a test.',
                                filename='test_doc.md')
        assert result.ok == True
        # Filename processing may modify the final name
        assert 'test_doc' in result.value['file_info']['file_path']
        assert result.value['content_length'] > 0

        # Check if file was created
        file_path = Path(result.value['file_info']['file_path'])
        assert file_path.exists()

        # Clean up
        file_path.unlink()

    def test_markdown_writer_multiple_files(self):
        """Test writing multiple files."""
        # First write
        result1 = registry.execute('markdown_writer',
                                 content='First file content.',
                                 filename='test1.md')
        assert result1.ok == True

        # Second write (different file)
        result2 = registry.execute('markdown_writer',
                                 content='Second file content.',
                                 filename='test2.md')
        assert result2.ok == True

        # Clean up
        Path(result1.value['file_info']['file_path']).unlink()
        Path(result2.value['file_info']['file_path']).unlink()

    def test_markdown_writer_parameter_validation(self):
        """Test parameter validation for markdown_writer."""
        # Missing required parameters
        result = registry.execute('markdown_writer', title='Test')
        assert result.ok == False
        assert "参数验证失败" in result.error and "content" in result.error

        result = registry.execute('markdown_writer', content='test content')
        assert result.ok == False
        assert "参数验证失败" in result.error and "filename" in result.error


class TestToolIntegration:
    """Test tool integration and registry."""

    def test_all_new_tools_registered(self):
        """Test that all new tools are properly registered."""
        tools = registry.list_tools()
        tool_names = [tool.name for tool in tools]

        new_tools = ['web_search', 'web_read', 'python_exec', 'file_read', 'tabular_qa', 'markdown_writer']
        for tool_name in new_tools:
            assert tool_name in tool_names, f"Tool {tool_name} not registered"

    def test_new_tools_have_correct_executors(self):
        """Test that new tools have correct executor assignments."""
        tools = registry.list_tools()
        tool_dict = {tool.name: tool for tool in tools}

        # Simple tools should use chat executor
        simple_tools = ['web_search', 'web_read', 'file_read', 'markdown_writer']
        for tool_name in simple_tools:
            assert tool_dict[tool_name].executor_default == 'chat'

        # Complex tools should use reasoner executor
        complex_tools = ['tabular_qa', 'python_exec']
        for tool_name in complex_tools:
            assert tool_dict[tool_name].executor_default == 'reasoner'

    def test_new_tools_have_complexity_metadata(self):
        """Test that new tools have complexity metadata."""
        tools = registry.list_tools()
        tool_dict = {tool.name: tool for tool in tools}

        # Check complexity assignments
        simple_tools = ['web_search', 'web_read', 'file_read', 'markdown_writer']
        complex_tools = ['tabular_qa', 'python_exec']

        for tool_name in simple_tools:
            assert tool_dict[tool_name].complexity == 'simple'

        for tool_name in complex_tools:
            assert tool_dict[tool_name].complexity == 'complex'
