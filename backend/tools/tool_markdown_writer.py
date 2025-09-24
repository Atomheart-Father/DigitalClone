"""
Markdown Writer Tool for the Digital Clone AI Assistant.

Formats analysis results and writes them to markdown files in the data/notes/ directory.
"""

import os
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "markdown_writer",
    "description": "将分析结果格式化为Markdown文档并保存到输出目录",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要写入的内容"
            },
            "filename": {
                "type": "string",
                "description": "文件名（会自动生成时间戳和.md扩展名）"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "标签列表",
                "default": []
            }
        },
        "required": ["content", "filename"],
        "additionalProperties": False
    },
    "strict": True,
    "executor_default": "chat",
    "complexity": "simple",
    "arg_hint": "content为文档内容；filename为文件名；tags为标签列表。文件会自动保存到OUTPUT_DIR目录。",
    "caller_snippet": "用于保存分析结果、调研报告、总结文档等。文件会保存在环境变量OUTPUT_DIR指定的目录中。"
}

def _get_output_directory() -> Path:
    """Get or create output directory from config."""
    try:
        from config import Config
        output_dir = Path(Config.OUTPUT_DIR)
    except ImportError:
        # Fallback if config import fails
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "data" / "output"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _generate_filename(filename: str) -> str:
    """Generate safe filename with timestamp."""
    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()

    # Add timestamp and extension
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if safe_filename:
        return f"{safe_filename}_{timestamp}.md"
    else:
        return f"document_{timestamp}.md"

def _format_header(filename: str, tags: List[str]) -> str:
    """Format markdown header."""
    # Extract title from filename (remove timestamp and extension)
    title = filename.replace('.md', '').split('_')[0] if '_' in filename else filename.replace('.md', '')

    header_lines = [
        f"# {title}",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if tags:
        tag_str = " ".join(f"`{tag}`" for tag in tags)
        header_lines.append(f"**标签**: {tag_str}")

    header_lines.extend([
        "",
        "---",
        ""
    ])

    return "\n".join(header_lines)

def _format_content(content: str) -> str:
    """Format content with basic markdown improvements."""
    # Ensure content ends with newline
    if not content.endswith('\n'):
        content += '\n'

    return content

def _format_footer(metadata: Dict[str, Any] = None) -> str:
    """Format footer with metadata."""
    footer_lines = [
        "",
        "---",
        "",
        "*本文档由赛博克隆AI助手生成*",
    ]

    if metadata:
        footer_lines.append("")
        footer_lines.append("**元数据**:")
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            footer_lines.append(f"- **{key}**: {value}")

    return "\n".join(footer_lines)

def _save_file(file_path: Path, content: str, append: bool = False) -> Dict[str, Any]:
    """Save content to file."""
    mode = 'a' if append else 'w'
    encoding = 'utf-8'

    try:
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)

        # Get file info
        stat = file_path.stat()
        return {
            "file_path": str(file_path),
            "size": stat.st_size,
            "mode": "追加" if append else "新建",
            "encoding": encoding
        }

    except Exception as e:
        raise ValueError(f"文件保存失败: {e}")

def run(content: str, filename: str, tags: List[str] = None) -> Dict[str, Any]:
    """
    Execute markdown_writer tool.

    Args:
        content: Content to write
        filename: Base filename (will be made safe and get timestamp)
        tags: List of tags

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not content or not content.strip():
            return {"ok": False, "error": "内容不能为空"}

        if not filename or not filename.strip():
            return {"ok": False, "error": "文件名不能为空"}

        content = content.strip()
        filename = filename.strip()
        tags = tags or []

        # Get output directory
        output_dir = _get_output_directory()

        # Generate safe filename with timestamp
        final_filename = _generate_filename(filename)
        file_path = output_dir / final_filename

        logger.info(f"Writing markdown file: {file_path}")

        # Always create new file (no append mode for simplicity)
        formatted_content = (
            _format_header(final_filename, tags) +
            _format_content(content) +
            _format_footer({"original_filename": filename, "tags": tags})
        )

        # Save file
        file_info = _save_file(file_path, formatted_content, append=False)

        result = {
            "filename": final_filename,
            "file_info": file_info,
            "content_length": len(content),
            "formatted_length": len(formatted_content),
            "tags": tags
        }

        logger.info(f"Markdown file written: {len(formatted_content)} characters to {file_path}")
        return {"ok": True, "value": result}

    except Exception as e:
        error_msg = f"Markdown写入失败: {str(e)}"
        logger.error(f"Markdown writer error: {e}")
        return {"ok": False, "error": error_msg}
