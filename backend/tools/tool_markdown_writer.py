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
    "description": "将分析结果格式化为Markdown文档并保存到data/notes/目录",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要写入的内容"
            },
            "title": {
                "type": "string",
                "description": "文档标题"
            },
            "filename": {
                "type": "string",
                "description": "文件名(可选，会自动生成)",
                "default": ""
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "标签列表",
                "default": []
            },
            "append": {
                "type": "boolean",
                "description": "是否追加到现有文件",
                "default": False
            }
        },
        "required": ["content", "title"],
        "additionalProperties": False
    },
    "strict": True,
    "executor_default": "chat",
    "complexity": "simple",
    "arg_hint": "content为文档内容；title为标题；filename可选；tags为标签列表；append决定是否追加",
    "caller_snippet": "用于保存分析结果、调研报告、总结文档等。文件会保存在data/notes/目录中。"
}

def _get_notes_directory() -> Path:
    """Get or create notes directory."""
    project_root = Path(__file__).resolve().parent.parent.parent
    notes_dir = project_root / "data" / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    return notes_dir

def _generate_filename(title: str, filename: str = "") -> str:
    """Generate safe filename."""
    if filename:
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_filename.endswith('.md'):
            safe_filename += '.md'
        return safe_filename

    # Generate from title
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_title}_{timestamp}.md"

def _format_header(title: str, tags: List[str]) -> str:
    """Format markdown header."""
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

def run(content: str, title: str, filename: str = "", tags: List[str] = None, append: bool = False) -> Dict[str, Any]:
    """
    Execute markdown_writer tool.

    Args:
        content: Content to write
        title: Document title
        filename: Optional filename
        tags: List of tags
        append: Whether to append to existing file

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not content or not content.strip():
            return {"ok": False, "error": "内容不能为空"}

        if not title or not title.strip():
            return {"ok": False, "error": "标题不能为空"}

        content = content.strip()
        title = title.strip()
        tags = tags or []

        # Get notes directory
        notes_dir = _get_notes_directory()

        # Generate filename
        final_filename = _generate_filename(title, filename)
        file_path = notes_dir / final_filename

        logger.info(f"Writing markdown file: {file_path}")

        # Check if file exists for append mode
        if append and not file_path.exists():
            append = False  # File doesn't exist, create new

        if append:
            # Just append content with separator
            separator = f"\n\n---\n\n## 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            formatted_content = separator + _format_content(content)
        else:
            # Create new file with full formatting
            formatted_content = (
                _format_header(title, tags) +
                _format_content(content) +
                _format_footer({"original_filename": filename, "tags": tags})
            )

        # Save file
        file_info = _save_file(file_path, formatted_content, append)

        result = {
            "title": title,
            "file_info": file_info,
            "content_length": len(content),
            "formatted_length": len(formatted_content),
            "tags": tags,
            "append_mode": append
        }

        logger.info(f"Markdown file written: {len(formatted_content)} characters to {file_path}")
        return {"ok": True, "value": result}

    except Exception as e:
        error_msg = f"Markdown写入失败: {str(e)}"
        logger.error(f"Markdown writer error: {e}")
        return {"ok": False, "error": error_msg}
