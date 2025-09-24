"""
File Read Tool for the Digital Clone AI Assistant.

Reads and extracts text content from various file formats (txt, md, pdf, etc.).
"""

import os
import logging
from typing import Dict, Any, List
from pathlib import Path
import chardet

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "file_read",
    "description": "读取本地文件的文本内容，支持txt、md、pdf等多种格式",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径，支持相对路径和绝对路径"
            },
            "max_length": {
                "type": "integer",
                "description": "最大返回字符数",
                "default": 8000,
                "minimum": 1000,
                "maximum": 20000
            },
            "encoding": {
                "type": "string",
                "description": "文件编码(可选，会自动检测)",
                "default": "auto"
            }
        },
        "required": ["file_path"],
        "additionalProperties": False
    },
    "strict": True,
    "executor_default": "chat",
    "complexity": "simple",
    "arg_hint": "file_path为文件路径；max_length控制返回内容长度；encoding通常自动检测",
    "caller_snippet": "用于读取本地文档、笔记、配置文件等。支持项目内data目录的文件。"
}

def _get_project_root() -> Path:
    """Get project root directory."""
    # Assume we're in backend/tools/, go up two levels
    current = Path(__file__).resolve()
    return current.parent.parent.parent

def _resolve_file_path(file_path: str) -> Path:
    """
    Resolve file path safely.

    Only allows access to project directory and common data directories.
    """
    project_root = _get_project_root()
    requested_path = Path(file_path).resolve()

    # Check if path is within allowed directories
    allowed_dirs = [
        project_root,
        project_root / "data",
        project_root / "docs",
        Path.home() / "Documents",  # Allow user's Documents
        Path.home() / "Desktop",    # Allow user's Desktop
    ]

    # Check if requested path is within any allowed directory
    for allowed_dir in allowed_dirs:
        try:
            requested_path.relative_to(allowed_dir)
            return requested_path
        except ValueError:
            continue

    # If not within allowed directories, reject
    raise ValueError(f"访问被拒绝: 文件路径超出允许范围: {file_path}")

def _detect_encoding(file_path: Path) -> str:
    """Detect file encoding."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)

            # Use detected encoding if confidence is high enough
            if confidence > 0.7 and encoding:
                return encoding.lower()

    except Exception:
        pass

    return 'utf-8'  # Default fallback

def _read_text_file(file_path: Path, encoding: str = 'auto', max_length: int = 8000) -> str:
    """Read text file with encoding detection."""
    if encoding == 'auto':
        encoding = _detect_encoding(file_path)

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read(max_length + 1000)  # Read a bit more to check truncation

        # Truncate if needed
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[内容已截断...]"

        return content

    except UnicodeDecodeError as e:
        # Try alternative encodings
        fallback_encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1']
        for fallback_encoding in fallback_encodings:
            if fallback_encoding != encoding:
                try:
                    with open(file_path, 'r', encoding=fallback_encoding) as f:
                        content = f.read(max_length + 1000)
                    if len(content) > max_length:
                        content = content[:max_length] + "\n\n[内容已截断...]"
                    return content
                except UnicodeDecodeError:
                    continue

        raise ValueError(f"无法读取文件编码: {e}")

def _read_pdf_file(file_path: Path, max_length: int = 8000) -> str:
    """Read PDF file and extract text."""
    try:
        # Try to import PyPDF2 or pdfplumber
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
            except ImportError:
                raise ImportError("需要安装PyPDF2或pdfplumber来读取PDF文件")

        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[内容已截断...]"

        return text.strip()

    except Exception as e:
        raise ValueError(f"PDF读取失败: {e}")

def _get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file information."""
    stat = file_path.stat()
    return {
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "extension": file_path.suffix.lower()
    }

def run(file_path: str, max_length: int = 8000, encoding: str = "auto") -> Dict[str, Any]:
    """
    Execute file_read tool.

    Args:
        file_path: Path to the file to read
        max_length: Maximum character length to return
        encoding: File encoding (auto-detect if not specified)

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not file_path or not file_path.strip():
            return {"ok": False, "error": "文件路径不能为空"}

        file_path = file_path.strip()

        # Resolve and validate file path
        resolved_path = _resolve_file_path(file_path)

        if not resolved_path.exists():
            return {"ok": False, "error": f"文件不存在: {file_path}"}

        if not resolved_path.is_file():
            return {"ok": False, "error": f"路径不是文件: {file_path}"}

        logger.info(f"Reading file: {resolved_path}")

        # Get file info
        file_info = _get_file_info(resolved_path)

        # Check file size (limit to 10MB)
        if file_info["size"] > 10 * 1024 * 1024:
            return {"ok": False, "error": "文件过大(超过10MB)"}

        # Read based on file type
        extension = file_info["extension"]

        if extension in ['.txt', '.md', '.py', '.json', '.yaml', '.yml', '.csv', '.log']:
            content = _read_text_file(resolved_path, encoding, max_length)
        elif extension == '.pdf':
            content = _read_pdf_file(resolved_path, max_length)
        else:
            # Try to read as text anyway
            try:
                content = _read_text_file(resolved_path, encoding, max_length)
            except Exception:
                return {"ok": False, "error": f"不支持的文件格式: {extension}"}

        if not content or not content.strip():
            return {"ok": False, "error": "文件为空或无法读取内容"}

        result = {
            "file_path": str(resolved_path),
            "file_info": file_info,
            "content": content,
            "encoding_used": encoding if encoding != "auto" else _detect_encoding(resolved_path),
            "truncated": len(content) > max_length
        }

        logger.info(f"File read completed: {len(content)} characters from {extension} file")
        return {"ok": True, "value": result}

    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        error_msg = f"文件读取失败: {str(e)}"
        logger.error(f"File read error for {file_path}: {e}")
        return {"ok": False, "error": error_msg}
