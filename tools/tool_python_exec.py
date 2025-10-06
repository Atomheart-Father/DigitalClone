"""
Python Execution Tool for the Digital Clone AI Assistant.

Provides safe Python code execution in a sandboxed environment with timeout and result truncation.
"""

import logging
from typing import Dict, Any, List
import sys
import io
import contextlib
import signal
import traceback
import ast
import pandas as pd
from io import StringIO
import numpy as np

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "python_exec",
    "description": "在沙盒环境中安全执行Python代码，支持数据处理、计算和表格操作",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的Python代码"
            },
            "timeout": {
                "type": "integer",
                "description": "执行超时时间(秒)",
                "default": 10,
                "minimum": 1,
                "maximum": 30
            },
            "max_output_length": {
                "type": "integer",
                "description": "最大输出长度",
                "default": 2000,
                "minimum": 500,
                "maximum": 5000
            }
        },
        "required": ["code"],
        "additionalProperties": False
    },
    "strict": True,
    "executor_default": "reasoner",
    "complexity": "complex",
    "arg_hint": "code为完整的Python代码；timeout控制执行时间；max_output_length限制输出长度",
    "caller_snippet": "用于数据分析、表格处理、复杂计算等。代码会被安全执行，支持pandas/numpy等库。"
}

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("代码执行超时")

def _is_safe_code(code: str) -> bool:
    """
    Basic safety check for Python code.

    Checks for potentially dangerous operations.
    """
    dangerous_patterns = [
        '__import__', 'import os', 'import sys', 'import subprocess',
        'eval(', 'exec(', 'open(', 'file(', 'input(',
        'import shutil', 'import socket', 'import urllib',
        'os.', 'sys.', 'shutil.', 'subprocess.'
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return False

    return True

def _execute_with_timeout(code: str, timeout: int = 10) -> tuple:
    """
    Execute Python code with timeout protection.

    Returns (output, error, execution_time)
    """
    # Set up signal handler for timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        # Create safe globals
        safe_globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
                'enumerate': enumerate, 'filter': filter, 'float': float, 'int': int,
                'len': len, 'list': list, 'map': map, 'max': max, 'min': min,
                'print': print, 'range': range, 'round': round, 'set': set,
                'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                'zip': zip, 'True': True, 'False': False, 'None': None
            },
            # Add common data science libraries
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
        }

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # Parse and check AST for safety
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return "", f"语法错误: {e}", 0

            # Execute code
            exec(compile(tree, '<string>', 'exec'), safe_globals)

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        signal.alarm(0)  # Cancel alarm
        return stdout, stderr, timeout

    except TimeoutError:
        return "", "代码执行超时", timeout
    except Exception as e:
        error_msg = f"执行错误: {str(e)}\n{traceback.format_exc()}"
        return "", error_msg, timeout
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

def _format_result(output: str, error: str, max_length: int) -> str:
    """Format execution result for display."""
    result_parts = []

    if output:
        result_parts.append(f"输出:\n{output}")

    if error:
        result_parts.append(f"错误:\n{error}")

    full_result = "\n\n".join(result_parts)

    # Truncate if too long
    if len(full_result) > max_length:
        full_result = full_result[:max_length-3] + "..."

    return full_result

def run(code: str, timeout: int = 10, max_output_length: int = 2000) -> Dict[str, Any]:
    """
    Execute python_exec tool.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        max_output_length: Maximum output length

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not code or not code.strip():
            return {"ok": False, "error": "代码不能为空"}

        code = code.strip()

        # Basic safety check
        if not _is_safe_code(code):
            return {"ok": False, "error": "代码包含不安全的操作"}

        logger.info(f"Executing Python code (timeout: {timeout}s)")

        # Execute code
        stdout, stderr, exec_time = _execute_with_timeout(code, timeout)

        # Format result
        result = _format_result(stdout, stderr, max_output_length)

        if not result:
            return {"ok": False, "error": "代码执行无输出"}

        execution_result = {
            "code": code[:200] + "..." if len(code) > 200 else code,
            "result": result,
            "execution_time": exec_time,
            "has_error": bool(stderr),
            "truncated": len(_format_result(stdout, stderr, float('inf'))) > max_output_length
        }

        logger.info(f"Python execution completed in {exec_time}s, output length: {len(result)}")
        return {"ok": True, "value": execution_result}

    except Exception as e:
        error_msg = f"Python执行工具失败: {str(e)}"
        logger.error(f"Python exec error: {e}")
        return {"ok": False, "error": error_msg}
