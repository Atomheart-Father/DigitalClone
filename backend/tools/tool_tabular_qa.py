"""
Tabular QA Tool for the Digital Clone AI Assistant.

Provides Q&A functionality for tabular data (CSV, Excel) using pandas.
"""

import os
import logging
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "tabular_qa",
    "description": "对表格数据进行查询和分析，支持CSV/Excel文件的数据选择、过滤和汇总",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "表格文件路径(CSV或Excel)"
            },
            "query": {
                "type": "string",
                "description": "查询描述或pandas操作代码"
            },
            "max_rows": {
                "type": "integer",
                "description": "返回的最大行数",
                "default": 50,
                "minimum": 5,
                "maximum": 200
            }
        },
        "required": ["file_path", "query"],
        "additionalProperties": False
    },
    "strict": True,
    "executor_default": "reasoner",
    "complexity": "complex",
    "arg_hint": "file_path为表格文件路径；query为查询描述或pandas代码；max_rows限制返回行数",
    "caller_snippet": "用于分析表格数据、筛选记录、计算统计信息等。支持自然语言查询和pandas代码。"
}

def _get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent

def _resolve_file_path(file_path: str) -> Path:
    """Resolve and validate file path."""
    project_root = _get_project_root()
    requested_path = Path(file_path).resolve()

    # Check if path is within allowed directories
    allowed_dirs = [
        project_root,
        project_root / "data",
        Path.home() / "Documents",
        Path.home() / "Desktop",
    ]

    for allowed_dir in allowed_dirs:
        try:
            requested_path.relative_to(allowed_dir)
            return requested_path
        except ValueError:
            continue

    raise ValueError(f"访问被拒绝: 文件路径超出允许范围: {file_path}")

def _load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load DataFrame from file."""
    extension = file_path.suffix.lower()

    try:
        if extension == '.csv':
            df = pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {extension}")

        return df

    except Exception as e:
        raise ValueError(f"文件加载失败: {e}")

def _execute_query(df: pd.DataFrame, query: str, max_rows: int) -> Dict[str, Any]:
    """
    Execute query on DataFrame.

    Supports both natural language queries and pandas code snippets.
    """
    query = query.strip().lower()

    try:
        result_df = None
        operation_desc = ""

        # Handle common natural language queries
        if any(word in query for word in ['显示', '查看', '看', 'show']):
            if '前' in query or 'first' in query:
                n = 5  # Default first 5
                for word in query.split():
                    if word.isdigit():
                        n = min(int(word), max_rows)
                        break
                result_df = df.head(n)
                operation_desc = f"显示前{n}行数据"
            elif '后' in query or 'last' in query:
                n = 5
                for word in query.split():
                    if word.isdigit():
                        n = min(int(word), max_rows)
                        break
                result_df = df.tail(n)
                operation_desc = f"显示后{n}行数据"
            else:
                result_df = df.head(min(max_rows, 10))
                operation_desc = "显示数据预览"

        elif any(word in query for word in ['统计', '汇总', 'summary', 'describe']):
            # Get basic statistics
            desc = df.describe(include='all')
            result_df = desc
            operation_desc = "数据统计汇总"

        elif any(word in query for word in ['列', 'columns', '字段']):
            # Show column information
            col_info = pd.DataFrame({
                '列名': df.columns,
                '数据类型': df.dtypes.astype(str),
                '非空值': df.count(),
                '唯一值': df.nunique()
            })
            result_df = col_info
            operation_desc = "列信息统计"

        elif '空值' in query or 'null' in query or 'na' in query:
            # Show null value information
            null_info = pd.DataFrame({
                '列名': df.columns,
                '空值数量': df.isnull().sum(),
                '空值比例': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%'
            })
            result_df = null_info
            operation_desc = "空值统计"

        else:
            # Try to execute as pandas code snippet
            # Create safe local environment
            local_vars = {'df': df, 'pd': pd, 'np': np}

            # Simple pandas operations
            if query.startswith('df.'):
                result_df = eval(query, {"__builtins__": {}}, local_vars)
                operation_desc = f"执行操作: {query}"
            else:
                # Default to showing head
                result_df = df.head(min(max_rows, 10))
                operation_desc = "显示数据预览"

        # Ensure result is a DataFrame
        if not isinstance(result_df, pd.DataFrame):
            if isinstance(result_df, pd.Series):
                result_df = result_df.to_frame()
            else:
                result_df = pd.DataFrame({'result': [result_df]})

        # Limit rows
        if len(result_df) > max_rows:
            result_df = result_df.head(max_rows)

        return {
            'operation': operation_desc,
            'shape': result_df.shape,
            'columns': list(result_df.columns),
            'data': result_df.to_dict('records'),
            'summary': {
                'total_rows': len(result_df),
                'total_columns': len(result_df.columns),
                'memory_usage': result_df.memory_usage(deep=True).sum()
            }
        }

    except Exception as e:
        raise ValueError(f"查询执行失败: {e}")

def run(file_path: str, query: str, max_rows: int = 50) -> Dict[str, Any]:
    """
    Execute tabular_qa tool.

    Args:
        file_path: Path to tabular file (CSV/Excel)
        query: Query description or pandas operation
        max_rows: Maximum rows to return

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not file_path or not file_path.strip():
            return {"ok": False, "error": "文件路径不能为空"}

        if not query or not query.strip():
            return {"ok": False, "error": "查询不能为空"}

        file_path = file_path.strip()
        query = query.strip()

        # Resolve file path
        resolved_path = _resolve_file_path(file_path)

        if not resolved_path.exists():
            return {"ok": False, "error": f"文件不存在: {file_path}"}

        logger.info(f"Processing tabular data: {resolved_path}")

        # Load DataFrame
        df = _load_dataframe(resolved_path)

        if df.empty:
            return {"ok": False, "error": "表格文件为空"}

        logger.info(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

        # Execute query
        result = _execute_query(df, query, max_rows)

        # Add file info
        result['file_info'] = {
            'path': str(resolved_path),
            'format': resolved_path.suffix.lower(),
            'original_shape': df.shape
        }

        logger.info(f"Tabular QA completed: {result['operation']}")
        return {"ok": True, "value": result}

    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        error_msg = f"表格查询失败: {str(e)}"
        logger.error(f"Tabular QA error for {file_path}: {e}")
        return {"ok": False, "error": error_msg}
