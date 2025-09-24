"""
Tool Registry for the Digital Clone AI Assistant.

This module provides dynamic tool discovery, registration, and execution
with JSON Schema validation and unified error handling.
"""

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import jsonschema

# Conditional imports to support both relative and absolute imports
try:
    from .message_types import ToolMeta, ToolExecutionResult
except ImportError:
    from message_types import ToolMeta, ToolExecutionResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tools dynamically."""

    def __init__(self, tools_dir: str = None):
        if tools_dir is None:
            # Auto-detect tools directory relative to this file
            current_dir = Path(__file__).parent
            self.tools_dir = current_dir / "tools"
        else:
            self.tools_dir = Path(tools_dir)

        self._tools: Dict[str, Dict[str, Any]] = {}
        self._load_tools()

    def _load_tools(self):
        """Dynamically load all tools from the tools directory."""
        if not self.tools_dir.exists():
            logger.warning(f"Tools directory not found: {self.tools_dir}")
            return

        logger.info(f"Loading tools from {self.tools_dir}")

        for tool_file in self.tools_dir.glob("tool_*.py"):
            try:
                self._load_tool_from_file(tool_file)
            except Exception as e:
                logger.error(f"Failed to load tool from {tool_file}: {e}")
                continue

        logger.info(f"Loaded {len(self._tools)} tools: {list(self._tools.keys())}")

    def _load_tool_from_file(self, tool_file: Path):
        """Load a single tool from a Python file."""
        try:
            # Execute the file in a controlled environment
            spec = importlib.util.spec_from_file_location("tool_module", tool_file)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load spec for {tool_file}")

            module = importlib.util.module_from_spec(spec)

            # Execute the module
            spec.loader.exec_module(module)

            # Check for required attributes
            if not hasattr(module, 'TOOL_META'):
                raise ValueError(f"Tool {tool_file.name} missing TOOL_META")

            if not hasattr(module, 'run'):
                raise ValueError(f"Tool {tool_file.name} missing run function")

            tool_meta = module.TOOL_META
            run_func = module.run

            # Validate TOOL_META structure
            required_keys = ['name', 'description', 'parameters']
            for key in required_keys:
                if key not in tool_meta:
                    raise ValueError(f"TOOL_META missing required key: {key}")

            tool_name = tool_meta['name']

            # Store the tool
            self._tools[tool_name] = {
                'meta': ToolMeta(**tool_meta),
                'run_func': run_func,
                'module': module
            }

            logger.debug(f"Loaded tool: {tool_name}")

        except Exception as e:
            raise ValueError(f"Failed to load tool from {tool_file}: {e}")

    def list_tools(self) -> List[ToolMeta]:
        """List all registered tools."""
        return [tool['meta'] for tool in self._tools.values()]

    def get_tool_meta(self, name: str) -> Optional[ToolMeta]:
        """Get metadata for a specific tool."""
        tool = self._tools.get(name)
        return tool['meta'] if tool else None

    def get_functions_schema(self) -> List[Dict[str, Any]]:
        """Get function schemas for LLM function calling."""
        functions = []
        for tool in self._tools.values():
            meta = tool['meta']
            functions.append({
                "name": meta.name,
                "description": meta.description,
                "parameters": meta.parameters
            })
        return functions

    def execute(self, name: str, **kwargs) -> ToolExecutionResult:
        """
        Execute a tool with given parameters.

        Args:
            name: Tool name to execute
            **kwargs: Parameters for the tool

        Returns:
            ToolExecutionResult with ok/value/error fields
        """
        try:
            logger.info(f"Executing tool: {name} with params: {kwargs}")

            # Check if tool exists
            if name not in self._tools:
                return ToolExecutionResult(
                    ok=False,
                    error=f"工具 '{name}' 未找到"
                )

            tool = self._tools[name]
            run_func = tool['run_func']
            meta = tool['meta']

            # Validate parameters against schema
            try:
                jsonschema.validate(instance=kwargs, schema=meta.parameters)
            except jsonschema.ValidationError as e:
                return ToolExecutionResult(
                    ok=False,
                    error=f"参数验证失败: {e.message}"
                )

            # Execute the tool
            result = run_func(**kwargs)

            # Ensure result format
            if not isinstance(result, dict):
                return ToolExecutionResult(
                    ok=False,
                    error="工具返回格式错误"
                )

            if 'ok' not in result:
                return ToolExecutionResult(
                    ok=False,
                    error="工具返回缺少 'ok' 字段"
                )

            # Convert to ToolExecutionResult
            execution_result = ToolExecutionResult(
                ok=result['ok'],
                value=result.get('value'),
                error=result.get('error')
            )

            logger.info(f"Tool {name} execution result: ok={execution_result.ok}")
            return execution_result

        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            logger.error(f"Tool execution error for {name}: {e}")
            return ToolExecutionResult(
                ok=False,
                error=error_msg
            )

    def reload_tools(self):
        """Reload all tools (useful for development)."""
        self._tools.clear()
        self._load_tools()
        logger.info("Tools reloaded")


# Global registry instance
registry = ToolRegistry()
