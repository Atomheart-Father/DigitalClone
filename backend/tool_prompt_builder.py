"""
Tool Prompt Builder for the DigitalClone AI Assistant.

This module automatically generates tool prompts and function schemas
from the tool registry for use in LLM prompts and API calls.
"""

import json
from typing import Dict, Any, List

# Conditional imports to support both relative and absolute imports
try:
    from .tool_registry import registry
except ImportError:
    from tool_registry import registry


def build_tool_prompts() -> Dict[str, Any]:
    """
    Build comprehensive tool prompts from the tool registry.

    Returns:
        Dictionary containing:
        - tools: List of function definitions for API calls
        - tools_text_chat: Formatted text for chat executor
        - tools_text_reasoner: Formatted text for reasoner executor
        - tool_name_index: Mapping of tool names to metadata
    """
    tools = []
    tools_text_chat_parts = []
    tools_text_reasoner_parts = []
    tool_name_index = {}

    for tool_meta in registry.list_tools():
        # Build function definition for API with strict schema
        function_def = {
            "type": "function",
            "function": {
                "name": tool_meta.name,
                "description": tool_meta.description,
                "strict": True,  # Enable strict JSON Schema validation
                "parameters": tool_meta.parameters
            }
        }
        tools.append(function_def)

        # Build text description for chat executor (simple tools)
        if tool_meta.executor_default in ["chat", "auto"] and tool_meta.complexity == "simple":
            chat_tool_text = f"""
### {tool_meta.name}
**描述**: {tool_meta.description}

**参数提示**: {tool_meta.arg_hint}

**使用说明**: {tool_meta.caller_snippet}

**参数规范**:
{json.dumps(tool_meta.parameters, ensure_ascii=False, indent=2)}
"""
            tools_text_chat_parts.append(chat_tool_text)

        # Build text description for reasoner executor (complex tools)
        if tool_meta.executor_default == "reasoner" or tool_meta.complexity == "complex":
            reasoner_tool_text = f"""
### {tool_meta.name}
**描述**: {tool_meta.description}

**复杂度**: {tool_meta.complexity}
**参数提示**: {tool_meta.arg_hint}
**使用说明**: {tool_meta.caller_snippet}

**参数规范**:
{json.dumps(tool_meta.parameters, ensure_ascii=False, indent=2)}
"""
            tools_text_reasoner_parts.append(reasoner_tool_text)

        # Index for quick lookup - ensure all values are basic types (no ToolMeta objects)
        tool_name_index[tool_meta.name] = {
            "description": str(tool_meta.description),
            "parameters": dict(tool_meta.parameters),  # Deep copy to avoid references
            "executor_default": str(tool_meta.executor_default),
            "complexity": str(tool_meta.complexity),
            "arg_hint": str(tool_meta.arg_hint or ""),
            "caller_snippet": str(tool_meta.caller_snippet or ""),
            "strict": bool(tool_meta.strict)
        }

    # Combine text descriptions
    tools_text_chat = "\n".join(tools_text_chat_parts) if tools_text_chat_parts else "当前执行者无可用工具。"
    tools_text_reasoner = "\n".join(tools_text_reasoner_parts) if tools_text_reasoner_parts else "当前执行者无可用工具。"

    return {
        "tools": tools,
        "tools_text_chat": tools_text_chat,
        "tools_text_reasoner": tools_text_reasoner,
        "tool_name_index": tool_name_index
    }


def get_system_prompt_with_tools(base_prompt: str, route: str, executor: str = "chat") -> str:
    """
    Get system prompt with tools integrated for specific executor.

    Args:
        base_prompt: Base system prompt text
        route: Route type ("chat" or "reasoner")
        executor: Executor type ("chat" or "reasoner")

    Returns:
        Complete system prompt with tools
    """
    tool_prompts = build_tool_prompts()
    tools_text = tool_prompts[f"tools_text_{executor}"]

    # Insert tools text into the base prompt
    complete_prompt = base_prompt.replace("{tools_text}", tools_text)

    return complete_prompt


def load_system_prompt(route: str, executor: str = "auto") -> str:
    """
    Load the appropriate system prompt for the given route and executor.

    Args:
        route: Route type ("chat" or "reasoner")
        executor: Executor type ("auto", "chat", or "reasoner")

    Returns:
        Complete system prompt with tools
    """
    import os
    from pathlib import Path

    # Get the prompts directory
    current_dir = Path(__file__).parent.parent
    prompts_dir = current_dir / "prompts"

    # Resolve executor
    if executor == "auto":
        executor = "chat" if route == "chat" else "reasoner"

    # Select prompt file
    if route == "reasoner":
        if executor == "reasoner":
            prompt_file = prompts_dir / "reasoner_tool_caller.txt"
        else:
            prompt_file = prompts_dir / "reasoner_planner.txt"
    else:
        if executor == "reasoner":
            prompt_file = prompts_dir / "reasoner_tool_caller.txt"
        else:
            prompt_file = prompts_dir / "chat_tool_caller.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            base_prompt = f.read()

        return get_system_prompt_with_tools(base_prompt, route, executor)

    except FileNotFoundError:
        # Fallback to simple prompt if file not found
        return f"You are a helpful AI assistant using the {route} model with {executor} execution. Use available tools when appropriate."


def validate_tool_schemas() -> List[str]:
    """
    Validate all tool schemas using JSON Schema validator.

    Returns:
        List of validation errors (empty if all valid)
    """
    import jsonschema

    errors = []

    for tool_meta in registry.list_tools():
        try:
            # Validate the schema itself
            jsonschema.Draft7Validator.check_schema(tool_meta.parameters)

            # Additional validation for our requirements
            if "type" not in tool_meta.parameters:
                errors.append(f"Tool {tool_meta.name}: missing 'type' in parameters")

            if tool_meta.parameters.get("type") != "object":
                errors.append(f"Tool {tool_meta.name}: parameters type must be 'object'")

            if "properties" not in tool_meta.parameters:
                errors.append(f"Tool {tool_meta.name}: missing 'properties' in parameters")

            # Check required field
            if "required" not in tool_meta.parameters:
                errors.append(f"Tool {tool_meta.name}: missing 'required' field")

        except jsonschema.SchemaError as e:
            errors.append(f"Tool {tool_meta.name}: invalid JSON schema - {e}")
        except Exception as e:
            errors.append(f"Tool {tool_meta.name}: validation error - {e}")

    return errors


# Global cache for tool prompts
_cached_tool_prompts = None


def get_cached_tool_prompts() -> Dict[str, Any]:
    """
    Get cached tool prompts, rebuilding if registry changed.

    Returns:
        Current tool prompts
    """
    global _cached_tool_prompts

    # Check if cache needs rebuild
    current_tool_count = len(registry.list_tools())
    cache_tool_count = len(_cached_tool_prompts.get("tools", [])) if _cached_tool_prompts else 0

    if _cached_tool_prompts is None or cache_tool_count != current_tool_count:
        _cached_tool_prompts = build_tool_prompts()

    return _cached_tool_prompts
