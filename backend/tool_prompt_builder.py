"""
Tool Prompt Builder for the DigitalClone AI Assistant.

This module automatically generates tool prompts and function schemas
from the tool registry for use in LLM prompts and API calls.
"""

import json
from typing import Dict, Any, List

from tool_registry import registry


def build_tool_prompts() -> Dict[str, Any]:
    """
    Build comprehensive tool prompts from the tool registry.

    Returns:
        Dictionary containing:
        - tools: List of function definitions for API calls
        - tools_text: Formatted text description for prompts
        - tool_name_index: Mapping of tool names to descriptions
    """
    tools = []
    tools_text_parts = []
    tool_name_index = {}

    for tool_meta in registry.list_tools():
        # Build function definition for API
        function_def = {
            "type": "function",
            "function": {
                "name": tool_meta.name,
                "description": tool_meta.description,
                "parameters": tool_meta.parameters
            }
        }
        tools.append(function_def)

        # Build text description for prompts
        tool_text = f"""
### {tool_meta.name}
**描述**: {tool_meta.description}

**参数**:
{json.dumps(tool_meta.parameters, ensure_ascii=False, indent=2)}
"""
        tools_text_parts.append(tool_text)

        # Index for quick lookup
        tool_name_index[tool_meta.name] = {
            "description": tool_meta.description,
            "parameters": tool_meta.parameters
        }

    # Combine text descriptions
    tools_text = "\n".join(tools_text_parts)

    return {
        "tools": tools,
        "tools_text": tools_text,
        "tool_name_index": tool_name_index
    }


def get_system_prompt_with_tools(base_prompt: str, route: str) -> str:
    """
    Get system prompt with tools integrated.

    Args:
        base_prompt: Base system prompt text
        route: Route type ("chat" or "reasoner")

    Returns:
        Complete system prompt with tools
    """
    tool_prompts = build_tool_prompts()
    tools_text = tool_prompts["tools_text"]

    # Insert tools text into the base prompt
    complete_prompt = base_prompt.replace("{tools_text}", tools_text)

    return complete_prompt


def load_system_prompt(route: str) -> str:
    """
    Load the appropriate system prompt for the given route.

    Args:
        route: Route type ("chat" or "reasoner")

    Returns:
        Complete system prompt with tools
    """
    import os
    from pathlib import Path

    # Get the prompts directory
    current_dir = Path(__file__).parent.parent
    prompts_dir = current_dir / "prompts"

    if route == "reasoner":
        prompt_file = prompts_dir / "reasoner_system.txt"
    else:
        prompt_file = prompts_dir / "chat_system.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            base_prompt = f.read()

        return get_system_prompt_with_tools(base_prompt, route)

    except FileNotFoundError:
        # Fallback to simple prompt if file not found
        return f"You are a helpful AI assistant using the {route} model. Use available tools when appropriate."


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

    # Simple cache invalidation - check if tool count changed
    current_tool_count = len(registry.list_tools())

    if _cached_tool_prompts is None or len(_cached_tool_prompts.get("tools", [])) != current_tool_count:
        _cached_tool_prompts = build_tool_prompts()

    return _cached_tool_prompts
