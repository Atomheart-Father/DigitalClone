"""
Tool Prompt Builder for the DigitalClone AI Assistant.

This module automatically generates tool prompts and function schemas
from the tool registry for use in LLM prompts and API calls.
"""

import json
from typing import Dict, Any, List, Optional

# Conditional imports to support both relative and absolute imports
try:
    from .tool_registry import registry
except ImportError:
    from tool_registry import registry

# Strict tool whitelist for planning phases (no invention allowed)
ALLOWED_TOOLS = [
    "file_read", "web_search", "web_read", "rag_search",
    "markdown_writer", "python_exec", "calculator", "datetime", "tabular_qa"
]

def get_allowed_tools_whitelist() -> List[str]:
    """Get the strict whitelist of allowed tools."""
    return ALLOWED_TOOLS.copy()

def get_tools_whitelist_text() -> str:
    """Get formatted whitelist text for prompts."""
    return "[" + ",".join(f'"{tool}"' for tool in ALLOWED_TOOLS) + "]"


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


def build_phase1_draft_prompt(task_summary: str, known_params: str = "",
                             missing_params: str = "", constraints: str = "") -> str:
    """
    Build Phase-1 draft planning prompt.

    Args:
        task_summary: Task overview and context
        known_params: Currently known parameters
        missing_params: Missing parameters that need user input
        constraints: Budget/time/network constraints

    Returns:
        Complete Phase-1 prompt
    """
    from pathlib import Path
    import os

    # Get the prompts directory
    current_dir = Path(__file__).parent.parent
    prompt_file = current_dir / "prompts" / "phase1_draft.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Fill template
        prompt = template.replace("{task_summary}", task_summary)
        prompt = prompt.replace("{tools_whitelist}", get_tools_whitelist_text())
        prompt = prompt.replace("{known_params}", known_params or "无")
        prompt = prompt.replace("{missing_params}", missing_params or "无")
        prompt = prompt.replace("{constraints}", constraints or "无特别约束")

        return prompt

    except FileNotFoundError:
        return f"任务：{task_summary}\n请给出简要的执行步骤草案，使用允许的工具：{get_tools_whitelist_text()}"


def build_phase2_review_prompt(goal: str, facts: str, draft_points: str) -> str:
    """
    Build Phase-2 micro-review prompt for Reasoner.

    Args:
        goal: Task goal (≤100 chars)
        facts: Key facts (≤150 chars)
        draft_points: Draft key points (≤80 chars)

    Returns:
        Complete Phase-2 prompt (≤300 tokens)
    """
    from pathlib import Path

    current_dir = Path(__file__).parent.parent
    prompt_file = current_dir / "prompts" / "phase2_review.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Fill template with relaxed length limits to avoid truncation
        prompt = template.replace("{goal}", goal[:100])
        prompt = prompt.replace("{facts}", facts[:150])
        prompt = prompt.replace("{draft_points}", draft_points[:80])

        return prompt

    except FileNotFoundError:
        return f"目标:{goal[:100]}\n事实:{facts[:150]}\n草案要点:{draft_points[:80]}\n改进准则:只回'保留/微调/重排/加一步/删一步'之一，并给出10~30字理由"


def build_phase3_json_plan_prompt(task: str, context_summary: str = "",
                                 known_params: str = "", missing_params: str = "",
                                 constraints: str = "") -> str:
    """
    Build Phase-3 JSON planning prompt for Chat.

    Args:
        task: Task description
        context_summary: Context summary
        known_params: Known parameters
        missing_params: Missing parameters
        constraints: Constraints

    Returns:
        Complete Phase-3 prompt
    """
    from pathlib import Path

    current_dir = Path(__file__).parent.parent
    prompt_file = current_dir / "prompts" / "phase3_plan_json.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Get tool descriptions for the prompt
        tool_prompts = get_cached_tool_prompts()
        tools_descriptions = tool_prompts.get("tools_text_chat", "无可用工具")


        # Fill template
        prompt = template.replace("{task}", task)
        prompt = template.replace("{context_summary}", context_summary or "无")
        prompt = template.replace("{known_params}", known_params or "{}")
        prompt = prompt.replace("{missing_params}", missing_params or "[]")
        prompt = prompt.replace("{tools_whitelist}", get_tools_whitelist_text())
        prompt = prompt.replace("{tools_descriptions}", tools_descriptions)
        prompt = prompt.replace("{constraints}", constraints or "无")

        return prompt

    except FileNotFoundError:
        return f'你只输出json。Schema: {{"strategy":"serial|parallel","todos":[{{"id":"T1","tool":"file_read","params":{{}},"depends_on":[],"why":"","cost":1}}],"ask_user":{{"needed":false,"missing_params":[],"ask_message":""}}}}\n允许工具:{get_tools_whitelist_text()}\n任务:{task}'


def build_tool_execution_prompt(task: str, current_state: str, todo_item: str, tool_name: str) -> str:
    """
    Build tool execution prompt for Function Calling.

    Args:
        task: Overall task
        current_state: Current execution state
        todo_item: Todo item to execute

    Returns:
        Tool execution prompt
    """
    from pathlib import Path

    current_dir = Path(__file__).parent.parent
    prompt_file = current_dir / "prompts" / "tool_execution.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Get tool descriptions for the specific tool
        tool_prompts = get_cached_tool_prompts()
        tool_descriptions = tool_prompts.get("tool_name_index", {})
        specific_tool_info = tool_descriptions.get(tool_name, {})

        # Build tool description
        tool_desc = f"{tool_name}：{specific_tool_info.get('description', '工具描述')}"
        if 'arg_hint' in specific_tool_info:
            tool_desc += f"\n参数提示：{specific_tool_info['arg_hint']}"

        # Fill template
        prompt = template.replace("{task}", task)
        prompt = template.replace("{current_state}", current_state or "初始状态")
        prompt = template.replace("{todo_item}", todo_item)
        prompt = template.replace("{tools_whitelist}", get_tools_whitelist_text())
        prompt = template.replace("{tools_descriptions}", tool_desc)

        return prompt

    except FileNotFoundError:
        return f"执行任务：{task}\n当前状态：{current_state}\n执行项：{todo_item}\n只使用允许工具：{get_tools_whitelist_text()}"


def build_reflective_replanning_prompt(goal: str, new_facts: str, current_plan: str) -> str:
    """
    Build reflective replanning prompt for Reasoner.

    Args:
        goal: Current goal
        new_facts: New information summary
        current_plan: Current plan summary

    Returns:
        Reflective replanning prompt (≤200 tokens)
    """
    from pathlib import Path

    current_dir = Path(__file__).parent.parent
    prompt_file = current_dir / "prompts" / "reflective_replanning.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Fill template with length limits
        prompt = template.replace("{goal}", goal[:40])
        prompt = template.replace("{new_facts}", new_facts[:100])
        prompt = template.replace("{current_plan}", current_plan[:40])

        return prompt

    except FileNotFoundError:
        return f"目标:{goal[:40]}\n新事实要点:{new_facts[:100]}\n现计划要点:{current_plan[:40]}\n问: 是否需要重排/补步/删步？只回'是/否+10~30字理由'"


def build_ask_user_degradation_prompt(task: str, current_params: str,
                                     missing_params: str, attempt_count: int) -> str:
    """
    Build AskUser degradation prompt for fallback handling.

    Args:
        task: Current task
        current_params: Currently available parameters
        missing_params: Still missing parameters
        attempt_count: Number of previous attempts

    Returns:
        Degradation handling prompt
    """
    from pathlib import Path

    current_dir = Path(__file__).parent.parent
    prompt_file = current_dir / "prompts" / "ask_user_degradation.txt"

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Fill template
        prompt = template.replace("{task}", task)
        prompt = template.replace("{current_params}", current_params or "{}")
        prompt = template.replace("{missing_params}", missing_params or "[]")
        prompt = template.replace("{attempt_count}", str(attempt_count))

        return prompt

    except FileNotFoundError:
        return f"任务：{task}\n当前参数：{current_params}\n缺失参数：{missing_params}\n已尝试{attempt_count}次，请决定是否可以安全推断默认值，或需要明确询问用户原因。"
