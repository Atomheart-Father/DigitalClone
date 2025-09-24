"""
Node definitions for the DigitalClone AI Assistant LangGraph.

This module contains all the node functions that make up the execution graph.
"""

import logging
from typing import Dict, Any, List

import sys
import os
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .state import AgentState, Route
# Conditional imports to support both relative and absolute imports
try:
    from backend.message_types import Message, Role, RouteDecision, ToolCall
    from backend.agent_core import AgentRouter
    from backend.llm_interface import create_llm_client
    from backend.tool_registry import registry
    from backend.tool_prompt_builder import build_tool_prompts
    from backend.config import config
except ImportError:
    # Fallback to relative imports if absolute imports fail
    from message_types import Message, Role, RouteDecision, ToolCall
    from agent_core import AgentRouter
    from llm_interface import create_llm_client
    from tool_registry import registry
    from tool_prompt_builder import build_tool_prompts
    from config import config

logger = logging.getLogger(__name__)


def user_input_node(state: AgentState) -> Dict[str, Any]:
    """
    User input processing node.

    This node receives user input and prepares the initial state.
    In our implementation, this is handled by create_initial_state,
    so this node mainly serves as an entry point.
    """
    logger.info(f"Processing user input: {state['messages'][-1].content[:50]}...")

    # Update execution path
    state["execution_path"].append("user_input")
    state["current_node"] = "user_input"

    # Return unchanged state - routing will happen next
    return state


def decide_route_node(state: AgentState) -> Dict[str, Any]:
    """
    Route decision node.

    Decides whether to use chat or reasoner model based on the input.
    """
    logger.info("Making routing decision...")

    if not state["messages"]:
        raise ValueError("No messages in state")

    user_message = None
    for msg in reversed(state["messages"]):
        if msg.role == Role.USER:
            user_message = msg
            break

    if not user_message:
        raise ValueError("No user message found")

    # Use existing router logic
    router = AgentRouter()
    route_decision = router.route(user_message.content, None)  # Simplified context

    # Map to our Route enum
    if route_decision.engine == "reasoner":
        route = Route.REASONER
    else:
        route = Route.CHAT

    logger.info(f"Routing decision: {route.value} (reason: {route_decision.reason})")

    # Update state
    state["route"] = route
    state["route_decision"] = route_decision
    state["execution_path"].append("decide_route")
    state["current_node"] = "decide_route"

    return state


def model_call_node(state: AgentState) -> Dict[str, Any]:
    """
    Model call node.

    Calls the appropriate LLM (chat or reasoner) with tools and system prompts.
    """
    logger.info(f"Making model call with route: {state['route'].value}")

    route = state["route"]
    messages = state["messages"]

    # Load appropriate system prompt
    try:
        from backend.tool_prompt_builder import load_system_prompt
    except ImportError:
        from tool_prompt_builder import load_system_prompt
    system_prompt = load_system_prompt(route.value)

    # Select appropriate LLM client
    if route == Route.REASONER:
        llm_client = create_llm_client("reasoner")
    else:
        llm_client = create_llm_client("chat")

    # Build tool prompts
    tool_prompts = build_tool_prompts()
    functions = tool_prompts["tools"]

    # For now, we'll handle streaming separately in CLI
    # This node focuses on the core logic
    response = llm_client.generate(
        messages,
        functions=functions,
        system_prompt=system_prompt,
        stream=False
    )

    # Handle response
    if response.tool_calls:
        # Has tool calls - store for execution
        tool_call = response.tool_calls[0]
        state["pending_tool_call"] = {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "id": getattr(tool_call, 'id', None)  # May not have ID yet
        }
        logger.info(f"Tool call detected: {tool_call.name}")

        # Add assistant message to conversation
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content,
            tool_call=tool_call
        )
        state["messages"].append(assistant_message)

    elif _needs_user_clarification(response.content):
        # Needs user clarification
        state["awaiting_user"] = True
        state["user_input_buffer"] = None

        # Add assistant message
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content
        )
        state["messages"].append(assistant_message)

        logger.info("AskUser clarification needed")

    else:
        # Final answer
        state["final_answer"] = response.content
        state["should_end"] = True

        # Add assistant message
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content
        )
        state["messages"].append(assistant_message)

        logger.info("Final answer generated")

    state["execution_path"].append("model_call")
    state["current_node"] = "model_call"

    return state


def tool_exec_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool execution node.

    Executes pending tool calls and adds results back to conversation.
    """
    logger.info("Executing tool...")

    if not state["pending_tool_call"]:
        raise ValueError("No pending tool call in state")

    tool_call = state["pending_tool_call"]
    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]

    logger.info(f"Executing tool: {tool_name} with args: {arguments}")

    # Execute tool
    tool_result = registry.execute(tool_name, **arguments)

    # Create tool message
    if tool_result.ok:
        content = f"工具执行成功: {tool_result.value}"
    else:
        content = f"工具执行失败: {tool_result.error}"

    tool_message = Message(
        role=Role.TOOL,
        content=content,
        tool_result=ToolCall(
            name=tool_name,
            arguments=arguments
        )
    )

    # Add to conversation and clear pending call
    state["messages"].append(tool_message)
    state["pending_tool_call"] = None
    state["tool_call_count"] += 1

    logger.info(f"Tool execution completed: {tool_name}")

    state["execution_path"].append("tool_exec")
    state["current_node"] = "tool_exec"

    return state


def need_user_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if user clarification is needed.

    This is a conditional node that determines the flow.
    """
    needs_user = state["awaiting_user"]
    logger.info(f"User clarification needed: {needs_user}")

    state["execution_path"].append("need_user")
    state["current_node"] = "need_user"

    return state


def ask_user_interrupt_node(state: AgentState) -> Dict[str, Any]:
    """
    Human-in-the-loop interrupt node.

    This node will be interrupted and resumed after user input.
    """
    logger.info("Entering AskUser interrupt mode")

    # In LangGraph, this would normally interrupt execution
    # For our CLI implementation, we'll handle this in the CLI layer

    state["execution_path"].append("ask_user_interrupt")
    state["current_node"] = "ask_user_interrupt"

    return state


def end_node(state: AgentState) -> Dict[str, Any]:
    """
    End node for successful completion.
    """
    logger.info("Conversation completed successfully")

    state["execution_path"].append("end")
    state["current_node"] = "end"
    state["should_end"] = True

    return state


# ===== PLANNER NODES =====

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify user intent and choose execution pipeline.

    Routes to: chat, planner, auto_rag
    """
    logger.info("Classifying user intent...")

    if not state["messages"]:
        raise ValueError("No messages in state")

    user_message = None
    for msg in reversed(state["messages"]):
        if msg.role.value == "user":
            user_message = msg
            break

    if not user_message:
        raise ValueError("No user message found")

    user_input = user_message.content.lower()

    # Check for explicit auto_rag triggers
    if any(keyword in user_input for keyword in ['自动扩充', '整理对话', '总结对话', 'auto_rag']):
        route = Route.AUTO_RAG
        reason = "用户明确请求自动知识扩充"
    # Check for planning keywords
    elif any(keyword in user_input for keyword in [
        '计划', '规划', '制定', '多步骤', '调研', '方案', '评估',
        '对比', '流程', '依赖', '阶段', '项目', '任务分解'
    ]) or len(user_input) > 100:  # Long inputs likely need planning
        route = Route.PLANNER
        reason = "检测到复杂规划任务特征"
    else:
        route = Route.CHAT
        reason = "简单对话任务，使用chat模型"

    logger.info(f"Intent classification: {route.value} ({reason})")

    state["route"] = route
    state["execution_path"].append("classify_intent")
    state["current_node"] = "classify_intent"

    return state


def sufficiency_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if we have sufficient information to proceed with planning.

    This is a simplified version - in production, you'd use more sophisticated analysis.
    """
    logger.info("Checking information sufficiency...")

    # For now, assume we have enough information
    # In production, this would analyze the plan and check for missing prerequisites
    state["sufficiency"] = "enough"
    state["execution_path"].append("sufficiency_check")
    state["current_node"] = "sufficiency_check"

    return state


def planner_generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a structured plan using the reasoner model.
    """
    logger.info("Generating structured plan...")

    if not state["messages"]:
        raise ValueError("No messages in state")

    # Get the user request
    user_request = ""
    for msg in state["messages"]:
        if msg.role.value == "user":
            user_request = msg.content
            break

    # Use reasoner to generate plan
    llm_client = create_llm_client("reasoner")

    system_prompt = """你是一个专业的项目规划师。用户会提出一个复杂任务，你需要将其分解为具体的、可执行的步骤。

请以JSON格式输出计划，格式如下：
{
  "goal": "任务目标描述",
  "success_criteria": "成功标准",
  "todos": [
    {
      "id": "T1",
      "title": "具体步骤标题",
      "why": "为什么需要这一步",
      "type": "tool|chat|reason|write|research",
      "tool": "工具名称（如果type=tool）",
      "input": "输入数据（可选）",
      "expected_output": "期望的输出格式",
      "needs": ["需要的用户信息"]
    }
  ]
}

只输出JSON，不要其他解释。"""

    prompt = f"用户任务：{user_request}\n\n请制定详细的执行计划。"

    try:
        response = llm_client.generate(
            messages=[Message(role=Role.USER, content=prompt)],
            system_prompt=system_prompt,
            stream=False
        )

        # Parse the JSON response
        import json
        plan_data = json.loads(response.content.strip())

        # Convert to TodoItem objects
        todos = []
        for todo_data in plan_data.get("todos", []):
            todos.append(TodoItem(
                id=todo_data["id"],
                title=todo_data["title"],
                why=todo_data["why"],
                type=TodoType(todo_data["type"]),
                tool=todo_data.get("tool"),
                input_data=todo_data.get("input"),
                expected_output=todo_data.get("expected_output", ""),
                needs=todo_data.get("needs", [])
            ))

        state["plan"] = todos
        logger.info(f"Generated plan with {len(todos)} todos")

    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        # Fallback to empty plan
        state["plan"] = []

    state["execution_path"].append("planner_generate")
    state["current_node"] = "planner_generate"

    return state


def planner_gate_node(state: AgentState) -> Dict[str, Any]:
    """
    Gate node that checks if we can proceed with plan execution.
    """
    logger.info("Checking planner gate...")

    # Check if we have missing information
    has_missing_info = any(
        todo.needs for todo in state["plan"]
    ) if state["plan"] else False

    if has_missing_info and state["limits"].max_ask_cycles > 0:
        state["sufficiency"] = "missing"
        state["awaiting_user"] = True
    else:
        state["sufficiency"] = "enough"
        state["awaiting_user"] = False

    state["execution_path"].append("planner_gate")
    state["current_node"] = "planner_gate"

    return state


def todo_dispatch_node(state: AgentState) -> Dict[str, Any]:
    """
    Dispatch the current todo item for execution.
    """
    logger.info("Dispatching todo item...")

    if not state["plan"]:
        logger.warning("No plan available for dispatch")
        state["should_end"] = True
        return state

    # Get current todo
    current_idx = state.get("current_todo", 0)
    if current_idx >= len(state["plan"]):
        # All todos completed
        state["should_end"] = True
        state["current_node"] = "end"
        return state

    todo = state["plan"][current_idx]
    state["current_todo"] = current_idx

    logger.info(f"Dispatching todo {todo.id}: {todo.title} (type: {todo.type.value})")

    # Route based on todo type
    if todo.type == TodoType.TOOL:
        # Prepare tool call
        state["pending_tool_call"] = {
            "name": todo.tool,
            "arguments": todo.input_data or {},
            "todo_id": todo.id
        }
    elif todo.type == TodoType.CHAT:
        # Use chat model
        llm_client = create_llm_client("chat")
        response = llm_client.generate(
            messages=state["messages"],
            system_prompt=f"执行任务：{todo.title}\n目标：{todo.expected_output}",
            stream=False
        )
        # Add response to messages
        assistant_msg = Message(role=Role.ASSISTANT, content=response.content)
        state["messages"].append(assistant_msg)

    elif todo.type in [TodoType.REASON, TodoType.WRITE, TodoType.RESEARCH]:
        # Use reasoner model
        llm_client = create_llm_client("reasoner")
        response = llm_client.generate(
            messages=state["messages"],
            system_prompt=f"执行任务：{todo.title}\n目标：{todo.expected_output}",
            stream=False
        )
        # Add response to messages
        assistant_msg = Message(role=Role.ASSISTANT, content=response.content)
        state["messages"].append(assistant_msg)

    # Move to next todo
    state["current_todo"] = current_idx + 1

    state["execution_path"].append("todo_dispatch")
    state["current_node"] = "todo_dispatch"

    return state


def aggregate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Aggregate results from all completed todos into final answer.
    """
    logger.info("Aggregating final answer...")

    # Collect all assistant responses from the execution
    assistant_responses = [
        msg.content for msg in state["messages"]
        if msg.role.value == "assistant"
    ]

    # Create a comprehensive answer
    if state["plan"]:
        final_answer = f"## 执行完成\n\n"
        for i, todo in enumerate(state["plan"]):
            final_answer += f"### {todo.title}\n{todo.why}\n\n"

        final_answer += f"## 结果汇总\n\n"
        for i, response in enumerate(assistant_responses[-len(state["plan"]):]):
            final_answer += f"**步骤{i+1}结果：**\n{response}\n\n"
    else:
        final_answer = "任务执行完成。" + " ".join(assistant_responses[-1:])

    state["final_answer"] = final_answer
    state["should_end"] = True

    state["execution_path"].append("aggregate_answer")
    state["current_node"] = "aggregate_answer"

    return state


def _needs_user_clarification(content: str) -> bool:
    """
    Determine if the response content indicates a need for user clarification.

    This is a simplified version - in practice, you might use more sophisticated
    detection based on the model's output patterns.
    """
    clarification_indicators = [
        "请告诉我",
        "您能提供",
        "需要更多信息",
        "请问",
        "我想了解",
        "能否告诉我",
        "需要澄清",
        "请补充"
    ]

    content_lower = content.lower()
    return any(indicator in content_lower for indicator in clarification_indicators)
