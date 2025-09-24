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
