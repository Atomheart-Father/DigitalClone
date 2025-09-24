"""
Edge definitions for the DigitalClone AI Assistant LangGraph.

This module contains the conditional routing logic that determines
the flow between nodes in the execution graph.
"""

from typing import Literal

from .state import AgentState, Route


def route_after_decision(state: AgentState) -> Literal["model_call", "end"]:
    """
    Route after route decision.

    Always proceed to model call for now.
    """
    return "model_call"


def route_after_model_call(state: AgentState) -> Literal["tool_exec", "need_user", "end"]:
    """
    Route after model call based on the response type.
    """
    if state["pending_tool_call"]:
        return "tool_exec"
    elif state["awaiting_user"]:
        return "need_user"
    else:
        return "end"


def route_after_tool_exec(state: AgentState) -> Literal["model_call", "end"]:
    """
    Route after tool execution.

    Usually go back to model for next step, but could end if complete.
    """
    # For now, always go back to model call
    # In a more sophisticated implementation, you might check if more tools are needed
    return "model_call"


def route_after_need_user(state: AgentState) -> Literal["ask_user_interrupt", "model_call"]:
    """
    Route based on whether user clarification is actually needed.
    """
    if state["awaiting_user"]:
        return "ask_user_interrupt"
    else:
        return "model_call"


def route_after_ask_user_interrupt(state: AgentState) -> Literal["model_call", "end"]:
    """
    Route after user provides clarification.

    Always go back to model call to continue.
    """
    return "model_call"


def should_end(state: AgentState) -> bool:
    """
    Determine if the conversation should end.
    """
    return state["should_end"] or state["retries"] >= state["max_retries"]
