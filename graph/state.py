"""
State definition for the DigitalClone AI Assistant LangGraph.

This module defines the typed state used throughout the graph execution.
"""

from typing import List, Optional, Dict, Any, TypedDict
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from message_types import Message, RouteDecision


class Route(str, Enum):
    """Route enumeration."""
    CHAT = "chat"
    REASONER = "reasoner"
    UNDECIDED = "undecided"


class AgentState(TypedDict):
    """Typed state for the LangGraph execution."""

    # Core conversation state
    messages: List[Message]

    # Routing state
    route: Route
    route_decision: Optional[RouteDecision]

    # Tool execution state
    pending_tool_call: Optional[Dict[str, Any]]
    tool_call_count: int

    # AskUser state
    awaiting_user: bool
    user_input_buffer: Optional[str]

    # Control flow
    retries: int
    max_retries: int

    # Status tracking
    current_node: str
    execution_path: List[str]

    # Context and metadata
    context: Dict[str, Any]
    session_id: Optional[str]

    # Final output
    final_answer: Optional[str]
    should_end: bool


def create_initial_state(user_input: str, session_id: Optional[str] = None) -> AgentState:
    """
    Create initial state for a new conversation turn.

    Args:
        user_input: User's input message
        session_id: Optional session identifier

    Returns:
        Initial AgentState
    """
    from ..backend.message_types import Message, Role

    initial_message = Message(
        role=Role.USER,
        content=user_input
    )

    return AgentState(
        messages=[initial_message],
        route=Route.UNDECIDED,
        route_decision=None,
        pending_tool_call=None,
        tool_call_count=0,
        awaiting_user=False,
        user_input_buffer=None,
        retries=0,
        max_retries=3,
        current_node="user_input",
        execution_path=["user_input"],
        context={},
        session_id=session_id,
        final_answer=None,
        should_end=False
    )
