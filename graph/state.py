"""
State definition for the DigitalClone AI Assistant LangGraph.

This module defines the typed state used throughout the graph execution.
"""

from typing import List, Optional, Dict, Any, TypedDict
from enum import Enum
from dataclasses import dataclass

import sys
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Conditional imports to support both relative and absolute imports
try:
    from backend.message_types import Message, RouteDecision
except ImportError:
    from message_types import Message, RouteDecision


class Route(str, Enum):
    """Route enumeration."""
    CHAT = "chat"
    REASONER = "reasoner"
    PLANNER = "planner"
    AUTO_RAG = "auto_rag"
    UNDECIDED = "undecided"


class TodoType(str, Enum):
    """Todo item types."""
    TOOL = "tool"
    CHAT = "chat"
    REASON = "reason"
    WRITE = "write"
    RESEARCH = "research"


@dataclass
class TodoItem:
    """Todo item structure for planning."""
    id: str
    title: str
    why: str
    type: TodoType
    tool: Optional[str] = None
    executor: Optional[str] = None  # "auto", "chat", or "reasoner"
    input_data: Optional[Dict[str, Any]] = None
    arg_template: Optional[Dict[str, Any]] = None  # Template for complex argument construction
    expected_output: str = ""
    needs: List[str] = None  # Information gaps that need user clarification
    output: Optional[str] = None  # Execution result

    def __post_init__(self):
        if self.needs is None:
            self.needs = []


@dataclass
class Limits:
    """Limits and constraints for planning."""
    max_tools_per_turn: int = 3
    max_ask_cycles: int = 2
    max_depth: int = 1


class AgentState(TypedDict):
    """Typed state for the LangGraph execution."""

    # Core conversation state
    messages: List[Message]

    # Routing state
    route: Route
    route_decision: Optional[RouteDecision]
    route_locked: bool  # Lock route during planner execution

    # Planning state
    plan: Optional[List[TodoItem]]  # Structured plan from planner_generate
    current_todo: Optional[int]  # Index of current todo being executed
    depth_budget: int

    # Tool execution state
    pending_tool_call: Optional[Dict[str, Any]]
    tool_call_count: int

    # AskUser state
    awaiting_user: bool
    user_input_buffer: Optional[str]
    sufficiency: str  # "unknown", "enough", "missing"
    ask_cycles_used: int

    # Control flow
    retries: int
    max_retries: int

    # Metrics and observability
    metrics: Dict[str, Any]  # Runtime metrics collection

    # Status tracking
    current_node: str
    execution_path: List[str]

    # Context and metadata
    context: Dict[str, Any]
    session_id: Optional[str]
    limits: Limits

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
    # Import Message and Role with conditional imports
    try:
        from backend.message_types import Message, Role
    except ImportError:
        from message_types import Message, Role

    initial_message = Message(
        role=Role.USER,
        content=user_input
    )

    return AgentState(
        messages=[initial_message],
        route=Route.UNDECIDED,
        route_decision=None,
        route_locked=False,
        plan=None,
        current_todo=None,
        depth_budget=1,
        pending_tool_call=None,
        tool_call_count=0,
        awaiting_user=False,
        user_input_buffer=None,
        sufficiency="unknown",
        ask_cycles_used=0,
        retries=0,
        max_retries=3,
        current_node="user_input",
        execution_path=["user_input"],
        context={},
        session_id=session_id,
        limits=Limits(),
        final_answer=None,
        should_end=False
    )
