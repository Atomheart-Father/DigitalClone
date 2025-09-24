"""
LangGraph implementation for the DigitalClone AI Assistant.

This module builds and configures the execution graph using LangGraph.
"""

import logging
import sys
from typing import Dict, Any
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    user_input_node,
    decide_route_node,
    model_call_node,
    tool_exec_node,
    need_user_node,
    ask_user_interrupt_node,
    end_node,
    # Planner nodes
    classify_intent_node,
    sufficiency_check_node,
    planner_generate_node,
    planner_gate_node,
    todo_dispatch_node,
    aggregate_answer_node
)
from .edges import (
    route_after_decision,
    route_after_model_call,
    route_after_tool_exec,
    route_after_need_user,
    route_after_ask_user_interrupt,
    should_end
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Build the LangGraph for the DigitalClone AI Assistant.

    Returns:
        Configured StateGraph instance
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("user_input", user_input_node)
    graph.add_node("decide_route", decide_route_node)
    graph.add_node("model_call", model_call_node)
    graph.add_node("tool_exec", tool_exec_node)
    graph.add_node("need_user", need_user_node)
    graph.add_node("ask_user_interrupt", ask_user_interrupt_node)
    graph.add_node("end", end_node)

    # Add edges
    graph.add_edge("user_input", "decide_route")
    graph.add_conditional_edges(
        "decide_route",
        route_after_decision,
        {
            "model_call": "model_call",
            "end": END
        }
    )
    graph.add_conditional_edges(
        "model_call",
        route_after_model_call,
        {
            "tool_exec": "tool_exec",
            "need_user": "need_user",
            "end": END
        }
    )
    graph.add_conditional_edges(
        "tool_exec",
        route_after_tool_exec,
        {
            "model_call": "model_call",
            "end": END
        }
    )
    graph.add_conditional_edges(
        "need_user",
        route_after_need_user,
        {
            "ask_user_interrupt": "ask_user_interrupt",
            "model_call": "model_call"
        }
    )
    graph.add_conditional_edges(
        "ask_user_interrupt",
        route_after_ask_user_interrupt,
        {
            "model_call": "model_call",
            "end": END
        }
    )

    # Set entry point
    graph.set_entry_point("user_input")

    return graph


def build_planner_graph() -> StateGraph:
    """
    Build the planner graph for complex task execution.

    Returns:
        Configured StateGraph for planner pipeline
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add planner nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("sufficiency_check", sufficiency_check_node)
    graph.add_node("planner_generate", planner_generate_node)
    graph.add_node("planner_gate", planner_gate_node)
    graph.add_node("todo_dispatch", todo_dispatch_node)
    graph.add_node("tool_exec", tool_exec_node)
    graph.add_node("ask_user_interrupt", ask_user_interrupt_node)
    graph.add_node("aggregate_answer", aggregate_answer_node)

    # Add edges for planner pipeline
    graph.add_edge("classify_intent", "sufficiency_check")
    graph.add_edge("sufficiency_check", "planner_generate")
    graph.add_edge("planner_generate", "planner_gate")

    # Conditional edges from planner_gate
    graph.add_conditional_edges(
        "planner_gate",
        lambda state: "ask_user_interrupt" if state["sufficiency"] == "missing" else "todo_dispatch"
    )

    graph.add_edge("ask_user_interrupt", "todo_dispatch")  # After user clarification
    graph.add_edge("tool_exec", "todo_dispatch")  # After tool execution

    # Loop back for next todo
    graph.add_conditional_edges(
        "todo_dispatch",
        lambda state: "tool_exec" if state.get("pending_tool_call") else "aggregate_answer"
    )

    # Set entry point
    graph.set_entry_point("classify_intent")

    return graph


def create_executable_graph():
    """
    Create executable graph with checkpointer.

    Returns:
        Compiled graph with memory checkpointer
    """
    graph = build_graph()

    # Add memory checkpointer for persistence
    checkpointer = MemorySaver()

    # Compile the graph
    app = graph.compile(checkpointer=checkpointer)

    logger.info("LangGraph compiled successfully with memory checkpointer")

    return app


def create_planner_graph():
    """
    Create executable planner graph with checkpointer.

    Returns:
        Compiled planner graph with memory checkpointer
    """
    graph = build_planner_graph()

    # Add memory checkpointer for persistence
    checkpointer = MemorySaver()

    # Compile the graph
    app = graph.compile(checkpointer=checkpointer)

    logger.info("Planner LangGraph compiled successfully with memory checkpointer")

    return app


# Global graph instances
graph_app = create_executable_graph()
planner_app = create_planner_graph()
