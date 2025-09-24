"""
LangGraph implementation for the DigitalClone AI Assistant.

This module builds and configures the execution graph using LangGraph.
"""

import logging
import sys
import os
from typing import Dict, Any

# Add current directory to path for relative imports
sys.path.append(os.path.dirname(__file__))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import (
    user_input_node,
    decide_route_node,
    model_call_node,
    tool_exec_node,
    need_user_node,
    ask_user_interrupt_node,
    end_node
)
from edges import (
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


# Global graph instance
graph_app = create_executable_graph()
