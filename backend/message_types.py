"""
Message type definitions for the Digital Clone AI Assistant.

This module defines the unified message protocol used throughout the system,
including roles, message structures, and function calling formats.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
    ASK_USER = "ask_user"


class ToolCall(BaseModel):
    """Represents a tool function call."""
    id: Optional[str] = Field(None, description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call")


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""
    name: str = Field(..., description="Name of the tool that was executed")
    content: str = Field(..., description="Result content from the tool execution")


class Message(BaseModel):
    """Unified message structure for all interactions."""
    role: Role = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content text")
    tool_call: Optional[ToolCall] = Field(None, description="Tool call information (assistant messages only)")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tool calls (OpenAI format)")
    tool_result: Optional[ToolResult] = Field(None, description="Tool execution result (tool messages only)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def is_function_call(self) -> bool:
        """Check if this message contains a function call."""
        return self.tool_call is not None or len(self.tool_calls) > 0

    def is_ask_user(self) -> bool:
        """Check if this message is asking the user for clarification."""
        return self.role == Role.ASK_USER

    def is_tool_result(self) -> bool:
        """Check if this message contains tool execution results."""
        return self.tool_result is not None


class StreamingChunk(BaseModel):
    """A chunk of streaming response."""
    content: str = Field("", description="Content chunk")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls in this chunk")
    finish_reason: Optional[str] = Field(None, description="Reason the response finished (only in final chunk)")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information (only in final chunk)")


class LLMResponse(BaseModel):
    """Response from LLM interface."""
    content: str = Field(..., description="Response content")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tool calls in the response")
    finish_reason: Optional[str] = Field(None, description="Reason the response finished")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")


class ConversationContext(BaseModel):
    """Context for a conversation turn."""
    messages: List[Message] = Field(default_factory=list, description="Conversation history")
    current_turn: int = Field(default=0, description="Current conversation turn number")
    ask_user_cycles: int = Field(default=0, description="Number of ask_user cycles in current turn")
    tool_calls_count: int = Field(default=0, description="Number of tool calls in current turn")


class RouteDecision(BaseModel):
    """Decision result from the routing system."""
    engine: str = Field(..., description="Selected engine ('chat' or 'reasoner')")
    reason: str = Field(..., description="Reason for the routing decision")
    confidence: float = Field(default=0.0, description="Confidence score (0.0-1.0)")


class ToolMeta(BaseModel):
    """Metadata for a tool."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for tool parameters")
    strict: bool = Field(True, description="Whether parameters must strictly match schema")
    executor_default: str = Field("chat", description="Default executor: auto, chat, or reasoner")
    complexity: str = Field("simple", description="Complexity level: simple or complex")
    arg_hint: str = Field("", description="Hints for parameter formatting and requirements")
    caller_snippet: str = Field("", description="Special calling instructions and examples")


class ToolExecutionResult(BaseModel):
    """Result of tool execution."""
    ok: bool = Field(..., description="Whether execution was successful")
    value: Optional[Any] = Field(None, description="Result value if successful")
    error: Optional[str] = Field(None, description="Error message if failed")


def create_user_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a user message."""
    return Message(
        role=Role.USER,
        content=content,
        metadata=metadata or {}
    )


def create_assistant_message(
    content: str,
    tool_call: Optional[ToolCall] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """Create an assistant message."""
    return Message(
        role=Role.ASSISTANT,
        content=content,
        tool_call=tool_call,
        metadata=metadata or {}
    )


def create_tool_message(
    tool_name: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """Create a tool result message."""
    return Message(
        role=Role.TOOL,
        content=content,
        tool_result=ToolResult(name=tool_name, content=content),
        metadata=metadata or {}
    )


def create_ask_user_message(
    question: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """Create an ask_user message for clarification."""
    return Message(
        role=Role.ASK_USER,
        content=question,
        metadata=metadata or {}
    )


def create_system_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a system message."""
    return Message(
        role=Role.SYSTEM,
        content=content,
        metadata=metadata or {}
    )
