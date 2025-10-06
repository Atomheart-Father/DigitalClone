"""
Schemas package - 统一契约定义

包含所有系统使用的数据结构定义：
- messages.py: 统一消息协议
- tool_io.py: 工具统一I/O契约  
- planning.py: 计划/步骤/搜索树结构
- trace.py: meta-trace记录格式
"""

from .messages import (
    Message, Role, ToolCall, ToolResult, StreamingChunk, LLMResponse,
    ConversationContext, RouteDecision, ToolMeta, ToolExecutionResult,
    create_user_message, create_assistant_message, create_tool_message,
    create_ask_user_message, create_system_message
)

from .tool_io import (
    ToolIO, CostInfo, TraceStep, ToolSchema, ToolExecutionError,
    CommonSchemas, create_success_result, create_error_result,
    validate_tool_io, convert_old_tool_result, convert_to_old_format
)

from .planning import (
    Plan, PlanStep, Candidate, SearchNode, PlanStatus, StepStatus, SearchStrategy,
    BestOfNResult, TreeOfThoughtsResult, MCTSResult, SearchConfig, SearchResult,
    create_plan, create_step, create_candidate, create_search_node
)

from .trace import (
    TraceEvent, ExecutionTrace, ABTestTrace, TraceAnalyzer,
    TraceLevel, OperationType, DecisionReason, CostMetrics, ScoreMetrics,
    create_trace_event, create_execution_trace, create_ab_test
)

__all__ = [
    # Messages
    "Message", "Role", "ToolCall", "ToolResult", "StreamingChunk", "LLMResponse",
    "ConversationContext", "RouteDecision", "ToolMeta", "ToolExecutionResult",
    "create_user_message", "create_assistant_message", "create_tool_message",
    "create_ask_user_message", "create_system_message",
    
    # Tool I/O
    "ToolIO", "CostInfo", "TraceStep", "ToolSchema", "ToolExecutionError",
    "CommonSchemas", "create_success_result", "create_error_result",
    "validate_tool_io", "convert_old_tool_result", "convert_to_old_format",
    
    # Planning
    "Plan", "PlanStep", "Candidate", "SearchNode", "PlanStatus", "StepStatus", "SearchStrategy",
    "BestOfNResult", "TreeOfThoughtsResult", "MCTSResult", "SearchConfig", "SearchResult",
    "create_plan", "create_step", "create_candidate", "create_search_node",
    
    # Trace
    "TraceEvent", "ExecutionTrace", "ABTestTrace", "TraceAnalyzer",
    "TraceLevel", "OperationType", "DecisionReason", "CostMetrics", "ScoreMetrics",
    "create_trace_event", "create_execution_trace", "create_ab_test"
]
