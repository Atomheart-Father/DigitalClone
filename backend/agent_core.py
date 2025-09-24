"""
Agent Core for the Digital Clone AI Assistant.

This module implements the main agent logic including routing decisions,
ReAct loop with tool calling, and AskUser state machine.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union, Generator

from config import config
from message_types import (
    Message, Role, RouteDecision, ConversationContext,
    LLMResponse, StreamingChunk, ToolCall, ToolExecutionResult
)
from llm_interface import create_llm_client
from tool_registry import registry

logger = logging.getLogger(__name__)


class AgentRouter:
    """Router for deciding which LLM engine to use."""

    # Keywords that indicate complex reasoning/planning tasks
    COMPLEX_KEYWORDS = {
        '计划', '规划', '制定', '分解', '多步骤', '调研', '写方案', '评估', '对比',
        '流程', '依赖', '里程碑', 'roadmap', 'strategy', 'systematic',
        'complex', 'comprehensive', 'detailed', 'step-by-step', 'breakdown'
    }

    # Patterns that indicate structured/complex tasks
    STRUCTURED_PATTERNS = [
        r'\d+\.',  # Numbered lists: 1. 2. 3.
        r'\d+\)',  # Numbered lists: 1) 2) 3)
        r'[•●○]',  # Bullet points
        r'->',     # Flow arrows
        r'→',      # Unicode arrows
        r'步骤',   # Chinese "steps"
        r'阶段',   # Chinese "stages/phases"
        r'第[一二三四五六七八九十]+步',  # Chinese numbered steps: 第一步、第二步等
        r'[一二三四五六七八九十]+、',  # Chinese numbered lists: 一、 二、 等
    ]

    def __init__(self):
        self.length_threshold = 80  # Chinese characters

    def route(self, user_input: str, context: ConversationContext) -> RouteDecision:
        """
        Decide which engine to use based on user input and context.

        Args:
            user_input: User's input text
            context: Current conversation context

        Returns:
            RouteDecision with engine choice and reasoning
        """
        # Check for explicit user requests
        lower_input = user_input.lower()
        if any(keyword in lower_input for keyword in ['复杂', '全面', '系统', '方案']):
            return RouteDecision(
                engine="reasoner",
                reason="用户明确要求复杂/全面/系统性分析",
                confidence=0.9
            )

        # Check keyword signals
        keyword_score = self._check_keywords(user_input)
        if keyword_score > 0:
            return RouteDecision(
                engine="reasoner",
                reason=f"检测到复杂任务关键词 (得分: {keyword_score})",
                confidence=min(0.8, 0.5 + keyword_score * 0.1)
            )

        # Check structural signals
        if self._check_structured_patterns(user_input):
            return RouteDecision(
                engine="reasoner",
                reason="检测到结构化任务模式（列表、流程图等）",
                confidence=0.7
            )

        # Check length threshold
        if len(user_input) > self.length_threshold:
            return RouteDecision(
                engine="reasoner",
                reason=f"输入长度超过阈值 ({len(user_input)} > {self.length_threshold})",
                confidence=0.6
            )

        # Check if tools are likely needed in combination (2+ different operations)
        if self._check_tool_combination_needed(user_input):
            return RouteDecision(
                engine="reasoner",
                reason="需要组合多个不同类型的工具操作",
                confidence=0.7
            )

        # Default to chat model
        return RouteDecision(
            engine="chat",
            reason="常规对话任务，使用chat模型",
            confidence=0.8
        )

    def _check_keywords(self, text: str) -> int:
        """Count complex task keywords in the text."""
        count = 0
        lower_text = text.lower()
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in lower_text:
                count += 1
        return count

    def _check_structured_patterns(self, text: str) -> bool:
        """Check if text contains structured patterns."""
        for pattern in self.STRUCTURED_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _check_tool_combination_needed(self, text: str) -> bool:
        """Check if multiple different tools might be needed."""
        # Define distinct tool operation categories
        tool_categories = {
            'math': ['计算', '算', '数学'],
            'time': ['时间', '日期', '几号', '现在'],
            'search': ['搜索', '查找', '查询']
        }

        # Count how many different tool categories are mentioned
        mentioned_categories = set()
        for category, keywords in tool_categories.items():
            if any(keyword in text for keyword in keywords):
                mentioned_categories.add(category)

        # Route to reasoner only if multiple different tool categories are needed
        return len(mentioned_categories) >= 2

    def route_with_reasoner_confirmation(
        self,
        user_input: str,
        context: ConversationContext
    ) -> RouteDecision:
        """
        Use reasoner to confirm routing decision when uncertain.

        This is an optional enhancement for more accurate routing.
        """
        initial_decision = self.route(user_input, context)

        # If confidence is low, ask reasoner for confirmation
        if initial_decision.confidence < 0.7:
            try:
                reasoner = create_llm_client("reasoner")

                prompt = f"""分析以下用户输入，判断是否需要复杂的推理和规划能力来回答。

用户输入: {user_input}

请只回答 "yes" 或 "no"，并简要说明原因（不超过20字）。
如果需要分解任务、多步骤推理、系统性分析，则回答 "yes"。"""

                messages = [Message(role=Role.USER, content=prompt)]
                response = reasoner.generate(messages)

                if "yes" in response.content.lower():
                    return RouteDecision(
                        engine="reasoner",
                        reason=f"Reasoner确认需要复杂推理: {response.content[:50]}",
                        confidence=0.9
                    )
                else:
                    return RouteDecision(
                        engine="chat",
                        reason=f"Reasoner确认可以使用简单回答: {response.content[:50]}",
                        confidence=0.8
                    )

            except Exception as e:
                logger.warning(f"Reasoner confirmation failed: {e}, using initial decision")
                return initial_decision

        return initial_decision


class AgentCore:
    """Main agent logic with ReAct loop and AskUser mechanism."""

    def __init__(self):
        self.router = AgentRouter()
        self.registry = registry  # Reference to global tool registry
        self.max_tool_calls = config.MAX_TOOL_CALLS_PER_TURN
        self.max_ask_cycles = config.MAX_ASK_USER_CYCLES

    def process_turn(
        self,
        user_input: str,
        conversation_history: List[Message],
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[StreamingChunk, None, None]]:
        """
        Process a single conversation turn.

        Args:
            user_input: User's input message
            conversation_history: Previous conversation messages

        Returns:
            Dictionary with response, route_decision, and execution details
        """
        context = ConversationContext(messages=conversation_history)
        route_decision = self.router.route(user_input, context)

        logger.info(f"Routing decision: {route_decision.engine} (reason: {route_decision.reason})")

        # Create appropriate LLM client
        llm_client = create_llm_client(route_decision.engine)

        # Prepare messages for LLM
        messages = conversation_history.copy()
        user_message = Message(role=Role.USER, content=user_input)
        messages.append(user_message)

        # Get available functions
        functions = registry.get_functions_schema()

        # Execute ReAct loop
        if stream:
            # Return streaming generator
            return self._react_loop_streaming(llm_client, messages, functions, context, route_decision)
        else:
            # Return regular response
            result = self._react_loop(llm_client, messages, functions, context, stream)
            return {
                "response": result["response"],
                "route_decision": route_decision,
                "tool_calls_made": result["tool_calls_count"],
                "ask_cycles_used": result["ask_cycles_used"],
                "final_messages": result["messages"]
            }

    def _react_loop(
        self,
        llm_client,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        context: ConversationContext,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the ReAct loop with tool calling and AskUser handling.

        Returns:
            Dictionary with final response and execution statistics
        """
        tool_calls_count = 0
        ask_cycles_used = 0

        while tool_calls_count < self.max_tool_calls and ask_cycles_used < self.max_ask_cycles:

            # Generate response from LLM
            try:
                response = llm_client.generate(messages, functions=functions)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return {
                    "response": f"抱歉，生成回复时出现错误: {str(e)}",
                    "tool_calls_count": tool_calls_count,
                    "ask_cycles_used": ask_cycles_used,
                    "messages": messages
                }

            # Add assistant message to history
            assistant_message = Message(
                role=Role.ASSISTANT,
                content=response.content
            )

            if response.tool_calls:
                assistant_message.tool_call = response.tool_calls[0]

            messages.append(assistant_message)

            # Check for function calls
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_calls_count += 1

                logger.info(f"Executing tool call: {tool_call.name} with args: {tool_call.arguments}")

                # Execute tool
                tool_result = registry.execute(tool_call.name, **tool_call.arguments)

                # Create tool result message
                if tool_result.ok:
                    content = f"工具执行成功: {tool_result.value}"
                else:
                    content = f"工具执行失败: {tool_result.error}"

                tool_message = Message(
                    role=Role.TOOL,
                    content=content,
                    tool_result=tool_call
                )
                messages.append(tool_message)

                # Continue the loop for another LLM call
                continue

            # Check for ask_user pattern
            if self._is_ask_user_response(response.content):
                ask_cycles_used += 1
                logger.info(f"AskUser cycle {ask_cycles_used} triggered")

                # Return ask_user response to CLI for user input
                return {
                    "response": response.content,
                    "tool_calls_count": tool_calls_count,
                    "ask_cycles_used": ask_cycles_used,
                    "messages": messages,
                    "awaiting_user_input": True
                }

            # No more actions needed, return final response
            break

        # Return final assistant response
        return {
            "response": response.content if 'response' in locals() else "我无法生成回复。",
            "tool_calls_count": tool_calls_count,
            "ask_cycles_used": ask_cycles_used,
            "messages": messages,
            "awaiting_user_input": False
        }

    def _is_ask_user_response(self, content: str) -> bool:
        """Check if the response indicates a need to ask the user for clarification."""
        ask_indicators = [
            "请告诉我",
            "您能提供",
            "需要更多信息",
            "请问",
            "我想了解",
            "能否告诉我",
            "需要澄清",
            "请补充"
        ]

        lower_content = content.lower()
        return any(indicator in lower_content for indicator in ask_indicators)

    def continue_with_user_clarification(
        self,
        clarification: str,
        previous_messages: List[Message],
        route_decision: RouteDecision
    ) -> Dict[str, Any]:
        """
        Continue conversation after user provides clarification.

        Args:
            clarification: User's clarification response
            previous_messages: Messages before asking for clarification
            route_decision: Original routing decision

        Returns:
            Dictionary with continued response
        """
        # Create LLM client (same as before)
        llm_client = create_llm_client(route_decision.engine)

        # Add user clarification to messages
        messages = previous_messages.copy()
        clarification_message = Message(role=Role.USER, content=clarification)
        messages.append(clarification_message)

        # Get available functions
        functions = registry.get_functions_schema()

        # Create new context (reset counters for continuation)
        context = ConversationContext(messages=messages)

        # Continue ReAct loop
        result = self._react_loop(llm_client, messages, functions, context)

        return {
            "response": result["response"],
            "tool_calls_made": result["tool_calls_count"],
            "final_messages": result["messages"]
        }

    def _react_loop_streaming(
        self,
        llm_client,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        context: ConversationContext,
        route_decision: RouteDecision
    ) -> Generator[StreamingChunk, None, None]:
        """
        Execute the ReAct loop with streaming response.
        For now, implement a simplified version that just streams the response.
        Complex tool calling and AskUser in streaming mode is not yet implemented.
        """
        try:
            # Generate streaming response
            response_generator = llm_client.generate(messages, functions=functions, stream=True)

            # Yield chunks as they come
            for chunk in response_generator:
                # For tool calls, we would need to handle them differently
                # For now, just yield the content chunks
                if chunk.content:
                    yield chunk

                # Handle tool calls (simplified)
                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        logger.info(f"Processing streaming tool call: {tool_call.name}")
                        # Execute tool
                        tool_result = self.registry.execute(tool_call.name, **tool_call.arguments)

                        # Add tool result to conversation
                        tool_message = Message(
                            role=Role.TOOL,
                            content=f"工具执行结果: {tool_result.value if tool_result.ok else tool_result.error}",
                            tool_result=tool_call
                        )
                        messages.append(tool_message)

                        # Yield a chunk indicating tool execution
                        yield StreamingChunk(
                            content=f"\n[执行工具: {tool_call.name}]\n"
                        )

                        # Note: In a full implementation, we'd need to get another streaming response
                        # after tool execution. For now, this is simplified.

            # Yield final chunk
            yield StreamingChunk(finish_reason="stop")

        except Exception as e:
            logger.error(f"Error in streaming ReAct loop: {e}")
            yield StreamingChunk(content=f"\n错误: {str(e)}", finish_reason="error")


# Global agent instance
agent = AgentCore()
