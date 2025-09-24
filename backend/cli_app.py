#!/usr/bin/env python3
"""
CLI Application for the Digital Clone AI Assistant.

Provides a command-line REPL interface with support for streaming output,
special commands, and AskUser state management.
"""

import sys
import argparse
import logging
from typing import List, Optional

import config
import message_types
import logger
from tool_prompt_builder import load_system_prompt
from graph import graph_app
from graph.state import create_initial_state

# 为了兼容性，创建别名
config = config.config
Message = message_types.Message
Role = message_types.Role
ConversationLogger = logger.ConversationLogger

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIApp:
    """Command-line interface for the AI assistant."""

    def __init__(self, stream: bool = False):
        self.stream = stream
        self.conversation_history: List[Message] = []
        self.awaiting_user_input = False
        self.logger = ConversationLogger()

    def run(self):
        """Main CLI application loop."""
        self._print_welcome()

        try:
            while True:
                if self.awaiting_user_input:
                    prompt = "请提供更多信息> "
                else:
                    prompt = "> "

                try:
                    user_input = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n再见！")
                    break

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith(':'):
                    if self._handle_special_command(user_input):
                        break
                    continue

                # Process user input
                self._process_user_input(user_input)

        except Exception as e:
            logger.error(f"CLI application error: {e}")
            print(f"\n发生错误: {e}")
            print("请检查日志以获取详细信息。")

    def _print_welcome(self):
        """Print welcome message and system information."""
        print("=" * 50)
        print("🎭 赛博克隆 AI 助手")
        print("=" * 50)

        # Show model information
        if config.is_api_key_available():
            print(f"✓ 已配置 API 密钥")
            print(f"📡 Chat模型: {config.MODEL_CHAT}")
            print(f"🤖 Reasoner模型: {config.MODEL_REASONER}")
        else:
            print("⚠️  未配置 API 密钥，使用 MockClient")

        # Show tool information
        tools = agent.registry.list_tools()
        print(f"🛠️  已加载 {len(tools)} 个工具: {[t.name for t in tools]}")

        print("\n输入您的消息，或使用特殊命令:")
        print("  :help  显示帮助信息")
        print("  :tools 列出可用工具")
        print("  :q     退出程序")
        print("  :clear 清空对话历史")
        print("  :graph 显示图状态")
        print("-" * 50)

    def _handle_special_command(self, command: str) -> bool:
        """
        Handle special commands starting with ':'.

        Returns:
            True if should exit, False otherwise
        """
        cmd = command[1:].lower()

        if cmd == 'q' or cmd == 'quit':
            print("再见！")
            return True

        elif cmd == 'help':
            self._show_help()
            return False

        elif cmd == 'tools':
            self._show_tools()
            return False

        elif cmd == 'clear':
            self.conversation_history.clear()
            self.awaiting_user_input = False
            print("✓ 已清空对话历史")
            return False

        elif cmd == 'history':
            self._show_history()
            return False

        elif cmd == 'graph':
            self._show_graph_status()
            return False

        else:
            print(f"未知命令: {command}")
            print("输入 :help 查看可用命令")
            return False

    def _show_help(self):
        """Show help information."""
        print("\n帮助信息:")
        print("  常规对话: 直接输入您的问题")
        print("  特殊命令:")
        print("    :help   - 显示此帮助信息")
        print("    :tools  - 列出可用工具")
        print("    :q      - 退出程序")
        print("    :clear  - 清空对话历史")
        print("    :history- 显示对话历史")
        print("\n  流式输出: 使用 --stream 参数启用")
        print("  AskUser: 当助手需要更多信息时，会提示您提供")

    def _show_tools(self):
        """Show available tools."""
        tools = agent.router.registry.list_tools()
        if not tools:
            print("暂无可用工具")
            return

        print(f"\n可用工具 ({len(tools)} 个):")
        for tool in tools:
            print(f"  • {tool.name}: {tool.description}")
            # Show parameters if available
            if tool.parameters.get('properties'):
                params = list(tool.parameters['properties'].keys())
                print(f"    参数: {', '.join(params)}")

    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("暂无对话历史")
            return

        print(f"\n对话历史 ({len(self.conversation_history)} 条消息):")
        for i, msg in enumerate(self.conversation_history, 1):
            role_name = {
                "user": "用户",
                "assistant": "助手",
                "tool": "工具",
                "system": "系统"
            }.get(msg.role.value, msg.role.value)

            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  {i}. {role_name}: {content_preview}")

    def _process_user_input(self, user_input: str):
        """Process user input and generate response."""
        try:
            if self.awaiting_user_input:
                # This is a clarification response
                self._handle_clarification(user_input)
            else:
                # This is a new conversation turn
                self._handle_new_turn(user_input)

        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            print(f"处理输入时出错: {e}")

    def _handle_new_turn(self, user_input: str):
        """Handle a new conversation turn using LangGraph."""
        if self.stream:
            print("正在思考...", end="", flush=True)
        else:
            print("正在思考...", end="", flush=True)

        try:
            # Create initial state
            initial_state = create_initial_state(user_input)

            # Add system prompt based on route (will be determined by graph)
            # For now, start with empty system prompt, graph will handle routing

            # Execute graph
            if self.stream:
                # Handle streaming with graph
                self._handle_graph_streaming(initial_state)
            else:
                # Handle regular graph execution
                self._handle_graph_execution(initial_state)

        except Exception as e:
            print(f"\r处理请求时出错: {e}")
            logger.error(f"Error in conversation turn: {e}")

    def _handle_graph_execution(self, initial_state):
        """Handle regular graph execution."""
        # Execute the graph
        final_state = graph_app.invoke(initial_state)

        # Extract the final answer
        if final_state["final_answer"]:
            print(f"\r{final_state['final_answer']}")
        elif final_state["messages"] and len(final_state["messages"]) > 1:
            # Get the last assistant message
            last_msg = final_state["messages"][-1]
            if last_msg.role == Role.ASSISTANT:
                print(f"\r{last_msg.content}")

        # Update conversation history
        self.conversation_history = final_state["messages"]

        # Log the conversation
        self.logger.log_turn(
            route_decision=final_state.get("route_decision"),
            messages=final_state["messages"],
            tool_calls_count=final_state["tool_call_count"],
            ask_cycles_used=final_state.get("ask_cycles_used", 0)
        )

    def _handle_graph_streaming(self, initial_state):
        """Handle streaming graph execution."""
        # For now, implement basic streaming
        # In a full implementation, we'd need to stream through the graph
        # This is a simplified version

        accumulated_content = ""

        try:
            # Execute graph step by step for demonstration
            # In practice, this would be more complex with actual streaming

            # For demonstration, just execute normally but simulate streaming
            final_state = graph_app.invoke(initial_state)

            # Simulate streaming by yielding content progressively
            if final_state["messages"] and len(final_state["messages"]) > 1:
                last_msg = final_state["messages"][-1]
                if last_msg.role == Role.ASSISTANT and last_msg.content:
                    # Simulate streaming by printing word by word
                    words = last_msg.content.split()
                    for word in words:
                        print(word + " ", end="", flush=True)
                        accumulated_content += word + " "
                    print()  # New line at end

            # Update conversation history
            self.conversation_history = final_state["messages"]

            # Log the conversation
            self.logger.log_turn(
                route_decision=final_state.get("route_decision"),
                messages=final_state["messages"],
                tool_calls_count=final_state["tool_call_count"],
                ask_cycles_used=final_state.get("ask_cycles_used", 0)
            )

        except Exception as e:
            print(f"\n流式输出处理出错: {e}")
            logger.error(f"Error in streaming graph execution: {e}")

    def _handle_regular_response(self, result):
        """Handle regular (non-streaming) response."""
        # Update conversation history
        self.conversation_history = result["final_messages"]

        # Log the conversation turn
        self.logger.log_turn(
            route_decision=result["route_decision"],
            messages=result["final_messages"],
            tool_calls_count=result["tool_calls_made"],
            ask_cycles_used=result["ask_cycles_used"]
        )

        # Handle response
        if result.get("awaiting_user_input"):
            self.awaiting_user_input = True
            print(f"\r{result['response']}")
        else:
            self.awaiting_user_input = False
            self._display_response(result)

    def _handle_streaming_response(self, result_generator, user_input: str):
        """Handle streaming response."""
        accumulated_content = ""
        tool_calls_made = 0
        ask_cycles_used = 0

        try:
            for chunk in result_generator:
                # Print content chunks
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    accumulated_content += chunk.content

                # Handle tool calls
                if chunk.tool_calls:
                    tool_calls_made += len(chunk.tool_calls)

                # Handle finish
                if chunk.finish_reason:
                    if chunk.finish_reason == "stop":
                        print()  # New line at the end
                    elif chunk.finish_reason == "error":
                        print(f"\n流式输出出错: {chunk.content}")
                        return

            # Create a synthetic result for logging
            route_decision = RouteDecision(
                engine="streaming",
                reason="streaming_response",
                confidence=1.0
            )

            # Log the conversation turn
            self.logger.log_turn(
                route_decision=route_decision,
                messages=[
                    Message(role=Role.USER, content=user_input),
                    Message(role=Role.ASSISTANT, content=accumulated_content)
                ],
                tool_calls_count=tool_calls_made,
                ask_cycles_used=ask_cycles_used
            )

            # Update conversation history
            self.conversation_history = [
                Message(role=Role.USER, content=user_input),
                Message(role=Role.ASSISTANT, content=accumulated_content)
            ]

        except Exception as e:
            print(f"\n流式输出处理出错: {e}")
            logger.error(f"Error handling streaming response: {e}")

    def _show_graph_status(self):
        """Show current graph execution status."""
        print("\n=== LangGraph 执行状态 ===")
        print(f"当前节点: user_input (等待用户输入)")
        print(f"对话轮数: {len([msg for msg in self.conversation_history if msg.role == Role.USER])}")
        print(f"消息总数: {len(self.conversation_history)}")
        print(f"等待用户输入: {getattr(self, 'awaiting_user_input', False)}")
        print(f"流式输出: {self.stream}")
        print("执行路径: user_input → decide_route → model_call → [tool_exec|need_user|end]")
        print("=" * 30)

    def _handle_clarification(self, clarification: str):
        """Handle user clarification after AskUser."""
        print("正在继续处理...", end="", flush=True)

        try:
            # Get the last route decision from history
            # In a more sophisticated implementation, we'd store this in context
            # For now, we'll re-route based on the clarification
            from .agent_core import RouteDecision

            # Simple heuristic: check if clarification is complex
            route_decision = agent.router.route(clarification, None)

            # Continue with clarification
            result = agent.continue_with_user_clarification(
                clarification,
                self.conversation_history,
                route_decision
            )

            # Update conversation history
            self.conversation_history = result["final_messages"]

            # Log continuation
            self.logger.log_continuation(
                clarification=clarification,
                messages=result["final_messages"],
                tool_calls_count=result["tool_calls_made"]
            )

            # Reset state and display response
            self.awaiting_user_input = False
            self._display_response(result)

        except Exception as e:
            print(f"\r处理澄清信息时出错: {e}")
            logger.error(f"Error handling clarification: {e}")
            self.awaiting_user_input = False

    def _display_response(self, result: dict):
        """Display the final response."""
        response = result["response"]

        if self.stream:
            # Simulate streaming by printing character by character
            print("\r", end="")
            for char in response:
                print(char, end="", flush=True)
                # Small delay for streaming effect (optional)
        else:
            print(f"\r{response}")

        # Show statistics if available
        tool_calls = result.get("tool_calls_made", 0)
        if tool_calls > 0:
            print(f"\n[使用了 {tool_calls} 个工具调用]")

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.logger, 'close'):
            self.logger.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="赛博克隆AI助手 CLI")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="启用流式输出"
    )

    args = parser.parse_args()

    app = CLIApp(stream=args.stream)
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
