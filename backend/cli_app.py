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

# 直接使用绝对导入（start.py已设置sys.path）
import config
import message_types
import logger

# Import graph modules (start.py已设置sys.path)
try:
    import graph
    import tool_prompt_builder
    graph_app = graph.graph_app
    planner_app = graph.planner_app
    create_initial_state = graph.state.create_initial_state
    load_system_prompt = tool_prompt_builder.load_system_prompt
    GRAPH_AVAILABLE = True
    logger.logger.info("Graph modules imported successfully")
except ImportError as e:
    logger.logger.warning(f"Graph modules not available: {e}")
    GRAPH_AVAILABLE = False
    load_system_prompt = None

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
        self.last_planner_state = None  # Save last planner state for continuation

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
        from tool_registry import registry
        tools = registry.list_tools()
        print(f"🛠️  已加载 {len(tools)} 个工具: {[t.name for t in tools]}")

        print("\n输入您的消息，或使用特殊命令:")
        print("  :help  显示帮助信息")
        print("  :tools 列出可用工具")
        print("  :q     退出程序")
        print("  :clear 清空对话历史")
        print("  :graph 显示图状态")
        print("  :route 显示路由决策")
        print("  :prompt [system|tools] 显示系统提示")
        print("  :plan  显示计划详情")
        print("  :todo <id> 显示特定TODO详情")
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

        elif cmd == 'route':
            self._show_route_status()
            return False

        elif cmd == 'prompt':
            self._show_prompt_status(command)
            return False

        elif cmd == 'plan':
            self._show_plan_status()
            return False

        elif cmd.startswith('todo'):
            # :todo <id> - show specific todo details
            parts = cmd.split()
            if len(parts) == 2:
                todo_id = parts[1]
                self._show_todo_details(todo_id)
            else:
                print("用法: :todo <id>")
                print("示例: :todo T1")
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
        from tool_registry import registry
        tools = registry.list_tools()
        if not tools:
            print("暂无可用工具")
            return

        print(f"\n可用工具 ({len(tools)} 个):")
        for tool in tools:
            print(f"  • {tool.name}: {tool.description}")
            # Show parameters if available
            if hasattr(tool, 'parameters') and tool.parameters.get('properties'):
                params = list(tool.parameters['properties'].keys())
                print(f"    参数: {', '.join(params)}")
            # Show executor info
            if hasattr(tool, 'executor_default'):
                print(f"    执行者: {tool.executor_default}")
            if hasattr(tool, 'complexity'):
                print(f"    复杂度: {tool.complexity}")

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
            if not GRAPH_AVAILABLE:
                print(f"\rLangGraph功能暂时不可用，请稍后重试。")
                logger.error("LangGraph not available - import failed during initialization")
                return

            # Create initial state
            initial_state = create_initial_state(user_input)

            # Use full routing logic to determine execution mode
            from agent_core import AgentRouter
            from message_types import ConversationContext

            router = AgentRouter()
            context = ConversationContext(
                conversation_history=self.conversation_history,
                current_tools=[],
                user_preferences={}
            )
            route_decision = router.route(user_input, context)

            # For complex tasks (reasoner), always use planner graph
            if route_decision.engine == "reasoner":
                if self.stream:
                    print(" (检测到复杂任务，使用规划模式)")
                self._handle_planner_execution(user_input, route_decision)
            else:
                # For simple tasks (chat), use streaming if enabled, otherwise regular graph
                if self.stream:
                    self._handle_direct_streaming(user_input)
                else:
                    self._handle_graph_execution(initial_state)

        except Exception as e:
            print(f"\r处理请求时出错: {e}")
            logger.error(f"Error in conversation turn: {e}")
            # Uncomment for debugging
            # import traceback
            # traceback.print_exc()

    def _handle_graph_execution(self, initial_state):
        """Handle regular graph execution."""
        # Execute the graph with checkpointer configuration
        config = {"configurable": {"thread_id": "cli-session"}}
        final_state = graph_app.invoke(initial_state, config=config)

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
        """Handle streaming graph execution using LangGraph's stream API."""
        try:
            config = {"configurable": {"thread_id": "cli-session"}}

            # Use LangGraph's stream method for true streaming
            accumulated_content = ""

            for event in graph_app.stream(initial_state, config=config):
                for node_name, node_state in event.items():
                    logger.debug(f"Streaming event from node: {node_name}")

                    # Check if we have a final answer
                    if node_state.get("final_answer"):
                        final_answer = node_state["final_answer"]
                        print(final_answer, end="", flush=True)
                        accumulated_content += final_answer

                    # Check for assistant messages with content
                    elif node_state.get("messages"):
                        messages = node_state["messages"]
                        if messages:
                            last_msg = messages[-1]
                            if (last_msg.role == Role.ASSISTANT and
                                hasattr(last_msg, 'content') and last_msg.content):

                                # Print new content progressively
                                new_content = last_msg.content
                                if len(new_content) > len(accumulated_content):
                                    # Print only the new part
                                    to_print = new_content[len(accumulated_content):]
                                    print(to_print, end="", flush=True)
                                    accumulated_content = new_content

            # Ensure we end with a newline
            if accumulated_content:
                print()

            # Get final state for logging
            final_state = graph_app.get_state(config=config)
            final_state_data = final_state.values

            # Update conversation history
            if final_state_data.get("messages"):
                self.conversation_history = final_state_data["messages"]

            # Log the conversation
            self.logger.log_turn(
                route_decision=final_state_data.get("route_decision"),
                messages=final_state_data.get("messages", []),
                tool_calls_count=final_state_data.get("tool_call_count", 0),
                ask_cycles_used=final_state_data.get("ask_cycles_used", 0)
            )

        except Exception as e:
            print(f"\n流式输出处理出错: {e}")
            logger.error(f"Error in streaming graph execution: {e}")
            # Fallback to non-streaming execution
            try:
                self._handle_graph_execution(initial_state)
            except Exception as fallback_error:
                logger.error(f"Fallback execution also failed: {fallback_error}")

    def _handle_direct_streaming(self, user_input: str):
        """Handle streaming using direct LLM calls with tool call support."""
        try:
            from agent_core import AgentCore
            from message_types import Message, Role

            agent = AgentCore()
            streaming_response = agent.process_turn(
                user_input=user_input,
                conversation_history=self.conversation_history,
                stream=True
            )

            # Process streaming chunks with tool call handling
            accumulated_content = ""
            tool_calls_count = 0
            pending_tool_calls = []

            for chunk in streaming_response:
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    accumulated_content += chunk.content

                if chunk.tool_calls:
                    # Handle tool calls in streaming mode
                    for tool_call in chunk.tool_calls:
                        pending_tool_calls.append(tool_call)
                        tool_calls_count += 1

                        # Execute tool immediately
                        print(f"[执行工具: {tool_call.name}]", end="", flush=True)
                        try:
                            from tool_registry import registry
                            tool_result = registry.execute(tool_call.name, **tool_call.arguments)

                            # Create tool result message
                            tool_message = Message(
                                role=Role.TOOL,
                                content=f"工具执行结果: {tool_result.value if tool_result.ok else tool_result.error}",
                                tool_result=tool_call
                            )

                            # Add to conversation history
                            self.conversation_history.append(tool_message)

                            # Show tool execution in streaming output
                            print(f"\n[执行工具: {tool_call.name}]", end="", flush=True)

                            # Continue the conversation with tool result
                            # This is a simplified approach - in production, you'd want to
                            # continue the streaming conversation with the tool result
                            follow_up_response = agent.process_turn(
                                user_input="",  # Empty input since we're continuing
                                conversation_history=self.conversation_history,
                                stream=True
                            )

                            # Process follow-up chunks
                            for follow_chunk in follow_up_response:
                                if follow_chunk.content:
                                    print(follow_chunk.content, end="", flush=True)
                                    accumulated_content += follow_chunk.content

                        except Exception as tool_error:
                            logger.error(f"Tool execution failed: {tool_error}")
                            print(f"\n[工具执行失败: {tool_call.name}]", end="", flush=True)

            # Ensure we end with a newline
            if accumulated_content:
                print()

            # Add user and assistant messages to history
            self.conversation_history.append(Message(role=Role.USER, content=user_input))
            if accumulated_content:
                self.conversation_history.append(Message(
                    role=Role.ASSISTANT,
                    content=accumulated_content
                ))

            # Log the conversation
            from agent_core import RouteDecision
            dummy_route_decision = RouteDecision(
                engine="chat",
                reason="Streaming mode with tool calls",
                confidence=1.0
            )
            self.logger.log_turn(
                route_decision=dummy_route_decision,
                messages=self.conversation_history[-2:],
                tool_calls_count=tool_calls_count,
                ask_cycles_used=0
            )

        except Exception as e:
            print(f"\n直接流式输出处理出错: {e}")
            logger.error(f"Error in direct streaming: {e}")
            # Fallback to non-streaming
            initial_state = create_initial_state(user_input)
            self._handle_graph_execution(initial_state)

    def _handle_planner_execution(self, user_input: str, route_decision):
        """Handle planner execution using LangGraph planner with heartbeat monitoring."""
        try:
            # Create initial state for planner
            initial_state = create_initial_state(user_input)

            # Execute planner graph with heartbeat monitoring
            config = {
                "configurable": {"thread_id": f"planner-{user_input[:20]}"},  # Use input prefix for thread ID
                "recursion_limit": 100  # Increase from default 25 to handle complex planning
            }

            # Hotfix-4: Add heartbeat logging for planner execution monitoring
            import time
            start_time = time.time()
            logger.info("Starting planner execution...")

            # Use stream mode to monitor progress and prevent blocking
            final_state = None
            try:
                # Try streaming mode first for better monitoring
                accumulated_state = initial_state.copy()
                for event in planner_app.stream(initial_state, config=config):
                    for node_name, node_state in event.items():
                        accumulated_state.update(node_state)

                        # Check for planner generation completion
                        if node_name == "planner_generate" and node_state.get("plan"):
                            logger.info(f"Planner generated {len(node_state['plan'])} todos")

                        # 🔴 关键：捕捉 ask_user_interrupt 中断并阻塞等待输入
                        if node_name == "ask_user_interrupt" and node_state.get("needs_user_input"):
                            logger.info("🛑 DETECTED USER INPUT REQUIREMENT - Blocking for user input")
                            needs_info = node_state["needs_user_input"]
                            prompt = f"需要为 '{needs_info.get('todo_title', '任务')}' 提供参数: {', '.join(needs_info.get('needs', []))}"

                            print(f"\n\033[33m[USER INPUT REQUIRED]\033[0m {prompt}")
                            try:
                                # 阻塞等待用户输入，最多等待120秒
                                import select
                                import sys

                                print("> ", end="", flush=True)
                                ready, _, _ = select.select([sys.stdin], [], [], 120.0)

                                if ready:
                                    user_text = input().strip()
                                else:
                                    print("\n输入超时，使用默认值继续...")
                                    user_text = ""  # 超时降级

                            except (EOFError, KeyboardInterrupt):
                                print("\n操作取消")
                                user_text = ""

                            # 将输入写回状态并同步推进
                            accumulated_state["user_provided_input"] = {param: user_text for param in needs_info.get('needs', [])}
                            accumulated_state["needs_info"] = needs_info

                            # 同步调用以处理用户输入并继续执行
                            accumulated_state = planner_app.invoke(accumulated_state, config=config)
                            logger.info("✅ User input injected and graph resumed")
                            continue

                        # Check for completion
                        if node_state.get("should_end") or node_state.get("final_answer"):
                            final_state = accumulated_state.copy()
                            # Ensure needs_user_input is included in final state
                            if node_state.get("needs_user_input"):
                                final_state["needs_user_input"] = node_state["needs_user_input"]
                                # Save state for continuation if user input is needed
                                self.last_planner_state = final_state.copy()
                            break

                    # Safety timeout check (5 minutes max)
                    if time.time() - start_time > 300:
                        logger.error("Planner execution timeout after 5 minutes")
                        raise TimeoutError("Planner execution exceeded 5 minute timeout")

                if final_state is None:
                    # Fallback to invoke if streaming didn't complete
                    logger.warning("Planner streaming didn't complete, falling back to invoke")
                    final_state = planner_app.invoke(initial_state, config=config)

            except Exception as stream_error:
                logger.warning(f"Planner streaming failed: {stream_error}, falling back to invoke")
                final_state = planner_app.invoke(initial_state, config=config)

            execution_time = time.time() - start_time
            logger.info(f"Planner execution completed in {execution_time:.2f}s")

            # Check if user input is needed before showing results
            if final_state.get("needs_user_input"):
                logger.info("User input required, triggering parameter collection")
                self._handle_parameter_collection(final_state["needs_user_input"])
                return

            # Extract the final answer
            if final_state["final_answer"]:
                print(final_state["final_answer"])
            elif final_state["messages"]:
                # Get the last assistant message
                last_msg = final_state["messages"][-1]
                if hasattr(last_msg, 'content') and last_msg.content:
                    print(last_msg.content)

            # Update conversation history
            if final_state.get("messages"):
                self.conversation_history = final_state["messages"]

            # Log the conversation
            self.logger.log_turn(
                route_decision=route_decision,
                messages=final_state.get("messages", []),
                tool_calls_count=final_state.get("tool_call_count", 0),
                ask_cycles_used=0  # Planner handles ask cycles internally
            )

        except Exception as e:
            print(f"\n规划执行出错: {e}")
            logger.error(f"Error in planner execution: {e}")
            # Fallback to direct streaming
            self._handle_direct_streaming(user_input)

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
        elif result.get("needs_user_input"):
            # New mechanism: collect specific parameters from user
            self._handle_parameter_collection(result["needs_user_input"])
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

    def _show_route_status(self):
        """Show current routing decision."""
        print("\n=== 路由决策状态 ===")
        print("支持的路由模式:")
        print("  • chat: 简单对话和基础工具调用")
        print("  • reasoner: 复杂推理和数学计算")
        print("  • planner: 复杂规划和多步骤任务")
        print("  • auto_rag: 自动知识扩充")
        print("\n决策依据:")
        print("  • 关键词检测（计划、方案、多步骤等）")
        print("  • 文本长度阈值（>100字符倾向规划）")
        print("  • 结构化模式（数字列表、流程图等）")
        print("  • 复杂度评估")
        print("=" * 30)

    def _show_prompt_status(self, command: str):
        """Show current system prompt and tools."""
        parts = command.split()
        show_system = len(parts) <= 1 or 'system' in parts[1:]
        show_tools = len(parts) <= 1 or 'tools' in parts[1:]

        print("\n=== 系统提示状态 ===")

        if show_system:
            print("\n📝 系统提示:")
            try:
                from tool_prompt_builder import load_system_prompt
                system_prompt = load_system_prompt("chat")  # Default to chat
                # Show first 200 chars and indicate if truncated
                if len(system_prompt) > 200:
                    print(f"{system_prompt[:200]}...")
                    print(f"\n(完整提示长度: {len(system_prompt)} 字符)")
                else:
                    print(system_prompt)
            except Exception as e:
                print(f"加载系统提示失败: {e}")

        if show_tools:
            print("\n🛠️ 工具清单:")
            try:
                from tool_prompt_builder import build_tool_prompts
                tool_prompts = build_tool_prompts()
                print(f"已注册工具数量: {len(tool_prompts['tools'])}")
                for tool in tool_prompts["tool_name_index"]:
                    desc = tool_prompts["tool_name_index"][tool]["description"]
                    print(f"  • {tool}: {desc[:50]}{'...' if len(desc) > 50 else ''}")
            except Exception as e:
                print(f"加载工具清单失败: {e}")

        print("=" * 30)

    def _show_plan_status(self):
        """Show current plan details."""
        print("\n=== 当前计划详情 ===")

        # This would ideally access the current graph state
        # For now, show general information
        print("Planner v1.0: 支持结构化任务规划与执行")
        print("计划生成: Reasoner模型 + JSON模式")
        print("执行分发: 智能路由到Chat/Reasoner执行者")
        print("工具调用: 严格两步协议 (tool_calls → role:tool → 继续)")
        print("\n计划包含字段:")
        print("  • id: 唯一标识符")
        print("  • title: 任务标题")
        print("  • type: tool/chat/reason/write/research")
        print("  • executor: auto/chat/reasoner")
        print("  • tool: 工具名称 (type=tool时)")
        print("  • input: 输入参数")
        print("  • expected_output: 期望输出")
        print("  • needs: 缺失信息需求")
        print("\n执行者选择优先级:")
        print("  1. TodoItem.executor (显式指定)")
        print("  2. TOOL_META.executor_default")
        print("  3. 复杂度自动判断")
        print("  4. Chat (默认)")
        print("=" * 30)

    def _show_todo_details(self, todo_id: str):
        """Show details for a specific todo item."""
        print(f"\n=== TODO {todo_id} 详情 ===")

        # This would access the current plan from graph state
        # For now, show placeholder information
        print(f"TODO ID: {todo_id}")
        print("状态: 计划中 (实际运行时会显示执行状态)")
        print("\n字段说明:")
        print("  • title: 任务具体描述")
        print("  • type: 执行类型 (tool/chat/reason/write/research)")
        print("  • executor: 执行者 (chat/reasoner)")
        print("  • tool: 工具名称 (type=tool时)")
        print("  • input: 输入参数")
        print("  • expected_output: 期望产出格式")
        print("  • output: 实际执行结果")
        print("  • needs: 缺失信息需求")
        print("\n在实际运行中，此命令会显示该TODO的完整状态信息。")
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

    def _handle_parameter_collection(self, needs_info: dict):
        """Handle collection of specific parameters from user."""
        print(f"\n🔍 需要收集参数用于: {needs_info['todo_title']}")
        print(f"请提供以下参数:")

        collected_params = {}
        for param in needs_info["needs"]:
            while True:
                try:
                    value = input(f"  {param}: ").strip()
                    if value:
                        collected_params[param] = value
                        break
                    else:
                        print(f"  ❌ {param}不能为空，请重新输入")
                except (EOFError, KeyboardInterrupt):
                    print("\n操作取消")
                    return

        print("✅ 参数收集完成，正在继续执行...")  # Continue with collected parameters
        self._continue_with_parameters(collected_params, needs_info)

    def _continue_with_parameters(self, parameters: dict, needs_info: dict):
        """Continue execution with collected parameters."""
        try:
            # Check if we have a saved planner state to continue
            if self.last_planner_state and self.last_planner_state.get("plan"):
                # Continue the previous planner execution with updated parameters
                self._continue_planner_execution(parameters, needs_info)
                return

            # Fallback to agent method for non-planner continuations
            result = agent.continue_with_parameter_input(
                parameters,
                self.conversation_history,
                needs_info
            )

            # Since this is a continuation of planner execution,
            # we need to handle the response differently
            if result.get("awaiting_user_input") or result.get("needs_user_input"):
                # Still needs more input
                if result.get("awaiting_user_input"):
                    self.awaiting_user_input = True
                    print(f"\r{result['response']}")
                elif result.get("needs_user_input"):
                    self._handle_parameter_collection(result["needs_user_input"])
                return

            # Execution completed
            self._display_response(result)

            # Update conversation history
            self.conversation_history = result["final_messages"]

            # Log continuation
            self.logger.log_continuation(
                clarification=f"Parameter input: {parameters}",
                messages=result["final_messages"],
                tool_calls_count=result["tool_calls_made"]
            )

            # Display response
            self._display_response(result)

        except Exception as e:
            print(f"处理参数输入时出错: {e}")
            logger.error(f"Error handling parameter input: {e}")

    def _continue_planner_execution(self, parameters: dict, needs_info: dict):
        """Continue a previously interrupted planner execution with user parameters."""
        try:
            if not self.last_planner_state:
                raise ValueError("No saved planner state to continue")

            # Update the saved state with user provided parameters
            updated_state = self.last_planner_state.copy()

            # Set up state for ask_user_interrupt_node to process
            updated_state["user_provided_input"] = parameters
            updated_state["needs_info"] = needs_info

            # Clear the needs_user_input flag since we have the parameters
            updated_state["needs_user_input"] = False

            # Add a system message indicating parameter collection completion
            param_summary = ", ".join([f"{k}='{v}'" for k, v in parameters.items()])
            system_message = Message(
                role=Role.SYSTEM,
                content=f"用户已提供所需参数: {param_summary}。继续执行计划。"
            )
            updated_state["messages"].append(system_message)

            logger.info(f"Continuing planner execution with parameters: {parameters}")

            # Continue the planner execution from the updated state
            config = {
                "configurable": {"thread_id": f"planner-continuation-{hash(str(parameters))}"},
                "recursion_limit": 100
            }

            # Execute the planner from the updated state
            final_state = planner_app.invoke(updated_state, config=config)

            # Handle the final result
            if final_state.get("final_answer"):
                print(final_state["final_answer"])
            elif final_state["messages"]:
                # Get the last assistant message
                for msg in reversed(final_state["messages"]):
                    if msg.role == Role.ASSISTANT:
                        print(msg.content)
                        break

            # Update conversation history
            self.conversation_history = final_state["messages"]

            # Clear the saved state since execution is complete
            self.last_planner_state = None

            # Log the completion
            self.logger.log_continuation(
                clarification=f"Planner continuation with parameters: {parameters}",
                messages=final_state["messages"],
                tool_calls_count=final_state.get("tool_call_count", 0)
            )

        except Exception as e:
            print(f"继续规划执行时出错: {e}")
            logger.error(f"Error continuing planner execution: {e}")
            # Reset state on error
            self.last_planner_state = None

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
