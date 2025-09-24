"""
Node definitions for the DigitalClone AI Assistant LangGraph.

This module contains all the node functions that make up the execution graph.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional

import sys
import os
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import state types first (always needed)
from .state import AgentState, Route, TodoItem, TodoType

# Conditional imports to support both relative and absolute imports
try:
    from backend.message_types import Message, Role, RouteDecision, ToolCall
    from backend.agent_core import AgentRouter
    from backend.llm_interface import create_llm_client
    from backend.tool_registry import registry
    from backend.tool_prompt_builder import build_tool_prompts, load_system_prompt
    from backend.config import config
    from backend.logger import ConversationLogger
    from backend.ask_user_policy import needs_user_clarification
except ImportError:
    # Fallback to relative imports if absolute imports fail
    from message_types import Message, Role, RouteDecision, ToolCall
    from agent_core import AgentRouter
    from llm_interface import create_llm_client
    from tool_registry import registry
    from tool_prompt_builder import build_tool_prompts, load_system_prompt
    from config import config
    from logger import ConversationLogger

logger = logging.getLogger(__name__)


def user_input_node(state: AgentState) -> Dict[str, Any]:
    """
    User input processing node.

    This node receives user input and prepares the initial state.
    In our implementation, this is handled by create_initial_state,
    so this node mainly serves as an entry point.
    """
    logger.info(f"Processing user input: {state['messages'][-1].content[:50]}...")

    # Update execution path
    state["execution_path"].append("user_input")
    state["current_node"] = "user_input"

    # Return unchanged state - routing will happen next
    return state


def decide_route_node(state: AgentState) -> Dict[str, Any]:
    """
    Route decision node.

    Decides whether to use chat or reasoner model based on the input.
    """
    logger.info("Making routing decision...")

    if not state["messages"]:
        raise ValueError("No messages in state")

    user_message = None
    for msg in reversed(state["messages"]):
        if msg.role == Role.USER:
            user_message = msg
            break

    if not user_message:
        raise ValueError("No user message found")

    # Use existing router logic
    router = AgentRouter()
    route_decision = router.route(user_message.content, None)  # Simplified context

    # Map to our Route enum
    if route_decision.engine == "reasoner":
        route = Route.REASONER
    else:
        route = Route.CHAT

    logger.info(f"Routing decision: {route.value} (reason: {route_decision.reason})")

    # Update state
    state["route"] = route
    state["route_decision"] = route_decision
    state["execution_path"].append("decide_route")
    state["current_node"] = "decide_route"

    return state


def model_call_node(state: AgentState) -> Dict[str, Any]:
    """
    Model call node.

    Calls the appropriate LLM (chat or reasoner) with tools and system prompts.
    """
    logger.info(f"Making model call with route: {state['route'].value}")

    route = state["route"]
    messages = state["messages"]

    # Load appropriate system prompt
    try:
        from backend.tool_prompt_builder import load_system_prompt
    except ImportError:
        from tool_prompt_builder import load_system_prompt
    system_prompt = load_system_prompt(route.value)

    # Select appropriate LLM client
    if route == Route.REASONER:
        llm_client = create_llm_client("reasoner")
    else:
        llm_client = create_llm_client("chat")

    # Build tool prompts
    tool_prompts = build_tool_prompts()
    functions = tool_prompts["tools"]

    # For now, we'll handle streaming separately in CLI
    # This node focuses on the core logic
    response = llm_client.generate(
        messages,
        functions=functions,
        system_prompt=system_prompt,
        stream=False
    )

    # Handle response
    if response.tool_calls:
        # Has tool calls - store for execution
        tool_call = response.tool_calls[0]
        state["pending_tool_call"] = {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "id": getattr(tool_call, 'id', None)  # May not have ID yet
        }
        logger.info(f"Tool call detected: {tool_call.name}")

        # Add assistant message to conversation
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content,
            tool_call=tool_call
        )
        state["messages"].append(assistant_message)

    elif needs_user_clarification(response.content):
        # Needs user clarification
        state["awaiting_user"] = True
        state["user_input_buffer"] = None

        # Add assistant message
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content
        )
        state["messages"].append(assistant_message)

        logger.info("AskUser clarification needed")

    else:
        # Final answer
        state["final_answer"] = response.content
        state["should_end"] = True

        # Add assistant message
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content
        )
        state["messages"].append(assistant_message)

        logger.info("Final answer generated")

    state["execution_path"].append("model_call")
    state["current_node"] = "model_call"

    return state


def tool_exec_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool execution node with executor routing and two-step protocol.

    Uses the specified executor to call tools with proper LLM interaction.
    """
    logger.info("Executing tool with executor routing...")

    if not state["pending_tool_call"]:
        raise ValueError("No pending tool call in state")

    tool_call_info = state["pending_tool_call"]
    tool_name = tool_call_info["tool"]
    input_data = tool_call_info["input_data"]
    todo_id = tool_call_info["todo_id"]
    executor = tool_call_info["executor"]

    logger.info(f"⚙️ TOOL EXECUTION START - {tool_name} with executor: {executor}")
    logger.info(f"   Todo ID: {todo_id}")
    logger.info(f"   Input Data: {input_data}")

    # Find the corresponding todo item
    current_todo = None
    if state["plan"]:
        for todo in state["plan"]:
            if todo.id == todo_id:
                current_todo = todo
                break

    if not current_todo:
        logger.error(f"❌ Could not find todo with id {todo_id}")
        state["pending_tool_call"] = None
        return state

    # Execute tool using the two-step protocol
    task_context = f"{current_todo.title}\n目标：{current_todo.expected_output}"
    logger.info(f"🔄 Calling tool with context: {task_context}")

    tool_result = call_tool_with_llm(executor, tool_name, task_context, state)

    if tool_result["success"]:
        # Store tool result for potential reflective replanning
        state["last_tool_result"] = tool_result

        # Update todo with result
        current_todo.output = tool_result["summary"]
        logger.info(f"✅ TOOL EXECUTION SUCCESS - {tool_name}")
        logger.info(f"   Result: {tool_result['summary'][:200]}{'...' if len(tool_result['summary']) > 200 else ''}")

        # Check if tool result contains content to add to chat context
        if "value" in tool_result and isinstance(tool_result["value"], dict):
            tool_value = tool_result["value"]
            if tool_value.get("add_to_context") and "content" in tool_value:
                # Add file content to chat context
                content = tool_value["content"]
                file_path = tool_value.get("file_path", "unknown file")

                context_message = Message(
                    role=Role.SYSTEM,
                    content=f"已读取文件内容（{file_path}）：\n\n{content[:2000]}{'...' if len(content) > 2000 else ''}\n\n请基于以上文件内容回答后续问题。"
                )
                state["messages"].append(context_message)
                logger.info(f"📝 Added file content to chat context ({len(content)} chars)")

        # Advance to next todo
        current_idx = state.get("current_todo", 0)
        state["current_todo"] = current_idx + 1
        logger.info(f"⏭️ ADVANCING to next todo (index: {current_idx + 1})")

    else:
        if "needs_clarification" in tool_result:
            # Trigger ask_user_interrupt - pause the tool call for later resumption
            state["paused_tool_call"] = state["pending_tool_call"]  # Save for resumption
            state["pending_tool_call"] = None  # Clear current pending call
            state["awaiting_user"] = True
            state["user_input_buffer"] = tool_result["needs_clarification"]
            logger.info(f"⏸️ TOOL EXECUTION PAUSED - Needs user clarification:")
            logger.info(f"   Question: {tool_result['needs_clarification']}")
            return state
        else:
            # Tool execution failed - add error to conversation for replanning
            error_msg = f"工具执行失败：{tool_result['error']}"
            current_todo.output = error_msg
            logger.error(f"❌ TOOL EXECUTION FAILED - {tool_name}: {tool_result['error']}")

            # Add error message to conversation so Chat model can see it
            from backend.message_types import Message, Role
            error_message = Message(
                role=Role.SYSTEM,
                content=f"工具调用失败反馈：{tool_name}执行失败，错误：{tool_result['error']}。请重新规划或提供替代方案。"
            )
            state["messages"].append(error_message)
            logger.info("📝 Added tool failure feedback to conversation")

            # Trigger replanning by going back to planner
            logger.info("🔄 TRIGGERING REPLAN due to tool failure")
            state["current_node"] = "planner_generate"
            state["pending_tool_call"] = None
            return state

    # Clear pending tool call
    state["pending_tool_call"] = None
    logger.info("🧹 Cleared pending tool call")

    state["execution_path"].append("tool_exec")
    state["current_node"] = "tool_exec"

    return state


def need_user_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if user clarification is needed.

    This is a conditional node that determines the flow.
    """
    needs_user = state["awaiting_user"]
    logger.info(f"User clarification needed: {needs_user}")

    state["execution_path"].append("need_user")
    state["current_node"] = "need_user"

    return state


def ask_user_interrupt_node(state: AgentState) -> Dict[str, Any]:
    """
    Human-in-the-loop interrupt node.

    This node handles resumption after user clarification or parameter input.
    It can handle both paused tool calls and missing parameter collection.
    """
    logger.info("👤 USER INTERACTION COMPLETE - Resuming execution")

    # Check if there's user input to process (from CLI parameter collection)
    if state.get("user_provided_input") and state.get("needs_info"):
        user_input = state["user_provided_input"]
        needs_info = state["needs_info"]

        logger.info(f"📝 PROCESSING user input from CLI: {user_input}")

        # Find and update the corresponding todo
        todo_id = needs_info["todo_id"]
        for todo in state["plan"]:
            if todo.id == todo_id:
                if not todo.input_data:
                    todo.input_data = {}

                # Update todo with user input
                for param, value in user_input.items():
                    todo.input_data[param] = value
                    logger.info(f"   Updated {param}: {value}")

                logger.info(f"✅ Todo {todo.id} parameters updated, ready for execution")
                break

        # Clear the user input state
        state.pop("user_provided_input", None)
        state.pop("needs_info", None)

    # Check if there's a paused tool call to resume
    elif state.get("paused_tool_call"):
        # Resume the paused tool call
        state["pending_tool_call"] = state["paused_tool_call"]
        state["paused_tool_call"] = None
        logger.info("▶️ RESUMING paused tool call after user input")
        logger.info(f"   Tool: {state['pending_tool_call']['tool']}")
        logger.info(f"   Todo: {state['pending_tool_call']['todo_id']}")

    # Clear clarification state
    state["awaiting_user"] = False
    state["user_input_buffer"] = None
    logger.info("🧹 Cleared user interaction state")

    state["execution_path"].append("ask_user_interrupt")
    state["current_node"] = "ask_user_interrupt"

    return state


def end_node(state: AgentState) -> Dict[str, Any]:
    """
    End node for successful completion.
    """
    logger.info("Conversation completed successfully")

    state["execution_path"].append("end")
    state["current_node"] = "end"
    state["should_end"] = True

    return state


# ===== PLANNER NODES =====

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify user intent and choose execution pipeline.

    Routes to: chat, planner, auto_rag
    Locks planner route until completion.
    """
    logger.info("Classifying user intent...")

    # Check if we have user input to process (continuation from CLI)
    if state.get("user_provided_input"):
        logger.info("Detected user input continuation, routing to ask_user_interrupt")
        state["execution_path"].append("classify_intent")
        state["current_node"] = "ask_user_interrupt"
        # Set a flag to indicate we should route to ask_user_interrupt
        state["_route_to_ask_user"] = True
        return state

    # Check if route is already locked (from previous planner execution)
    if state.get("route_locked", False):
        logger.info("Route locked, continuing with planner")
        state["execution_path"].append("classify_intent")
        state["current_node"] = "classify_intent"
        return state

    if not state["messages"]:
        raise ValueError("No messages in state")

    user_message = None
    for msg in reversed(state["messages"]):
        if msg.role.value == "user":
            user_message = msg
            break

    if not user_message:
        raise ValueError("No user message found")

    user_input = user_message.content.lower()

    # Check for explicit auto_rag triggers
    if any(keyword in user_input for keyword in ['自动扩充', '整理对话', '总结对话', 'auto_rag']):
        route = Route.AUTO_RAG
        reason = "用户明确请求自动知识扩充"
    # Check for planning keywords (must match AgentRouter.COMPLEX_KEYWORDS)
    elif any(keyword in user_input for keyword in [
        '计划', '规划', '制定', '分解', '多步骤', '调研', '写方案', '评估', '对比',
        '流程', '依赖', '里程碑', 'roadmap', 'strategy', 'systematic',
        'complex', 'comprehensive', 'detailed', 'step-by-step', 'breakdown',
        '分析', '总结', '报告', '查找', '搜索', '研究', '调查', '整理',
        '综合', '整合', '比较', '评估', '撰写', '生成', '创建'
    ]) or len(user_input) > 100:  # Long inputs likely need planning
        route = Route.PLANNER
        reason = "检测到复杂规划任务特征"
        # Lock the route for planner execution
        state["route_locked"] = True
    else:
        route = Route.CHAT
        reason = "简单对话任务，使用chat模型"

    logger.info(f"Intent classification: {route.value} ({reason})")

    state["route"] = route
    state["execution_path"].append("classify_intent")
    state["current_node"] = "classify_intent"

    return state


def sufficiency_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if we have sufficient information to proceed with planning.

    This is a simplified version - in production, you'd use more sophisticated analysis.
    """
    logger.info("Checking information sufficiency...")

    # For now, assume we have enough information
    # In production, this would analyze the plan and check for missing prerequisites
    state["sufficiency"] = "enough"
    state["execution_path"].append("sufficiency_check")
    state["current_node"] = "sufficiency_check"

    return state


def planner_generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a structured plan using the new three-phase approach:
    1. Chat model creates quick draft (<100 words)
    2. Reasoner model reviews and improves (<200 word prompt)
    3. Chat model outputs final JSON plan with execution dependencies

    This ensures better planning quality while avoiding Reasoner instability.
    """
    logger.info("Generating structured plan using three-phase approach...")

    if not state["messages"]:
        raise ValueError("No messages in state")

    # Get the user request
    user_request = ""
    for msg in state["messages"]:
        if msg.role.value == "user":
            user_request = msg.content
            break

    try:
        # Phase 1: Chat model creates quick draft (<100 words)
        logger.info("📝 Phase 1: Chat model creating quick draft...")
        chat_client = create_llm_client("chat")

        # Get available tools summary for context
        tools_summary = "可用工具：文件读取(file_read)、网络搜索(web_search)、计算器(calculator)、文档写入(markdown_writer)、RAG搜索(rag_search)等。"

        quick_draft_prompt = f"""用户任务：{user_request}

用<100字规划执行方案：
1. 步骤序列（串行/并行标注）
2. 各步骤工具及参数需求
3. 依赖关系
4. 用户需提供的参数

{tools_summary}

格式：步骤N(串行/并行)：工具(参数=来源) → 结果"""

        # Manage conversation history - compress if too long
        total_chars = sum(len(str(msg.content or "")) for msg in state["messages"])
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
        estimated_tokens = total_chars // 4
        if estimated_tokens > 8000:  # 8k tokens limit
            logger.info(f"📚 Compressing conversation history: ~{estimated_tokens} tokens -> summarizing")
            state["messages"] = _compress_conversation_history(state["messages"])

        try:
            quick_response = chat_client.generate(
                messages=[Message(role=Role.USER, content=quick_draft_prompt)],
                stream=False
            )
            draft_plan = quick_response.content.strip()
            logger.info(f"📝 PHASE 1 COMPLETE - Quick draft ({len(draft_plan)} chars):")
            logger.info(f"   Draft: {draft_plan}")
        except Exception as e:
            logger.warning(f"❌ Phase 1 failed: {e}, using fallback plan")
            draft_plan = f"分析用户需求：{user_request[:50]}... 使用相关工具获取信息并生成结果。需要文件读取、网络搜索、文档写入等工具。"
            logger.info(f"📝 PHASE 1 FALLBACK - Draft: {draft_plan}")

        # Phase 2: Reasoner model reviews the draft (<200 word prompt)
        logger.info("🤔 PHASE 2 START - Reasoner model reviewing draft...")

        # Get available tools for context
        try:
            from backend.tool_prompt_builder import build_tool_prompts
            tool_prompts = build_tool_prompts()
            available_tools = tool_prompts.get("tool_name_index", {})
            tools_list = list(available_tools.keys())
            tools_summary = ', '.join(tools_list)
        except Exception as e:
            logger.warning(f"Could not load tools for planning review: {e}")
            tools_summary = "file_read, web_search, calculator, markdown_writer, rag_search"

        # Fixed reasoner prompt template (<200 words)
        reasoner_review_template = f"""审查执行方案：

用户需求：{{user_request}}

方案：{{draft_plan}}

工具参数：
{tools_summary}

重要说明：你只能建议使用上述列表中的工具，不能发明或假设不存在的工具！

分析(<80字)：
- 流程合理性？
- 参数完整性？
- 用户需提供哪些参数？

输出改进建议。"""

        # Don't truncate inputs too aggressively - let the model handle longer context
        reasoner_prompt = reasoner_review_template.format(
            user_request=user_request,
            draft_plan=draft_plan
        )

        logger.info(f"🤔 PHASE 2 PROMPT ({len(reasoner_prompt)} chars):")
        # Only log first 500 chars to avoid log spam, but don't truncate the actual prompt
        logger.info(f"   Prompt: {reasoner_prompt[:500]}{'...' if len(reasoner_prompt) > 500 else ''}")

        # Remove hard truncation - let the model handle the full context
        # The 200 token limit is a soft guideline, not a hard requirement

        try:
            reasoner_client = create_llm_client("reasoner")
            review_response = reasoner_client.generate(
                messages=[Message(role=Role.USER, content=reasoner_prompt)],
                stream=False
            )
            review_feedback = review_response.content.strip()
            logger.info(f"🤔 PHASE 2 COMPLETE - Reasoner review ({len(review_feedback)} chars):")
            logger.info(f"   Review: {review_feedback}")
        except Exception as e:
            logger.warning(f"❌ Phase 2 failed: {e}, skipping review")
            review_feedback = "方案基本合理，可以按原计划执行。确保区分串行和并行任务。"
            logger.info(f"🤔 PHASE 2 FALLBACK - Review: {review_feedback}")

        # Phase 3: Chat model creates final JSON plan incorporating feedback
        logger.info("📋 PHASE 3 START - Chat model creating final JSON plan...")

        final_planning_prompt = f"""基于用户需求、初步方案和专家反馈，制定最终执行计划。

用户需求：{user_request}
初步方案：{draft_plan}
专家建议：{review_feedback}

🔴 基于执行流程制定详细计划
参考初步方案中的步骤、依赖关系和串并行控制，制定完整的执行计划。

注意：只能使用系统已有的工具，不要发明不存在的工具。

请输出JSON格式的详细执行计划：

{{
  "goal": "任务目标",
  "success_criteria": "成功标准",
  "execution_strategy": "serial",
  "todos": [
    {{
      "id": "T1",
      "title": "读取用户提供的文件内容",
      "why": "获取文件内容作为报告的基础",
      "type": "tool",
      "tool": "file_read",
      "executor": "chat",
      "dependencies": [],
      "parallel_group": null,
      "execution_order": 1,
      "input": {{"file_path": "将在执行时从用户输入获取"}},
      "expected_output": "文件内容的文本",
      "needs": ["file_path"]
    }},
    {{
      "id": "T2",
      "title": "将文件内容存入RAG系统",
      "why": "为后续查询和分析建立知识库",
      "type": "tool",
      "tool": "rag_upsert",
      "executor": "chat",
      "dependencies": ["T1"],
      "parallel_group": null,
      "execution_order": 2,
      "input": {{"documents": "从T1读取的文件内容"}},
      "expected_output": "文件内容已存入RAG系统",
      "needs": []
    }},
    {{
      "id": "T3",
      "title": "搜索互联网相关信息",
      "why": "补充背景信息",
      "type": "tool",
      "tool": "web_search",
      "executor": "chat",
      "dependencies": ["T1"],
      "parallel_group": null,
      "execution_order": 3,
      "input": {{"query": "基于文件内容的关键搜索词"}},
      "expected_output": "搜索结果",
      "needs": []
    }},
    {{
      "id": "T4",
      "title": "生成分析报告",
      "why": "整合文件内容和搜索结果生成最终报告",
      "type": "tool",
      "tool": "markdown_writer",
      "executor": "chat",
      "dependencies": ["T1", "T2", "T3"],
      "parallel_group": null,
      "execution_order": 4,
      "input": {{"content": "整合的文件内容和搜索结果", "filename": "analysis_report"}},
      "expected_output": "保存到OUTPUT_DIR的markdown文件",
      "needs": []
    }}
  ]
}}

执行控制规范：
- dependencies: ["T1"] - 必须完成的前置任务ID数组
- parallel_group: "group1" - 同组任务可并行执行（相同组名）
- execution_order: 1 - 组内执行顺序（从小到大）
- needs: ["file_path"] - 需要用户提供的参数，系统会中断询问

执行规则：
1. 相同parallel_group的任务按execution_order顺序执行
2. 不同parallel_group间按dependencies关系串行执行
3. 有needs字段的任务会中断执行收集用户输入
4. markdown_writer自动使用OUTPUT_DIR环境变量

只输出JSON格式。"""

        logger.info(f"📋 PHASE 3 PROMPT ({len(final_planning_prompt)} chars):")
        logger.info(f"   User Request: {user_request}")
        logger.info(f"   Draft Plan: {draft_plan}")
        logger.info(f"   Review Feedback: {review_feedback}")

        # Manage conversation history for Phase 3 as well
        total_chars = sum(len(str(msg.content or "")) for msg in state["messages"])
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
        estimated_tokens = total_chars // 4
        if estimated_tokens > 8000:  # 8k tokens limit
            logger.info(f"📚 Compressing conversation history for Phase 3: ~{estimated_tokens} tokens")
            state["messages"] = _compress_conversation_history(state["messages"])

        try:
            final_response = chat_client.generate(
                messages=[Message(role=Role.USER, content=final_planning_prompt)],
                stream=False,
                response_format={"type": "json_object"}
            )

            # Parse final JSON plan
            content = final_response.content.strip()
            logger.info(f"📋 PHASE 3 RESPONSE ({len(content)} chars):")
            logger.info(f"   Raw JSON: {content}")

            try:
                plan_data = json.loads(content)
                logger.info("📋 PHASE 3 COMPLETE - Final JSON plan parsing successful")
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing failed: {e}")
                # Fallback: try to extract JSON
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group())
                    logger.info("📋 PHASE 3 COMPLETE - JSON extracted and parsed from fallback")
                else:
                    raise ValueError(f"Failed to parse JSON plan: {content[:200]}...")

        except Exception as e:
            logger.warning(f"Phase 3 failed: {e}, using fallback plan")
            # Create a simple fallback plan
            plan_data = {
                "goal": f"处理用户请求：{user_request[:50]}...",
                "success_criteria": "成功完成用户任务",
                "execution_strategy": "serial",
                "todos": [
                    {
                        "id": "T1",
                        "title": "分析用户需求并执行任务",
                        "why": "直接响应用户请求",
                        "type": "tool",
                        "tool": "file_read",  # Default tool
                        "executor": "chat",
                        "dependencies": [],
                        "parallel_group": None,
                        "execution_order": 1,
                        "input": {"file_path": "需要用户指定"},
                "expected_output": "任务执行结果",
                "needs": ["file_path"]
                    }
                ]
            }
            logger.info("✅ Fallback plan created")

        # Validate and convert to TodoItem objects
        required_keys = ["goal", "success_criteria", "todos"]
        for key in required_keys:
            if key not in plan_data:
                logger.warning(f"Missing required key: {key}, using default")
                if key == "goal":
                    plan_data["goal"] = f"处理用户请求：{user_request[:50]}..."
                elif key == "success_criteria":
                    plan_data["success_criteria"] = "成功完成用户任务"
                elif key == "todos":
                    plan_data["todos"] = []

        todos = []
        for todo_data in plan_data["todos"]:
            required_todo_keys = ["id", "title", "type"]
            for key in required_todo_keys:
                if key not in todo_data:
                    logger.warning(f"Todo missing required key: {key}, skipping")
                    continue

            todos.append(TodoItem(
                id=todo_data["id"],
                title=todo_data["title"],
                why=todo_data.get("why", ""),
                type=TodoType(todo_data["type"]),
                tool=todo_data.get("tool"),
                executor=todo_data.get("executor", "chat"),
                input_data=todo_data.get("input"),
                dependencies=todo_data.get("dependencies", []),
                parallel_group=todo_data.get("parallel_group"),
                execution_order=todo_data.get("execution_order", 0),
                expected_output=todo_data.get("expected_output", ""),
                needs=todo_data.get("needs", [])
            ))

        state["plan"] = todos
        state["execution_strategy"] = plan_data.get("execution_strategy", "serial")

        logger.info(f"🎯 FINAL PLAN GENERATED - {len(todos)} todos (strategy: {state['execution_strategy']})")
        logger.info(f"📊 Goal: {plan_data.get('goal', 'N/A')}")
        logger.info(f"✅ Success Criteria: {plan_data.get('success_criteria', 'N/A')}")

        for i, todo in enumerate(todos, 1):
            deps = f" ← {todo.dependencies}" if todo.dependencies else ""
            parallel = f" [并行组:{todo.parallel_group}]" if todo.parallel_group else ""
            needs = f" [需要用户输入:{todo.needs}]" if todo.needs else ""
            tool_info = f"[{todo.tool}]" if todo.tool else ""
            executor_info = f"({todo.executor})" if todo.executor else ""
            logger.info(f"   {i}. {todo.id}: {todo.title} {tool_info}{executor_info}{deps}{parallel}{needs}")
            if todo.why:
                logger.info(f"      原因: {todo.why}")
            if todo.expected_output:
                logger.info(f"      预期输出: {todo.expected_output}")

    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Create minimal fallback plan
        state["plan"] = [
            TodoItem(
                id="T1",
                title="处理用户请求",
                why="执行用户的基本需求",
                type=TodoType.TOOL,
                tool="file_read",
                executor="chat",
                input_data={"file_path": "需要用户指定"},
                dependencies=[],
                parallel_group=None,
                execution_order=1,
                expected_output="任务结果",
                needs=["file_path"]
            )
        ]
        state["execution_strategy"] = "serial"
        logger.info("✅ Emergency fallback plan created")

    state["execution_path"].append("planner_generate")
    state["current_node"] = "planner_generate"

    return state


def planner_gate_node(state: AgentState) -> Dict[str, Any]:
    """
    Gate node that checks if we can proceed with plan execution.
    """
    logger.info("Checking planner gate...")

    # Check if we have missing information
    has_missing_info = any(
        todo.needs for todo in state["plan"]
    ) if state["plan"] else False

    if has_missing_info and state["limits"].max_ask_cycles > 0:
        state["sufficiency"] = "missing"
        state["awaiting_user"] = True
    else:
        state["sufficiency"] = "enough"
        state["awaiting_user"] = False

    state["execution_path"].append("planner_gate")
    state["current_node"] = "planner_gate"

    return state


def resolve_executor(todo: TodoItem) -> str:
    """
    Resolve which executor to use for a todo item.

    Returns:
        "chat" or "reasoner"
    """
    # Priority: todo.executor > TOOL_META.executor_default > "chat"
    if todo.executor and todo.executor != "auto":
        return todo.executor

    # Check tool metadata
    if todo.tool:
        tool_meta = registry.get_tool_meta(todo.tool)
        if tool_meta:
            if tool_meta.executor_default == "reasoner":
                return "reasoner"
            if tool_meta.complexity == "complex":
                return "reasoner"

    # Auto-upgrade conditions for chat → reasoner
    input_data = todo.input_data or {}
    input_str = json.dumps(input_data, ensure_ascii=False)

    # Condition 1: Parameter object too large (>512 chars)
    if len(input_str) > 512:
        logger.info(f"Auto-upgrade to reasoner: large parameter object ({len(input_str)} chars)")
        return "reasoner"

    # Condition 2: Previous validation failures (placeholder for future implementation)
    # This would track validation failures and upgrade after N attempts

    # Condition 3: Complex argument construction needed
    if todo.arg_template:
        logger.info(f"Auto-upgrade to reasoner: argument template required")
        return "reasoner"

    # Condition 4: Complex tool types (placeholder for specific tool types)

    # Default to chat
    return "chat"


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    Truncate text to max_chars, adding suffix if truncated.

    Args:
        text: Text to truncate
        max_chars: Maximum character count
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    # Ensure we don't go negative
    keep_chars = max(0, max_chars - len(suffix))
    return text[:keep_chars] + suffix




def call_tool_with_llm(executor: str, tool_name: str, task_context: str, state: AgentState) -> Dict[str, Any]:
    """
    Call a tool using the specified executor with proper two-step protocol.

    Returns:
        Dict with result information
    """
    tool_meta = registry.get_tool_meta(tool_name)
    if not tool_meta:
        return {"success": False, "error": f"Tool {tool_name} not found"}

    # Get appropriate system prompt
    system_prompt = load_system_prompt("chat" if executor == "chat" else "reasoner", executor)

    # For Reasoner executor, provide only the specific tool information
    # For Chat executor, provide full context as before
    if executor == "reasoner":
        # Limit all inputs for Reasoner to prevent empty responses
        task_context = truncate_text(task_context, 200)
        tool_desc = truncate_text(tool_meta.description, 150)
        arg_hint = truncate_text(tool_meta.arg_hint or "", 100)

        # Create focused context for Reasoner - only this specific tool
        context_message = f"""任务：{task_context}

你可以使用以下工具来完成任务：
- {tool_name}：{tool_desc}
  参数提示：{arg_hint}

请调用 {tool_name} 工具来完成这个任务。只允许调用这个工具和 ask_user 工具（如果需要用户信息）。
"""
        logger.info(f"Reasoner tool execution: task_context={len(task_context)} chars, "
                   f"tool_desc={len(tool_desc)} chars, arg_hint={len(arg_hint)} chars")
    else:
        # Chat executor gets full context
        tool_desc = tool_meta.description
        arg_hint = tool_meta.arg_hint or ""

        # Create full context message for Chat
        context_message = f"""任务：{task_context}

请调用 {tool_name} 工具来完成这个任务。
工具描述：{tool_desc}
参数提示：{arg_hint}

只允许调用 {tool_name} 工具和 ask_user 工具（如果需要澄清信息）。
"""

    # Add context to messages
    execution_messages = state["messages"].copy()
    execution_messages.append(Message(role=Role.USER, content=context_message))

    # Create LLM client
    llm_client = create_llm_client(executor)

    retry_count = 0
    max_retries = 2

    while retry_count <= max_retries:
        try:
            # Step 1: Generate tool call
            # For Reasoner, provide only the specific tool function definition
            # For Chat, provide both the tool and ask_user functions
            if executor == "reasoner":
                functions = [{
                    "name": tool_name,
                    "description": tool_desc,  # Use truncated description for Reasoner
                    "parameters": tool_meta.parameters
                }, {
                    "name": "ask_user",
                    "description": "向用户询问缺失的信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "要问用户的问题"}
                        },
                        "required": ["question"]
                    }
                }]
            else:
                functions = [{
                    "name": tool_name,
                    "description": tool_meta.description,  # Full description for Chat
                    "parameters": tool_meta.parameters
                }, {
                    "name": "ask_user",
                    "description": "向用户询问缺失的信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "要问用户的问题"}
                        },
                        "required": ["question"]
                    }
                }]

            response = llm_client.generate(
                messages=execution_messages,
                functions=functions,
                stream=False
            )

            # Check if we got tool calls
            if not response.tool_calls:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"No tool calls generated, retrying ({retry_count}/{max_retries})")
                    continue
                else:
                    return {"success": False, "error": "No tool calls generated after retries"}

            # Process tool calls
            for tool_call in response.tool_calls:
                if tool_call.name == "ask_user":
                    # Need user clarification - this would trigger ask_user_interrupt
                    question = tool_call.arguments.get("question", "需要更多信息")
                    logger.info(f"Tool execution needs clarification: {question}")
                    return {"success": False, "needs_clarification": question}

                elif tool_call.name == tool_name:
                    # Execute the tool
                    try:
                        result = registry.execute(tool_name, **tool_call.arguments)
                        state["tool_call_count"] += 1

                        # Add tool call to conversation
                        assistant_msg = Message(
                            role=Role.ASSISTANT,
                            content="",
                            tool_calls=[ToolCall(
                                id=tool_call.id or f"call_{state['tool_call_count']}",
                                name=tool_call.name,
                                arguments=tool_call.arguments
                            )]
                        )
                        state["messages"].append(assistant_msg)

                        # Add tool response
                        tool_msg = Message(
                            role=Role.TOOL,
                            content=str(result.value) if hasattr(result, 'value') and result.value else str(result.error or "Tool executed"),
                            tool_call_id=tool_call.id or f"call_{state['tool_call_count']}"
                        )
                        state["messages"].append(tool_msg)

                        # Step 2: Generate summary
                        summary_messages = state["messages"].copy()
                        summary_messages.append(Message(
                            role=Role.USER,
                            content="请基于工具执行结果，给出1-2句简洁的总结。"
                        ))

                        summary_response = llm_client.generate(
                            messages=summary_messages,
                            stream=False
                        )

                        return {
                            "success": True,
                            "result": result,
                            "summary": summary_response.content,
                            "executor": executor
                        }

                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        return {"success": False, "error": str(e)}

            # If we get here, no valid tool calls
            return {"success": False, "error": "No valid tool calls found"}

        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(f"Tool call failed, retrying ({retry_count}/{max_retries}): {e}")
            else:
                return {"success": False, "error": f"Tool call failed after retries: {e}"}

    return {"success": False, "error": "Unexpected error in tool execution"}


def _compress_conversation_history(messages: List[Message]) -> List[Message]:
    """Compress conversation history when it gets too long."""
    if len(messages) <= 2:
        return messages

    # Keep the first system message and last few exchanges
    compressed = []

    # Always keep system messages
    for msg in messages:
        if msg.role.value == "system":
            compressed.append(msg)

    # Keep last 3 user/assistant pairs
    recent_messages = []
    for msg in reversed(messages):
        if msg.role.value in ["user", "assistant"]:
            recent_messages.insert(0, msg)
            if len(recent_messages) >= 6:  # 3 pairs
                break

    compressed.extend(recent_messages)

    # Add a summary message
    summary_msg = Message(
        role=Role.SYSTEM,
        content="历史对话已压缩以提高性能。保留了关键系统信息和最近的对话内容。"
    )
    compressed.insert(0, summary_msg)

    logger.info(f"Compressed {len(messages)} messages to {len(compressed)}")
    return compressed


def get_next_executable_todos(state: AgentState) -> List[TodoItem]:
    """
    Determine which todos are ready for execution based on dependencies and parallel groups.
    Returns a list of todos that can be executed (may be multiple for parallel execution).
    """
    plan = state["plan"]
    completed_todos = {t.id for t in plan if t.output is not None}

    # Group todos by parallel groups
    parallel_groups = {}
    serial_todos = []

    for todo in plan:
        if todo.parallel_group:
            if todo.parallel_group not in parallel_groups:
                parallel_groups[todo.parallel_group] = []
            parallel_groups[todo.parallel_group].append(todo)
        else:
            serial_todos.append(todo)

    executable_todos = []

    # Check serial todos (those with dependencies)
    for todo in serial_todos:
        if todo.output is not None:
            continue  # Already completed

        # Check if all dependencies are completed
        deps_completed = all(dep_id in completed_todos for dep_id in (todo.dependencies or []))
        if deps_completed:
            executable_todos.append(todo)

    # Check parallel groups
    for group_id, group_todos in parallel_groups.items():
        # Group todos by execution order
        by_order = {}
        for todo in group_todos:
            order = todo.execution_order
            if order not in by_order:
                by_order[order] = []
            by_order[order].append(todo)

        # Execute todos in order, but allow parallel execution within the same order
        min_order = min(by_order.keys())
        current_order_todos = [t for t in by_order[min_order] if t.output is None]

        # Check if previous orders in this group are completed
        prev_orders_completed = True
        for order in range(1, min_order):
            if order in by_order:
                if not all(t.output is not None for t in by_order[order]):
                    prev_orders_completed = False
                    break

        if prev_orders_completed:
            executable_todos.extend(current_order_todos)

    return executable_todos


def todo_dispatch_node(state: AgentState) -> Dict[str, Any]:
    """
    Dispatch the next executable todo items for execution with proper serial/parallel control.

    Handles execution dependencies, parallel groups, and user interaction requirements.
    """
    logger.info("🔄 EXECUTION DISPATCH - Checking next executable todos...")

    if not state["plan"]:
        logger.warning("❌ No plan available for dispatch")
        state["should_end"] = True
        state["current_node"] = "end"
        return state

    # Check if all todos are completed
    all_completed = all(todo.output is not None for todo in state["plan"])
    if all_completed:
        logger.info("🎉 EXECUTION COMPLETE - All todos finished successfully")
        state["should_end"] = True
        state["current_node"] = "end"
        return state

    # Get next executable todos (may be multiple for parallel execution)
    executable_todos = get_next_executable_todos(state)

    if not executable_todos:
        logger.warning("⏳ EXECUTION WAITING - No todos ready (dependencies not satisfied)")
        state["should_end"] = True  # This will trigger ask_user if needed
        state["current_node"] = "end"
        return state

    # If multiple todos are executable, we need to decide execution strategy
    if len(executable_todos) > 1:
        logger.info(f"⚖️ MULTIPLE EXECUTABLE - {len(executable_todos)} todos ready: {[t.id for t in executable_todos]}")

        # Check if they are in the same parallel group
        parallel_groups = set(t.parallel_group for t in executable_todos if t.parallel_group)
        if len(parallel_groups) == 1:
            logger.info("🔀 PARALLEL EXECUTION - Same group, executing concurrently")
            # For now, execute the first one and mark others as pending
            # TODO: Implement true parallel execution
            todo = executable_todos[0]
        else:
            # Different groups or mixed, execute first available
            logger.info("🔀 SEQUENTIAL EXECUTION - Different groups, executing first available")
            todo = executable_todos[0]
    else:
        todo = executable_todos[0]

    logger.info(f"🚀 EXECUTING TODO - {todo.id}: {todo.title} (type: {todo.type.value})")

    # Check if this todo needs user input (new mechanism)
    if todo.needs and len(todo.needs) > 0:
        # Check if we already have user input for these needs
        has_all_needed = True
        missing_params = []

        for param in todo.needs:
            current_value = todo.input_data.get(param) if todo.input_data else None
            # Check if value is missing or is a placeholder
            if not current_value or current_value in ["需要用户指定", "用户将提供", "将在执行时从用户输入获取"]:
                missing_params.append(param)
                has_all_needed = False

        if not has_all_needed:
            logger.info(f"👤 USER INPUT REQUIRED - Todo {todo.id} missing: {missing_params}")
            logger.info(f"   Todo: {todo.title}")
            # Set state for user input collection and end execution to return to CLI
            state["needs_user_input"] = {
                "todo_id": todo.id,
                "needs": missing_params,
                "todo_title": todo.title
            }
            state["should_end"] = True
            state["current_node"] = "end"
            return state

    try:
        if todo.type == TodoType.TOOL:
            # Tool and executor should already be decided in planning phase
            if not todo.tool:
                logger.error(f"Tool todo {todo.id} missing tool name from planning phase")
                todo.output = f"错误：规划阶段未指定工具名称"
                state["current_todo"] = state.get("current_todo", 0) + 1
            else:
                # Executor should be set in planning phase, fallback to auto-resolution
                executor = todo.executor if todo.executor and todo.executor != "auto" else resolve_executor(todo)

                # Set up tool call for tool_exec node
                state["pending_tool_call"] = {
                    "tool": todo.tool,
                    "input_data": todo.input_data or {},
                    "todo_id": todo.id,
                    "executor": executor
                }
                logger.info(f"🔧 TOOL CALL PREPARED - {todo.tool} with executor {executor}")
                logger.info(f"   Input: {todo.input_data or {}}")
                logger.info(f"   Expected: {todo.expected_output or 'N/A'}")
                if todo.why:
                    logger.info(f"   Reason: {todo.why}")
                # Don't advance current_todo yet - wait for tool_exec completion

        elif todo.type == TodoType.CHAT:
            # Use chat model for this todo
            system_prompt = load_system_prompt("chat", "chat")
            additional_context = f"\n\n当前执行任务：{todo.title}\n期望输出：{todo.expected_output}"

            llm_client = create_llm_client("chat")
            response = llm_client.generate(
                messages=state["messages"],
                system_prompt=system_prompt + additional_context,
                stream=False
            )

            # Add response to messages
            assistant_msg = Message(role=Role.ASSISTANT, content=response.content)
            state["messages"].append(assistant_msg)

            # Store result in todo
            todo.output = response.content
            logger.info(f"Chat model executed for todo {todo.id}")

        elif todo.type in [TodoType.REASON, TodoType.WRITE, TodoType.RESEARCH]:
            # Use reasoner model for complex tasks
            system_prompt = load_system_prompt("reasoner", "reasoner")
            additional_context = f"\n\n当前执行任务：{todo.title}\n期望输出：{todo.expected_output}"

            llm_client = create_llm_client("reasoner")
            response = llm_client.generate(
                messages=state["messages"],
                system_prompt=system_prompt + additional_context,
                stream=False
            )

            # Add response to messages
            assistant_msg = Message(role=Role.ASSISTANT, content=response.content)
            state["messages"].append(assistant_msg)

            # Store result in todo
            todo.output = response.content
            logger.info(f"Reasoner model executed for todo {todo.id}")

        # Check if this todo needs sub-planning
        if todo.type in [TodoType.REASON, TodoType.RESEARCH] and state["depth_budget"] > 0:
            complexity_indicators = ['分解', '细化', '子任务', '多步骤', '分阶段']
            title_lower = todo.title.lower()
            output_lower = (todo.expected_output or "").lower()
            needs_subplanning = any(
                indicator in title_lower or indicator in output_lower
                for indicator in complexity_indicators
            )
            if needs_subplanning:
                logger.info(f"Todo {todo.id} may benefit from sub-planning (depth_budget: {state['depth_budget']})")
                # Could trigger planner_refine_local here

        # For tool todos, execution will continue in tool_exec node
        # For other types, we've completed execution here

    except Exception as e:
        logger.error(f"Error executing todo {todo.id}: {e}")
        # Mark todo as failed but continue
        todo.output = f"执行失败：{str(e)}"

    # Don't advance current_todo here - let the execution flow handle it
    # For tool calls, tool_exec node will handle completion
    # For direct execution, the todo is already marked as completed above

    state["execution_path"].append("todo_dispatch")
    state["current_node"] = "todo_dispatch"

    return state


def log_execution_metrics(state: AgentState) -> None:
    """
    Log execution metrics for observability.

    Args:
        state: Current agent state with metrics
    """
    metrics = state.get("metrics", {})

    if not metrics:
        logger.info("No execution metrics recorded")
        return

    # Log micro-reasoning metrics
    micro_calls = metrics.get("reasoner_micro_calls", 0)
    micro_empty = metrics.get("reasoner_micro_empty", 0)
    micro_fallback = metrics.get("reasoner_micro_fallback", 0)

    if micro_calls > 0:
        empty_rate = micro_empty / micro_calls * 100 if micro_calls > 0 else 0
        fallback_rate = micro_fallback / micro_calls * 100 if micro_calls > 0 else 0

        logger.info(f"Micro-reasoning metrics: calls={micro_calls}, empty={micro_empty} ({empty_rate:.1f}%), "
                   f"fallback={micro_fallback} ({fallback_rate:.1f}%)")

    # Log execution summary
    plan_size = len(state.get("plan", []))
    completed_todos = sum(1 for todo in state.get("plan", []) if todo.output)
    tool_calls = state.get("tool_call_count", 0)

    logger.info(f"Execution summary: plan_size={plan_size}, completed={completed_todos}, "
               f"tool_calls={tool_calls}, execution_path={state.get('execution_path', [])}")


def aggregate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Aggregate results from all completed todos into final answer.
    """
    logger.info("📊 AGGREGATING FINAL RESULTS - Compiling execution summary")

    # Log execution metrics
    log_execution_metrics(state)

    # Collect all assistant responses from the execution
    assistant_responses = [
        msg.content for msg in state["messages"]
        if msg.role.value == "assistant"
    ]

    logger.info(f"📋 EXECUTION SUMMARY - {len(state['plan']) if state['plan'] else 0} todos completed")

    # Log detailed results for each todo
    if state["plan"]:
        for i, todo in enumerate(state["plan"], 1):
            status = "✅" if todo.output else "❌"
            logger.info(f"   {i}. {status} {todo.id}: {todo.title}")
            if todo.output:
                output_preview = todo.output[:100] + "..." if len(todo.output) > 100 else todo.output
                logger.info(f"      Result: {output_preview}")
            else:
                logger.info("      Result: 未执行")

    # Create a comprehensive answer
    if state["plan"]:
        final_answer = f"## 执行完成\n\n"
        for i, todo in enumerate(state["plan"]):
            final_answer += f"### {todo.title}\n{todo.why}\n\n"

        final_answer += f"## 结果汇总\n\n"
        for i, response in enumerate(assistant_responses[-len(state["plan"]):]):
            final_answer += f"**步骤{i+1}结果：**\n{response}\n\n"
    else:
        final_answer = "任务执行完成。" + " ".join(assistant_responses[-1:])

    logger.info("🎯 FINAL ANSWER GENERATED")
    logger.info(f"   Length: {len(final_answer)} characters")
    logger.info(f"   Preview: {final_answer[:200]}...")

    state["final_answer"] = final_answer
    state["should_end"] = True

    state["execution_path"].append("aggregate_answer")
    state["current_node"] = "aggregate_answer"

    return state


# ===== REFLECTIVE REPLANNING NODES =====

def reflective_replanning_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if reflective replanning is needed after major information gathering.

    This node is called after tool execution to determine if we should trigger
    the expensive reflective replanning process based on new information volume.
    """
    from backend.config import Config

    # Skip if reflective replanning is disabled
    if not Config.ENABLE_REFLECTIVE_REPLANNING:
        logger.debug("Reflective replanning disabled, skipping")
        state["execution_path"].append("reflective_check")
        state["current_node"] = "reflective_check"
        return state

    # Check if we just executed a tool that might have gathered significant information
    if not state.get("pending_tool_call") or not state.get("last_tool_result"):
        logger.debug("No recent tool execution to check for reflection")
        state["execution_path"].append("reflective_check")
        state["current_node"] = "reflective_check"
        return state

    tool_name = state["pending_tool_call"]["tool"]
    tool_result = state["last_tool_result"]

    # Check if this tool typically gathers large amounts of information
    info_gathering_tools = ["file_read", "web_search", "rag_search", "tabular_qa"]

    if tool_name not in info_gathering_tools:
        logger.debug(f"Tool {tool_name} not considered info-gathering, skipping reflection")
        state["execution_path"].append("reflective_check")
        state["current_node"] = "reflective_check"
        return state

    # Estimate the amount of new information gathered
    info_size = 0

    # Check tool result for content
    if "value" in tool_result and isinstance(tool_result["value"], dict):
        tool_value = tool_result["value"]
        if "content" in tool_value:
            info_size += len(str(tool_value["content"]))
        if "summary" in tool_result:
            info_size += len(str(tool_result["summary"]))

    # Check if information size exceeds threshold
    if info_size >= Config.REFLECTIVE_REPLANNING_MIN_INFO_SIZE:
        logger.info(f"🧠 DETECTED LARGE INFO GATHERING - {tool_name} returned ~{info_size} chars, triggering reflective replanning")
        state["trigger_reflective_replanning"] = True
        state["new_information_summary"] = _extract_information_summary(tool_result, tool_name)
    else:
        logger.debug(f"Information size {info_size} below threshold {Config.REFLECTIVE_REPLANNING_MIN_INFO_SIZE}, skipping reflection")

    state["execution_path"].append("reflective_check")
    state["current_node"] = "reflective_check"

    return state


def reflective_replanning_node(state: AgentState) -> Dict[str, Any]:
    """
    Perform reflective replanning after major information gathering.

    This expensive process compresses context and allows the AI to reconsider
    and potentially modify the execution plan based on new information.
    """
    from backend.config import Config

    if not state.get("trigger_reflective_replanning"):
        logger.debug("No reflective replanning triggered")
        state["execution_path"].append("reflective_replanning")
        state["current_node"] = "reflective_replanning"
        return state

    logger.info("🧠 STARTING REFLECTIVE REPLANNING - Compressing context and evaluating plan changes")

    try:
        # Phase 1: Chat model compresses information and current plan
        chat_client = create_llm_client("chat")

        current_plan = state.get("plan", [])
        new_info = state.get("new_information_summary", "")

        compression_prompt = f"""基于新获取的信息，压缩当前计划的关键信息到{Config.REFLECTIVE_REPLANNING_MAX_TOKENS}token内：

新获取的信息：
{new_info[:2000]}...（如有更多内容已省略）

当前执行计划：
{_summarize_current_plan(current_plan)}

请用<100字总结：
1. 新信息的核心要点
2. 当前计划的状态
3. 潜在的计划调整方向

输出格式：直接给出总结，不要多余内容。"""

        logger.info("🧠 REFLECTIVE PHASE 1 - Chat compressing context")
        compression_response = chat_client.generate(
            messages=[Message(role=Role.USER, content=compression_prompt)],
            stream=False
        )
        compressed_context = compression_response.content.strip()
        logger.info(f"🧠 COMPRESSED CONTEXT ({len(compressed_context)} chars): {compressed_context[:200]}...")

        # Phase 2: Reasoner evaluates if plan changes are needed
        # Get available tools for context
        try:
            from backend.tool_prompt_builder import build_tool_prompts
            tool_prompts = build_tool_prompts()
            available_tools = tool_prompts.get("tool_name_index", {})
            tools_list = list(available_tools.keys())
        except Exception as e:
            logger.warning(f"Could not load tools for reflection: {e}")
            tools_list = ["file_read", "web_search", "rag_search", "tabular_qa", "calculator", "datetime", "markdown_writer"]

        reasoner_client = create_llm_client("reasoner")

        reflection_prompt = f"""评估是否需要修改执行计划：

压缩的上下文：{compressed_context}

可用工具列表：{', '.join(tools_list)}

重要说明：你只能建议使用上述列表中的工具，不能发明或假设不存在的工具！

请判断：
- 当前计划是否仍合适？
- 新信息是否需要调整步骤？
- 是否需要添加/删除/修改任务？

如果需要修改，给出具体的修改建议（必须使用上述可用工具）。
如果不需要修改，只回复"无需修改"。

输出格式：简洁的判断和建议。"""

        logger.info("🧠 REFLECTIVE PHASE 2 - Reasoner evaluating plan changes")
        reflection_response = reasoner_client.generate(
            messages=[Message(role=Role.USER, content=reflection_prompt)],
            stream=False
        )
        reflection_result = reflection_response.content.strip()
        logger.info(f"🧠 REFLECTION RESULT: {reflection_result[:200]}...")

        # Check if changes are needed
        if "无需修改" not in reflection_result.lower() and len(reflection_result.strip()) > 0:
            # Phase 3: Chat model modifies the plan based on reflection
            logger.info("🧠 REFLECTIVE PHASE 3 - Chat modifying plan based on reflection")

            plan_modification_prompt = f"""基于反思结果修改执行计划：

压缩上下文：{compressed_context}

反思建议：{reflection_result}

当前计划详情：
{_format_plan_for_modification(current_plan)}

请输出修改后的完整JSON格式执行计划。格式要求与原始计划相同。

注意：只能使用系统已有的工具，不要发明不存在的工具。"""

            modification_response = chat_client.generate(
                messages=[Message(role=Role.USER, content=plan_modification_prompt)],
                stream=False
            )

            try:
                import json
                import re
                content = modification_response.content.strip()

                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    modified_plan_data = json.loads(json_match.group())
                    logger.info("🧠 PLAN MODIFIED - Applying new execution plan")

                    # Update the plan in state
                    state["plan"] = _create_todos_from_json(modified_plan_data)
                    state["plan_modified"] = True
                else:
                    logger.warning("🧠 PLAN MODIFICATION FAILED - Could not parse JSON")
                    state["plan_modified"] = False

            except Exception as e:
                logger.error(f"🧠 PLAN MODIFICATION ERROR: {e}")
                state["plan_modified"] = False
        else:
            logger.info("🧠 NO PLAN CHANGES NEEDED - Continuing with current plan")
            state["plan_modified"] = False

    except Exception as e:
        logger.error(f"🧠 REFLECTIVE REPLANNING ERROR: {e}")
        state["plan_modified"] = False

    # Clear the trigger
    state.pop("trigger_reflective_replanning", None)
    state.pop("new_information_summary", None)

    state["execution_path"].append("reflective_replanning")
    state["current_node"] = "reflective_replanning"

    return state


def _extract_information_summary(tool_result: dict, tool_name: str) -> str:
    """Extract and summarize information from tool result."""
    summary_parts = []

    if "value" in tool_result and isinstance(tool_result["value"], dict):
        tool_value = tool_result["value"]

        if tool_name == "file_read" and "content" in tool_value:
            content = tool_value["content"]
            summary_parts.append(f"文件内容：{content[:500]}..." if len(content) > 500 else f"文件内容：{content}")

        elif tool_name == "web_search" and "results" in tool_value:
            results = tool_value["results"]
            summary_parts.append(f"网络搜索结果：找到{len(results)}个结果")

        elif tool_name == "rag_search" and "results" in tool_value:
            results = tool_value["results"]
            summary_parts.append(f"RAG搜索结果：找到{len(results)}个相关文档片段")

        elif tool_name == "tabular_qa" and "answer" in tool_value:
            summary_parts.append(f"表格问答结果：{tool_value['answer']}")

    if "summary" in tool_result:
        summary_parts.append(f"工具总结：{tool_result['summary']}")

    return " | ".join(summary_parts) if summary_parts else "无新信息"


def _summarize_current_plan(plan: List) -> str:
    """Summarize the current execution plan."""
    if not plan:
        return "无当前计划"

    summaries = []
    for i, todo in enumerate(plan, 1):
        status = "✅" if todo.output else "⏳" if todo.input_data else "❌"
        summaries.append(f"{i}. {status} {todo.title}")

    return "\n".join(summaries)


def _format_plan_for_modification(plan: List) -> str:
    """Format plan for modification prompt."""
    if not plan:
        return "当前计划为空"

    formatted = []
    for todo in plan:
        status = "已完成" if todo.output else "待执行"
        formatted.append(f"""
- ID: {todo.id}
  标题: {todo.title}
  状态: {status}
  类型: {todo.type.value}
  工具: {todo.tool or 'N/A'}
  依赖: {todo.dependencies or []}
  并行组: {todo.parallel_group or 'N/A'}
  执行顺序: {todo.execution_order}
  需要参数: {todo.needs or []}
  预期输出: {todo.expected_output}
""")

    return "".join(formatted)


def _create_todos_from_json(plan_data: dict) -> List:
    """Create Todo objects from JSON plan data."""
    from backend.message_types import Todo, TodoType

    todos = []
    if "todos" in plan_data:
        for todo_data in plan_data["todos"]:
            try:
                todo = Todo(
                    id=todo_data["id"],
                    title=todo_data["title"],
                    why=todo_data.get("why", ""),
                    type=TodoType(todo_data.get("type", "tool")),
                    tool=todo_data.get("tool"),
                    executor=todo_data.get("executor", "chat"),
                    dependencies=todo_data.get("dependencies", []),
                    parallel_group=todo_data.get("parallel_group"),
                    execution_order=todo_data.get("execution_order", 0),
                    input=todo_data.get("input", {}),
                    expected_output=todo_data.get("expected_output", ""),
                    needs=todo_data.get("needs", [])
                )
                todos.append(todo)
            except Exception as e:
                logger.error(f"Error creating todo from {todo_data}: {e}")

    return todos


