"""
Node definitions for the DigitalClone AI Assistant LangGraph.

This module contains all the node functions that make up the execution graph.
"""

import logging
import json
import re
from typing import Dict, Any, List

import sys
import os
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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

    elif _needs_user_clarification(response.content):
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
    Tool execution node.

    Executes pending tool calls and adds results back to conversation.
    """
    logger.info("Executing tool...")

    if not state["pending_tool_call"]:
        raise ValueError("No pending tool call in state")

    tool_call = state["pending_tool_call"]
    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]

    logger.info(f"Executing tool: {tool_name} with args: {arguments}")

    # Execute tool
    tool_result = registry.execute(tool_name, **arguments)

    # Create tool message
    if tool_result.ok:
        content = f"工具执行成功: {tool_result.value}"
    else:
        content = f"工具执行失败: {tool_result.error}"

    tool_message = Message(
        role=Role.TOOL,
        content=content,
        tool_result=ToolCall(
            name=tool_name,
            arguments=arguments
        )
    )

    # Add to conversation and clear pending call
    state["messages"].append(tool_message)
    state["pending_tool_call"] = None
    state["tool_call_count"] += 1

    logger.info(f"Tool execution completed: {tool_name}")

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

    This node will be interrupted and resumed after user input.
    """
    logger.info("Entering AskUser interrupt mode")

    # In LangGraph, this would normally interrupt execution
    # For our CLI implementation, we'll handle this in the CLI layer

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
    # Check for planning keywords
    elif any(keyword in user_input for keyword in [
        '计划', '规划', '制定', '多步骤', '调研', '方案', '评估',
        '对比', '流程', '依赖', '阶段', '项目', '任务分解'
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
    Generate a structured plan using the reasoner model with JSON mode.

    Uses response_format=json_object to ensure strict JSON output.
    """
    logger.info("Generating structured plan...")

    if not state["messages"]:
        raise ValueError("No messages in state")

    # Get the user request
    user_request = ""
    for msg in state["messages"]:
        if msg.role.value == "user":
            user_request = msg.content
            break

    # Load planner system prompt
    try:
        with open(project_root / "prompts" / "reasoner_planner.txt", 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        # Fallback system prompt
        system_prompt = """你是一个专业的项目规划师。用户会提出一个复杂任务，你需要将其分解为具体的、可执行的步骤。

只输出JSON格式的计划，不要其他解释。格式如下：
{
  "goal": "任务目标描述",
  "success_criteria": "成功标准",
  "todos": [
    {
      "id": "T1",
      "title": "具体步骤标题",
      "why": "为什么需要这一步",
      "type": "tool|chat|reason|write|research",
      "tool": "工具名称（如果type=tool）",
      "input": {"参数名": "参数值"},
      "expected_output": "期望的输出格式",
      "needs": ["需要的用户信息"]
    }
  ]
}"""

    # Create LLM client
    llm_client = create_llm_client("reasoner")

    # Build tool information for planner
    tool_prompts = build_tool_prompts()
    tools_text = f"""
可用工具：
{tool_prompts["tools_text"]}

在制定计划时，请考虑使用以下工具：
- calculator: 数学计算
- datetime: 时间处理
- rag_search: 知识库搜索
- rag_upsert: 文档入库
"""

    # Construct user prompt
    user_prompt = f"""用户任务：{user_request}

请基于可用工具制定详细的执行计划。{tools_text}

请以JSON格式响应，只输出JSON，不要任何其他解释。"""

    try:
        # Use JSON mode for strict structured output
        response = llm_client.generate(
            messages=[Message(role=Role.USER, content=user_prompt)],
            system_prompt=system_prompt,
            stream=False,
            response_format={"type": "json_object"}  # Force JSON output
        )

        # Parse the JSON response
        content = response.content.strip()
        logger.debug(f"Raw planner response: {content[:500]}...")

        # Try direct JSON parsing first
        try:
            plan_data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text using regex
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    plan_data = json.loads(json_match.group())
                    logger.info("Recovered JSON from regex extraction")
                except json.JSONDecodeError as recovery_error:
                    logger.error(f"JSON recovery failed: {recovery_error}")
                    raise ValueError(f"Could not parse JSON response: {content[:200]}...")
            else:
                raise ValueError(f"No JSON found in response: {content[:200]}...")

        # Validate JSON against schema (basic validation)
        required_keys = ["goal", "success_criteria", "todos"]
        for key in required_keys:
            if key not in plan_data:
                raise ValueError(f"Missing required key: {key}")

        if not isinstance(plan_data["todos"], list):
            raise ValueError("todos must be a list")

        # Convert to TodoItem objects
        todos = []
        for todo_data in plan_data["todos"]:
            # Validate todo structure
            required_todo_keys = ["id", "title", "type"]
            for key in required_todo_keys:
                if key not in todo_data:
                    raise ValueError(f"Todo missing required key: {key}")

            todos.append(TodoItem(
                id=todo_data["id"],
                title=todo_data["title"],
                why=todo_data.get("why", ""),
                type=TodoType(todo_data["type"]),
                tool=todo_data.get("tool"),
                executor=todo_data.get("executor"),
                input_data=todo_data.get("input"),
                arg_template=todo_data.get("arg_template"),
                expected_output=todo_data.get("expected_output", ""),
                needs=todo_data.get("needs", [])
            ))

        state["plan"] = todos
        logger.info(f"Generated plan with {len(todos)} todos")
        for todo in todos:
            logger.info(f"  - {todo.id}: {todo.title} ({todo.type.value})")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Raw response: {response.content}")
        # Try to extract JSON from text using regex
        json_match = re.search(r'\{.*\}', response.content.strip(), re.DOTALL)
        if json_match:
            try:
                plan_data = json.loads(json_match.group())
                todos = [TodoItem(
                    id=todo_data["id"],
                    title=todo_data["title"],
                    why=todo_data.get("why", ""),
                    type=TodoType(todo_data["type"]),
                    tool=todo_data.get("tool"),
                    input_data=todo_data.get("input"),
                    expected_output=todo_data.get("expected_output", ""),
                    needs=todo_data.get("needs", [])
                ) for todo_data in plan_data.get("todos", [])]
                state["plan"] = todos
                logger.info(f"Recovered plan with {len(todos)} todos from regex")
            except Exception as recovery_error:
                logger.error(f"JSON recovery failed: {recovery_error}")
                state["plan"] = []
        else:
            state["plan"] = []
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        logger.error(f"Response content: {response.content[:500] if response else 'No response'}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        state["plan"] = []

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

    # Create execution context message
    context_message = f"""任务：{task_context}

请调用 {tool_name} 工具来完成这个任务。
工具描述：{tool_meta.description}
参数提示：{tool_meta.arg_hint}

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
            response = llm_client.generate(
                messages=execution_messages,
                functions=[{
                    "name": tool_name,
                    "description": tool_meta.description,
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
                }],
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
                        result = registry.execute(tool_name, tool_call.arguments)
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


def todo_dispatch_node(state: AgentState) -> Dict[str, Any]:
    """
    Dispatch the current todo item for execution with executor routing.

    Implements sophisticated executor selection and two-step tool calling protocol.
    """
    logger.info("Dispatching todo item...")

    if not state["plan"]:
        logger.warning("No plan available for dispatch")
        state["should_end"] = True
        state["current_node"] = "end"
        return state

    # Get current todo
    current_idx = state.get("current_todo", 0)
    if current_idx >= len(state["plan"]):
        # All todos completed
        state["should_end"] = True
        state["current_node"] = "end"
        return state

    todo = state["plan"][current_idx]

    logger.info(f"Dispatching todo {todo.id}: {todo.title} (type: {todo.type.value})")

    try:
        if todo.type == TodoType.TOOL:
            # Tool execution with executor routing
            if not todo.tool:
                logger.error(f"Tool todo {todo.id} missing tool name")
                todo.output = f"错误：工具调用缺少工具名称"
            else:
                # Resolve executor
                executor = resolve_executor(todo)
                logger.info(f"Using executor '{executor}' for tool '{todo.tool}'")

                # Execute tool with LLM
                task_context = f"{todo.title}\n目标：{todo.expected_output}"
                tool_result = call_tool_with_llm(executor, todo.tool, task_context, state)

                if tool_result["success"]:
                    todo.output = tool_result["summary"]
                    logger.info(f"Tool {todo.tool} executed successfully with {executor} executor")
                else:
                    if "needs_clarification" in tool_result:
                        # Trigger ask_user_interrupt
                        state["awaiting_user"] = True
                        state["user_input_buffer"] = tool_result["needs_clarification"]
                        logger.info(f"Tool execution paused for user clarification: {tool_result['needs_clarification']}")
                        return state  # Don't advance to next todo
                    else:
                        todo.output = f"工具执行失败：{tool_result['error']}"
                        logger.error(f"Tool execution failed: {tool_result['error']}")

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
            if any(indicator in todo.title.lower() or indicator in (todo.expected_output or "").lower()):
                logger.info(f"Todo {todo.id} may benefit from sub-planning (depth_budget: {state['depth_budget']})")
                # Could trigger planner_refine_local here

        # Move to next todo
        state["current_todo"] = current_idx + 1

    except Exception as e:
        logger.error(f"Error executing todo {todo.id}: {e}")
        # Mark todo as failed but continue with next one
        todo.output = f"执行失败：{str(e)}"
        state["current_todo"] = current_idx + 1

    state["execution_path"].append("todo_dispatch")
    state["current_node"] = "todo_dispatch"

    return state


def aggregate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Aggregate results from all completed todos into final answer.
    """
    logger.info("Aggregating final answer...")

    # Collect all assistant responses from the execution
    assistant_responses = [
        msg.content for msg in state["messages"]
        if msg.role.value == "assistant"
    ]

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

    state["final_answer"] = final_answer
    state["should_end"] = True

    state["execution_path"].append("aggregate_answer")
    state["current_node"] = "aggregate_answer"

    return state


def _needs_user_clarification(content: str) -> bool:
    """
    Determine if the response content indicates a need for user clarification.

    This is a simplified version - in practice, you might use more sophisticated
    detection based on the model's output patterns.
    """
    clarification_indicators = [
        "请告诉我",
        "您能提供",
        "需要更多信息",
        "请问",
        "我想了解",
        "能否告诉我",
        "需要澄清",
        "请补充"
    ]

    content_lower = content.lower()
    return any(indicator in content_lower for indicator in clarification_indicators)
