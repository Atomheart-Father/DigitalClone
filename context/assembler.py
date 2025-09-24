"""
Context Assembler - Intelligent message assembly from memory layers.

This implements the core logic for assembling conversation context from
multiple memory layers into optimized message sequences.
"""

import time
from typing import List, Dict, Any, Optional

# Conditional imports
try:
    from backend.message_types import Message, Role
    from backend.config import Config
    from memory.working_buffer import WorkingBuffer
    from memory.rolling_summary import RollingSummary
    from memory.profile_store import ProfileStore
    from memory.rag_store import RAGStore
except ImportError:
    # Fallback for testing
    Message = Dict
    Role = str
    Config = None
    WorkingBuffer = None
    RollingSummary = None
    ProfileStore = None
    RAGStore = None


class ContextAssembler:
    """
    Intelligent context assembler for multi-turn conversations.

    Assembles context from four memory layers into optimized message sequences
    that maximize DeepSeek's context caching effectiveness.
    """

    def __init__(self, working_buffer: Optional['WorkingBuffer'] = None,
                 rolling_summary: Optional['RollingSummary'] = None,
                 profile_store: Optional['ProfileStore'] = None,
                 rag_store: Optional['RAGStore'] = None):
        """
        Initialize context assembler.

        Args:
            working_buffer: Working memory buffer instance
            rolling_summary: Rolling summary instance
            profile_store: User profile store instance
            rag_store: RAG store instance
        """
        self.working_buffer = working_buffer or (WorkingBuffer() if WorkingBuffer else None)
        self.rolling_summary = rolling_summary or (RollingSummary() if RollingSummary else None)
        self.profile_store = profile_store or (ProfileStore() if ProfileStore else None)
        self.rag_store = rag_store or (RAGStore() if RAGStore else None)

    def assemble(self, current_query: str, budget_tokens: int = 8000,
                include_rag: bool = True) -> Dict[str, Any]:
        """
        Assemble complete context for LLM inference.

        Args:
            current_query: Current user query
            budget_tokens: Total token budget
            include_rag: Whether to include RAG retrieval

        Returns:
            Dictionary containing assembled messages and metadata
        """
        start_time = time.time()

        # Initialize message list
        messages = []

        # Reserve tokens for different sections
        system_budget = min(500, budget_tokens // 10)  # 10% for system
        summary_budget = min(1000, budget_tokens // 8)  # 12.5% for summary
        task_budget = min(800, budget_tokens // 12)    # ~8% for task state
        buffer_budget = budget_tokens // 3             # 33% for working buffer
        rag_budget = budget_tokens // 4               # 25% for RAG

        # 1. System/Instruction (highest priority)
        system_msg = self._create_system_message()
        messages.append(system_msg)

        # 2. Rolling Summary (stable, cacheable)
        if self.rolling_summary:
            summary_content = self.rolling_summary.get_summary()
            if summary_content:
                # Truncate to budget
                truncated_summary = self._truncate_to_budget(summary_content, summary_budget)
                if truncated_summary:
                    summary_msg = Message(
                        role=Role.SYSTEM,
                        content=f"对话历史摘要：{truncated_summary}"
                    )
                    messages.append(summary_msg)

        # 3. Task State & TODO Summary
        task_summary = self._create_task_summary()
        if task_summary:
            truncated_task = self._truncate_to_budget(task_summary, task_budget)
            if truncated_task:
                task_msg = Message(
                    role=Role.SYSTEM,
                    content=f"当前任务状态：{truncated_task}"
                )
                messages.append(task_msg)

        # 4. Working Buffer (recent conversation)
        if self.working_buffer:
            buffer_messages = self._assemble_working_buffer(buffer_budget)
            messages.extend(buffer_messages)

        # 5. RAG Evidence (query-relevant)
        if include_rag and self.rag_store and current_query:
            rag_messages = self._assemble_rag_evidence(current_query, rag_budget)
            messages.extend(rag_messages)

        # Calculate final statistics
        assembly_time = time.time() - start_time
        total_chars = sum(len(str(getattr(msg, 'content', ''))) for msg in messages)
        estimated_tokens = total_chars // 4

        # Calculate section breakdown
        section_tokens = self._calculate_section_tokens(messages)

        result = {
            'messages': messages,
            'metadata': {
                'total_chars': total_chars,
                'estimated_tokens': estimated_tokens,
                'budget_tokens': budget_tokens,
                'utilization_percent': (estimated_tokens / budget_tokens) * 100 if budget_tokens > 0 else 0,
                'section_breakdown': section_tokens,
                'assembly_time_ms': assembly_time * 1000,
                'cacheable_prefix_len': self._estimate_cacheable_prefix(messages),
                'message_count': len(messages)
            }
        }

        return result

    def update_memories(self, new_messages: List[Dict[str, Any]],
                       dropped_content: Optional[str] = None) -> None:
        """
        Update memory layers after conversation progress.

        Args:
            new_messages: New conversation messages
            dropped_content: Content dropped from working buffer
        """
        # Update working buffer
        if self.working_buffer:
            for msg in new_messages:
                self.working_buffer.append_turn(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    turn_type='conversation'
                )

        # Update rolling summary if content was dropped
        if dropped_content and self.rolling_summary:
            current_context = self._extract_current_context()
            self.rolling_summary.update_summary(dropped_content, current_context)

    def add_tool_result_to_memory(self, tool_name: str, result: Dict[str, Any]) -> None:
        """
        Add tool execution result to appropriate memory layers.

        Args:
            tool_name: Name of the executed tool
            result: Tool execution result
        """
        # Always add summary to working buffer
        if self.working_buffer:
            self.working_buffer.append_tool_result(tool_name, result)

        # Add full content to RAG for information-gathering tools
        if self.rag_store and self._is_information_tool(tool_name):
            content = self._extract_content_for_rag(tool_name, result)
            if content:
                metadata = {
                    'tool': tool_name,
                    'timestamp': time.time(),
                    'source_type': 'tool_execution'
                }
                self.rag_store.add_document(content, f"tool_{tool_name}", metadata)

    def add_user_profile_fact(self, key: str, value: Any,
                            confidence: float = 0.7, ttl_days: Optional[int] = None) -> None:
        """
        Add a fact to user profile.

        Args:
            key: Fact key
            value: Fact value
            confidence: Confidence score
            ttl_days: Time-to-live in days
        """
        if self.profile_store:
            self.profile_store.upsert_profile_fact(key, value, confidence, "conversation", ttl_days)

    def _create_system_message(self) -> Dict[str, Any]:
        """Create the system instruction message."""
        system_content = """你是DigitalClone，一个强大的AI助手。

能力：
- 可以使用各种工具执行任务
- 进行复杂的多步骤规划
- 管理长时间对话的上下文

工作模式：
- 接收用户指令后，先理解需求
- 如需要工具调用，会明确标识
- 保持对话连贯性和逻辑性
- 遇到问题时会主动寻求澄清

工具使用原则：
- 只使用系统提供的工具
- 工具调用前会说明用途
- 执行结果会用于后续推理

回复风格：
- 清晰简洁，不啰嗦
- 逻辑严谨，有条理
- 遇到不确定时会明确说明"""

        return Message(role=Role.SYSTEM, content=system_content)

    def _create_task_summary(self) -> str:
        """Create a summary of current task state."""
        if not self.working_buffer:
            return ""

        task_state = self.working_buffer.get_task_state('current_task')
        if not task_state:
            return ""

        summary_parts = []
        summary_parts.append(f"当前任务：{task_state.get('goal', '未指定')}")

        if task_state.get('progress'):
            summary_parts.append(f"进度：{task_state['progress']}")

        if task_state.get('next_steps'):
            steps = task_state['next_steps'][:3]  # Limit to 3 steps
            summary_parts.append(f"下一步：{', '.join(steps)}")

        return " | ".join(summary_parts)

    def _assemble_working_buffer(self, budget_tokens: int) -> List[Dict[str, Any]]:
        """Assemble messages from working buffer within budget."""
        if not self.working_buffer:
            return []

        messages = []
        current_tokens = 0
        max_chars = budget_tokens * 4

        # Get recent turns (most recent first, but we'll reverse them)
        recent_turns = self.working_buffer.get_recent_turns(20)

        for turn in reversed(recent_turns):  # Process in chronological order
            content_chars = len(turn.content)

            if current_tokens + (content_chars // 4) > budget_tokens:
                # Would exceed budget, stop here
                break

            # Convert to Message format
            msg = Message(
                role=Role(turn.role) if hasattr(Role, turn.role.upper()) else Role.USER,
                content=turn.content
            )
            messages.append(msg)
            current_tokens += content_chars // 4

        return messages

    def _assemble_rag_evidence(self, query: str, budget_tokens: int) -> List[Dict[str, Any]]:
        """Assemble relevant RAG evidence for the query."""
        if not self.rag_store:
            return []

        # Search for relevant chunks
        results = self.rag_store.search(query, k=3)

        if not results:
            return []

        # Deduplicate and assemble evidence
        seen_fingerprints = set()
        evidence_parts = []
        current_chars = 0
        max_chars = budget_tokens * 4

        for result in results:
            chunk = result['chunk']
            content = result['content']

            # Skip duplicates
            if chunk.fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(chunk.fingerprint)

            # Check if we have space
            if current_chars + len(content) > max_chars:
                # Truncate if needed
                available_chars = max_chars - current_chars
                if available_chars > 100:  # Only add if we have meaningful space
                    content = content[:available_chars - 3] + "..."
                else:
                    break

            evidence_parts.append(f"来自 {chunk.source}：{content}")
            current_chars += len(content)

        if evidence_parts:
            evidence_content = "\n\n".join(evidence_parts)
            evidence_msg = Message(
                role=Role.SYSTEM,
                content=f"相关参考信息：\n{evidence_content}"
            )
            return [evidence_msg]

        return []

    def _truncate_to_budget(self, content: str, budget_tokens: int) -> str:
        """Truncate content to fit within token budget."""
        max_chars = budget_tokens * 4
        if len(content) <= max_chars:
            return content

        return content[:max_chars - 3] + "..."

    def _calculate_section_tokens(self, messages: List) -> Dict[str, int]:
        """Calculate token usage breakdown by section."""
        breakdown = {
            'system': 0,
            'summary': 0,
            'task': 0,
            'buffer': 0,
            'rag': 0
        }

        for i, msg in enumerate(messages):
            content = getattr(msg, 'content', '')
            tokens = len(content) // 4

            if i == 0:
                breakdown['system'] = tokens
            elif '历史摘要' in content:
                breakdown['summary'] = tokens
            elif '当前任务状态' in content:
                breakdown['task'] = tokens
            elif '相关参考信息' in content:
                breakdown['rag'] = tokens
            else:
                breakdown['buffer'] += tokens

        return breakdown

    def _estimate_cacheable_prefix(self, messages: List) -> int:
        """Estimate the length of cacheable prefix in characters."""
        # System message + first summary message are typically stable
        cacheable_chars = 0

        if len(messages) >= 1:
            cacheable_chars += len(getattr(messages[0], 'content', ''))

        if len(messages) >= 2 and '历史摘要' in getattr(messages[1], 'content', ''):
            cacheable_chars += len(getattr(messages[1], 'content', ''))

        return cacheable_chars

    def _extract_current_context(self) -> Optional[str]:
        """Extract current conversation context for summary updates."""
        if not self.working_buffer:
            return None

        recent_turns = self.working_buffer.get_recent_turns(3)
        context_parts = []

        for turn in recent_turns:
            if turn.turn_type == 'conversation':
                context_parts.append(f"{turn.role}: {turn.content[:100]}")

        return " | ".join(context_parts) if context_parts else None

    def _is_information_tool(self, tool_name: str) -> bool:
        """Check if a tool typically gathers information."""
        info_tools = {'file_read', 'web_search', 'web_read', 'rag_search', 'tabular_qa'}
        return tool_name in info_tools

    def _extract_content_for_rag(self, tool_name: str, result: Dict[str, Any]) -> Optional[str]:
        """Extract content from tool result for RAG storage."""
        if not result.get('success'):
            return None

        if tool_name == 'file_read':
            return result.get('value', {}).get('content')
        elif tool_name in ['web_search', 'web_read']:
            return result.get('value', {}).get('content', result.get('summary'))
        elif tool_name == 'tabular_qa':
            return result.get('value', {}).get('answer', result.get('summary'))
        elif tool_name == 'rag_search':
            # RAG results are already in RAG, don't circular reference
            return None

        return result.get('summary')
