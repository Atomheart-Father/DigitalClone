"""
Working Buffer Memory - Recent conversation turns and task state.

This implements the working memory layer that maintains recent conversation
turns, task progress, and key variables within token budget.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Conditional imports for LLM client
try:
    from backend.llm_interface import create_llm_client
except ImportError:
    create_llm_client = None


@dataclass
class ConversationTurn:
    """A single conversation turn with metadata."""
    role: str
    content: str
    timestamp: float
    turn_type: str  # 'user', 'assistant', 'tool_result', 'system'
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WorkingBuffer:
    """
    Working memory buffer for recent conversation turns and task state.

    Maintains recent N turns within token budget, with intelligent compression
    when limits are exceeded.
    """

    def __init__(self, max_tokens: int = 4000, max_turns: int = 20):
        """
        Initialize working buffer.

        Args:
            max_tokens: Maximum token budget for the buffer
            max_turns: Maximum number of conversation turns to keep
        """
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.task_state: Dict[str, Any] = {}

    def append_turn(self, role: str, content: str, turn_type: str = 'conversation',
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Append a new conversation turn to the buffer.

        Args:
            role: Message role ('user', 'assistant', 'system', 'tool')
            content: Message content
            turn_type: Type of turn for categorization
            metadata: Additional metadata for the turn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=time.time(),
            turn_type=turn_type,
            metadata=metadata or {}
        )

        self.turns.append(turn)

        # Enforce limits
        self._enforce_limits()

    def append_tool_result(self, tool_name: str, result: Dict[str, Any],
                          summary: Optional[str] = None) -> None:
        """
        Append a tool execution result to the buffer.

        Args:
            tool_name: Name of the executed tool
            result: Full tool execution result
            summary: Optional human-readable summary
        """
        # Create a compact summary for the buffer
        content = summary or self._create_tool_summary(tool_name, result)

        self.append_turn(
            role='tool',
            content=content,
            turn_type='tool_result',
            metadata={
                'tool_name': tool_name,
                'full_result': result,
                'summary_length': len(content)
            }
        )

    def update_task_state(self, key: str, value: Any) -> None:
        """
        Update task state information.

        Args:
            key: State key
            value: State value
        """
        self.task_state[key] = value

    def get_task_state(self, key: str, default: Any = None) -> Any:
        """
        Get task state value.

        Args:
            key: State key
            default: Default value if key not found
        """
        return self.task_state.get(key, default)

    def compact(self, target_tokens: Optional[int] = None) -> str:
        """
        Compact the buffer to fit within token limits.

        Args:
            target_tokens: Target token count (uses max_tokens if None)

        Returns:
            Summary of compression operations performed
        """
        if target_tokens is None:
            target_tokens = self.max_tokens

        # Estimate current token usage (rough approximation: 1 token ≈ 4 chars)
        current_chars = sum(len(turn.content) for turn in self.turns)
        current_tokens = current_chars // 4

        if current_tokens <= target_tokens:
            return "No compression needed"

        # Remove oldest turns first
        while self.turns and current_tokens > target_tokens:
            removed_turn = self.turns.pop(0)
            removed_chars = len(removed_turn.content)
            current_tokens -= removed_chars // 4

        # If still over limit, apply more aggressive compression
        if current_tokens > target_tokens:
            self._aggressive_compaction(target_tokens)

        return f"Compressed buffer: removed {len(self.turns)} turns, ~{current_tokens} tokens remaining"

    def get_recent_turns(self, n: int = 10) -> List[ConversationTurn]:
        """
        Get the most recent N conversation turns.

        Args:
            n: Number of recent turns to return

        Returns:
            List of recent conversation turns
        """
        return self.turns[-n:] if len(self.turns) > n else self.turns

    def get_buffer_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current buffer state.

        Returns:
            Dictionary with buffer statistics and content summary
        """
        total_chars = sum(len(turn.content) for turn in self.turns)
        estimated_tokens = total_chars // 4

        return {
            'turn_count': len(self.turns),
            'estimated_tokens': estimated_tokens,
            'total_chars': total_chars,
            'max_tokens': self.max_tokens,
            'max_turns': self.max_turns,
            'task_state_keys': list(self.task_state.keys()),
            'turn_types': [turn.turn_type for turn in self.turns[-5:]]  # Last 5 turns
        }

    def clear(self) -> None:
        """Clear all buffer contents."""
        self.turns.clear()
        self.task_state.clear()

    def _enforce_limits(self) -> None:
        """Enforce buffer limits by removing old content."""
        # Remove excess turns
        if len(self.turns) > self.max_turns:
            removed_count = len(self.turns) - self.max_turns
            self.turns = self.turns[-self.max_turns:]

        # Check token limits
        self.compact()

    def _aggressive_compaction(self, target_tokens: int) -> None:
        """
        Apply aggressive compression when basic removal isn't enough.

        Args:
            target_tokens: Target token count
        """
        if not create_llm_client:
            # Fallback: truncate content
            for turn in self.turns:
                if len(turn.content) > 200:
                    turn.content = turn.content[:200] + "..."
            return

        try:
            # Use LLM to create compact summaries
            client = create_llm_client("chat")

            for turn in self.turns:
                if len(turn.content) > 500:  # Only compress long content
                    summary_prompt = f"请将以下内容压缩为50字以内，同时保留关键信息：\n\n{turn.content[:1000]}"

                    response = client.generate(
                        messages=[{"role": "user", "content": summary_prompt}],
                        stream=False
                    )

                    if response.content:
                        turn.content = response.content.strip()
                        turn.metadata['compressed'] = True

        except Exception as e:
            # Fallback to simple truncation
            for turn in self.turns:
                if len(turn.content) > 200:
                    turn.content = turn.content[:200] + "..."

    def _create_tool_summary(self, tool_name: str, result: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of tool execution result.

        Args:
            tool_name: Name of the executed tool
            result: Tool execution result

        Returns:
            Compact summary string
        """
        if not result.get('success', False):
            return f"工具 {tool_name} 执行失败: {result.get('error', '未知错误')}"

        # Different summary strategies for different tools
        if tool_name == 'file_read':
            if 'value' in result and isinstance(result['value'], dict):
                content = result['value'].get('content', '')
                return f"读取文件成功: {len(content)} 字符内容"
        elif tool_name == 'web_search':
            if 'value' in result and isinstance(result['value'], dict):
                results = result['value'].get('results', [])
                return f"网络搜索完成: 找到 {len(results)} 个结果"
        elif tool_name == 'rag_search':
            if 'value' in result and isinstance(result['value'], dict):
                results = result['value'].get('results', [])
                return f"RAG搜索完成: 返回 {len(results)} 个相关片段"
        elif tool_name == 'calculator':
            return f"计算结果: {result.get('value', '未知')}"

        # Default summary
        summary = result.get('summary', '')
        if summary:
            return summary[:200] + ('...' if len(summary) > 200 else '')

        return f"工具 {tool_name} 执行成功"
