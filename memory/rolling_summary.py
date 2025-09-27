"""
Rolling Summary Memory - Recursive summarization of conversation history.

This implements recursive summarization where past conversations are
compressed into hierarchical summaries that maintain semantic integrity.
"""

import time
from typing import Dict, Any, Optional

# Conditional imports for LLM client
try:
    from backend.llm_interface import create_llm_client
except ImportError:
    create_llm_client = None


class RollingSummary:
    """
    Rolling summary memory using recursive summarization.

    Maintains a hierarchical summary of conversation history that gets
    recursively updated as new content is added.
    """

    def __init__(self, max_tokens: int = 800):
        """
        Initialize rolling summary.

        Args:
            max_tokens: Maximum token budget for the summary
        """
        self.max_tokens = max_tokens
        self.current_summary = ""
        self.last_updated = time.time()
        self.update_count = 0

    def update_summary(self, dropped_span: str, current_context: Optional[str] = None) -> str:
        """
        Update the rolling summary by incorporating new dropped content.

        Args:
            dropped_span: The conversation span being dropped from working memory
            current_context: Current task context for better summarization

        Returns:
            Updated summary string
        """
        if not dropped_span.strip():
            return self.current_summary

        # Rule-based compression first
        compressed_span = self._rule_based_compress(dropped_span)

        # If we have an LLM client, use it for better compression
        if create_llm_client and len(compressed_span) > 200:
            compressed_span = self._llm_based_compress(compressed_span, current_context)

        # Recursive merge with existing summary
        if self.current_summary:
            self.current_summary = self._merge_summaries(self.current_summary, compressed_span)
        else:
            self.current_summary = compressed_span

        # Enforce token limits
        self._enforce_limits()

        self.last_updated = time.time()
        self.update_count += 1

        return self.current_summary

    def get_summary(self) -> str:
        """Get the current rolling summary."""
        return self.current_summary

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        estimated_tokens = len(self.current_summary) // 4

        return {
            'summary_length_chars': len(self.current_summary),
            'estimated_tokens': estimated_tokens,
            'max_tokens': self.max_tokens,
            'last_updated': self.last_updated,
            'update_count': self.update_count,
            'utilization_percent': (estimated_tokens / self.max_tokens) * 100 if self.max_tokens > 0 else 0
        }

    def clear(self) -> None:
        """Clear the rolling summary."""
        self.current_summary = ""
        self.last_updated = time.time()
        self.update_count = 0

    def _rule_based_compress(self, text: str) -> str:
        """
        Apply rule-based compression to reduce text length while preserving key information.

        Args:
            text: Text to compress

        Returns:
            Compressed text
        """
        if not text or len(text) < 100:
            return text

        lines = text.split('\n')
        compressed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip redundant conversational elements
            if any(skip in line.lower() for skip in [
                '好的', '谢谢', '请问', '你好', '再见', '明白了', '知道了',
                '是的', '不是', '可以', '当然', '没问题'
            ]):
                continue

            # Keep important elements
            if any(keep in line.lower() for keep in [
                '任务', '目标', '计划', '步骤', '完成', '失败', '错误',
                '结果', '总结', '分析', '结论', '建议', '决策'
            ]):
                compressed_lines.append(line)
            elif len(line) > 50:  # Keep substantial lines
                compressed_lines.append(line)

        # Limit to reasonable length
        compressed = '\n'.join(compressed_lines)
        if len(compressed) > 500:
            compressed = compressed[:500] + "..."

        return compressed

    def _llm_based_compress(self, text: str, context: Optional[str] = None) -> str:
        """
        Use LLM to create a more intelligent compression.

        Args:
            text: Text to compress
            context: Current task context

        Returns:
            LLM-compressed text
        """
        if not create_llm_client:
            return text

        try:
            client = create_llm_client("chat")

            context_str = f"\n当前任务上下文：{context}" if context else ""

            prompt = f"""请将以下对话内容压缩为100字以内，保留关键信息、决策、结果和重要事实：{context_str}

内容：
{text[:1500]}...

压缩要求：
- 保留实体名称、数值、引用
- 保留决策和结果
- 合并相似信息
- 直接输出压缩结果，不要其他内容"""

            response = client.generate(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=150  # Limit output length
            )

            if response.content and len(response.content.strip()) > 0:
                compressed = response.content.strip()
                # Ensure it's not longer than original
                if len(compressed) < len(text):
                    return compressed

        except Exception as e:
            # Log error but continue with rule-based compression
            print(f"LLM compression failed: {e}")

        return text

    def _merge_summaries(self, existing: str, new: str) -> str:
        """
        Recursively merge existing summary with new compressed content.

        Args:
            existing: Existing summary
            new: New content to merge

        Returns:
            Merged summary
        """
        if not existing:
            return new
        if not new:
            return existing

        # If combined length is reasonable, just concatenate
        combined = f"{existing}\n{new}"
        if len(combined) <= 1000:
            return combined

        # Need to merge intelligently
        if create_llm_client:
            try:
                client = create_llm_client("chat")

                merge_prompt = f"""请合并以下两个摘要，创建一个连贯的整体摘要，控制在150字以内：

现有摘要：
{existing}

新增内容：
{new}

要求：
- 保持时间顺序
- 合并重复信息
- 保留关键决策和结果
- 直接输出合并结果"""

                response = client.generate(
                    messages=[{"role": "user", "content": merge_prompt}],
                    stream=False,
                    max_tokens=200
                )

                if response.content:
                    return response.content.strip()

            except Exception as e:
                print(f"Summary merge failed: {e}")

        # Fallback: simple concatenation with length limit
        return combined[:800] + "..."

    def _enforce_limits(self) -> None:
        """Enforce token limits on the summary."""
        estimated_tokens = len(self.current_summary) // 4

        if estimated_tokens > self.max_tokens:
            # Truncate to fit
            max_chars = self.max_tokens * 4
            if len(self.current_summary) > max_chars:
                self.current_summary = self.current_summary[:max_chars - 3] + "..."
