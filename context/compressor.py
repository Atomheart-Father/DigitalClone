"""
Compressor - Intelligent text compression with fidelity preservation.

This implements the compression strategy described in the design:
- Rule-based compression first (remove redundancy, keep entities/numbers/references)
- Model-based compression for better preservation (LLM-generated summaries)
- Fidelity evaluation and monitoring
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Conditional imports for LLM client
try:
    from backend.llm_interface import create_llm_client
except ImportError:
    create_llm_client = None


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: str
    entities_preserved: List[str]
    quality_score: float


class TextCompressor:
    """
    Intelligent text compressor that preserves semantic fidelity.

    Implements the compression strategy: rule-based first, then model-based,
    with monitoring of compression quality.
    """

    def __init__(self, max_tokens: int = 8000):
        """
        Initialize compressor.

        Args:
            max_tokens: Maximum token budget
        """
        self.max_tokens = max_tokens

        # Patterns for entity extraction
        self.entity_patterns = {
            'numbers': re.compile(r'\b\d+\.?\d*\b'),
            'dates': re.compile(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'file_paths': re.compile(r'[./\\][\w./\\-]+(?:\.\w+)?'),
            'quoted_text': re.compile(r'"([^"]*)"|\'([^\']*)\''),
        }

    def compress_text(self, text: str, target_tokens: int,
                     method: str = "hybrid") -> CompressionResult:
        """
        Compress text to fit within token budget while preserving fidelity.

        Args:
            text: Text to compress
            target_tokens: Target token count
            method: Compression method ("rule", "model", "hybrid")

        Returns:
            CompressionResult with metrics
        """
        original_tokens = self._estimate_tokens(text)
        entities = self._extract_entities(text)

        if original_tokens <= target_tokens:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                method="none",
                entities_preserved=entities,
                quality_score=1.0
            )

        compressed_text = text
        final_method = "rule"

        # Rule-based compression first
        if method in ["rule", "hybrid"]:
            compressed_text = self._rule_based_compress(text)
            final_method = "rule"

        compressed_tokens = self._estimate_tokens(compressed_text)

        # If still over budget and method allows, use model-based compression
        if compressed_tokens > target_tokens and method in ["model", "hybrid"]:
            compressed_text = self._model_based_compress(compressed_text, target_tokens)
            compressed_tokens = self._estimate_tokens(compressed_text)
            final_method = "hybrid" if method == "hybrid" else "model"

        # Final truncation if still over budget
        if compressed_tokens > target_tokens:
            compressed_text = self._truncate_to_tokens(compressed_text, target_tokens)
            compressed_tokens = target_tokens

        # Calculate quality metrics
        quality_score = self._calculate_quality_score(text, compressed_text, entities)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            method=final_method,
            entities_preserved=entities,
            quality_score=quality_score
        )

    def compress_conversation_history(self, messages: List[Dict[str, Any]],
                                     target_tokens: int) -> Tuple[List[Dict[str, Any]], CompressionResult]:
        """
        Compress conversation history to fit within token budget.

        Args:
            messages: List of conversation messages
            target_tokens: Target token count for entire history

        Returns:
            Tuple of (compressed_messages, compression_metrics)
        """
        if not messages:
            return messages, CompressionResult("", "", 0, 0, 1.0, "none", [], 1.0)

        # Calculate current token usage
        total_tokens = sum(self._estimate_tokens(str(msg.get('content', ''))) for msg in messages)

        if total_tokens <= target_tokens:
            return messages, CompressionResult("", "", total_tokens, total_tokens, 1.0, "none", [], 1.0)

        # Keep system messages and most recent exchanges
        compressed_messages = []

        # Always keep system messages
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        compressed_messages.extend(system_messages)

        # Keep recent user/assistant pairs
        recent_messages = []
        non_system_messages = [msg for msg in messages if msg.get('role') != 'system']

        # Try to keep last N pairs
        pairs_to_keep = 3
        while pairs_to_keep > 0 and len(recent_messages) < len(non_system_messages):
            # Get next pair (user + assistant response)
            pair_start = len(non_system_messages) - (pairs_to_keep * 2)
            if pair_start < 0:
                pair_start = 0

            pair_messages = non_system_messages[pair_start:]
            recent_messages = pair_messages
            break

        compressed_messages.extend(recent_messages)

        # If still over budget, compress individual message contents
        compressed_content = ""
        original_content = ""

        for msg in compressed_messages:
            content = str(msg.get('content', ''))
            original_content += content

            # Compress individual message if needed
            result = self.compress_text(content, target_tokens // len(compressed_messages))
            msg['content'] = result.compressed_text
            compressed_content += result.compressed_text

        # Calculate overall compression metrics
        original_tokens = self._estimate_tokens(original_content)
        compressed_tokens = self._estimate_tokens(compressed_content)

        metrics = CompressionResult(
            original_text=original_content,
            compressed_text=compressed_content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            method="conversation",
            entities_preserved=self._extract_entities(original_content),
            quality_score=self._calculate_quality_score(original_content, compressed_content,
                                                       self._extract_entities(original_content))
        )

        return compressed_messages, metrics

    def _rule_based_compress(self, text: str) -> str:
        """
        Apply rule-based compression to reduce text length.

        Args:
            text: Text to compress

        Returns:
            Compressed text
        """
        if not text or len(text) < 100:
            return text

        compressed = text

        # Remove excessive whitespace
        compressed = re.sub(r'\n\s*\n\s*\n+', '\n\n', compressed)
        compressed = re.sub(r' +', ' ', compressed)

        # Remove redundant conversational patterns
        redundant_patterns = [
            r'\b(好的?|嗯|啊|哦|哎呀|哎哟)\b',
            r'\b(谢谢?|感谢|多谢)\b',
            r'\b(请问?|请教|询问)\b',
            r'\b(我想|我要|我想说)\b',
            r'\b(也就是说|也就是说|换句话说)\b',
            r'\b(其实|实际上|事实上)\b',
        ]

        for pattern in redundant_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)

        # Keep important structural elements
        # Preserve numbered/bulleted lists
        lines = compressed.split('\n')
        preserved_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Keep lines with important indicators
            if any(indicator in line.lower() for indicator in [
                '任务', '目标', '计划', '步骤', '完成', '失败', '错误',
                '结果', '总结', '分析', '结论', '建议', '决策',
                '文件', '路径', '数据', '信息', '查询', '搜索'
            ]) or line.startswith(('- ', '* ', '• ', '1.', '2.', '3.')):
                preserved_lines.append(line)
            elif len(line) > 20:  # Keep substantial lines
                preserved_lines.append(line)

        compressed = '\n'.join(preserved_lines)

        # Final cleanup
        compressed = re.sub(r'\n\s*\n\s*\n+', '\n\n', compressed)

        return compressed.strip()

    def _model_based_compress(self, text: str, target_tokens: int) -> str:
        """
        Use LLM to create intelligent compression.

        Args:
            text: Text to compress
            target_tokens: Target token count

        Returns:
            LLM-compressed text
        """
        if not create_llm_client or len(text) < 200:
            return self._truncate_to_tokens(text, target_tokens)

        try:
            client = create_llm_client("chat")

            # Calculate target length (rough character estimate)
            target_chars = target_tokens * 4

            prompt = f"""请将以下内容压缩为不超过{target_chars}字符，同时保留关键信息、决策和结果：

内容：
{text[:3000]}...

要求：
- 保留实体名称、数值、引用和重要事实
- 保留决策过程和结论
- 保持逻辑连贯性
- 直接输出压缩结果，不要其他内容"""

            response = client.generate(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=min(target_tokens, 1000)
            )

            if response.content and len(response.content.strip()) > 0:
                compressed = response.content.strip()
                # Ensure it's actually shorter
                if len(compressed) < len(text):
                    return compressed

        except Exception as e:
            # Log error but fall back to rule-based
            print(f"Model compression failed: {e}")

        return self._rule_based_compress(text)

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract important entities that should be preserved during compression.

        Args:
            text: Text to analyze

        Returns:
            List of important entities
        """
        entities = []

        for name, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            entities.extend(matches[:5])  # Limit to avoid explosion

        # Also extract capitalized words (potential names)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized[:5])

        return list(set(entities))  # Remove duplicates

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 characters for most languages).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token budget.

        Args:
            text: Text to truncate
            max_tokens: Maximum token count

        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        last_sentence_end = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? '),
            truncated.rfind('\n')
        )

        if last_sentence_end > max_chars * 0.8:  # If sentence end is reasonably close
            return text[:last_sentence_end + 1]

        return truncated + "..."

    def _calculate_quality_score(self, original: str, compressed: str,
                                entities: List[str]) -> float:
        """
        Calculate compression quality score based on entity preservation and length.

        Args:
            original: Original text
            compressed: Compressed text
            entities: Important entities that should be preserved

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not original:
            return 1.0

        # Entity preservation score
        preserved_entities = sum(1 for entity in entities if entity in compressed)
        entity_score = preserved_entities / len(entities) if entities else 1.0

        # Length preservation score (prefer keeping more content when possible)
        length_ratio = len(compressed) / len(original)
        length_score = min(length_ratio * 2, 1.0)  # Boost shorter compressions slightly

        # Combine scores
        return (entity_score * 0.7) + (length_score * 0.3)
