"""
Micro Decision Maker - Reasoner for constrained decision scenarios.

This implements micro-decision making using DeepSeek Reasoner with
strict token limits and natural language output for scenarios requiring
careful evaluation.
"""

import time
from typing import Optional, Dict, Any, List

# Conditional imports
try:
    from backend.llm_interface import create_llm_client
    from backend.message_types import Message, Role
except ImportError:
    # Fallback for testing
    create_llm_client = None
    Message = Dict
    Role = str


class MicroDecider:
    """
    Micro-decision maker using DeepSeek Reasoner.

    Optimized for scenarios requiring evaluation of multiple options
    with limited context and strict response constraints.
    """

    def __init__(self, max_tokens: int = 200, timeout_seconds: int = 30):
        """
        Initialize micro decider.

        Args:
            max_tokens: Maximum tokens for micro-decisions
            timeout_seconds: Timeout for Reasoner calls
        """
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def decide(self, goal: str, facts: str, candidates: List[str]) -> Optional[int]:
        """
        Make a micro-decision from multiple candidates.

        Args:
            goal: Decision goal/target
            facts: Key facts/context (compressed)
            candidates: List of candidate options

        Returns:
            Selected candidate index (0-based) or None if failed/timeout
        """
        if not candidates or len(candidates) < 2:
            return 0 if candidates else None

        # Construct micro-prompt
        prompt = self._construct_micro_prompt(goal, facts, candidates)

        # Check token budget
        estimated_tokens = len(prompt) // 4
        if estimated_tokens > self.max_tokens:
            # Truncate if needed
            available_chars = (self.max_tokens - 20) * 4  # Reserve for response
            prompt = prompt[:available_chars] + "..."

        try:
            # Call Reasoner with constraints
            client = create_llm_client("reasoner")

            messages = [Message(role=Role.USER, content=prompt)]

            start_time = time.time()
            response = client.generate(
                messages=messages,
                stream=False,
                response_format=None,  # Natural language, not JSON
                max_tokens=50  # Very constrained response
            )

            elapsed = time.time() - start_time

            if elapsed > self.timeout_seconds:
                return None  # Timeout

            # Parse natural language response
            choice = self._parse_choice_response(response.content, len(candidates))

            return choice

        except Exception as e:
            print(f"Micro-decision failed: {e}")
            return None

    def _construct_micro_prompt(self, goal: str, facts: str, candidates: List[str]) -> str:
        """
        Construct a micro-prompt for decision making.

        Args:
            goal: Decision goal
            facts: Key facts
            candidates: Candidate options

        Returns:
            Formatted micro-prompt
        """
        # Truncate inputs to fit budget
        goal = goal[:50] if len(goal) > 50 else goal
        facts = facts[:80] if len(facts) > 80 else facts

        prompt_parts = [
            f"目标:{goal}",
            f"事实:{facts}",
        ]

        # Add candidates with strict character limits
        for i, candidate in enumerate(candidates[:5]):  # Max 5 candidates
            truncated_candidate = candidate[:25] if len(candidate) > 25 else candidate
            prompt_parts.append(f"{i}:{truncated_candidate}")

        prompt_parts.append("规则:只回数字编号，不解释")

        return " ".join(prompt_parts)

    def _parse_choice_response(self, response: str, num_candidates: int) -> Optional[int]:
        """
        Parse natural language choice response.

        Args:
            response: Reasoner response
            num_candidates: Number of available candidates

        Returns:
            Selected candidate index or None if invalid
        """
        if not response:
            return None

        response = response.strip()

        # Try to extract first digit
        import re
        digits = re.findall(r'\d+', response)

        if digits:
            choice = int(digits[0])
            if 0 <= choice < num_candidates:
                return choice

        # Check for explicit "no change" or similar
        if any(word in response.lower() for word in ['不变', '保持', 'no change', '无需']):
            return 0  # Default to first option

        return None  # Invalid response

    def get_stats(self) -> Dict[str, Any]:
        """Get micro-decider statistics."""
        return {
            'max_tokens': self.max_tokens,
            'timeout_seconds': self.timeout_seconds,
            'decision_type': 'micro_reasoner'
        }


# Convenience function for backward compatibility
def decide_with_reasoner(goal: str, facts: str, candidates: List[str]) -> Optional[int]:
    """
    Convenience function for micro-decisions.

    Args:
        goal: Decision goal
        facts: Key facts
        candidates: Candidate options

    Returns:
        Selected candidate index or None
    """
    decider = MicroDecider()
    return decider.decide(goal, facts, candidates)
