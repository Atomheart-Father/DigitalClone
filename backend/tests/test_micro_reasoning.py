"""
Tests for text processing utilities in the agent core.
"""

import sys
import os

# Add the backend directory to Python path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)

# Add the graph directory to Python path
graph_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'graph')
sys.path.insert(0, graph_dir)

from graph.nodes import truncate_text


def test_truncate_text():
    """Test text truncation utility."""
    # No truncation needed
    assert truncate_text("short text", 20) == "short text"

    # Truncation with suffix
    long_text = "This is a very long text that should be truncated"
    result = truncate_text(long_text, 20)
    assert len(result) <= 20
    assert result.endswith("...")
    assert result.startswith("This is a very")

    # Custom suffix
    result = truncate_text(long_text, 20, "[CUT]")
    assert len(result) <= 20
    assert result.endswith("[CUT]")
    assert result.startswith("This is a very")

    # Edge case: text exactly at limit
    exact_text = "exactly 20 chars!!"
    assert truncate_text(exact_text, 20) == exact_text

    # Edge case: very short limit
    result = truncate_text("hello", 2)
    assert result == "..."  # 2-3 = -1, so keep_chars = 0, result = "..."
