"""
Tests for routing logic in the agent core.
"""

import pytest
import sys
import os

# Add the backend directory to Python path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)

from ..agent_core import AgentRouter
from ..message_types import ConversationContext, RouteDecision


def test_router_initialization():
    """Test that router initializes correctly."""
    router = AgentRouter()
    assert router is not None
    assert hasattr(router, 'route')


def test_chat_routing():
    """Test routing decisions that should use chat model."""
    router = AgentRouter()
    context = ConversationContext()

    # Simple greeting
    decision = router.route("你好，今天怎么样？", context)
    assert decision.engine == "chat"
    assert decision.confidence >= 0.5

    # Simple question
    decision = router.route("今天天气不错，讲个笑话", context)
    assert decision.engine == "chat"

    # Short question
    decision = router.route("1+1等于几？", context)
    assert decision.engine == "chat"


def test_reasoner_routing_keywords():
    """Test routing to reasoner based on keywords."""
    router = AgentRouter()
    context = ConversationContext()

    # Planning keywords
    decision = router.route("给我做个计划", context)
    assert decision.engine == "reasoner"

    decision = router.route("帮我制定方案", context)
    assert decision.engine == "reasoner"

    # Complex task keywords
    decision = router.route("需要系统性分析这个问题", context)
    assert decision.engine == "reasoner"

    decision = router.route("这是一个复杂的任务", context)
    assert decision.engine == "reasoner"


def test_reasoner_routing_structure():
    """Test routing to reasoner based on structural patterns."""
    router = AgentRouter()
    context = ConversationContext()

    # Numbered list
    decision = router.route("请帮我：1. 分析问题 2. 提出解决方案 3. 制定实施计划", context)
    assert decision.engine == "reasoner"

    # Flow arrows
    decision = router.route("数据处理 -> 分析结果 -> 生成报告", context)
    assert decision.engine == "reasoner"


def test_reasoner_routing_length():
    """Test routing to reasoner based on input length."""
    router = AgentRouter()
    context = ConversationContext()

    # Short input (should be chat)
    short_input = "简单的问候"
    decision = router.route(short_input, context)
    assert decision.engine == "chat"

    # Long input (should be reasoner)
    long_input = "我需要你帮助我制定一个详细的学习计划。这个计划应该包含多个阶段，每个阶段都有具体的目标和任务。我希望能够系统性地提高我的编程技能，特别关注Python开发、算法思维和软件工程实践等方面。计划应该考虑时间安排、学习资源、练习项目以及进度评估等多个方面。请给我一个全面的建议。" * 3
    decision = router.route(long_input, context)
    assert decision.engine == "reasoner"


def test_explicit_user_requests():
    """Test routing based on explicit user complexity requests."""
    router = AgentRouter()
    context = ConversationContext()

    decision = router.route("请详细分析这个复杂问题", context)
    assert decision.engine == "reasoner"

    decision = router.route("给我一个全面的解决方案", context)
    assert decision.engine == "reasoner"

    decision = router.route("系统性地处理这个任务", context)
    assert decision.engine == "reasoner"


def test_routing_confidence():
    """Test that routing provides appropriate confidence scores."""
    router = AgentRouter()
    context = ConversationContext()

    # High confidence for clear cases
    decision = router.route("今天天气怎么样", context)
    assert decision.confidence >= 0.8

    # Lower confidence for ambiguous cases
    decision = router.route("帮我做个什么事", context)  # Somewhat ambiguous
    assert decision.confidence >= 0.0
    assert decision.confidence <= 1.0


def test_multiple_operations_detection():
    """Test detection of multiple tool operations."""
    router = AgentRouter()
    context = ConversationContext()

    # Single operation (chat)
    decision = router.route("帮我算个数学题", context)
    assert decision.engine == "chat"

    # Multiple operations (reasoner)
    decision = router.route("先搜索一下信息，然后计算结果，最后生成报告", context)
    assert decision.engine == "reasoner"


def test_router_with_context():
    """Test routing with conversation context."""
    router = AgentRouter()

    # Empty context
    context = ConversationContext()
    decision = router.route("简单问题", context)
    assert decision.engine == "chat"

    # Context with previous messages
    context = ConversationContext()
    # In a more sophisticated test, we could add previous messages
    # For now, just test that context parameter works
    decision = router.route("继续讨论", context)
    assert isinstance(decision, RouteDecision)


def test_route_decision_structure():
    """Test that route decisions have proper structure."""
    router = AgentRouter()
    context = ConversationContext()

    decision = router.route("测试输入", context)

    assert hasattr(decision, 'engine')
    assert hasattr(decision, 'reason')
    assert hasattr(decision, 'confidence')

    assert decision.engine in ['chat', 'reasoner']
    assert isinstance(decision.reason, str)
    assert len(decision.reason) > 0
    assert isinstance(decision.confidence, float)
    assert 0.0 <= decision.confidence <= 1.0


@pytest.mark.parametrize("input_text,expected_engine", [
    ("你好", "chat"),
    ("今天天气真好", "chat"),
    ("帮我算算2+2", "chat"),
    ("制定学习计划", "reasoner"),
    ("系统分析问题", "reasoner"),
    ("1.步骤一 2.步骤二 3.步骤三", "reasoner"),
    ("数据 -> 处理 -> 输出", "reasoner"),
])
def test_routing_parametrized(input_text, expected_engine):
    """Parametrized test for various routing scenarios."""
    router = AgentRouter()
    context = ConversationContext()

    decision = router.route(input_text, context)
    assert decision.engine == expected_engine
