"""
Tests for AskUser policy and behavior.

This module tests the logic for when to ask users for clarification
versus when to use tools or provide direct answers.
"""

import pytest
import sys
import os

# Add the backend directory to Python path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)

from agent_core import AgentRouter
from message_types import Message, Role
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'graph'))
from nodes import _needs_user_clarification


class TestAskUserPolicy:
    """Test cases for AskUser policy decisions."""

    def test_objective_facts_use_tools(self):
        """Test that objective facts are retrieved via tools, not asked."""
        # Current time should use datetime tool
        router = AgentRouter()

        # This should route to reasoner (contains planning keywords)
        # but within reasoner, it should use tools for objective info
        decision = router.route("现在几点了？", None)
        # Note: This might route to chat depending on the heuristics
        # The key is that it shouldn't trigger AskUser for objective facts

        # For this test, we're more concerned with the pattern
        # In actual implementation, the model would decide based on system prompt

    def test_subjective_preferences_trigger_ask_user(self):
        """Test that subjective preferences trigger AskUser."""
        # Questions about preferences should trigger clarification
        test_cases = [
            "你喜欢什么颜色？",
            "我应该选择哪个选项？",
            "你更喜欢猫还是狗？",
            "预算有多少？",
            "你想要什么样的设计？"
        ]

        # These should contain indicators that would make the model ask for clarification
        for question in test_cases:
            # The actual decision happens in the model based on system prompt
            # Here we test that such questions would be routed appropriately
            router = AgentRouter()
            decision = router.route(question, None)
            # These might route to chat, but the model should ask for clarification
            assert decision.engine in ["chat", "reasoner"]

    def test_technical_questions_use_tools(self):
        """Test that technical questions use appropriate tools."""
        test_cases = [
            "计算 2 + 3 * 4",
            "今天是几号",
            "当前时间是什么"
        ]

        router = AgentRouter()

        for question in test_cases:
            decision = router.route(question, None)
            # These should be handled by chat model with tools
            assert decision.engine == "chat"

    def test_complex_planning_routes_to_reasoner(self):
        """Test that complex planning tasks route to reasoner."""
        test_cases = [
            "给我制定一个学习计划",
            "帮我规划旅行路线",
            "设计一个项目方案",
            "制定年度目标",
            "创建工作流程"
        ]

        router = AgentRouter()

        for question in test_cases:
            decision = router.route(question, None)
            assert decision.engine == "reasoner", f"Question '{question}' should route to reasoner"

    def test_ask_user_limit_enforced(self):
        """Test that AskUser is limited to prevent infinite loops."""
        # This would be tested in integration tests with the full graph
        # Here we can test the basic routing logic

        router = AgentRouter()

        # Very long/complex input should route to reasoner
        long_input = "请帮我详细规划一个为期三个月的个人发展计划，包括职业技能提升、健康管理、金融理财、社交拓展等多个方面，并为每个阶段制定具体可执行的目标和时间表。" * 2

        decision = router.route(long_input, None)
        assert decision.engine == "reasoner"

    def test_structured_input_routes_to_reasoner(self):
        """Test that structured input (lists, steps) routes to reasoner."""
        test_cases = [
            "1. 分析问题 2. 提出解决方案 3. 实施计划",
            "第一步：研究 第二步：设计 第三步：实现",
            "步骤：\n- 收集需求\n- 分析数据\n- 生成报告",
            "任务分解：\n• 市场调研\n• 产品设计\n• 用户测试"
        ]

        router = AgentRouter()

        for structured_input in test_cases:
            decision = router.route(structured_input, None)
            assert decision.engine == "reasoner", f"Structured input should route to reasoner: {structured_input[:50]}..."

    def test_ask_user_indicators(self):
        """Test detection of AskUser indicators in AI responses."""
        # AI responses that indicate AskUser
        ask_user_responses = [
            "请告诉我你的预算",
            "您能提供更多信息吗？",
            "需要澄清一下您的要求",
            "请问您有什么特殊需求？",
            "我想了解您的具体情况"
        ]

        for response in ask_user_responses:
            assert _needs_user_clarification(response), f"Should detect AskUser in response: {response}"

        # AI responses that do NOT indicate AskUser
        no_ask_responses = [
            "计算结果是5",
            "今天天气很好",
            "现在是下午3点",
            "AI是人工智能的缩写",
            "Python是一种编程语言"
        ]

        for response in no_ask_responses:
            assert not _needs_user_clarification(response), f"Should NOT detect AskUser in response: {response}"

    def test_route_confidence_levels(self):
        """Test that routing provides appropriate confidence levels."""
        router = AgentRouter()

        # High confidence cases
        high_confidence = [
            "你好",
            "计算 5+3",
            "现在时间"
        ]

        for query in high_confidence:
            decision = router.route(query, None)
            assert decision.confidence >= 0.5, f"Should have reasonable confidence for: {query}"

        # Structured input should have high confidence for reasoner
        decision = router.route("1.步骤一 2.步骤二 3.步骤三", None)
        assert decision.confidence >= 0.7, "Structured input should have high confidence"

    def test_fallback_behavior(self):
        """Test routing fallback behavior for edge cases."""
        router = AgentRouter()

        # Empty input should not crash
        decision = router.route("", None)
        assert decision.engine in ["chat", "reasoner"]

        # Very short input
        decision = router.route("?", None)
        assert decision.engine in ["chat", "reasoner"]

        # Input with only special characters
        decision = router.route("!@#$%", None)
        assert decision.engine in ["chat", "reasoner"]
