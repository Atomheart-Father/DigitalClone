#!/usr/bin/env python3
"""
Basic test for planner functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.llm_interface import create_llm_client
from backend.message_types import Message, Role
from graph.nodes import planner_generate_node
from graph.state import create_initial_state

def test_planner_json_generation():
    """Test planner JSON generation."""
    print("Testing planner JSON generation...")

    # Force mock client for testing
    import os
    original_env = os.environ.get('ENABLE_MOCK_CLIENT_IF_NO_KEY')
    os.environ['ENABLE_MOCK_CLIENT_IF_NO_KEY'] = 'true'

    # Temporarily remove API key to force mock
    from backend import config
    original_key = config.config.DEEPSEEK_API_KEY
    config.config.DEEPSEEK_API_KEY = None

    try:
        # Create test state
        state = create_initial_state("帮我制定一个学习计划，包括学习编程和数学")

        # Run planner generate node
        result = planner_generate_node(state)

        print("✅ Planner generate completed")
        plan = result.get('plan', [])
        print(f"Plan generated: {len(plan)} todos")

        if plan:
            for todo in plan:
                print(f"  - {todo.id}: {todo.title} ({todo.type.value})")

        return len(plan) > 0
    except Exception as e:
        print(f"❌ Planner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore environment
        if original_env is not None:
            os.environ['ENABLE_MOCK_CLIENT_IF_NO_KEY'] = original_env
        else:
            os.environ.pop('ENABLE_MOCK_CLIENT_IF_NO_KEY', None)
        config.config.DEEPSEEK_API_KEY = original_key

def test_llm_json_mode():
    """Test LLM JSON mode directly."""
    print("\nTesting LLM JSON mode...")

    client = create_llm_client('reasoner')

    test_prompt = """请制定一个学习计划，包括编程和数学。

请以JSON格式响应，只输出JSON，不要任何其他解释。

格式：
{
  "goal": "学习目标",
  "success_criteria": "成功标准",
  "todos": [
    {
      "id": "T1",
      "title": "任务标题",
      "why": "为什么需要",
      "type": "tool",
      "tool": "calculator",
      "input": {"expression": "1+1"},
      "expected_output": "结果",
      "needs": []
    }
  ]
}"""

    try:
        response = client.generate(
            messages=[Message(role=Role.USER, content=test_prompt)],
            system_prompt="You are a helpful planning assistant.",
            response_format={"type": "json_object"},
            stream=False
        )

        print("✅ LLM JSON mode works")
        print(f"Response: {response.content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ LLM JSON mode failed: {e}")
        return False

if __name__ == "__main__":
    print("Running basic planner tests...\n")

    test1_passed = test_llm_json_mode()
    test2_passed = test_planner_json_generation()

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
