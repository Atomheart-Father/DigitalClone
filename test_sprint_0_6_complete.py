#!/usr/bin/env python3
"""
Complete test suite for Sprint 0.6 - Tool-Executor Routing
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Force use real DeepSeek clients for testing blocking issues
os.environ["ENABLE_MOCK_CLIENT_IF_NO_KEY"] = "false"

def test_planner_json_generation():
    """Test planner generates proper JSON structure."""
    print("🧪 Testing planner JSON generation...")

    try:
        from graph.nodes import planner_generate_node
        from graph.state import create_initial_state

        state = create_initial_state('做一个公司竞品两周调研并拟方案')
        result = planner_generate_node(state)

        assert result.get('plan') is not None, "Plan should be generated"
        plan = result['plan']
        assert len(plan) > 0, "Plan should have todos"

        # Check plan structure
        for todo in plan:
            assert hasattr(todo, 'id'), "Todo should have id"
            assert hasattr(todo, 'title'), "Todo should have title"
            assert hasattr(todo, 'type'), "Todo should have type"
            assert todo.type.value in ['tool', 'chat', 'reason', 'write', 'research'], f"Invalid todo type: {todo.type.value}"

        print("✅ Planner JSON generation works")
        return True

    except Exception as e:
        print(f"❌ Planner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tool_call_protocol():
    """Test two-step tool calling protocol."""
    print("🧪 Testing tool call protocol...")

    try:
        from graph.nodes import call_tool_with_llm
        from graph.state import create_initial_state

        state = create_initial_state('test')
        result = call_tool_with_llm('chat', 'rag_search', '搜索测试信息', state)

        assert result.get('success') == True, f"Tool call should succeed, got: {result}"
        assert 'summary' in result, "Result should have summary"

        print("✅ Tool call protocol works")
        return True

    except Exception as e:
        print(f"❌ Tool call protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_executor_routing():
    """Test executor routing logic."""
    print("🧪 Testing executor routing...")

    from graph.nodes import resolve_executor
    from graph.state import TodoItem, TodoType

    # Test default routing
    todo1 = TodoItem(id="T1", title="Test", why="Test calculator", type=TodoType.TOOL, tool="calculator")
    executor1 = resolve_executor(todo1)
    assert executor1 == "chat", f"Calculator should route to chat, got {executor1}"

    # Test explicit executor
    todo2 = TodoItem(id="T2", title="Test", why="Test calculator with reasoner", type=TodoType.TOOL, tool="calculator", executor="reasoner")
    executor2 = resolve_executor(todo2)
    assert executor2 == "reasoner", f"Explicit executor should be respected, got {executor2}"

    print("✅ Executor routing works")
    return True

def test_tool_metadata():
    """Test tool metadata and registry."""
    print("🧪 Testing tool metadata...")

    from backend.tool_registry import registry

    tools = registry.list_tools()
    assert len(tools) >= 4, f"Should have at least 4 tools, got {len(tools)}"

    # Check required metadata
    for tool in tools:
        assert hasattr(tool, 'executor_default'), f"Tool {tool.name} missing executor_default"
        assert hasattr(tool, 'complexity'), f"Tool {tool.name} missing complexity"
        assert hasattr(tool, 'arg_hint'), f"Tool {tool.name} missing arg_hint"
        assert hasattr(tool, 'caller_snippet'), f"Tool {tool.name} missing caller_snippet"

    print("✅ Tool metadata works")
    return True

def test_complete_planner_execution():
    """Test complete planner execution."""
    print("🧪 Testing complete planner execution...")

    try:
        from graph import planner_app
        from graph.state import create_initial_state

        initial_state = create_initial_state('做一个公司竞品两周调研并拟方案')
        config = {'configurable': {'thread_id': 'test-planner'}}
        final_state = planner_app.invoke(initial_state, config=config)

        # Check results
        assert final_state.get('plan') is not None, "Should have plan"
        assert final_state.get('final_answer') is not None, "Should have final answer"
        assert len(final_state['plan']) > 0, "Plan should have todos"
        assert final_state.get('tool_call_count', 0) >= 0, "Should track tool calls"

        # Check execution path
        exec_path = final_state.get('execution_path', [])
        required_steps = ['user_input', 'classify_intent', 'planner_generate', 'todo_dispatch', 'aggregate_answer']
        for step in required_steps:
            assert step in exec_path, f"Missing execution step: {step}"

        print("✅ Complete planner execution works")
        return True

    except Exception as e:
        print(f"❌ Complete planner execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Running Sprint 0.6 Complete Test Suite")
    print("=" * 50)

    tests = [
        test_tool_metadata,
        test_executor_routing,
        test_planner_json_generation,
        test_tool_call_protocol,
        test_complete_planner_execution,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Sprint 0.6 is ready! 🎊")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
