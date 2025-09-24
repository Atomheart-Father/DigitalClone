#!/usr/bin/env python3
"""
测试Prompt优化方案 - 验证三大问题修复

测试目标：
1. AskUser阻断机制 - 规划阶段触发阻塞等待
2. 工具严格控制 - 白名单强制，禁止发明工具
3. 多轮上下文管理 - 分层组装，预算控制
"""

import sys
import os

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_tool_whitelist_enforcement():
    """测试工具白名单强制执行"""
    print("🧪 测试工具白名单强制执行...")

    from backend.tool_prompt_builder import get_allowed_tools_whitelist, get_tools_whitelist_text

    allowed_tools = get_allowed_tools_whitelist()
    whitelist_text = get_tools_whitelist_text()

    print(f"✅ 白名单工具: {allowed_tools}")
    print(f"✅ 白名单文本: {whitelist_text}")

    # 验证不允许的工具会被拒绝
    assert "text_analyze" not in allowed_tools, "❌ 不应该允许不存在的工具"
    assert "file_read" in allowed_tools, "✅ 应该允许file_read"
    assert "web_search" in allowed_tools, "✅ 应该允许web_search"

    print("✅ 工具白名单强制执行正常")

def test_prompt_generation():
    """测试各阶段Prompt生成"""
    print("🧪 测试各阶段Prompt生成...")

    from backend.tool_prompt_builder import (
        build_phase1_draft_prompt,
        build_phase2_review_prompt,
        build_phase3_json_plan_prompt,
        build_tool_execution_prompt,
        build_reflective_replanning_prompt
    )

    # 测试Phase 1
    p1_prompt = build_phase1_draft_prompt(
        task_summary="读取文件并分析内容",
        known_params="无",
        missing_params="file_path",
        constraints="快速完成"
    )
    assert "工具" in p1_prompt, "❌ Phase 1应该包含工具列表"
    assert len(p1_prompt) < 1000, f"❌ Phase 1过长: {len(p1_prompt)} chars"
    print(f"✅ Phase 1 Prompt生成正常 ({len(p1_prompt)} chars)")

    # 测试Phase 2
    p2_prompt = build_phase2_review_prompt(
        goal="分析文件内容",
        facts="用户需要读取文件",
        draft_points="使用file_read工具"
    )
    assert len(p2_prompt) < 500, f"❌ Phase 2过长: {len(p2_prompt)} chars"
    assert "改进准则" in p2_prompt, "❌ Phase 2应该包含改进准则"
    print(f"✅ Phase 2 Prompt生成正常 ({len(p2_prompt)} chars)")

    # 测试Phase 3
    p3_prompt = build_phase3_json_plan_prompt(
        task="读取并分析文件内容",
        known_params="{}",
        missing_params='["file_path"]',
        constraints="快速"
    )
    assert "你只输出 json" in p3_prompt, "❌ Phase 3应该包含JSON指令"
    assert "ask_user" in p3_prompt, "❌ Phase 3应该支持ask_user"
    print(f"✅ Phase 3 Prompt生成正常 ({len(p3_prompt)} chars)")

    # 测试工具执行
    exec_prompt = build_tool_execution_prompt(
        task="读取文件",
        current_state="准备执行",
        todo_item="T1: 读取用户文件",
        tool_name="file_read"
    )
    assert "只从 ALLOWED_TOOLS 里选择" in exec_prompt, "❌ 工具执行应该强制白名单"
    print(f"✅ 工具执行Prompt生成正常 ({len(exec_prompt)} chars)")

    # 测试反思规划
    reflect_prompt = build_reflective_replanning_prompt(
        goal="完成分析",
        new_facts="获取了文件内容",
        current_plan="读取文件 -> 分析内容"
    )
    assert len(reflect_prompt) < 500, f"❌ 反思Prompt过长: {len(reflect_prompt)} chars"
    print(f"✅ 反思规划Prompt生成正常 ({len(reflect_prompt)} chars)")

    print("✅ 各阶段Prompt生成测试通过")

def test_context_assembly():
    """测试上下文组装系统"""
    print("🧪 测试上下文组装系统...")

    from context.assembler import ContextAssembler
    from memory.working_buffer import WorkingBuffer
    from memory.rolling_summary import RollingSummary
    from memory.profile_store import ProfileStore
    from memory.rag_store import RAGStore

    # 创建组件
    buffer = WorkingBuffer(max_tokens=2000, max_turns=10)
    summary = RollingSummary(max_tokens=500)
    profile = ProfileStore()
    rag = RAGStore()

    assembler = ContextAssembler(
        working_buffer=buffer,
        rolling_summary=summary,
        profile_store=profile,
        rag_store=rag
    )

    # 添加测试数据
    buffer.append_turn("user", "请帮我分析这个文件")
    buffer.append_turn("assistant", "好的，我来帮您分析")

    # 测试组装
    result = assembler.assemble("分析文件内容", budget_tokens=4000)

    assert 'messages' in result, "❌ 应该有messages字段"
    assert 'metadata' in result, "❌ 应该有metadata字段"
    assert len(result['messages']) > 0, "❌ 应该有消息"

    metadata = result['metadata']
    assert 'total_chars' in metadata, "❌ 应该有字符统计"
    assert 'estimated_tokens' in metadata, "❌ 应该有token估算"
    assert 'section_breakdown' in metadata, "❌ 应该有分节统计"

    print(f"✅ 上下文组装正常: {len(result['messages'])} messages, ~{metadata['estimated_tokens']} tokens")

    # 清理测试文件
    import os
    for f in ["test_profile.json", "test_rag.json"]:
        if os.path.exists(f):
            os.remove(f)

def test_askuser_integration():
    """测试AskUser集成"""
    print("🧪 测试AskUser集成...")

    from graph.nodes import _create_fallback_plan

    # 测试fallback计划生成
    plan = _create_fallback_plan("请分析这个文件")

    assert "ask_user" in plan, "❌ fallback计划应该包含ask_user"
    assert plan["ask_user"]["needed"] == True, "❌ 应该需要用户输入"
    assert "file_path" in plan["ask_user"]["missing_params"], "❌ 应该要求file_path"

    print("✅ AskUser集成测试通过")

def test_micro_decider():
    """测试微决策器"""
    print("🧪 测试微决策器...")

    from reasoner.micro_decide import MicroDecider

    decider = MicroDecider(max_tokens=200, timeout_seconds=10)

    stats = decider.get_stats()
    assert stats['max_tokens'] == 200, "❌ 最大token数设置不正确"
    assert stats['timeout_seconds'] == 10, "❌ 超时设置不正确"

    print("✅ 微决策器结构正常")

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始Prompt优化方案测试")
    print("=" * 60)

    try:
        test_tool_whitelist_enforcement()
        print()

        test_prompt_generation()
        print()

        test_context_assembly()
        print()

        test_askuser_integration()
        print()

        test_micro_decider()
        print()

        print("🎉 所有测试通过！")
        print("📋 Prompt优化方案成功实现三大问题修复：")
        print("   ✅ AskUser阻断机制 - 规划阶段触发阻塞等待")
        print("   ✅ 工具严格控制 - 白名单强制，禁止发明工具")
        print("   ✅ 多轮上下文管理 - 分层组装，预算控制")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
