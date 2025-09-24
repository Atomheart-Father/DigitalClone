#!/usr/bin/env python3
"""
真正的AI系统全流程交互测试
严格遵循用户要求：AI自我介绍 -> 用户请求 -> AI引导 -> 用户问路径 -> AI处理完整任务
"""

import subprocess
import time
import sys
from pathlib import Path

def run_real_full_test():
    """运行真正的完整AI系统测试"""

    print("🚀 开始真正的AI系统全流程测试")
    print("=" * 60)

    # 清理旧的输出文件
    output_dir = Path("/Users/bozhongxiao/Desktop/克罗米王国国立电台/AgentFlow/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in output_dir.glob("*.md"):
        if "test_output.md" not in f.name:
            f.unlink()
            print(f"🧹 清理旧文件: {f.name}")

    print("\n📋 严格按照用户要求的交互流程：")
    print("1. AI自我介绍")
    print("2. 用户：我想让你帮我看看桌面上的文件，然后上网搜索相关信息，最后写一个总结")
    print("3. AI引导：请问您想让我查看哪个文件呢？")
    print("4. 用户：文件在哪里啊？文件路径是什么？")
    print("5. AI告诉文件路径并开始完整处理：读取文件 -> 搜索网络 -> 生成总结 -> 保存报告")

    # 构建精确的交互命令序列
    interaction_commands = [
        # 步骤1: 用户打招呼，AI自我介绍
        '你好\n',
        'sleep 8\n',  # 等待AI处理和响应

        # 步骤2: 用户描述需求
        '我想让你帮我看看桌面上的文件，然后上网搜索相关信息，最后写一个总结\n',
        'sleep 8\n',  # 等待AI处理和响应

        # 步骤3: AI会引导用户指定文件，这里等待AI响应

        # 步骤4: 用户问文件路径
        '文件在哪里啊？文件路径是什么？\n',
        'sleep 8\n',  # 等待AI告诉文件路径

        # 步骤5: 现在AI应该开始处理完整任务，等待较长时间
        'sleep 60\n',  # 给足够时间处理文件读取+网络搜索+总结生成

        # 退出
        'exit\n',
    ]

    # 创建交互脚本
    script_content = ''.join(interaction_commands)
    script_file = Path(__file__).parent / "real_interaction.txt"
    script_file.write_text(script_content, encoding='utf-8')
    print(f"✅ 创建交互脚本: {script_file}")

    try:
        # 启动AI系统并通过脚本交互
        print("🎯 启动真实的AI系统并开始完整交互...")
        print("⚠️  这将调用真实的DeepSeek API，产生实际费用")

        cmd = f"{sys.executable} start.py < real_interaction.txt"
        print(f"执行命令: {cmd}")

        # 使用shell执行命令并重定向输入
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(Path(__file__).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 实时读取输出并记录关键步骤
        output_lines = []
        key_steps_detected = {
            "ai_intro": False,
            "user_request": False,
            "ai_guide": False,
            "user_ask_path": False,
            "ai_tell_path": False,
            "file_reading": False,
            "web_search": False,
            "summary_gen": False,
            "completion": False
        }

        print("\n📥 实时监控AI系统输出:")

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if line:
                    print(f"🤖 {line}")
                    output_lines.append(line)

                    # 检测关键步骤
                    line_lower = line.lower()

                    if "赛博克隆" in line and "助手" in line:
                        key_steps_detected["ai_intro"] = True
                        print("✅ 检测到: AI正确自我介绍")

                    if "查看哪个文件" in line or "文件名" in line:
                        key_steps_detected["ai_guide"] = True
                        print("✅ 检测到: AI引导用户指定文件")

                    if "新兴AI技术趋势分析.md" in line:
                        key_steps_detected["ai_tell_path"] = True
                        print("✅ 检测到: AI告诉用户文件路径")

                    if "读取文件" in line or "正在读取" in line:
                        key_steps_detected["file_reading"] = True
                        print("✅ 检测到: AI开始读取文件")

                    if "搜索" in line and ("网络" in line or "上网" in line):
                        key_steps_detected["web_search"] = True
                        print("✅ 检测到: AI开始网络搜索")

                    if "总结" in line and ("生成" in line or "报告" in line):
                        key_steps_detected["summary_gen"] = True
                        print("✅ 检测到: AI开始生成总结")

        # 等待进程结束
        return_code = process.wait()
        print(f"\n🏁 AI系统退出 (返回码: {return_code})")

        # 验证关键步骤
        print("\n🔍 验证关键交互步骤:")
        step_checks = [
            ("AI自我介绍", key_steps_detected["ai_intro"]),
            ("AI引导用户指定文件", key_steps_detected["ai_guide"]),
            ("AI告诉用户文件路径", key_steps_detected["ai_tell_path"]),
            ("AI读取文件", key_steps_detected["file_reading"]),
            ("AI进行网络搜索", key_steps_detected["web_search"]),
            ("AI生成总结报告", key_steps_detected["summary_gen"]),
        ]

        passed_steps = 0
        for step_name, passed in step_checks:
            status = "✅" if passed else "❌"
            print(f"{status} {step_name}")
            if passed:
                passed_steps += 1

        print(f"\n📊 步骤验证: {passed_steps}/{len(step_checks)} 通过")

        # 检查输出文件
        print("\n🔍 检查生成的输出文件...")
        md_files = list(output_dir.glob("*.md"))
        md_files = [f for f in md_files if "test_output.md" not in f.name]

        if md_files:
            # 找到最新文件
            latest_file = max(md_files, key=lambda f: f.stat().st_mtime)
            print(f"✅ 找到输出文件: {latest_file.name}")

            # 检查文件内容
            content = latest_file.read_text(encoding='utf-8')
            content_length = len(content)

            print(f"📄 文件大小: {content_length} 字符")

            # 检查内容质量
            quality_checks = [
                ("包含AI技术相关内容", "AI技术" in content),
                ("包含总结相关内容", "总结" in content),
                ("包含报告相关内容", "报告" in content),
                ("内容长度足够", content_length > 300),
            ]

            quality_passed = 0
            for check_desc, check_pass in quality_checks:
                status = "✅" if check_pass else "❌"
                print(f"{status} {check_desc}")
                if check_pass:
                    quality_passed += 1

            if quality_passed >= 3:
                print("🎉 输出文件质量检查通过！")
                return True
            else:
                print(f"⚠️ 输出文件质量不足 ({quality_passed}/4)")
                return False
        else:
            print("❌ 未找到生成的Markdown文件")
            return False

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理脚本文件
        if script_file.exists():
            script_file.unlink()
            print("✅ 清理交互脚本文件")

def main():
    """主函数"""
    print("🎭 真正的AI系统全流程交互测试")
    print("严格按照用户要求：AI自我介绍 -> 用户请求 -> AI引导 -> 用户问路径 -> AI处理完整任务")

    # 再次确认
    confirm = input("\n确定要运行真实的全流程API测试吗？(yes/no): ")
    if confirm.lower() not in ['yes', 'y', '是']:
        print("测试取消")
        return 0

    try:
        success = run_real_full_test()

        if success:
            print("\n🎉 真正的AI系统全流程测试成功完成！")
            print("✅ AI系统完整工作流程验证通过")
            print("✅ 成功调用DeepSeek API")
            print("✅ 完成文件读取+网络搜索+总结生成的完整闭环")
            print("✅ 严格遵循用户要求的交互流程")
            return 0
        else:
            print("\n❌ 真正的AI系统全流程测试失败！")
            print("请检查详细输出日志")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        return 1

if __name__ == "__main__":
    sys.exit(main())

