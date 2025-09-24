#!/usr/bin/env python3
"""
启动脚本 for 赛博克隆AI助手

解决模块导入冲突问题。
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# 添加backend目录到Python路径
backend_dir = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_dir)

# 现在可以安全导入我们的模块
try:
    from cli_app import main
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

if __name__ == "__main__":
    main()
