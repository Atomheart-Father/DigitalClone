# 赛博克隆 AI 助理系统

一个支持工具调用和复杂任务规划的对话式AI助手。

## 特性

- 🔄 **双模型路由**: DeepSeek-chat处理日常问答，DeepSeek-reasoner处理复杂任务规划
- 🛠️ **工具系统**: 统一注册的工具系统，支持计算器、日期时间等工具
- ❓ **智能追问**: AskUser机制，支持模型主动追问用户以获取必要信息
- 📝 **对话日志**: 持久化对话记录，为后续RAG功能提供数据基础
- 💻 **CLI界面**: 命令行REPL，支持流式输出和特殊命令
- 🤔 **反思规划**: 高级功能，支持在获取大量信息后动态调整执行计划（可配置开关）

## 快速开始

### 环境要求
- Python 3.10+
- DeepSeek API Key

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd deepclone-assistant

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 DEEPSEEK_API_KEY
```

### 运行

```bash
# 启动CLI应用
python start.py

# 开启流式输出
python start.py --stream
```

### 示例交互

```
==================================================
🎭 赛博克隆 AI 助手
==================================================
✓ 已配置 API 密钥
📡 Chat模型: deepseek-chat
🤖 Reasoner模型: deepseek-reasoner
🛠️  已加载 2 个工具: ['calculator', 'datetime']

输入您的消息，或使用特殊命令:
  :help  显示帮助信息
  :tools 列出可用工具
  :q     退出程序
  :clear 清空对话历史
--------------------------------------------------
> 今天天气不错，讲个笑话
[Chat模型] 正在思考...
小明问老师："为什么天空是蓝色的？"老师说："因为我不知道红色去哪儿了！"

> 帮我算 (12+7)*3**2
[Chat模型] 正在调用工具...
调用: calculator(expression="(12+7)*3**2")
结果: 171

> 给我做个两周学习计划，包含每日任务和阶段目标
[Reasoner模型] 这是一个复杂的规划任务，需要reasoner模型...
正在规划学习方案...

阶段1 (第1-7天): 基础知识构建
- Day 1: 核心概念学习
- Day 2: 实践练习
...
```

## 架构设计

### 核心组件

- **`agent_core.py`**: 主要代理逻辑，包含ReAct循环和路由决策
- **`llm_interface.py`**: LLM客户端抽象，支持DeepSeek chat/reasoner和mock后备
- **`tool_registry.py`**: 工具注册系统，动态发现和执行工具
- **`message_types.py`**: 统一消息协议定义
- **`cli_app.py`**: 命令行界面和REPL循环
- **`logger.py`**: 对话日志记录系统

### 工具系统

工具放置在 `backend/tools/` 目录下，每个工具需要：

- `TOOL_META`: 工具元数据（名称、描述、参数Schema）
- `run(**kwargs)`: 工具执行函数
- 返回格式: `{"ok": bool, "value": any, "error": str}`

### 消息协议

支持以下角色：
- `user`: 用户输入
- `assistant`: 助手回复
- `tool`: 工具调用结果
- `system`: 系统消息
- `ask_user`: 模型追问用户

## 开发

### 项目结构

```
deepclone-assistant/
├── backend/
│   ├── agent_core.py          # 代理核心逻辑
│   │   ├── llm_interface.py   # LLM接口抽象
│   │   ├── tool_registry.py   # 工具注册系统
│   │   ├── message_types.py   # 消息类型定义
│   │   ├── logger.py          # 日志系统
│   │   ├── config.py          # 配置管理
│   │   └── cli_app.py        # CLI应用
│   ├── tools/                 # 工具目录
│   │   ├── tool_calculator.py # 计算器工具
│   │   └── tool_datetime.py   # 日期时间工具
│   └── tests/                 # 测试文件
├── data/
│   └── logs/                  # 日志存储
├── docs/
│   └── ADR-0001.md           # 架构决策记录
├── .env.example              # 环境变量模板
├── requirements.txt          # 依赖列表
└── README.md
```

### 添加新工具

1. 在 `backend/tools/` 下创建新文件 `tool_yourtool.py`
2. 定义工具元数据和执行函数：

```python
TOOL_META = {
    "name": "your_tool",
    "description": "Your tool description",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param1"]
    }
}

def run(param1: str) -> dict:
    try:
        # 工具逻辑
        result = process(param1)
        return {"ok": True, "value": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

3. 重启应用，工具会自动注册

### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest backend/tests/test_tools.py
pytest backend/tests/test_router.py
```

## 贡献

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

## 高级配置

### 反思规划功能

这是一个高级功能，用于处理重量级任务。当系统获取到大量信息后，会自动触发反思流程重新评估和调整执行计划。

**配置选项：**

```bash
# 开启反思规划（默认关闭）
ENABLE_REFLECTIVE_REPLANNING=true

# 触发反思的最小信息量（字符数，默认1000）
REFLECTIVE_REPLANNING_MIN_INFO_SIZE=2000

# 压缩上下文的最大token数（默认200）
REFLECTIVE_REPLANNING_MAX_TOKENS=150
```

**工作流程：**
1. 工具执行后，如果返回信息超过阈值 → 触发反思
2. Chat模型压缩新信息和当前计划到200token内
3. Reasoner模型判断是否需要修改计划
4. 如需要，Chat模型基于反思建议生成新的执行计划

**⚠️ 注意：** 此功能会显著增加API调用次数和响应时间，仅在处理复杂任务时开启。

## 许可证

[License信息]
