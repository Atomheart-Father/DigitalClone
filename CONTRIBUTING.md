# 贡献指南

欢迎为赛博克隆AI助手系统贡献代码！请遵循以下指南确保代码质量和一致性。

## 开发原则

遵循以下原则，具体见 [ADR-0001](docs/ADR-0001.md)：
- **CLI优先**: 以命令行界面为首要交付方式
- **模块清晰**: 保持模块内高内聚、跨模块低耦合
- **复用现有**: 优先使用现有接口，避免创造新轮子
- **主动测试**: 每个功能都有相应的测试和验证
- **遵循规范**: 遵守项目架构和编码规范

## 开发流程

### 1. 准备开发环境

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API keys
```

### 2. 创建功能分支

```bash
# 从 main 分支创建功能分支
git checkout -b feat/your-feature-name
# 或修复分支
git checkout -b fix/issue-description
# 或维护分支
git checkout -b chore/maintenance-task
```

### 3. 开发和测试

```bash
# 运行测试确保现有功能正常
pytest

# 启动应用进行手动测试
python -m backend.cli_app

# 开发完成后再次运行测试
pytest
```

### 4. 提交代码

```bash
# 添加更改的文件
git add .

# 提交时使用清晰的提交信息
git commit -m "feat: add new tool for weather querying

- Implement weather API integration
- Add JSON schema validation
- Include error handling for API failures
- Add unit tests for success/error cases"

# 推送分支
git push origin feat/your-feature-name
```

### 5. 创建 Pull Request

使用 [PULL_REQUEST_TEMPLATE.md](PULL_REQUEST_TEMPLATE.md) 创建PR，确保：

- [ ] 完成自检清单中的所有项目
- [ ] 通过手动验收测试
- [ ] 更新相关文档
- [ ] 添加必要的测试

## 编码规范

### Python 代码风格

- 使用 **类型注解** 和 **docstring**
- 遵循 PEP 8 命名规范
- 使用描述性的变量和函数名
- 函数复杂度控制在合理范围内

```python
from typing import Dict, Any

def process_user_input(input_text: str) -> Dict[str, Any]:
    """
    Process user input and return structured response.

    Args:
        input_text: Raw user input string

    Returns:
        Dictionary containing processing results

    Raises:
        ValueError: If input is invalid
    """
    if not input_text:
        raise ValueError("Input text cannot be empty")

    # Processing logic here
    return {"processed": True, "result": input_text.upper()}
```

### 错误处理

- 所有外部调用都要有异常处理
- 提供人类可读的错误信息
- 记录详细的错误日志用于调试

```python
import logging

logger = logging.getLogger(__name__)

def call_external_api(params: dict) -> dict:
    try:
        response = requests.post(API_URL, json=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error(f"API call timed out for params: {params}")
        return {"ok": False, "error": "请求超时，请稍后重试"}
    except requests.HTTPError as e:
        logger.error(f"API error {e.response.status_code}: {e.response.text}")
        return {"ok": False, "error": f"API错误: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Unexpected error in API call: {e}")
        return {"ok": False, "error": "发生未知错误，请联系管理员"}
```

### 日志记录

- 使用适当的日志级别
- 不要记录敏感信息
- 为每个模块创建独立的logger

```python
import logging

logger = logging.getLogger(__name__)

def process_request(request_data: dict):
    logger.info("Processing request", extra={"request_id": request_data.get("id")})
    logger.debug("Request details", extra={"data": request_data})

    # Processing logic

    logger.info("Request processed successfully")
```

## 测试要求

### 单元测试

- 每个工具至少有一个成功和一个失败的测试用例
- 测试边界条件和异常情况
- 使用描述性的测试名称

```python
import pytest
from backend.tools.tool_calculator import run as calculator_run

def test_calculator_simple_addition():
    """Test basic addition operation."""
    result = calculator_run(expression="2 + 3")
    assert result["ok"] is True
    assert result["value"] == 5

def test_calculator_invalid_expression():
    """Test handling of invalid expressions."""
    result = calculator_run(expression="invalid expression")
    assert result["ok"] is False
    assert "error" in result
```

### 集成测试

- 测试完整的对话流程
- 测试工具调用链
- 测试错误恢复机制

### 手动测试清单

提交PR前，请手动验证：

1. **模型路由**: 简单问答使用chat模型，复杂规划使用reasoner模型
2. **工具调用**: 计算器和日期时间工具工作正常
3. **AskUser机制**: 模型能正确追问并处理用户补充信息
4. **错误处理**: 网络超时、API错误等情况有适当处理
5. **日志记录**: 对话记录正确保存到指定目录

## 文档要求

### 代码文档

- 所有公共函数要有docstring
- 复杂逻辑要有注释说明
- 更新README中的使用示例

### API文档

- 新工具要更新工具列表
- 配置变更要更新环境变量说明
- 新功能要添加使用示例

## 安全注意事项

- 不要在日志中记录API密钥或其他敏感信息
- 工具实现要避免安全风险（如代码注入）
- 外部API调用要有超时和重试限制
- 输入验证要防止恶意输入

## 提交信息规范

提交信息格式：
```
type(scope): description

[optional body]

[optional footer]
```

类型：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码风格调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 维护任务

示例：
```
feat(tools): add weather query tool

- Implement OpenWeatherMap API integration
- Add location validation and error handling
- Include unit tests for different weather conditions

Closes #123
```

## 代码审查

PR创建后会进行代码审查，请：

- 及时响应审查意见
- 解释设计决策的理由
- 根据反馈进行必要的修改
- 确保所有CI检查通过

## 问题报告

发现bug或有功能建议：

1. 检查是否已有相关issue
2. 如果没有，创建新的issue
3. 提供详细的复现步骤
4. 包含环境信息和错误日志

感谢你的贡献！🎉
