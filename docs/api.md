# 赛博克隆AI助手 API 接口文档

## 概述

本文档描述了赛博克隆AI助手的API接口规范，为后续Web界面或其他客户端集成提供参考。

## 核心接口

### 1. 对话接口

#### POST /api/chat

处理用户对话请求，支持流式和非流式响应。

**请求格式:**
```json
{
  "message": "用户输入的消息",
  "stream": false,
  "session_id": "可选的会话ID",
  "context": {
    "additional_context": "可选的额外上下文信息"
  }
}
```

**响应格式 (非流式):**
```json
{
  "response": "AI助手的回复内容",
  "route": "chat|reasoner",
  "tool_calls": 0,
  "session_id": "会话ID",
  "finished": true
}
```

**响应格式 (流式):**
```
data: {"chunk": "响应内容块", "finished": false}
data: {"chunk": "更多内容", "finished": false}
data: {"finished": true, "tool_calls": 1}
data: [DONE]
```

**错误响应:**
```json
{
  "error": "错误描述信息",
  "code": "ERROR_CODE",
  "session_id": "会话ID"
}
```

### 2. 工具结果接口

#### POST /api/tool_result

提交工具执行结果，继续对话流程。

**请求格式:**
```json
{
  "session_id": "会话ID",
  "tool_call_id": "工具调用ID",
  "tool_name": "工具名称",
  "result": {
    "success": true,
    "data": "工具执行结果",
    "error": null
  }
}
```

**响应格式:**
```json
{
  "response": "基于工具结果的AI回复",
  "finished": true,
  "session_id": "会话ID"
}
```

## 数据格式规范

### 消息格式

所有消息遵循统一的格式：

```json
{
  "role": "user|assistant|tool|system",
  "content": "消息内容",
  "metadata": {
    "timestamp": "ISO 8601时间戳",
    "tool_call_id": "工具调用ID（仅tool消息）",
    "tool_name": "工具名称（仅tool消息）"
  }
}
```

### 工具调用格式

```json
{
  "id": "工具调用唯一ID",
  "name": "工具名称",
  "arguments": {
    "参数名": "参数值"
  }
}
```

### 路由决策格式

```json
{
  "engine": "chat|reasoner",
  "reason": "路由决策理由",
  "confidence": 0.85
}
```

## 认证与安全

### API Key认证

所有请求需要在HTTP头部包含API Key：

```
Authorization: Bearer YOUR_API_KEY
X-API-Key: YOUR_API_KEY
```

### 请求限制

- 每个API Key每分钟最多100个请求
- 单个请求最大内容长度：10KB
- 会话保持时间：30分钟

## 错误码定义

| 错误码 | 描述 | HTTP状态码 |
|--------|------|------------|
| INVALID_REQUEST | 请求格式错误 | 400 |
| UNAUTHORIZED | 未授权访问 | 401 |
| FORBIDDEN | 权限不足 | 403 |
| NOT_FOUND | 资源不存在 | 404 |
| RATE_LIMITED | 请求频率超限 | 429 |
| INTERNAL_ERROR | 服务器内部错误 | 500 |
| MODEL_TIMEOUT | AI模型响应超时 | 504 |

## 示例用例

### 简单对话

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "message": "你好，请介绍一下你自己",
    "stream": false
  }'
```

### 复杂任务（带工具调用）

```bash
# 第一步：发送复杂请求
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "message": "帮我计算 (12+7)*3^2 并且告诉我当前时间",
    "stream": false
  }'

# 响应可能包含工具调用信息
# 客户端需要执行工具并提交结果
```

### 流式响应

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "message": "给我讲个故事",
    "stream": true
  }' \
  --no-buffer
```

## 实现注意事项

### 客户端实现建议

1. **连接管理**: 实现自动重连和指数退避重试
2. **流式处理**: 正确处理SSE (Server-Sent Events) 格式
3. **状态管理**: 维护会话状态和上下文信息
4. **错误处理**: 实现完善的错误处理和用户反馈

### 服务端实现要求

1. **并发处理**: 支持多个并发会话
2. **状态持久化**: 会话状态的可靠存储和恢复
3. **工具执行**: 安全的工具调用和结果验证
4. **日志记录**: 完整的请求响应日志记录

## 版本信息

- **API版本**: v1.0
- **最后更新**: 2025-09-24
- **兼容性**: 向后兼容，支持渐进式升级
