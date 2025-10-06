"""
Tool I/O Schema - 工具统一输入输出契约

定义所有工具的统一输入输出格式，支持JSON Schema强校验与重试机制。
遵循 {ok: bool, out: {...}, conf: float, cost: {...}, trace: [...]} 格式。
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
import jsonschema
import json


class CostInfo(BaseModel):
    """工具执行成本信息"""
    tokens_input: int = Field(0, description="输入token数量")
    tokens_output: int = Field(0, description="输出token数量") 
    compute_time: float = Field(0.0, description="计算时间(秒)")
    api_calls: int = Field(0, description="API调用次数")
    memory_usage: int = Field(0, description="内存使用量(字节)")
    
    def total_tokens(self) -> int:
        """总token数量"""
        return self.tokens_input + self.tokens_output
    
    def total_cost(self, input_price: float = 0.001, output_price: float = 0.002) -> float:
        """估算总成本(美元)"""
        return self.tokens_input * input_price + self.tokens_output * output_price


class TraceStep(BaseModel):
    """单步执行轨迹"""
    step_id: str = Field(..., description="步骤ID")
    operation: str = Field(..., description="操作名称")
    input: Dict[str, Any] = Field(default_factory=dict, description="输入参数")
    output: Optional[Dict[str, Any]] = Field(None, description="输出结果")
    duration: float = Field(0.0, description="执行时长(秒)")
    error: Optional[str] = Field(None, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class ToolIO(BaseModel):
    """工具统一输入输出格式"""
    ok: bool = Field(..., description="执行是否成功")
    out: Optional[Dict[str, Any]] = Field(None, description="输出结果")
    conf: float = Field(0.0, ge=0.0, le=1.0, description="置信度(0-1)")
    cost: CostInfo = Field(default_factory=CostInfo, description="执行成本")
    trace: List[TraceStep] = Field(default_factory=list, description="执行轨迹")
    
    @validator('out')
    def validate_output(cls, v, values):
        """验证输出格式"""
        if values.get('ok') and v is None:
            raise ValueError("成功执行时必须有输出结果")
        return v
    
    def add_trace_step(self, step_id: str, operation: str, 
                      input_data: Dict[str, Any], 
                      output_data: Optional[Dict[str, Any]] = None,
                      duration: float = 0.0,
                      error: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """添加轨迹步骤"""
        step = TraceStep(
            step_id=step_id,
            operation=operation,
            input=input_data,
            output=output_data,
            duration=duration,
            error=error,
            metadata=metadata or {}
        )
        self.trace.append(step)
    
    def update_cost(self, **kwargs):
        """更新成本信息"""
        for key, value in kwargs.items():
            if hasattr(self.cost, key):
                setattr(self.cost, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.dict()


class ToolSchema(BaseModel):
    """工具Schema定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: Dict[str, Any] = Field(..., description="参数JSON Schema")
    returns: Dict[str, Any] = Field(..., description="返回JSON Schema")
    strict: bool = Field(True, description="是否严格模式")
    retry_count: int = Field(3, description="重试次数")
    timeout: float = Field(30.0, description="超时时间(秒)")
    
    def validate_input(self, params: Dict[str, Any]) -> bool:
        """验证输入参数"""
        try:
            jsonschema.validate(params, self.parameters)
            return True
        except jsonschema.ValidationError:
            return False
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """验证输出结果"""
        try:
            jsonschema.validate(output, self.returns)
            return True
        except jsonschema.ValidationError:
            return False
    
    def get_openai_tool_definition(self) -> Dict[str, Any]:
        """获取OpenAI工具定义格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolExecutionError(Exception):
    """工具执行异常"""
    def __init__(self, message: str, tool_name: str, 
                 input_params: Optional[Dict[str, Any]] = None,
                 trace: Optional[List[TraceStep]] = None):
        self.message = message
        self.tool_name = tool_name
        self.input_params = input_params or {}
        self.trace = trace or []
        super().__init__(f"Tool {tool_name} execution failed: {message}")


def create_success_result(output: Dict[str, Any], 
                         confidence: float = 1.0,
                         cost: Optional[CostInfo] = None,
                         trace: Optional[List[TraceStep]] = None) -> ToolIO:
    """创建成功结果"""
    return ToolIO(
        ok=True,
        out=output,
        conf=confidence,
        cost=cost or CostInfo(),
        trace=trace or []
    )


def create_error_result(error: str,
                       input_params: Optional[Dict[str, Any]] = None,
                       trace: Optional[List[TraceStep]] = None) -> ToolIO:
    """创建错误结果"""
    return ToolIO(
        ok=False,
        out=None,
        conf=0.0,
        cost=CostInfo(),
        trace=trace or []
    )


def validate_tool_io(io_result: ToolIO, schema: ToolSchema) -> bool:
    """验证工具I/O是否符合Schema"""
    if not io_result.ok:
        return True  # 错误结果不需要验证输出
    
    if not schema.validate_output(io_result.out):
        return False
    
    return True


# 常用工具Schema模板
class CommonSchemas:
    """常用工具Schema模板"""
    
    @staticmethod
    def math_expression() -> Dict[str, Any]:
        """数学表达式工具Schema"""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    
    @staticmethod
    def math_result() -> Dict[str, Any]:
        """数学计算结果Schema"""
        return {
            "type": "object", 
            "properties": {
                "result": {
                    "type": "number",
                    "description": "计算结果"
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "计算步骤"
                }
            },
            "required": ["result"],
            "additionalProperties": False
        }
    
    @staticmethod
    def search_query() -> Dict[str, Any]:
        """搜索查询Schema"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "结果数量限制"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    
    @staticmethod
    def search_results() -> Dict[str, Any]:
        """搜索结果Schema"""
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "url": {"type": "string"},
                            "score": {"type": "number"}
                        },
                        "required": ["title", "content"]
                    },
                    "description": "搜索结果列表"
                },
                "total": {
                    "type": "integer",
                    "description": "总结果数"
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }


# 兼容层：从旧格式转换
def convert_old_tool_result(old_result: Dict[str, Any]) -> ToolIO:
    """从旧格式转换到新格式"""
    return ToolIO(
        ok=old_result.get("ok", False),
        out=old_result.get("value"),
        conf=1.0 if old_result.get("ok", False) else 0.0,
        cost=CostInfo(),
        trace=[]
    )


def convert_to_old_format(tool_io: ToolIO) -> Dict[str, Any]:
    """转换到旧格式以保持兼容性"""
    return {
        "ok": tool_io.ok,
        "value": tool_io.out,
        "error": None if tool_io.ok else "Tool execution failed"
    }
