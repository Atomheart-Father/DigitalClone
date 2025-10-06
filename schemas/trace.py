"""
Trace Schema - Meta-trace记录格式

用于记录全链路的执行轨迹，包括步骤、分数、代价、置信度、决策原因等。
支持A/B对比、性能分析、决策回放等功能。
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
from datetime import datetime
import json


class TraceLevel(str, Enum):
    """轨迹级别"""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class OperationType(str, Enum):
    """操作类型"""
    # 核心操作
    THINK = "think"           # 思考
    ACT = "act"              # 行动
    OBSERVE = "observe"      # 观察
    
    # 搜索操作
    SEARCH = "search"        # 搜索
    EXPAND = "expand"        # 扩展
    EVALUATE = "evaluate"    # 评估
    SELECT = "select"        # 选择
    
    # 验证操作
    VERIFY = "verify"        # 验证
    CHECK = "check"          # 检查
    
    # 工具操作
    TOOL_CALL = "tool_call"  # 工具调用
    TOOL_RESULT = "tool_result"  # 工具结果
    
    # 决策操作
    DECIDE = "decide"        # 决策
    ROUTE = "route"          # 路由
    
    # 系统操作
    INIT = "init"            # 初始化
    FINALIZE = "finalize"    # 完成


class DecisionReason(BaseModel):
    """决策原因"""
    reason_type: str = Field(..., description="原因类型")
    description: str = Field(..., description="原因描述")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="证据列表")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="置信度")
    
    def add_evidence(self, key: str, value: Any, weight: float = 1.0):
        """添加证据"""
        self.evidence.append({
            "key": key,
            "value": value,
            "weight": weight
        })


class CostMetrics(BaseModel):
    """成本指标"""
    tokens_input: int = Field(0, description="输入token数")
    tokens_output: int = Field(0, description="输出token数")
    compute_time: float = Field(0.0, description="计算时间(秒)")
    memory_usage: int = Field(0, description="内存使用(字节)")
    api_calls: int = Field(0, description="API调用次数")
    cost_usd: float = Field(0.0, description="成本(美元)")
    
    def add_cost(self, other: 'CostMetrics'):
        """累加成本"""
        self.tokens_input += other.tokens_input
        self.tokens_output += other.tokens_output
        self.compute_time += other.compute_time
        self.memory_usage = max(self.memory_usage, other.memory_usage)
        self.api_calls += other.api_calls
        self.cost_usd += other.cost_usd


class ScoreMetrics(BaseModel):
    """评分指标"""
    accuracy: float = Field(0.0, description="准确性")
    relevance: float = Field(0.0, description="相关性")
    completeness: float = Field(0.0, description="完整性")
    efficiency: float = Field(0.0, description="效率")
    quality: float = Field(0.0, description="质量")
    overall: float = Field(0.0, description="总体评分")
    
    @validator('accuracy', 'relevance', 'completeness', 'efficiency', 'quality', 'overall')
    def validate_score_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("评分必须在0-1之间")
        return v
    
    def calculate_overall(self, weights: Optional[Dict[str, float]] = None):
        """计算总体评分"""
        if weights is None:
            weights = {
                "accuracy": 0.3,
                "relevance": 0.2,
                "completeness": 0.2,
                "efficiency": 0.15,
                "quality": 0.15
            }
        
        self.overall = sum(getattr(self, key) * weight for key, weight in weights.items())
        return self.overall


class TraceEvent(BaseModel):
    """轨迹事件"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="事件ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    level: TraceLevel = Field(TraceLevel.INFO, description="级别")
    operation: OperationType = Field(..., description="操作类型")
    message: str = Field(..., description="消息")
    data: Dict[str, Any] = Field(default_factory=dict, description="数据")
    cost: CostMetrics = Field(default_factory=CostMetrics, description="成本")
    score: Optional[ScoreMetrics] = Field(None, description="评分")
    decision_reason: Optional[DecisionReason] = Field(None, description="决策原因")
    parent_event_id: Optional[str] = Field(None, description="父事件ID")
    children_event_ids: List[str] = Field(default_factory=list, description="子事件ID列表")
    
    def add_child(self, child_event_id: str):
        """添加子事件"""
        if child_event_id not in self.children_event_ids:
            self.children_event_ids.append(child_event_id)
    
    def set_decision_reason(self, reason_type: str, description: str, 
                           confidence: float = 0.0, evidence: List[Dict[str, Any]] = None):
        """设置决策原因"""
        self.decision_reason = DecisionReason(
            reason_type=reason_type,
            description=description,
            confidence=confidence,
            evidence=evidence or []
        )


class ExecutionTrace(BaseModel):
    """执行轨迹"""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="轨迹ID")
    session_id: str = Field(..., description="会话ID")
    user_query: str = Field(..., description="用户查询")
    strategy: str = Field("simple", description="执行策略")
    events: List[TraceEvent] = Field(default_factory=list, description="事件列表")
    total_cost: CostMetrics = Field(default_factory=CostMetrics, description="总成本")
    final_score: Optional[ScoreMetrics] = Field(None, description="最终评分")
    success: bool = Field(False, description="是否成功")
    error: Optional[str] = Field(None, description="错误信息")
    duration: float = Field(0.0, description="总时长(秒)")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    def add_event(self, event: TraceEvent):
        """添加事件"""
        self.events.append(event)
        self.total_cost.add_cost(event.cost)
    
    def calculate_final_score(self, weights: Optional[Dict[str, float]] = None):
        """计算最终评分"""
        if not self.events:
            return
        
        # 聚合所有事件的评分
        scores = [event.score for event in self.events if event.score is not None]
        if not scores:
            return
        
        # 计算加权平均
        if weights is None:
            weights = {
                "accuracy": 0.3,
                "relevance": 0.2,
                "completeness": 0.2,
                "efficiency": 0.15,
                "quality": 0.15
            }
        
        final_score = ScoreMetrics()
        for key in ["accuracy", "relevance", "completeness", "efficiency", "quality"]:
            values = [getattr(score, key) for score in scores]
            if values:
                setattr(final_score, key, sum(values) / len(values))
        
        final_score.calculate_overall(weights)
        self.final_score = final_score
    
    def get_events_by_operation(self, operation: OperationType) -> List[TraceEvent]:
        """根据操作类型获取事件"""
        return [event for event in self.events if event.operation == operation]
    
    def get_events_by_level(self, level: TraceLevel) -> List[TraceEvent]:
        """根据级别获取事件"""
        return [event for event in self.events if event.level == level]
    
    def get_decision_chain(self) -> List[TraceEvent]:
        """获取决策链"""
        return [event for event in self.events if event.decision_reason is not None]


class ABTestTrace(BaseModel):
    """A/B测试轨迹"""
    test_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="测试ID")
    test_name: str = Field(..., description="测试名称")
    variant_a: ExecutionTrace = Field(..., description="变体A轨迹")
    variant_b: ExecutionTrace = Field(..., description="变体B轨迹")
    comparison_metrics: Dict[str, Any] = Field(default_factory=dict, description="对比指标")
    winner: Optional[str] = Field(None, description="获胜变体")
    confidence_level: float = Field(0.0, description="置信水平")
    
    def compare_variants(self) -> Dict[str, Any]:
        """对比变体"""
        comparison = {
            "cost_comparison": {
                "variant_a": self.variant_a.total_cost.dict(),
                "variant_b": self.variant_b.total_cost.dict()
            },
            "score_comparison": {
                "variant_a": self.variant_a.final_score.dict() if self.variant_a.final_score else None,
                "variant_b": self.variant_b.final_score.dict() if self.variant_b.final_score else None
            },
            "duration_comparison": {
                "variant_a": self.variant_a.duration,
                "variant_b": self.variant_b.duration
            },
            "success_rate": {
                "variant_a": self.variant_a.success,
                "variant_b": self.variant_b.success
            }
        }
        
        self.comparison_metrics = comparison
        return comparison
    
    def determine_winner(self, metric: str = "overall") -> str:
        """确定获胜者"""
        if metric == "overall" and self.variant_a.final_score and self.variant_b.final_score:
            if self.variant_a.final_score.overall > self.variant_b.final_score.overall:
                self.winner = "variant_a"
            else:
                self.winner = "variant_b"
        elif metric == "cost":
            if self.variant_a.total_cost.cost_usd < self.variant_b.total_cost.cost_usd:
                self.winner = "variant_a"
            else:
                self.winner = "variant_b"
        elif metric == "duration":
            if self.variant_a.duration < self.variant_b.duration:
                self.winner = "variant_a"
            else:
                self.winner = "variant_b"
        
        return self.winner or "tie"


class TraceAnalyzer(BaseModel):
    """轨迹分析器"""
    traces: List[ExecutionTrace] = Field(default_factory=list, description="轨迹列表")
    
    def add_trace(self, trace: ExecutionTrace):
        """添加轨迹"""
        self.traces.append(trace)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.traces:
            return {}
        
        costs = [trace.total_cost for trace in self.traces]
        durations = [trace.duration for trace in self.traces]
        scores = [trace.final_score.overall for trace in self.traces if trace.final_score]
        success_rates = [trace.success for trace in self.traces]
        
        return {
            "total_traces": len(self.traces),
            "success_rate": sum(success_rates) / len(success_rates),
            "avg_cost_usd": sum(cost.cost_usd for cost in costs) / len(costs),
            "avg_duration": sum(durations) / len(durations),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "total_tokens": sum(cost.total_tokens() for cost in costs),
            "total_api_calls": sum(cost.api_calls for cost in costs)
        }
    
    def get_strategy_comparison(self) -> Dict[str, Any]:
        """获取策略对比"""
        strategies = {}
        for trace in self.traces:
            if trace.strategy not in strategies:
                strategies[trace.strategy] = []
            strategies[trace.strategy].append(trace)
        
        comparison = {}
        for strategy, strategy_traces in strategies.items():
            costs = [trace.total_cost for trace in strategy_traces]
            durations = [trace.duration for trace in strategy_traces]
            scores = [trace.final_score.overall for trace in strategy_traces if trace.final_score]
            success_rates = [trace.success for trace in strategy_traces]
            
            comparison[strategy] = {
                "count": len(strategy_traces),
                "success_rate": sum(success_rates) / len(success_rates),
                "avg_cost_usd": sum(cost.cost_usd for cost in costs) / len(costs),
                "avg_duration": sum(durations) / len(durations),
                "avg_score": sum(scores) / len(scores) if scores else 0.0
            }
        
        return comparison


# 工厂函数
def create_trace_event(operation: OperationType, message: str, 
                      level: TraceLevel = TraceLevel.INFO,
                      data: Dict[str, Any] = None,
                      cost: CostMetrics = None,
                      score: ScoreMetrics = None) -> TraceEvent:
    """创建轨迹事件"""
    return TraceEvent(
        operation=operation,
        message=message,
        level=level,
        data=data or {},
        cost=cost or CostMetrics(),
        score=score
    )


def create_execution_trace(session_id: str, user_query: str, 
                          strategy: str = "simple") -> ExecutionTrace:
    """创建执行轨迹"""
    return ExecutionTrace(
        session_id=session_id,
        user_query=user_query,
        strategy=strategy
    )


def create_ab_test(test_name: str, variant_a: ExecutionTrace, 
                  variant_b: ExecutionTrace) -> ABTestTrace:
    """创建A/B测试"""
    return ABTestTrace(
        test_name=test_name,
        variant_a=variant_a,
        variant_b=variant_b
    )
