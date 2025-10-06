"""
Planning Schema - 计划/步骤/候选/评分/搜索树结构

用于Best-of-N、Tree-of-Thoughts(ToT)、MCTS等搜索算法的数据结构。
支持计划生成、步骤分解、候选评估、评分聚合等功能。
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
from datetime import datetime


class PlanStatus(str, Enum):
    """计划状态"""
    DRAFT = "draft"           # 草稿
    ACTIVE = "active"         # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 取消


class StepStatus(str, Enum):
    """步骤状态"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 完成
    FAILED = "failed"         # 失败
    SKIPPED = "skipped"       # 跳过


class SearchStrategy(str, Enum):
    """搜索策略"""
    BEST_OF_N = "best_of_n"   # Best-of-N并行
    TREE_OF_THOUGHTS = "tot"  # Tree-of-Thoughts
    MCTS = "mcts"            # Monte Carlo Tree Search
    BEAM_SEARCH = "beam"      # 束搜索


class PlanStep(BaseModel):
    """计划步骤"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="步骤ID")
    name: str = Field(..., description="步骤名称")
    description: str = Field("", description="步骤描述")
    action: str = Field(..., description="执行动作")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="步骤参数")
    status: StepStatus = Field(StepStatus.PENDING, description="步骤状态")
    result: Optional[Dict[str, Any]] = Field(None, description="步骤结果")
    error: Optional[str] = Field(None, description="错误信息")
    dependencies: List[str] = Field(default_factory=list, description="依赖的步骤ID")
    estimated_time: float = Field(0.0, description="预估执行时间(秒)")
    actual_time: float = Field(0.0, description="实际执行时间(秒)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    def is_ready(self, completed_steps: List[str]) -> bool:
        """检查步骤是否可以执行"""
        return all(dep in completed_steps for dep in self.dependencies)
    
    def mark_completed(self, result: Dict[str, Any], execution_time: float):
        """标记步骤完成"""
        self.status = StepStatus.COMPLETED
        self.result = result
        self.actual_time = execution_time
    
    def mark_failed(self, error: str, execution_time: float):
        """标记步骤失败"""
        self.status = StepStatus.FAILED
        self.error = error
        self.actual_time = execution_time


class Candidate(BaseModel):
    """候选方案"""
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="候选ID")
    content: str = Field(..., description="候选内容")
    score: float = Field(0.0, description="评分")
    confidence: float = Field(0.0, description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    def update_score(self, score: float, confidence: float):
        """更新评分"""
        self.score = score
        self.confidence = confidence


class SearchNode(BaseModel):
    """搜索树节点"""
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="节点ID")
    parent_id: Optional[str] = Field(None, description="父节点ID")
    content: str = Field(..., description="节点内容")
    children: List[str] = Field(default_factory=list, description="子节点ID列表")
    visits: int = Field(0, description="访问次数")
    value: float = Field(0.0, description="节点价值")
    confidence: float = Field(0.0, description="置信度")
    depth: int = Field(0, description="节点深度")
    is_terminal: bool = Field(False, description="是否为终端节点")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def add_child(self, child_id: str):
        """添加子节点"""
        if child_id not in self.children:
            self.children.append(child_id)
    
    def update_value(self, value: float):
        """更新节点价值"""
        self.visits += 1
        # 使用指数移动平均更新价值
        alpha = 0.1
        self.value = (1 - alpha) * self.value + alpha * value
    
    def ucb_score(self, c: float = 1.414) -> float:
        """计算UCB分数(用于MCTS)"""
        import math
        if self.visits == 0:
            return float('inf')
        return self.value + c * (2 * math.log(self.visits) / self.visits) ** 0.5


class Plan(BaseModel):
    """执行计划"""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="计划ID")
    name: str = Field(..., description="计划名称")
    description: str = Field("", description="计划描述")
    goal: str = Field(..., description="计划目标")
    steps: List[PlanStep] = Field(default_factory=list, description="计划步骤")
    status: PlanStatus = Field(PlanStatus.DRAFT, description="计划状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def add_step(self, step: PlanStep):
        """添加步骤"""
        self.steps.append(step)
        self.updated_at = datetime.now()
    
    def get_ready_steps(self) -> List[PlanStep]:
        """获取可执行的步骤"""
        completed_ids = [s.step_id for s in self.steps if s.status == StepStatus.COMPLETED]
        return [s for s in self.steps if s.is_ready(completed_ids) and s.status == StepStatus.PENDING]
    
    def is_completed(self) -> bool:
        """检查计划是否完成"""
        return all(step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] for step in self.steps)
    
    def get_progress(self) -> float:
        """获取进度百分比"""
        if not self.steps:
            return 0.0
        completed = sum(1 for step in self.steps if step.status == StepStatus.COMPLETED)
        return completed / len(self.steps) * 100


class BestOfNResult(BaseModel):
    """Best-of-N搜索结果"""
    candidates: List[Candidate] = Field(..., description="候选列表")
    best_candidate: Optional[Candidate] = Field(None, description="最佳候选")
    consensus_score: float = Field(0.0, description="共识评分")
    diversity_score: float = Field(0.0, description="多样性评分")
    execution_time: float = Field(0.0, description="执行时间")
    
    def select_best(self) -> Optional[Candidate]:
        """选择最佳候选"""
        if not self.candidates:
            return None
        self.best_candidate = max(self.candidates, key=lambda c: c.score)
        return self.best_candidate


class TreeOfThoughtsResult(BaseModel):
    """Tree-of-Thoughts搜索结果"""
    root_node: Optional[SearchNode] = Field(None, description="根节点")
    nodes: List[SearchNode] = Field(default_factory=list, description="所有节点")
    best_path: List[str] = Field(default_factory=list, description="最佳路径")
    best_leaf: Optional[SearchNode] = Field(None, description="最佳叶节点")
    tree_depth: int = Field(0, description="树深度")
    total_nodes: int = Field(0, description="总节点数")
    execution_time: float = Field(0.0, description="执行时间")
    
    def get_node_by_id(self, node_id: str) -> Optional[SearchNode]:
        """根据ID获取节点"""
        return next((node for node in self.nodes if node.node_id == node_id), None)
    
    def get_children(self, node_id: str) -> List[SearchNode]:
        """获取子节点"""
        node = self.get_node_by_id(node_id)
        if not node:
            return []
        return [self.get_node_by_id(child_id) for child_id in node.children if self.get_node_by_id(child_id)]


class MCTSResult(BaseModel):
    """MCTS搜索结果"""
    root_node: Optional[SearchNode] = Field(None, description="根节点")
    nodes: List[SearchNode] = Field(default_factory=list, description="所有节点")
    best_action: Optional[str] = Field(None, description="最佳动作")
    iterations: int = Field(0, description="迭代次数")
    tree_size: int = Field(0, description="树大小")
    execution_time: float = Field(0.0, description="执行时间")
    
    def get_best_path(self) -> List[str]:
        """获取最佳路径"""
        if not self.root_node:
            return []
        
        path = []
        current = self.root_node
        
        while current and current.children:
            # 选择访问次数最多的子节点
            best_child_id = max(current.children, 
                              key=lambda cid: next((n.visits for n in self.nodes if n.node_id == cid), 0))
            path.append(best_child_id)
            current = next((n for n in self.nodes if n.node_id == best_child_id), None)
        
        return path


class SearchConfig(BaseModel):
    """搜索配置"""
    strategy: SearchStrategy = Field(SearchStrategy.BEST_OF_N, description="搜索策略")
    max_candidates: int = Field(5, description="最大候选数")
    max_depth: int = Field(4, description="最大深度")
    max_width: int = Field(3, description="最大宽度")
    iterations: int = Field(100, description="迭代次数")
    temperature: float = Field(0.7, description="生成温度")
    timeout: float = Field(60.0, description="超时时间(秒)")
    parallel: bool = Field(True, description="是否并行执行")
    
    @validator('max_candidates', 'max_depth', 'max_width', 'iterations')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("值必须大于0")
        return v


class SearchResult(BaseModel):
    """搜索结果"""
    success: bool = Field(..., description="是否成功")
    strategy: SearchStrategy = Field(..., description="使用的策略")
    result: Union[BestOfNResult, TreeOfThoughtsResult, MCTSResult] = Field(..., description="具体结果")
    error: Optional[str] = Field(None, description="错误信息")
    config: SearchConfig = Field(..., description="搜索配置")
    
    def get_best_answer(self) -> Optional[str]:
        """获取最佳答案"""
        if not self.success:
            return None
            
        if isinstance(self.result, BestOfNResult):
            return self.result.best_candidate.content if self.result.best_candidate else None
        elif isinstance(self.result, TreeOfThoughtsResult):
            return self.result.best_leaf.content if self.result.best_leaf else None
        elif isinstance(self.result, MCTSResult):
            # MCTS结果需要根据最佳路径构建答案
            return self.result.best_action
    
    def get_confidence(self) -> float:
        """获取整体置信度"""
        if not self.success:
            return 0.0
            
        if isinstance(self.result, BestOfNResult):
            return self.result.best_candidate.confidence if self.result.best_candidate else 0.0
        elif isinstance(self.result, TreeOfThoughtsResult):
            return self.result.best_leaf.confidence if self.result.best_leaf else 0.0
        elif isinstance(self.result, MCTSResult):
            return self.result.root_node.confidence if self.result.root_node else 0.0


# 工厂函数
def create_plan(name: str, goal: str, description: str = "") -> Plan:
    """创建新计划"""
    return Plan(
        name=name,
        goal=goal,
        description=description
    )


def create_step(name: str, action: str, parameters: Dict[str, Any] = None) -> PlanStep:
    """创建计划步骤"""
    return PlanStep(
        name=name,
        action=action,
        parameters=parameters or {}
    )


def create_candidate(content: str, score: float = 0.0, confidence: float = 0.0) -> Candidate:
    """创建候选方案"""
    return Candidate(
        content=content,
        score=score,
        confidence=confidence
    )


def create_search_node(content: str, parent_id: Optional[str] = None) -> SearchNode:
    """创建搜索节点"""
    return SearchNode(
        content=content,
        parent_id=parent_id
    )
