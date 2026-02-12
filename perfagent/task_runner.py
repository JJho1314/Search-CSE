"""
BaseTaskRunner 抽象接口

定义任务特定插件（TaskRunner）的标准接口。
每种任务类型（如 EffiBench 代码优化、LiveCodeBench 代码生成、AIME 数学推理）
需实现此接口，从而使 Agent 核心循环保持任务无关。

设计要点:
- Agent 通过 TaskRunner 的方法与任务数据交互，而不直接处理任务特定的数据结构。
- ``load_metadata`` 为类方法，可在不实例化 TaskRunner 的情况下提取最小元数据，
  供 SE_Perf 层在首次调用 Agent 之前使用。
- ``evaluate`` 返回 ``(metric, artifacts)``，其中 metric 统一为越低越好（lower is better），
  artifacts 包含评估补充信息（problem_description 由 TaskMetadata 提供）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .protocols import TaskMetadata


class BaseTaskRunner(ABC):
    """任务特定插件的抽象基类

    每个具体的 TaskRunner 负责:
    1. 加载并解析任务数据
    2. 提取或生成初始解
    3. 评估解的质量并返回标量指标 + artifacts
    4. 构建 system prompt 和 optimization prompt
    5. 从 LLM 响应中提取新的解
    """

    # ------------------------------------------------------------------
    # 元数据提取（类方法，无需实例化）
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def load_metadata(cls, path: Path) -> TaskMetadata:
        """从任务数据文件中提取最小元数据

        供 SE_Perf 在首次调用 Agent 之前使用（例如用于全局记忆检索、
        轨迹池匹配等），无需加载完整的任务数据。

        Args:
            path: 任务数据文件路径

        Returns:
            TaskMetadata，包含 instance_id 和 problem_description
        """
        ...

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    @abstractmethod
    def load_instance(self, path: Path) -> Any:
        """加载完整的任务数据

        返回值类型由具体 TaskRunner 定义（如 EffiBenchXInstance、dict 等），
        Agent 核心循环不对其做任何假设，仅作为不透明对象传递给其他 TaskRunner 方法。

        Args:
            path: 任务数据文件路径

        Returns:
            任务数据对象（类型由子类定义）
        """
        ...

    # ------------------------------------------------------------------
    # 初始解
    # ------------------------------------------------------------------

    @abstractmethod
    def get_initial_solution(self, instance_data: Any, config: Any) -> str:
        """提取或生成初始解

        根据任务数据和配置，返回初始解字符串。
        - 对于代码优化任务，这可能是从实例中提取的初始代码或占位符代码。
        - 对于数学推理任务，这可能是空字符串或初始推理框架。

        Args:
            instance_data: load_instance 返回的任务数据对象
            config: Agent 配置对象

        Returns:
            初始解字符串
        """
        ...

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        solution: str,
        instance_data: Any,
        config: Any,
    ) -> tuple[float, dict[str, Any]]:
        """评估解的质量

        返回一个标量指标和一个 artifacts 字典。

        约定:
        - metric 的比较方向由 PerfAgentConfig.metric_higher_is_better 控制：
          - False（默认）：越小越好，如运行时间、错误率
          - True：越大越好，如准确率、通过率
          TaskRunner 可以选择直接返回原始指标（如 pass rate 0-1），
          由 Agent 配置控制比较方向，或自行转换后统一返回
        - artifacts 是任务特定的评估补充信息，SE_Perf 层不对其内容做假设。
          例如：test_results、failure_details、evaluation_metadata 等。
          注意：artifacts 不应包含 problem_description，它已经在顶层提供。

        Args:
            solution: 待评估的解
            instance_data: load_instance 返回的任务数据对象
            config: Agent 配置对象

        Returns:
            (metric, artifacts) 元组
        """
        ...

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    @abstractmethod
    def build_system_prompt(self, instance_data: Any, **context: Any) -> str:
        """构建系统 prompt

        根据任务数据和上下文信息，生成发送给 LLM 的系统级指令。

        Args:
            instance_data: load_instance 返回的任务数据对象
            **context: 额外上下文（如 language, optimization_target,
                       additional_requirements, local_memory, global_memory 等）

        Returns:
            系统 prompt 字符串
        """
        ...

    @abstractmethod
    def build_optimization_prompt(
        self,
        solution: str,
        metric: float,
        artifacts: dict[str, Any],
        **context: Any,
    ) -> str:
        """构建优化/求解 prompt

        根据当前解、指标和 artifacts，生成发送给 LLM 的用户级优化指令。

        Args:
            solution: 当前解
            metric: 当前标量指标
            artifacts: 当前 artifacts 字典
            **context: 额外上下文（如 language、history 等）

        Returns:
            优化 prompt 字符串
        """
        ...

    # ------------------------------------------------------------------
    # LLM 停止序列（可选，由子类覆盖）
    # ------------------------------------------------------------------

    @property
    def llm_stop_sequences(self) -> list[str] | None:
        """返回 LLM 生成时的停止序列列表。

        子类可覆盖此属性，让 PerfAgent 在调用 LLM 时传入 stop 参数。
        例如 SearchR1Runner 返回 ["</search>"]，使 LLM 在生成搜索查询后
        立即停止，而不是继续生成幻觉的搜索结果。

        Returns:
            停止序列列表，或 None 表示不使用停止序列
        """
        return None

    # ------------------------------------------------------------------
    # 解提取
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_solution(self, llm_response: str, current_solution: str) -> str:
        """从 LLM 响应中提取新的解

        根据任务类型的不同，提取逻辑可能差异很大：
        - 代码优化: 提取 Markdown 代码块或应用 SEARCH/REPLACE diff
        - 数学推理: 提取最终数值答案
        - 代码生成: 提取完整函数实现

        Args:
            llm_response: LLM 的原始响应文本
            current_solution: 当前解（用于 diff-based 方式的基准）

        Returns:
            新的解字符串；若提取失败，可返回 current_solution（不变）
        """
        ...
