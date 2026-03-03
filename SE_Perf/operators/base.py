#!/usr/bin/env python3
"""
SE Operators Base Classes

定义了所有算子的基类和核心接口。
算子是模块化的、可重用的组件，用于执行特定的轨迹操作，如生成、交叉或过滤。
"""

from __future__ import annotations

import abc
import random
import re
from dataclasses import dataclass, field
from typing import Any

from core.utils.llm_client import LLMClient
from core.utils.se_logger import get_se_logger
from core.utils.traj_pool_manager import TrajPoolManager
from perf_config import StepConfig


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryItem:
    """轨迹池中单条轨迹的结构化表示。

    Attributes:
        label: 轨迹标签（如 "sol1", "iter3"）。
        metric: 标量性能指标（越低越好）。
        solution: 解/代码文本。
        summary: 轨迹摘要（可为 dict 或 str）。
        extras: 其他未列举的字段（保留原始 JSON 中的所有额外字段）。
    """

    label: str = ""
    metric: float | None = None
    solution: str = ""
    summary: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TrajectoryItem":
        known = {"label", "metric", "solution", "summary"}
        metric_raw = d.get("metric")
        try:
            metric = float(metric_raw) if metric_raw is not None else None
        except (ValueError, TypeError):
            metric = None
        return TrajectoryItem(
            label=str(d.get("label") or ""),
            metric=metric,
            solution=str(d.get("solution") or ""),
            summary=d.get("summary"),
            extras={k: v for k, v in d.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": self.label,
            "solution": self.solution,
        }
        if self.metric is not None:
            out["metric"] = self.metric
        if self.summary is not None:
            out["summary"] = self.summary
        out.update(self.extras)
        return out


@dataclass
class InstanceTrajectories:
    """实例在轨迹池中的所有轨迹数据（结构化）。

    提供类型安全的属性访问，所有算子通过 `.trajectories` 直接操作。

    Attributes:
        problem: 问题描述文本（来自轨迹池 "problem" 键）。
        trajectories: 按标签索引的轨迹字典。
    """

    problem: str = ""
    trajectories: dict[str, TrajectoryItem] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.problem) or bool(self.trajectories)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "InstanceTrajectories":
        """从轨迹池的原始 JSON dict 构建。"""
        if not isinstance(d, dict):
            return InstanceTrajectories()
        problem = str(d.get("problem") or "")
        trajectories: dict[str, TrajectoryItem] = {}
        for k, v in d.items():
            if k == "problem" or not isinstance(v, dict):
                continue
            trajectories[k] = TrajectoryItem.from_dict(v)
        return InstanceTrajectories(problem=problem, trajectories=trajectories)

    def to_dict(self) -> dict[str, Any]:
        """转回原始 dict 格式。"""
        out: dict[str, Any] = {}
        if self.problem:
            out["problem"] = self.problem
        for k, traj in self.trajectories.items():
            out[k] = traj.to_dict()
        return out


@dataclass
class OperatorContext:
    """算子执行的共享上下文。

    封装算子所需的模型配置、提示词配置和选择模式，
    替代原先通过 dict 传递的 operator_config。

    Attributes:
        model_config: LLM 模型配置（保留 dict，因为需透传给 LLMClient）。
        prompt_config: 提示词配置。
        selection_mode: 默认轨迹选择模式（"weighted" 或 "random"）。
        metric_higher_is_better: metric 方向。True=越大越好（如 EM），False=越小越好（如运行时间）。
    """

    model_config: dict[str, Any] = field(default_factory=dict)
    prompt_config: dict[str, Any] = field(default_factory=dict)
    selection_mode: str = "weighted"
    metric_higher_is_better: bool = False


@dataclass
class OperatorResult:
    """单实例算子执行结果

    这是 Operator 返回给 perf_run.py 的标准化结果对象。
    包含用于构建 AgentRequest 的全部信息。

    Attributes:
        additional_requirements: 额外的 prompt 要求（来自算子分析）
        initial_solution: 可选的初始解覆盖
        source_labels: 使用的源轨迹标签列表
    """

    additional_requirements: str | None = None
    initial_solution: str | None = None
    source_labels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------


class BaseOperator(abc.ABC):
    """
    SE算子基类，定义通用功能和新的 `run` 接口。
    所有算子都应继承自此类。
    """

    def __init__(self, context: OperatorContext):
        """
        初始化算子。

        Args:
            context: OperatorContext 实例。
        """
        self.context = context
        self.llm_client: LLMClient | None = None
        self.logger = get_se_logger(f"operator.{self.get_name()}", emoji="🔧")

    def _setup_model(self) -> None:
        """设置LLM客户端实例。"""
        if self.llm_client is not None:
            return
        model_config_data = self.context.model_config
        self.llm_client = LLMClient(model_config_data)
        self.logger.info(f"LLM客户端已初始化: {model_config_data.get('name')}")

    def _call_llm_api(self, prompt: str, system_prompt: str = "") -> str:
        """
        调用LLM API。

        Args:
            prompt: 用户提示。
            system_prompt: 系统提示。

        Returns:
            LLM生成的响应文本。
        """
        self._setup_model()
        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": prompt})

        try:
            model_cfg = self.context.model_config
            temp = model_cfg.get("temperature", 0.3)
            max_out = model_cfg.get("max_output_tokens")
            self.logger.debug(f"LLM系统提示词:\n{system_prompt}")
            self.logger.debug(f"LLM用户提示词:\n{prompt}")
            message = self.llm_client.call_llm(
                history,
                temperature=temp,
                max_tokens=max_out,
                usage_context=f"operator.{self.get_name()}",
            )
            self.logger.debug(f"LLM原始响应:\n{message}")
            if message:
                message = self.llm_client.clean_think_tags(message)
            self.logger.debug(f"LLM清理后响应:\n{message}")
            return message or ""
        except Exception as e:
            self.logger.error(f"LLM API调用失败: {e}")
            return ""
            
    def _format_entry(self, entry: InstanceTrajectories, compact: bool = False) -> str:
        return TrajPoolManager.format_entry(entry, compact=compact)

    def _count_failed_answers(
        self,
        instance_entry: InstanceTrajectories,
    ) -> int:
        """统计 instance_entry 中去重后的失败答案数量。

        用于判断是否需要启用 compact 模式来节省 token。
        """
        seen: set[str] = set()
        for _key, traj in instance_entry.trajectories.items():
            metric = traj.metric
            extras = traj.extras or {}
            artifacts = extras.get("artifacts") or {}
            if isinstance(traj.summary, dict):
                artifacts = artifacts or traj.summary.get("artifacts") or {}
            is_failed = False
            if metric is not None and metric == 0.0:
                is_failed = True
            if artifacts.get("em_match") is False or artifacts.get("subem_match") is False:
                is_failed = True
            if not is_failed:
                continue
            answer = artifacts.get("extracted_answer") or ""
            if not answer:
                sol = traj.solution or ""
                m = re.search(r"<answer>\s*(.*?)\s*</answer>", sol, re.DOTALL)
                if m:
                    answer = m.group(1).strip()
            if answer:
                seen.add(answer)
        return len(seen)

    def _build_failed_answers_summary(
        self,
        instance_entry: InstanceTrajectories,
        exclude_labels: list[str] | None = None,
    ) -> str:
        """从所有历史轨迹中提取失败答案，构建错误答案黑名单摘要。

        遍历 instance_entry 中的全部轨迹，收集 metric==0 或 artifacts 中标记失败的条目，
        去重后生成一段文本供下一次迭代参考，避免模型重复犯同样的错误。

        Args:
            instance_entry: 当前实例在轨迹池中的所有轨迹。
            exclude_labels: 可选，需排除的标签列表。

        Returns:
            格式化后的失败答案摘要文本；若无失败记录则返回空字符串。
        """
        exclude = set(exclude_labels or [])
        # 收集: (answer, failure_reason, iteration, label)
        failed_records: list[tuple[str, str, int | None, str]] = []
        seen_answers: dict[str, list[str]] = {}  # answer -> [failure_reasons]

        for key, traj in instance_entry.trajectories.items():
            if key in exclude:
                continue
            # 判断是否失败
            metric = traj.metric
            extras = traj.extras or {}
            artifacts = extras.get("artifacts") or {}
            if isinstance(traj.summary, dict):
                artifacts = artifacts or traj.summary.get("artifacts") or {}

            is_failed = False
            if metric is not None and metric == 0.0:
                is_failed = True
            if artifacts.get("em_match") is False or artifacts.get("subem_match") is False:
                is_failed = True

            if not is_failed:
                continue

            # 提取答案和失败原因
            answer = artifacts.get("extracted_answer") or ""
            if not answer:
                # 尝试从 solution 中提取 <answer>...</answer>
                sol = traj.solution or ""
                m = re.search(r"<answer>\s*(.*?)\s*</answer>", sol, re.DOTALL)
                if m:
                    answer = m.group(1).strip()
            if not answer:
                continue

            failure_reason = artifacts.get("failure_reason") or "Unknown"
            iteration = extras.get("iteration")
            try:
                iteration = int(iteration) if iteration is not None else None
            except (ValueError, TypeError):
                iteration = None

            failed_records.append((answer, failure_reason, iteration, key))
            if answer not in seen_answers:
                seen_answers[answer] = []
            if failure_reason not in seen_answers[answer]:
                seen_answers[answer].append(failure_reason)

        if not seen_answers:
            return ""

        # 构建精简摘要：只列出错误答案，不含冗余的 iteration/reason 细节
        unique_answers = list(seen_answers.keys())
        answer_list = ", ".join(f'"{a}"' for a in unique_answers)

        lines: list[str] = [
            "### WRONG ANSWERS (DO NOT REPEAT ANY)",
            answer_list,
            f"All {len(unique_answers)} answers above are INCORRECT. You MUST give a DIFFERENT answer.",
            "Try a fundamentally different approach to find the correct answer.",
        ]

        return "\n".join(lines)

    def _weighted_select_labels(
        self, entry: InstanceTrajectories, k: int = 1, allowed_labels: list[str] | None = None
    ) -> list[str]:
        """基于 performance 的线性加权采样选择子标签。

        权重方向由 ``self.context.metric_higher_is_better`` 控制：
        - False（默认）：metric 越低权重越高（如运行时间：1/perf）
        - True：metric 越高权重越高（如 EM/准确率：直接使用 perf）

        若提供 allowed_labels，则仅在该集合中进行采样（忽略不存在的标签）。
        """
        higher_is_better = getattr(self.context, "metric_higher_is_better", False)
        items: list[tuple[str, float]] = []
        for subkey, traj in entry.trajectories.items():
            if allowed_labels is not None:
                if subkey not in allowed_labels and (not traj.label or traj.label not in allowed_labels):
                    continue
            perf_val = traj.metric if traj.metric is not None else (0.0 if higher_is_better else 1.0)
            items.append((subkey, perf_val))
        if not items:
            return []
        eps = 1e-9
        selected: list[str] = []
        remaining = items.copy()
        for _ in range(min(k, len(remaining))):
            if higher_is_better:
                # metric 越大越好：直接用 metric 值作为权重
                weights = [max(0.001, perf + eps) for _, perf in remaining]
            else:
                # metric 越小越好：用 1/metric 作为权重
                weights = [max(0.001, 1.0 / max(eps, perf)) for _, perf in remaining]
            total = sum(weights)
            if total <= 0:
                choice = random.choice(remaining)[0]
            else:
                weights = [w / total for w in weights]
                r = random.random()
                s = 0.0
                choice = remaining[-1][0]
                for (label_key, perf), w in zip(remaining, weights):
                    s += w
                    if r <= s:
                        choice = label_key
                        break
            selected.append(choice)
            remaining = [it for it in remaining if it[0] != choice]
        return selected

    def _random_select_labels(
        self, entry: InstanceTrajectories, k: int = 1, allowed_labels: list[str] | None = None
    ) -> list[str]:
        candidates: list[str] = []
        for subkey, traj in entry.trajectories.items():
            if allowed_labels is not None:
                if subkey not in allowed_labels and (not traj.label or traj.label not in allowed_labels):
                    continue
            candidates.append(subkey)
        if not candidates:
            return []
        k = min(k, len(candidates))
        return random.sample(candidates, k)

    def _get_selection_mode(self, step_config: StepConfig) -> str:
        try:
            v = step_config.selection_mode
            if isinstance(v, str) and v.strip():
                m = v.strip().lower()
                if m in ("weighted", "random"):
                    return m
            g = self.context.selection_mode
            if isinstance(g, str) and g.strip():
                m = g.strip().lower()
                if m in ("weighted", "random"):
                    return m
        except Exception:
            pass
        return "weighted"

    def _resolve_label_subkey(self, entry: InstanceTrajectories, label: str) -> str | None:
        """将外部提供的标签解析为 entry 的子键。
        优先匹配子键名，其次匹配子项内部的 `label` 字段。
        """
        lab = str(label)
        if lab in entry.trajectories:
            return lab
        for subkey, traj in entry.trajectories.items():
            if traj.label == lab:
                return subkey
        return None

    def _select_source_labels(self, entry: InstanceTrajectories, step_config: StepConfig, required_n: int) -> list[str]:
        """统一选择源轨迹标签。
        规则：
        - 若 `inputs` 标签数目 == required_n：直接使用 `inputs`
        - 若 `inputs` 标签数目 >  required_n：在 `inputs` 范围内加权采样 required_n 个
        - 若 `inputs` 标签数目 <  required_n：先使用已有 `inputs`，剩余从整个 entry 中加权采样补齐
        返回 entry 子键名列表，唯一且最多 required_n 个。
        """
        inputs = step_config.inputs or []
        provided_labels = [str(i.get("label")) for i in inputs if isinstance(i, dict) and i.get("label")]
        # 解析为 entry 子键
        resolved = []
        seen: set[str] = set()
        for lab in provided_labels:
            subkey = self._resolve_label_subkey(entry, lab)
            if subkey and subkey not in seen:
                resolved.append(subkey)
                seen.add(subkey)

        need = max(0, int(required_n))
        count = len(resolved)
        if count == need:
            return resolved
        if count > need:
            mode = self._get_selection_mode(step_config)
            if mode == "random":
                sampled = self._random_select_labels(entry, k=need, allowed_labels=resolved)
            else:
                sampled = self._weighted_select_labels(entry, k=need, allowed_labels=resolved)
            return list(dict.fromkeys(sampled))

        # count < need：先用已有，再补齐
        out = list(resolved)
        used = set(out)
        remaining = [k for k in entry.trajectories if k not in used]
        if remaining:
            mode = self._get_selection_mode(step_config)
            if mode == "random":
                sampled_more = self._random_select_labels(entry, k=need - count, allowed_labels=remaining)
            else:
                sampled_more = self._weighted_select_labels(entry, k=need - count, allowed_labels=remaining)
            for s in sampled_more:
                if s not in used:
                    out.append(s)
                    used.add(s)
                if len(out) >= need:
                    break
        return out[:need]

    @abc.abstractmethod
    def get_name(self) -> str:
        """获取算子名称。"""
        pass

    @abc.abstractmethod
    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: InstanceTrajectories,
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例，返回结构化结果。

        这是单实例模式下的标准调用接口。子类必须实现此方法。

        Args:
            step_config: 当前步骤的配置（StepConfig 对象）。
            instance_name: 实例名称。
            instance_entry: 该实例在轨迹池中的结构化数据。
            problem_description: 问题描述文本（由调用方显式传入）。

        Returns:
            OperatorResult 对象，包含 additional_requirements、initial_solution 等。
        """
        ...


class TemplateOperator(BaseOperator):
    """
    模板算子基类，用于为下一次 PerfAgent 运行生成初始代码。
    """


class EnhanceOperator(BaseOperator):
    """
    增强算子基类，用于为下一次 PerfAgent 运行生成增强历史配置。
    """
