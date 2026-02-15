#!/usr/bin/env python3

"""
Trajectory Pool Summary Operator

分析轨迹池中的全部历史尝试，综合其优点与失败模式，
生成新的附加需求文本，用于下一次迭代评估。
"""

import textwrap
from typing import Any

from perf_config import StepConfig

from operators.base import InstanceTrajectories, TrajectoryItem, OperatorResult, TemplateOperator


class TrajPoolSummaryOperator(TemplateOperator):
    """轨迹池总结算子：综合历史轨迹，生成附加需求"""

    def get_name(self) -> str:
        return "traj_pool_summary"

    def get_strategy_prefix(self) -> str:
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("traj_pool_summary", {}) if isinstance(pcfg, dict) else {}
        return str(opcfg.get("prefix") or pcfg.get("traj_pool_summary_prefix") or "RISK-AWARE PROBLEM SOLVING GUIDANCE")

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: InstanceTrajectories,
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例的轨迹池总结。"""
        if not instance_entry.trajectories:
            return OperatorResult()

        problem_statement = problem_description
        if not problem_statement:
            return OperatorResult()

        content = self._build_additional_requirements(str(problem_statement), instance_entry.trajectories, instance_entry)
        if not content:
            return OperatorResult()

        return OperatorResult(additional_requirements=content)

    def _format_approaches_data(self, trajectories: dict[str, TrajectoryItem]) -> str:
        """格式化历史尝试数据为通用的嵌套文本结构。"""

        def _fmt(value: Any, indent: int) -> str:
            prefix = "  " * indent
            if isinstance(value, dict):
                lines: list[str] = []
                for k, v in value.items():
                    if v is None or v == "" or (isinstance(v, (list, dict)) and len(v) == 0):
                        continue
                    if isinstance(v, (int, float)):
                        lines.append(f"{prefix}{k}: {v}")
                    elif isinstance(v, bool):
                        lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
                    elif isinstance(v, str) and "\n" not in v:
                        lines.append(f"{prefix}{k}: {v}")
                    else:
                        lines.append(f"{prefix}{k}:")
                        child = _fmt(v, indent + 1)
                        if child:
                            lines.append(child)
                return "\n".join(lines)
            if isinstance(value, list):
                lines: list[str] = []
                for item in value:
                    if item is None or item == "":
                        continue
                    if isinstance(item, (int, float)):
                        lines.append(f"{prefix}- {item}")
                    elif isinstance(item, bool):
                        lines.append(f"{prefix}- {'true' if item else 'false'}")
                    elif isinstance(item, str) and "\n" not in item:
                        lines.append(f"{prefix}- {item}")
                    else:
                        child = _fmt(item, indent + 1)
                        if child:
                            child_lines = child.splitlines()
                            if child_lines:
                                child_prefix = "  " * (indent + 1)
                                first = child_lines[0]
                                if first.startswith(child_prefix):
                                    first = first[len(child_prefix) :]
                                lines.append(f"{prefix}- {first}")
                                for cl in child_lines[1:]:
                                    if cl.startswith(child_prefix):
                                        cl = cl[len(child_prefix) :]
                                    lines.append(f"{prefix}  {cl}")
                return "\n".join(lines)
            if isinstance(value, str):
                if "\n" in value:
                    return textwrap.indent(value, "  " * indent)
                return f"{prefix}{value}"
            if isinstance(value, bool):
                return f"{prefix}{'true' if value else 'false'}"
            if isinstance(value, (int, float)):
                return f"{prefix}{value}"
            return f"{prefix}{str(value)}"

        parts: list[str] = []
        for key, traj in trajectories.items():
            parts.append(f"ATTEMPT {key}:")
            body = _fmt(traj.to_dict(), 0)
            if body:
                parts.append(body)
        return "\n".join(parts)

    def _build_additional_requirements(
        self,
        problem_statement: str,
        trajectories: dict[str, TrajectoryItem],
        instance_entry: InstanceTrajectories | None = None,
    ) -> str:
        formatted_attempts = self._format_approaches_data(trajectories)
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("traj_pool_summary", {}) if isinstance(pcfg, dict) else {}
        header = str(
            opcfg.get("header") or pcfg.get("traj_pool_summary_header") or "RISK-AWARE PROBLEM SOLVING GUIDANCE"
        )
        prob = textwrap.indent(str(problem_statement).strip(), "  ")
        attempts = textwrap.indent(formatted_attempts.strip(), "  ")

        # 注入历史错误答案黑名单
        failed_section = ""
        if instance_entry is not None:
            failed_summary = self._build_failed_answers_summary(instance_entry)
            if failed_summary:
                failed_section = "\n" + failed_summary + "\n"

        body = (
            opcfg.get("guidance")
            or pcfg.get("traj_pool_summary_guidance")
            or (
                "Guidance:\n"
                "1. Identify and avoid previously failed techniques and blind spots.\n"
                "2. Prefer robust alternatives with clearer performance characteristics.\n"
                "3. Integrate proven components across attempts when helpful.\n"
                "4. Keep code simple, correct, and maintainable.\n"
            )
        )
        return f"{header}\n\nPROBLEM:\n{prob}\n{failed_section}\nHISTORY COMPONENTS:\n{attempts}\n\n{body}".strip()


# 注册算子
from .registry import register_operator

register_operator("traj_pool_summary", TrajPoolSummaryOperator)
