#!/usr/bin/env python3
"""
Crossover Operator

当轨迹池中有效条数大于等于2时，结合两条轨迹的特性生成新的策略。
当有效条数不足时，记录错误并跳过处理。
"""

import textwrap

from perf_config import StepConfig

from operators.base import BaseOperator, InstanceTrajectories, OperatorResult

class CrossoverOperator(BaseOperator):
    """交叉算子：综合两条轨迹的优点，生成新的初始代码"""

    def get_name(self) -> str:
        return "crossover"

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: InstanceTrajectories,
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例的交叉操作。

        从 instance_entry 中选择两条轨迹，生成交叉策略的 additional_requirements。
        """

        # 自适应选择两个源轨迹（不重复）
        chosen = self._select_source_labels(instance_entry, step_config, required_n=2)
        pick1 = chosen[0] if len(chosen) >= 1 else None
        pick2 = chosen[1] if len(chosen) >= 2 else None

        if pick1 and pick2 and pick1 == pick2:
            # 保障不重复，若重复则尝试再选一个不同的
            extra = [l for l in self._weighted_select_labels(instance_entry, k=3) if l != pick1]
            if extra:
                pick2 = extra[0]

        traj1 = instance_entry.trajectories.get(pick1) if pick1 else None
        traj2 = instance_entry.trajectories.get(pick2) if pick2 else None
        used = [s for s in [pick1, pick2] if isinstance(s, str) and s]

        self.logger.info(f" crossover 算子: pick1={pick1}, pick2={pick2}, has_traj1={traj1 is not None}, has_traj2={traj2 is not None}")

        if traj1 is None or traj2 is None:
            self.logger.warning(f" crossover 算子: traj1 或 traj2 为 None，跳过")
            return OperatorResult(source_labels=used)

        summary1 = self._format_entry(InstanceTrajectories(trajectories={pick1 or "iter1": traj1}))
        summary2 = self._format_entry(InstanceTrajectories(trajectories={pick2 or "iter2": traj2}))

        self.logger.info(f" crossover 算子: has_problem_description={bool(problem_description)}, has_summary1={bool(summary1)}, has_summary2={bool(summary2)}")

        if not problem_description or not summary1 or not summary2:
            self.logger.warning(f" crossover 算子: 缺少必要信息，无法构建 additional_requirements")
            return OperatorResult(source_labels=used)

        content = self._build_additional_requirements(summary1, summary2, instance_entry)
        self.logger.info(f" crossover 算子: 构建的 additional_requirements 长度={len(content) if content else 0}")

        if not content:
            self.logger.warning(f" crossover 算子: _build_additional_requirements 返回空字符串")
            return OperatorResult(source_labels=used)

        return OperatorResult(
            additional_requirements=content,
            source_labels=used,
        )

    def _build_additional_requirements(
        self, trajectory1: str, trajectory2: str, instance_entry: InstanceTrajectories | None = None
    ) -> str:
        t1 = textwrap.indent(trajectory1.strip(), "  ")
        t2 = textwrap.indent(trajectory2.strip(), "  ")
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("crossover", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("crossover_header")
            or "### STRATEGY MODE: CROSSOVER STRATEGY\nYou are tasked with synthesizing a SUPERIOR hybrid solution by intelligently combining the best elements of two prior optimization trajectories described below."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("crossover_guidelines")
            or (
                """
### SYNTHESIS GUIDELINES
1. **Complementary Combination**: Actively combine specific strengths.
- Example: If T1 has a better Core Algorithm but slow I/O, and T2 has fast I/O but a naive algorithm, implement T1's algorithm using T2's I/O technique.
- Example: If T1 used a correct Stack logic but slow List, and T2 used a fast Array but had logic bugs, implement T1's logic using T2's structure.
2. **Avoid Shared Weaknesses**: If both trajectories failed at a specific sub-task, you must introduce a novel fix for that specific part.
3. **Seamless Integration**: Do not just concatenate code. The resulting logic must be a single, cohesive implementation.
            """
            )
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())

        # 注入历史错误答案黑名单
        if instance_entry is not None:
            failed_summary = self._build_failed_answers_summary(instance_entry)
            if failed_summary:
                parts.append("\n" + failed_summary)

        parts.append("\n### TRAJECTORY 1 SUMMARY\n" + t1)
        parts.append("\n### TRAJECTORY 2 SUMMARY\n" + t2)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("crossover", CrossoverOperator)
