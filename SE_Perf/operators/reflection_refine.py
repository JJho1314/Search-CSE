#!/usr/bin/env python3
"""
Reflection and Refine Operator

根据给定的源轨迹（source trajectory）进行反思与改进，生成更优的实现策略要求，
用于在下一次 PerfAgent 迭代中指导代码优化。
"""

import textwrap

from perf_config import StepConfig

from operators.base import BaseOperator, InstanceTrajectories, OperatorResult


class ReflectionRefineOperator(BaseOperator):
    """
    反思与改进算子：
    输入：step_config.inputs 中给定的单个源轨迹标签，如 {"label": "sol1"}
    输出：带有反思与具体改进指令的 additional_requirements 文本。
    """

    def get_name(self) -> str:
        return "reflection_refine"

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: InstanceTrajectories,
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例的反思与改进。"""

        src_summary = None
        used_labels: list[str] = []

        self.logger.info(f" reflection_refine 算子: 开始处理 instance={instance_name}, has_problem_description={bool(problem_description)}")

        # 若未提供输入标签，进行线性加权采样选择源轨迹
        chosen = self._select_source_labels(instance_entry, step_config, required_n=1)
        self.logger.info(f" reflection_refine 算子: chosen labels={chosen}")

        if chosen:
            traj = instance_entry.trajectories.get(chosen[0])
            if traj is not None:
                src_summary = self._format_entry(InstanceTrajectories(trajectories={chosen[0]: traj}))
                used_labels = [chosen[0]]
                self.logger.info(f" reflection_refine: 从 chosen 选择了轨迹，生成摘要")
            else:
                self.logger.warning(f" reflection_refine: chosen[0] 在 instance_entry 中不存在")
        else:
            self.logger.info(f" reflection_refine: 没有 chosen，进行加权采样")
            keys = self._weighted_select_labels(instance_entry, k=1)
            self.logger.info(f" reflection_refine: 加权采样得到 keys={keys}")
            if keys:
                traj = instance_entry.trajectories.get(keys[0])
                if traj is not None:
                    src_summary = self._format_entry(InstanceTrajectories(trajectories={keys[0]: traj}))
                    used_labels = [keys[0]]
                    self.logger.info(f" reflection_refine: 从采样选择轨迹，生成摘要")
                else:
                    self.logger.warning(f" reflection_refine: keys[0] 在 instance_entry 中不存在")
            else:
                self.logger.warning(f" reflection_refine: 加权采样没有得到任何键")

        # 最后回退：使用最新条目摘要
        if not src_summary:
            self.logger.info(f" reflection_refine: src_summary 仍为空，使用整个 instance_entry 作为回退")
            src_summary = self._format_entry(instance_entry)

        self.logger.info(f" reflection_refine: has_problem_description={bool(problem_description)}, has_src_summary={bool(src_summary)}")

        if not problem_description or not src_summary:
            self.logger.warning(f" reflection_refine: 缺少必要信息，无法构建 additional_requirements")
            return OperatorResult(source_labels=used_labels)

        content = self._build_additional_requirements(src_summary, instance_entry)
        self.logger.info(f" reflection_refine: 构建的 additional_requirements 长度={len(content) if content else 0}")

        if not content:
            self.logger.warning(f" reflection_refine: _build_additional_requirements 返回空字符串")
            return OperatorResult(source_labels=used_labels)

        return OperatorResult(
            additional_requirements=content,
            source_labels=used_labels,
        )

    def _build_additional_requirements(
        self, source_summary: str, instance_entry: InstanceTrajectories | None = None
    ) -> str:
        """
        构造带有反思与改进要求的 additional_requirements 文本。

        在源轨迹摘要之前注入历史错误答案黑名单，确保模型不会重复已知的错误。
        """
        src = textwrap.indent((source_summary or "").strip(), "  ")
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("reflection_refine", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("reflection_header")
            or "### STRATEGY MODE: REFLECTION AND REFINE STRATEGY\nYou must explicitly reflect on the previous trajectory and implement concrete improvements."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("reflection_guidelines")
            or (
                """
### REFINEMENT GUIDELINES
1. **Diagnose**: Identify the main shortcomings (correctness risks, bottlenecks, redundant work, I/O overhead).
2. **Fixes**: Propose targeted code-level changes (algorithmic upgrade, data structure replacement, caching/precomputation, I/O batching).
3. **Maintain Correctness**: Prioritize correctness; add guards/tests if necessary before optimizing runtime.
4. **Performance Goal**: Aim for measurable runtime improvement. Prefer asymptotic gains over micro-optimizations.
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

        parts.append("\n### SOURCE TRAJECTORY SUMMARY\n" + src)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("reflection_refine", ReflectionRefineOperator)
