#!/usr/bin/env python3
"""
Trajectory Pool Manager (Label-based, Single Instance)

管理一个以"标签"为键的轨迹池（单实例）。每个轨迹都是一个独立的实体，
包含了执行摘要、性能数据、代码路径等元信息。

Pool 数据结构（扁平化，无外层 instance_name 嵌套）：
{
  "problem": "...",
  "iter0": { "label": "iter0", "solution": "...", ... },
  "iter1": { "label": "iter1", "solution": "...", ... }
}
"""

import copy
import json
import math
from pathlib import Path
from typing import Any, Optional

from core.utils.local_memory_manager import LocalMemoryManager
from core.utils.se_logger import get_se_logger


class TrajPoolManager:
    """
    轨迹池管理器（单实例，基于标签）。
    负责加载、保存、查询和修改存储在 traj.pool 文件中的轨迹数据。
    轨迹池是一个以字符串标签为键的扁平字典，顶层包含 "problem" 和各轨迹条目。
    """

    def __init__(
        self,
        pool_path: str,
        instance_name: str = "",
        llm_client=None,
        memory_manager: Optional["LocalMemoryManager"] = None,  # noqa: F821
        prompt_config: dict[str, Any] | None = None,
        metric_higher_is_better: bool = False,
    ):
        """
        初始化轨迹池管理器。

        Args:
            pool_path: traj.pool 文件路径。
            instance_name: 实例名称（唯一实例）。
            llm_client: LLM 客户端实例，用于轨迹总结。
            memory_manager: 本地记忆管理器。
            prompt_config: 提示词配置字典。
            metric_higher_is_better: metric 方向。True=越大越好（如 EM），False=越小越好（如运行时间）。
        """
        self.pool_path = Path(pool_path)
        self.instance_name = instance_name
        self.llm_client = llm_client
        self.logger = get_se_logger("traj_pool", emoji="🏊")
        self.memory_manager = memory_manager
        self.prompt_config = prompt_config or {}
        self.metric_higher_is_better = metric_higher_is_better
        self._best_label: str | None = None

    # -----------------------------------------------------------------------
    # 池的加载 / 保存 / 初始化
    # -----------------------------------------------------------------------

    def initialize_pool(self) -> None:
        """初始化轨迹池文件。如果文件不存在，则创建一个空的 JSON 对象。"""
        try:
            self.pool_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.pool_path.exists():
                with open(self.pool_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                self.logger.info(f"初始化空的轨迹池: {self.pool_path}")
            else:
                self.logger.info(f"轨迹池已存在: {self.pool_path}")
        except Exception as e:
            self.logger.error(f"初始化轨迹池失败: {e}")
            raise
        try:
            self.refresh_best_label()
        except Exception:
            pass

    def load_pool(self) -> dict[str, Any]:
        """从文件加载整个轨迹池。"""
        try:
            if not self.pool_path.exists():
                self.logger.warning("轨迹池文件不存在，返回空池")
                return {}
            with open(self.pool_path, encoding="utf-8") as f:
                pool_data = json.load(f)
            self.logger.debug(f"加载了 {len(pool_data)} 条轨迹")
            return pool_data
        except Exception as e:
            self.logger.error(f"加载轨迹池失败: {e}")
            return {}

    def save_pool(self, pool_data: dict[str, Any]) -> None:
        """将轨迹池数据完整保存到文件。"""
        try:
            with open(self.pool_path, "w", encoding="utf-8") as f:
                json.dump(pool_data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"保存了 {len(pool_data)} 条轨迹到轨迹池")
        except Exception as e:
            self.logger.error(f"保存轨迹池失败: {e}")
            raise

    # -----------------------------------------------------------------------
    # 查询
    # -----------------------------------------------------------------------

    def get_trajectory(self, label: str) -> dict[str, Any] | None:
        """
        通过标签获取单个轨迹。

        Args:
            label: 轨迹标签。

        Returns:
            找到的轨迹字典，否则返回 None。
        """
        pool_data = self.load_pool()
        # 优先匹配子键名
        if label in pool_data and isinstance(pool_data[label], dict):
            return pool_data[label]
        # 其次匹配子条目内的 "label" 字段
        for subkey, subval in pool_data.items():
            if subkey == "problem":
                continue
            if isinstance(subval, dict) and str(subval.get("label")) == label:
                return subval
        return None

    def get_all_trajectories(self) -> dict[str, Any]:
        """获取池中所有的轨迹。"""
        return self.load_pool()

    def get_all_labels(self) -> list[str]:
        """获取所有唯一的轨迹标签。"""
        pool_data = self.load_pool()
        labels: set[str] = set()
        for subkey, subval in pool_data.items():
            if subkey == "problem" or not isinstance(subval, dict):
                continue
            if "label" in subval:
                labels.add(str(subval["label"]))
            else:
                labels.add(subkey)
        return sorted(labels)

    # -----------------------------------------------------------------------
    # 写入 / 更新
    # -----------------------------------------------------------------------

    def add_or_update_entry(self, entry: dict[str, Any]) -> None:
        """
        添加或更新一个轨迹条目。

        Args:
            entry: 要添加或更新的轨迹条目，必须包含 'label'。
        """
        pool_data = self.load_pool()

        # 保持顶层 "problem" 描述
        problem_text = entry.get("problem") or pool_data.get("problem")
        if problem_text is not None:
            pool_data["problem"] = problem_text

        # 将本次迭代的 "label" 作为子键，保存条目内容
        iter_label = entry.get("label")
        if not iter_label:
            raise ValueError("缺少 'label' 用于轨迹条目的子键")

        detail = entry.copy()
        detail.pop("problem", None)  # 避免在子条目中重复存储
        pool_data[str(iter_label)] = detail

        self.save_pool(pool_data)
        self.logger.info(f"已更新条目: {iter_label}")
        try:
            best = self._select_best_label(pool_data)
            if best:
                self._best_label = best
        except Exception:
            pass

    def add_trajectory(self, label: str, traj_info: dict[str, Any]) -> None:
        """
        添加单条轨迹记录。

        Args:
            label: 轨迹标签。
            traj_info: 轨迹信息字典。
        """
        # 统一处理 trajectory_raw，确保其为 JSON 对象
        raw_content = traj_info.get("trajectory_raw")
        if isinstance(raw_content, str):
            try:
                trajectory_raw = json.loads(raw_content)
            except json.JSONDecodeError:
                self.logger.warning(f"无法将 trajectory_raw 解析为 JSON (标签: {label})，将作为原始文本存储。")
                trajectory_raw = {"_raw_text": raw_content}
        else:
            trajectory_raw = raw_content

        entry = {
            "problem": traj_info.get("problem_description") or traj_info.get("problem_statement"),
            "label": label,
            "summary": traj_info.get("summary") or {},
            "solution": traj_info.get("solution") or "",
            "metric": traj_info.get("metric"),
            "artifacts": traj_info.get("artifacts") or {},
            "source_dir": traj_info.get("source_dir"),
            "trajectory_raw": trajectory_raw,
            "iteration": traj_info.get("iteration"),
        }
        self.add_or_update_entry(entry)

    def relabel(
        self,
        old_label: str,
        new_label: str,
        operator_name: str | None = None,
        delete_old: bool = False,
    ) -> None:
        """重命名轨迹标签。"""
        pool_data = self.load_pool()
        if old_label not in pool_data:
            raise ValueError(f"标签 '{old_label}' 不存在，无法重命名。")

        old_entry = pool_data.get(old_label)
        new_entry = copy.deepcopy(old_entry) if isinstance(old_entry, dict) else old_entry
        if isinstance(new_entry, dict):
            new_entry["label"] = new_label
            if operator_name is not None:
                new_entry["operator_name"] = operator_name
            new_entry["source_entry_labels"] = [old_label]
        pool_data[str(new_label)] = new_entry
        if delete_old:
            try:
                del pool_data[old_label]
            except Exception:
                pass
        # 更新顶层当前标签
        pool_data["label"] = new_label

        self.save_pool(pool_data)
        self.logger.info(f"重命名 '{old_label}' -> '{new_label}'，operator={operator_name or 'unchanged'}。")

    def delete_trajectories(self, labels: list[str]) -> None:
        """删除指定标签的轨迹。"""
        pool_data = self.load_pool()
        deleted_count = 0
        for lb in labels:
            if lb in pool_data and lb != "problem":
                del pool_data[lb]
                deleted_count += 1
        if deleted_count > 0:
            self.save_pool(pool_data)
        self.logger.info(f"从轨迹池中删除了 {deleted_count} 条轨迹。")

    # -----------------------------------------------------------------------
    # 轨迹总结
    # -----------------------------------------------------------------------

    def summarize_trajectory(
        self,
        trajectory_content: str,
        solution_content: str,
        iteration: int,
        label: str,
        problem_description: str | None = None,
        best_solution_text: str | None = None,
        target_solution_text: str | None = None,
    ) -> dict[str, Any]:
        """
        使用 LLM（或备用方法）总结单条轨迹的内容。

        Args:
            trajectory_content: .tra 文件内容。
            solution_content: 解/代码文本，或 "FAILED_NO_SOLUTION"。
            iteration: 迭代号（用于上下文）。
            label: 轨迹标签（用于日志）。
            problem_description: 问题描述。

        Returns:
            轨迹总结字典。
        """
        from .llm_client import TrajectorySummarizer
        from .traj_summarizer import TrajSummarizer

        summarizer = TrajSummarizer()
        is_failed = not solution_content or solution_content == "FAILED_NO_SOLUTION"

        try:
            if self.llm_client:
                traj_summarizer = TrajectorySummarizer(self.llm_client, prompt_config=self.prompt_config)
                summary = traj_summarizer.summarize_trajectory(
                    trajectory_content,
                    solution_content,
                    iteration,
                    problem_description=problem_description,
                    best_solution_text=best_solution_text,
                    target_solution_text=target_solution_text,
                )
                if is_failed:
                    summary["strategy_status"] = "FAILED"
                    summary["failure_reason"] = "No solution generated"
                self.logger.debug(f"LLM 轨迹总结 (标签 '{label}'): {summary.get('approach_summary', 'N/A')}")
                return summary
            else:
                self.logger.info(f"未配置 LLM 客户端，使用备用总结 (标签 '{label}')")
                summary = summarizer.create_fallback_summary(trajectory_content, solution_content or "", iteration)
                self.logger.debug(f"备用轨迹总结 (标签 '{label}'): {summary.get('approach_summary', 'N/A')}")
                return summary
        except Exception as e:
            self.logger.error(f"轨迹总结失败 (标签 '{label}'): {e}")
            return {
                "error": "summarization_failed",
                "details": str(e),
                "iteration": iteration,
                "label": label,
            }

    def _gather_memory_context(self, res: dict[str, Any]) -> dict[str, Any]:
        """
        准备 Memory 模块所需的上下文信息。

        Args:
            res: 当前轨迹结果字典。

        Returns:
            包含 extract_and_update 所需参数的字典。
        """
        pool_data = self.load_pool()

        # 1. Source Entries（Old Code & Context）
        source_entries = []
        src_labels = res.get("source_entry_labels")
        if src_labels and isinstance(src_labels, list):
            for sl in src_labels:
                sl_str = str(sl)
                if sl_str in pool_data and isinstance(pool_data[sl_str], dict):
                    source_entries.append(pool_data[sl_str])

        # 2. Best Entry（Best Code & Context）
        best_entry = None
        best_label = self._best_label
        if not best_label:
            best_label = self._select_best_label(pool_data)
        if best_label and str(best_label) in pool_data:
            best_entry = pool_data[str(best_label)]

        # 从 artifacts 或顶层字段获取语言和优化目标
        artifacts = res.get("artifacts") or {}
        language = artifacts.get("language") or res.get("language")
        optimization_target = artifacts.get("optimization_target") or res.get("optimization_target")

        return {
            "instance_name": self.instance_name,
            "current_entry": res,
            "source_entries": source_entries,
            "best_entry": best_entry,
            "problem_description": pool_data.get("problem"),
            "language": language,
            "optimization_target": optimization_target,
        }

    def _process_single_trajectory_summary(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """总结单条轨迹并构建完整的 TrajectoryInfo 对象。"""
        try:
            # 从 prompt_config.summarizer.enable_summary 读取是否执行 LLM 总结；默认 True
            do_summary = True
            try:
                summarizer_cfg = (
                    self.prompt_config.get("summarizer", {}) if isinstance(self.prompt_config, dict) else {}
                )
                flag = summarizer_cfg.get("enable_summary")
                if isinstance(flag, bool):
                    do_summary = flag
            except Exception:
                pass

            # 获取当前最佳解的文本
            best_solution_text = ""
            try:
                best_label = self.get_best_label()
                if best_label:
                    pool_data = self.load_pool()
                    cand = pool_data.get(str(best_label))
                    if isinstance(cand, dict):
                        best_solution_text = self.format_entry({str(best_label): cand})
            except Exception:
                best_solution_text = ""

            # 格式化当前目标解的文本
            target_solution_text = ""
            try:
                lab = str(item.get("label") or "target")
                target_solution_text = self.format_entry(
                    {
                        lab: {
                            "label": lab,
                            "iteration": item.get("iteration"),
                            "solution": item.get("solution") or "",
                            "metric": item.get("metric"),
                            "operator_name": item.get("operator_name"),
                        }
                    }
                )
            except Exception:
                target_solution_text = str(item.get("solution") or "")

            summary = None
            if do_summary:
                summary = self.summarize_trajectory(
                    trajectory_content=item["trajectory_content"],
                    solution_content=item.get("solution") or "",
                    iteration=item["iteration"],
                    label=item["label"],
                    problem_description=item.get("problem_description"),
                    best_solution_text=best_solution_text,
                    target_solution_text=target_solution_text,
                )
            else:
                summary = {}

            # 解析 .tra 原始内容为 JSON 对象，如果失败则作为原始文本
            raw_content = item.get("trajectory_content")
            if isinstance(raw_content, str):
                try:
                    trajectory_raw_obj = json.loads(raw_content)
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"无法将 trajectory_raw 解析为 JSON (标签: {item.get('label')})，将作为原始文本存储。"
                    )
                    trajectory_raw_obj = {"_raw_text": raw_content}
            else:
                trajectory_raw_obj = raw_content

            return {
                "label": item["label"],
                "iteration": item["iteration"],
                "solution": item.get("solution") or "",
                "metric": item.get("metric"),
                "artifacts": item.get("artifacts") or {},
                "source_dir": item.get("source_dir"),
                "summary": summary,
                "problem_description": item.get("problem_description"),
                "trajectory_raw": trajectory_raw_obj,
                "source_entry_labels": item.get("source_entry_labels"),
                "operator_name": item.get("operator_name"),
                "meta": {"summary_enabled": bool(do_summary)},
            }
        except Exception as e:
            self.logger.error(f"轨迹总结任务失败 (标签 '{item.get('label')}'): {e}")
            return None

    def summarize_and_add_trajectory(self, trajectory_item: dict[str, Any]) -> bool:
        """
        总结单条轨迹并将其添加到轨迹池中。

        Args:
            trajectory_item: 待处理轨迹信息字典，包含:
                - "label": str
                - "problem_description": str
                - "trajectory_content": str  (.tra 内容)
                - "solution": str            (解/代码文本)
                - "metric": float | str | None  (标量指标，越低越好)
                - "artifacts": dict | None   (任务特定上下文)
                - "iteration": int
                - "source_dir": str
                - "operator_name": str | None
                - "source_entry_labels": list[str] | None

        Returns:
            是否成功处理并添加。
        """
        if not trajectory_item:
            return False

        try:
            label = trajectory_item.get("label", "unknown")
            res = self._process_single_trajectory_summary(trajectory_item)
            if not res:
                self.logger.warning(f"轨迹总结失败 (标签 '{label}')。")
                return False

            # --- 写入轨迹池 --- #
            pool_data = self.load_pool()
            problem_text = res.get("problem_description") or pool_data.get("problem")
            if problem_text is not None:
                pool_data["problem"] = problem_text

            iter_label = res.get("label")
            if not iter_label:
                self.logger.warning(f"跳过缺少 'label' 的轨迹: {res}")
                return False

            detail = res.copy()
            detail.pop("problem_description", None)
            pool_data[str(iter_label)] = detail
            try:
                best = self._select_best_label(pool_data)
                if best:
                    self._best_label = best
            except Exception:
                pass

            # 记忆提炼与更新
            try:
                if self.memory_manager:
                    ctx = self._gather_memory_context(res)
                    self.memory_manager.extract_and_update(**ctx)
            except Exception as me:
                self.logger.warning(f"本地记忆提炼失败（标签 '{iter_label}'): {me}")

            self.save_pool(pool_data)
            self.logger.info(f"成功总结并添加轨迹: {iter_label}")
            return True

        except Exception as e:
            self.logger.error(f"总结与写入轨迹失败: {e}")
            raise

    # -----------------------------------------------------------------------
    # Best Label 管理
    # -----------------------------------------------------------------------

    def _select_best_label(self, pool_data: dict[str, Any]) -> str | None:
        """从池数据中选出性能最优的标签。

        根据 ``self.metric_higher_is_better`` 决定排序方向：
        - False（默认）：metric 越低越好（如运行时间）
        - True：metric 越高越好（如 EM/准确率）
        """
        higher = self.metric_higher_is_better
        # 无效 metric 的哨兵值：始终为"最差"
        worst_val = -float("inf") if higher else float("inf")

        candidates: list[tuple[str, float, int]] = []  # (label, perf, iteration)
        for k, v in pool_data.items():
            if k == "problem" or not isinstance(v, dict):
                continue
            perf_val = v.get("metric")
            try:
                if isinstance(perf_val, (int, float)):
                    val = float(perf_val)
                elif isinstance(perf_val, str):
                    s = perf_val.strip().lower()
                    if s in ("inf", "+inf", "infinity", "+infinity"):
                        val = worst_val
                    elif s in ("-inf", "-infinity"):
                        val = worst_val
                    elif s == "nan":
                        val = worst_val
                    else:
                        val = float(s)
                else:
                    val = worst_val
            except Exception:
                val = worst_val

            label_txt = str(v.get("label") or k)
            it_raw = v.get("iteration")
            try:
                iter_num = int(it_raw) if it_raw is not None else -1
            except Exception:
                iter_num = -1
            candidates.append((label_txt, val, iter_num))

        if not candidates:
            return None

        finite = [c for c in candidates if math.isfinite(c[1])]
        if finite:
            if higher:
                # metric 越大越好：降序排列，相同 metric 时取更新的迭代
                finite.sort(key=lambda t: (-t[1], -t[2]))
            else:
                # metric 越小越好：升序排列，相同 metric 时取更新的迭代
                finite.sort(key=lambda t: (t[1], -t[2]))
            return finite[0][0]
        candidates.sort(key=lambda t: (-t[2], t[0]))
        return candidates[0][0]

    def get_best_label(self) -> str | None:
        """获取当前最佳轨迹的标签。"""
        if isinstance(self._best_label, str) and self._best_label:
            return self._best_label
        pool_data = self.load_pool()
        try:
            best = self._select_best_label(pool_data)
            if best:
                self._best_label = best
            return best
        except Exception:
            return None

    def refresh_best_label(self) -> None:
        """刷新最佳标签缓存。"""
        self._best_label = None
        pool_data = self.load_pool()
        try:
            best = self._select_best_label(pool_data)
            if best:
                self._best_label = best
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 统计 / 提取
    # -----------------------------------------------------------------------

    def get_pool_stats(self) -> dict[str, Any]:
        """获取轨迹池的统计信息。"""
        try:
            pool_data = self.load_pool()
            # 统计轨迹条目数（排除 "problem" 等非轨迹键）
            traj_count = sum(1 for k, v in pool_data.items() if k != "problem" and isinstance(v, dict))
            stats = {
                "total_trajectories": traj_count,
                "labels": self.get_all_labels(),
            }
            self.logger.debug(f"轨迹池统计: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"获取轨迹池统计失败: {e}")
            return {"total_trajectories": 0, "labels": []}

    def _parse_perf(self, val: Any) -> float:
        """将性能值解析为 float（非有限值返回 inf）。"""
        try:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("inf", "+inf", "infinity", "+infinity", "nan"):
                    return float("inf")
                return float(s)
            return float("inf")
        except Exception:
            return float("inf")

    def extract_steps(self) -> list[dict[str, Any]]:
        """提取所有优化步骤（包含前后性能对比）。"""
        steps: list[dict[str, Any]] = []
        pool_data = self.load_pool()

        for key, val in pool_data.items():
            if key == "problem" or not isinstance(val, dict):
                continue
            opn = val.get("operator_name")
            if opn is None or opn in ["filter_trajectories", "plan"]:
                continue
            src_labels = val.get("source_entry_labels")
            if not isinstance(src_labels, list) or not src_labels:
                continue

            sources: list[tuple[str, dict, float]] = []
            for sl in src_labels:
                sl_str = str(sl)
                src = self.get_trajectory(sl_str)
                if isinstance(src, dict):
                    perf_prev = self._parse_perf(src.get("metric"))
                    if math.isfinite(perf_prev):
                        sources.append((sl_str, src, perf_prev))
            perf_curr = self._parse_perf(val.get("metric"))
            if not math.isfinite(perf_curr) or not sources:
                continue
            best_src = min(sources, key=lambda t: t[2])
            best_label_key, best_detail, perf_prev_best = best_src[0], best_src[1], best_src[2]
            improved = perf_curr < min(t[2] for t in sources)
            delta = perf_prev_best - perf_curr
            pct = (
                (delta / perf_prev_best * 100.0) if perf_prev_best != 0 and math.isfinite(perf_prev_best) else None
            )
            prev_it = best_detail.get("iteration")
            try:
                prev_iter = int(prev_it) if prev_it is not None else -1
            except Exception:
                prev_iter = -1
            curr_it = val.get("iteration")
            try:
                curr_iter = int(curr_it) if curr_it is not None else -1
            except Exception:
                curr_iter = -1

            steps.append(
                {
                    "instance_name": self.instance_name,
                    "prev_label": str(best_detail.get("label") or best_label_key),
                    "curr_label": str(val.get("label") or key),
                    "prev_iter": int(prev_iter),
                    "curr_iter": int(curr_iter),
                    "perf_prev": perf_prev_best,
                    "perf_curr": perf_curr,
                    "delta": delta,
                    "pct": pct,
                    "prev_detail": best_detail,
                    "curr_detail": val,
                    "source_labels": [str(sl) for sl, _, _ in sources],
                    "operator_name": str(opn) if opn is not None else None,
                    "improved": bool(improved),
                }
            )
        return steps

    # -----------------------------------------------------------------------
    # 格式化工具（静态方法，被多处引用）
    # -----------------------------------------------------------------------

    @staticmethod
    def _strip_blacklist_messages(text: str) -> str:
        """去除轨迹文本中的黑名单注入消息，减少 token 开销。

        匹配并删除以下模式：
        - ⚠️ IMPORTANT: These answers are ALL WRONG ...
        - ⚠️ STOP: Your answer "..." has been tried ...
        - ⚠️ Your answer "..." is STILL WRONG ...
        以及黑名单消息后附带的指示行（You MUST:, 1., 2., 3. 等）。
        """
        if not text:
            return text
        lines = text.split("\n")
        cleaned: list[str] = []
        skip_until_blank = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("⚠️") or stripped.startswith("\u26a0"):
                skip_until_blank = True
                continue
            if skip_until_blank:
                if not stripped:
                    skip_until_blank = False
                    continue
                # 黑名单消息的延续行
                if stripped.startswith(("You MUST", "1.", "2.", "3.", "The search results")):
                    continue
                skip_until_blank = False
            cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _compress_search_solution(
        solution: str,
        max_info_chars: int = 200,
        max_total_chars: int = 1500,
    ) -> str:
        """压缩 Search-R1 风格的多轮搜索轨迹 solution 文本。

        策略：
        - 保留 <think>...</think> 推理（截断过长的推理块）
        - 保留 <search>query</search> 原文
        - 将 <information>...</information> 块压缩为 "[搜索结果: N 个文档, 约 M 字符]"
        - 保留 <answer>...</answer> 原文
        - 保留 invalid action 提示原文
        - 去除黑名单注入消息（⚠️ 行），大幅减少冗余 token
        - 总体保留轨迹结构可读性，大幅减少 token 数
        - 压缩后若仍超过 max_total_chars，只保留最后几轮搜索

        Args:
            solution: 原始 solution 文本
            max_info_chars: <information> 块保留的最大字符数（0 表示全部替换为摘要）
            max_total_chars: 压缩后的最大总字符数硬上限，超出则截断保留尾部

        Returns:
            压缩后的 solution 文本
        """
        import re as _re

        if not solution:
            return solution

        compressed = solution

        # ---- 去除黑名单注入消息 ----
        compressed = TrajPoolManager._strip_blacklist_messages(compressed)

        if "<information>" not in compressed:
            # 即使没有 <information>，也需要检查总长度
            if max_total_chars > 0 and len(compressed) > max_total_chars:
                compressed = "... [earlier rounds truncated] ...\n" + compressed[-max_total_chars:]
            return compressed

        def _compress_info_block(match: _re.Match) -> str:
            content = match.group(1)
            # 统计文档数量
            doc_count = len(_re.findall(r"Doc \d+\(", content))
            char_count = len(content)
            if max_info_chars > 0 and char_count <= max_info_chars:
                return match.group(0)  # 短的保留原样
            summary = f"[{doc_count} docs, {char_count} chars]"
            return f"<information>{summary}</information>"

        compressed = _re.sub(
            r"<information>(.*?)</information>",
            _compress_info_block,
            compressed,
            flags=_re.DOTALL,
        )

        # 截断过长的 <think> 块（保留首尾各 150 字符）
        def _compress_think_block(match: _re.Match) -> str:
            content = match.group(1)
            if len(content) <= 400:
                return match.group(0)
            head = content[:150].rstrip()
            tail = content[-150:].lstrip()
            return f"<think>{head}\n... [truncated {len(content)} chars] ...\n{tail}</think>"

        compressed = _re.sub(
            r"<think>(.*?)</think>",
            _compress_think_block,
            compressed,
            flags=_re.DOTALL,
        )

        # ---- 硬上限：若压缩后仍超限，保留尾部（最近的搜索轮次更有价值）----
        if max_total_chars > 0 and len(compressed) > max_total_chars:
            compressed = "... [earlier rounds truncated] ...\n" + compressed[-max_total_chars:]

        return compressed

    @staticmethod
    def format_entry(
        data: Any,
        include_keys: set[str] | None = None,
        compact: bool = False,
    ) -> str:
        """格式化轨迹条目为可读文本。

        接受 InstanceTrajectories 或兼容的 dict（向后兼容）。
        选取最新迭代的轨迹并格式化输出。

        Args:
            data: InstanceTrajectories 对象或原始 dict。
            include_keys: 若非 None，仅格式化顶层中属于此集合的键。
            compact: 若为 True，只输出 summary 摘要中对 LLM 有用的字段
                     （approach_summary, search_queries, information_found, answer_given），
                     去掉 solution 原文、metric、artifacts、meta 等元数据，
                     大幅减少 token 消耗。
        """
        import re

        # ---- 统一转换为 {key: TrajectoryItem} ----
        trajectories: dict[str, Any] = {}

        if hasattr(data, "trajectories"):
            trajectories = data.trajectories
        elif isinstance(data, dict):
            for k, v in data.items():
                if k == "problem" or not isinstance(v, dict):
                    continue
                trajectories[str(k)] = v
        if not trajectories:
            return ""

        # ---- 找到最新迭代的轨迹 ----
        def _parse_key_num(k: str) -> int | None:
            if k.isdigit():
                return int(k)
            m = re.search(r"(\d+)$", k)
            return int(m.group(1)) if m else None

        best_num = -1
        latest_key = ""
        latest_item: Any = None

        for key, item in trajectories.items():
            iter_num: int | None = None
            if hasattr(item, "extras"):
                raw = item.extras.get("iteration")
            elif isinstance(item, dict):
                raw = item.get("iteration")
            else:
                raw = None
            if raw is not None:
                try:
                    iter_num = int(raw)
                except (ValueError, TypeError):
                    iter_num = None
            if iter_num is None:
                iter_num = _parse_key_num(key)
            use_num = iter_num if iter_num is not None else -1
            if use_num >= best_num:
                best_num = use_num
                latest_key = key
                latest_item = item

        if latest_item is None:
            return ""

        # ---- 将 TrajectoryItem 转为 dict 以统一格式化 ----
        if hasattr(latest_item, "to_dict"):
            latest_data = latest_item.to_dict()
            chosen_label = latest_item.label
        elif isinstance(latest_item, dict):
            latest_data = latest_item
            chosen_label = latest_item.get("label")
        else:
            return ""

        header = str(chosen_label or latest_key).strip()

        # ---- compact 模式：只输出 summary 中对 LLM 有用的字段 ----
        if compact:
            return TrajPoolManager._format_entry_compact(latest_data, header)

        # ---- 完整格式化 ----
        def indent_str(level: int) -> str:
            return "  " * level

        def fmt_value(val: Any, level: int) -> str:
            if val is None:
                return "null"
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, str):
                if "\n" in val:
                    lines = val.splitlines()
                    pad = indent_str(level + 1)
                    return "|\n" + "\n".join(f"{pad}{line}" for line in lines)
                return val
            if isinstance(val, dict):
                lines: list[str] = []
                for k, v in val.items():
                    if str(k) in {"trajectory_raw", "source_dir"}:
                        continue
                    if level == 0 and include_keys is not None and str(k) not in include_keys:
                        continue
                    key_line = f"{indent_str(level)}{k}:"
                    if str(k) == "solution" and isinstance(v, str):
                        # 对包含搜索轨迹的 solution 做压缩
                        compressed_v = TrajPoolManager._compress_search_solution(v)
                        lines.append(key_line)
                        lines.append(f"```\n{compressed_v}\n```")
                    elif isinstance(v, (dict, list)) or (isinstance(v, str) and "\n" in v):
                        lines.append(key_line)
                        lines.append(fmt_value(v, level + 1))
                    else:
                        lines.append(f"{key_line} {fmt_value(v, 0)}")
                return "\n".join(lines)
            if isinstance(val, list):
                lines: list[str] = []
                for item in val:
                    if isinstance(item, (dict, list)) or (isinstance(item, str) and "\n" in item):
                        lines.append(f"{indent_str(level)}-")
                        lines.append(fmt_value(item, level + 1))
                    else:
                        lines.append(f"{indent_str(level)}- {fmt_value(item, 0)}")
                return "\n".join(lines)
            return str(val)

        body = fmt_value(latest_data, 0)
        return f"{header}\n{body}".strip() if header else body

    @staticmethod
    def _format_entry_compact(data: dict, header: str) -> str:
        """compact 模式的轨迹格式化：只输出 summary 摘要中的关键字段。

        提取 summary 中的 approach_summary、search_queries、information_found、
        answer_given，去掉 solution 原文、metric、artifacts、meta 等元数据。

        Args:
            data: 轨迹的 dict 表示。
            header: 轨迹标签/标题。

        Returns:
            精简后的格式化文本。
        """
        summary = data.get("summary")
        if not summary:
            # 没有 summary，回退到只输出 answer
            solution = data.get("solution", "")
            import re as _re
            m = _re.search(r"<answer>\s*(.*?)\s*</answer>", solution, _re.DOTALL)
            answer = m.group(1).strip() if m else "N/A"
            return f"{header}\n  answer_given: {answer}".strip()

        if isinstance(summary, str):
            # summary 是纯文本，直接返回
            return f"{header}\n{summary}".strip()

        # summary 是 dict，提取关键字段
        useful_keys = [
            "approach_summary",
            "search_queries",
            "information_found",
            "answer_given",
        ]
        lines: list[str] = []
        if header:
            lines.append(header)
        for key in useful_keys:
            val = summary.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                items = "\n".join(f"    - {item}" for item in val)
                lines.append(f"  {key}:\n{items}")
            elif isinstance(val, str):
                lines.append(f"  {key}: {val}")
            else:
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)
