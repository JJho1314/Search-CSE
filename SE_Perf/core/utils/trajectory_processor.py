#!/usr/bin/env python3

"""
SE Framework Trajectory Processor

为SE框架提供轨迹文件处理功能，在每个iteration后生成简化的.tra文件。
基于converter_old.py的逻辑，适配SE框架的目录结构。
"""

import json
import re
from pathlib import Path
from typing import Any

# 延迟导入避免循环导入问题
# from .se_logger import get_se_logger


class TrajectoryProcessor:
    """轨迹文件处理器，用于生成简化的.tra文件"""

    def __init__(self):
        """初始化轨迹处理器"""
        # 延迟导入避免循环导入
        try:
            from .se_logger import get_se_logger

            self.logger = get_se_logger("trajectory_processor", emoji="🎬")
        except ImportError:
            # 如果导入失败，使用标准日志
            import logging

            self.logger = logging.getLogger("trajectory_processor")
            self.logger.setLevel(logging.INFO)

    def _count_tokens(self, text: str) -> int:
        """简单的token计数近似算法"""
        if not text or not isinstance(text, str):
            return 0
        # 基础token计数 - 按空白符和常见标点分割
        tokens = re.findall(r"\b\w+\b", text.lower())
        return len(tokens)

    def _compress_search_content(self, text: str, max_info_chars: int = 200) -> str:
        """压缩包含搜索轨迹的文本内容。

        将 <information>...</information> 块替换为简短摘要，
        截断过长的 <think> 块，大幅减少 .tra 文件体积。

        Args:
            text: 原始文本内容
            max_info_chars: <information> 块保留的最大字符数

        Returns:
            压缩后的文本
        """
        if not text or "<information>" not in text:
            return text

        def _compress_info_block(match: re.Match) -> str:
            content = match.group(1)
            doc_count = len(re.findall(r"Doc \d+\(", content))
            char_count = len(content)
            if max_info_chars > 0 and char_count <= max_info_chars:
                return match.group(0)
            summary = f"[{doc_count} docs, {char_count} chars]"
            return f"<information>{summary}</information>"

        compressed = re.sub(
            r"<information>(.*?)</information>",
            _compress_info_block,
            text,
            flags=re.DOTALL,
        )

        # 截断过长的 <think> 块
        def _compress_think_block(match: re.Match) -> str:
            content = match.group(1)
            if len(content) <= 400:
                return match.group(0)
            head = content[:150].rstrip()
            tail = content[-150:].lstrip()
            return f"<think>{head}\n... [truncated {len(content)} chars] ...\n{tail}</think>"

        compressed = re.sub(
            r"<think>(.*?)</think>",
            _compress_think_block,
            compressed,
            flags=re.DOTALL,
        )

        return compressed

    def _truncate_text(self, text: str, first_percent: float = 0.2, last_percent: float = 0.1) -> str:
        """
        使用字符百分比约束截断文本内容

        Args:
            text: 要截断的文本
            first_percent: 保留开头的百分比（默认20%）
            last_percent: 保留结尾的百分比（默认10%）

        Returns:
            截断后的文本
        """
        if not text or not isinstance(text, str):
            return text

        text_length = len(text)

        # 只对足够长的内容进行截断
        if text_length < 300:
            return text

        # 计算首部长度（20%，约束在30-150字符）
        first_length = int(text_length * first_percent)
        first_length = max(30, min(150, first_length))

        # 计算尾部长度（10%，约束在30-100字符）
        last_length = int(text_length * last_percent)
        last_length = max(30, min(100, last_length))

        # 检查截断是否有意义（保留超过80%时不截断）
        truncated_length = first_length + last_length + len("... [TRUNCATED] ...")
        if truncated_length >= text_length * 0.8:
            return text

        # 提取首尾部分
        first_part = text[:first_length]
        last_part = text[-last_length:]

        # 组合截断标记
        return f"{first_part}... [TRUNCATED] ...{last_part}"

    def _truncate_tool_content(self, content) -> str:
        """截断工具输出内容"""
        if not content:
            return content

        # 处理列表格式: [{"type": "text", "text": "..."}]
        if isinstance(content, list) and len(content) > 0:
            first_item = content[0]
            if isinstance(first_item, dict) and "text" in first_item:
                text_content = first_item["text"]
                if isinstance(text_content, str):
                    return self._truncate_text(text_content)

        # 处理字符串格式
        if isinstance(content, str):
            return self._truncate_text(content)

        return content

    def _create_tra_from_traj(self, traj_file: Path, tra_file: Path) -> dict[str, int]:
        """
        从.traj文件创建.tra文件，只保留history的role/content

        Args:
            traj_file: 原始轨迹文件路径
            tra_file: 目标.tra文件路径

        Returns:
            处理统计信息字典
        """
        try:
            with open(traj_file, encoding="utf-8") as f:
                traj_data = json.load(f)

            # 提取并简化 history
            history = traj_data.get("history", [])

            # perf 轨迹：若历史项包含 message_type，则视为 perfagent 轨迹。
            # 仅保留 role 与 content，不做截断或字段改写。
            is_perf_traj = any(isinstance(it, dict) and ("message_type" in it) for it in history)

            simplified_history = []
            total_tokens = 0
            original_tokens = 0  # 原始token数统计

            # perf 轨迹处理
            if is_perf_traj:
                for item in history:
                    if not isinstance(item, dict) or "role" not in item:
                        continue

                    simplified_item = {"role": item["role"]}
                    if "content" in item:
                        content_val = item["content"]
                        # 对包含搜索轨迹的 content 做压缩
                        # （压缩 <information> 块和过长的 <think> 块）
                        if isinstance(content_val, str):
                            content_val = self._compress_search_content(content_val)
                        simplified_item["content"] = content_val
                        content_str = str(content_val) if content_val is not None else ""
                        tokens = self._count_tokens(content_str)
                        original_tokens += tokens
                        total_tokens += tokens

                    if len(simplified_item) > 1:
                        simplified_history.append(simplified_item)

                # 创建.tra文件内容（仅 role/content）
                tra_data = {"Trajectory": simplified_history}

                with open(tra_file, "w", encoding="utf-8") as f:
                    json.dump(tra_data, f, indent=2)

                return {
                    "total_tokens": total_tokens,
                    "original_tokens": original_tokens,
                    "saved_tokens": 0,
                    "compression_ratio": 0,
                    "history_items": len(simplified_history),
                }

            for item in history:
                if "role" not in item:
                    continue

                simplified_item = {"role": item["role"]}

                # 首先统计原始内容的token数
                for field in ["content", "thought", "action"]:
                    if item.get(field):
                        original_field_str = str(item[field]) if item[field] else ""
                        original_tokens += self._count_tokens(original_field_str)

                # 根据角色类型处理不同字段
                if item["role"] == "assistant":
                    # assistant角色：提取thought而非content
                    if item.get("thought"):
                        simplified_item["thought"] = item["thought"]

                    # 包含action并应用截断
                    if item.get("action"):
                        original_action = item["action"]
                        action = original_action

                        # 对str_replace_editor或长action(>350字符)应用截断
                        if isinstance(action, str):
                            if "str_replace_editor" in action or len(action) > 350:
                                action = self._truncate_text(action)
                        elif isinstance(action, dict):
                            action_str = str(action)
                            if "str_replace_editor" in action_str or len(action_str) > 350:
                                action = self._truncate_text(action_str)

                        simplified_item["action"] = action

                else:
                    # 非assistant角色：使用content
                    if item.get("content"):
                        original_content = item["content"]
                        content = original_content

                        # 对tool角色的长观察结果应用截断
                        if item["role"] == "tool":
                            content = self._truncate_tool_content(content)

                        simplified_item["content"] = content

                # 只添加有意义内容的项（不只是role）
                if len(simplified_item) > 1:
                    simplified_history.append(simplified_item)

                    # 统计压缩后字段的token数
                    for field in ["content", "thought", "action"]:
                        if field in simplified_item:
                            field_str = str(simplified_item[field]) if simplified_item[field] else ""
                            total_tokens += self._count_tokens(field_str)

            # 创建.tra文件内容
            tra_data = {"Trajectory": simplified_history}

            # 写入.tra文件
            with open(tra_file, "w", encoding="utf-8") as f:
                json.dump(tra_data, f, indent=2)

            # 计算节省的token数
            saved_tokens = original_tokens - total_tokens
            compression_ratio = (saved_tokens / original_tokens * 100) if original_tokens > 0 else 0

            return {
                "total_tokens": total_tokens,
                "original_tokens": original_tokens,
                "saved_tokens": saved_tokens,
                "compression_ratio": compression_ratio,
                "history_items": len(simplified_history),
            }

        except Exception as e:
            self.logger.error(f"创建.tra文件失败 {traj_file}: {e}")
            return {
                "total_tokens": 0,
                "original_tokens": 0,
                "saved_tokens": 0,
                "compression_ratio": 0,
                "history_items": 0,
            }

    def process_iteration_directory(self, iteration_dir: Path) -> dict[str, Any]:
        """
        处理单个iteration目录，为所有实例生成.tra文件

        Args:
            iteration_dir: iteration目录路径（如iteration_1/）

        Returns:
            处理结果统计信息
        """
        self.logger.info(f"开始处理iteration目录: {iteration_dir}")

        if not iteration_dir.exists() or not iteration_dir.is_dir():
            self.logger.warning(f"目录不存在或不是目录: {iteration_dir}")
            return {}

        processing_stats = {
            "iteration_dir": str(iteration_dir),
            "processed_instances": [],
            "total_tokens": 0,
            "total_tra_files": 0,
            "failed_instances": [],
        }

        # 收集需要处理的 (traj_file, 所在目录, 实例名) 列表
        traj_entries: list[tuple[Path, Path, str]] = []

        # 1) 先检查 iteration_dir 下直接存在的 .traj 文件（单实例扁平模式）
        direct_traj_files = list(iteration_dir.glob("*.traj"))
        for traj_file in direct_traj_files:
            traj_entries.append((traj_file, iteration_dir, traj_file.stem))

        # 2) 再遍历子目录中的 .traj 文件（多实例 / 旧目录结构兼容）
        for instance_dir in iteration_dir.iterdir():
            if not instance_dir.is_dir() or instance_dir.name.startswith("."):
                continue
            for traj_file in instance_dir.glob("*.traj"):
                traj_entries.append((traj_file, instance_dir, instance_dir.name))

        if not traj_entries:
            self.logger.debug(f"迭代目录 {iteration_dir.name} 中没有 .traj 文件")

        # 按实例名分组处理
        instances_map: dict[str, list[tuple[Path, Path]]] = {}
        for traj_file, parent_dir, inst_name in traj_entries:
            instances_map.setdefault(inst_name, []).append((traj_file, parent_dir))

        for inst_name, file_list in instances_map.items():
            instance_stats = {
                "instance_name": inst_name,
                "tra_files_created": [],
                "total_tokens": 0,
                "total_history_items": 0,
            }

            for traj_file, parent_dir in file_list:
                tra_file = parent_dir / (traj_file.stem + ".tra")

                # 生成.tra文件
                file_stats = self._create_tra_from_traj(traj_file, tra_file)

                if file_stats["history_items"] > 0:
                    instance_stats["tra_files_created"].append(
                        {
                            "traj_file": traj_file.name,
                            "tra_file": tra_file.name,
                            "tokens": file_stats["total_tokens"],
                            "original_tokens": file_stats["original_tokens"],
                            "saved_tokens": file_stats["saved_tokens"],
                            "compression_ratio": file_stats["compression_ratio"],
                            "history_items": file_stats["history_items"],
                        }
                    )
                    instance_stats["total_tokens"] += file_stats["total_tokens"]
                    instance_stats["total_history_items"] += file_stats["history_items"]

                    self.logger.info(
                        f"已创建 {tra_file.name}: {file_stats['history_items']} 历史项, "
                        f"{file_stats['total_tokens']} tokens "
                        f"(原始: {file_stats['original_tokens']}, "
                        f"节省: {file_stats['saved_tokens']}, "
                        f"压缩率: {file_stats['compression_ratio']:.1f}%)"
                    )
                else:
                    processing_stats["failed_instances"].append(
                        {
                            "instance_name": inst_name,
                            "traj_file": traj_file.name,
                            "reason": "No valid history items",
                        }
                    )

            if instance_stats["tra_files_created"]:
                processing_stats["processed_instances"].append(instance_stats)
                processing_stats["total_tokens"] += instance_stats["total_tokens"]
                processing_stats["total_tra_files"] += len(instance_stats["tra_files_created"])

                self.logger.info(
                    f"实例 {inst_name}: 创建了 {len(instance_stats['tra_files_created'])} 个.tra文件"
                )

        # 记录处理结果
        self.logger.info(
            f"iteration处理完成: 创建了 {processing_stats['total_tra_files']} 个.tra文件, "
            f"总计 ~{processing_stats['total_tokens']} tokens"
        )

        if processing_stats["failed_instances"]:
            self.logger.warning(f"失败的实例数: {len(processing_stats['failed_instances'])}")

        return processing_stats

    def process_workspace_directory(
        self, workspace_dir: Path, target_iterations: list[int] | None = None
    ) -> dict[str, Any]:
        """
        处理整个workspace目录的所有iterations

        Args:
            workspace_dir: workspace目录路径
            target_iterations: 指定要处理的iteration列表，None表示处理所有

        Returns:
            整体处理结果统计
        """
        self.logger.info(f"开始处理workspace目录: {workspace_dir}")

        if not workspace_dir.exists() or not workspace_dir.is_dir():
            self.logger.error(f"Workspace目录不存在: {workspace_dir}")
            return {}

        workspace_stats = {
            "workspace_dir": str(workspace_dir),
            "iterations_processed": [],
            "total_tokens": 0,
            "total_tra_files": 0,
            "processing_errors": [],
        }

        # 查找所有iteration目录
        iteration_pattern = re.compile(r"^iteration_(\d+)$")
        iteration_dirs = []

        for item in workspace_dir.iterdir():
            if item.is_dir():
                match = iteration_pattern.match(item.name)
                if match:
                    iteration_num = int(match.group(1))
                    if target_iterations is None or iteration_num in target_iterations:
                        iteration_dirs.append((iteration_num, item))

        # 按iteration号排序
        iteration_dirs.sort(key=lambda x: x[0])

        if not iteration_dirs:
            self.logger.warning("未找到任何iteration目录")
            return workspace_stats

        # 处理每个iteration
        for iteration_num, iteration_dir in iteration_dirs:
            try:
                iteration_stats = self.process_iteration_directory(iteration_dir)
                if iteration_stats:
                    workspace_stats["iterations_processed"].append(
                        {"iteration_number": iteration_num, "stats": iteration_stats}
                    )
                    workspace_stats["total_tokens"] += iteration_stats["total_tokens"]
                    workspace_stats["total_tra_files"] += iteration_stats["total_tra_files"]
            except Exception as e:
                error_info = {"iteration_number": iteration_num, "iteration_dir": str(iteration_dir), "error": str(e)}
                workspace_stats["processing_errors"].append(error_info)
                self.logger.error(f"处理iteration_{iteration_num}时出错: {e}")

        # 最终统计
        processed_iterations = len(workspace_stats["iterations_processed"])
        self.logger.info(
            f"Workspace处理完成: {processed_iterations} 个iteration, "
            f"{workspace_stats['total_tra_files']} 个.tra文件, "
            f"~{workspace_stats['total_tokens']} tokens"
        )

        return workspace_stats

    def extract_problem_from_tra(self, tra_file: Path, problem_file: Path) -> bool:
        """
        从.tra文件中提取problem描述并保存为.problem文件

        Args:
            tra_file: .tra文件路径
            problem_file: 目标.problem文件路径

        Returns:
            True表示成功提取，False表示失败
        """
        try:
            with open(tra_file, encoding="utf-8") as f:
                tra_data = json.load(f)

            # 定位到Trajectory[1]["content"][0]["text"]
            trajectory = tra_data.get("Trajectory", [])
            if len(trajectory) < 2:
                self.logger.warning(f"tra文件格式异常，trajectory长度不足: {tra_file}")
                return False

            user_entry = trajectory[1]
            if user_entry.get("role") != "user":
                self.logger.warning(f"trajectory[1]不是user角色: {tra_file}")
                return False

            content = user_entry.get("content", [])
            if not isinstance(content, list) or len(content) == 0:
                self.logger.warning(f"user content格式异常: {tra_file}")
                return False

            text_content = content[0].get("text", "")
            if not text_content:
                self.logger.warning(f"未找到text内容: {tra_file}")
                return False

            # 使用正则提取<pr_description>标签中的内容
            import re

            match = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", text_content, re.DOTALL)
            if not match:
                self.logger.warning(f"未找到pr_description标签: {tra_file}")
                return False

            problem_description = match.group(1).strip()
            if not problem_description:
                self.logger.warning(f"pr_description内容为空: {tra_file}")
                return False

            # 写入.problem文件
            with open(problem_file, "w", encoding="utf-8") as f:
                f.write(problem_description)

            # 统计信息
            problem_tokens = self._count_tokens(problem_description)
            self.logger.info(f"已提取problem: {problem_file.name} ({problem_tokens} tokens)")

            return True

        except Exception as e:
            self.logger.error(f"提取problem失败 {tra_file}: {e}")
            return False

    def process_problems_in_iteration(self, iteration_dir: Path) -> dict[str, Any]:
        """
        为iteration目录中的所有实例提取problem文件

        Args:
            iteration_dir: iteration目录路径

        Returns:
            提取结果统计
        """
        self.logger.info(f"开始提取iteration目录的problems: {iteration_dir}")

        if not iteration_dir.exists() or not iteration_dir.is_dir():
            self.logger.warning(f"目录不存在或不是目录: {iteration_dir}")
            return {}

        problem_stats = {
            "iteration_dir": str(iteration_dir),
            "problems_extracted": [],
            "total_problems": 0,
            "failed_extractions": [],
        }

        # 遍历所有实例目录
        for instance_dir in iteration_dir.iterdir():
            if not instance_dir.is_dir() or instance_dir.name.startswith("."):
                continue

            # 查找.tra文件
            tra_files = list(instance_dir.glob("*.tra"))
            if not tra_files:
                self.logger.debug(f"实例 {instance_dir.name} 没有.tra文件")
                continue

            # 处理每个.tra文件（通常只有一个）
            for tra_file in tra_files:
                problem_file = instance_dir / (instance_dir.name + ".problem")

                success = self.extract_problem_from_tra(tra_file, problem_file)

                if success:
                    problem_stats["problems_extracted"].append(
                        {
                            "instance_name": instance_dir.name,
                            "tra_file": tra_file.name,
                            "problem_file": problem_file.name,
                        }
                    )
                    problem_stats["total_problems"] += 1
                else:
                    problem_stats["failed_extractions"].append(
                        {
                            "instance_name": instance_dir.name,
                            "tra_file": tra_file.name,
                            "reason": "Problem extraction failed",
                        }
                    )

        self.logger.info(
            f"problem提取完成: 成功 {problem_stats['total_problems']} 个, "
            f"失败 {len(problem_stats['failed_extractions'])} 个"
        )

        return problem_stats


def process_trajectory_files(workspace_dir: str, iterations: list[int] | None = None) -> dict[str, Any]:
    """
    便捷函数：处理轨迹文件

    Args:
        workspace_dir: workspace目录路径
        iterations: 要处理的iteration列表，None表示处理所有

    Returns:
        处理结果统计
    """
    processor = TrajectoryProcessor()
    return processor.process_workspace_directory(Path(workspace_dir), iterations)


def extract_problems_from_workspace(workspace_dir: str, iterations: list[int] | None = None) -> dict[str, Any]:
    """
    便捷函数：从workspace提取problem文件

    Args:
        workspace_dir: workspace目录路径
        iterations: 要处理的iteration列表，None表示处理所有

    Returns:
        提取结果统计
    """
    import re

    processor = TrajectoryProcessor()
    workspace_path = Path(workspace_dir)

    if not workspace_path.exists():
        return {"error": f"Workspace目录不存在: {workspace_path}"}

    # 查找iteration目录
    iteration_pattern = re.compile(r"^iteration_(\d+)$")
    iteration_dirs = []

    for item in workspace_path.iterdir():
        if item.is_dir():
            match = iteration_pattern.match(item.name)
            if match:
                iteration_num = int(match.group(1))
                if iterations is None or iteration_num in iterations:
                    iteration_dirs.append((iteration_num, item))

    iteration_dirs.sort(key=lambda x: x[0])

    workspace_results = {
        "workspace_dir": str(workspace_path),
        "iterations_processed": [],
        "total_problems": 0,
        "total_failed": 0,
    }

    for iteration_num, iteration_dir in iteration_dirs:
        problem_stats = processor.process_problems_in_iteration(iteration_dir)
        if problem_stats:
            workspace_results["iterations_processed"].append(
                {"iteration_number": iteration_num, "stats": problem_stats}
            )
            workspace_results["total_problems"] += problem_stats.get("total_problems", 0)
            workspace_results["total_failed"] += len(problem_stats.get("failed_extractions", []))

    return workspace_results


# 使用示例
if __name__ == "__main__":
    # 示例：处理Demo_Structure目录
    demo_workspace = "/home/uaih3k9x/630_swe/SE/trajectories/Demo_Structure"

    processor = TrajectoryProcessor()
    results = processor.process_workspace_directory(Path(demo_workspace))

    print("处理结果:")
    print(f"- 处理的iterations: {len(results['iterations_processed'])}")
    print(f"- 创建的.tra文件: {results['total_tra_files']}")
    print(f"- 总token数: ~{results['total_tokens']}")
