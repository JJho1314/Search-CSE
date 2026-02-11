#!/usr/bin/env python3
"""
批量执行 nanoCSE Search-R1 用例并汇总统计指标。

用法:
    # 跑整个 instances 目录
    uv run python SE_Perf/batch_run.py \
        --config configs/search_r1.yaml \
        --instances_dir instances/search_r1_bamboogle

    # 指定部分用例
    uv run python SE_Perf/batch_run.py \
        --config configs/search_r1.yaml \
        --instances instances/search_r1_bamboogle/test_0.json instances/search_r1_bamboogle/test_1.json

    # 并发跑（默认串行）
    uv run python SE_Perf/batch_run.py \
        --config configs/search_r1.yaml \
        --instances_dir instances/search_r1_bamboogle \
        --workers 4

    # 指定范围（从第 10 个到第 20 个，不包含 20）
    uv run python SE_Perf/batch_run.py \
        --config configs/search_r1.yaml \
        --instances_dir instances/search_r1_bamboogle \
        --start 10 --end 20

    # 断点续跑（需要指定 --output_dir 为已有输出目录）
    uv run python SE_Perf/batch_run.py \
        --config configs/search_r1.yaml \
        --instances_dir instances/search_r1_bamboogle \
        --output_dir batch_results/search_r1_batch_20260211_130230 \
        --resume

指标计算参考 Search-R1 ray_trainer.py _validate + compute_evolve_metrics：
- 按 data_source 分组统计 EM (Exact Match)
- 多轮进化时统计 baseline / best / improved_ratio / solved_ratio
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

# 添加 SE 根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))


def _natural_sort_key(path: Path):
    """将文件名中的数字部分按数值排序，实现自然排序（1, 2, 3, ... 10, 11）。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', path.name)
    ]

from perf_config import SEPerfRunSEConfig
from perfagent.task_registry import create_task_runner

# ---------------------------------------------------------------------------
# 单实例执行（可在子进程中调用）
# ---------------------------------------------------------------------------


def run_single_instance(
    config_path: str,
    instance_path: str,
    output_root: str,
) -> dict:
    """
    对单个实例执行完整 nanoCSE 自进化流程，返回结果字典。

    与 perf_run.py main() 逻辑一致，但返回结构化结果而非打印。
    """
    # 延迟导入，避免子进程 pickle 问题
    from core.utils.llm_client import LLMClient
    from core.utils.local_memory_manager import LocalMemoryManager
    from core.utils.se_logger import get_se_logger, setup_se_logging
    from core.utils.traj_pool_manager import TrajPoolManager
    from core.global_memory.utils.config import GlobalMemoryConfig
    from core.utils.global_memory_manager import GlobalMemoryManager
    from iteration_executor import execute_iteration
    from results_io import log_token_usage, aggregate_all_iterations_preds
    from run_helpers import build_perf_agent_config
    from perf_config import LocalMemoryConfig

    instance_path = Path(instance_path)
    result_entry = {
        "instance_id": instance_path.stem,
        "instance_path": str(instance_path),
        "data_source": "unknown",
        "iterations": [],       # 每轮 metric
        "best_metric": None,
        "error": None,
    }

    try:
        # 1. 加载配置
        with open(config_path, encoding="utf-8") as f:
            se_raw = yaml.safe_load(f) or {}
        se_cfg = SEPerfRunSEConfig.from_dict(se_raw)

        # 2. 加载元数据
        task_type = se_cfg.task_type or "search_r1"
        task_runner = create_task_runner(task_type)
        metadata = task_runner.load_metadata(instance_path)
        instance_name = metadata.instance_id or instance_path.stem
        problem_description = metadata.problem_description or ""
        result_entry["instance_id"] = instance_name

        # 提取 data_source
        try:
            raw_data = json.loads(instance_path.read_text(encoding="utf-8"))
            result_entry["data_source"] = raw_data.get("data_source", "unknown")
        except Exception:
            pass

        # 3. 输出目录: {output_root}/{data_source}/{instance_name}/
        ds_name = result_entry["data_source"]
        instance_output = str(Path(output_root) / ds_name / instance_name)
        if Path(instance_output).exists():
            shutil.rmtree(instance_output)
        Path(instance_output).mkdir(parents=True, exist_ok=True)

        log_file = setup_se_logging(instance_output)
        logger = get_se_logger(f"batch_{instance_name}", emoji="⚡")

        os.environ["SE_TOKEN_LOG_PATH"] = str(Path(instance_output) / "token_usage.jsonl")
        os.environ["SE_LLM_IO_LOG_PATH"] = str(Path(instance_output) / "llm_io.jsonl")

        # 4. 初始化组件
        llm_client = None
        try:
            llm_client = LLMClient(se_cfg.model.to_dict())
        except Exception as e:
            logger.warning(f"LLM客户端初始化失败: {e}")

        local_memory = None
        memory_config = se_cfg.local_memory
        if isinstance(memory_config, LocalMemoryConfig) and memory_config.enabled:
            try:
                memory_path = Path(instance_output) / "memory.json"
                local_memory = LocalMemoryManager(
                    memory_path, llm_client=llm_client,
                    format_mode=memory_config.format_mode,
                )
                local_memory.initialize()
            except Exception:
                pass

        traj_pool_path = str(Path(instance_output) / "traj.pool")
        traj_pool_manager = TrajPoolManager(
            traj_pool_path,
            instance_name=instance_name,
            llm_client=llm_client,
            memory_manager=local_memory,
            prompt_config=se_cfg.prompt_config.to_dict(),
            metric_higher_is_better=se_cfg.metric_higher_is_better,
        )
        traj_pool_manager.initialize_pool()

        global_memory = None
        global_memory_config = se_cfg.global_memory_bank
        if isinstance(global_memory_config, GlobalMemoryConfig) and global_memory_config.enabled:
            try:
                global_memory = GlobalMemoryManager(llm_client=llm_client, bank_config=global_memory_config)
            except Exception:
                pass

        # 5. 执行迭代
        iterations = se_cfg.strategy.iterations
        next_iteration_idx = 1

        for step in iterations:
            next_iteration_idx = execute_iteration(
                step=step,
                task_data_path=instance_path,
                instance_id=instance_name,
                problem_description=problem_description,
                se_cfg=se_cfg,
                traj_pool_manager=traj_pool_manager,
                local_memory=local_memory,
                global_memory=global_memory,
                output_dir=instance_output,
                iteration_idx=next_iteration_idx,
                mode="execute",
                logger=logger,
                task_runner=task_runner,
            )

        # 6. 收集每轮 metric
        agg_path = aggregate_all_iterations_preds(Path(instance_output), logger)
        if agg_path and agg_path.exists():
            with open(agg_path, encoding="utf-8") as f:
                preds = json.load(f)
            for iid, entries in preds.items():
                for entry in entries:
                    m = entry.get("metric")
                    result_entry["iterations"].append({
                        "iteration": entry.get("iteration"),
                        "metric": m,
                        "success": entry.get("success", False),
                    })

        # best metric = 最高 metric（EM 中 1.0 = 正确）
        metrics = [
            e["metric"] for e in result_entry["iterations"]
            if e["metric"] is not None
        ]
        result_entry["best_metric"] = max(metrics) if metrics else 0.0

    except Exception as e:
        result_entry["error"] = str(e)

    return result_entry


# ---------------------------------------------------------------------------
# 指标汇总（参考 ray_trainer compute_evolve_metrics）
# ---------------------------------------------------------------------------


def compute_batch_metrics(results: list[dict]) -> dict:
    """
    从批量执行结果中计算汇总指标。

    参考 Search-R1 ray_trainer.py _validate + compute_evolve_metrics：
    - 按 data_source 分组
    - 统计 EM baseline / best / improved / solved
    """
    # 按 data_source 分组
    ds_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        ds = r.get("data_source", "unknown")
        ds_results[ds].append(r)

    metrics = {}
    all_baseline = []
    all_best = []

    for ds, ds_items in sorted(ds_results.items()):
        baselines = []
        bests = []
        for item in ds_items:
            iters = item.get("iterations", [])
            if not iters:
                baselines.append(0.0)
                bests.append(item.get("best_metric", 0.0) or 0.0)
                continue
            # baseline = 第一轮 metric
            first_metric = iters[0].get("metric")
            baselines.append(float(first_metric) if first_metric is not None else 0.0)
            bests.append(float(item.get("best_metric", 0.0) or 0.0))

        n = len(ds_items)
        baseline_mean = sum(baselines) / max(1, n)
        best_mean = sum(bests) / max(1, n)
        improved = sum(1 for b, e in zip(baselines, bests) if e > b)
        solved = sum(1 for r in bests if r >= 1.0)
        errors = sum(1 for item in ds_items if item.get("error"))

        metrics[f"em/{ds}/baseline"] = baseline_mean
        metrics[f"em/{ds}/best"] = best_mean
        metrics[f"em/{ds}/improved"] = f"{improved}/{n}"
        metrics[f"em/{ds}/solved"] = f"{solved}/{n}"
        metrics[f"em/{ds}/solved_ratio"] = solved / max(1, n)
        metrics[f"em/{ds}/errors"] = errors

        all_baseline.extend(baselines)
        all_best.extend(bests)

    # 总体
    total = len(results)
    if total > 0:
        metrics["em/overall/baseline"] = sum(all_baseline) / total
        metrics["em/overall/best"] = sum(all_best) / total
        metrics["em/overall/n"] = total
        metrics["em/overall/improved"] = sum(1 for b, e in zip(all_baseline, all_best) if e > b)
        metrics["em/overall/solved"] = sum(1 for r in all_best if r >= 1.0)
        metrics["em/overall/solved_ratio"] = metrics["em/overall/solved"] / total
        metrics["em/overall/errors"] = sum(1 for r in results if r.get("error"))

    return metrics


def print_metrics(metrics: dict, results: list[dict]):
    """打印汇总指标表格。"""
    print("\n" + "=" * 70)
    print("  nanoCSE Batch Evaluation Summary")
    print("=" * 70)

    # 按 data_source 打印
    ds_set = set()
    for key in metrics:
        parts = key.split("/")
        if len(parts) >= 3:
            ds_set.add(parts[1])

    for ds in sorted(ds_set):
        if ds == "overall":
            continue
        bl = metrics.get(f"em/{ds}/baseline", 0)
        bt = metrics.get(f"em/{ds}/best", 0)
        imp = metrics.get(f"em/{ds}/improved", "0/0")
        slv = metrics.get(f"em/{ds}/solved", "0/0")
        sr = metrics.get(f"em/{ds}/solved_ratio", 0)
        err = metrics.get(f"em/{ds}/errors", 0)
        print(f"\n  [{ds}]")
        print(f"    Baseline EM:   {bl:.4f}")
        print(f"    Best EM:       {bt:.4f}  (+{bt - bl:.4f})")
        print(f"    Improved:      {imp}")
        print(f"    Solved (EM=1): {slv}  ({sr:.1%})")
        if err > 0:
            print(f"    Errors:        {err}")

    # Overall
    bl = metrics.get("em/overall/baseline", 0)
    bt = metrics.get("em/overall/best", 0)
    imp = metrics.get("em/overall/improved", "0/0")
    slv = metrics.get("em/overall/solved", "0/0")
    sr = metrics.get("em/overall/solved_ratio", 0)
    err = metrics.get("em/overall/errors", 0)
    print(f"\n  [OVERALL] ({len(results)} instances)")
    print(f"    Baseline EM:   {bl:.4f}")
    print(f"    Best EM:       {bt:.4f}  (+{bt - bl:.4f})")
    print(f"    Improved:      {imp}")
    print(f"    Solved (EM=1): {slv}  ({sr:.1%})")
    if err > 0:
        print(f"    Errors:        {err}")
    print("=" * 70)


def build_summary_json(metrics: dict, results: list[dict]) -> dict:
    """构建结构化汇总 JSON，包含 per-dataset 和 overall 统计。"""
    ds_set = set()
    for key in metrics:
        parts = key.split("/")
        if len(parts) >= 3:
            ds_set.add(parts[1])

    summary: dict = {"datasets": {}, "overall": {}}

    for ds in sorted(ds_set):
        if ds == "overall":
            continue
        summary["datasets"][ds] = {
            "baseline_em": metrics.get(f"em/{ds}/baseline", 0),
            "best_em": metrics.get(f"em/{ds}/best", 0),
            "improvement": metrics.get(f"em/{ds}/best", 0) - metrics.get(f"em/{ds}/baseline", 0),
            "improved": metrics.get(f"em/{ds}/improved", "0/0"),
            "solved": metrics.get(f"em/{ds}/solved", "0/0"),
            "solved_ratio": metrics.get(f"em/{ds}/solved_ratio", 0),
            "errors": metrics.get(f"em/{ds}/errors", 0),
        }

    summary["overall"] = {
        "total_instances": len(results),
        "baseline_em": metrics.get("em/overall/baseline", 0),
        "best_em": metrics.get("em/overall/best", 0),
        "improvement": metrics.get("em/overall/best", 0) - metrics.get("em/overall/baseline", 0),
        "improved": metrics.get("em/overall/improved", "0/0"),
        "solved": metrics.get("em/overall/solved", "0/0"),
        "solved_ratio": metrics.get("em/overall/solved_ratio", 0),
        "errors": metrics.get("em/overall/errors", 0),
    }

    return summary


# ---------------------------------------------------------------------------
# Resume: 从已有输出目录中收集已完成实例的结果
# ---------------------------------------------------------------------------


def _collect_iteration_metrics_from_preds(preds_path: Path) -> list[dict]:
    """从 preds.json 收集每轮迭代的 metric（用于 resume 读取已有结果）。"""
    iterations = []
    if not preds_path.exists():
        return iterations
    try:
        with open(preds_path, encoding="utf-8") as f:
            preds = json.load(f)
        for _iid, entries in preds.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                iterations.append({
                    "iteration": entry.get("iteration"),
                    "metric": entry.get("metric"),
                    "success": entry.get("success", False),
                })
    except Exception:
        pass
    return iterations


def _check_instance_completed(output_root: str, data_source: str, instance_name: str) -> Path | None:
    """检查实例是否已完成（preds.json 存在），返回其输出目录或 None。"""
    instance_output = Path(output_root) / data_source / instance_name
    preds_path = instance_output / "preds.json"
    if preds_path.exists():
        return instance_output
    return None


def _load_completed_result(instance_path: Path, instance_name: str, data_source: str, output_dir: Path) -> dict:
    """从已完成的实例输出目录中加载结果。"""
    result_entry = {
        "instance_id": instance_name,
        "instance_path": str(instance_path),
        "data_source": data_source,
        "iterations": [],
        "best_metric": None,
        "error": None,
        "status": "skipped",
    }

    preds_path = output_dir / "preds.json"
    result_entry["iterations"] = _collect_iteration_metrics_from_preds(preds_path)

    metrics = [
        e["metric"] for e in result_entry["iterations"]
        if e["metric"] is not None
    ]
    result_entry["best_metric"] = max(metrics) if metrics else 0.0

    return result_entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="nanoCSE 批量执行 Search-R1 用例并汇总指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 跑整个目录
  uv run python SE_Perf/batch_run.py \\
      --config configs/search_r1.yaml \\
      --instances_dir instances/search_r1_bamboogle

  # 指定范围 [10, 20)
  uv run python SE_Perf/batch_run.py \\
      --config configs/search_r1.yaml \\
      --instances_dir instances/search_r1_bamboogle \\
      --start 10 --end 20

  # 断点续跑
  uv run python SE_Perf/batch_run.py \\
      --config configs/search_r1.yaml \\
      --instances_dir instances/search_r1_bamboogle \\
      --output_dir batch_results/search_r1_batch_20260211_130230 \\
      --resume
        """,
    )
    parser.add_argument("--config", default="configs/search_r1.yaml", help="SE 配置文件")
    parser.add_argument("--instances_dir", type=str, default=None,
                        help="实例目录，跑目录下所有 .json 文件")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="指定实例文件路径列表")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出根目录（默认自动生成带时间戳的目录；resume 时需指定已有目录）")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 worker 数（默认 1 = 串行）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制跑的实例数量（调试用，已废弃，建议使用 --start/--end）")
    parser.add_argument("--start", type=int, default=None,
                        help="起始索引（从 0 开始），例如 --start 10 表示从第 11 个实例开始")
    parser.add_argument("--end", type=int, default=None,
                        help="结束索引（不包含），例如 --end 20 表示到第 20 个（不包含）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：跳过输出目录中已有 preds.json 的实例")
    args = parser.parse_args()

    # 收集实例文件列表
    instance_files: list[Path] = []
    if args.instances:
        instance_files = [Path(p) for p in args.instances]
    elif args.instances_dir:
        instance_files = sorted(Path(args.instances_dir).glob("*.json"), key=_natural_sort_key)
    else:
        print("错误: 必须指定 --instances_dir 或 --instances")
        sys.exit(1)

    total_discovered = len(instance_files)

    # 范围选择：优先使用 start/end，兼容 limit
    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end if args.end is not None else len(instance_files)
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(instance_files):
            end_idx = len(instance_files)
        if start_idx >= end_idx:
            print(f"错误: start ({start_idx}) >= end ({end_idx})")
            sys.exit(1)
        instance_files = instance_files[start_idx:end_idx]
        print(f"发现 {total_discovered} 个实例文件，选择范围 [{start_idx}:{end_idx}]，共 {len(instance_files)} 个")
    elif args.limit:
        instance_files = instance_files[:args.limit]
        print(f"发现 {total_discovered} 个实例文件，限制执行前 {args.limit} 个")
    else:
        print(f"发现 {len(instance_files)} 个实例文件")

    if not instance_files:
        print("未找到任何实例文件")
        sys.exit(1)

    print(f"待处理 {len(instance_files)} 个实例，workers={args.workers}")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or f"batch_results/search_r1_batch_{timestamp}"
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Resume 预过滤：跳过已完成的实例，收集其结果
    # -----------------------------------------------------------------
    pending_files: list[Path] = []
    skipped_results: list[dict] = []

    if args.resume:
        # 预加载配置以获取 instance_name（需要 task_runner.load_metadata）
        with open(args.config, encoding="utf-8") as f:
            se_raw = yaml.safe_load(f) or {}
        se_cfg = SEPerfRunSEConfig.from_dict(se_raw)
        task_type = se_cfg.task_type or "search_r1"
        task_runner = create_task_runner(task_type)

        for inst_path in instance_files:
            # 获取 data_source
            try:
                raw_data = json.loads(inst_path.read_text(encoding="utf-8"))
                data_source = raw_data.get("data_source", "unknown")
            except Exception:
                data_source = "unknown"

            # 获取 instance_name（与 run_single_instance 保持一致）
            try:
                metadata = task_runner.load_metadata(inst_path)
                instance_name = metadata.instance_id or inst_path.stem
            except Exception:
                instance_name = inst_path.stem

            completed_dir = _check_instance_completed(output_root, data_source, instance_name)
            if completed_dir is not None:
                result = _load_completed_result(inst_path, instance_name, data_source, completed_dir)
                skipped_results.append(result)
                print(f"  [SKIP] {instance_name} (已完成)")
            else:
                pending_files.append(inst_path)

        print(f"Resume: 跳过 {len(skipped_results)} 个已完成实例，{len(pending_files)} 个待执行")
    else:
        pending_files = instance_files

    # 执行
    all_results: list[dict] = list(skipped_results)
    start_time = time.time()

    if not pending_files:
        print("所有实例均已完成，无需执行")
    elif args.workers <= 1:
        # 串行执行
        for idx, inst_path in enumerate(pending_files):
            print(f"\n[{idx + 1}/{len(pending_files)}] Running {inst_path.name} ...")
            result = run_single_instance(
                config_path=args.config,
                instance_path=str(inst_path),
                output_root=output_root,
            )
            all_results.append(result)
            bm = result.get("best_metric", 0)
            err = result.get("error")
            status = f"EM={bm:.1f}" if not err else f"ERROR: {err[:60]}"
            print(f"  -> {status}")
    else:
        # 并行执行
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for inst_path in pending_files:
                fut = executor.submit(
                    run_single_instance,
                    config_path=args.config,
                    instance_path=str(inst_path),
                    output_root=output_root,
                )
                futures[fut] = inst_path

            for idx, fut in enumerate(as_completed(futures)):
                inst_path = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "instance_id": inst_path.stem,
                        "data_source": "unknown",
                        "iterations": [],
                        "best_metric": 0.0,
                        "error": str(e),
                    }
                all_results.append(result)
                bm = result.get("best_metric", 0)
                err = result.get("error")
                status = f"EM={bm:.1f}" if not err else f"ERROR: {err[:60]}"
                print(f"[{idx + 1}/{len(pending_files)}] {inst_path.name} -> {status}")

    elapsed = time.time() - start_time

    # -----------------------------------------------------------------
    # 按 data_source 分组保存每个用例的详细分数
    # -----------------------------------------------------------------
    ds_grouped: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        ds_grouped[r.get("data_source", "unknown")].append(r)

    for ds, items in ds_grouped.items():
        ds_dir = Path(output_root) / ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        # 每个 data_source 下保存一份 scores.json：每个用例的分数明细
        scores = {}
        for item in items:
            iid = item["instance_id"]
            iter_metrics = [
                {"iteration": e.get("iteration"), "metric": e.get("metric")}
                for e in item.get("iterations", [])
            ]
            scores[iid] = {
                "best_metric": item.get("best_metric", 0.0),
                "baseline_metric": iter_metrics[0]["metric"] if iter_metrics else None,
                "iterations": iter_metrics,
                "error": item.get("error"),
            }
        scores_path = ds_dir / "scores.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

    # 保存全部原始结果
    results_path = Path(output_root) / "batch_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------
    # 计算汇总指标并保存
    # -----------------------------------------------------------------
    metrics = compute_batch_metrics(all_results)
    print_metrics(metrics, all_results)

    # 保存结构化汇总 JSON（包含 per-dataset 和 overall）
    summary = build_summary_json(metrics, all_results)
    summary_path = Path(output_root) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 也保存扁平化的指标
    metrics_path = Path(output_root) / "batch_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 打印最终摘要
    n_skipped = len(skipped_results)
    n_executed = len(all_results) - n_skipped
    print(f"\n  耗时: {elapsed:.1f}s ({elapsed / max(1, n_executed):.1f}s/instance)")
    if n_skipped > 0:
        print(f"  跳过(resume): {n_skipped} 个, 新执行: {n_executed} 个")
    print(f"  汇总: {summary_path}")
    print(f"  指标: {metrics_path}")
    print(f"  输出: {output_root}")
    print(f"\n  目录结构:")
    for ds in sorted(ds_grouped.keys()):
        n = len(ds_grouped[ds])
        print(f"    {output_root}/{ds}/           ({n} instances)")
        print(f"    {output_root}/{ds}/scores.json")


if __name__ == "__main__":
    main()

