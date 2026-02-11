#!/usr/bin/env python3
"""
将 Search-R1 的 parquet 数据集转换为 nanoCSE 单实例 JSON 文件。

用法：
    python scripts/convert_search_r1_parquet.py \
        --parquet /path/to/nq_hotpotqa_train/test.parquet \
        --output_dir instances/search_r1/ \
        --data_source nq \
        --max_instances 100

输入格式（parquet 列）：
    - id: str                   实例 ID
    - question: str             问题文本
    - golden_answers: list[str] 正确答案列表
    - data_source: str          数据源（nq / hotpotqa / ...）
    - prompt: list[dict]        chat 格式的 prompt
    - reward_model: dict        包含 ground_truth.target

输出格式（JSON 文件）：
    {
        "question_id": "test_0",
        "question": "who got the first nobel prize in physics?",
        "ground_truth": ["Wilhelm Conrad Röntgen"],
        "data_source": "nq",
        "prompt": [...]
    }
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert Search-R1 parquet to nanoCSE instance JSONs")
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSON files")
    parser.add_argument("--data_source", default=None,
                        help="Filter by data_source (e.g. 'nq', 'hotpotqa'). None = all")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Max number of instances to convert. None = all")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Which split this is (for naming)")
    args = parser.parse_args()

    # 使用 pyarrow 读取（避免 pandas numpy 兼容性问题）
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow is required. Install with: pip install pyarrow")
        sys.exit(1)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Error: parquet file not found: {parquet_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {parquet_path}...")
    table = pq.read_table(str(parquet_path))
    total_rows = table.num_rows
    print(f"Total rows: {total_rows}")
    print(f"Columns: {table.column_names}")

    converted = 0
    skipped = 0

    for i in range(total_rows):
        if args.max_instances is not None and converted >= args.max_instances:
            break

        row = {col: table.column(col)[i].as_py() for col in table.column_names}

        # 过滤 data_source
        ds = row.get("data_source", "unknown")
        if args.data_source and ds != args.data_source:
            skipped += 1
            continue

        # 提取字段
        instance_id = row.get("id", f"{args.split}_{i}")
        question = row.get("question", "")

        # ground_truth: 优先从 golden_answers，回退到 reward_model.ground_truth.target
        golden_answers = row.get("golden_answers")
        if not golden_answers:
            rm = row.get("reward_model", {})
            if isinstance(rm, dict):
                gt = rm.get("ground_truth", {})
                if isinstance(gt, dict):
                    golden_answers = gt.get("target", [])
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        if not golden_answers:
            golden_answers = []

        # 构建输出
        instance = {
            "question_id": instance_id,
            "question": question,
            "ground_truth": golden_answers,
            "data_source": ds,
        }

        # 可选：保留原始 prompt
        prompt = row.get("prompt")
        if prompt:
            instance["prompt"] = prompt

        # 可选：保留 extra_info
        extra_info = row.get("extra_info")
        if extra_info:
            instance["extra_info"] = extra_info

        # 写入 JSON
        safe_id = str(instance_id).replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"{safe_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(instance, f, ensure_ascii=False, indent=2)

        converted += 1

    print(f"\nDone! Converted {converted} instances, skipped {skipped}")
    print(f"Output directory: {output_dir}")

    # 打印示例
    if converted > 0:
        example_files = sorted(output_dir.glob("*.json"))[:3]
        for ef in example_files:
            print(f"\nExample: {ef.name}")
            with open(ef, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  question: {data['question'][:80]}...")
            print(f"  ground_truth: {data['ground_truth']}")
            print(f"  data_source: {data['data_source']}")


if __name__ == "__main__":
    main()

