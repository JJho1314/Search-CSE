# 只转换 hotpotqa
python scripts/convert_search_r1_parquet.py \
    --parquet /data/user/rli112/junjie/workspace/LLM_Agent/data/nq_hotpotqa_train/test.parquet \
    --output_dir instances/search_r1_hotpotqa/ \
    --data_source hotpotqa

uv run python SE_Perf/perf_run.py     --config ./configs/search_r1.yaml     --instance ./instances/search_r1_bamboogle/test_0.json

# 并行 4 workers
uv run python SE_Perf/batch_run.py \
    --config configs/search_r1.yaml \
    --instances_dir instances/search_r1_bamboogle