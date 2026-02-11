#!/bin/bash
# 启动 vLLM OpenAI 兼容 API 服务器
# 用于 nanoCSE 的 Search-R1 自进化评测
#
# 对应 Search-R1 evaluate.sh 中的模型配置：
#   actor_rollout_ref.model.path=$BASE_MODEL
#   actor_rollout_ref.rollout.tensor_model_parallel_size=1
#   actor_rollout_ref.rollout.gpu_memory_utilization=0.6
#
# 用法：
#   bash scripts/launch_vllm_server.sh              # 默认配置
#   bash scripts/launch_vllm_server.sh --port 8000  # 自定义端口
#
# 注意：需要先启动检索服务器（另一个终端）：
#   cd /data/user/rli112/junjie/workspace/LLM_Agent/Search-R1
#   conda activate retriever
#   bash retrieval_launch.sh

# 模型路径（与 evaluate.sh 中的 BASE_MODEL 一致）
MODEL_PATH="${MODEL_PATH:-/data/user/rli112/junjie/workspace/LLM_Agent/weight/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3}"

# 服务端口（nanoCSE search_r1.yaml 中 model.api_base 对应的端口）
PORT="${PORT:-8000}"

# GPU 配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=XFORMERS  # 与 evaluate.sh 一致

echo "============================================"
echo "  vLLM Server for Search-R1 + nanoCSE"
echo "============================================"
echo "  Model:  ${MODEL_PATH}"
echo "  Port:   ${PORT}"
echo "  GPU:    ${CUDA_VISIBLE_DEVICES}"
echo "============================================"
echo ""
echo "API endpoint: http://0.0.0.1:${PORT}/v1"
echo ""
echo "Test with:"
echo "  curl http://0.0.0.1:${PORT}/v1/models"
echo ""

# 使用 Search-R1 的 venv（vllm 0.6.3 + 兼容的 transformers）
SEARCH_R1_VENV="/data/user/rli112/junjie/workspace/LLM_Agent/Search-R1/.venv"

# 从路径中提取短名作为 served-model-name（与 search_r1.yaml 中 model.name 一致）
MODEL_NAME="$(basename "${MODEL_PATH}")"

"${SEARCH_R1_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --trust-remote-code \
    "$@"

