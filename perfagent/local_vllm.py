"""
本地 vLLM 推理引擎，完全对齐 Search-R1 的 token 级续写逻辑。

Search-R1 在 generation.py 中使用 vLLM 进行多轮搜索生成：
  1. tokenize(system_prompt + question) → input_ids
  2. vLLM.generate(prompt_token_ids=input_ids) → response_ids
  3. postprocess: 截断到 </search> 或 </answer>
  4. 判断 action → 调搜索 API → 拼接 <information> → tokenize → 拼接到 input_ids
  5. 重复直到 done 或 max_turns
  6. final rollout: 再生成一次（不调搜索）

本模块将此流程封装为 LocalVLLMEngine，供 SearchR1Runner 调用。
与 Search-R1 的关键对齐点：
- 使用 vLLM.generate(prompt_token_ids=...) 做 token 级续写（非 Chat API）
- _postprocess_responses: 截断到 </search> 或 </answer>
- execute_predictions: 判断 action（search/answer/invalid）
- _update_rolling_state: input_ids = [prev_input + response + next_obs]
- final rollout: 循环后再生成一次，do_search=False
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class LocalVLLMConfig:
    """本地 vLLM 推理配置，对应 Search-R1 的 GenerationConfig"""
    model_path: str
    max_turns: int = 4
    max_response_length: int = 1024    # 每轮最大生成 token 数
    max_prompt_length: int = 8192      # prompt 最大 token 数
    max_obs_length: int = 500          # 搜索结果最大 token 数
    temperature: float = 0.7
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.6
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    # 搜索配置
    search_url: str = "http://127.0.0.1:8001/retrieve"
    topk: int = 3
    search_timeout: int = 10


class LocalVLLMEngine:
    """本地 vLLM 推理引擎，完全对齐 Search-R1 的 run_llm_loop。

    使用方式：
        engine = LocalVLLMEngine(config)
        engine.load_model()
        trajectory = engine.run_search_loop(system_prompt)
    """

    def __init__(self, config: LocalVLLMConfig, _logger: logging.Logger | None = None):
        self._config = config
        self._logger = _logger or logger
        self._llm = None  # vllm.LLM 实例
        self._tokenizer = None
        self._sampling_params = None

    # ------------------------------------------------------------------
    # 修复 transformers / vLLM 兼容性
    # ------------------------------------------------------------------

    def _patch_rope_scaling_if_needed(self) -> None:
        """修复 transformers 自动填充 rope_scaling 与 vLLM 不兼容的问题。

        transformers 对某些模型 (如 Qwen2) 会在 AutoConfig 中自动注入
            rope_scaling = {"rope_theta": ..., "rope_type": "default"}
        但 vLLM 的 _get_and_verify_max_len 期望有 "factor" 键，
        当 rope_type 不在白名单 (su/longrope/llama3/mrope) 且缺少 "factor" 时会报
            AssertionError: assert "factor" in rope_scaling

        修复方法：monkey-patch vLLM 的验证函数，遇到这种情况时临时置 rope_scaling 为 None。
        """
        import json
        import os

        config_path = os.path.join(self._config.model_path, "config.json")
        if not os.path.exists(config_path):
            return

        with open(config_path, "r") as f:
            raw_cfg = json.load(f)

        raw_rope = raw_cfg.get("rope_scaling", None)
        if raw_rope is not None:
            # config.json 里已显式设置了 rope_scaling，不干预
            return

        # 检查 transformers 是否会自动注入
        try:
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(self._config.model_path)
            auto_rope = getattr(hf_cfg, "rope_scaling", None)
        except Exception:
            return

        if auto_rope is None:
            return  # 没有自动注入，没问题

        # transformers 自动注入了 rope_scaling，但原始 JSON 里是 null/缺失
        if isinstance(auto_rope, dict):
            rope_type = auto_rope.get("type") or auto_rope.get("rope_type", "")
            safe_types = {"su", "longrope", "llama3", "mrope"}
            if rope_type not in safe_types and "factor" not in auto_rope:
                self._logger.warning(
                    f"transformers 自动注入了不兼容的 rope_scaling: {auto_rope}, "
                    f"monkey-patch vLLM _get_and_verify_max_len 以绕过"
                )
                self._monkey_patch_vllm_rope_scaling()

    @staticmethod
    def _monkey_patch_vllm_rope_scaling() -> None:
        """Monkey-patch vLLM 的 _get_and_verify_max_len，
        使其在遇到 rope_type='default' 且缺少 'factor' 时跳过 scaling。
        """
        import vllm.config as vllm_config

        original_fn = vllm_config._get_and_verify_max_len

        def _patched_get_and_verify_max_len(
            hf_config, max_model_len, disable_sliding_window, sliding_window_len,
            spec_target_max_model_len=None, **kwargs,
        ):
            rope_scaling = getattr(hf_config, "rope_scaling", None)
            needs_patch = False
            if isinstance(rope_scaling, dict):
                rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type", "")
                safe_types = {"su", "longrope", "llama3", "mrope"}
                if rope_type not in safe_types and "factor" not in rope_scaling:
                    needs_patch = True
                    hf_config.rope_scaling = None

            try:
                result = original_fn(
                    hf_config, max_model_len, disable_sliding_window,
                    sliding_window_len, spec_target_max_model_len, **kwargs,
                )
            finally:
                if needs_patch:
                    hf_config.rope_scaling = rope_scaling

            return result

        vllm_config._get_and_verify_max_len = _patched_get_and_verify_max_len

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """加载 vLLM 模型，对应 Search-R1 的 vLLMRollout.__init__"""
        from vllm import LLM, SamplingParams

        self._logger.info(
            f"加载本地 vLLM 模型: {self._config.model_path}, "
            f"dtype={self._config.dtype}, tp={self._config.tensor_parallel_size}, "
            f"gpu_mem={self._config.gpu_memory_utilization}, "
            f"max_model_len={self._config.max_model_len}"
        )

        # 修复 rope_scaling 兼容性
        self._patch_rope_scaling_if_needed()

        load_start = time.time()
        self._llm = LLM(
            model=self._config.model_path,
            dtype=self._config.dtype,
            tensor_parallel_size=self._config.tensor_parallel_size,
            gpu_memory_utilization=self._config.gpu_memory_utilization,
            max_model_len=self._config.max_model_len,
            trust_remote_code=True,
        )
        self._tokenizer = self._llm.get_tokenizer()

        # 搜索循环中的 sampling params（生成时截断由 postprocess 处理）
        self._sampling_params = SamplingParams(
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            max_tokens=self._config.max_response_length,
            # 不用 stop 参数，靠 postprocess 截断，与 Search-R1 一致
        )

        elapsed = time.time() - load_start
        self._logger.info(f"模型加载完成，耗时 {elapsed:.1f}s")

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # ------------------------------------------------------------------
    # 核心方法：对齐 Search-R1 的 run_llm_loop（单条）
    # ------------------------------------------------------------------

    def run_search_loop(self, system_prompt: str) -> str:
        """执行多轮搜索生成循环，完全对齐 Search-R1 的 run_llm_loop。

        Args:
            system_prompt: 完整的 system prompt（包含问题和上下文）

        Returns:
            完整的生成轨迹文本（包含 <think>, <search>, <information>, <answer> 等标签）
        """
        assert self._llm is not None, "模型未加载，请先调用 load_model()"

        # 1. tokenize 初始 prompt → input_ids (与 Search-R1 的 initial_input_ids 对应)
        input_ids = self._tokenizer.encode(system_prompt, add_special_tokens=True)
        self._logger.info(
            f"初始 prompt token 数: {len(input_ids)}, "
            f"max_turns={self._config.max_turns}"
        )

        # 用于记录完整文本轨迹
        all_response_text = ""
        active = True

        # 2. 主循环 (对应 Search-R1 generation.py run_llm_loop 的 for step in range(max_turns))
        for step in range(self._config.max_turns):
            if not active:
                break

            # 截断 input_ids 到 max_prompt_length（从左边截断，保留最新的 context）
            if len(input_ids) > self._config.max_prompt_length:
                input_ids = input_ids[-self._config.max_prompt_length:]

            self._logger.info(
                f"[轮次 {step + 1}/{self._config.max_turns}] "
                f"input_ids 长度: {len(input_ids)}"
            )

            # vLLM generate（对应 actor_rollout_wg.generate_sequences）
            gen_start = time.time()
            outputs = self._llm.generate(
                prompt_token_ids=[input_ids],
                sampling_params=self._sampling_params,
            )
            gen_elapsed = time.time() - gen_start

            response_text = outputs[0].outputs[0].text
            response_ids = list(outputs[0].outputs[0].token_ids)
            self._logger.info(
                f"[轮次 {step + 1}] 生成 {len(response_ids)} tokens, "
                f"耗时 {gen_elapsed:.1f}s"
            )

            # 3. postprocess: 截断到 </search> 或 </answer>
            #    对应 Search-R1 generation.py _postprocess_responses
            truncated_text = self._postprocess_response(response_text)
            truncated_ids = self._tokenizer.encode(
                truncated_text, add_special_tokens=False
            )

            # 4. execute_prediction: 判断 action, 调搜索 API
            #    对应 Search-R1 generation.py execute_predictions
            action, next_obs, done = self._execute_prediction(
                truncated_text, do_search=True
            )

            self._logger.info(
                f"[轮次 {step + 1}] action={action}, done={done}, "
                f"next_obs 长度={len(next_obs)}"
            )

            # 记录文本
            all_response_text += truncated_text
            if next_obs:
                all_response_text += next_obs

            if done:
                active = False
                break

            # 5. _update_rolling_state: 拼接 input_ids + response + next_obs
            next_obs_ids = self._tokenize_obs(next_obs)
            input_ids = input_ids + truncated_ids + next_obs_ids

        # 6. final rollout（对应 Search-R1 generation.py 的 "final LLM rollout"）
        if active:
            if len(input_ids) > self._config.max_prompt_length:
                input_ids = input_ids[-self._config.max_prompt_length:]

            self._logger.info(
                f"[Final Rollout] input_ids 长度: {len(input_ids)}"
            )

            gen_start = time.time()
            outputs = self._llm.generate(
                prompt_token_ids=[input_ids],
                sampling_params=self._sampling_params,
            )
            gen_elapsed = time.time() - gen_start

            response_text = outputs[0].outputs[0].text
            response_ids = list(outputs[0].outputs[0].token_ids)
            self._logger.info(
                f"[Final Rollout] 生成 {len(response_ids)} tokens, "
                f"耗时 {gen_elapsed:.1f}s"
            )

            # postprocess final
            truncated_text = self._postprocess_response(response_text)

            # execute_prediction with do_search=False
            action, next_obs, done = self._execute_prediction(
                truncated_text, do_search=False
            )

            all_response_text += truncated_text
            self._logger.info(
                f"[Final Rollout] action={action}, done={done}"
            )

        full_text = system_prompt + all_response_text
        self._logger.info(
            f"搜索循环完成，总轨迹长度: {len(full_text)} chars"
        )
        return full_text

    # ------------------------------------------------------------------
    # 辅助方法：对齐 Search-R1 generation.py
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_response(response_text: str) -> str:
        """截断到 </search> 或 </answer>，对应 _postprocess_responses。

        与 Search-R1 完全一致的逻辑:
          if '</search>' in resp: resp = resp.split('</search>')[0] + '</search>'
          elif '</answer>' in resp: resp = resp.split('</answer>')[0] + '</answer>'
          else: resp = resp
        """
        if "</search>" in response_text:
            return response_text.split("</search>")[0] + "</search>"
        elif "</answer>" in response_text:
            return response_text.split("</answer>")[0] + "</answer>"
        return response_text

    def _execute_prediction(
        self, prediction: str, do_search: bool = True
    ) -> tuple[str, str, bool]:
        """判断 action 并执行，对应 execute_predictions（单条版本）。

        Returns:
            (action, next_obs, done)
            - action: "search" / "answer" / None
            - next_obs: 下一轮的 observation 文本
            - done: 是否结束
        """
        action, content = self._postprocess_prediction(prediction)

        if action == "answer":
            return "answer", "", True
        elif action == "search":
            if do_search:
                try:
                    search_results = self._batch_search([content])
                    result_str = search_results[0].strip() if search_results else ""
                except Exception as e:
                    self._logger.error(f"搜索 API 调用失败: {e}")
                    result_str = ""
                next_obs = f"\n\n<information>{result_str}</information>\n\n"
            else:
                next_obs = ""
            return "search", next_obs, False
        else:
            # invalid action，与 Search-R1 一致
            next_obs = (
                "\nMy previous action is invalid. "
                "If I want to search, I should put the query between <search> and </search>. "
                "If I want to give the final answer, I should put the answer between "
                "<answer> and </answer>. Let me try again.\n"
            )
            return None, next_obs, False

    @staticmethod
    def _postprocess_prediction(prediction: str) -> tuple[str | None, str]:
        """从 prediction 文本中提取 action 和 content。

        对应 Search-R1 generation.py postprocess_predictions（单条版本）。
        """
        pattern = r"<(search|answer)>(.*?)</\1>"
        match = re.search(pattern, prediction, re.DOTALL)
        if match:
            action = match.group(1)
            content = match.group(2).strip()
            return action, content
        return None, ""

    def _tokenize_obs(self, obs_text: str) -> list[int]:
        """Tokenize observation 文本并截断到 max_obs_length。

        对应 Search-R1 generation.py _process_next_obs（单条版本）。
        """
        if not obs_text:
            return []
        ids = self._tokenizer.encode(obs_text, add_special_tokens=False)
        if len(ids) > self._config.max_obs_length:
            self._logger.warning(
                f"[OBS 截断] {len(ids)} → {self._config.max_obs_length} tokens"
            )
            ids = ids[: self._config.max_obs_length]
        return ids

    def _batch_search(self, queries: list[str]) -> list[str]:
        """调用搜索 API，对应 Search-R1 generation.py batch_search。"""
        if not queries:
            return []

        payload = {
            "queries": queries,
            "topk": self._config.topk,
            "return_scores": True,
        }

        try:
            resp = requests.post(
                self._config.search_url,
                json=payload,
                timeout=self._config.search_timeout,
            )
            resp.raise_for_status()
            results = resp.json()["result"]
            return [self._passages2string(r) for r in results]
        except Exception as e:
            self._logger.error(f"搜索 API 错误: {e}")
            return [""]

    @staticmethod
    def _passages2string(retrieval_result: list[dict]) -> str:
        """格式化搜索结果，与 Search-R1 generation.py _passages2string 完全一致。"""
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference
