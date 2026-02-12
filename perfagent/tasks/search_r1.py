"""
Search-R1 TaskRunner 实现

支持 Search-R1 格式的 QA 搜索推理任务，使用搜索增强型 LLM 进行多轮检索与回答。

核心流程（对应 ray_trainer_backup.py 的 _validate）：
1. 加载 QA 实例（question + ground_truth）
2. LLM 通过多轮搜索进行推理（<search>query</search> → <information>...</information>）
3. 提取最终答案（<answer>...</answer>）
4. 使用 Exact Match (EM) 评估答案正确性

多轮搜索交互（方案 B）：
- PerfAgent._call_llm() 返回第一轮 LLM 响应
- extract_solution() 检测 <search> 标签 → 调用搜索 API → 拼接 <information>
  → 再次调用 LLM → 循环直到 <answer> 或达到 max_search_turns
- 对应 ray_trainer.py 中 LLMGenerationManager.run_llm_loop() 的多轮逻辑
"""

from __future__ import annotations

import json
import logging
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import requests

from perfagent.task_runner import BaseTaskRunner
from perfagent.protocols import TaskMetadata

if TYPE_CHECKING:
    from perfagent.llm_client import LLMClient
    from perfagent.local_vllm import LocalVLLMEngine


# ======================================================================
# 配置
# ======================================================================

@dataclass
class SearchR1TaskConfig:
    """Search-R1 任务特定配置"""
    search_url: str = "http://127.0.0.1:8001/retrieve"
    topk: int = 3
    max_search_turns: int = 5
    timeout: int = 10
    # 搜索结果最大字符数（对应 Search-R1 的 max_obs_length=500 tokens）
    # 500 tokens ≈ 2000 字符（英文为主），设为 2000 字符作为安全默认值
    max_obs_chars: int = 2000

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None) -> SearchR1TaskConfig:
        if config_dict is None:
            return cls()
        return cls(
            search_url=config_dict.get("search_url", "http://127.0.0.1:8001/retrieve"),
            topk=config_dict.get("topk", 3),
            max_search_turns=config_dict.get("max_search_turns", 5),
            timeout=config_dict.get("timeout", 10),
            max_obs_chars=config_dict.get("max_obs_chars", 2000),
        )


# ======================================================================
# 实例数据
# ======================================================================

@dataclass
class SearchR1Instance:
    """Search-R1 QA 实例数据"""
    id: str
    question: str
    ground_truth: list[str]
    data_source: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, file_path: Path | None = None) -> SearchR1Instance:
        instance_id = data.get("question_id") or data.get("id") or "unknown"
        if file_path and instance_id == "unknown":
            instance_id = file_path.stem

        gt = data.get("ground_truth") or data.get("target") or data.get("answer", [])
        if isinstance(gt, str):
            gt = [gt]
        elif isinstance(gt, dict):
            gt = gt.get("target", [])
            if isinstance(gt, str):
                gt = [gt]

        question = data.get("question", "")
        if not question:
            prompt = data.get("prompt", [])
            if isinstance(prompt, list):
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg.get("content", "")
                        break
            elif isinstance(prompt, str):
                question = prompt

        return cls(
            id=str(instance_id),
            question=question,
            ground_truth=gt,
            data_source=data.get("data_source", "unknown"),
            metadata=data.get("metadata", {}),
        )


# ======================================================================
# EM 评估工具（移植自 Search-R1 verl/utils/reward_score/qa_em.py）
# ======================================================================

def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _em_check(prediction: str, golden_answers: list[str]) -> bool:
    normalized_pred = _normalize_answer(prediction)
    return any(_normalize_answer(ga) == normalized_pred for ga in golden_answers)


def _subem_check(prediction: str, golden_answers: list[str]) -> bool:
    normalized_pred = _normalize_answer(prediction)
    return any(_normalize_answer(ga) in normalized_pred for ga in golden_answers)


def _extract_answer(text: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def _extract_search_query(text: str) -> str | None:
    """从 LLM 响应末尾提取最后一个 <search>query</search>。"""
    matches = list(re.finditer(r"<search>(.*?)</search>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


# ======================================================================
# 搜索工具
# ======================================================================

def _batch_search(queries: list[str], search_url: str, topk: int = 3,
                  timeout: int = 10, max_obs_chars: int = 2000) -> list[str]:
    """批量搜索并格式化结果。

    对应 Search-R1 的 ``max_obs_length`` 截断逻辑：
    Search-R1 在 token 级别截断搜索结果（默认 500 tokens），防止
    ``<information>`` 块过长挤占模型的推理和回答空间。
    此处使用字符级截断作为等效实现（500 tokens ≈ 2000 字符）。
    """
    if not queries:
        return []
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    try:
        resp = requests.post(search_url, json=payload, timeout=timeout)
        results = resp.json().get("result", [])
    except Exception:
        return ["[Search failed]"] * len(queries)

    formatted = []
    for result in results:
        parts = []
        for idx, doc_item in enumerate(result):
            content = doc_item.get("document", {}).get("contents", "")
            title = content.split("\n")[0] if content else ""
            text = "\n".join(content.split("\n")[1:]) if content else ""
            parts.append(f"Doc {idx + 1}(Title: {title}) {text}")
        combined = "\n".join(parts)

        # 截断过长的搜索结果（对应 Search-R1 的 max_obs_length 截断）
        if max_obs_chars > 0 and len(combined) > max_obs_chars:
            combined = combined[:max_obs_chars]

        formatted.append(combined)
    return formatted


# ======================================================================
# TaskRunner 实现
# ======================================================================

class SearchR1Runner(BaseTaskRunner):
    """Search-R1 QA 搜索推理任务的 TaskRunner 实现。

    多轮搜索交互（方案 B）：
    - extract_solution() 中实现搜索循环
    - 检测 LLM 响应中的 <search> 标签 → 调用搜索 API → 拼接结果 → 继续调用 LLM
    - 对应 ray_trainer.py 中 LLMGenerationManager.run_llm_loop() 的多轮逻辑
    """

    def __init__(
        self,
        *,
        task_config: dict[str, Any] | None = None,
        _logger: logging.Logger | None = None,
    ):
        self._logger = _logger or logging.getLogger(__name__)
        self._task_config = SearchR1TaskConfig.from_dict(task_config)
        # LLM client 由 PerfAgent 在运行时注入（Chat API 模式，作为 fallback）
        self._llm_client: LLMClient | None = None
        self._llm_temperature: float = 0.7
        self._llm_max_tokens: int | None = None
        # 本地 vLLM 推理引擎（优先使用，完全对齐 Search-R1 的 token 级续写）
        self._local_engine: LocalVLLMEngine | None = None
        # 上下文长度安全阈值（字符数）；由 set_llm_client 从 max_input_tokens 推算
        # 粗略估算：1 token ≈ 3.5 字符（英文为主），保留 output 余量
        self._max_context_chars: int = 90000  # 默认 ~25k tokens
        # 缓存最近一次 build_system_prompt 的结果，供 _run_search_loop 使用
        self._last_system_prompt: str = ""

    def set_llm_client(self, llm_client: LLMClient | None,
                       temperature: float = 0.7,
                       max_tokens: int | None = None):
        """注入 LLM 客户端，供 extract_solution 中的搜索循环使用。

        由 PerfAgent 在 _init_run_context 或 _process_single_iteration 中调用。
        """
        self._llm_client = llm_client
        self._llm_temperature = temperature
        self._llm_max_tokens = max_tokens

        # 从 LLM client config 推算上下文安全阈值
        if llm_client is not None:
            cfg = getattr(llm_client, 'config', {}) or {}
            max_input = cfg.get('max_input_tokens', 0)
            max_output = cfg.get('max_output_tokens', max_tokens or 5000)
            if max_input > 0:
                # 安全余量：input 上限减去 output 预留，再留 10% buffer
                safe_tokens = int((max_input - max_output) * 0.9)
                self._max_context_chars = max(safe_tokens * 4, 20000)  # 1 token ≈ 4 chars
                self._logger.info(
                    f"上下文安全阈值: {self._max_context_chars} 字符 "
                    f"(max_input={max_input}, max_output={max_output})"
                )

    def set_local_engine(self, engine: LocalVLLMEngine) -> None:
        """注入本地 vLLM 推理引擎。

        当 local_engine 可用时，extract_solution 会使用 token 级续写
        （完全对齐 Search-R1 的 run_llm_loop），而不是 Chat API。

        由 PerfAgent 在 _init_run_context 中调用。
        """
        self._local_engine = engine
        self._logger.info(
            f"已注入本地 vLLM 引擎: loaded={engine.is_loaded}"
        )

    # ------------------------------------------------------------------
    # LLM 停止序列
    # ------------------------------------------------------------------

    @property
    def llm_stop_sequences(self) -> list[str] | None:
        """让 PerfAgent 在首次 LLM 调用时也使用 stop=["</search>"]。

        这对应 Search-R1 中 StopOnSequence(["</search>"]) 的机制，
        防止 LLM 在生成 <search>query</search> 后继续伪造 <information> 块。
        """
        return ["</search>"]

    # ------------------------------------------------------------------
    # 元数据提取
    # ------------------------------------------------------------------

    @classmethod
    def load_metadata(cls, path: Path) -> TaskMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        instance = SearchR1Instance.from_dict(data, path)
        return TaskMetadata(
            instance_id=instance.id,
            problem_description=instance.question,
        )

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_instance(self, path: Path) -> SearchR1Instance:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SearchR1Instance.from_dict(data, path)

    # ------------------------------------------------------------------
    # 初始解
    # ------------------------------------------------------------------

    def get_initial_solution(self, instance_data: Any, config: Any) -> str:
        return ""

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(
        self,
        solution: str,
        instance_data: Any,
        config: Any,
    ) -> tuple[float, dict[str, Any]]:
        instance: SearchR1Instance = instance_data
        answer = _extract_answer(solution)

        # 注意：ground_truth 绝不能放进 artifacts，否则会泄露到 prompt 里
        # artifacts 会被拼进 optimization prompt 和 trajectory summary，模型能看到
        artifacts: dict[str, Any] = {
            "extracted_answer": answer,
            "data_source": instance.data_source,
        }

        if answer is None:
            artifacts["failure_reason"] = "No <answer>...</answer> tag found"
            self._logger.info(f"实例 {instance.id}: 未找到答案标签")
            return 0.0, artifacts

        em_passed = _em_check(answer, instance.ground_truth)
        subem_passed = _subem_check(answer, instance.ground_truth)

        metric = 1.0 if em_passed else 0.0
        artifacts["em_match"] = em_passed
        artifacts["subem_match"] = subem_passed

        if not em_passed:
            artifacts["failure_reason"] = "Answer does not match ground truth"

        # ground_truth 只写日志，不进 artifacts
        self._logger.info(
            f"实例 {instance.id}: EM={'通过' if em_passed else '未通过'}, "
            f"answer='{answer}', ground_truth(first 3)={instance.ground_truth[:3]}"
        )

        return metric, artifacts

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def build_system_prompt(self, instance_data: Any, **context: Any) -> str:
        instance: SearchR1Instance = instance_data
        config = context.get("config")

        additional_requirements = (
            context.get("additional_requirements")
            or getattr(getattr(config, "prompts", None), "additional_requirements", None)
            or ""
        )

        base_prompt = (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between "
            "<information> and </information>. "
            "You can search as many times as you want. "
            "If you find no further external knowledge needed, you can directly provide the answer "
            "inside <answer> and </answer>, without detailed illustrations. "
            "For example, <answer> Beijing </answer>.\n\n"
            f"Question: {instance.question}\n"
        )

        if additional_requirements:
            base_prompt += f"\n{additional_requirements}\n"

        # 缓存 system prompt，供 _run_search_loop 构建消息时使用
        self._last_system_prompt = base_prompt
        return base_prompt

    def build_optimization_prompt(
        self,
        solution: str,
        metric: float,
        artifacts: dict[str, Any],
        **context: Any,
    ) -> str:
        passed = metric >= 1.0
        extracted_answer = artifacts.get("extracted_answer", "None")
        failure_reason = artifacts.get("failure_reason", "")

        prompt_parts = [
            "## Previous Attempt Result\n",
            f"- Status: {'✅ Correct' if passed else '❌ Incorrect'}",
            f"- Extracted answer: {extracted_answer}",
        ]

        if not passed and failure_reason:
            prompt_parts.append(f"- Failure reason: {failure_reason}")

        # 精炼之前的搜索轨迹，而非简单截断
        if solution and len(solution) > 1500:
            condensed = self._condense_trajectory_for_prompt(solution)
            prompt_parts.append(
                "\n## Your Previous Search & Reasoning Summary\n"
                f"{condensed}\n"
            )
        else:
            prompt_parts.append(
                "\n## Your Previous Reasoning Trajectory\n"
                f"{solution}\n"
            )

        if not passed:
            prompt_parts.append(
                "\n## Instructions\n"
                "Your previous answer was incorrect. Please:\n"
                "1. Re-read the question carefully\n"
                "2. Search for more relevant information using <search>query</search>\n"
                "3. Reason step by step inside <think>...</think>\n"
                "4. Provide a corrected answer inside <answer>...</answer>\n"
            )
        else:
            prompt_parts.append(
                "\n## Instructions\n"
                "Your previous answer was correct. "
                "Try to provide the answer again, potentially with better reasoning.\n"
            )

        return "\n".join(prompt_parts)

    @staticmethod
    def _condense_trajectory_for_prompt(solution: str, max_chars: int = 1500) -> str:
        """将长的多轮搜索轨迹精炼为紧凑摘要，用于 optimization prompt。

        提取关键信息：搜索查询、发现的关键事实、推理链、最终答案。
        去掉大段重复的搜索结果原文。
        """
        parts: list[str] = []

        # 提取搜索查询（去重）
        queries = re.findall(r"<search>(.*?)</search>", solution, re.DOTALL)
        if queries:
            parts.append(f"Searches performed ({len(queries)} queries):")
            seen: set[str] = set()
            for i, q in enumerate(queries, 1):
                qs = q.strip()
                if qs not in seen:
                    parts.append(f"  {i}. {qs}")
                    seen.add(qs)

        # 提取搜索结果摘要（只保留标题）
        info_blocks = re.findall(r"<information>(.*?)</information>", solution, re.DOTALL)
        if info_blocks:
            parts.append(f"\nInformation found ({len(info_blocks)} results):")
            for i, info in enumerate(info_blocks, 1):
                titles = re.findall(r'Title:\s*"?([^")\n]+)', info)
                if titles:
                    parts.append(f"  {i}. {'; '.join(t.strip() for t in titles[:3])}")
                else:
                    snippet = info.strip().replace("\n", " ")[:80]
                    if snippet:
                        parts.append(f"  {i}. {snippet}")

        # 提取 think 块中的关键推理（只取最后一个有实质内容的）
        thinks = re.findall(r"<think>(.*?)</think>", solution, re.DOTALL)
        if thinks:
            key_thinks = [t.strip() for t in thinks if len(t.strip()) > 30]
            if key_thinks:
                last_think = key_thinks[-1]
                if len(last_think) > 300:
                    last_think = last_think[:300] + "..."
                parts.append(f"\nLast reasoning step:\n  {last_think}")

        # 提取最终答案
        answer_match = re.search(r"<answer>(.*?)</answer>", solution, re.DOTALL)
        if answer_match:
            parts.append(f"\nAnswer given: {answer_match.group(1).strip()}")
        else:
            parts.append("\n[No <answer> tag found in trajectory]")

        condensed = "\n".join(parts)
        if len(condensed) > max_chars:
            condensed = condensed[:max_chars] + "\n... [truncated]"
        return condensed

    # ------------------------------------------------------------------
    # 解提取 + 多轮搜索交互循环
    # ------------------------------------------------------------------

    def extract_solution(self, llm_response: str, current_solution: str) -> str:
        """从 LLM 响应中提取完整的搜索推理轨迹。

        优先使用本地 vLLM 引擎（token 级续写，完全对齐 Search-R1）；
        如果未配置本地引擎，回退到 Chat API 模式。

        本地 vLLM 模式：
        - 忽略 PerfAgent 传入的 llm_response（因为 PerfAgent 是通过 Chat API 生成的）
        - 直接使用 LocalVLLMEngine.run_search_loop(system_prompt) 完成整个多轮推理
        - 完全对齐 Search-R1 的 run_llm_loop: token 级续写 + postprocess + execute_predictions

        Chat API fallback 模式：
        - 使用 PerfAgent 的 llm_response 作为第一轮输出
        - 通过 _run_search_loop() 进行多轮搜索（存在格式对齐风险）
        """
        # ---- 优先使用本地 vLLM 引擎 ----
        if self._local_engine is not None and self._local_engine.is_loaded:
            self._logger.info("使用本地 vLLM 引擎执行多轮搜索（token 级续写）")
            try:
                trajectory = self._local_engine.run_search_loop(self._last_system_prompt)
                if trajectory and trajectory.strip():
                    return trajectory.strip()
                self._logger.warning("本地 vLLM 引擎返回空轨迹，回退到 Chat API")
            except Exception as e:
                self._logger.error(f"本地 vLLM 引擎执行失败: {e}，回退到 Chat API")

        # ---- Chat API fallback ----
        if not llm_response or not llm_response.strip():
            self._logger.warning("LLM 响应为空，返回当前解")
            return current_solution

        # OpenAI API 的 stop 参数会在匹配时停止，但返回的文本不包含 stop 序列本身。
        if "<search>" in llm_response and "</search>" not in llm_response:
            llm_response += "</search>"
            self._logger.info("补充被 stop 参数截断的 </search> 标签")

        # 如果已经有 <answer>，无需搜索循环
        if "</answer>" in llm_response:
            return llm_response.strip()

        # 如果没有 <search>，也无需循环
        if "</search>" not in llm_response:
            if "<answer>" in llm_response or "<think>" in llm_response:
                return llm_response.strip()
            self._logger.warning("无法从响应中提取有效轨迹，返回当前解")
            return current_solution

        # 没有 LLM client 则无法执行搜索循环
        if self._llm_client is None:
            self._logger.warning(
                "检测到 <search> 标签但无 LLM client，无法执行搜索循环，返回单轮结果"
            )
            return llm_response.strip()

        # ---- 多轮搜索交互循环（Chat API fallback）----
        return self._run_search_loop(llm_response)

    @staticmethod
    def _detect_repetition(text: str, min_fragment_len: int = 60, max_repeats: int = 3) -> bool:
        """检测文本中是否存在大量重复片段（LLM 退化生成）。

        策略：取文本末尾的一个片段，检查该片段在整段文本中出现的次数。
        如果超过阈值则判定为重复退化。
        """
        if len(text) < min_fragment_len * 2:
            return False
        # 取末尾片段
        fragment = text[-min_fragment_len:]
        count = text.count(fragment)
        return count > max_repeats

    @staticmethod
    def _classify_action(text: str) -> tuple[str | None, str]:
        """对应 Search-R1 的 postprocess_predictions + _postprocess_responses。

        将 LLM 响应截断到第一个 </search> 或 </answer>，并判断 action 类型：
        - 'search': 包含 <search>query</search>
        - 'answer': 包含 <answer>answer</answer>
        - None: 无有效 action（无效响应）
        """
        # 对应 _postprocess_responses: 截断到 </search> 或 </answer>
        if '</search>' in text:
            text = text.split('</search>')[0] + '</search>'
        elif '</answer>' in text:
            text = text.split('</answer>')[0] + '</answer>'

        # 对应 postprocess_predictions: 用正则判断 action
        match = re.search(r'<(search|answer)>(.*?)</\1>', text, re.DOTALL)
        if match:
            action = match.group(1)  # 'search' or 'answer'
            return action, text
        return None, text

    def _build_continuation_messages(self, accumulated_text: str) -> list[dict[str, str]]:
        """构建搜索循环中用于 LLM 续写的消息列表。

        Search-R1 的 run_llm_loop 是 token 级续写：模型看到的 input_ids 为
        [system_prompt_tokens + 之前轨迹_tokens]，然后从轨迹末尾继续生成。
        模型始终能看到 system prompt 中的格式指令（如 "provide the answer inside
        <answer> and </answer>, without detailed illustrations"）。

        Chat API 无法做 token 级续写，因此需要用 assistant prefix continuation 模式
        来模拟：
        - system: 原始的 system prompt（包含格式指令和问题）
        - assistant: 之前的轨迹（作为 assistant 的部分响应，让模型续写）

        这样模型知道：
        1. 自己的角色和格式要求（来自 system prompt）
        2. 之前已经做了什么（来自 assistant 前缀）
        3. 需要继续当前的结构化输出，而不是重新回答
        """
        messages = [
            {"role": "system", "content": self._last_system_prompt},
            {"role": "assistant", "content": accumulated_text},
        ]
        return messages

    def _run_search_loop(self, initial_response: str) -> str:
        """
        多轮搜索交互循环，严格对齐 Search-R1 的 run_llm_loop()。

        Search-R1 原始流程 (generation.py run_llm_loop):
          for step in range(max_turns):
            1. generate_sequences() → 模型生成 response
            2. _postprocess_responses() → 截断到 </search> 或 </answer>
            3. execute_predictions() → 判断 action:
               - 'search' → 调搜索 API → next_obs = <information>...</information>
               - 'answer' → done=True → 退出
               - None → next_obs = 错误提示
            4. _update_rolling_state() → [input + response + next_obs] 拼成新 input
          # final LLM rollout (do_search=False): 让模型给出 <answer>

        此处的等效实现：
        - 使用 Chat API + stop=["</search>"] 替代 vLLM generate_sequences
        - 使用 assistant prefix continuation 模式（system + assistant前缀）模拟 token 级续写
        - 使用字符串拼接替代 tensor 拼接
        - 关键：循环结束后有 final rollout（不带 stop），让模型给出 <answer>
        """
        max_turns = self._task_config.max_search_turns
        accumulated_text = initial_response  # 完整的多轮轨迹文本

        # ---- 对应 Search-R1 的 _postprocess_responses: 截断并判断 action ----
        action, accumulated_text = self._classify_action(accumulated_text)
        self._logger.info(
            f"搜索循环初始: action={action}, 响应长度={len(accumulated_text)}"
        )

        # 如果首次就是 answer，直接返回
        if action == 'answer':
            return accumulated_text.strip()

        # ---- 主循环：对应 Search-R1 的 for step in range(max_turns) ----
        for turn in range(max_turns):
            if action != 'search':
                # 对应 Search-R1 execute_predictions 中 action==None 的情况：
                # 给一个错误提示让模型重试
                self._logger.warning(
                    f"搜索循环 Turn {turn + 1}: 无效 action (非 search/answer)，注入纠正提示"
                )
                accumulated_text += (
                    "\nMy previous action is invalid. "
                    "If I want to search, I should put the query between <search> and </search>. "
                    "If I want to give the final answer, I should put the answer between "
                    "<answer> and </answer>. Let me try again.\n"
                )
            else:
                # action == 'search': 提取 query，调搜索 API，拼接 <information>
                query = _extract_search_query(accumulated_text)
                if not query:
                    self._logger.warning(f"搜索循环 Turn {turn + 1}: search 标签内为空，结束")
                    break

                self._logger.info(f"搜索循环 Turn {turn + 1}: query='{query[:80]}'")
                search_start = time.time()
                search_result = self.search(query)
                search_elapsed = time.time() - search_start
                self._logger.info(
                    f"搜索循环 Turn {turn + 1}: 搜索耗时 {search_elapsed:.2f}s, "
                    f"结果长度 {len(search_result)} 字符 "
                    f"(上限 {self._task_config.max_obs_chars})"
                )

                # 对应 Search-R1 execute_predictions: next_obs 格式
                accumulated_text += f"\n\n<information>{search_result.strip()}</information>\n\n"

            # ---- 安全检查 ----
            if self._detect_repetition(accumulated_text):
                self._logger.warning(
                    f"搜索循环 Turn {turn + 1}: 检测到重复退化，提前结束"
                )
                break

            if len(accumulated_text) > self._max_context_chars:
                self._logger.warning(
                    f"搜索循环 Turn {turn + 1}: 上下文 {len(accumulated_text)} 字符 "
                    f"超过阈值 {self._max_context_chars}，提前结束"
                )
                break

            # ---- 调用 LLM 继续生成（对应 generate_sequences）----
            # 使用 assistant prefix continuation 模式，让模型续写结构化轨迹
            # system prompt 包含格式指令和问题，assistant 前缀是之前的轨迹
            messages = self._build_continuation_messages(accumulated_text)

            try:
                llm_start = time.time()
                continuation = self._llm_client.call_llm(
                    messages,
                    temperature=self._llm_temperature,
                    max_tokens=self._llm_max_tokens,
                    usage_context="search_r1.search_loop",
                    stop=["</search>"],
                )
                llm_elapsed = time.time() - llm_start

                # stop 参数截停时不包含 stop 序列本身，需要补上
                if "<search>" in continuation and "</search>" not in continuation:
                    continuation += "</search>"

                self._logger.info(
                    f"搜索循环 Turn {turn + 1}: LLM 耗时 {llm_elapsed:.2f}s, "
                    f"响应长度 {len(continuation)} 字符"
                )

                if hasattr(self._llm_client, 'clean_think_tags'):
                    continuation = self._llm_client.clean_think_tags(continuation)

                if self._detect_repetition(continuation, min_fragment_len=40, max_repeats=5):
                    self._logger.warning(
                        f"搜索循环 Turn {turn + 1}: 单轮响应重复退化，截断"
                    )
                    continuation = self._truncate_repetition(continuation)

                # 对应 _postprocess_responses: 截断并判断 action
                action, continuation = self._classify_action(continuation)
                accumulated_text += continuation

                self._logger.info(f"搜索循环 Turn {turn + 1}: action={action}")

                # 如果是 answer，立即结束
                if action == 'answer':
                    self._logger.info(f"搜索循环 Turn {turn + 1}: 检测到 <answer>，结束循环")
                    break

            except Exception as e:
                self._logger.error(f"搜索循环 Turn {turn + 1}: LLM 调用失败: {e}")
                break

        # ---- final LLM rollout: 对应 Search-R1 循环后的 final rollout ----
        # 如果循环结束后仍没有 <answer>，再做一次生成（不带 stop），让模型给出答案
        if "</answer>" not in accumulated_text:
            self._logger.info("搜索循环: 无 <answer>，执行 final rollout")
            # 同样使用 assistant prefix continuation，模型看到 system prompt + 轨迹，
            # 自然地在轨迹末尾续写 <answer>简洁答案</answer>
            messages = self._build_continuation_messages(accumulated_text)
            try:
                llm_start = time.time()
                # final rollout 不带 stop，让模型自由生成直到 <answer>
                final_response = self._llm_client.call_llm(
                    messages,
                    temperature=self._llm_temperature,
                    max_tokens=self._llm_max_tokens,
                    usage_context="search_r1.final_rollout",
                )
                llm_elapsed = time.time() - llm_start
                self._logger.info(
                    f"搜索循环 final rollout: LLM 耗时 {llm_elapsed:.2f}s, "
                    f"响应长度 {len(final_response)} 字符"
                )

                if hasattr(self._llm_client, 'clean_think_tags'):
                    final_response = self._llm_client.clean_think_tags(final_response)

                # 截断到 </answer>（如果有的话）
                _, final_response = self._classify_action(final_response)
                accumulated_text += final_response
            except Exception as e:
                self._logger.error(f"搜索循环 final rollout: LLM 调用失败: {e}")

        # 统计搜索轮次
        total_searches = len(re.findall(r"<search>", accumulated_text))
        has_answer = "</answer>" in accumulated_text
        self._logger.info(
            f"搜索循环结束: {total_searches} 次搜索, "
            f"{'有' if has_answer else '无'}答案, "
            f"轨迹长度 {len(accumulated_text)} 字符"
        )

        return accumulated_text.strip()

    @staticmethod
    def _truncate_repetition(text: str, min_fragment_len: int = 40) -> str:
        """截断重复退化的文本，保留有效内容。

        从文本末尾取一个片段，向前查找该片段第一次出现的位置，
        保留到第二次出现为止的内容。
        """
        if len(text) < min_fragment_len * 3:
            return text
        fragment = text[-min_fragment_len:]
        # 找到该片段首次出现的位置
        first_pos = text.find(fragment)
        if first_pos < 0:
            return text
        # 找到第二次出现的位置
        second_pos = text.find(fragment, first_pos + 1)
        if second_pos < 0:
            return text
        # 保留到第二次出现为止（包含一次完整的内容）
        truncated = text[:second_pos].rstrip()
        return truncated if truncated else text

    # ------------------------------------------------------------------
    # 搜索辅助
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        results = _batch_search(
            [query],
            search_url=self._task_config.search_url,
            topk=self._task_config.topk,
            timeout=self._task_config.timeout,
            max_obs_chars=self._task_config.max_obs_chars,
        )
        return results[0] if results else "[Search failed]"

    def batch_search(self, queries: list[str]) -> list[str]:
        return _batch_search(
            queries,
            search_url=self._task_config.search_url,
            topk=self._task_config.topk,
            timeout=self._task_config.timeout,
            max_obs_chars=self._task_config.max_obs_chars,
        )
