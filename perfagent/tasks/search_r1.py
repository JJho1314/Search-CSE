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

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None) -> SearchR1TaskConfig:
        if config_dict is None:
            return cls()
        return cls(
            search_url=config_dict.get("search_url", "http://127.0.0.1:8001/retrieve"),
            topk=config_dict.get("topk", 3),
            max_search_turns=config_dict.get("max_search_turns", 5),
            timeout=config_dict.get("timeout", 10),
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
                  timeout: int = 10) -> list[str]:
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
        formatted.append("\n".join(parts))
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
        # LLM client 由 PerfAgent 在运行时注入
        self._llm_client: LLMClient | None = None
        self._llm_temperature: float = 0.7
        self._llm_max_tokens: int | None = None

    def set_llm_client(self, llm_client: LLMClient | None,
                       temperature: float = 0.7,
                       max_tokens: int | None = None):
        """注入 LLM 客户端，供 extract_solution 中的搜索循环使用。

        由 PerfAgent 在 _init_run_context 或 _process_single_iteration 中调用。
        """
        self._llm_client = llm_client
        self._llm_temperature = temperature
        self._llm_max_tokens = max_tokens

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

        artifacts: dict[str, Any] = {
            "extracted_answer": answer,
            "ground_truth": instance.ground_truth,
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
            artifacts["failure_reason"] = (
                f"Answer '{answer}' does not match any of {instance.ground_truth}"
            )

        self._logger.info(
            f"实例 {instance.id}: EM={'通过' if em_passed else '未通过'}, "
            f"answer='{answer}', ground_truth={instance.ground_truth}"
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

        prompt_parts.append(
            "\n## Your Previous Reasoning Trajectory\n"
            f"{solution[:2048]}\n"
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

    # ------------------------------------------------------------------
    # 解提取 + 多轮搜索交互循环
    # ------------------------------------------------------------------

    def extract_solution(self, llm_response: str, current_solution: str) -> str:
        """从 LLM 响应中提取完整的搜索推理轨迹。

        方案 B 核心：在此方法中实现多轮搜索交互循环。

        流程对应 ray_trainer.py 中 LLMGenerationManager.run_llm_loop()：
        1. 检查 LLM 响应是否包含 <search>query</search>
        2. 如果有 → 调用搜索 API → 拼接 <information>结果</information>
        3. 将完整上下文（system + 已有对话 + 搜索结果）再次发给 LLM
        4. 重复直到出现 <answer> 或达到 max_search_turns
        5. 返回完整的多轮轨迹文本

        如果没有 LLM client（未注入），则回退到单轮模式。
        """
        if not llm_response or not llm_response.strip():
            self._logger.warning("LLM 响应为空，返回当前解")
            return current_solution

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

        # ---- 多轮搜索交互循环（对应 run_llm_loop）----
        return self._run_search_loop(llm_response)

    def _run_search_loop(self, initial_response: str) -> str:
        """
        多轮搜索交互循环。

        对应 ray_trainer.py 中 LLMGenerationManager.run_llm_loop()：
        - 每轮：检测 </search> → 提取 query → 调用搜索 API → 拼接 <information>
        - 将完整对话历史发给 LLM 继续生成
        - 直到 </answer> 出现或达到 max_search_turns

        与 ray_trainer 的差异：
        - ray_trainer 使用 vLLM 的 generate_sequences（token 级控制）
        - 此处使用 LLM API 的多轮对话（message 级控制）
        - 效果等价：LLM 看到完整的搜索历史并继续推理
        """
        max_turns = self._task_config.max_search_turns
        accumulated_text = initial_response  # 完整的多轮轨迹文本

        for turn in range(max_turns):
            # 检查是否已有答案
            if "</answer>" in accumulated_text:
                self._logger.info(f"搜索循环: 在第 {turn} 轮检测到 <answer>，结束循环")
                break

            # 提取搜索查询
            query = _extract_search_query(accumulated_text)
            if not query:
                self._logger.info(f"搜索循环: 第 {turn} 轮无 <search> 标签，结束循环")
                break

            # 截断到最后一个 </search>（移除 </search> 之后的内容）
            last_search_end = accumulated_text.rfind("</search>")
            if last_search_end >= 0:
                accumulated_text = accumulated_text[:last_search_end + len("</search>")]

            # 调用搜索 API
            self._logger.info(f"搜索循环 Turn {turn + 1}: query='{query[:80]}'")
            search_start = time.time()
            search_result = self.search(query)
            search_elapsed = time.time() - search_start
            self._logger.info(f"搜索循环 Turn {turn + 1}: 搜索耗时 {search_elapsed:.2f}s")

            # 拼接搜索结果（对应 ray_trainer 中的 next_obs 拼接）
            search_block = f"\n\n<information>{search_result.strip()}</information>\n\n"
            accumulated_text += search_block

            # 构建消息并继续调用 LLM
            messages = [
                {"role": "user", "content": accumulated_text},
            ]

            try:
                llm_start = time.time()
                continuation = self._llm_client.call_llm(
                    messages,
                    temperature=self._llm_temperature,
                    max_tokens=self._llm_max_tokens,
                    usage_context="search_r1.search_loop",
                )
                llm_elapsed = time.time() - llm_start
                self._logger.info(
                    f"搜索循环 Turn {turn + 1}: LLM 耗时 {llm_elapsed:.2f}s, "
                    f"响应长度 {len(continuation)} 字符"
                )

                # 清理 think tags（如果 LLM client 有此方法）
                if hasattr(self._llm_client, 'clean_think_tags'):
                    continuation = self._llm_client.clean_think_tags(continuation)

                accumulated_text += continuation

            except Exception as e:
                self._logger.error(f"搜索循环 Turn {turn + 1}: LLM 调用失败: {e}")
                break

        # 统计搜索轮次
        total_searches = len(re.findall(r"<search>", accumulated_text))
        has_answer = "</answer>" in accumulated_text
        self._logger.info(
            f"搜索循环结束: {total_searches} 次搜索, "
            f"{'有' if has_answer else '无'}答案, "
            f"轨迹长度 {len(accumulated_text)} 字符"
        )

        return accumulated_text.strip()

    # ------------------------------------------------------------------
    # 搜索辅助
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        results = _batch_search(
            [query],
            search_url=self._task_config.search_url,
            topk=self._task_config.topk,
            timeout=self._task_config.timeout,
        )
        return results[0] if results else "[Search failed]"

    def batch_search(self, queries: list[str]) -> list[str]:
        return _batch_search(
            queries,
            search_url=self._task_config.search_url,
            topk=self._task_config.topk,
            timeout=self._task_config.timeout,
        )
