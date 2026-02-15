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
    max_obs_chars: int = 4000  # 单次搜索结果最大字符数（对应 Search-R1 max_obs_length）

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None) -> SearchR1TaskConfig:
        if config_dict is None:
            return cls()
        return cls(
            search_url=config_dict.get("search_url", "http://127.0.0.1:8001/retrieve"),
            topk=config_dict.get("topk", 3),
            max_search_turns=config_dict.get("max_search_turns", 5),
            timeout=config_dict.get("timeout", 10),
            max_obs_chars=config_dict.get("max_obs_chars", 4000),
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
    """从文本中提取第一个 <search>query</search>（对齐 Search-R1 取第一个 action 的行为）。"""
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
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
            parts.append(f"Doc {idx + 1}(Title: {title}) {text}\n")
        formatted.append("".join(parts))
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
        # 上下文长度安全阈值（字符数）；由 set_llm_client 从 max_input_tokens 推算
        # 粗略估算：1 token ≈ 3.5 字符（英文为主），保留 output 余量
        self._max_context_chars: int = 90000  # 默认 ~25k tokens
        # 当前实例的 user prompt（system_prompt 内容），在 extract_solution 前由
        # _process_single_iteration 设置，供 _run_search_loop 构造续写消息使用
        self._current_user_prompt: str = ""
        # 历史错误答案追踪（跨 SE 迭代 + PerfAgent 内部迭代）
        self._blacklisted_answers: set[str] = set()
        self._failed_answers_set: set[str] = set()

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

        # 从 additional_requirements 中提取历史错误答案列表，
        # 供 _run_search_loop 的 final rollout 和 forced answer 消息使用
        if additional_requirements:
            self._extract_blacklisted_answers(additional_requirements)

        base_prompt = (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between "
            "<information> and </information>. "
            "You can search as many times as you want. "
            "If you find no further external knowledge needed, you can directly provide the answer "
            "inside <answer> and </answer>, without detailed illustrations. "
            "For example, <answer> Beijing </answer>.\n"
            "IMPORTANT: Your answer inside <answer> tags must be a short, direct entity name or value "
            "— NOT a full sentence. For example, write <answer> Giuseppe Cesari </answer> "
            "instead of <answer> Giuseppe Cesari lived longer than Nicos Poulantzas </answer>.\n\n"
            f"Question: {instance.question}\n"
        )

        if additional_requirements:
            base_prompt += f"\n{additional_requirements}\n"

        return base_prompt

    def _extract_blacklisted_answers(self, additional_requirements: str) -> None:
        """从 additional_requirements 中解析已知错误答案列表。

        解析 HISTORICAL FAILED ANSWERS 段中的 'WRONG answer: "xxx"' 模式，
        存储到 self._blacklisted_answers 供后续 prompt 使用。
        """
        if not hasattr(self, "_blacklisted_answers"):
            self._blacklisted_answers: set[str] = set()
        # 匹配 'WRONG answer: "xxx"' 模式
        for m in re.finditer(r'WRONG answer:\s*"([^"]+)"', additional_requirements):
            self._blacklisted_answers.add(m.group(1))

    def _get_all_blacklisted(self) -> set[str]:
        """获取所有黑名单中的错误答案（原始形式）。"""
        answers = getattr(self, "_blacklisted_answers", set())
        internal = getattr(self, "_failed_answers_set", set())
        return answers | internal

    def _is_blacklisted(self, answer: str) -> bool:
        """检查答案是否在黑名单中（使用归一化匹配，与 EM 评估一致）。

        同时匹配原始字符串和 _normalize_answer 归一化后的结果，
        避免 "Carrickfergus" vs "carrickfergus" 等大小写/标点差异导致漏判。
        """
        if not answer:
            return False
        normalized = _normalize_answer(answer)
        for wrong in self._get_all_blacklisted():
            if _normalize_answer(wrong) == normalized:
                return True
        return False

    def _build_blacklist_reminder(self) -> str:
        """构建简短的错误答案提醒文本，用于注入到搜索循环的关键位置。"""
        all_wrong = self._get_all_blacklisted()
        if not all_wrong:
            return ""
        wrong_list = ", ".join(f'"{a}"' for a in sorted(all_wrong))
        return (
            f"\n⚠️ IMPORTANT: These answers are ALL WRONG — do NOT use them: {wrong_list}. "
            "You must find a DIFFERENT answer.\n"
        )

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

        # 在错误情况下，提取并注入历史错误答案列表作为强化提醒
        if not passed and extracted_answer and extracted_answer != "None":
            # 追踪本 PerfAgent 内部迭代的错误答案
            if not hasattr(self, "_failed_answers_set"):
                self._failed_answers_set: set[str] = set()
            self._failed_answers_set.add(str(extracted_answer))

            if self._failed_answers_set:
                wrong_list = ", ".join(f'"{a}"' for a in sorted(self._failed_answers_set))
                prompt_parts.append(
                    f"\n⚠️ REMINDER: The following answers are ALL WRONG and must NOT be repeated: {wrong_list}"
                )

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
                "IMPORTANT: Your answer must be a short, direct entity name or value "
                "— NOT a full sentence or explanation. "
                "For example: <answer> Giuseppe Cesari </answer>\n"
            )
        else:
            prompt_parts.append(
                "\n## Instructions\n"
                "Your previous answer was correct. "
                "Try to provide the answer again, potentially with better reasoning.\n"
                "IMPORTANT: Your answer must be a short, direct entity name or value "
                "— NOT a full sentence or explanation.\n"
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

        注意：action 优先级为 search > answer（对齐 Search-R1 _postprocess_responses），
        即如果同时存在 </search> 和 </answer>，优先处理搜索。
        """
        if not llm_response or not llm_response.strip():
            self._logger.warning("LLM 响应为空，返回当前解")
            return current_solution

        # search 优先于 answer（对齐 Search-R1 _postprocess_responses 优先级）
        if "</search>" in llm_response:
            if self._llm_client is None:
                self._logger.warning(
                    "检测到 <search> 标签但无 LLM client，无法执行搜索循环，返回单轮结果"
                )
                return llm_response.strip()
            # ---- 多轮搜索交互循环（对应 run_llm_loop）----
            return self._run_search_loop(llm_response)

        # 没有 <search>，检查是否有 <answer>
        if "</answer>" in llm_response:
            return llm_response.strip()

        # 既无 search 也无 answer
        if "<answer>" in llm_response or "<think>" in llm_response:
            return llm_response.strip()

        self._logger.warning("无法从响应中提取有效轨迹，返回当前解")
        return current_solution

    def _run_search_loop(self, initial_response: str) -> str:
        """
        多轮搜索交互循环（对齐 Search-R1 LLMGenerationManager.run_llm_loop()）。

        关键对齐点：
        1. 每步截断到第一个 </search>（匹配 _postprocess_responses）
        2. search 优先于 answer（匹配 _postprocess_responses 优先级）
        3. 无效 action 给纠正提示并重试（匹配 execute_predictions）
        4. 搜索结果截断到 max_obs_chars（匹配 _process_next_obs）
        5. 循环结束后 final rollout 强制输出答案（匹配 final LLM rollout）

        与 ray_trainer 的差异：
        - ray_trainer 使用 vLLM 的 generate_sequences（token 级控制）
        - 此处使用 LLM API 的多轮对话（message 级控制）
        - 效果等价：LLM 看到完整的搜索历史并继续推理
        """
        max_turns = self._task_config.max_search_turns
        accumulated_text = ""       # 已处理并确认的完整轨迹
        latest_output = initial_response  # 待处理的最新 LLM 输出

        # 黑名单答案拦截计数器：同一搜索循环内最多拦截 N 次黑名单答案，
        # 超过后放行交给 final rollout 处理。
        # 目的：避免模型因搜索引擎始终返回相同误导信息而陷入
        # "回答 X → 被拦截 → 再回答 X → 再被拦截" 的死循环，浪费所有搜索轮次。
        MAX_BLACKLIST_REJECTIONS = 1
        blacklist_reject_count = 0

        # 总迭代次数安全上限（包含搜索 + 拦截 + 无效 action 等所有循环）
        # 即使 turn 不递增（如黑名单拦截），total_iterations 也递增，
        # 防止极端情况下 while 循环无限运行并生成超长轨迹
        MAX_TOTAL_ITERATIONS = max_turns * 3 + 2
        total_iterations = 0

        # 对齐 Search-R1 execute_predictions 中的 invalid action 反馈
        INVALID_ACTION_MSG = (
            "\nMy previous action is invalid. "
            "If I want to search, I should put the query between <search> and </search>. "
            "If I want to give the final answer, I should put the answer between "
            "<answer> and </answer>. Let me try again.\n"
        )

        # 使用 while 而非 for，手动控制 turn 递增：
        # - 真正的搜索操作（has_search）和无效 action 才消耗 turn
        # - 黑名单拦截（has_answer 被拒）不消耗 turn，把搜索机会留给实际探索
        turn = 0
        while turn < max_turns:
            # ---- 总迭代次数保护：防止极端情况下生成超长轨迹 ----
            total_iterations += 1
            if total_iterations > MAX_TOTAL_ITERATIONS:
                self._logger.warning(
                    f"搜索循环: 总迭代次数达到安全上限 {MAX_TOTAL_ITERATIONS}，"
                    f"强制结束（turn={turn}, max_turns={max_turns}）"
                )
                accumulated_text += latest_output
                break

            # ---- 上下文长度保护 ----
            if len(accumulated_text) + len(latest_output) > self._max_context_chars:
                self._logger.warning(
                    f"搜索循环 Turn {turn + 1}: 上下文超过安全阈值 "
                    f"{self._max_context_chars}，提前结束循环"
                )
                accumulated_text += latest_output
                break

            # 检测 latest_output 中的 action（search 优先于 answer，匹配 Search-R1）
            has_search = "</search>" in latest_output
            has_answer = "</answer>" in latest_output

            if has_search:
                # ---- 截断到第一个 </search>（匹配 Search-R1 _postprocess_responses）----
                first_search_end = latest_output.find("</search>") + len("</search>")
                truncated_output = latest_output[:first_search_end]
                accumulated_text += truncated_output

                # 提取查询
                match = re.search(r"<search>(.*?)</search>", truncated_output, re.DOTALL)
                if not match or not match.group(1).strip():
                    self._logger.warning(
                        f"搜索循环 Turn {turn + 1}: </search> 存在但无法提取有效查询"
                    )
                    break
                query = match.group(1).strip()

                # 调用搜索 API
                self._logger.info(f"搜索循环 Turn {turn + 1}: query='{query[:80]}'")
                search_start = time.time()
                search_result = self.search(query)
                search_elapsed = time.time() - search_start
                self._logger.info(
                    f"搜索循环 Turn {turn + 1}: 搜索耗时 {search_elapsed:.2f}s"
                )

                # ---- 截断搜索结果（匹配 Search-R1 _process_next_obs 的 max_obs_length）----
                max_obs = self._task_config.max_obs_chars
                if max_obs > 0 and len(search_result) > max_obs:
                    self._logger.warning(
                        f"搜索循环 Turn {turn + 1}: 搜索结果 {len(search_result)} 字符 "
                        f"超过 max_obs_chars={max_obs}，截断"
                    )
                    search_result = search_result[:max_obs] + "\n[... truncated]"

                # 拼接搜索结果（对齐 Search-R1 infer.py 的 curr_search_template）
                search_block = (
                    f"<information>{search_result.strip()}</information>\n\n"
                )
                accumulated_text += search_block

                # 方案 A: 每轮搜索结果返回后注入黑名单提醒
                blacklist_inline = self._build_blacklist_reminder()
                if blacklist_inline:
                    accumulated_text += blacklist_inline
                    self._logger.info(
                        f"搜索循环 Turn {turn + 1}: 已注入黑名单提醒"
                    )

                # 拼接后检查长度
                if len(accumulated_text) > self._max_context_chars:
                    self._logger.warning(
                        f"搜索循环 Turn {turn + 1}: 拼接搜索结果后上下文超长，提前结束"
                    )
                    break

                # 调用 LLM 继续生成
                continuation = self._call_llm_for_search_loop(
                    accumulated_text, turn + 1
                )
                if continuation is None:
                    break
                latest_output = continuation
                turn += 1  # 真正的搜索操作消耗 turn

            elif has_answer:
                # 方案 B: 拦截黑名单答案（限制拦截次数，避免死循环）
                candidate_answer = _extract_answer(latest_output)
                if (candidate_answer
                        and self._is_blacklisted(candidate_answer)
                        and blacklist_reject_count < MAX_BLACKLIST_REJECTIONS):
                    # 拦截次数未超限，拒绝该答案并要求模型重新搜索
                    blacklist_reject_count += 1
                    accumulated_text += latest_output
                    blacklist_reject_msg = (
                        f'\n⚠️ STOP: Your answer "{candidate_answer}" has been tried before '
                        f"and is CONFIRMED WRONG. You MUST NOT use this answer.\n"
                        f"The search results may be about a DIFFERENT person/entity with "
                        f"a similar name. There might be MULTIPLE films/entities with the "
                        f"same name — you need to find the RIGHT one.\n"
                        f"You MUST:\n"
                        f"1. Search using COMPLETELY DIFFERENT queries (try adding year, "
                        f"language, or other distinguishing details)\n"
                        f"2. Carefully verify that search results are about the CORRECT entity\n"
                        f"3. Provide a DIFFERENT answer\n"
                    )
                    accumulated_text += blacklist_reject_msg
                    self._logger.info(
                        f"搜索循环 Turn {turn + 1}: 拦截黑名单答案 "
                        f'"{candidate_answer}" ({blacklist_reject_count}/{MAX_BLACKLIST_REJECTIONS})，'
                        f"注入纠正消息并继续"
                    )

                    if len(accumulated_text) > self._max_context_chars:
                        self._logger.warning(
                            f"搜索循环 Turn {turn + 1}: 拦截后上下文超长，结束"
                        )
                        break

                    continuation = self._call_llm_for_search_loop(
                        accumulated_text, turn + 1
                    )
                    if continuation is None:
                        break
                    latest_output = continuation
                    # 拦截不递增 turn，不消耗搜索轮次配额
                    continue

                # 答案不在黑名单 或 拦截次数已达上限 → 结束循环
                accumulated_text += latest_output
                if candidate_answer and self._is_blacklisted(candidate_answer):
                    self._logger.info(
                        f"搜索循环: 黑名单拦截已达上限 {MAX_BLACKLIST_REJECTIONS} 次，"
                        f'放行答案 "{candidate_answer}"，交给 final rollout 处理'
                    )
                else:
                    self._logger.info(
                        f"搜索循环: 在第 {turn + 1} 轮检测到 <answer>，结束循环"
                    )
                break

            else:
                # ---- 无效 action（匹配 Search-R1 execute_predictions）----
                accumulated_text += latest_output
                accumulated_text += INVALID_ACTION_MSG
                self._logger.info(
                    f"搜索循环 Turn {turn + 1}: 无有效 <search> 或 <answer>，"
                    f"发送纠正提示"
                )

                if len(accumulated_text) > self._max_context_chars:
                    self._logger.warning(
                        f"搜索循环 Turn {turn + 1}: 拼接纠正提示后上下文超长，结束"
                    )
                    break

                continuation = self._call_llm_for_search_loop(
                    accumulated_text, turn + 1
                )
                if continuation is None:
                    break
                latest_output = continuation
                turn += 1

        if turn >= max_turns:
            # while 循环正常结束（max_turns 耗尽），追加最后的 latest_output
            accumulated_text += latest_output

        # ---- Final Rollout（匹配 Search-R1 run_llm_loop 的 final LLM rollout）----
        # 如果循环结束后仍无答案，或答案在黑名单中，强制再生成
        # 检查已有答案是否在黑名单中
        existing_answer = _extract_answer(accumulated_text)
        need_final_rollout = "</answer>" not in accumulated_text
        if not need_final_rollout and existing_answer and self._is_blacklisted(existing_answer):
            need_final_rollout = True
            self._logger.info(
                f"搜索循环: 已有答案 \"{existing_answer}\" 在黑名单中，触发 final rollout"
            )

        if need_final_rollout:
            if len(accumulated_text) <= self._max_context_chars:
                blacklist_reminder = self._build_blacklist_reminder()
                force_answer_msg = (
                    "\nYou have reached the maximum number of search attempts. "
                    "You must now provide your final answer based on all the "
                    "information you have gathered so far. "
                    "Put your answer inside <answer> and </answer> tags. "
                    "Your answer must be a short, direct entity name or value "
                    "— NOT a full sentence. "
                    "For example, <answer> Beijing </answer>.\n"
                    f"{blacklist_reminder}"
                )
                accumulated_text += force_answer_msg
                self._logger.info("搜索循环: 执行 final rollout，强制生成答案")

                # 最多尝试 2 次 final rollout（防止反复给出黑名单答案时死循环）
                for final_attempt in range(2):
                    continuation = self._call_llm_for_search_loop(
                        accumulated_text, max_turns + 1 + final_attempt
                    )
                    if not continuation:
                        break
                    accumulated_text += continuation

                    # 检查 final rollout 的答案是否仍在黑名单中
                    final_answer = _extract_answer(accumulated_text)
                    if final_answer and self._is_blacklisted(final_answer):
                        self._logger.warning(
                            f"搜索循环: final rollout 第 {final_attempt + 1} 次 "
                            f"答案 \"{final_answer}\" 仍在黑名单中"
                        )
                        if final_attempt < 1:
                            # 再给一次机会
                            accumulated_text += (
                                f'\n⚠️ Your answer "{final_answer}" is STILL WRONG '
                                f"(it was already tried). Give a DIFFERENT answer.\n"
                            )
                            continue
                    break  # 答案不在黑名单，或已用完重试次数

        # 统计搜索轮次
        total_searches = len(re.findall(r"<search>", accumulated_text))
        has_answer = "</answer>" in accumulated_text
        raw_len = len(accumulated_text)

        # ---- 清理黑名单注入消息，减少存储和后续处理的 token 开销 ----
        accumulated_text = self._strip_blacklist_messages_from_trajectory(
            accumulated_text
        )

        self._logger.info(
            f"搜索循环结束: {total_searches} 次搜索, "
            f"{'有' if has_answer else '无'}答案, "
            f"原始轨迹 {raw_len} 字符, 清理后 {len(accumulated_text)} 字符"
        )

        return accumulated_text.strip()

    @staticmethod
    def _strip_blacklist_messages_from_trajectory(text: str) -> str:
        """清理轨迹文本中的黑名单注入消息。

        这些消息在搜索循环中对 LLM 有用，但在存储和后续处理中
        是冗余噪声，会导致：
        1. .tra 文件过大
        2. 轨迹总结 LLM 输入超长
        3. 算子（reflection/crossover）prompt 膨胀

        保留轨迹结构（<search>、<information>、<think>、<answer>），
        仅去除 ⚠️ 开头的黑名单提醒行及其附属指示行。
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
                if stripped.startswith(
                    ("You MUST", "1.", "2.", "3.", "The search results")
                ):
                    continue
                skip_until_blank = False
            cleaned.append(line)
        return "\n".join(cleaned)

    def _call_llm_for_search_loop(self, assistant_prefix: str, turn: int) -> str | None:
        """搜索循环中调用 LLM 的统一入口，包含错误处理和日志。

        **关键对齐点（Search-R1 infer.py）**：
        Search-R1 的 infer.py 将 system_prompt+question 作为初始 prompt，
        然后把模型输出和搜索结果**直接拼接到同一个 prompt 字符串**上，
        让模型在已有文本基础上**续写**。

        为了用 OpenAI Chat API 复现这一行为，我们使用：
        - role=user: 原始的 system_prompt（包含 question）
        - role=assistant: 已有的搜索轨迹文本（accumulated_text）
        这样模型会认为 assistant_prefix 是自己之前的输出，从而续写。

        Args:
            assistant_prefix: 已有的搜索轨迹文本，作为 assistant 消息的 prefix
            turn: 当前轮次编号（用于日志）

        Returns:
            LLM 的续写文本，失败时返回 None
        """
        # 构造续写消息：user=原始prompt, assistant=已有轨迹
        # 对齐 Search-R1 infer.py 的 prompt 拼接方式
        if self._current_user_prompt:
            messages = [
                {"role": "user", "content": self._current_user_prompt},
                {"role": "assistant", "content": assistant_prefix},
            ]
        else:
            # fallback: 如果 user prompt 未设置，退化为旧行为
            self._logger.warning("搜索循环: _current_user_prompt 未设置，退化为单 user 消息模式")
            messages = [{"role": "user", "content": assistant_prefix}]
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
                f"搜索循环 Turn {turn}: LLM 耗时 {llm_elapsed:.2f}s, "
                f"响应长度 {len(continuation)} 字符"
            )

            # 注意：不清理 <think> tags！
            # Search-R1 infer.py 保留 <think> 内容在 prompt 中，
            # 模型需要看到自己之前的推理过程才能正确续写。

            return continuation

        except Exception as e:
            self._logger.error(f"搜索循环 Turn {turn}: LLM 调用失败: {e}")
            return None

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
