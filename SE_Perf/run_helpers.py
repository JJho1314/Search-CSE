"""
perf_run 辅助函数

提供配置构建、内存检索等 perf_run 流水线的通用辅助功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.utils.global_memory_manager import GlobalMemoryManager
from operators.base import OperatorContext
from perf_config import SEPerfRunSEConfig, StepConfig
from run_models import GlobalMemoryContext

from perfagent.config import PerfAgentConfig, load_config


def extract_optimization_info(perf_config_path: str | None) -> tuple[str | None, str | None]:
    """从 PerfAgent 配置文件中提取优化目标和语言配置。

    支持新格式（task_config）和旧格式（optimization/language_cfg）。
    """
    if not perf_config_path:
        return None, None
    try:
        with open(perf_config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        # 新格式：从 task_config 读取
        task_cfg = config.get("task_config", {}) or {}
        opt_target = task_cfg.get("target")
        language = task_cfg.get("language")
        # 旧格式回退
        if not opt_target:
            opt_target = (config.get("optimization", {}) or {}).get("target")
        if not language:
            language = (config.get("language_cfg", {}) or {}).get("language")
        target_str = str(opt_target) if isinstance(opt_target, str) and opt_target.strip() else None
        language_str = str(language) if isinstance(language, str) and language.strip() else None
        return target_str, language_str
    except Exception:
        return None, None


def retrieve_global_memory(
    global_memory: GlobalMemoryManager | None,
    instance_name: str,
    problem_description: str,
    additional_requirements: str | None,
    local_memory_text: str | None,
) -> str | None:
    """纯内存函数：为单实例检索 Global Memory，返回文本或 None。

    不写入任何文件，不修改任何外部状态。
    语言和优化目标不再由 SE 层提供，Global Memory 应从 problem_description
    和 additional_requirements 推断上下文。
    """
    if not global_memory:
        return None

    try:
        context = GlobalMemoryContext(
            problem_description=problem_description,
            additional_requirements=additional_requirements or "",
            local_memory=local_memory_text or "",
        )
        queries = global_memory.generate_queries(context.to_dict())
        if not queries:
            return None
        mem_content = global_memory.retrieve(queries, context=context.to_dict())
        return mem_content if mem_content else None
    except Exception:
        return None


def build_perf_agent_config(
    base_config_path: str | None,
    se_model_config: dict[str, Any],
    max_iterations: int,
    output_dir: Path,
) -> PerfAgentConfig:
    """构建 PerfAgentConfig，合并 SE 层的模型和迭代覆盖。"""
    if base_config_path and Path(base_config_path).exists():
        config = load_config(Path(base_config_path))
    else:
        config = PerfAgentConfig()

    # 应用 SE 模型配置覆盖
    # 包含本地 vLLM 推理相关的键，使 SE config 能完整覆盖 base_config 的模型配置
    allowed_model_keys = {
        "name", "api_base", "api_key", "max_input_tokens", "max_output_tokens", "temperature",
        # 本地 vLLM 推理配置
        "local_model_path", "gpu_memory_utilization", "tensor_parallel_size",
        "max_model_len", "max_response_length",
        # LLM 客户端参数
        "request_timeout", "max_retries", "retry_delay",
    }
    for key, val in (se_model_config or {}).items():
        if key in allowed_model_keys and val is not None and str(val).strip():
            setattr(config.model, key, val)

    config.max_iterations = max_iterations

    # 设置输出目录
    config.logging.trajectory_dir = output_dir
    config.logging.log_dir = output_dir

    return config


def build_operator_context(se_cfg: SEPerfRunSEConfig, step: StepConfig) -> OperatorContext:
    """从 SE 配置和步骤配置构建 OperatorContext。

    算子层（Plan/Reflection/Crossover/Summarizer）的 LLM 调用使用 API 模式：
    - 优先使用 operator_models 配置（独立于 PerfAgent 的搜索推理模型）
    - 如果未配置 operator_models，回退到 model 配置的 API 部分
    """
    # prompt_config: 步骤级覆盖 > SE 全局配置；OperatorContext 仍使用 dict 以兼容算子
    if step.prompt_config is not None:
        pc = step.prompt_config.to_dict()
    else:
        pc = se_cfg.prompt_config.to_dict()

    # 算子层模型配置：优先用 operator_models（在 extras 里），否则回退到 model
    operator_model_cfg = se_cfg.extras.get("operator_models")
    if isinstance(operator_model_cfg, dict) and operator_model_cfg:
        model_cfg = operator_model_cfg
    else:
        model_cfg = se_cfg.model.to_dict()

    return OperatorContext(
        model_config=model_cfg,
        prompt_config=pc,
        selection_mode=step.selection_mode or "weighted",
        metric_higher_is_better=se_cfg.metric_higher_is_better,
    )
