"""
PerfAgent 配置系统

提供通用的 agent 参数配置。任务特定配置通过 task_config 字典传递给 TaskRunner。
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """模型配置"""

    name: str = "gpt-4"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_input_tokens: int = 4000
    max_output_tokens: int = 4000
    use_llm: bool = False
    # 本地 vLLM 推理（优先于 API 模式，完全对齐 Search-R1 的 token 级续写）
    # 设为模型权重目录路径即可启用，例如：
    #   local_model_path: /data/.../weight/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3
    local_model_path: str | None = None
    # vLLM 推理参数
    gpu_memory_utilization: float = 0.6
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    max_response_length: int = 1024  # 每轮最大生成 token 数
    # LLM 客户端增强参数
    request_timeout: float = 60.0
    max_retries: int = 10
    retry_delay: float = 2.0
    retry_backoff_factor: float = 2.0
    retry_jitter: float = 0.5
    log_inputs_outputs: bool = True
    log_sanitize: bool = True


@dataclass
class LoggingConfig:
    save_trajectory: bool = True
    trajectory_dir: Path = Path("./trajectories")
    log_dir: Path = Path("./logs")
    log_level: str = "INFO"


@dataclass
class PromptConfig:
    system_template: str = ""
    optimization_template: str = ""
    # 额外的系统提示内容：用于格式化到模板中的 {additional_requirements}
    additional_requirements: str | None = None
    include_all_history: bool = False
    local_memory: str | None = None
    global_memory: str | None = None


@dataclass
class PerfAgentConfig:
    """PerfAgent 配置类（通用，任务无关）

    任务特定的配置（如优化目标、运行时限制、语言等）统一放入
    task_config 字典，由 TaskRunner 负责解释。
    """

    # 迭代控制
    max_iterations: int = 10
    # 早停控制：连续未改进达到阈值后停止；0 表示不启用
    early_stop_no_improve: int = 0

    # 从 OptimizationConfig 提升的通用字段
    adopt_only_if_improved: bool = False
    # metric 比较方向：False=越小越好(默认)，True=越大越好
    metric_higher_is_better: bool = False

    # 组件配置
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)

    # 任务特定配置（不透明字典，由 TaskRunner 解释）
    task_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理：验证参数"""
        self._validate_config()

    def ensure_dirs(self) -> None:
        """按需创建日志和轨迹目录。

        不在 __post_init__ 中自动调用，避免仅实例化配置对象（如测试、序列化）
        时在项目根目录留下空目录。调用方在真正需要写入文件前调用此方法。
        """
        self.logging.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)

    def apply_cli_overrides(self, args: Any) -> None:
        """根据 CLI 参数覆盖配置。"""
        # 基础覆盖
        if getattr(args, "max_iterations", None) is not None:
            self.max_iterations = args.max_iterations
        if getattr(args, "model", None):
            self.model.name = args.model
        if getattr(args, "trajectory_dir", None):
            self.logging.trajectory_dir = args.trajectory_dir
        if getattr(args, "log_dir", None):
            self.logging.log_dir = args.log_dir
        if getattr(args, "log_level", None):
            self.logging.log_level = args.log_level

        # LLM 客户端配置
        if getattr(args, "llm_use", False):
            self.model.use_llm = True
        if getattr(args, "llm_api_base", None):
            self.model.api_base = args.llm_api_base
        if getattr(args, "llm_api_key", None):
            self.model.api_key = args.llm_api_key
        if getattr(args, "llm_model", None):
            self.model.name = args.llm_model
        if getattr(args, "llm_temp", None) is not None:
            self.model.temperature = args.llm_temp
        if getattr(args, "llm_max_output", None) is not None:
            self.model.max_output_tokens = args.llm_max_output
        # LLM 请求控制与日志开关
        if getattr(args, "llm_timeout", None) is not None:
            self.model.request_timeout = args.llm_timeout
        if getattr(args, "llm_max_retries", None) is not None:
            self.model.max_retries = args.llm_max_retries
        if getattr(args, "llm_retry_delay", None) is not None:
            self.model.retry_delay = args.llm_retry_delay
        if getattr(args, "llm_retry_backoff", None) is not None:
            self.model.retry_backoff_factor = args.llm_retry_backoff
        if getattr(args, "llm_retry_jitter", None) is not None:
            self.model.retry_jitter = args.llm_retry_jitter
        if getattr(args, "llm_log_io", False):
            self.model.log_inputs_outputs = True
        if getattr(args, "llm_log_sanitize", False):
            self.model.log_sanitize = True
        # 早停
        if getattr(args, "early_stop_no_improve", None) is not None:
            self.early_stop_no_improve = args.early_stop_no_improve

        # 任务特定覆盖写入 task_config
        if getattr(args, "language", None):
            self.task_config["language"] = args.language
        if getattr(args, "opt_target", None):
            self.task_config["target"] = args.opt_target
        if getattr(args, "include_other_metrics", None) is not None:
            self.task_config["include_other_metrics_in_summary"] = args.include_other_metrics

    def _validate_config(self):
        """验证配置参数"""
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not (0.0 <= self.model.temperature <= 1.0):
            raise ValueError("model.temperature must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（嵌套组件序列化，并处理 Path）"""
        data = asdict(self)
        # 处理嵌套 Path 字段
        if "logging" in data:
            if isinstance(data["logging"].get("log_dir"), Path):
                data["logging"]["log_dir"] = str(data["logging"]["log_dir"])
            if isinstance(data["logging"].get("trajectory_dir"), Path):
                data["logging"]["trajectory_dir"] = str(data["logging"]["trajectory_dir"])
        return data

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PerfAgentConfig":
        """从字典创建配置

        支持新格式（task_config）和旧格式（optimization/runtime/language_cfg）的向后兼容。
        """
        cfg = config_dict.copy() if config_dict else {}

        # model
        model_dict = cfg.get("model", {}) or {}
        model_cfg = ModelConfig(**model_dict)

        # logging
        logging_dict = cfg.get("logging", {}) or {}
        if "trajectory_dir" in logging_dict and isinstance(logging_dict["trajectory_dir"], str):
            logging_dict["trajectory_dir"] = Path(logging_dict["trajectory_dir"])
        if "log_dir" in logging_dict and isinstance(logging_dict["log_dir"], str):
            logging_dict["log_dir"] = Path(logging_dict["log_dir"])
        logging_cfg = LoggingConfig(**logging_dict)

        # prompts
        prompts_dict = cfg.get("prompts", {}) or {}
        prompts_cfg = PromptConfig(**prompts_dict)

        # task_config（新格式优先）
        task_config = cfg.get("task_config", {}) or {}

        # 向后兼容：从旧的嵌套键（optimization/runtime/language_cfg）迁移到 task_config
        if not task_config:
            old_opt = cfg.get("optimization", {}) or {}
            old_runtime = cfg.get("runtime", {}) or {}
            old_lang = cfg.get("language_cfg", {}) or {}
            merged: dict[str, Any] = {}
            for k, v in old_opt.items():
                if k != "adopt_only_if_improved":
                    merged[k] = v
            merged.update(old_runtime)
            if "language" in old_lang:
                merged["language"] = old_lang["language"]
            if merged:
                task_config = merged

        # adopt_only_if_improved: 顶层 > 旧 optimization section
        adopt_only_if_improved = cfg.get("adopt_only_if_improved", None)
        if adopt_only_if_improved is None:
            old_opt = cfg.get("optimization", {}) or {}
            adopt_only_if_improved = old_opt.get("adopt_only_if_improved", False)

        # 顶层标量字段
        max_iterations = cfg.get("max_iterations", 10)
        early_stop_no_improve = cfg.get("early_stop_no_improve", 0)
        metric_higher_is_better = cfg.get("metric_higher_is_better", False)

        return cls(
            max_iterations=max_iterations,
            model=model_cfg,
            logging=logging_cfg,
            prompts=prompts_cfg,
            early_stop_no_improve=early_stop_no_improve,
            adopt_only_if_improved=adopt_only_if_improved,
            metric_higher_is_better=metric_higher_is_better,
            task_config=task_config,
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PerfAgentConfig":
        """从 YAML 文件加载配置"""
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        return cls.from_dict(config_data)

    def to_yaml(self, config_path: Path) -> None:
        """保存配置到 YAML 文件"""
        config_data = self.to_dict()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)


def load_config(config_path: Path | None = None) -> PerfAgentConfig:
    """加载配置文件"""
    if config_path and Path(config_path).exists():
        return PerfAgentConfig.from_yaml(Path(config_path))
    env_config_path = os.getenv("PERFAGENT_CONFIG")
    if env_config_path:
        env_path = Path(env_config_path)
        if env_path.exists():
            return PerfAgentConfig.from_yaml(env_path)
    return PerfAgentConfig()
