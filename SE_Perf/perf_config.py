from dataclasses import dataclass, field
from typing import Any, Literal



from core.global_memory.utils.config import (
    ChromaBackendConfig,
    GlobalMemoryConfig,
    MemoryConfig,
    OpenAIEmbeddingConfig,
)

# 轨迹选择模式
SelectionMode = Literal["weighted", "random"]

@dataclass
class PerfRunCLIConfig:
    config: str = "SE/configs/se_configs/dpsk.yaml"
    mode: str = "execute"


@dataclass
class ModelConfig:
    name: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.name is not None:
            out["name"] = self.name
        if self.api_base is not None:
            out["api_base"] = self.api_base
        if self.api_key is not None:
            out["api_key"] = self.api_key
        if self.max_input_tokens is not None:
            out["max_input_tokens"] = self.max_input_tokens
        if self.max_output_tokens is not None:
            out["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            out["temperature"] = self.temperature
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ModelConfig":
        known_keys = {"name", "api_base", "api_key", "max_input_tokens", "max_output_tokens", "temperature"}
        extras = {k: v for k, v in (d or {}).items() if k not in known_keys}
        return ModelConfig(
            name=(d or {}).get("name"),
            api_base=(d or {}).get("api_base"),
            api_key=(d or {}).get("api_key"),
            max_input_tokens=(d or {}).get("max_input_tokens"),
            max_output_tokens=(d or {}).get("max_output_tokens"),
            temperature=(d or {}).get("temperature"),
            extras=extras,
        )


@dataclass
class InstancesConfig:
    instances_dir: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {"instances_dir": self.instances_dir}
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "InstancesConfig":
        extras = {k: v for k, v in (d or {}).items() if k != "instances_dir"}
        return InstancesConfig(instances_dir=str((d or {}).get("instances_dir") or ""), extras=extras)


@dataclass
class LocalMemoryConfig:
    enabled: bool = True
    format_mode: Literal["full", "short"] = "short"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {"enabled": self.enabled, "format_mode": self.format_mode}
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LocalMemoryConfig":
        enabled_val = (d or {}).get("enabled")
        enabled = True if enabled_val is None else bool(enabled_val)
        fmt = str((d or {}).get("format_mode") or "short")
        extras = {k: v for k, v in (d or {}).items() if k not in {"enabled", "format_mode"}}
        return LocalMemoryConfig(enabled=enabled, format_mode=fmt, extras=extras)


@dataclass
class PromptConfig:
    """算子和轨迹摘要使用的提示词配置。

    将原先的 dict[str, Any] 结构化，为各算子和摘要器提供命名子配置。

    Attributes:
        plan: Plan 算子的提示词配置（system_prompt, user_prompt_template, fallback_patterns 等）。
        crossover: Crossover 算子的提示词配置（header, guidelines 等）。
        reflection_refine: ReflectionRefine 算子的提示词配置。
        alternative_strategy: AlternativeStrategy 算子的提示词配置。
        trajectory_analyzer: TrajectoryAnalyzer 算子的提示词配置。
        traj_pool_summary: TrajPoolSummary 算子的提示词配置。
        base_operator: 所有算子共享的提示词配置（enforce_tail, imports_block 等）。
        summarizer: 轨迹摘要器的提示词配置。
        extras: 其他未列举的提示词配置。
    """

    plan: dict[str, Any] = field(default_factory=dict)
    crossover: dict[str, Any] = field(default_factory=dict)
    reflection_refine: dict[str, Any] = field(default_factory=dict)
    alternative_strategy: dict[str, Any] = field(default_factory=dict)
    trajectory_analyzer: dict[str, Any] = field(default_factory=dict)
    traj_pool_summary: dict[str, Any] = field(default_factory=dict)
    base_operator: dict[str, Any] = field(default_factory=dict)
    summarizer: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.plan:
            out["plan"] = self.plan
        if self.crossover:
            out["crossover"] = self.crossover
        if self.reflection_refine:
            out["reflection_refine"] = self.reflection_refine
        if self.alternative_strategy:
            out["alternative_strategy"] = self.alternative_strategy
        if self.trajectory_analyzer:
            out["trajectory_analyzer"] = self.trajectory_analyzer
        if self.traj_pool_summary:
            out["traj_pool_summary"] = self.traj_pool_summary
        if self.base_operator:
            out["base_operator"] = self.base_operator
        if self.summarizer:
            out["summarizer"] = self.summarizer
        out.update(self.extras)
        return out

    def get(self, key: str, default: Any = None) -> Any:
        """兼容 dict 风格的 get 访问（便于算子代码平滑迁移）。"""
        if hasattr(self, key):
            val = getattr(self, key)
            if val:
                return val
        return self.extras.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """兼容 dict 风格的 setdefault（便于 trajectory_handler 等平滑迁移）。"""
        if hasattr(self, key) and key != "extras":
            val = getattr(self, key)
            if val:
                return val
            setattr(self, key, default)
            return default
        return self.extras.setdefault(key, default)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PromptConfig":
        if not isinstance(d, dict):
            return PromptConfig()
        known_keys = {
            "plan", "crossover", "reflection_refine", "alternative_strategy",
            "trajectory_analyzer", "traj_pool_summary", "base_operator", "summarizer",
        }
        extras = {k: v for k, v in d.items() if k not in known_keys}
        return PromptConfig(
            plan=d.get("plan") or {},
            crossover=d.get("crossover") or {},
            reflection_refine=d.get("reflection_refine") or {},
            alternative_strategy=d.get("alternative_strategy") or {},
            trajectory_analyzer=d.get("trajectory_analyzer") or {},
            traj_pool_summary=d.get("traj_pool_summary") or {},
            base_operator=d.get("base_operator") or {},
            summarizer=d.get("summarizer") or {},
            extras=extras,
        )


@dataclass
class StepConfig:
    """单个迭代步骤的配置。

    替代原先的 dict[str, Any]，提供类型安全的属性访问。
    """

    operator: str | None = None
    num: int | None = None
    trajectory_labels: list[str] = field(default_factory=list)
    trajectory_label: str | None = None
    source_trajectories: list[str] = field(default_factory=list)
    source_trajectory: str | None = None
    inputs: list[dict[str, str]] = field(default_factory=list)
    selection_mode: SelectionMode | None = None
    filter_strategy: dict[str, Any] | None = None
    prompt_config: PromptConfig | None = None
    perf_base_config: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def source_labels(self) -> list[str]:
        """统一获取源轨迹标签列表。"""
        if self.source_trajectories:
            return [str(x) for x in self.source_trajectories]
        if self.source_trajectory:
            return [str(self.source_trajectory)]
        return []

    @property
    def is_filter(self) -> bool:
        return str(self.operator) in ("filter", "filter_trajectories")

    @property
    def is_plan(self) -> bool:
        return self.operator == "plan"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.operator is not None:
            out["operator"] = self.operator
        if self.num is not None:
            out["num"] = self.num
        if self.trajectory_labels:
            out["trajectory_labels"] = self.trajectory_labels
        if self.trajectory_label is not None:
            out["trajectory_label"] = self.trajectory_label
        if self.source_trajectories:
            out["source_trajectories"] = self.source_trajectories
        if self.source_trajectory is not None:
            out["source_trajectory"] = self.source_trajectory
        if self.inputs:
            out["inputs"] = self.inputs
        if self.selection_mode is not None:
            out["selection_mode"] = self.selection_mode
        if self.filter_strategy is not None:
            out["filter_strategy"] = self.filter_strategy
        if self.prompt_config is not None:
            out["prompt_config"] = self.prompt_config.to_dict()
        if self.perf_base_config is not None:
            out["perf_base_config"] = self.perf_base_config
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StepConfig":
        if not isinstance(d, dict):
            return StepConfig()

        known_keys = {
            "operator", "num", "trajectory_labels", "trajectory_label",
            "source_trajectories", "source_trajectory", "inputs",
            "selection_mode", "filter_strategy", "strategy",
            "prompt_config", "perf_base_config",
        }

        num_val = d.get("num")
        try:
            num = int(num_val) if num_val is not None else None
        except (ValueError, TypeError):
            num = None

        trajectory_labels = list(d.get("trajectory_labels") or [])
        source_trajectories = list(d.get("source_trajectories") or [])
        inputs = list(d.get("inputs") or [])

        # filter_strategy: 优先使用 filter_strategy，回退到 strategy
        fs = d.get("filter_strategy")
        if not isinstance(fs, dict):
            fs = d.get("strategy")
        filter_strategy = fs if isinstance(fs, dict) else None

        extras = {k: v for k, v in d.items() if k not in known_keys}

        return StepConfig(
            operator=d.get("operator"),
            num=num,
            trajectory_labels=trajectory_labels,
            trajectory_label=d.get("trajectory_label"),
            source_trajectories=source_trajectories,
            source_trajectory=d.get("source_trajectory"),
            inputs=inputs,
            selection_mode=d.get("selection_mode"),
            filter_strategy=filter_strategy,
            prompt_config=PromptConfig.from_dict(d["prompt_config"]) if isinstance(d.get("prompt_config"), dict) else None,
            perf_base_config=d.get("perf_base_config"),
            extras=extras,
        )


@dataclass
class StrategyConfig:
    iterations: list[StepConfig] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"iterations": [s.to_dict() for s in self.iterations]}
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StrategyConfig":
        raw_iterations = list((d or {}).get("iterations") or [])
        iterations = [
            StepConfig.from_dict(item) if isinstance(item, dict) else StepConfig()
            for item in raw_iterations
        ]
        extras = {k: v for k, v in (d or {}).items() if k != "iterations"}
        return StrategyConfig(iterations=iterations, extras=extras)


@dataclass
class SEPerfRunSEConfig:
    base_config: str | None = None
    output_dir: str = ""
    task_type: str = "effibench"
    model: ModelConfig = field(default_factory=ModelConfig)
    instances: InstancesConfig = field(default_factory=InstancesConfig)
    max_iterations: int = 1
    # metric 比较方向：False=越小越好(默认，如运行时间)，True=越大越好(如 EM/准确率)
    # 若未显式设置，会在运行时从 base_config 中自动检测
    metric_higher_is_better: bool = False
    local_memory: LocalMemoryConfig | None = None
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    global_memory_bank: GlobalMemoryConfig | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base_config": self.base_config,
            "output_dir": self.output_dir,
            "task_type": self.task_type,
            "model": self.model.to_dict(),
            "instances": self.instances.to_dict(),
            "max_iterations": self.max_iterations,
            "metric_higher_is_better": self.metric_higher_is_better,
            "prompt_config": self.prompt_config.to_dict(),
            "strategy": self.strategy.to_dict(),
        }
        if self.local_memory is not None:
            out["local_memory"] = self.local_memory.to_dict()
        if self.global_memory_bank is not None:
            out["global_memory_bank"] = {
                "enabled": bool(self.global_memory_bank.enabled),
                "embedding_model": {
                    "provider": self.global_memory_bank.embedding_model.provider,
                    "api_base": self.global_memory_bank.embedding_model.api_base,
                    "api_key": self.global_memory_bank.embedding_model.api_key,
                    "model": self.global_memory_bank.embedding_model.model,
                    "request_timeout": self.global_memory_bank.embedding_model.request_timeout,
                },
                "memory": {
                    "backend": self.global_memory_bank.memory.backend,
                    "chroma": {
                        "collection_name": self.global_memory_bank.memory.chroma.collection_name,
                        "persist_path": self.global_memory_bank.memory.chroma.persist_path,
                    },
                },
            }
        out.update(self.extras)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SEPerfRunSEConfig":
        base_config = (d or {}).get("base_config")
        output_dir = str((d or {}).get("output_dir") or "")
        task_type = str((d or {}).get("task_type") or "effibench")
        model = ModelConfig.from_dict((d or {}).get("model") or {})
        instances = InstancesConfig.from_dict((d or {}).get("instances") or {})
        mi_val = (d or {}).get("max_iterations", 10)
        try:
            max_iterations = int(mi_val)
        except Exception:
            max_iterations = 10
        lm_dict = (d or {}).get("local_memory")
        local_memory = LocalMemoryConfig.from_dict(lm_dict) if isinstance(lm_dict, dict) else None
        prompt_config = PromptConfig.from_dict((d or {}).get("prompt_config") or {})
        strategy = StrategyConfig.from_dict((d or {}).get("strategy") or {})
        # metric_higher_is_better: 优先从 SE 配置读取，未设置时从 base_config 自动检测
        mhib_raw = (d or {}).get("metric_higher_is_better")
        if mhib_raw is not None:
            metric_higher_is_better = bool(mhib_raw)
        elif base_config:
            # 自动从 base_config 中检测
            metric_higher_is_better = False
            try:
                from pathlib import Path as _P
                bc_path = _P(base_config)
                if bc_path.exists():
                    import yaml as _yaml
                    with open(bc_path, encoding="utf-8") as _f:
                        _base_raw = _yaml.safe_load(_f) or {}
                    metric_higher_is_better = bool(_base_raw.get("metric_higher_is_better", False))
            except Exception:
                pass
        else:
            metric_higher_is_better = False

        known = {
            "base_config",
            "output_dir",
            "task_type",
            "model",
            "instances",
            "max_iterations",
            "metric_higher_is_better",
            "local_memory",
            "prompt_config",
            "strategy",
            "global_memory_bank",
        }
        extras = {k: v for k, v in (d or {}).items() if k not in known}
        gmb_dict = (d or {}).get("global_memory_bank") or None
        gmb = None
        if isinstance(gmb_dict, dict):
            enabled_val = gmb_dict.get("enabled")
            enabled = True if enabled_val is None else bool(enabled_val)
            em_raw = gmb_dict.get("embedding_model") or {}
            em_cfg = OpenAIEmbeddingConfig(
                provider=str(em_raw.get("provider") or "openai"),
                api_base=em_raw.get("api_base") or em_raw.get("base_url"),
                api_key=em_raw.get("api_key"),
                model=em_raw.get("model"),
                request_timeout=em_raw.get("request_timeout"),
            )
            m_raw = gmb_dict.get("memory") or {}
            c_raw = m_raw.get("chroma") or {}
            chroma_cfg = ChromaBackendConfig(
                collection_name=str(c_raw.get("collection_name") or "global_memory"),
                persist_path=c_raw.get("persist_path"),
            )
            mem_cfg = MemoryConfig(backend=str(m_raw.get("backend") or "chroma"), chroma=chroma_cfg)
            gmb = GlobalMemoryConfig(enabled=enabled, embedding_model=em_cfg, memory=mem_cfg)
        return SEPerfRunSEConfig(
            base_config=base_config,
            output_dir=output_dir,
            task_type=task_type,
            model=model,
            instances=instances,
            max_iterations=max_iterations,
            metric_higher_is_better=metric_higher_is_better,
            local_memory=local_memory,
            prompt_config=prompt_config,
            strategy=strategy,
            extras=extras,
            global_memory_bank=gmb,
        )
