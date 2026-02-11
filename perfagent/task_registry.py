"""
TaskRunner 注册表与工厂

提供 TaskRunner 类型的集中注册和按名称创建实例的能力。
SE_Perf 配置中通过 ``task_type`` 字段选择对应的 TaskRunner。

使用方式:
    from perfagent.task_registry import create_task_runner, register_task_runner

    # 注册（通常在 tasks 子包的 __init__.py 中完成）
    register_task_runner("effibench", "perfagent.tasks.effibench.EffiBenchRunner")

    # 创建
    runner = create_task_runner("effibench")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task_runner import BaseTaskRunner

# ---------------------------------------------------------------------------
# 内置注册表：task_type -> 完整限定类路径（延迟导入）
# ---------------------------------------------------------------------------
_TASK_RUNNERS: dict[str, str] = {
    "effibench": "perfagent.tasks.effibench.EffiBenchRunner",
    "livecodebench": "perfagent.tasks.livecodebench.LiveCodeBenchRunner",
    "aime": "perfagent.tasks.aime.AIMERunner",
    "search_r1": "perfagent.tasks.search_r1.SearchR1Runner",
}


def register_task_runner(task_type: str, class_path: str) -> None:
    """注册一个 TaskRunner 类型

    Args:
        task_type: 任务类型名称（如 "effibench"）
        class_path: TaskRunner 类的完整限定路径
                    （如 "perfagent.tasks.effibench.EffiBenchRunner"）
    """
    _TASK_RUNNERS[task_type] = class_path


def list_task_types() -> list[str]:
    """列出所有已注册的任务类型名称"""
    return list(_TASK_RUNNERS.keys())


def _import_class(class_path: str) -> type:
    """根据完整限定路径动态导入并返回类对象

    Args:
        class_path: 形如 "package.module.ClassName" 的字符串

    Returns:
        导入的类对象

    Raises:
        ImportError: 模块不存在
        AttributeError: 类在模块中不存在
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ImportError(f"无效的类路径: {class_path!r}（缺少模块路径）")

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_task_runner(task_type: str, **kwargs) -> "BaseTaskRunner":
    """根据任务类型名称创建 TaskRunner 实例

    Args:
        task_type: 已注册的任务类型名称（如 "effibench"）
        **kwargs: 传递给 TaskRunner 构造函数的额外参数

    Returns:
        对应的 BaseTaskRunner 子类实例

    Raises:
        KeyError: 未注册的任务类型
        ImportError: 模块导入失败
        AttributeError: 类在模块中不存在
        TypeError: 类实例化失败
    """
    if task_type not in _TASK_RUNNERS:
        available = ", ".join(sorted(_TASK_RUNNERS.keys()))
        raise KeyError(
            f"未知的任务类型: {task_type!r}。已注册的类型: {available}"
        )

    class_path = _TASK_RUNNERS[task_type]
    runner_cls = _import_class(class_path)

    # 基本校验：确保是 BaseTaskRunner 的子类
    from .task_runner import BaseTaskRunner

    if not issubclass(runner_cls, BaseTaskRunner):
        raise TypeError(
            f"{class_path} 不是 BaseTaskRunner 的子类"
        )

    return runner_cls(**kwargs)
