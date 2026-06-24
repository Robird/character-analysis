#!/usr/bin/env python3
"""可复用的批量提问框架。

模式：遍历角色名录 → 对每个角色构造 prompt → LLM Agent 产出结构化结果 → 落盘。
换任务只需换：system prompt、产物 schema、prompt 构造器、输出文件名。

用法::

    from batch_ask import BatchAskConfig, run_batch_ask
    from pydantic import BaseModel

    class MyTag(BaseModel):
        field_a: str
        field_b: str

    config = BatchAskConfig(
        job_name="my-tags",
        system_prompt="你是……",
        output_tool_name="output_tag",
        output_schema=MyTag,
        output_tool_desc="用此工具输出……",
        build_task_prompt=lambda raw, stored: f"请标注 {raw.name}……",
    )
    run_batch_ask(config, limit=100, workers=8)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable

from pydantic import BaseModel

from agent import Agent
from api import LLMClient
from character_gist import CharacterGist
from character_profile import GIST_FILENAME
from character_profile import StoredProfile
from character_profile import load_profile
from character_profile import safe_dir_name
from character_reader import iter_characters

logger = logging.getLogger(__name__)


@dataclass
class BatchAskConfig:
    """批量提问任务的配置。

    Attributes:
        job_name: 任务名，同时用作输出文件名（``{job_name}.json``）。
        system_prompt: Agent 的 system 提示词。
        output_tool_name: 产出工具的函数名。
        output_schema: 产出结构的 pydantic 模型。
        output_tool_desc: 产出工具的说明文字。
        build_task_prompt: 构造 user prompt 的可调用对象。
            入参为 ``(raw: CharacterGist, stored: StoredProfile | None)``。
        max_iterations: 最大工具调用轮数；简单分类 1-2 轮足够。
        temperature: 采样温度；分类任务建议 0.0-0.2。
        query_status: 是否在循环后追加状态汇报轮次；简单标注可关闭以节省一次请求。
        model: LLM 模型名；None 使用 LLMClient 默认值。
    """

    job_name: str
    system_prompt: str
    output_tool_name: str
    output_schema: type[BaseModel]
    output_tool_desc: str
    build_task_prompt: Callable[[CharacterGist, StoredProfile | None], str]
    max_iterations: int = 4
    temperature: float = 0.1
    query_status: bool = False
    model: str | None = None


def _target_dir(base: Path, raw: CharacterGist) -> Path:
    """计算原始条目对应的角色目录（与 :func:`make_dir.target_dir` 一致）。"""
    segments = [safe_dir_name(seg) for seg in raw.classification]
    return base.joinpath(*segments, safe_dir_name(raw.name))


def _build_agent(config: BatchAskConfig, client: LLMClient) -> Agent:
    """按配置构造一个已注册产出工具的 Agent。"""
    agent = Agent(config.system_prompt, client=client, max_iterations=config.max_iterations)
    agent.add_output_tool(
        config.output_tool_name,
        config.output_schema,
        config.output_tool_desc,
    )
    return agent


def _try_load_stored(char_dir: Path) -> StoredProfile | None:
    """尝试加载 gist.json；不存在或损坏时返回 None。"""
    gist_path = char_dir / GIST_FILENAME
    if not gist_path.exists():
        return None
    try:
        return load_profile(char_dir)
    except Exception:
        logger.debug("加载 gist.json 失败: %s", char_dir, exc_info=True)
        return None


def _build_envelope(
    raw: CharacterGist,
    stored: StoredProfile | None,
    config: BatchAskConfig,
    result_value: BaseModel,
    model: str,
    status: str,
    iterations: int,
    max_iterations_reached: bool = False,
) -> dict[str, Any]:
    """构造与 phase 文件一致的信封结构。"""
    if stored:
        profile = stored.profile
        header = {
            "character": profile.chinese_name,
            "native_name": profile.native_name,
            "aliases": list(profile.aliases),
            "source": profile.source,
            "gist": profile.gist,
            "classification": list(stored.classification),
        }
    else:
        header = {
            "character": raw.name,
            "native_name": raw.name,
            "aliases": [],
            "source": "",
            "gist": raw.gist,
            "classification": list(raw.classification),
        }

    return {
        **header,
        "pass": config.job_name,
        "data": result_value.model_dump(),
        "run": {
            "model": model,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": status,
            "iterations": iterations,
            "max_iterations_reached": max_iterations_reached,
        },
    }


def _try_parse_final_text(
    final_text: str, config: BatchAskConfig
) -> BaseModel | None:
    """尝试从模型收尾正文中提取 JSON 并校验为产出结构。

    部分模型偶发将合法 JSON 写在正文中（如 Markdown 代码块）而不调用工具。
    此 fallback 可挽救这类调用，避免因工具调用合规性问题而丢失整条数据。
    """
    if not final_text:
        return None
    text = final_text.strip()
    # 去掉可能的 Markdown 代码块包裹
    if text.startswith("```"):
        # 去掉开头的 ```json 或 ```
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return config.output_schema.model_validate(data)
    except Exception:
        return None


def _process_one(
    raw: CharacterGist,
    base: Path,
    config: BatchAskConfig,
    client: LLMClient,
) -> Path | None:
    """处理单个人物；返回输出路径，失败返回 None 并记日志。"""
    char_dir = _target_dir(base, raw)
    output_path = char_dir / f"{config.job_name}.json"

    if output_path.exists():
        return output_path  # 幂等：已完成则跳过

    stored = _try_load_stored(char_dir)
    prompt = config.build_task_prompt(raw, stored)

    agent = _build_agent(config, client)
    result = agent.run(
        prompt,
        temperature=config.temperature,
        query_status=config.query_status,
    )

    values = result.by_tool(config.output_tool_name)
    if not values:
        # 兜底：部分模型偶发把合法 JSON 写入正文而不调用工具，
        # 尝试从 final_text 中解析以挽救本次调用。
        tag = _try_parse_final_text(result.final_text, config)
        if tag is None:
            raise RuntimeError(
                f"未产出 {config.output_tool_name}：{raw.name!r} "
                f"（final_text={result.final_text[:120]!r}）"
            )
    else:
        tag = values[-1]  # 取最后一条（若有自我修正）
        assert isinstance(tag, config.output_schema)

    record = _build_envelope(
        raw,
        stored,
        config,
        result_value=tag,
        model=client.model,
        status=result.status.value,
        iterations=result.iterations,
        max_iterations_reached=result.max_iterations_reached,
    )

    char_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def run_batch_ask(
    config: BatchAskConfig,
    *,
    base: str | Path = "output",
    limit: int | None = None,
    workers: int = 8,
) -> None:
    """执行一次批量提问任务。

    遍历全部角色，对尚未产出 ``{job_name}.json`` 的角色依次调用 LLM 标注，
    结果写入角色目录。单条失败被隔离，不影响其余条目。

    Args:
        config: 任务配置。
        base: 输出根目录。
        limit: 仅处理前 N 个角色；None 表示全部。
        workers: 并发线程数。
    """
    base = Path(base)
    raws = list(iter_characters())
    if limit is not None:
        raws = raws[:limit]

    pending = [
        raw for raw in raws
        if not (_target_dir(base, raw) / f"{config.job_name}.json").exists()
    ]

    logger.info(
        "任务 %s：名录 %d 条，已完成 %d 条，待处理 %d 条，并发 %d",
        config.job_name, len(raws), len(raws) - len(pending), len(pending), workers,
    )

    if not pending:
        logger.info("全部完成，无需处理。")
        return

    done = 0
    failed = 0
    client_kwargs: dict[str, Any] = {}
    if config.model:
        client_kwargs["model"] = config.model
    with LLMClient(**client_kwargs) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, raw, base, config, client): raw
                for raw in pending
            }
            for future in as_completed(futures):
                raw = futures[future]
                try:
                    future.result()
                    done += 1
                except Exception:
                    failed += 1
                    logger.warning("标注失败：%s", raw.name, exc_info=True)
                processed = done + failed
                if processed % 20 == 0 or processed == len(pending):
                    logger.info(
                        "进度 %d/%d（成功 %d，失败 %d）",
                        processed, len(pending), done, failed,
                    )

    logger.info("任务 %s 完成：成功 %d，失败 %d", config.job_name, done, failed)
