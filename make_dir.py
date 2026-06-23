#!/usr/bin/env python3
"""数据清洗 Pass：把原始角色名录规范化为结构化档案并落盘。

早期枚举名录时尚无结构化输出能力，``name`` 字段格式不统一（译名 / 罗马音 / 别号
混入括号、以 ``/`` 分隔）。本脚本遍历 ``characters/`` 下的原始条目，用 LLM 将其
清洗为 :class:`~character_profile.CharacterProfile`——母语本名、中文名、别名数组、
出处、重写简介——写入 ``output/<分类...>/<安全目录名>/gist.json``。

目录各段均经 :func:`~character_profile.safe_dir_name` 处理，根治原始名称含 ``/`` 等
字符破坏目录结构的问题。

工程性质：
* 幂等可续跑——已存在 ``gist.json`` 的条目直接跳过，可随时中断重跑。
* 单条失败被隔离——某个人物清洗失败只记录告警，不影响其余条目。
* 并发——以线程池并行发起 LLM 请求（``httpx.Client`` 线程安全，可共享）。

用法::

    python make_dir.py [--limit N] [--workers W] [--base output]
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime
from itertools import islice
from pathlib import Path

from agent import Agent
from api import LLMClient
from character_gist import CharacterGist
from character_profile import CharacterProfile
from character_profile import GIST_FILENAME
from character_profile import StoredProfile
from character_profile import safe_dir_name
from character_profile import save_profile
from character_reader import iter_characters

logger = logging.getLogger(__name__)

_PROFILE_TOOL = "output_profile"

_CLEAN_SYSTEM_PROMPT = (
    "你是一位严谨的多语言角色数据规范化专家，精通世界各国语言、历史与文艺作品。"
    "你的任务是把一条格式不统一的角色名录记录，规范化为字段一致、可机读的结构化档案。"
    "你能准确判断一个人物 / 角色的母语原名应使用何种语言书写，"
    "并把混杂在一起的译名、别名、罗马音、外号、不同语言写法拆分到对应字段。"
    "你只通过 output_profile 工具输出唯一一条结构化结果，不在正文中堆砌罗列。"
)


def build_cleaner_agent(client: LLMClient) -> Agent:
    """构造一个已注册 output_profile 产出工具的清洗 Agent。"""
    agent = Agent(_CLEAN_SYSTEM_PROMPT, client=client, max_iterations=4)
    agent.add_output_tool(
        _PROFILE_TOOL,
        CharacterProfile,
        "用此工具输出规范化后的角色档案。整个任务只调用一次。",
    )
    return agent


def _build_clean_prompt(raw: CharacterGist) -> str:
    classification = "/".join(raw.classification)
    return (
        "请规范化下面这条角色名录记录，并用 output_profile 工具输出（仅调用一次）：\n\n"
        f"- 原始名称字段：{raw.name}\n"
        f"- 原始简介字段：{raw.gist}\n"
        f"- 分类路径（含真实/虚构、文化或作品等出处线索）：{classification}\n\n"
        "要点：\n"
        "- native_name：按母语 / 原作语言规则判断并用该语言书写本名。\n"
        "- chinese_name：中文最通用的单一规范名。\n"
        "- aliases：把原始名称括号里、以斜杠分隔的其它写法，以及你已知的其他译名 / "
        "别号 / 罗马音 / 外号，逐个拆成数组元素。\n"
        "- source：判断出处。虚构角色填作品名（尽量含作者与年代）；真实人物用自然语言"
        "概括其所属领域与核心身份。分类路径仅作判断线索，不要照抄到 source 里。\n"
        "- gist：重写为一句更准确、信息更完整的中文简介。\n"
        "- 一切以公认事实为准，不臆造与人物不符的内容。"
    )


class CleaningError(RuntimeError):
    """LLM 未能产出合法的规范化档案。"""


def clean_character(raw: CharacterGist, client: LLMClient) -> CharacterProfile:
    """对一条原始条目执行 LLM 清洗，返回规范化档案。

    Raises:
        CleaningError: 工具调用循环结束后仍未收集到任何合法 profile。
    """
    agent = build_cleaner_agent(client)
    result = agent.run(_build_clean_prompt(raw), temperature=0.2, query_status=False)
    profiles = result.by_tool(_PROFILE_TOOL)
    if not profiles:
        raise CleaningError(
            f"未产出 profile：{raw.name!r}（final_text={result.final_text[:80]!r}）"
        )
    # 正常只产出一条；若模型多调用一次，取最后一条（视作其自我修正后的最终版本）。
    profile = profiles[-1]
    assert isinstance(profile, CharacterProfile)
    return profile


def target_dir(base: Path, raw: CharacterGist) -> Path:
    """计算原始条目对应的安全人物目录（不依赖清洗结果，便于清洗前判重跳过）。"""
    segments = [safe_dir_name(seg) for seg in raw.classification]
    return base.joinpath(*segments, safe_dir_name(raw.name))


def _process_one(raw: CharacterGist, base: Path, client: LLMClient) -> Path:
    char_dir = target_dir(base, raw)
    profile = clean_character(raw, client)
    record = StoredProfile(
        profile=profile,
        classification=raw.classification,
        raw_name=raw.name,
        raw_gist=raw.gist,
        clean_model=client.model,
        clean_timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    return save_profile(char_dir, record)


def run(base: str | Path = "output", limit: int | None = None, workers: int = 8) -> None:
    """遍历原始名录，清洗未完成的条目并落盘。

    Args:
        base: 输出根目录。
        limit: 仅处理名录前 N 条（用于小规模试跑）；None 表示全部。
        workers: 并发线程数。
    """
    base = Path(base)
    source = iter_characters()
    raws = list(islice(source, limit)) if limit is not None else list(source)
    pending = [raw for raw in raws if not (target_dir(base, raw) / GIST_FILENAME).exists()]
    logger.info(
        "名录 %d 条，已完成 %d 条，待清洗 %d 条，并发 %d",
        len(raws), len(raws) - len(pending), len(pending), workers,
    )
    if not pending:
        return

    done = failed = 0
    with LLMClient() as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, raw, base, client): raw for raw in pending}
            for future in as_completed(futures):
                raw = futures[future]
                try:
                    future.result()
                    done += 1
                except Exception:  # noqa: BLE001 - 隔离单条失败，不影响整体批处理
                    failed += 1
                    logger.warning("清洗失败：%s", raw.name, exc_info=True)
                processed = done + failed
                if processed % 20 == 0 or processed == len(pending):
                    logger.info(
                        "进度 %d/%d（成功 %d，失败 %d）", processed, len(pending), done, failed
                    )
    logger.info("清洗完成：成功 %d，失败 %d", done, failed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 清洗原始角色名录为结构化档案。")
    parser.add_argument("--limit", type=int, default=None, help="仅处理名录前 N 条（试跑用）。")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）。")
    parser.add_argument("--base", default="output", help="输出根目录（默认 output）。")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # 第三方库的 INFO 日志（如每次 HTTP 请求）会淹没进度，降一档。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _parse_args()
    run(base=args.base, limit=args.limit, workers=args.workers)