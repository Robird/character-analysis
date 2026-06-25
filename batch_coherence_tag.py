#!/usr/bin/env python3
"""虚构角色心理自洽性批量标注。

仅处理 ``classification[0] == 'fiction'`` 的虚构角色，判断其行为在多大程度上
可以从内在逻辑（性格、信念、处境）合理推导，而非仅仅服务于作者的剧情需要。

标注维度：

* **psychological_coherence** — 高 / 中 / 低
  - 高：行为与内在逻辑高度一致，读者能理解「他/她为什么会这样做」。
  - 中：大体自洽，但部分行为受剧情需求驱动而非角色内在逻辑。
  - 低：角色主要服务于戏剧功能，行为缺乏可信的心理动机。
* **coherence_note** — 若为中/低，用一句话说明主要的不自洽之处。

筛选价值：低自洽性的角色（无故的恶、无由的爱、性格突变、纯粹的功能性存在）
产出的 belief→action 轨迹质量低，不适合作为 Role-Play-Agent 的 SFT 训练数据。

用法::

    python batch_coherence_tag.py [--limit N] [--workers W] [--base output]
"""

from __future__ import annotations

import argparse
import logging
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from batch_ask import BatchAskConfig
from batch_ask import run_batch_ask
from character_gist import CharacterGist
from character_profile import StoredProfile

logger = logging.getLogger(__name__)


# ── 产出结构 ──────────────────────────────────────────────────────────────────


class CoherenceTag(BaseModel):
    """单个虚构角色的心理自洽性标注。"""

    psychological_coherence: Literal["高", "中", "低"] = Field(
        description=(
            "角色的行为在多大程度上可以从其性格、信念和处境中合理推导。\n"
            "高：行为与内在逻辑高度一致。即使在奇幻/科幻设定中，角色面对处境时的反应、"
            "选择与变化都让人感到「这个人就是会这样做」。其欲望、恐惧、原则与行动之间"
            "存在清晰的因果链。\n"
            "中：大体自洽，但存在部分行为更像作者安排而非角色自主——如关键情节处做出"
            "与既有性格不完全吻合的决定，或情感转折缺乏足够铺垫。\n"
            "低：角色主要作为戏剧工具存在。典型表现：无铺垫地爱上主角、无理由地作恶、"
            "性格为配合剧情突变、仅承担解说/送道具/牺牲以推动主角成长等功能。"
        ),
    )
    coherence_note: str = Field(
        default="",
        description=(
            "若 coherence 不为「高」，用一句话点明主要的不自洽之处（如「对主角的爱"
            "缺乏任何心理铺垫」「作恶动机从未被建立」「性格在第三幕突然反转以配合结局」）。"
            "若为「高」则留空。"
        ),
    )


# ── 提示词 ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是一位文学分析与角色心理学专家，兼具编剧顾问的经验。"
    "你的任务不是评判一个角色「写得好不好」，而是判断其行为在多大程度上具有"
    "**可信的心理逻辑**——即角色的行动是否可以从其性格、信念和处境中合理推导，"
    "而非仅仅服务于作者的情节需要或戏剧效果。\n\n"
    "关键区分：\n"
    "- 奇幻/科幻角色可以是高度自洽的（一条龙按龙的逻辑行事，只要内部一致即可）。\n"
    "- 现实主义背景下的角色也可以不自洽（如果一个普通人突然做出极端行为而毫无铺垫）。\n"
    "- 「自洽」不等于「善良」或「理性」——一个有严重缺陷、甚至自我毁灭倾向的角色，"
    "只要其行为链条可被理解，就是高自洽的。\n\n"
    "判断时请基于原著/史料中的实际描写，如信息不足以做出可靠判断则倾向于保守标注（中而非低）。"
    "你只通过 output_tag 工具输出结构化结果，不在正文中堆砌罗列。"
)

_OUTPUT_TOOL_DESC = (
    "用此工具输出角色的心理自洽性标注。整个任务只调用一次。"
)


def _build_task_prompt(raw: CharacterGist, stored: StoredProfile | None) -> str:
    """构造单个人物的自洽性标注 prompt。"""
    if stored is not None:
        profile = stored.profile
        name = profile.chinese_name
        native = profile.native_name
        source = profile.source
        gist = profile.gist
        classification = "/".join(stored.classification)
    else:
        name = raw.name
        native = raw.name
        source = ""
        gist = raw.gist
        classification = "/".join(raw.classification)

    return (
        "请判断以下虚构角色的心理自洽性，用 output_tag 工具输出（仅调用一次）：\n\n"
        f"- 角色名：{name}\n"
        f"- 母语名：{native}\n"
        f"- 出处：{source}\n"
        f"- 简介：{gist}\n"
        f"- 分类：{classification}\n\n"
        "判断指引：\n"
        "- 问自己：这个角色的关键行为是否能从其性格和处境中找到可信的「为什么」？\n"
        "- 警惕「因为剧情需要」的痕迹——突如其来的爱情、无铺垫的背叛、性格突变等。\n"
        "- 如果简介信息量不足以深入判断，倾向于标「中」而非「低」。\n"
        "- 若标「中」或「低」，请在 coherence_note 中用一句话指出问题所在。\n"
    )


# ── 筛选：仅虚构角色 ──────────────────────────────────────────────────────────


def _fiction_only(classification: tuple[str, ...]) -> bool:
    """仅保留 classification 顶层为 'fiction' 的角色。"""
    return len(classification) > 0 and classification[0] == "fiction"


# ── 任务配置 ──────────────────────────────────────────────────────────────────

CONFIG = BatchAskConfig(
    job_name="batch-coherence-tag",
    system_prompt=_SYSTEM_PROMPT,
    output_tool_name="output_tag",
    output_schema=CoherenceTag,
    output_tool_desc=_OUTPUT_TOOL_DESC,
    build_task_prompt=_build_task_prompt,
    max_iterations=3,
    temperature=0.0,
    query_status=False,
    classification_filter=_fiction_only,
    # model="deepseek-v4-flash",  # 若需降成本可切换
)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量标注虚构角色的心理自洽性（仅处理 fiction 分类）。"
    )
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个虚构角色。")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）。")
    parser.add_argument("--base", default="output", help="输出根目录（默认 output）。")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _parse_args()
    run_batch_ask(CONFIG, base=args.base, limit=args.limit, workers=args.workers)
