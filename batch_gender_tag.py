#!/usr/bin/env python3
"""性别+能动性+结构密度 批量标注。

对全部角色做一次快速三维标注，产出 ``batch-gender-tag.json`` 写入各角色目录。

三个维度：

* **gender** — 男 / 女 / 非二元 / 不确定
* **agency_level** — 高（主动塑造命运）/ 中（有主动性但受制约）/ 低（被动承受）
* **structural_density** — 丰富（大量内心与决策细节）/ 适中（足够叙事细节）/ 稀疏（仅轮廓记载）

这三个维度直接支持「杰出女性」开发子集的筛选：
``gender=女 ∧ agency_level∈{高,中} ∧ structural_density∈{丰富,适中}``。

用法::

    python batch_gender_tag.py [--limit N] [--workers W] [--base output]

幂等可续跑——已存在 ``batch-gender-tag.json`` 的条目自动跳过。
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


class CharacterTag(BaseModel):
    """单个人物的快速三维标注。"""

    gender: Literal["男", "女", "非二元", "不确定"] = Field(
        description=(
            "人物的性别身份。根据姓名、简介、出处综合判断。"
            "非二元=原作中明确为非二元性别或性别流动；不确定=信息不足以做出合理判断。"
        ),
    )
    agency_level: Literal["高", "中", "低"] = Field(
        description=(
            "该角色在多大程度上是自身命运的主动塑造者。"
            "高=主要驱动者，主动发起行动改变处境（如武则天、贞德、斯嘉丽）；"
            "中=有主动行为但受制于环境或他人（如简·爱、安娜·卡列尼娜）；"
            "低=主要被动承受事件，极少主动发起行动（如功能性配角、受害者角色）。"
        ),
    )
    structural_density: Literal["丰富", "适中", "稀疏"] = Field(
        description=(
            "该角色在原著/史料中可供提取的内心世界与决策细节的丰富程度。"
            "丰富=大量内心独白、决策过程、信念变化的文本支撑（如第一人称主角、有传记的历史人物）；"
            "适中=有足够叙事细节但非全程深入心理描写（如第三人称重要配角）；"
            "稀疏=仅轮廓性记载或功能性背景角色。"
        ),
    )


# ── 提示词 ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是一位角色数据标注专家，精通世界各国的历史人物与虚构角色。"
    "你的任务是根据角色的姓名与简介，快速判断其性别、能动性水平和生平可提取的结构密度。"
    "判断时以公认史料或原著为准，不臆造；若信息不足以判断则标注为「不确定」或「稀疏」。"
    "你只通过 output_tag 工具输出结构化结果，不在正文中堆砌罗列。"
)

_OUTPUT_TOOL_DESC = (
    "用此工具输出角色的性别、能动性水平和结构密度标注。整个任务只调用一次。"
)


def _build_task_prompt(raw: CharacterGist, stored: StoredProfile | None) -> str:
    """构造单个人物的标注 prompt。

    优先使用已清洗的 gist.json 数据（更准确），不存在时回退原始条目。
    """
    if stored is not None:
        profile = stored.profile
        name = profile.chinese_name
        native = profile.native_name
        source = profile.source
        gist = profile.gist
        classification = "/".join(stored.classification)
        aliases = "、".join(profile.aliases) if profile.aliases else "无"
    else:
        name = raw.name
        native = raw.name
        source = ""
        gist = raw.gist
        classification = "/".join(raw.classification)
        aliases = "无"

    return (
        "请根据以下信息判断角色属性，用 output_tag 工具输出（仅调用一次）：\n\n"
        f"- 角色名：{name}\n"
        f"- 母语名：{native}\n"
        f"- 别名：{aliases}\n"
        f"- 出处：{source}\n"
        f"- 简介：{gist}\n"
        f"- 分类路径：{classification}\n\n"
        "字段判断指引：\n"
        "- gender：根据姓名、代词、社会身份综合判断。\n"
        "- agency_level：从简介中判断此人多大程度上「自己推动」了生平走向。\n"
        "- structural_density：判断此人是否有足够丰富的原著/史料可供深挖内心与决策过程。\n"
    )


# ── 任务配置 ──────────────────────────────────────────────────────────────────

CONFIG = BatchAskConfig(
    job_name="batch-gender-tag",
    system_prompt=_SYSTEM_PROMPT,
    output_tool_name="output_tag",
    output_schema=CharacterTag,
    output_tool_desc=_OUTPUT_TOOL_DESC,
    build_task_prompt=_build_task_prompt,
    max_iterations=9,  # 给足自愈余地：首轮产出 + 可能的校验重修 + 收尾
    temperature=0.0,  # 分类任务，零温保证一致性
    query_status=False,  # 简单标注无需状态汇报，省一次请求
    # model="deepseek-v4-flash",  # 可切换为便宜模型，降低 5-10× 成本
)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量标注角色性别、能动性与结构密度。"
    )
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个角色。")
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
