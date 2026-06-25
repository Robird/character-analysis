#!/usr/bin/env python3
"""Phase 0 试点：人物元信息提取。

用 :class:`agent.Agent` 驱动真实 LLM，对单个人物提取结构化元信息——
身份与角色、重要关系人、活动领域、主要活动地点、标志性事件。这些正是
``docs/动作挖掘思路.md`` 中后续「多轴交叉枚举」所需的维度值。

每个 pass 的产出写入该人物在 ``output/`` 下子目录里的独立 JSON 文件，与
``gist.json`` 并列。用独立文件分别保存各 pass，便于并行执行与进度查询：
列出目录中的 ``phase*.json`` 即可得知已完成哪些分析。

用法::

    python extract_meta.py [character_dir]

``character_dir`` 为包含 ``gist.json`` 的人物目录，缺省为简·爱。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from agent import Agent
from agent import AgentResult
from analysis_shared import CharacterHeader
from analysis_shared import CharacterWorkspace
from analysis_shared import DEFAULT_CHARACTER_DIR
from analysis_shared import Domain
from analysis_shared import Location
from analysis_shared import Phase0MetaData
from analysis_shared import Phase0MetaRecord
from analysis_shared import Relationship
from analysis_shared import Role
from analysis_shared import SignatureEvent
from api import LLMClient
from character_profile import StoredProfile

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是一位资深的人物分析专家，兼具文学研究者与传记作家的素养。"
    "你的任务是为一个人物建立结构化的「元信息档案」，供后续更细粒度的动作挖掘使用。"
    "你熟悉古今中外的历史人物与虚构角色，能够基于公认的史料或原著文本，"
    "系统、全面、忠实地梳理一个人物的身份、关系、领域、地点与标志性事件。"
    "你只通过工具调用输出结构化结果，不在正文里堆砌罗列。"
)


# 工具名 → (schema, 工具说明)。也用作产出聚合的索引。
_OUTPUT_TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "output_role": (Role, "用此工具输出人物的一个身份/角色/职位。每个身份单独一次调用。"),
    "output_relationship": (
        Relationship,
        "用此工具输出一个与人物有重要交互的关系人。每个关系人单独一次调用。",
    ),
    "output_domain": (Domain, "用此工具输出人物涉及的一个活动领域。每个领域单独一次调用。"),
    "output_location": (Location, "用此工具输出一个人物主要活动地点/场所。每个地点单独一次调用。"),
    "output_signature_event": (
        SignatureEvent,
        "用此工具输出一个人物的标志性事件。每个事件单独一次调用。",
    ),
}

# 输出 JSON 里 data 各字段 ← 对应工具名，保持稳定的下游契约。
_FIELD_BY_TOOL: dict[str, str] = {
    "output_role": "roles",
    "output_relationship": "relationships",
    "output_domain": "domains",
    "output_location": "locations",
    "output_signature_event": "signature_events",
}


def build_meta_agent(client: LLMClient, *, max_iterations: int | None = None) -> Agent:
    """构造一个已注册全部元信息产出工具的 Agent。"""
    agent = Agent(_SYSTEM_PROMPT, client=client, max_iterations=max_iterations)
    for name, (schema, description) in _OUTPUT_TOOLS.items():
        agent.add_output_tool(name, schema, description)
    return agent


def _build_task_prompt(stored: StoredProfile, *, quick: bool = False) -> str:
    header = CharacterHeader.from_stored_profile(stored)
    aliases = "、".join(header.aliases) if header.aliases else "无"
    prompt = (
        f"请系统性地分析人物「{header.character}」"
        f"（母语原名：{header.native_name}；别名：{aliases}；"
        f"出处：{header.source}；分类：{header.classification_path}；简介：{header.gist}），"
        "提取以下五类结构化元信息，每一条都用一次对应的工具调用输出：\n"
        "1. 身份与角色（output_role）：此人一生中承担过的各种身份/角色/职位。\n"
        "2. 重要关系人（output_relationship）：与此人有重要交互的人物，标注关系类型与时期。\n"
        "3. 活动领域（output_domain）：此人涉及的活动领域。\n"
        "4. 主要活动地点（output_location）：此人生平的主要活动地点/场所。\n"
        "5. 标志性事件（output_signature_event）：此人最为人知的标志性事件。\n\n"
        "要求：\n"
        "- 力求全面：既覆盖最显著的，也主动挖掘容易被忽略的次要项"
        "（次要配角、过渡时期的身份、不起眼的场所等）。\n"
        "- 每条信息独立一次工具调用；同类信息可在同一轮内并行多次调用。\n"
        "- 忠实于公认史料或原著，不臆造与人物不符的内容。\n"
        "- 五类信息全部输出完毕后即可结束。"
    )
    if quick:
        prompt += (
            "\n- 当前为快速流程验证模式：每一类只需抽取 1-2 条最具代表性的样例，"
            "允许明显不完整，但字段必须合法、结构必须正确。"
        )
    return prompt


def extract_meta(
    stored: StoredProfile, client: LLMClient, *, quick: bool = False
) -> tuple[Phase0MetaData, AgentResult]:
    """对 *stored* 人物运行 Phase 0 元信息提取，返回 (data, 原始 AgentResult)。"""
    agent = build_meta_agent(client, max_iterations=16 if quick else 64)
    result = agent.run(_build_task_prompt(stored, quick=quick), temperature=0.4)
    data = Phase0MetaData(
        roles=list(result.by_tool("output_role")),
        relationships=list(result.by_tool("output_relationship")),
        domains=list(result.by_tool("output_domain")),
        locations=list(result.by_tool("output_location")),
        signature_events=list(result.by_tool("output_signature_event")),
    )
    return data, result


def _build_record(
    stored: StoredProfile,
    client: LLMClient,
    data: Phase0MetaData,
    result: AgentResult,
    *,
    quick: bool = False,
) -> Phase0MetaRecord:
    """组装写盘的完整记录：data 为产出主体，run 为便于复盘的运行元数据。"""
    header = CharacterHeader.from_stored_profile(stored)
    counts = {
        field: len(getattr(data, field))
        for field in _FIELD_BY_TOOL.values()
    }
    return Phase0MetaRecord.from_parts(
        header,
        data=data,
        run={
            "model": client.model,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": result.status.value,
            "status_detail": result.status_detail,
            "iterations": result.iterations,
            "max_iterations_reached": result.max_iterations_reached,
            "coverage_mode": "quick" if quick else "full",
            "counts": counts,
            "final_text": result.final_text,
            "intermediate_texts": list(result.intermediate_texts),
        },
    )


def run(character_dir: str | Path, *, quick: bool = False) -> Path:
    """对 *character_dir* 中的人物执行 Phase 0 提取并写出结果文件，返回文件路径。"""
    workspace = CharacterWorkspace.from_path(character_dir)
    stored = workspace.load_profile()
    logger.info(
        "开始分析人物：%s（模式=%s）",
        stored.profile.chinese_name,
        "quick" if quick else "full",
    )

    with LLMClient() as client:
        data, result = extract_meta(stored, client, quick=quick)
        record = _build_record(stored, client, data, result, quick=quick)

    output_path = workspace.phase0_meta_path
    workspace.write_json(output_path, record.model_dump(by_alias=True))
    counts = record.run["counts"]
    logger.info(
        "分析完成：状态=%s，轮数=%d，产出 %s",
        result.status.value,
        result.iterations,
        "，".join(f"{field}×{n}" for field, n in counts.items()),
    )
    logger.info("已写入：%s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHARACTER_DIR
    run(target_dir)
