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

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from agent import Agent
from agent import AgentResult
from api import LLMClient
from character_gist import CharacterGist

logger = logging.getLogger(__name__)

# 本试点固定分析的人物目录：简·爱。换人物时把它作为 CLI 参数传入即可。
_DEFAULT_CHARACTER_DIR = (
    "output/fiction/文学/英国文学/古典至19世纪/勃朗特姐妹/Jane Eyre（简·爱）"
)

# 产出文件名。带 phase 编号便于多 pass 并行时用 glob 查询进度。
_OUTPUT_FILENAME = "phase0-meta.json"

_SYSTEM_PROMPT = (
    "你是一位资深的人物分析专家，兼具文学研究者与传记作家的素养。"
    "你的任务是为一个人物建立结构化的「元信息档案」，供后续更细粒度的动作挖掘使用。"
    "你熟悉古今中外的历史人物与虚构角色，能够基于公认的史料或原著文本，"
    "系统、全面、忠实地梳理一个人物的身份、关系、领域、地点与标志性事件。"
    "你只通过工具调用输出结构化结果，不在正文里堆砌罗列。"
)


class Role(BaseModel):
    """人物一生中承担过的一个身份 / 角色 / 职位。"""

    name: str = Field(description="身份或角色名，如：孤女、家庭教师、继承人、妻子。")
    period: str = Field(default="", description="该身份对应的人生时期，如：童年、桑菲尔德时期。")
    note: str = Field(default="", description="一句话说明此身份的处境或内涵。")


class Relationship(BaseModel):
    """与人物有重要交互的一个关系人。"""

    name: str = Field(description="关系人姓名。")
    relation_type: str = Field(description="关系类型，如：恩人/对手/恋人/监护人/挚友。")
    period: str = Field(default="", description="关系发生或最重要的时期。")
    note: str = Field(default="", description="一句话说明此关系的性质或张力。")


class Domain(BaseModel):
    """人物涉及的一个活动领域。"""

    name: str = Field(description="活动领域名，如：教育、绘画、宗教、情感与婚姻、自立谋生。")
    note: str = Field(default="", description="一句话说明此人在该领域的具体涉入。")


class Location(BaseModel):
    """人物生平的一个主要活动地点 / 场所。"""

    name: str = Field(description="地点或场所名，如：盖茨黑德、劳渥德学校、桑菲尔德庄园。")
    period: str = Field(default="", description="在此地活动的人生时期。")
    note: str = Field(default="", description="一句话说明此地点对人物的意义。")


class SignatureEvent(BaseModel):
    """人物最为人知的一个标志性事件（后续分析中用于排除高频内容）。"""

    name: str = Field(description="标志性事件名。")
    period: str = Field(default="", description="事件发生的人生时期。")
    note: str = Field(default="", description="一句话概括此事件。")


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


def build_meta_agent(client: LLMClient) -> Agent:
    """构造一个已注册全部元信息产出工具的 Agent。"""
    agent = Agent(_SYSTEM_PROMPT, client=client)
    for name, (schema, description) in _OUTPUT_TOOLS.items():
        agent.add_output_tool(name, schema, description)
    return agent


def _build_task_prompt(character: CharacterGist) -> str:
    classification = "/".join(character.classification)
    return (
        f"请系统性地分析人物「{character.name}」"
        f"（分类：{classification}；一句话简介：{character.gist}），"
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


def extract_meta(character: CharacterGist, client: LLMClient) -> tuple[dict[str, Any], AgentResult]:
    """对 *character* 运行 Phase 0 元信息提取，返回 (data, 原始 AgentResult)。"""
    agent = build_meta_agent(client)
    result = agent.run(_build_task_prompt(character), temperature=0.4)

    data: dict[str, list[dict[str, Any]]] = {field: [] for field in _FIELD_BY_TOOL.values()}
    for tool_name, field in _FIELD_BY_TOOL.items():
        data[field] = [item.model_dump() for item in result.by_tool(tool_name)]
    return data, result


def _build_record(character: CharacterGist, client: LLMClient, data: dict[str, Any], result: AgentResult) -> dict[str, Any]:
    """组装写盘的完整记录：data 为产出主体，run 为便于复盘的运行元数据。"""
    return {
        "character": character.name,
        "gist": character.gist,
        "classification": list(character.classification),
        "pass": "phase0-meta",
        "data": data,
        "run": {
            "model": client.model,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": result.status.value,
            "status_detail": result.status_detail,
            "iterations": result.iterations,
            "max_iterations_reached": result.max_iterations_reached,
            "counts": {field: len(items) for field, items in data.items()},
            "final_text": result.final_text,
            "intermediate_texts": list(result.intermediate_texts),
        },
    }


def run(character_dir: str | Path) -> Path:
    """对 *character_dir* 中的人物执行 Phase 0 提取并写出结果文件，返回文件路径。"""
    character_dir = Path(character_dir)
    character = CharacterGist.LoadFromJson(character_dir)
    logger.info("开始分析人物：%s", character.name)

    with LLMClient() as client:
        data, result = extract_meta(character, client)
        record = _build_record(character, client, data, result)

    output_path = character_dir / _OUTPUT_FILENAME
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = record["run"]["counts"]
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
    target_dir = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_CHARACTER_DIR
    run(target_dir)
