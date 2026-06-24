#!/usr/bin/env python3
"""Phase 1 试点：生平骨架（时间轴）提取。

用 :class:`agent.Agent` 驱动真实 LLM，将人物一生划分为主要阶段，每个阶段进一步
拆解为若干子时期，并为每个子时期标注：时间范围、核心处境、开端触发事件、结束
转折事件。这是后续 Phase 2 多轴交叉枚举所需的时间轴骨架。

若人物目录下已存在 ``phase0-meta.json``，其「标志性事件」列表会被提取出来作为
时间节点锚点一并提供给 LLM，以提升子时期划分的精度和完整性。

每个主要阶段（LifeStage）作为一次独立工具调用输出，其下的子时期（SubPeriod）
嵌套在同一对象内，与 Phase 0「一条记录一次调用」的扁平模式不同。这是因为阶段
与子时期之间存在强归属关系，拆成两个工具反而需要额外的关联字段，不如嵌套自然。

用法::

    python extract_timeline.py [character_dir]

``character_dir`` 为包含 ``gist.json`` 的人物目录，缺省为简·爱。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from agent import Agent
from agent import AgentResult
from analysis_shared import CharacterHeader
from analysis_shared import CharacterWorkspace
from analysis_shared import DEFAULT_CHARACTER_DIR
from analysis_shared import LifeStage
from analysis_shared import Phase1TimelineData
from analysis_shared import Phase1TimelineRecord
from analysis_shared import SubPeriod
from analysis_shared import load_signature_event_names
from api import LLMClient
from character_profile import StoredProfile

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是一位资深的传记研究者与人物分析专家。"
    "你的任务是将一个人物的一生拆解为结构化的时间轴：先划分主要阶段，再在每个阶段内"
    "进一步细分子时期，并为每个子时期标注核心处境与前后转折事件。"
    "你熟悉古今中外的历史人物与虚构角色，能基于公认史料或原著忠实还原生平轨迹。"
    "分析时请特别注意：不要只记录高光时刻，平淡的过渡期、低谷期、等待期同样要列出——"
    "这些往往是性格塑造和心理变化的关键阶段。"
    "你只通过工具调用输出结构化结果，每个主要阶段输出一次，按时间先后顺序。"
)

_TOOL_NAME = "output_life_stage"
_TOOL_DESC = (
    "用此工具输出人物的一个主要人生阶段（含内嵌的子时期列表）。"
    "每个主要阶段独立调用一次，按时间先后顺序依次输出。"
)


def build_timeline_agent(client: LLMClient, *, max_iterations: int | None = None) -> Agent:
    agent = Agent(_SYSTEM_PROMPT, client=client, max_iterations=max_iterations)
    agent.add_output_tool(_TOOL_NAME, LifeStage, _TOOL_DESC)
    return agent


def _build_task_prompt(
    stored: StoredProfile, signature_events: list[str], *, quick: bool = False
) -> str:
    header = CharacterHeader.from_stored_profile(stored)
    aliases = "、".join(header.aliases) if header.aliases else "无"

    lines = [
        f"请将人物「{header.character}」的一生系统性地拆解为结构化时间轴。",
        f"人物信息：母语原名={header.native_name}；别名={aliases}；"
        f"出处={header.source}；分类={header.classification_path}；简介={header.gist}。",
        "",
        "输出规则：",
        "1. 将一生划分为若干主要阶段（通常 4-8 个），每个阶段用一次 output_life_stage 工具调用输出。",
        "2. 每个主要阶段内进一步细分为子时期（sub_periods），按时间顺序嵌套在同一工具调用中。",
        "3. 对每个子时期填写：",
        "   - name：子时期名称",
        "   - time_range：时间范围",
        "   - core_situation：核心处境（一句话）",
        "   - opening_event：触发此子时期的事件（一句话）",
        "   - closing_event：结束此子时期的转折事件（一句话；末尾子时期无明确结束事件时留空）",
        "4. 覆盖原则：不要只关注高光时刻；平淡的过渡期、低谷期、等待期同样要列出。",
        "5. 按时间先后顺序依次输出各主要阶段；忠实于公认史料或原著，不臆造内容。",
    ]
    if quick:
        lines += [
            "6. 当前为快速流程验证模式：只需输出 1-2 个主要阶段，且每个阶段 1-2 个子时期即可。",
            "7. 允许明显不完整，但结构必须正确、时间顺序必须自洽。",
        ]

    if signature_events:
        lines += [
            "",
            "参考标志性事件（Phase 0 已提取，可作为子时期开端/结尾的时间节点锚点，请确保它们"
            "都能在某个子时期的 opening_event 或 closing_event 中体现）：",
        ]
        for ev in signature_events:
            lines.append(f"  · {ev}")

    return "\n".join(lines)


# ── 核心流程 ──────────────────────────────────────────────────────────────────


def extract_timeline(
    stored: StoredProfile, client: LLMClient, character_dir: Path, *, quick: bool = False
) -> tuple[Phase1TimelineData, AgentResult]:
    """对 *stored* 人物运行 Phase 1 时间轴提取，返回 (data, 原始 AgentResult)。"""
    signature_events = load_signature_event_names(character_dir)
    agent = build_timeline_agent(client, max_iterations=2 if quick else None)
    result = agent.run(
        _build_task_prompt(stored, signature_events, quick=quick),
        temperature=0.3,
    )
    data = Phase1TimelineData(life_stages=list(result.by_tool(_TOOL_NAME)))
    return data, result


def _build_record(
    stored: StoredProfile,
    client: LLMClient,
    data: Phase1TimelineData,
    result: AgentResult,
    *,
    quick: bool = False,
) -> Phase1TimelineRecord:
    header = CharacterHeader.from_stored_profile(stored)
    return Phase1TimelineRecord.from_parts(
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
            "counts": {
                "life_stages": len(data.life_stages),
                "sub_periods_total": data.total_sub_periods(),
            },
            "final_text": result.final_text,
            "intermediate_texts": list(result.intermediate_texts),
        },
    )


def run(character_dir: str | Path, *, quick: bool = False) -> Path:
    """对 *character_dir* 中的人物执行 Phase 1 提取并写出结果文件，返回文件路径。"""
    workspace = CharacterWorkspace.from_path(character_dir)
    stored = workspace.load_profile()
    logger.info(
        "开始提取时间轴：%s（模式=%s）",
        stored.profile.chinese_name,
        "quick" if quick else "full",
    )

    with LLMClient() as client:
        data, result = extract_timeline(stored, client, workspace.root, quick=quick)
        record = _build_record(stored, client, data, result, quick=quick)

    output_path = workspace.phase1_timeline_path
    workspace.write_json(output_path, record.model_dump(by_alias=True))
    counts = record.run["counts"]
    logger.info(
        "提取完成：状态=%s，轮数=%d，主要阶段×%d，子时期共%d个",
        result.status.value,
        result.iterations,
        counts["life_stages"],
        counts["sub_periods_total"],
    )
    logger.info("已写入：%s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHARACTER_DIR
    run(target_dir)
