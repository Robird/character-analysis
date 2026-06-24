#!/usr/bin/env python3
"""Phase 2 实施：沿时间线逐步展开提取人物动作。

依据 ``docs/基于时间线的展开思路3.md`` 落地。对 Phase 1 产出的每个子时期，依次执行：

* **Phase 2A 场景展开**：把子时期拆为有序的场景序列（singular / recurring）。
  跨度极短的单场景子时期跳过展开，直接整理为一个场景。是否展开由一次轻量 LLM 判断决定。
* **Phase 2B Round 1 动作提取**：以 14 类动作 taxonomy 为引导，逐条提取场景内的外部与内部动作，
  保留时间 / 因果顺序。
* **Phase 2B Round 2 决策上下文**：仅对决策类动作（decision / suppression / social）补充
  备选方案、决策因素、确信程度与后续影响——直接服务于 belief-observation-action 训练格式。
* **recurring 演变追问**：对反复发生的场景额外提取行为模式的演变轨迹。

每个动作在内存中以自包含上下文的 :class:`Phase2Record` 表示；磁盘上以树形
（:class:`StoredSubPeriod` → scenes → actions，场景上下文每场景只存一次）紧凑存储，
通过 :func:`load_records` 展平回 ``Phase2Record`` 列表供下游消费——“磁盘紧凑、内存完整”。

落盘与断点续跑（checkpoint-by-product）：

* 分片：``phase2-actions/{stage:02d}-{sub:02d}-{子时期名}.json``，每个子时期一棵
  ``StoredSubPeriod`` 树。分片文件的存在性即进度标记——重跑时已存在且合法的分片直接
  跳过（旧版扁平分片会自动迁移为树形），损坏分片删除后重来。
* 聚合：``phase2-actions.json``，与 ``phase0-meta.json`` / ``phase1-timeline.json`` 同构的
  元数据信封（``character`` 等 + ``data.sub_periods`` 树形列表 + ``run``）。下游用
  :func:`load_records` 读取并展平。

用法::

    python extract_actions.py [character_dir] [--limit N] [--workers W] [--status]

``character_dir`` 为包含 ``gist.json`` 与 ``phase1-timeline.json`` 的人物目录，缺省为简·爱。
``--limit N`` 只处理前 N 个子时期（试点 / 增量）；``--workers W`` 子时期级并发度（默认 1）；
``--status`` 为主 Agent 追加状态查询轮（默认关闭）。
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from agent import Agent
from analysis_shared import CharacterHeader
from analysis_shared import CharacterWorkspace
from analysis_shared import DEFAULT_CHARACTER_DIR
from analysis_shared import LifeStage
from analysis_shared import PHASE2_ACTIONS_NAME
from analysis_shared import SubPeriod
from analysis_shared import load_phase1_timeline
from api import LLMClient
from character_profile import StoredProfile

logger = logging.getLogger(__name__)


# ── 类型别名 ──────────────────────────────────────────────────────────────────

SceneType: TypeAlias = Literal["singular", "recurring"]
ActionType: TypeAlias = Literal[
    "verbal",
    "physical",
    "social",
    "instrumental",
    "expressive",
    "perception",
    "interpretation",
    "emotion",
    "recall",
    "judgment",
    "decision",
    "desire",
    "suppression",
    "belief_update",
]
DecisionConfidence: TypeAlias = Literal["坚定", "犹豫", "被迫", "冲动"]

# 仅对这些动作类型触发 Round 2 决策上下文补充。
DECISION_RELATED_ACTION_TYPES: tuple[ActionType, ...] = ("decision", "suppression", "social")


# ── Phase 2 中间产出模型 ───────────────────────────────────────────────────────


class SceneDecompositionVerdict(BaseModel):
    """对单个子时期是否需要场景展开的结构化判断。"""

    needs_decomposition: bool = Field(description="是否需要场景展开")
    reason: str = Field(description="判断依据，如'单日内的单一事件'或'跨度数年的漫长时期'")


class Scene(BaseModel):
    """Phase 2A 产出：一个时空连续的具体事件或情境。"""

    name: str = Field(description="场景名")
    scene_type: SceneType = Field(
        description="singular: 一次性事件；recurring: 反复发生的行为模式"
    )
    time_in_period: str = Field(description="此场景在子时期中的相对时间位置")
    participants: list[str] = Field(default_factory=list, description="在场人物")
    setting: str = Field(default="", description="地点/环境")
    mood: str = Field(default="", description="主基调")
    sketch: str = Field(description="50-80字的场景概述，描述发生了什么")
    frequency: str = Field(
        default="",
        description="[仅 recurring] 大约发生的频率和持续时间跨度",
    )


class Action(BaseModel):
    """Phase 2B Round 1 产出：场景内的一个动作单元。"""

    seq: int = Field(ge=1, description="在场景内的顺序号，从1开始")
    type: ActionType = Field(
        description=(
            "动作类型，必须为 taxonomy 中定义的14类之一："
            "verbal / physical / social / instrumental / expressive / "
            "perception / interpretation / emotion / recall / judgment / "
            "decision / desire / suppression / belief_update"
        )
    )
    description: str = Field(description="动作的具体描述，一句话")
    detail: str = Field(default="", description="可选的补充细节或上下文")


class DecisionContext(BaseModel):
    """Phase 2B Round 2 产出：一个决策点的深度上下文（条件触发）。"""

    ref_seq: int = Field(ge=1, description="关联的 Action.seq")
    alternatives: list[str] = Field(min_length=2, description="未选择的备选行动，至少2个")
    decision_factors: list[str] = Field(
        min_length=1,
        description=(
            "驱动实际选择的因素："
            "道德约束/性格驱动/能力限制/信息不足/情绪冲动/外部压力/过往经验"
        ),
    )
    confidence: DecisionConfidence = Field(description="确信程度：坚定/犹豫/被迫/冲动")
    consequence: str = Field(description="此决定的后续影响（场景内可知）")


class SceneEvolution(BaseModel):
    """recurring 场景额外追问产出：行为模式的演变轨迹。"""

    scene_name: str = Field(description="对应的 recurring 场景名")
    has_evolution: bool = Field(description="行为模式是否有可辨识的演变")
    trajectory: str = Field(default="", description="[如有演变] 从时期早期到晚期的变化轨迹")
    exception_instance: str = Field(default="", description="[如有] 打破常规模式的特殊实例")
    others_change: str = Field(default="", description="[如有] 其他在场人物反应/行为的变化")


# ── Phase 2 最终产出模型 ───────────────────────────────────────────────────────


class Phase2Record(BaseModel):
    """Phase 2 最终产出：一条附着完整上下文的动作记录。

    每条记录对应一个场景内的一个动作单元，携带其所属 life_stage / sub_period / scene
    的全部上下文，以及可选的决策深度与场景演变信息，不依赖外部 join。
    """

    # ── 人物标识 ──
    character: str = Field(description="人物中文名")

    # ── 时间轴上下文（来自 Phase 1） ──
    life_stage: str = Field(description="所属主要人生阶段名")
    sub_period: str = Field(description="所属子时期名")

    # ── 场景上下文（来自 Phase 2A） ──
    scene_index_in_sub_period: int = Field(
        ge=1, description="该场景在所属 sub_period 内的顺序号，从 1 开始"
    )
    scene_name: str = Field(description="所属场景名")
    scene_type: SceneType = Field(
        description="场景类型：singular=一次性事件，recurring=反复发生的行为模式"
    )
    scene_time_in_period: str = Field(default="", description="场景在子时期中的相对时间位置")
    scene_mood: str = Field(default="", description="场景主基调")
    scene_sketch: str = Field(description="场景概述，保留 Phase 2A 的原始摘要")
    scene_frequency: str = Field(
        default="", description="[仅 recurring] 大约发生的频率和持续时间跨度；singular 留空"
    )
    participants: list[str] = Field(default_factory=list, description="场景在场人物")
    setting: str = Field(default="", description="场景地点/环境")

    # ── 动作本体（来自 Phase 2B Round 1） ──
    seq_in_scene: int = Field(ge=1, description="在场景内的顺序号，从 1 开始")
    action_type: ActionType = Field(
        description=(
            "动作类型，14 类 taxonomy 之一："
            "verbal / physical / social / instrumental / expressive / "
            "perception / interpretation / emotion / recall / judgment / "
            "decision / desire / suppression / belief_update"
        )
    )
    action_description: str = Field(description="动作的具体描述，一句话")
    action_detail: str = Field(default="", description="可选的补充细节或上下文")

    # ── 决策上下文（来自 Phase 2B Round 2，条件存在） ──
    decision_alternatives: list[str] = Field(default_factory=list, description="[如有] 未选择的备选行动")
    decision_factors: list[str] = Field(default_factory=list, description="[如有] 驱动实际选择的因素")
    decision_confidence: DecisionConfidence | None = Field(
        default=None, description="[如有] 确信程度：坚定/犹豫/被迫/冲动"
    )
    decision_consequence: str = Field(default="", description="[如有] 此决定的后续影响（场景内可知）")

    # ── 场景演变（来自 recurring 追问，条件存在） ──
    scene_has_evolution: bool = Field(default=False, description="[仅 recurring] 行为模式是否有可辨识的演变")
    scene_evolution_trajectory: str = Field(default="", description="[如有演变] 从时期早期到晚期的变化轨迹")
    scene_evolution_exception: str = Field(default="", description="[如有] 打破常规模式的特殊实例")
    scene_evolution_others_change: str = Field(default="", description="[如有] 其他在场人物反应/行为的变化")

    @classmethod
    def from_enriched(
        cls,
        *,
        character: str,
        life_stage: str,
        sub_period: str,
        scene_index_in_sub_period: int,
        scene: Scene,
        action: Action,
        decision_context: Optional[DecisionContext] = None,
        scene_evolution: Optional[SceneEvolution] = None,
    ) -> Phase2Record:
        """从 Phase 2 各子阶段产出的分散对象组装为一条完整记录。"""
        return cls(
            character=character,
            life_stage=life_stage,
            sub_period=sub_period,
            scene_index_in_sub_period=scene_index_in_sub_period,
            scene_name=scene.name,
            scene_type=scene.scene_type,
            scene_time_in_period=scene.time_in_period,
            scene_mood=scene.mood,
            scene_sketch=scene.sketch,
            scene_frequency=scene.frequency,
            participants=list(scene.participants),
            setting=scene.setting,
            seq_in_scene=action.seq,
            action_type=action.type,
            action_description=action.description,
            action_detail=action.detail,
            decision_alternatives=decision_context.alternatives if decision_context else [],
            decision_factors=decision_context.decision_factors if decision_context else [],
            decision_confidence=decision_context.confidence if decision_context else None,
            decision_consequence=decision_context.consequence if decision_context else "",
            scene_has_evolution=scene_evolution.has_evolution if scene_evolution else False,
            scene_evolution_trajectory=scene_evolution.trajectory if scene_evolution else "",
            scene_evolution_exception=scene_evolution.exception_instance if scene_evolution else "",
            scene_evolution_others_change=scene_evolution.others_change if scene_evolution else "",
        )


# ── 树形落盘模型（磁盘紧凑 / 内存完整） ───────────────────────────────────────


class StoredAction(BaseModel):
    """落盘用：场景内的一个动作（含可选决策上下文，去掉冗余的场景/人物上下文）。"""

    seq: int
    type: ActionType
    description: str
    detail: str = ""
    decision_alternatives: list[str] = Field(default_factory=list)
    decision_factors: list[str] = Field(default_factory=list)
    decision_confidence: DecisionConfidence | None = None
    decision_consequence: str = ""


class StoredScene(BaseModel):
    """落盘用：一个场景及其有序动作序列（场景上下文只存一次）。"""

    scene_index: int
    name: str
    scene_type: SceneType
    time_in_period: str = ""
    mood: str = ""
    sketch: str = ""
    frequency: str = ""
    participants: list[str] = Field(default_factory=list)
    setting: str = ""
    has_evolution: bool = False
    evolution_trajectory: str = ""
    evolution_exception: str = ""
    evolution_others_change: str = ""
    actions: list[StoredAction] = Field(default_factory=list)


class StoredSubPeriod(BaseModel):
    """一个子时期的完整产出树（= 一个分片）。

    磁盘上以树形存储（场景上下文每场景只存一次），消除扁平 ``Phase2Record`` 列表
    在每条动作上重复场景/人物上下文造成的冗余；通过 :meth:`to_records` 展平回
    ``Phase2Record`` 列表供 Phase 5 与体检消费，实现“磁盘紧凑、内存完整”。
    """

    character: str
    life_stage: str
    sub_period: str
    scenes: list[StoredScene] = Field(default_factory=list)

    @classmethod
    def from_records(
        cls,
        records: list[Phase2Record],
        *,
        character: str | None = None,
        life_stage: str | None = None,
        sub_period: str | None = None,
    ) -> StoredSubPeriod:
        """把一个子时期的扁平 ``Phase2Record`` 列表归并为树。

        身份字段（character/life_stage/sub_period）缺省从 ``records[0]`` 推断；records 为空时
        必须显式提供。按 ``scene_index_in_sub_period`` 分组并保持首次出现顺序，场景级字段
        取该场景首条记录（同场景各动作的场景/演变上下文一致）。
        """
        if records:
            first = records[0]
            character = character if character is not None else first.character
            life_stage = life_stage if life_stage is not None else first.life_stage
            sub_period = sub_period if sub_period is not None else first.sub_period
        if character is None or life_stage is None or sub_period is None:
            raise ValueError("records 为空时必须显式提供 character/life_stage/sub_period")

        order: list[int] = []
        grouped: dict[int, list[Phase2Record]] = {}
        for record in records:
            idx = record.scene_index_in_sub_period
            if idx not in grouped:
                grouped[idx] = []
                order.append(idx)
            grouped[idx].append(record)

        scenes: list[StoredScene] = []
        for idx in order:
            group = grouped[idx]
            ctx = group[0]
            scenes.append(
                StoredScene(
                    scene_index=idx,
                    name=ctx.scene_name,
                    scene_type=ctx.scene_type,
                    time_in_period=ctx.scene_time_in_period,
                    mood=ctx.scene_mood,
                    sketch=ctx.scene_sketch,
                    frequency=ctx.scene_frequency,
                    participants=list(ctx.participants),
                    setting=ctx.setting,
                    has_evolution=ctx.scene_has_evolution,
                    evolution_trajectory=ctx.scene_evolution_trajectory,
                    evolution_exception=ctx.scene_evolution_exception,
                    evolution_others_change=ctx.scene_evolution_others_change,
                    actions=[
                        StoredAction(
                            seq=r.seq_in_scene,
                            type=r.action_type,
                            description=r.action_description,
                            detail=r.action_detail,
                            decision_alternatives=list(r.decision_alternatives),
                            decision_factors=list(r.decision_factors),
                            decision_confidence=r.decision_confidence,
                            decision_consequence=r.decision_consequence,
                        )
                        for r in group
                    ],
                )
            )
        return cls(character=character, life_stage=life_stage, sub_period=sub_period, scenes=scenes)

    def to_records(self) -> list[Phase2Record]:
        """展平回 ``Phase2Record`` 列表（把场景/人物上下文回填到每条动作）。"""
        records: list[Phase2Record] = []
        for scene in self.scenes:
            for action in scene.actions:
                records.append(
                    Phase2Record(
                        character=self.character,
                        life_stage=self.life_stage,
                        sub_period=self.sub_period,
                        scene_index_in_sub_period=scene.scene_index,
                        scene_name=scene.name,
                        scene_type=scene.scene_type,
                        scene_time_in_period=scene.time_in_period,
                        scene_mood=scene.mood,
                        scene_sketch=scene.sketch,
                        scene_frequency=scene.frequency,
                        participants=list(scene.participants),
                        setting=scene.setting,
                        seq_in_scene=action.seq,
                        action_type=action.type,
                        action_description=action.description,
                        action_detail=action.detail,
                        decision_alternatives=list(action.decision_alternatives),
                        decision_factors=list(action.decision_factors),
                        decision_confidence=action.decision_confidence,
                        decision_consequence=action.decision_consequence,
                        scene_has_evolution=scene.has_evolution,
                        scene_evolution_trajectory=scene.evolution_trajectory,
                        scene_evolution_exception=scene.evolution_exception,
                        scene_evolution_others_change=scene.evolution_others_change,
                    )
                )
        return records

    def action_count(self) -> int:
        """该子时期的动作总数。"""
        return sum(len(scene.actions) for scene in self.scenes)

    def to_storage_dict(self) -> dict[str, Any]:
        """紧凑落盘：省略默认值/空值字段（嵌套场景与动作一并紧凑化）。"""
        return self.model_dump(exclude_defaults=True, exclude_none=True)


# ── Phase 2A：场景展开 ─────────────────────────────────────────────────────────

_SCENE_JUDGMENT_SYSTEM_PROMPT = (
    "你是一个文本分析辅助程序。你的任务是根据子时期的时间范围描述，"
    "判断该子时期是否需要进行场景展开（decomposition）。"
    "规则：如果时间范围描述表明这是一个单日/单场景的短事件"
    "（如'一个傍晚''当天下午''那个凌晨''某次交谈'），则不需要展开；"
    "如果时间跨度达到数周或更长，则需要展开。"
    "你只通过工具调用输出判断结果，不输出其他内容。"
)


def needs_scene_decomposition(sub_period: SubPeriod, client: LLMClient) -> bool:
    """用一次轻量 LLM 调用判断子时期是否需要场景展开。"""
    agent = Agent(_SCENE_JUDGMENT_SYSTEM_PROMPT, client=client, max_iterations=4)
    agent.add_output_tool(
        "output_verdict",
        SceneDecompositionVerdict,
        "用此工具输出对当前子时期是否需要场景展开的判断。",
    )
    prompt = (
        f"子时期名称：{sub_period.name}\n"
        f"时间范围描述：{sub_period.time_range}\n"
        f"核心处境：{sub_period.core_situation}\n\n"
        "请判断此子时期是否需要场景展开。"
    )
    result = agent.run(prompt, temperature=0.0, query_status=False)
    verdicts = result.by_tool("output_verdict")
    if not verdicts:
        # 兜底：未能取得结构化判断时，保守地展开。
        return True
    return verdicts[-1].needs_decomposition


_SCENE_DECOMP_SYSTEM_PROMPT = (
    "你是一位叙事分析专家，擅长将一段人生时期拆解为具体、有序的场景序列。"
    "每个场景是一个时空连续的、可'拍成一个镜头'的具体事件或情境。"
    "你只通过 output_scene 工具调用输出场景——每个场景对应一次工具调用。"
    "请在同一轮内、按时间/因果顺序，一次性并行发出全部场景的工具调用，不要每轮只发一个。"
)


def decompose_scenes(
    character_context: str,
    sub_period: SubPeriod,
    client: LLMClient,
    *,
    query_status: bool = True,
) -> list[Scene]:
    """将子时期展开为有序的场景序列。"""
    agent = Agent(_SCENE_DECOMP_SYSTEM_PROMPT, client=client)
    agent.add_output_tool(
        "output_scene",
        Scene,
        "用此工具输出一个场景。每个场景一次调用；请在同一轮内按时间顺序并行发出全部场景。",
    )
    prompt = (
        f"人物：{character_context}\n"
        f"时期：{sub_period.name}\n"
        f"时间范围：{sub_period.time_range}\n"
        f"处境：{sub_period.core_situation}\n"
        f"起始事件：{sub_period.opening_event}\n"
        f"结束事件：{sub_period.closing_event}\n\n"
        "请将此时期展开为一个有序的场景序列。\n"
        "要求：\n"
        "1. 覆盖完整时间线，不跳过'平淡'的部分。\n"
        "2. 标志性事件、日常典型情境、过渡场景都要包括。\n"
        "3. 对长时间跨度中反复发生的行为模式，用 recurring 场景来代表"
        "——在 frequency 字段说明大约频率，sketch 描述一次典型实例。\n"
        "4. 场景之间保持时间/因果顺序。\n"
        "5. 注意：opening_event 和 closing_event 是子时期的转折锚点——它们中蕴含的关键动作"
        "和事件必须被覆盖，宁可边界处偶发重叠，也不能遗漏。"
        "opening_event 可作为本时期第一个场景的核心内容自然展开；"
        "closing_event 标志本时期的结束，可在最后一个场景中体现。\n\n"
        "目标数量（根据时期跨度自适应）：\n"
        "  - 数周跨度：5-8 个场景\n"
        "  - 数月跨度：8-12 个场景\n"
        "  - 数年跨度：12-18 个场景\n"
        "注意：若时期信息稀疏（仅有静态处境描述而无具体事件链），"
        "以 recurring 场景为主，不强凑 singular 数量。"
    )
    result = agent.run(prompt, temperature=0.4, query_status=query_status)
    return result.by_tool("output_scene")


_SCENE_FROM_SP_SYSTEM_PROMPT = (
    "你是一个文本分析辅助程序。将一个人生子时期的信息整理为一个 Scene 结构。"
    "你只通过 output_scene 工具调用输出结果。"
)


def scene_from_sub_period(sub_period: SubPeriod, client: LLMClient) -> Scene:
    """将单场景子时期整理为一个 Scene（跳过展开路径）。"""
    agent = Agent(_SCENE_FROM_SP_SYSTEM_PROMPT, client=client, max_iterations=4)
    agent.add_output_tool(
        "output_scene",
        Scene,
        "用此工具输出从子时期提取的单个场景。",
    )
    prompt = (
        f"子时期名称：{sub_period.name}\n"
        f"时间范围：{sub_period.time_range}\n"
        f"核心处境：{sub_period.core_situation}\n"
        f"起始事件：{sub_period.opening_event}\n"
        f"结束事件：{sub_period.closing_event}\n\n"
        "请将以上信息整理为一个 scene_type='singular' 的场景。"
        "从核心处境和起止事件中推断 participants / setting / mood，"
        "以 core_situation 为主要依据撰写 sketch。"
    )
    result = agent.run(prompt, temperature=0.0, query_status=False)
    scenes = result.by_tool("output_scene")
    if scenes:
        return scenes[0]
    # 兜底：用最直接的字段映射构造一个最小 Scene。
    return Scene(
        name=sub_period.name,
        scene_type="singular",
        time_in_period=sub_period.time_range,
        participants=[],
        setting="",
        mood="",
        sketch=sub_period.core_situation,
    )


# ── Phase 2B Round 1：taxonomy 引导的动作提取 ──────────────────────────────────

_ACTION_TAXONOMY = (
    "外部动作："
    "verbal（说了什么、对谁、什么语气、什么意图）、"
    "physical（身体动作、位移、姿态）、"
    "social（服从/反抗/讨好/回避/合作/对抗）、"
    "instrumental（使用物品、读写、劳作）、"
    "expressive（哭、笑、沉默、颤抖、叹息）；"
    "内部动作："
    "perception（注意到/观察到了什么）、"
    "interpretation（如何理解当前情境）、"
    "emotion（产生什么情绪、情绪如何变化）、"
    "recall（想起了什么）、"
    "judgment（对人或事做出什么评价）、"
    "decision（做了什么决定，包括'决定忍耐''决定不说'）、"
    "desire（想要什么、渴望什么）、"
    "suppression（压抑了什么冲动）、"
    "belief_update（信念/认知是否发生变化）"
)

_ACTION_EXTRACT_SYSTEM_PROMPT = (
    "你是一位动作分析专家。给定一个场景，你逐条列举目标人物在其中的全部动作，"
    "包括外部可观察行为和内部心理过程。"
    "你只通过 output_action 工具调用输出动作——每个动作对应一次工具调用。"
    "请在同一轮内、按时间/因果顺序，一次性并行发出该场景的全部动作调用，不要每轮只发一个。"
    f"动作类型必须从以下 taxonomy 中选择：{_ACTION_TAXONOMY}。"
    "每种类型都考虑是否存在，但不强求每种都有。"
)


def extract_actions_with_taxonomy(
    character_context: str,
    scene: Scene,
    sub_period: SubPeriod,
    client: LLMClient,
    *,
    query_status: bool = True,
) -> list[Action]:
    """对单个场景做 taxonomy 引导的全量动作提取。"""
    agent = Agent(_ACTION_EXTRACT_SYSTEM_PROMPT, client=client)
    agent.add_output_tool(
        "output_action",
        Action,
        "用此工具输出场景中的一个动作。每个动作一次调用；请在同一轮内按时间/因果顺序并行发出全部动作。",
    )
    prompt = (
        f"人物：{character_context}\n"
        f"场景：{scene.name}\n"
        f"场景类型：{scene.scene_type}\n"
        f"在场人物：{', '.join(scene.participants) if scene.participants else '（未明确，请据概述推断）'}\n"
        f"环境：{scene.setting}\n"
        f"基调：{scene.mood}\n"
        f"概述：{scene.sketch}\n"
        f"所属子时期背景：{sub_period.core_situation}\n\n"
        f"请列举 {character_context.split('（')[0]} 在此场景中的全部动作，包括外部动作和内部动作。\n"
        "按时间/因果顺序排列，每个动作标注 type 与从 1 开始递增的 seq；"
        "请在同一轮内并行发出所有动作调用。\n\n"
        "注意：\n"
        "- 如果场景类型是 recurring，描述的是一次典型实例中的动作序列。\n"
        "- 粒度要求：一个动作 ≈ 一个可观察的行为单元或一次内心转变。\n"
        "- 不要合并多个动作为一条笼统描述。\n"
        "- 预期产出 15-30 个动作。"
    )
    result = agent.run(prompt, temperature=0.4, query_status=query_status)
    return result.by_tool("output_action")


def _renumber_actions(actions: list[Action]) -> list[Action]:
    """按收集顺序把 seq 重写为 1..N，保证唯一且连续。

    模型自报的 seq 可能重复或跳号，而 Round 2 的 ``DecisionContext.ref_seq`` 依赖 seq
    精确匹配。在送入决策轮之前统一重编号，可消除这一脆弱点。
    """
    return [action.model_copy(update={"seq": i}) for i, action in enumerate(actions, start=1)]


# ── Phase 2B Round 2：决策上下文 ───────────────────────────────────────────────

_DECISION_CONTEXT_SYSTEM_PROMPT = (
    "你是一位决策分析专家。给定一个场景和其中涉及选择/决策的动作，"
    "你为每个决策点补充备选方案、决策因素、确信程度与后续影响。"
    "你只通过 output_decision_context 工具调用输出——每个决策点对应一次工具调用。"
    "请在同一轮内并行发出全部决策点调用，并在 ref_seq 中写明所关联动作的 seq。"
)


def extract_decision_context(
    character_context: str,
    scene: Scene,
    decision_actions: list[Action],
    client: LLMClient,
    *,
    query_status: bool = True,
) -> list[DecisionContext]:
    """对含决策/抑制/社会性选择的动作补充决策深度。"""
    agent = Agent(_DECISION_CONTEXT_SYSTEM_PROMPT, client=client, max_iterations=16)
    agent.add_output_tool(
        "output_decision_context",
        DecisionContext,
        "用此工具补充一个决策点的深度上下文。每个决策点单独一次调用。",
    )
    actions_text = "\n".join(
        f"  seq={a.seq} type={a.type} description={a.description}" for a in decision_actions
    )
    prompt = (
        f"人物：{character_context}\n"
        f"场景：{scene.name}\n"
        f"场景概述：{scene.sketch}\n\n"
        f"以下是此场景中涉及选择/决策的动作：\n{actions_text}\n\n"
        "对每个决策点，请补充（在 ref_seq 中写明对应动作的 seq）：\n"
        "1. 备选方案：此人当时还有哪些可选行动？（至少列2个未选择的路径）\n"
        "2. 决策因素：是什么让此人选择了实际路径而非备选？\n"
        "   （道德约束/性格驱动/能力限制/信息不足/情绪冲动/外部压力/过往经验）\n"
        "3. 确信程度：此人对这个决定有多确信？（坚定/犹豫/被迫/冲动）\n"
        "4. 后续影响：这个决定导致了什么后果？（如果在当前场景内可知）"
    )
    result = agent.run(prompt, temperature=0.3, query_status=query_status)
    return result.by_tool("output_decision_context")


# ── recurring 场景演变追问 ─────────────────────────────────────────────────────

_EVOLUTION_SYSTEM_PROMPT = (
    "你是一位行为模式分析专家。给定一个反复发生的场景及其典型动作序列，"
    "你判断该行为模式在时期进程中是否有演变，并结构化地描述演变轨迹。"
    "你只通过 output_evolution 工具调用输出结果。"
)


def extract_evolution(
    character_context: str,
    scene: Scene,
    actions: list[Action],
    client: LLMClient,
) -> SceneEvolution:
    """对 recurring 场景追问行为演变。"""
    agent = Agent(_EVOLUTION_SYSTEM_PROMPT, client=client, max_iterations=4)
    agent.add_output_tool(
        "output_evolution",
        SceneEvolution,
        "用此工具输出 recurring 场景的行为演变分析。",
    )
    actions_text = "\n".join(
        f"  seq={a.seq} type={a.type} description={a.description}" for a in actions
    )
    prompt = (
        f"人物：{character_context}\n"
        f"反复发生的场景：{scene.name}\n"
        f"频率：{scene.frequency}\n"
        f"典型实例的动作列表：\n{actions_text}\n\n"
        "这个场景在此时期内反复发生。请问：\n"
        "1. 从时期开始到结束，此人在这个场景中的行为模式有无演变？\n"
        "   如果有，描述从早期到晚期的变化轨迹。\n"
        "2. 有没有某一次'例外'——打破了常规模式的特殊实例？\n"
        "3. 其他在场人物的反应/行为有无随时间变化？"
    )
    result = agent.run(prompt, temperature=0.3, query_status=False)
    evolutions = result.by_tool("output_evolution")
    if evolutions:
        return evolutions[0]
    return SceneEvolution(scene_name=scene.name, has_evolution=False)


# ── 单个场景 → 记录 ───────────────────────────────────────────────────────────


def _process_scene(
    character_context: str,
    character_name: str,
    life_stage: LifeStage,
    sub_period: SubPeriod,
    scene_index: int,
    scene: Scene,
    client: LLMClient,
    *,
    query_status: bool,
) -> list[Phase2Record]:
    """对单个场景执行 2B Round1 + Round2（+ recurring 演变），返回该场景的记录列表。"""
    actions = extract_actions_with_taxonomy(
        character_context, scene, sub_period, client, query_status=query_status
    )
    actions = _renumber_actions(actions)
    if not actions:
        logger.warning("场景未产出任何动作：%s / %s", sub_period.name, scene.name)
        return []

    decision_actions = [a for a in actions if a.type in DECISION_RELATED_ACTION_TYPES]
    ctx_by_seq: dict[int, DecisionContext] = {}
    if decision_actions:
        contexts = extract_decision_context(
            character_context, scene, decision_actions, client, query_status=query_status
        )
        ctx_by_seq = {dc.ref_seq: dc for dc in contexts}

    evolution: SceneEvolution | None = None
    if scene.scene_type == "recurring":
        evolution = extract_evolution(character_context, scene, actions, client)

    return [
        Phase2Record.from_enriched(
            character=character_name,
            life_stage=life_stage.name,
            sub_period=sub_period.name,
            scene_index_in_sub_period=scene_index,
            scene=scene,
            action=action,
            decision_context=ctx_by_seq.get(action.seq),
            scene_evolution=evolution,
        )
        for action in actions
    ]


def _process_sub_period(
    character_context: str,
    character_name: str,
    life_stage: LifeStage,
    sub_period: SubPeriod,
    client: LLMClient,
    *,
    query_status: bool,
) -> tuple[list[Phase2Record], int]:
    """对单个子时期执行完整的 2A→2B→演变流程，返回 (记录列表, 场景数)。"""
    if needs_scene_decomposition(sub_period, client):
        scenes = decompose_scenes(character_context, sub_period, client, query_status=query_status)
        if not scenes:
            logger.warning("场景展开为空，回退为单场景：%s", sub_period.name)
            scenes = [scene_from_sub_period(sub_period, client)]
    else:
        scenes = [scene_from_sub_period(sub_period, client)]

    records: list[Phase2Record] = []
    for scene_index, scene in enumerate(scenes, start=1):
        records.extend(
            _process_scene(
                character_context,
                character_name,
                life_stage,
                sub_period,
                scene_index,
                scene,
                client,
                query_status=query_status,
            )
        )
    return records, len(scenes)


# ── 分片 I/O 与断点续跑 ────────────────────────────────────────────────────────


def _load_existing_shard(
    shard_path: Path, *, workspace: CharacterWorkspace | None = None
) -> StoredSubPeriod | None:
    """读取分片为子时期树；旧版扁平分片自动迁移为树形；损坏则删除并返回 ``None``。"""
    workspace = workspace or CharacterWorkspace.from_path(shard_path.parent.parent)
    try:
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("分片 JSON 解析失败，删除后重跑：%s", shard_path, exc_info=True)
        shard_path.unlink(missing_ok=True)
        return None
    try:
        if isinstance(payload, list):
            # 旧版扁平分片（list[Phase2Record]）→ 迁移为树形并重写。
            records = [Phase2Record.model_validate(item) for item in payload]
            tree = StoredSubPeriod.from_records(records)
            workspace.write_json(shard_path, tree.to_storage_dict(), atomic=True)
            logger.info("已将扁平分片迁移为树形：%s", shard_path)
            return tree
        tree = StoredSubPeriod.model_validate(payload)
        compact = tree.to_storage_dict()
        if payload != compact:  # 旧的非紧凑树 → 紧凑化（幂等）。
            workspace.write_json(shard_path, compact, atomic=True)
        return tree
    except (ValidationError, KeyError, TypeError, ValueError):
        logger.warning("检测到损坏分片，删除后重跑：%s", shard_path, exc_info=True)
        shard_path.unlink(missing_ok=True)
        return None


def load_records(character_dir: str | Path) -> list[Phase2Record]:
    """读取一个人物的 Phase 2 产出并展平为 ``Phase2Record`` 列表（供 Phase 5 / 体检消费）。

    优先读聚合 ``phase2-actions.json`` 的 ``data.sub_periods``；缺则拼接 ``phase2-actions/``
    下各分片。兼容旧版扁平格式（``data.records`` 或分片 ``list[Phase2Record]``）。
    """
    workspace = CharacterWorkspace.from_path(character_dir)
    records: list[Phase2Record] = []
    aggregate = workspace.phase2_actions_path
    if aggregate.exists():
        data = json.loads(aggregate.read_text(encoding="utf-8")).get("data", {})
        if "sub_periods" in data:
            for tree_payload in data["sub_periods"]:
                records.extend(StoredSubPeriod.model_validate(tree_payload).to_records())
            return records
        return [Phase2Record.model_validate(item) for item in data.get("records", [])]

    shard_dir = workspace.phase2_shard_dir
    if not shard_dir.is_dir():
        return records
    for shard in sorted(shard_dir.glob("*.json")):
        payload = json.loads(shard.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend(Phase2Record.model_validate(item) for item in payload)
        else:
            records.extend(StoredSubPeriod.model_validate(payload).to_records())
    return records


# ── 聚合记录 ──────────────────────────────────────────────────────────────────


def _build_aggregate_record(
    stored: StoredProfile,
    model: str,
    trees: list[StoredSubPeriod],
    summaries: list[dict[str, Any]],
    failures: list[dict[str, str]],
    *,
    coverage_mode: str = "full",
) -> dict[str, Any]:
    """组装与 phase0/phase1 同构的聚合信封：data.sub_periods 为树形产出主体，run 为复盘元数据。"""
    header = CharacterHeader.from_stored_profile(stored)
    scenes_total = sum(len(t.scenes) for t in trees)
    actions_total = sum(len(s.actions) for t in trees for s in t.scenes)
    decision_enriched = sum(
        1 for t in trees for s in t.scenes for a in s.actions if a.decision_factors
    )
    recurring_actions = sum(
        len(s.actions) for t in trees for s in t.scenes if s.scene_type == "recurring"
    )
    return {
        **header.to_record_base(PHASE2_ACTIONS_NAME),
        "data": {"sub_periods": [tree.to_storage_dict() for tree in trees]},
        "run": {
            "model": model,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "coverage_mode": coverage_mode,
            "counts": {
                "sub_periods_processed": len(summaries),
                "sub_periods_failed": len(failures),
                "scenes": scenes_total,
                "actions": actions_total,
                "decision_enriched_actions": decision_enriched,
                "recurring_actions": recurring_actions,
            },
            "sub_period_summaries": summaries,
            "failures": failures,
        },
    }


def _summary_from_tree(tree: StoredSubPeriod, *, resumed: bool) -> dict[str, Any]:
    """从一个子时期树回填进度摘要。"""
    return {
        "life_stage": tree.life_stage,
        "sub_period": tree.sub_period,
        "scenes": len(tree.scenes),
        "actions": tree.action_count(),
        "resumed": resumed,
    }


# ── 单子时期 worker（并发单元） ────────────────────────────────────


@dataclass
class _UnitOutcome:
    """一个子时期的处理结果。tree/summary 与 failure 互斥（失败时 tree、summary 均为 None）。"""

    tree: StoredSubPeriod | None
    summary: dict[str, Any] | None
    failure: dict[str, str] | None


def _process_unit(
    unit: tuple[int, LifeStage, int, SubPeriod],
    *,
    workspace: CharacterWorkspace,
    character_context: str,
    character_name: str,
    client: LLMClient,
    query_status: bool,
) -> _UnitOutcome:
    """处理单个子时期：断点续跑命中则复用分片树，否则提取并写树形分片。线程安全（各写各的分片）。"""
    stage_index, life_stage, sub_period_index, sub_period = unit
    shard_path = workspace.phase2_shard_path(stage_index, sub_period_index, sub_period.name)

    if shard_path.exists():
        tree = _load_existing_shard(shard_path, workspace=workspace)
        if tree is not None:
            logger.info(
                "跳过已完成子时期：%s（%d 场景，%d 动作）",
                sub_period.name, len(tree.scenes), tree.action_count(),
            )
            return _UnitOutcome(tree, _summary_from_tree(tree, resumed=True), None)

    logger.info("处理子时期：%s / %s", life_stage.name, sub_period.name)
    try:
        records, scene_count = _process_sub_period(
            character_context, character_name, life_stage, sub_period, client, query_status=query_status
        )
    except Exception:  # noqa: BLE001 - 单子时期失败需隔离，等待后续重跑覆盖
        logger.warning(
            "子时期处理失败，跳过等待重跑：%s / %s", life_stage.name, sub_period.name, exc_info=True
        )
        return _UnitOutcome(None, None, {"life_stage": life_stage.name, "sub_period": sub_period.name})

    tree = StoredSubPeriod.from_records(
        records, character=character_name, life_stage=life_stage.name, sub_period=sub_period.name
    )
    workspace.write_json(shard_path, tree.to_storage_dict(), atomic=True)
    logger.info("完成子时期：%s（%d 场景，%d 动作）", sub_period.name, scene_count, len(records))
    return _UnitOutcome(tree, _summary_from_tree(tree, resumed=False), None)


# ── 主流程 ────────────────────────────────────────────────────────────────────


def run(
    character_dir: str | Path,
    *,
    max_sub_periods: int | None = None,
    query_status: bool = False,
    workers: int = 1,
    coverage_mode: str = "full",
) -> Path:
    """对 *character_dir* 中的人物执行 Phase 2 动作提取，写出分片与聚合文件。

    Args:
        character_dir: 含 ``gist.json`` 与 ``phase1-timeline.json`` 的人物目录。
        max_sub_periods: 只处理前 N 个子时期（试点/增量），None 表示全部。
        query_status: 主 Agent（场景展开/动作提取/决策上下文）是否追加状态查询轮。
            默认关闭：Phase 2 不消费中间 Agent 的状态，且结构校验/自愈不依赖它。
        workers: 子时期级并发度。子时期相互独立（各写各的分片），>1 时用线程池并发；
            缺省 1 为串行。注意 API 速率限制，过高可能触发 429。

    Returns:
        聚合产出 ``phase2-actions.json`` 的路径。
    """
    workspace = CharacterWorkspace.from_path(character_dir)
    stored = workspace.load_profile()
    character_context = workspace.load_header().format_prompt_context()
    character_name = stored.profile.chinese_name
    timeline = load_phase1_timeline(workspace)

    # 展平 (阶段序号, 阶段, 子时期序号, 子时期)，并按需截断。
    units: list[tuple[int, LifeStage, int, SubPeriod]] = timeline.data.iter_units()
    if max_sub_periods is not None:
        units = units[:max_sub_periods]

    workers = max(1, workers)
    logger.info(
        "开始 Phase 2 动作提取：%s（共 %d 个子时期，并发 %d）",
        character_name, len(units), workers,
    )

    with LLMClient() as client:
        model = client.model

        def work(unit: tuple[int, LifeStage, int, SubPeriod]) -> _UnitOutcome:
            return _process_unit(
                unit,
                workspace=workspace,
                character_context=character_context,
                character_name=character_name,
                client=client,
                query_status=query_status,
            )

        # 子时期彼此独立（各写各的分片），可安全并发；workers=1 即退化为串行。
        if workers == 1:
            outcomes = [work(unit) for unit in units]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                outcomes = list(executor.map(work, units))

    # 按子时期顺序汇总（executor.map 保序）。
    trees: list[StoredSubPeriod] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for outcome in outcomes:
        if outcome.tree is not None:
            trees.append(outcome.tree)
        if outcome.summary is not None:
            summaries.append(outcome.summary)
        if outcome.failure is not None:
            failures.append(outcome.failure)

    record = _build_aggregate_record(
        stored, model, trees, summaries, failures, coverage_mode=coverage_mode
    )
    output_path = workspace.phase2_actions_path
    workspace.write_json(output_path, record, atomic=True)

    counts = record["run"]["counts"]
    logger.info(
        "Phase 2 完成：子时期 %d（失败 %d），场景 %d，动作 %d，决策增强 %d，recurring %d",
        counts["sub_periods_processed"], counts["sub_periods_failed"], counts["scenes"],
        counts["actions"], counts["decision_enriched_actions"], counts["recurring_actions"],
    )
    logger.info("已写入：%s", output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2：沿时间线逐步展开提取人物动作。")
    parser.add_argument(
        "character_dir",
        nargs="?",
        default=DEFAULT_CHARACTER_DIR,
        help="含 gist.json 与 phase1-timeline.json 的人物目录（缺省为简·爱）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="只处理前 N 个子时期（试点/增量）。",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="为主 Agent 追加状态查询轮（默认关闭，Phase 2 不消费中间 Agent 状态）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="W",
        help="子时期级并发度（默认 1 串行）。子时期独立可并发，但注意 API 速率限制。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run(args.character_dir, max_sub_periods=args.limit, query_status=args.status, workers=args.workers)
