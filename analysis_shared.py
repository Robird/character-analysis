#!/usr/bin/env python3
"""跨阶段共享的人物分析 DTO、路径约定与 I/O helper。

本模块收拢 Phase 0/1/2 之间重复出现的几类抽象：

* 人物公共头部字段（character/native_name/aliases/source/gist/classification）
* phase 文件名、默认试点目录、Phase 2 分片路径规则
* 旧产物兼容读取（尤其是 phase0 / phase1 与 phase2 分片）
* 与时间轴相关的跨阶段 DTO（Phase 1 产出会被 Phase 2 消费）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from character_profile import StoredProfile
from character_profile import load_profile
from character_profile import safe_dir_name

DEFAULT_CHARACTER_DIR = Path(
    "output/fiction/文学/英国文学/古典至19世纪/勃朗特姐妹/Jane Eyre（简·爱）"
)

PHASE0_META_NAME = "phase0-meta"
PHASE1_TIMELINE_NAME = "phase1-timeline"
PHASE2_ACTIONS_NAME = "phase2-actions"

PHASE0_META_FILENAME = f"{PHASE0_META_NAME}.json"
PHASE1_TIMELINE_FILENAME = f"{PHASE1_TIMELINE_NAME}.json"
PHASE2_ACTIONS_FILENAME = f"{PHASE2_ACTIONS_NAME}.json"
PHASE2_SHARD_DIRNAME = PHASE2_ACTIONS_NAME


def read_json(path: str | Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, atomic: bool = False) -> None:
    """写 JSON 文件；必要时用临时文件 + replace 保证原子替换。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if atomic:
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
        return
    path.write_text(text, encoding="utf-8")


class CharacterHeader(BaseModel):
    """跨 phase 重复出现的人物公共头部字段。"""

    character: str = Field(description="人物中文规范名。")
    native_name: str = Field(description="人物母语/原作语言本名。")
    aliases: list[str] = Field(default_factory=list, description="其他别名、译名、罗马音等。")
    source: str = Field(default="", description="出处或身份概述。")
    gist: str = Field(default="", description="人物一句话简介。")
    classification: list[str] = Field(default_factory=list, description="分类路径分段。")

    @classmethod
    def from_stored_profile(cls, stored: StoredProfile) -> Self:
        profile = stored.profile
        return cls(
            character=profile.chinese_name,
            native_name=profile.native_name,
            aliases=list(profile.aliases),
            source=profile.source,
            gist=profile.gist,
            classification=list(stored.classification),
        )

    @property
    def classification_path(self) -> str:
        """把分类数组格式化为 prompt 友好的单行路径。"""
        return "/".join(self.classification)

    def format_prompt_context(self) -> str:
        """生成紧凑人物上下文，供各 Agent prompt 复用。"""
        return f"{self.character}（{self.native_name}），{self.source}。{self.gist}"

    def to_record_base(self, phase_name: str) -> dict[str, Any]:
        """生成各 phase 聚合 JSON 的公共头部。"""
        return {
            "character": self.character,
            "native_name": self.native_name,
            "aliases": list(self.aliases),
            "source": self.source,
            "gist": self.gist,
            "classification": list(self.classification),
            "pass": phase_name,
        }


class PhaseRecordBase(BaseModel):
    """带公共头部和 run 元数据的 phase 记录基类。"""

    model_config = ConfigDict(populate_by_name=True)

    character: str
    native_name: str
    aliases: list[str] = Field(default_factory=list)
    source: str = ""
    gist: str = ""
    classification: list[str] = Field(default_factory=list)
    phase_name: str = Field(alias="pass")
    run: dict[str, Any] = Field(default_factory=dict)

    def header(self) -> CharacterHeader:
        return CharacterHeader(
            character=self.character,
            native_name=self.native_name,
            aliases=list(self.aliases),
            source=self.source,
            gist=self.gist,
            classification=list(self.classification),
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
    """人物最为人知的一个标志性事件。"""

    name: str = Field(description="标志性事件名。")
    period: str = Field(default="", description="事件发生的人生时期。")
    note: str = Field(default="", description="一句话概括此事件。")


class Phase0MetaData(BaseModel):
    """Phase 0 元信息主体。"""

    roles: list[Role] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    domains: list[Domain] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    signature_events: list[SignatureEvent] = Field(default_factory=list)

    def signature_event_names(self) -> list[str]:
        """给后续 phase 返回可直接拼进 prompt 的标志性事件名列表。"""
        return [event.name for event in self.signature_events if event.name]


class Phase0MetaRecord(PhaseRecordBase):
    """`phase0-meta.json` 的结构化 DTO。"""

    data: Phase0MetaData

    @classmethod
    def from_parts(
        cls, header: CharacterHeader, *, run: dict[str, Any], data: Phase0MetaData
    ) -> Self:
        payload = header.to_record_base(PHASE0_META_NAME)
        payload["data"] = data.model_dump()
        payload["run"] = run
        return cls.model_validate(payload)


class SubPeriod(BaseModel):
    """一个主要阶段内的子时期。"""

    name: str = Field(description="子时期名称，简洁概括其主要特征。")
    time_range: str = Field(description="时间范围，可用年龄、年份或相对描述。")
    core_situation: str = Field(description="此时期此人的核心处境。")
    opening_event: str = Field(description="触发或标志此子时期开始的事件。")
    closing_event: str = Field(
        default="",
        description="结束此子时期的转折事件；若无明确结束事件则留空。",
    )


class LifeStage(BaseModel):
    """人物一生中的一个主要阶段。"""

    name: str = Field(description="主要阶段名称。")
    time_range: str = Field(description="此阶段整体时间范围。")
    summary: str = Field(description="此阶段的一句话概括。")
    sub_periods: list[SubPeriod] = Field(default_factory=list)


class Phase1TimelineData(BaseModel):
    """Phase 1 时间轴主体。"""

    life_stages: list[LifeStage] = Field(default_factory=list)

    def total_sub_periods(self) -> int:
        return sum(len(stage.sub_periods) for stage in self.life_stages)

    def iter_units(self) -> list[tuple[int, LifeStage, int, SubPeriod]]:
        """展平为 `(阶段序号, 阶段, 子时期序号, 子时期)` 列表。"""
        return [
            (stage_index, life_stage, sub_period_index, sub_period)
            for stage_index, life_stage in enumerate(self.life_stages)
            for sub_period_index, sub_period in enumerate(life_stage.sub_periods)
        ]


class Phase1TimelineRecord(PhaseRecordBase):
    """`phase1-timeline.json` 的结构化 DTO。"""

    data: Phase1TimelineData

    @classmethod
    def from_parts(
        cls, header: CharacterHeader, *, run: dict[str, Any], data: Phase1TimelineData
    ) -> Self:
        payload = header.to_record_base(PHASE1_TIMELINE_NAME)
        payload["data"] = data.model_dump()
        payload["run"] = run
        return cls.model_validate(payload)


@dataclass(frozen=True)
class CharacterWorkspace:
    """单个人物目录的路径约定与文件 I/O 入口。"""

    root: Path

    @classmethod
    def from_path(cls, character_dir: str | Path) -> Self:
        return cls(Path(character_dir).resolve())

    @property
    def phase0_meta_path(self) -> Path:
        return self.root / PHASE0_META_FILENAME

    @property
    def phase1_timeline_path(self) -> Path:
        return self.root / PHASE1_TIMELINE_FILENAME

    @property
    def phase2_actions_path(self) -> Path:
        return self.root / PHASE2_ACTIONS_FILENAME

    @property
    def phase2_shard_dir(self) -> Path:
        return self.root / PHASE2_SHARD_DIRNAME

    def phase_output_path(self, phase_name: str) -> Path:
        return self.root / f"{phase_name}.json"

    def phase2_shard_path(self, stage_index: int, sub_period_index: int, sub_period_name: str) -> Path:
        """按统一规则构造 Phase 2 子时期分片路径。"""
        filename = (
            f"{stage_index:02d}-{sub_period_index:02d}-{safe_dir_name(sub_period_name)}.json"
        )
        return self.phase2_shard_dir / filename

    def load_profile(self) -> StoredProfile:
        return load_profile(self.root)

    def load_header(self) -> CharacterHeader:
        return CharacterHeader.from_stored_profile(self.load_profile())

    def read_json(self, path: str | Path) -> Any:
        path = Path(path)
        return read_json(path if path.is_absolute() else self.root / path)

    def write_json(self, path: str | Path, payload: Any, *, atomic: bool = False) -> None:
        path = Path(path)
        write_json(path if path.is_absolute() else self.root / path, payload, atomic=atomic)


def _coerce_workspace(character_dir: str | Path | CharacterWorkspace) -> CharacterWorkspace:
    if isinstance(character_dir, CharacterWorkspace):
        return character_dir
    return CharacterWorkspace.from_path(character_dir)


def load_phase0_meta(character_dir: str | Path | CharacterWorkspace) -> Phase0MetaRecord | None:
    """读取 `phase0-meta.json`；文件不存在时返回 `None`。"""
    workspace = _coerce_workspace(character_dir)
    if not workspace.phase0_meta_path.exists():
        return None
    return Phase0MetaRecord.model_validate(read_json(workspace.phase0_meta_path))


def load_signature_event_names(character_dir: str | Path | CharacterWorkspace) -> list[str]:
    """读取 Phase 0 产出的标志性事件名，兼容旧数据。"""
    record = load_phase0_meta(character_dir)
    if record is None:
        return []
    return record.data.signature_event_names()


def load_phase1_timeline(character_dir: str | Path | CharacterWorkspace) -> Phase1TimelineRecord:
    """读取并校验 `phase1-timeline.json`。"""
    workspace = _coerce_workspace(character_dir)
    return Phase1TimelineRecord.model_validate(read_json(workspace.phase1_timeline_path))
