#!/usr/bin/env python3
"""规范化角色档案（CharacterProfile）及其 gist.json 读写。

``character_reader.iter_characters`` 产出的 :class:`~character_gist.CharacterGist`
是从 Markdown 名录里原样解析出的「原始条目」，其 ``name`` 字段格式不统一：常把
译名、罗马音、别号混入括号并以 ``/`` 分隔（既不利机读，又会在用作目录名时被
``/`` 拆成多层目录）。

本模块定义经 LLM 清洗后的规范化档案 :class:`CharacterProfile`，把母语本名、中文名、
别名、出处、简介拆分为一致字段；并提供把完整记录读写为人物子目录下 ``gist.json``
的工具，以及把任意名称转为文件系统安全目录名的 :func:`safe_dir_name`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

# gist.json 内人物子目录的文件名。
GIST_FILENAME = "gist.json"

# 跨平台不安全的路径字符：Unix 仅禁 '/' 与 NUL，但为兼容 Windows 一并替换
# : * ? " < > | 及控制字符。
_ILLEGAL_PATH_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def safe_dir_name(name: str) -> str:
    """把任意名称转为文件系统安全的单层目录名。

    替换路径非法字符为下划线，折叠空白，并去除首尾空白与点（Windows 不允许
    目录名以点/空格结尾）。结果为空时回退为占位名，避免产生非法路径。
    """
    cleaned = _ILLEGAL_PATH_CHARS.sub("_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" .")
    return cleaned or "_unnamed_"


class CharacterProfile(BaseModel):
    """LLM 清洗产出的规范化角色档案。字段刻意拆分，便于后续机读与检索。"""

    native_name: str = Field(
        description=(
            "人物在其母语 / 原作语言中的本名，用该语言原文书写。"
            "判断规则：真实人物用其本人母语（日本人用日语、法国人用法语、俄国人用俄语…）；"
            "虚构角色用原作语言（日漫角色用日语、英文小说角色用英语…）；"
            "中国人物或华语作品角色用中文。"
        )
    )
    native_language: str = Field(
        description="native_name 所用的语言，如：中文、日语、英语、法语、俄语、蒙古语。"
    )
    chinese_name: str = Field(
        description=(
            "中文世界最通用的单一规范名称。若本就是中文则与 native_name 一致或为其常用简称。"
            "只填一个规范名，其余写法一律放入 aliases。"
        )
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "所有其他称呼，逐个独立成数组元素：译名、别号、罗马音 / 拼音、外号、"
            "其他语言写法、本名全称等。没有则留空数组。"
        ),
    )
    source: str = Field(
        description=(
            "出处来源。虚构角色填作品名（尽量含作者与年代）；"
            "真实人物填其所属领域与核心身份的概述。"
        )
    )
    gist: str = Field(
        description="一句精炼、准确的中文简介，概括此人物最核心的身份与特征；可在原简介上修订补全。"
    )


@dataclass(frozen=True)
class StoredProfile:
    """落盘 ``gist.json`` 的完整记录：规范化档案 + 溯源与清洗元数据。

    Attributes:
        profile: 规范化后的角色档案。
        classification: 分类路径分段（与原始条目一致）。
        raw_name: 清洗前的原始名称字段，保留以便溯源与复查清洗质量。
        raw_gist: 清洗前的原始简介字段。
        clean_model: 执行清洗的模型名。
        clean_timestamp: 清洗时间戳（ISO 8601）。
    """

    profile: CharacterProfile
    classification: tuple[str, ...]
    raw_name: str
    raw_gist: str
    clean_model: str
    clean_timestamp: str


def save_profile(character_dir: str | Path, record: StoredProfile) -> Path:
    """把 *record* 写入 ``character_dir/gist.json``，必要时自动创建目录。

    Args:
        character_dir: 目标人物目录（调用方已用 :func:`safe_dir_name` 处理过各段）。
        record: 要保存的完整记录。

    Returns:
        写入的 ``gist.json`` 完整路径。
    """
    character_dir = Path(character_dir)
    character_dir.mkdir(parents=True, exist_ok=True)
    data = {
        **record.profile.model_dump(),
        "classification": list(record.classification),
        "raw_name": record.raw_name,
        "raw_gist": record.raw_gist,
        "clean": {"model": record.clean_model, "timestamp": record.clean_timestamp},
    }
    gist_path = character_dir / GIST_FILENAME
    gist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return gist_path


def load_profile(character_dir: str | Path) -> StoredProfile:
    """从 ``character_dir/gist.json`` 反序列化出一条 :class:`StoredProfile`。

    Raises:
        FileNotFoundError: ``gist.json`` 不存在。
        KeyError: JSON 缺少必要字段。
    """
    gist_path = Path(character_dir) / GIST_FILENAME
    data = json.loads(gist_path.read_text(encoding="utf-8"))
    clean = data.get("clean", {})
    profile = CharacterProfile(
        native_name=data["native_name"],
        native_language=data["native_language"],
        chinese_name=data["chinese_name"],
        aliases=list(data.get("aliases", [])),
        source=data.get("source", ""),
        gist=data.get("gist", ""),
    )
    return StoredProfile(
        profile=profile,
        classification=tuple(data.get("classification", ())),
        raw_name=data.get("raw_name", ""),
        raw_gist=data.get("raw_gist", ""),
        clean_model=clean.get("model", ""),
        clean_timestamp=clean.get("timestamp", ""),
    )
