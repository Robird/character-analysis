#!/usr/bin/env python3
"""CharacterEntry 数据类型及其 JSON 序列化工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


class CharacterGist(NamedTuple):
    """一条角色记录。兼容元组解包：name, gist, classification = entry。

    Attributes:
        name: 破折号前的角色名。
        gist: 破折号后的简介。
        classification: 分类路径分段。末段为所属文件的 stem 或最近 ATX 标题。
            例: ("real", "中华文化") / ("fiction", "文学", "英国文学", "古典至19世纪")。
    """

    name: str
    gist: str
    classification: tuple[str, ...]

    # ── JSON 序列化 ──────────────────────────────────

    @staticmethod
    def SaveToJson(entry: CharacterGist, dir_path: str | Path) -> Path:
        """将 *entry* 序列化为 ``dir_path/gist.json``，必要时自动创建目录。

        Args:
            entry: 要保存的角色条目。
            dir_path: 目标目录路径。

        Returns:
            写入的 ``gist.json`` 完整路径。
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        gist_path = dir_path / "gist.json"
        data = {
            "name": entry.name,
            "classification": list(entry.classification),
            "gist": entry.gist,
        }
        gist_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return gist_path

    @staticmethod
    def LoadFromJson(dir_path: str | Path) -> CharacterGist:
        """从 ``dir_path/gist.json`` 反序列化出一条 CharacterEntry。

        Args:
            dir_path: 包含 ``gist.json`` 的目录路径。

        Returns:
            反序列化得到的 CharacterEntry。

        Raises:
            FileNotFoundError: ``gist.json`` 不存在。
            KeyError: JSON 缺少必要字段。
        """
        gist_path = Path(dir_path) / "gist.json"
        with open(gist_path, encoding="utf-8") as f:
            data = json.load(f)
        return CharacterGist(
            name=data["name"],
            gist=data["gist"],
            classification=tuple(data["classification"]),
        )
