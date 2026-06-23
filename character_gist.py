#!/usr/bin/env python3
"""CharacterGist 数据类型：从名录 Markdown 原样解析出的「原始角色条目」。

经 LLM 清洗后的规范化档案见 :mod:`character_profile`。"""

from __future__ import annotations

from typing import NamedTuple


class CharacterGist(NamedTuple):
    """一条原始角色记录。兼容元组解包：name, gist, classification = entry。

    Attributes:
        name: 破折号前的角色名（格式不统一，可能混入译名 / 别号 / 罗马音）。
        gist: 破折号后的简介。
        classification: 分类路径分段。末段为所属文件的 stem 或最近 ATX 标题。
            例: ("real", "中华文化") / ("fiction", "文学", "英国文学", "古典至19世纪")。
    """

    name: str
    gist: str
    classification: tuple[str, ...]
