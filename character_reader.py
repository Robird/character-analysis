#!/usr/bin/env python3
"""遍历 characters 目录下所有 .md 文件，yield CharacterEntry 命名元组。

用法::

    from character_reader import iter_characters
    from itertools import islice

    for entry in islice(iter_characters(), 20):
        print(entry.name, entry.gist, entry.classification)
"""

import re
from pathlib import Path
from collections.abc import Generator

from character_gist import CharacterGist


# ATX heading: 1-6 个 # + 至少一个空格 + 标题文本
_ATX_RE = re.compile(r"^#{1,6}\s+(.+)")


def iter_characters(
    base_dir: str | Path = "characters",
) -> Generator[CharacterGist, None, None]:
    """生成器：遍历 base_dir 下所有 .md 文件中的角色行。

    Args:
        base_dir: characters 目录的路径。

    Yields:
        CharacterEntry — 含 name、gist、classification 三个字段。
    """
    root = Path(base_dir)

    for md_file in sorted(root.rglob("*.md")):
        # 相对路径分段：去掉 base_dir 前缀，拆目录+文件名，末段去 .md
        rel = md_file.relative_to(root)
        parts = rel.parts
        stem = parts[-1]
        if stem.endswith(".md"):
            stem = stem[:-3]
        path_segments = parts[:-1] + (stem,) if len(parts) > 1 else (stem,)
        while(len(path_segments) > 1 and path_segments[-1]==path_segments[-2]):
            path_segments = path_segments[:-1]

        current_heading: str | None = None

        with open(md_file, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()

                # 检测 ATX heading（必须在 bullet 之前判断）
                m = _ATX_RE.match(stripped)
                if m:
                    current_heading = m.group(1)
                    continue

                # 只处理 bullet 行，且含破折号（有简介）
                if not stripped.startswith("- "):
                    continue
                if "—" not in stripped:
                    continue

                # 切分：第一个 — 之前为 name，之后为 gist
                body = stripped[2:]  # 去掉 "- "
                em_idx = body.index("—")
                name = body[:em_idx].strip()
                gist = body[em_idx + 1 :].strip()

                if name:  # 跳过空名称
                    if current_heading and current_heading != path_segments[-1]:
                        classification = (*path_segments, current_heading)
                    else:
                        classification = path_segments
                    yield CharacterGist(name, gist, classification)


# ── 测试 ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from itertools import islice

    max_lines = 20  # 限制输出，防止工具截断
    if len(sys.argv) > 1:
        max_lines = int(sys.argv[1])

    print(f"# 前 {max_lines} 条角色\n")
    for i, entry in enumerate(islice(iter_characters(), max_lines), 1):
        classification = "/".join(entry.classification)
        
        user_prompt = f"`{classification}` 中的 `{entry.name}` — {entry.gist}"
        print(f"{i:4d}")
        print(user_prompt)
        print()
