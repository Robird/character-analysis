#!/usr/bin/env python3
"""遍历 characters 目录下所有 .md 文件，yield 每行角色的 (name, gist, path_segments, heading) 元组。"""

import re
from pathlib import Path
from collections.abc import Generator

# ATX heading: 1-6 个 # + 至少一个空格 + 标题文本
_ATX_RE = re.compile(r"^#{1,6}\s+(.+)")


def iter_characters(
    base_dir: str | Path = "characters",
    *,
    limit: int | None = None,
) -> Generator[tuple[str, str, tuple[str, ...], str | None], None, None]:
    """生成器：遍历所有 .md 文件中的角色行。

    Args:
        base_dir: characters 目录的路径。
        limit: 最多 yield 的条目数（None = 不限）。

    Yields:
        (name, gist, path_segments, heading) —
            name           — 破折号前的角色名。
            gist           — 破折号后的简介。
            path_segments  — 相对于 base_dir 的路径分段 tuple，
                             末段为去掉 .md 后缀的文件名。
            heading        — 该角色上方最近的 ATX 标题文本
                             （不含 # 号），无标题时为 None。
    """
    root = Path(base_dir)
    count = 0

    for md_file in sorted(root.rglob("*.md")):
        # 相对路径分段：去掉 base_dir 前缀，拆目录+文件名，末段去 .md
        rel = md_file.relative_to(root)
        parts = rel.parts
        # 末段去掉 .md 后缀
        stem = parts[-1]
        if stem.endswith(".md"):
            stem = stem[:-3]
        path_segments = parts[:-1] + (stem,) if len(parts) > 1 else (stem,)

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

                # 切分：第一个 —  之前为 name，之后为 gist
                body = stripped[2:]  # 去掉 "- "
                em_idx = body.index("—")
                name = body[:em_idx].strip()
                gist = body[em_idx + 1 :].strip()

                if name:  # 跳过空名称
                    yield (name, gist, path_segments, current_heading)
                    count += 1
                    if limit is not None and count >= limit:
                        return


# ── 测试 ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    max_lines = 20  # 限制输出，防止工具截断
    if len(sys.argv) > 1:
        max_lines = int(sys.argv[1])

    print(f"# 前 {max_lines} 条角色\n")
    for i, (name, gist, path_seg, heading) in enumerate(
        iter_characters(limit=max_lines), 1
    ):
        path_str = "/".join(path_seg)
        head_str = f"§ {heading}" if heading else "(无标题)"
        print(f"{i:4d}. {name}")
        print(f"      [{path_str}]  {head_str}")
        print(f"      {gist}")
        print()
