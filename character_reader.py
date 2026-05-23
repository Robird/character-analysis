#!/usr/bin/env python3
"""遍历 characters 目录下所有 .md 文件，yield 每行角色的 (name, gist, path_segments) 元组。"""

from pathlib import Path
from collections.abc import Generator


def iter_characters(
    base_dir: str | Path = "characters",
    *,
    limit: int | None = None,
) -> Generator[tuple[str, str, tuple[str, ...]], None, None]:
    """生成器：遍历所有 .md 文件中的角色行。

    Args:
        base_dir: characters 目录的路径。
        limit: 最多 yield 的条目数（None = 不限）。

    Yields:
        (name, gist, path_segments) —
            name           — 破折号前的角色名。
            gist           — 破折号后的简介。
            path_segments  — 相对于 base_dir 的路径分段 tuple，
                             末段为去掉 .md 后缀的文件名。
                             例: ("real", "中华文化")
                                 ("fiction", "文学", "英国文学", "古典至19世纪")
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

        with open(md_file, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
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
                    yield (name, gist, path_segments)
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
    for i, (name, gist, path_seg) in enumerate(iter_characters(limit=max_lines), 1):
        path_str = "/".join(path_seg)
        print(f"{i:4d}. {name}")
        print(f"      [{path_str}]")
        print(f"      {gist}")
        print()
