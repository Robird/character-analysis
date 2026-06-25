#!/usr/bin/env python3
"""筛选高质量 SFT 训练子集并导出。

从全部已标注人物中按四维条件筛选：

    gender=女 ∧ agency_level=高 ∧ structural_density=丰富 ∧ psychological_coherence=高

产出单个 JSON 文件，每项包含人物名、母语名、出处、分类、简介和 output 相对路径。

用法::

    python select_gold_subset.py [--out gold_subset.json]

后续可通过 ``--limit N`` 进一步缩减（如只取前 100 人做首轮实验）。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "gold_subset.json"
OUTPUT_BASE = Path("output")


def _load_tag(char_dir: Path, filename: str) -> dict | None:
    """加载单个人物目录下的标注文件，不存在或损坏返回 None。"""
    path = char_dir / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("读取失败: %s", path, exc_info=True)
        return None


def select(
    *,
    gender: str = "女",
    agency: str = "高",
    density: str = "丰富",
    coherence: str = "高",
) -> list[dict]:
    """遍历全部已标注人物，返回符合条件的条目列表。

    Args:
        gender: 性别筛选值。
        agency: 能动性筛选值。
        density: 结构密度筛选值。
        coherence: 心理自洽性筛选值。

    Returns:
        按人物名排序的条目列表，每项包含 character / native_name / source /
        gist / classification / rel_path。
    """
    results: list[dict] = []

    for gender_path in sorted(OUTPUT_BASE.rglob("batch-gender-tag.json")):
        char_dir = gender_path.parent
        rel_dir = char_dir.relative_to(OUTPUT_BASE)

        gtag = _load_tag(char_dir, "batch-gender-tag.json")
        if gtag is None:
            continue
        gdata = gtag["data"]

        # 性别 + 能动性 + 密度
        if (
            gdata.get("gender") != gender
            or gdata.get("agency_level") != agency
            or gdata.get("structural_density") != density
        ):
            continue

        # 心理自洽性
        ctag = _load_tag(char_dir, "batch-coherence-tag.json")
        if ctag is None:
            continue
        if ctag["data"].get("psychological_coherence") != coherence:
            continue

        results.append(
            {
                "character": gtag["character"],
                "native_name": gtag.get("native_name", ""),
                "source": gtag.get("source", ""),
                "gist": gtag.get("gist", ""),
                "classification": gtag.get("classification", []),
                "rel_path": str(rel_dir),
            }
        )

    results.sort(key=lambda r: r["character"])
    return results


def run(out_path: str | Path = DEFAULT_OUTPUT, *, limit: int | None = None) -> Path:
    """执行筛选并写出 JSON，返回输出路径。"""
    results = select()
    total = len(results)

    if limit is not None and limit < total:
        results = results[:limit]
        logger.info("符合条件 %d 人，取前 %d 人", total, limit)
    else:
        logger.info("符合条件 %d 人（全量）", total)

    out_path = Path(out_path)
    payload = {
        "description": (
            "高质量 SFT 训练子集：女性 + 高能动性 + 丰富结构密度 + 高心理自洽性"
        ),
        "filters": {
            "gender": "女",
            "agency_level": "高",
            "structural_density": "丰富",
            "psychological_coherence": "高",
        },
        "total": len(results),
        "total_available": total,
        "characters": results,
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("已写入: %s", out_path.resolve())
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="筛选高质量 SFT 角色子集（女+高能动+丰富密度+高自洽）。"
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"输出 JSON 路径（默认 {DEFAULT_OUTPUT}）。")
    parser.add_argument("--limit", type=int, default=None, help="仅取前 N 人（用于小规模实验）。")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    run(out_path=args.out, limit=args.limit)
