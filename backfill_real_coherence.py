#!/usr/bin/env python3
"""为所有非虚构角色补填「高」心理自洽性标注。

真实人物天然具有心理自洽性（他们真实存在过），无需 LLM 判断。
此脚本为所有 ``classification[0] != 'fiction'`` 且尚未有
``batch-coherence-tag.json`` 的角色直接写入 coherence=高。

用法::

    python backfill_real_coherence.py [--base output] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from character_profile import StoredProfile
from character_profile import load_profile
from character_profile import safe_dir_name
from character_reader import iter_characters

logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "batch-coherence-tag.json"


def _target_dir(base: Path, classification: tuple[str, ...], raw_name: str) -> Path:
    segments = [safe_dir_name(seg) for seg in classification]
    return base.joinpath(*segments, safe_dir_name(raw_name))


def is_real(classification: tuple[str, ...]) -> bool:
    """非虚构角色：顶层不是 'fiction'。"""
    return len(classification) > 0 and classification[0] != "fiction"


def _build_envelope(
    raw_name: str,
    raw_gist: str,
    classification: tuple[str, ...],
    stored: StoredProfile | None,
) -> dict:
    if stored is not None:
        profile = stored.profile
        header = {
            "character": profile.chinese_name,
            "native_name": profile.native_name,
            "aliases": list(profile.aliases),
            "source": profile.source,
            "gist": profile.gist,
            "classification": list(stored.classification),
        }
    else:
        header = {
            "character": raw_name,
            "native_name": raw_name,
            "aliases": [],
            "source": "",
            "gist": raw_gist,
            "classification": list(classification),
        }

    return {
        **header,
        "pass": "batch-coherence-tag",
        "data": {
            "psychological_coherence": "高",
            "coherence_note": "",
        },
        "run": {
            "model": "backfill (real person)",
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "backfilled",
            "iterations": 0,
            "max_iterations_reached": False,
        },
    }


def run(base: str | Path = "output", *, dry_run: bool = False) -> None:
    base = Path(base)
    created = 0
    skipped_existing = 0
    total_real = 0

    for raw in iter_characters():
        if not is_real(raw.classification):
            continue
        total_real += 1

        char_dir = _target_dir(base, raw.classification, raw.name)
        output_path = char_dir / OUTPUT_FILENAME

        if output_path.exists():
            skipped_existing += 1
            continue

        # Try to load gist.json for cleaner header
        stored = None
        try:
            stored = load_profile(char_dir)
        except Exception:
            pass

        record = _build_envelope(raw.name, raw.gist, raw.classification, stored)

        if dry_run:
            logger.info("  [dry-run] would create: %s", output_path)
        else:
            char_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        created += 1

    logger.info(
        "真实角色 %d 人：新增 %d，已有 %d（%s）",
        total_real,
        created,
        skipped_existing,
        "dry-run" if dry_run else "done",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为所有非虚构角色补填 coherence=高。"
    )
    parser.add_argument("--base", default="output", help="输出根目录（默认 output）。")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际写入。")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    run(base=args.base, dry_run=args.dry_run)
