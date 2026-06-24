#!/usr/bin/env python3
"""Phase 2 产出质量体检（QA report）。

读取一个人物的 ``phase2-actions.json``（缺省回退到 ``phase2-actions/`` 分片），
输出一份可读的统计报告，用于在大规模推广前系统性地评估产出质量与发现问题：

* 概览：子时期 / 场景 / 动作 / 决策覆盖 / recurring 占比
* 动作类型（14 类 taxonomy）分布与内部 / 外部占比
* 每子时期的场景数 / 动作数 / 动作密度，并标记异常（过度展开、密度趋同）
* 决策确信度分布与偏倚标记
* 精确重复动作（语义去重的廉价下界代理）
* 去规范化存储冗余（上下文字段 vs 动作本体字节占比）
* 若干样本记录

只读，不修改任何产出文件。

用法::

    python inspect_phase2.py [character_dir]

``character_dir`` 缺省为简·爱。
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_DEFAULT_CHARACTER_DIR = (
    "output/fiction/文学/英国文学/古典至19世纪/勃朗特姐妹/Jane Eyre（简·爱）"
)
_AGGREGATE_FILENAME = "phase2-actions.json"
_SHARD_DIRNAME = "phase2-actions"

# 动作类型语义分组（与 extract_actions 的 taxonomy 一致）。
_EXTERNAL_TYPES = {"verbal", "physical", "social", "instrumental", "expressive"}
_INTERNAL_TYPES = {
    "perception",
    "interpretation",
    "emotion",
    "recall",
    "judgment",
    "decision",
    "desire",
    "suppression",
    "belief_update",
}

# 启发式阈值（仅用于提示，不是硬性判定）。
_OVER_DECOMP_SCENES = 12  # 单子时期场景数超过此值 → 提示复核是否过度展开
_DENSITY_CV_FLAT = 0.18  # 动作/场景 的变异系数低于此值 → 提示密度趋同
_CONFIDENCE_SKEW = 0.6  # 单一确信度占比超过此值 → 提示分布偏倚

# 落盘时被省略默认值的字段，统计前需补回，避免 KeyError。
_RECORD_DEFAULTS: dict[str, Any] = {
    "scene_time_in_period": "",
    "scene_mood": "",
    "scene_frequency": "",
    "participants": [],
    "setting": "",
    "action_detail": "",
    "decision_alternatives": [],
    "decision_factors": [],
    "decision_confidence": None,
    "decision_consequence": "",
    "scene_has_evolution": False,
    "scene_evolution_trajectory": "",
    "scene_evolution_exception": "",
    "scene_evolution_others_change": "",
}


def _load_records(character_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """加载记录列表，返回 (records, 来源说明)。优先聚合文件，回退到分片。"""
    aggregate = character_dir / _AGGREGATE_FILENAME
    if aggregate.exists():
        doc = json.loads(aggregate.read_text(encoding="utf-8"))
        return doc.get("data", {}).get("records", []), f"聚合文件 {_AGGREGATE_FILENAME}"

    shard_dir = character_dir / _SHARD_DIRNAME
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"未找到 {aggregate} 或 {shard_dir}")
    records: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("*.json")):
        records.extend(json.loads(shard.read_text(encoding="utf-8")))
    return records, f"{shard_dir.name}/ 下的分片"


def _fill_defaults(record: dict[str, Any]) -> dict[str, Any]:
    """把紧凑落盘时省略的默认字段补回，便于统一统计。"""
    return {**_RECORD_DEFAULTS, **record}


def _print_header(title: str) -> None:
    print(f"\n{'─' * 4} {title} {'─' * max(0, 56 - len(title))}")


def _report_overview(records: list[dict[str, Any]], source: str) -> None:
    sub_periods = {(r["life_stage"], r["sub_period"]) for r in records}
    scenes = {(r["life_stage"], r["sub_period"], r["scene_index_in_sub_period"]) for r in records}
    decisions = sum(1 for r in records if r.get("decision_factors"))
    recurring = sum(1 for r in records if r.get("scene_type") == "recurring")
    total = len(records)
    _print_header("概览")
    print(f"  数据来源    {source}")
    print(f"  子时期      {len(sub_periods)}")
    print(f"  场景        {len(scenes)}")
    print(f"  动作记录    {total}")
    if total:
        print(f"  决策增强    {decisions}  ({decisions / total * 100:.1f}%)")
        print(f"  recurring   {recurring}  ({recurring / total * 100:.1f}%)")
        print(f"  场景均动作  {total / max(1, len(scenes)):.1f}")


def _report_action_types(records: list[dict[str, Any]]) -> None:
    counter = collections.Counter(r["action_type"] for r in records)
    total = len(records) or 1
    _print_header("动作类型分布")
    for action_type, count in counter.most_common():
        group = "内" if action_type in _INTERNAL_TYPES else "外"
        bar = "█" * round(count / total * 40)
        print(f"  [{group}] {action_type:16s} {count:5d} {count / total * 100:5.1f}% {bar}")
    external = sum(c for t, c in counter.items() if t in _EXTERNAL_TYPES)
    internal = total - external
    print(f"\n  外部 {external} ({external / total * 100:.0f}%) | 内部 {internal} ({internal / total * 100:.0f}%)")


def _iter_sub_period_stats(records: list[dict[str, Any]]):
    """按出现顺序产出每个子时期的 (life_stage, sub_period, 场景数, 动作数, recurring占比)。"""
    grouped: dict[tuple[str, str], dict[str, Any]] = collections.OrderedDict()
    for r in records:
        key = (r["life_stage"], r["sub_period"])
        entry = grouped.setdefault(key, {"scenes": set(), "actions": 0, "recurring": 0})
        entry["scenes"].add(r["scene_index_in_sub_period"])
        entry["actions"] += 1
        if r.get("scene_type") == "recurring":
            entry["recurring"] += 1
    for (life_stage, sub_period), entry in grouped.items():
        yield life_stage, sub_period, len(entry["scenes"]), entry["actions"], entry["recurring"]


def _report_density(records: list[dict[str, Any]]) -> None:
    _print_header("每子时期密度（场景 / 动作 / 均值 / recurring占比）")
    per_scene_means: list[float] = []
    flags: list[str] = []
    for life_stage, sub_period, n_scenes, n_actions, n_recurring in _iter_sub_period_stats(records):
        mean = n_actions / max(1, n_scenes)
        per_scene_means.append(mean)
        rec_pct = n_recurring / max(1, n_actions) * 100
        mark = " ⚠过度展开?" if n_scenes >= _OVER_DECOMP_SCENES else ""
        print(f"  {sub_period[:22]:24s} 场景{n_scenes:3d} 动作{n_actions:4d} 均{mean:4.1f} rec{rec_pct:3.0f}%{mark}")
        if n_scenes >= _OVER_DECOMP_SCENES:
            flags.append(sub_period)

    # 动作/场景 密度趋同检测（变异系数）。
    if len(per_scene_means) >= 3:
        mean = statistics.mean(per_scene_means)
        cv = statistics.pstdev(per_scene_means) / mean if mean else 0.0
        _print_header("密度趋同检测")
        print(f"  每场景动作数：均值 {mean:.1f}，标准差 {statistics.pstdev(per_scene_means):.1f}，变异系数 {cv:.2f}")
        if cv < _DENSITY_CV_FLAT:
            print(
                f"  ⚠ 变异系数 {cv:.2f} < {_DENSITY_CV_FLAT}：各场景动作密度高度趋同，"
                "疑似锚定在 prompt 上限附近，未按场景重要性分配粒度。"
            )
    if flags:
        print(f"\n  ⚠ 场景数 ≥{_OVER_DECOMP_SCENES} 的子时期（建议复核是否过度展开）：{len(flags)} 个")


def _report_decisions(records: list[dict[str, Any]]) -> None:
    decisions = [r for r in records if r.get("decision_factors")]
    _print_header("决策上下文")
    if not decisions:
        print("  （无决策增强记录）")
        return
    conf = collections.Counter(r.get("decision_confidence") for r in decisions)
    total = len(decisions)
    print(f"  决策点 {total}，确信度分布：")
    for value, count in conf.most_common():
        print(f"    {str(value):6s} {count:4d}  {count / total * 100:5.1f}%")
    top_value, top_count = conf.most_common(1)[0]
    if top_count / total > _CONFIDENCE_SKEW:
        print(
            f"  ⚠ 确信度 '{top_value}' 占 {top_count / total * 100:.0f}% > {_CONFIDENCE_SKEW * 100:.0f}%："
            "分布偏倚，可能是模型默认倾向，宜在 prompt 中引导更诚实的变化。"
        )
    # 类型分布
    by_type = collections.Counter(r["action_type"] for r in decisions)
    print("  决策点动作类型：" + "，".join(f"{t}×{c}" for t, c in by_type.most_common()))


def _report_duplication(records: list[dict[str, Any]]) -> None:
    _print_header("重复与冗余")
    desc_counter = collections.Counter(r["action_description"] for r in records)
    dups = {k: v for k, v in desc_counter.items() if v > 1}
    dup_records = sum(v for v in dups.values())
    total = len(records) or 1
    print(f"  精确重复描述：{len(dups)} 种，涉及 {dup_records} 条（{dup_records / total * 100:.1f}%）")
    for text, count in sorted(dups.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    ×{count}  {text[:48]}")

    # 去规范化冗余：上下文字段 vs 动作本体字段 字节占比。
    ctx_keys = {
        "character", "life_stage", "sub_period", "scene_name", "scene_sketch",
        "participants", "setting", "scene_mood", "scene_time_in_period",
        "scene_frequency", "scene_index_in_sub_period", "scene_type",
    }
    body_keys = {"seq_in_scene", "action_type", "action_description", "action_detail"}

    def byte_size(record: dict[str, Any], keys: set[str]) -> int:
        subset = {k: record[k] for k in record if k in keys}
        return len(json.dumps(subset, ensure_ascii=False).encode("utf-8"))

    ctx_bytes = sum(byte_size(r, ctx_keys) for r in records)
    body_bytes = sum(byte_size(r, body_keys) for r in records)
    denom = ctx_bytes + body_bytes or 1
    print(
        f"  去规范化冗余：上下文字段 {ctx_bytes / 1e6:.1f}MB vs 动作本体 {body_bytes / 1e6:.1f}MB"
        f" → 重复上下文占 {ctx_bytes / denom * 100:.0f}%"
    )


def _report_samples(records: list[dict[str, Any]], count: int = 2) -> None:
    _print_header(f"样本记录（前 {count} 条决策增强）")
    shown = 0
    for r in records:
        if not r.get("decision_factors"):
            continue
        print(f"  · [{r['action_type']}] {r['action_description']}")
        print(f"      备选：{r.get('decision_alternatives')}")
        print(f"      因素：{r.get('decision_factors')}")
        print(f"      确信：{r.get('decision_confidence')}  后果：{r.get('decision_consequence', '')[:40]}")
        shown += 1
        if shown >= count:
            break


def inspect(character_dir: str | Path) -> None:
    """对 *character_dir* 的 Phase 2 产出生成质量体检报告并打印到标准输出。"""
    character_dir = Path(character_dir)
    raw_records, source = _load_records(character_dir)
    if not raw_records:
        print(f"未在 {character_dir} 找到任何 Phase 2 记录。")
        return
    records = [_fill_defaults(r) for r in raw_records]

    print(f"# Phase 2 质量体检：{character_dir.name}")
    _report_overview(records, source)
    _report_action_types(records)
    _report_density(records)
    _report_decisions(records)
    _report_duplication(records)
    _report_samples(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 产出质量体检报告。")
    parser.add_argument(
        "character_dir",
        nargs="?",
        default=_DEFAULT_CHARACTER_DIR,
        help="含 phase2-actions.json 的人物目录（缺省为简·爱）。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        inspect(args.character_dir)
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
