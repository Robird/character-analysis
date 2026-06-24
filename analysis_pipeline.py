#!/usr/bin/env python3
"""人物分析一键 pipeline 入口。

给定一个已包含 ``gist.json`` 的人物目录，按 Phase 0 -> Phase 1 -> Phase 2 顺序运行：

* ``phase0-meta.json``：人物元信息
* ``phase1-timeline.json``：时间轴骨架
* ``phase2-actions.json``：沿时间线展开的动作树

特性：

* 断点续跑：自动跳过已满足当前模式的 phase；Phase 2 不完整时自动续跑。
* 快速验证：``quick`` 模式让每个 phase 只产出 1-2 步的样例，便于通路验证。
* 全量运行：``full`` 模式产出完整结果，也是默认模式。
* 双入口：既可作为库函数调用，也可直接作为 CLI 执行。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

from analysis_shared import CharacterWorkspace
from analysis_shared import DEFAULT_CHARACTER_DIR
from analysis_shared import load_phase1_timeline
from extract_actions import run as run_phase2_actions
from extract_meta import run as run_phase0_meta
from extract_timeline import run as run_phase1_timeline

logger = logging.getLogger(__name__)

PipelineMode = Literal["full", "quick"]
PhaseStatus = Literal["completed", "skipped"]


@dataclass(frozen=True)
class PhaseRunResult:
    """一次 phase 执行结果。"""

    phase_name: str
    status: PhaseStatus
    output_path: Path
    coverage_mode: str
    reason: str = ""
    counts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_path"] = str(self.output_path)
        return payload


@dataclass(frozen=True)
class PipelineRunResult:
    """整条 pipeline 的执行摘要。"""

    character_dir: Path
    requested_mode: PipelineMode
    phase0: PhaseRunResult
    phase1: PhaseRunResult
    phase2: PhaseRunResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_dir": str(self.character_dir),
            "requested_mode": self.requested_mode,
            "phase0": self.phase0.to_dict(),
            "phase1": self.phase1.to_dict(),
            "phase2": self.phase2.to_dict(),
        }


@dataclass
class PipelineRunner:
    """编排单个人物目录的三阶段分析流程。"""

    character_dir: str | Path
    mode: PipelineMode = "full"
    phase2_workers: int = 1
    phase2_query_status: bool = False
    quick_phase2_sub_periods: int = 2

    def __post_init__(self) -> None:
        self.workspace = CharacterWorkspace.from_path(self.character_dir)

    workspace: CharacterWorkspace = field(init=False, repr=False)

    def run(self) -> PipelineRunResult:
        self._ensure_seed_exists()

        phase0 = self._run_phase0()
        phase1 = self._run_phase1(force_rerun=(phase0.status == "completed"))
        phase2 = self._run_phase2(force_restart=(phase1.status == "completed"))
        return PipelineRunResult(
            character_dir=self.workspace.root,
            requested_mode=self.mode,
            phase0=phase0,
            phase1=phase1,
            phase2=phase2,
        )

    def _ensure_seed_exists(self) -> None:
        gist_path = self.workspace.root / "gist.json"
        if not gist_path.exists():
            raise FileNotFoundError(f"未找到种子文件 gist.json：{gist_path}")

    def _run_phase0(self) -> PhaseRunResult:
        path = self.workspace.phase0_meta_path
        if self._phase_record_satisfies(path):
            return self._build_skip_result("phase0-meta", path, "现有 phase0 结果已满足当前模式")

        self._backup_path_if_exists(path)
        output_path = run_phase0_meta(self.workspace.root, quick=self.mode == "quick")
        return self._build_result("phase0-meta", "completed", output_path)

    def _run_phase1(self, *, force_rerun: bool) -> PhaseRunResult:
        path = self.workspace.phase1_timeline_path
        if not force_rerun and self._phase_record_satisfies(path):
            return self._build_skip_result("phase1-timeline", path, "现有 phase1 结果已满足当前模式")

        if force_rerun:
            logger.info("Phase 0 本轮已更新，Phase 1 需要重跑")
        self._backup_path_if_exists(path)
        output_path = run_phase1_timeline(self.workspace.root, quick=self.mode == "quick")
        return self._build_result("phase1-timeline", "completed", output_path)

    def _run_phase2(self, *, force_restart: bool) -> PhaseRunResult:
        aggregate = self.workspace.phase2_actions_path
        expected_sub_periods = self._expected_phase2_sub_periods()

        if force_restart:
            logger.info("Phase 1 本轮已更新，Phase 2 旧产物将备份后重跑")
            self._backup_phase2_artifacts()
        else:
            verdict = self._inspect_phase2_state(expected_sub_periods)
            if verdict == "skip":
                return self._build_skip_result(
                    "phase2-actions", aggregate, "现有 phase2 结果已满足当前模式"
                )
            if verdict == "restart":
                logger.info("现有 phase2 为 quick 结果，切换到 full 模式前先备份旧产物")
                self._backup_phase2_artifacts()

        output_path = run_phase2_actions(
            self.workspace.root,
            max_sub_periods=self._phase2_limit(),
            query_status=self.phase2_query_status,
            workers=self.phase2_workers,
            coverage_mode=self.mode,
        )
        return self._build_result("phase2-actions", "completed", output_path)

    def _phase2_limit(self) -> int | None:
        if self.mode == "quick":
            return max(1, self.quick_phase2_sub_periods)
        return None

    def _phase_record_satisfies(self, path: Path) -> bool:
        """判断 phase0/1 文件是否已满足当前模式。"""
        if not path.exists():
            return False
        coverage_mode = self._read_coverage_mode(path)
        if self.mode == "quick":
            return coverage_mode in {"quick", "full"}
        return coverage_mode == "full"

    def _inspect_phase2_state(self, expected_sub_periods: int) -> Literal["skip", "resume", "restart"]:
        """判断 phase2 是可跳过、可续跑，还是应从头重跑。"""
        aggregate = self.workspace.phase2_actions_path
        if not aggregate.exists():
            return "resume"

        payload = self.workspace.read_json(aggregate)
        run = payload.get("run", {})
        data = payload.get("data", {})
        coverage_mode = run.get("coverage_mode", "full")
        processed = len(data.get("sub_periods", []))

        if self.mode == "quick":
            return "skip" if processed >= expected_sub_periods else "resume"

        if coverage_mode == "quick":
            return "restart"
        if processed >= expected_sub_periods:
            return "skip"
        return "resume"

    def _expected_phase2_sub_periods(self) -> int:
        timeline = load_phase1_timeline(self.workspace)
        total = timeline.data.total_sub_periods()
        if self.mode == "quick":
            return min(total, max(1, self.quick_phase2_sub_periods))
        return total

    def _read_coverage_mode(self, path: Path) -> str:
        payload = self.workspace.read_json(path)
        return payload.get("run", {}).get("coverage_mode", "full")

    def _read_counts(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        payload = self.workspace.read_json(path)
        return dict(payload.get("run", {}).get("counts", {}))

    def _build_result(self, phase_name: str, status: PhaseStatus, path: Path) -> PhaseRunResult:
        return PhaseRunResult(
            phase_name=phase_name,
            status=status,
            output_path=path,
            coverage_mode=self._read_coverage_mode(path),
            counts=self._read_counts(path),
        )

    def _build_skip_result(self, phase_name: str, path: Path, reason: str) -> PhaseRunResult:
        logger.info("%s：跳过（%s）", phase_name, reason)
        return PhaseRunResult(
            phase_name=phase_name,
            status="skipped",
            output_path=path,
            coverage_mode=self._read_coverage_mode(path),
            reason=reason,
            counts=self._read_counts(path),
        )

    def _backup_phase2_artifacts(self) -> None:
        self._backup_path_if_exists(self.workspace.phase2_actions_path)
        self._backup_path_if_exists(self.workspace.phase2_shard_dir)

    def _backup_path_if_exists(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = path.with_name(f"{path.name}.bak-pipeline-{timestamp}")
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = path.with_name(f"{path.name}.bak-pipeline-{timestamp}-{suffix}")
        path.rename(candidate)
        logger.info("已备份旧产物：%s -> %s", path, candidate)
        return candidate


def run_pipeline(
    character_dir: str | Path,
    *,
    mode: PipelineMode = "full",
    phase2_workers: int = 1,
    phase2_query_status: bool = False,
    quick_phase2_sub_periods: int = 2,
) -> PipelineRunResult:
    """库函数入口：执行单个人物目录的一键 pipeline。"""
    runner = PipelineRunner(
        character_dir=character_dir,
        mode=mode,
        phase2_workers=phase2_workers,
        phase2_query_status=phase2_query_status,
        quick_phase2_sub_periods=quick_phase2_sub_periods,
    )
    return runner.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行人物分析 pipeline（phase0 -> phase1 -> phase2）。")
    parser.add_argument(
        "character_dir",
        nargs="?",
        default=DEFAULT_CHARACTER_DIR,
        help="人物目录，需已包含 gist.json。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速流程验证模式：每个 phase 只跑 1-2 步样例；默认关闭（即 full）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="W",
        help="Phase 2 子时期级并发度（默认 1）。",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Phase 2 主 Agent 是否追加状态查询轮（默认关闭）。",
    )
    parser.add_argument(
        "--quick-sub-periods",
        type=int,
        default=2,
        metavar="N",
        help="quick 模式下，Phase 2 最多处理前 N 个子时期（默认 2）。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    result = run_pipeline(
        args.character_dir,
        mode="quick" if args.quick else "full",
        phase2_workers=args.workers,
        phase2_query_status=args.status,
        quick_phase2_sub_periods=args.quick_sub_periods,
    )
    logger.info("pipeline 完成：%s", result.to_dict())
