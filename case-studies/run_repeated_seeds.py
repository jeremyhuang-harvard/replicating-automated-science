"""Rerun the original pipelines with new seeds (the repeated-seed study, November 2026).

This script only *calls the original command-line entry points* with new seeds and new output
directories; it changes no prompt, temperature, parser or design. Run it from the repository
root on an Apple-silicon Mac with `mlx-lm` installed (see README):

    python3 case-studies/run_repeated_seeds.py pilot      # one draw of each stage, to time it
    python3 case-studies/run_repeated_seeds.py all        # everything (resumable, sequential)
    python3 case-studies/run_repeated_seeds.py all --k 5 --s 2   # smaller study

Stages and seeds (originals in brackets):

1. scientist designs: `assr-replication generate-designs --base-seed 20261014 + 100k`, k = 1..K
   [20261014]. Each run draws one design per scenario, retrying invalid output as the original did.
2. Qwen-designed experiments: `assr_replication.generated_cli --base-seed 2026101400 + s*10^7`,
   s = 1..S [2026101400]. The original run's four accepted execution designs (`finalized.json`) are
   copied into the new directory first, which the runner treats as already finalized (its own
   resume path), so only the subject and measurement draws change.
3. GPT-Sol-designed experiments: `assr_replication.frontier_cli full-run --base-seed
   2026101400 + s*10^7` [2026101400], same design file, default 384 max tokens.
4. Expected Parrot replication: `experiments/expected_parrot_qwen_replication.py --seed 42 + s`
   [42], s = 1..S. The three prompts are fixed; only the seed changes.

Every finished run directory gets a `provenance.json` (command, seed, start/end time, runtime).
Directories that already hold a completed run are skipped, and the frontier and generated runners
resume an interrupted run in place, so the script can be restarted at any time. Absolute paths
(the stimulus image path in Expected Parrot) are replaced by bare file names, as in the originals.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSR = ROOT / "case-studies" / "automated-social-science"
PARROT = ROOT / "case-studies" / "expected-parrot"
DATE = "2026-11-11"
MODEL = "mlx-community/Qwen3-8B-4bit"
MLX_PYTHON = Path.home() / ".local/share/uv/tools/mlx-lm/bin/python"
IMAGE = Path(os.environ.get("PARROT_IMAGE", "image_09ac93.png"))

SCIENTIST_BASE, SCIENTIST_STEP = 20261014, 100
SUBJECT_BASE, SUBJECT_STEP = 2026101400, 10_000_000
PARROT_BASE = 42
ORIGINAL_QWEN_RUN = ASSR / "outputs" / "generated-experiments-qwen3-8b-2026-10-14"
CANONICAL_DESIGNS = "outputs/canonical-designs-qwen3-8b-2026-10-14/designs"
SCIENTIST_PARENT = ASSR / "outputs" / f"scientist-design-repeats-qwen3-8b-{DATE}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run(cmd: list[str], cwd: Path, out_dir: Path, seed: int, stage: str, env: dict | None = None,
        after=None) -> None:
    print(f"[{now()}] {stage} seed={seed} -> {rel(out_dir)}", flush=True)
    started, t0 = now(), time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    runtime = time.monotonic() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{stage} seed={seed} failed with exit code {proc.returncode}")
    if after is not None:
        after()
    shown = [str(c).replace(str(Path.home()), "~") for c in cmd]
    prov = {"stage": stage, "seed": seed, "command": shown, "cwd": rel(cwd), "started_at": started,
            "completed_at": now(), "runtime_seconds": round(runtime, 1), "model": MODEL}
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"[{now()}] done {stage} seed={seed} in {runtime / 60:.1f} min", flush=True)


def scientist(k: int) -> None:
    seed = SCIENTIST_BASE + SCIENTIST_STEP * k
    out = SCIENTIST_PARENT / f"base-seed-{seed}"
    if (out / "provenance.json").exists():
        return
    if out.exists():  # interrupted mid-run: the runner refuses to overwrite, so redraw from scratch
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["uv", "run", "assr-replication", "generate-designs", "--output-dir", str(out.relative_to(ASSR)),
         "--model", MODEL, "--max-tokens", "2200", "--base-seed", str(seed)], ASSR, out, seed, "scientist-designs")


def generated(s: int) -> None:
    seed = SUBJECT_BASE + SUBJECT_STEP * s
    out = ASSR / "outputs" / f"generated-experiments-qwen3-8b-{DATE}-seed-{seed}"
    if (out / "provenance.json").exists():
        return
    for fin in sorted((ORIGINAL_QWEN_RUN / "finalizations").glob("*/finalized.json")):
        target = out / "finalizations" / fin.parent.name / "finalized.json"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fin, target)
    run([str(MLX_PYTHON), "-m", "assr_replication.generated_cli", "--initial-dir", CANONICAL_DESIGNS,
         "--output-dir", str(out.relative_to(ASSR)), "--model", MODEL, "--base-seed", str(seed)],
        ASSR, out, seed, "qwen-designed-experiments", env={"PYTHONPATH": "src"})


def frontier(s: int) -> None:
    seed = SUBJECT_BASE + SUBJECT_STEP * s
    out = ASSR / "outputs" / f"frontier-planned-qwen-subjects-{DATE}-seed-{seed}"
    if (out / "provenance.json").exists():
        return
    rel_out = str(out.relative_to(ASSR))

    def after() -> None:
        n = sum(1 for p in (out / "studies").glob("*/cells.jsonl") for line in p.read_text().splitlines()
                if line.strip())
        if n != 108:
            raise SystemExit(f"frontier run has {n} of 108 cells; rerun to resume")
        subprocess.run([str(MLX_PYTHON), "-m", "assr_replication.frontier_cli", "analyze", "--output-dir",
                        rel_out], cwd=ASSR, env={**os.environ, "PYTHONPATH": "src"}, check=True)

    run([str(MLX_PYTHON), "-m", "assr_replication.frontier_cli", "full-run", "--output-dir", rel_out,
         "--model", MODEL, "--base-seed", str(seed)], ASSR, out, seed, "gpt-sol-designed-experiments",
        env={"PYTHONPATH": "src"}, after=after)


def parrot(s: int) -> None:
    seed = PARROT_BASE + s
    out = PARROT / ".artifacts" / f"expected-parrot-qwen-replication-{DATE}-seed-{seed}"
    if (out / "provenance.json").exists():
        return
    if out.exists():
        shutil.rmtree(out)
    image = IMAGE.resolve()
    run([str(MLX_PYTHON), "experiments/expected_parrot_qwen_replication.py", "--output-dir",
         str(out.relative_to(PARROT)), "--seed", str(seed), "--image", str(image)], PARROT, out, seed,
        "expected-parrot-replication")
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    prov["command"][prov["command"].index("--image") + 1] = image.name
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    path = out / "results.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    res["image"]["path"] = image.name  # scrub the absolute local path, as in the original runs
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("pilot", "all", "scientist", "generated", "frontier", "parrot"))
    p.add_argument("--k", type=int, default=10, help="scientist design draws per scenario")
    p.add_argument("--s", type=int, default=4, help="extra subject-stage seeds per cell")
    a = p.parse_args()
    if a.phase == "pilot":
        scientist(1), parrot(1), generated(1), frontier(1)
        return 0
    if a.phase in ("all", "scientist"):
        for k in range(1, a.k + 1):
            scientist(k)
    if a.phase in ("all", "parrot"):
        for s in range(1, a.s + 1):
            parrot(s)
    if a.phase in ("all", "generated"):
        for s in range(1, a.s + 1):
            generated(s)
    if a.phase in ("all", "frontier"):
        for s in range(1, a.s + 1):
            frontier(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
