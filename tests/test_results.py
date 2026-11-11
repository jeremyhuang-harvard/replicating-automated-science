"""The saved outputs must reproduce the numbers we report."""

import csv
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_results", ROOT / "analysis" / "make_results.py")
assert spec is not None and spec.loader is not None
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)

TABLES = mr.build()


def by_lever(rows):
    return {(r["study"], r["lever"]): r for r in rows}


def test_headline_fragility_counts():
    qwen, sol = TABLES["summary"]
    assert (qwen["levers"], qwen["reversed_by_1_cell"], qwen["reversed_by_2_cells"],
            qwen["reversed_by_3_cells"], qwen["survive_3_deletions"], qwen["abs_t_below_2"]) == (12, 2, 2, 0, 8, 6)
    assert (sol["levers"], sol["reversed_by_1_cell"], sol["reversed_by_2_cells"], sol["reversed_by_3_cells"],
            sol["survive_3_deletions"], sol["zero_effect"], sol["abs_t_below_2"]) == (20, 0, 2, 4, 13, 1, 9)


def test_qwen_nine_of_twelve_and_which_levers_are_fragile():
    levers = by_lever(TABLES["qwen"])
    assert sum(r["prediction_holds"] for r in TABLES["qwen"]) == 9
    assert levers[("bail setting", "Severity of Tax Fraud")]["cells_to_reverse"] == 2
    assert levers[("art auction", "Bidding Strategy")]["cells_to_reverse"] == 2
    assert levers[("mug bargaining", "mediator's presence")]["cells_to_reverse"] == 1
    assert levers[("art auction", "Bidder Risk Tolerance")]["cells_to_reverse"] == 1


def test_qwen_contrasts_match_the_runner_analysis_files():
    for study in mr.QWEN_STUDIES:
        saved = json.loads((mr.QWEN_RUN / "studies" / study / "analysis.json").read_text())
        mine = {r["lever"]: r for r in TABLES["qwen"] if r["study"] == study[3:].replace("-", " ")}
        for lever, contrast in saved["contrasts"].items():
            assert round(contrast, 4) == mine[lever]["high_minus_low"]
            assert round(saved["ols"]["coefficients"][lever], 4) == mine[lever]["ols_beta"]
            assert saved["direction_agreement"][lever] == mine[lever]["prediction_holds"]


def test_sol_regressions_match_the_users_saved_table():
    path = ROOT / "case-studies/automated-social-science/docs/runs/2026-10-15-sol-qwen-regressions.csv"
    saved = list(csv.DictReader(path.open()))
    mine = TABLES["sol"]
    assert len(saved) == len(mine) == 20
    for s, m in zip(saved, mine):
        assert round(float(s["beta"]), 6) == m["ols_beta"]
        assert round(float(s["hc1_se"]), 6) == m["hc1_se"]
        assert round(float(s["r_squared"]), 6) == m["r_squared"]


def test_sol_descriptives_match_the_writeup():
    stats = {(r["study"], r["statistic"]): r["value"] for r in TABLES["sol_desc"]}
    assert stats[("mug bargaining", "deal rate")] == 0.5185
    assert stats[("mug bargaining", "mean price given a deal")] == 14.3571
    assert stats[("bail setting", "mean bail ($)")] == 22222.22
    assert stats[("lawyer interview", "hire rate")] == 0.6296
    assert stats[("lawyer interview", "mean rating (0-100)")] == 72.4074
    assert stats[("art auction", "mean final price ($)")] == 171.8519
    assert stats[("art auction", "mean allocative efficiency")] == 0.537
    assert stats[("art auction", "share efficiently allocated")] == 0.4074
    assert stats[("art auction", "no-sale cells")] == 10
    assert stats[("art auction", "winners paying above their own valuation")] == 3


def test_parrot_segment_means_match_the_saved_comparison():
    saved = json.loads((mr.PARROT / "expected-parrot-qwen-replication-v2" / "comparison.json").read_text())
    rows = {r["condition"]: r for r in TABLES["segs"]}
    for key, label in mr.PARROT_CONDITIONS.items():
        seg = saved[key]["by_segment"]
        assert rows[label]["appeal_adventurous"] == seg["adventurous_foodie"]["overall_appeal"]
        assert rows[label]["purchase_mainstream"] == seg["mainstream_drinker"]["purchase_intent"]
        assert rows[label]["appeal_range"] == saved[key]["polarization"]["appeal_segment_range"]
        assert rows[label]["purchase_range"] == saved[key]["polarization"]["purchase_segment_range"]
    anchored = rows["minimal personas, first prompt (anchored)"]
    assert anchored["appeal_range"] == anchored["purchase_range"] == 0.0


def test_parrot_answer_agreement():
    a, b = TABLES["agree"]
    assert (a["ratings_compared"], a["identical"], a["differ_by_2_or_more"]) == (48, 34, 2)
    assert (b["ratings_compared"], b["identical"], b["differ_by_2_or_more"]) == (48, 22, 11)


def test_fixed_seeds_replay_identically():
    rows = TABLES["replays"]
    assert [(r["identical"], r["total"]) for r in rows] == [(8, 8), (6, 6), (8, 8), (0, 8), (12, 12), (12, 12)]


def test_committed_results_match_a_fresh_build():
    assert mr.markdown(**TABLES) == (ROOT / "results" / "RESULTS.md").read_text()
    assert mr.svg_fragility(TABLES["qwen"], TABLES["sol"]) == (ROOT / "results" / "lever_fragility.svg").read_text()


def test_vendored_upstream_files_are_byte_identical():
    provenance = (mr.VENDOR / "PROVENANCE.md").read_text()
    for digest, name in re.findall(r"^([0-9a-f]{64})  (.+\.json)$", provenance, flags=re.M):
        assert hashlib.sha256((mr.VENDOR / name).read_bytes()).hexdigest() == digest


def test_no_credentials_or_machine_paths_in_tracked_files():
    pattern = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{32,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                         r"/Users/[A-Za-z]|/home/[a-z]+/|BEGIN [A-Z ]*PRIVATE KEY")
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split("\0")
    for name in filter(None, tracked):
        path = ROOT / name
        if path.suffix in {".png", ".pdf"} or path.name == "test_results.py":
            continue
        assert not pattern.search(path.read_text(errors="ignore")), name
