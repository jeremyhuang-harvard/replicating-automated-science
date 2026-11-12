"""The saved outputs must reproduce the numbers we report."""

import csv
import hashlib
import itertools
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


# ----------------------------------------------------------------------------- repeated seeds (Nov 2026)

def test_every_repeated_seed_run_is_complete():
    assert len(mr.scientist_runs()) == 11
    assert len(mr.qwen_subject_runs()) == len(mr.sol_subject_runs()) == len(mr.parrot_runs()) == 5
    for run in mr.qwen_subject_runs() + mr.sol_subject_runs():
        cells = [(p.parent.name, c["cell_id"]) for p in (run / "studies").glob("*/cells.jsonl") for c in mr.load_jsonl(p)]
        assert len(cells) == len(set(cells)) == 108, run.name
    for run in mr.scientist_runs():
        assert len(mr.load_json(run / "validation_report.json")["scenarios"]) == 4


def test_repeated_runs_change_only_the_seed():
    """Same execution designs and design bundle as the originals; only seeds differ."""
    for run in mr.qwen_subject_runs()[1:]:
        for study in mr.QWEN_STUDIES:
            assert (run / "finalizations" / study / "finalized.json").read_bytes() == \
                (mr.QWEN_RUN / "finalizations" / study / "finalized.json").read_bytes()
            assert mr.load_json(run / "studies" / study / "finalized_design.json") == \
                mr.load_json(mr.QWEN_RUN / "studies" / study / "finalized_design.json")
        assert not list(run.glob("finalizations/*/raw_calls.jsonl"))  # no compiler calls were made
    original = mr.load_json(mr.SOL_RUN / "prompt_config_snapshot.json")
    for run in mr.sol_subject_runs()[1:]:
        assert mr.load_json(run / "design_bundle.json") == mr.load_json(mr.SOL_RUN / "design_bundle.json")
        snap = mr.load_json(run / "prompt_config_snapshot.json")
        assert {k: v for k, v in snap.items() if k != "base_seed"} == \
            {k: v for k, v in original.items() if k != "base_seed"}
    v2 = mr.load_json(mr.parrot_runs()[0] / "results.json")
    for run in mr.parrot_runs()[1:]:
        res = mr.load_json(run / "results.json")
        assert res["image"] == v2["image"] and res["model"] == v2["model"]
        for cond in mr.PARROT_CONDITIONS:
            assert [x["persona"] for x in res[cond]["responses"]] == [x["persona"] for x in v2[cond]["responses"]]


def test_no_two_runs_of_a_stage_share_a_seed():
    def seeds(paths):
        return {mr.json.dumps(c["request"]["metadata"]["seed"]) for p in paths for c in mr.load_jsonl(p)}
    for runs in (mr.qwen_subject_runs(), mr.sol_subject_runs()):
        sets = [seeds(sorted((run / "studies").glob("*/raw_calls.jsonl"))) for run in runs]
        for a, b in itertools.combinations(sets, 2):
            assert not a & b
    sets = [set(mr.load_json(run / "manifest.json")["seeds_used"]) for run in mr.scientist_runs()]
    for a, b in itertools.combinations(sets, 2):
        assert not a & b
    assert len({mr.load_json(run / "results.json")["seed"] for run in mr.parrot_runs()}) == 5


def test_every_committed_results_file_matches_a_fresh_build(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "OUT", tmp_path)
    monkeypatch.setattr(mr, "ROOT", tmp_path.parent)
    mr.main()
    fresh = sorted(p.name for p in tmp_path.iterdir())
    assert fresh == sorted(p.name for p in (ROOT / "results").iterdir())
    for name in fresh:
        assert (tmp_path / name).read_bytes() == (ROOT / "results" / name).read_bytes(), name


def test_repeated_qwen_subject_draws_headline():
    q = TABLES["rep_summary"][0]
    assert q["draws_per_cell"] == 5
    assert (q["verdict_same_in_all_draws"], q["verdict_flips_in_some_new_draw"]) == (10, 2)
    assert (q["fragile_1_2_cells"], q["fragile_1_2_cells_that_flip"], q["robust"], q["robust_that_flip"]) == (4, 1, 8, 1)
    assert (q["abs_t_above_2_in_every_draw"], q["abs_t_above_2_in_some_draws"], q["abs_t_above_2_in_no_draw"]) == (4, 3, 5)
    assert q["predictions_held_by_draw"] == "9; 10; 10; 9; 10"
    assert (q["single_draws_matching_pooled"], q["single_draws"], q["original_differs_from_pooled"]) == (58, 60, 1)
    stab = by_lever(TABLES["rep_qwen_stab"])
    # the one deletion-fragile lever whose verdict changes: mediator presence fails once, holds four times
    assert stab[("mug bargaining", "mediator's presence")]["draws_prediction_holds"] == 4
    assert stab[("bail setting", "Severity of Tax Fraud")]["draws_prediction_holds"] == 5
    assert stab[("art auction", "Bidding Strategy")]["draws_prediction_holds"] == 5
    assert stab[("art auction", "Bidder Risk Tolerance")]["draws_prediction_holds"] == 0
    assert stab[("bail setting", "Evidence Strength")]["draws_prediction_holds"] == 4


def test_repeated_scientist_draws_headline():
    rows = {r["scenario"]: r for r in TABLES["sci_summary"]}
    assert all(r["draws"] == r["valid_designs"] == 11 for r in rows.values())
    assert sum(r["invalid_attempts"] for r in rows.values()) == 5
    assert [rows[s]["distinct_lever_sets_names"] for s in rows] == [6, 9, 6, 9]
    assert [rows[s]["new_draws_with_original_name_set"] for s in rows] == [0, 2, 3, 0]
    assert [rows[s]["mean_jaccard_names"] for s in rows] == [0.236, 0.242, 0.431, 0.189]
    assert [rows[s]["mean_jaccard_concepts"] for s in rows] == [0.405, 0.415, 0.567, 0.307]
    orig = [x for x in TABLES["sci_freq"] if x["in_original_draw"]]
    assert len(orig) == 12 and sum(x["draws_with_lever"] - 1 for x in orig) == 68
    freq = {(x["scenario"], x["concept"]): x for x in TABLES["sci_freq"]}
    assert freq[("mug bargaining", "mediator's presence")]["draws_with_lever"] == 1
    assert freq[("lawyer interview", "communication")]["draws_with_lever"] == 11
    assert freq[("mug bargaining", "buyer's budget / willingness to pay")]["predicted_directions"] == "decrease 3; increase 6"


def test_repeated_parrot_seeds_separate_seed_noise_from_wording():
    rows = {r["comparison"]: r for r in TABLES["parrot_seed_agree"]}
    same = [r for k, r in rows.items() if k.startswith("same prompt")]
    assert [r["identical_mean"] for r in same] == [44.0, 43.6, 40.6]
    user_vs_system = rows["prompt changed, any seeds: persona in user prompt vs system prompt (separate calls)"]
    assert (user_vs_system["pairs"], user_vs_system["identical_mean"]) == (25, 34.6)
    joint_vs_sep = rows["prompt changed, any seeds: one call with full context vs separate calls"]
    assert joint_vs_sep["identical_mean"] == 22.0


def test_repeated_sol_subject_draws_headline():
    s = TABLES["rep_summary"][1]
    assert s["draws_per_cell"] == 5
    assert (s["verdict_same_in_all_draws"], s["verdict_flips_in_some_new_draw"]) == (15, 5)
    assert (s["fragile_1_2_cells"], s["fragile_1_2_cells_that_flip"], s["fragile_3_cells"],
            s["fragile_3_cells_that_flip"], s["robust"], s["robust_that_flip"]) == (2, 2, 4, 1, 13, 1)
    assert (s["abs_t_above_2_in_every_draw"], s["abs_t_above_2_in_some_draws"], s["abs_t_above_2_in_no_draw"]) == (2, 14, 4)
    assert (s["single_draws_matching_pooled"], s["single_draws"]) == (95, 100)
    stab = by_lever(TABLES["rep_sol_stab"])
    differ = sorted(lever for (_, lever), x in stab.items() if x["new_draws_same_verdict"] < x["new_draws"])
    assert differ == ["Bidder A valuation", "Bidder B valuation", "Candidate height", "Outside option",
                      "Seller attachment"]
    assert (stab[("mug bargaining", "Outside option")]["draws_zero"],
            stab[("mug bargaining", "Outside option")]["draws_negative"]) == (1, 4)


def test_repeated_cells_often_replay_the_same_outcome():
    rows = {(r["designs"], r["study"]): r for r in TABLES["cell_redraw"]}
    assert rows[("Qwen designs", "bail setting")]["share_identical_outcome"] == 0.833
    assert rows[("Qwen designs", "art auction")]["share_identical_transcript"] == 0.602
    assert all(r["cell_draw_pairs"] == 108 for r in rows.values())


def test_parrot_polarisation_is_steady_across_seeds():
    full = [r["purchase_range"] for r in TABLES["parrot_seed_segs"]
            if r["condition"] == "EP personas, one call, full context"]
    assert sorted(full) == [3, 3, 3, 3, 3.25]


def test_repeat_ledger_runtime():
    ledger = TABLES["rep_ledger"]
    assert len(ledger) == 22
    assert round(sum(x["minutes"] for x in ledger)) == 237
    assert sum(x["failed"] for x in ledger) == 2
    assert sum(x["model_calls"] for x in ledger) == 6709
    assert sum(x["model_calls"] for x in ledger if x["stage"] == "Qwen scientist designs") == 45
    assert sum(x["model_calls"] for x in TABLES["ledger"]) + 6709 == 8498  # README: "nearly 8,500"
