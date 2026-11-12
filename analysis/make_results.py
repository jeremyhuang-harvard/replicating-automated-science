"""Rebuild every table in `results/` from the saved run outputs.

No API key, no model, no network, standard library only:

    python3 analysis/make_results.py

Inputs (all shipped in this repository):

- case-studies/automated-social-science/outputs/   raw calls, cell records and analyses
- case-studies/automated-social-science/vendor/    the four upstream Manning et al. SCM files
- case-studies/expected-parrot/.artifacts/         raw persona answers and summaries

Vocabulary. A *lever* is a cause the scientist agent theorised and turned into a treatment
(e.g. "buyer's budget"). Each experiment cell was run exactly once, so every lever's estimated
effect rests on single stochastic model draws. The *fragility* of a lever is the smallest
number of cells (out of 27) whose deletion reverses its verdict; we search all deletions of
1, 2 and 3 cells exhaustively, so the result is exact and deterministic.
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSR = ROOT / "case-studies" / "automated-social-science"
OUT_ASSR = ASSR / "outputs"
QWEN_RUN = OUT_ASSR / "generated-experiments-qwen3-8b-2026-10-14"
SOL_RUN = OUT_ASSR / "frontier-planned-qwen-subjects-2026-10-14"
PILOT = OUT_ASSR / "frontier-pilot-2026-10-14"
PILOT_FAILED = OUT_ASSR / "frontier-pilot-2026-10-14-schema-v1-failed"
CANONICAL = OUT_ASSR / "canonical-designs-qwen3-8b-2026-10-14"
VENDOR = ASSR / "vendor" / "original_scm"
PARROT = ROOT / "case-studies" / "expected-parrot" / ".artifacts"
OUT = ROOT / "results"

MAX_DELETIONS = 3
SURVIVES = "more than 3"
QWEN_STUDIES = ["01-mug-bargaining", "02-bail-setting", "03-lawyer-interview", "04-art-auction"]
SOL_STUDIES = {  # study id -> (label, primary outcome used in the user's regression slide)
    "frontier-mug-bargaining": ("mug bargaining", "deal"),
    "frontier-bail-setting": ("bail setting", "bail_amount"),
    "frontier-lawyer-interview": ("lawyer interview", "rating"),
    "frontier-art-auction": ("art auction", "final_price"),
}
RATINGS = ["overall_appeal", "concept_clarity", "uniqueness", "purchase_intent"]
SEGMENTS = ["adventurous_foodie", "mainstream_drinker", "skeptical_health_conscious"]
PARROT_CONDITIONS = {
    "rich_personas_joint_full_context": "EP personas, one call, full context",
    "expected_parrot_like_separate_context_only_q1": "EP personas, separate calls, context only in Q1",
    "expected_parrot_like_system_personas_separate_context_only_q1":
        "EP personas as system prompt, separate calls, context only in Q1",
}

# Hand-coded (not computed): which levers name the same underlying concept across the three
# scientists. Keys are concepts; values list the lever names each scientist used for it.
# Used only for the overlap count in table 7. Edit here if you disagree with a coding.
SHARED_CONCEPTS = {
    "mug bargaining": {"buyer's budget": ("buyer's budget", "buyer's budget", "Buyer budget"),
                       "seller's minimum price": ("seller's minimum acceptable price for the mug", None,
                                                  "Seller reservation price"),
                       "seller's attachment": ("seller's feelings of love towards the mug", None,
                                               "Seller attachment")},
    "bail setting": {"criminal history": ("Defendant's criminal history", "Defendant's Criminal History",
                                          "Prior convictions"),
                     "evidence strength": (None, "Evidence Strength", "Evidence strength")},
    "lawyer interview": {"bar exam": ("whether the applicant passed the bar exam", None, "Bar status"),
                         "height (placebo)": ("the job applicant's height", None, "Candidate height"),
                         "communication clarity": (None, "Communication clarity", "Communication clarity"),
                         "legal knowledge / reasoning": (None, "Legal knowledge", "Legal reasoning performance")},
    "art auction": {"bidder valuations / budgets": ("bidder 1's maximum budget for the piece of art", None,
                                                    "Bidder A valuation")},
}


# ----------------------------------------------------------------------------- helpers

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else None


def solve(matrix: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan solve of matrix @ X = rhs (rhs may have several columns)."""
    n = len(matrix)
    aug = [list(map(float, matrix[i])) + list(map(float, rhs[i])) for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def ols_hc1(x: list[list[float]], y: list[float]) -> tuple[list[float], list[float], float]:
    """OLS coefficients, HC1 robust standard errors and R^2."""
    n, k = len(x), len(x[0])
    xtx = [[sum(r[i] * r[j] for r in x) for j in range(k)] for i in range(k)]
    identity = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    inv = solve(xtx, identity)
    xty = [sum(r[i] * v for r, v in zip(x, y)) for i in range(k)]
    beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
    resid = [v - sum(b * xi for b, xi in zip(beta, r)) for r, v in zip(x, y)]
    meat = [[sum(r[i] * r[j] * e * e for r, e in zip(x, resid)) for j in range(k)] for i in range(k)]
    tmp = [[sum(inv[i][a] * meat[a][j] for a in range(k)) for j in range(k)] for i in range(k)]
    cov = [[sum(tmp[i][a] * inv[a][j] for a in range(k)) for j in range(k)] for i in range(k)]
    se = [(cov[i][i] * n / (n - k)) ** 0.5 for i in range(k)]
    ybar = mean(y)
    r2 = 1 - sum(e * e for e in resid) / sum((v - ybar) ** 2 for v in y)
    return beta, se, r2


def level_means(rows: list[tuple[list[int], float]], lever: int, dropped: frozenset = frozenset()):
    out = []
    for level in range(3):
        vals = [v for i, (idx, v) in enumerate(rows) if idx[lever] == level and i not in dropped]
        out.append(mean(vals))
    return out


def fragility(rows, lever: int, verdict) -> int | None:
    """Smallest number of deleted cells that changes verdict(means); None if > MAX_DELETIONS."""
    base = verdict(level_means(rows, lever))
    for k in range(1, MAX_DELETIONS + 1):
        for combo in itertools.combinations(range(len(rows)), k):
            means = level_means(rows, lever, frozenset(combo))
            if None in means:
                continue
            if verdict(means) != base:
                return k
    return None


def sign(x: float) -> int:
    return (x > 1e-12) - (x < -1e-12)


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def r(x, nd=4):
    return None if x is None else round(x, nd)


# ----------------------------------------------------------------------------- case study 1: ASSR

def qwen_verdict(direction: str, m: list[float]) -> bool:
    """Does Qwen's predicted direction hold for level means m = (low, mid, high)?"""
    if direction == "increase":
        return m[2] - m[0] > 0
    if direction == "decrease":
        return m[2] - m[0] < 0
    return not (min(m[0], m[2]) <= m[1] <= max(m[0], m[2]))  # non-monotonic


def qwen_scientist_levers(run: Path = QWEN_RUN) -> list[dict]:
    """All-Qwen pipeline: Qwen theorised the levers, played the subjects and measured outcomes."""
    out = []
    for study in QWEN_STUDIES:
        design = load_json(run / "studies" / study / "finalized_design.json")["source_design"]
        cells = load_jsonl(run / "studies" / study / "cells.jsonl")
        valid = [c for c in cells if c.get("measurement_valid") and c.get("measurement") is not None]
        rows = [(c["indices"], float(c["measurement"]["value"])) for c in valid]
        x = [[1.0] + [float(i) for i in idx] for idx, _ in rows]
        beta, se, _ = ols_hc1(x, [v for _, v in rows])
        for j, cause in enumerate(design["causes"]):
            direction = cause["predicted_direction"]

            def verdict(m, direction=direction):
                return qwen_verdict(direction, m)

            means = level_means(rows, j)
            out.append({
                "study": study[3:].replace("-", " "),
                "lever": cause["name"],
                "predicted": direction,
                "n_valid_cells": len(rows),
                "mean_low": r(means[0]), "mean_mid": r(means[1]), "mean_high": r(means[2]),
                "high_minus_low": r(means[2] - means[0]),
                "prediction_holds": verdict(means),
                "ols_beta": r(beta[j + 1]), "hc1_se": r(se[j + 1]),
                "abs_t": r(abs(beta[j + 1]) / se[j + 1], 2) if se[j + 1] > 0 else None,
                "cells_to_reverse": fragility(rows, j, verdict) or SURVIVES,
            })
    return out


def sol_cells(study: str, run: Path = SOL_RUN) -> list[dict]:
    """Frontier run cells, with the one separately recovered interview measurement merged in."""
    cells = load_jsonl(run / "studies" / study / "cells.jsonl")
    rec_path = run / "studies" / study / "measurement_recoveries.jsonl"
    recovered = {x["cell_id"]: x for x in load_jsonl(rec_path)} if rec_path.exists() else {}
    for c in cells:
        if c["status"] != "complete" and c["cell_id"] in recovered:
            c["derived"] = dict(recovered[c["cell_id"]]["measurement"]["outcomes"])
            c["recovered"] = True
    return cells


def sol_scientist_levers(run: Path = SOL_RUN) -> list[dict]:
    """GPT-Sol theorised the levers (five per study); Qwen played the subjects and measured."""
    bundle = {s["study_id"]: s for s in load_json(run / "design_bundle.json")["studies"]}
    out = []
    for study, (label, outcome) in SOL_STUDIES.items():
        cells = sol_cells(study, run)
        # every cell of the original run is measured (one after recovery); in the repeated-seed
        # runs a cell whose measurement failed three times has no outcome and is left out
        rows = [(c["indices"], float(c["derived"][outcome])) for c in cells
                if c.get("derived") and c["derived"].get(outcome) is not None]
        x = [[1.0] + [float(i) for i in idx] for idx, _ in rows]
        beta, se, r2 = ols_hc1(x, [v for _, v in rows])
        for j, factor in enumerate(bundle[study]["factors"]):
            means = level_means(rows, j)
            base = sign(means[2] - means[0])
            frag = None if base == 0 else (fragility(rows, j, lambda m: sign(m[2] - m[0])) or SURVIVES)
            out.append({
                "study": label,
                "outcome": outcome,
                "lever": factor["name"],
                "scientist_prediction": factor["prediction"],
                "mean_low": r(means[0]), "mean_mid": r(means[1]), "mean_high": r(means[2]),
                "high_minus_low": r(means[2] - means[0]),
                "ols_beta": r(beta[j + 1], 6), "hc1_se": r(se[j + 1], 6), "r_squared": r(r2, 6),
                "abs_t": r(abs(beta[j + 1]) / se[j + 1], 2) if se[j + 1] > 0 else None,
                "cells_to_reverse": "zero effect" if base == 0 else frag,
            })
    return out


def fragility_summary(qwen: list[dict], sol: list[dict]) -> list[dict]:
    rows = []
    for label, levers in (("Qwen scientist + Qwen subjects", qwen), ("GPT-Sol scientist + Qwen subjects", sol)):
        scored = [lv for lv in levers if lv["cells_to_reverse"] != "zero effect"]
        rows.append({
            "pipeline": label,
            "levers": len(levers),
            "reversed_by_1_cell": sum(lv["cells_to_reverse"] == 1 for lv in scored),
            "reversed_by_2_cells": sum(lv["cells_to_reverse"] == 2 for lv in scored),
            "reversed_by_3_cells": sum(lv["cells_to_reverse"] == 3 for lv in scored),
            "survive_3_deletions": sum(lv["cells_to_reverse"] == SURVIVES for lv in scored),
            "zero_effect": len(levers) - len(scored),
            "abs_t_below_2": sum(lv["abs_t"] is not None and lv["abs_t"] < 2 for lv in levers),
        })
    return rows


def sol_study_summary() -> list[dict]:
    """The descriptive numbers quoted in the user's frontier write-up."""
    mug = [c["derived"] for c in sol_cells("frontier-mug-bargaining")]
    bail = [c["derived"] for c in sol_cells("frontier-bail-setting")]
    law = sol_cells("frontier-lawyer-interview")
    auc = [c["derived"] for c in sol_cells("frontier-art-auction")]
    prices = [d["price"] for d in mug if d["deal"]]
    return [
        {"study": "mug bargaining", "statistic": "deal rate", "value": r(mean([bool(d["deal"]) for d in mug]))},
        {"study": "mug bargaining", "statistic": "mean price given a deal", "value": r(mean(prices))},
        {"study": "bail setting", "statistic": "mean bail ($)", "value": r(mean([d["bail_amount"] for d in bail]), 2)},
        {"study": "lawyer interview", "statistic": "hire rate", "value": r(mean([bool(c["derived"]["hire"]) for c in law]))},
        {"study": "lawyer interview", "statistic": "mean rating (0-100)", "value": r(mean([c["derived"]["rating"] for c in law]))},
        {"study": "lawyer interview", "statistic": "cells needing the recovery call", "value": sum(bool(c.get("recovered")) for c in law)},
        {"study": "art auction", "statistic": "mean final price ($)", "value": r(mean([d["final_price"] for d in auc]))},
        {"study": "art auction", "statistic": "mean allocative efficiency", "value": r(mean([d["allocative_efficiency"] for d in auc]))},
        {"study": "art auction", "statistic": "share efficiently allocated", "value": r(mean([bool(d["efficient_allocation"]) for d in auc]))},
        {"study": "art auction", "statistic": "no-sale cells", "value": sum(d["winner"] == "no sale" for d in auc)},
        {"study": "art auction", "statistic": "winners paying above their own valuation",
         "value": sum(d["winner"] != "no sale" and d["final_price"] > d["winner_valuation"] for d in auc)},
    ]


def run_ledger() -> list[dict]:
    """Every saved run, including the failed and superseded ones."""
    rows = []
    report = load_json(CANONICAL / "validation_report.json")
    attempts = load_jsonl(CANONICAL / "raw_calls.jsonl")
    rows.append({"case_study": "automated-social-science", "run": CANONICAL.name,
                 "what": "Qwen proposes one design per scenario",
                 "units": "4 scenarios", "succeeded": len(list((CANONICAL / "designs").glob("*.json"))),
                 "failed": 0, "model_calls": len(attempts),
                 "minutes": "",
                 "note": f"{sum(len(x['attempts']) == 1 and x['valid'] for x in report['scenarios'])} of "
                         f"{len(report['scenarios'])} designs valid on the first attempt"})
    fin_calls = sum(len(load_jsonl(p)) for p in (QWEN_RUN / "finalizations").glob("*/raw_calls.jsonl"))
    cells = [c for s in QWEN_STUDIES for c in load_jsonl(QWEN_RUN / "studies" / s / "cells.jsonl")]
    calls = sum(len(load_jsonl(QWEN_RUN / "studies" / s / "raw_calls.jsonl")) for s in QWEN_STUDIES)
    rows.append({"case_study": "automated-social-science", "run": QWEN_RUN.name,
                 "what": "Qwen-designed factorials run with Qwen subjects",
                 "units": f"{len(cells)} cells", "succeeded": sum(bool(c.get("measurement_valid")) for c in cells),
                 "failed": sum(not c.get("measurement_valid") for c in cells), "model_calls": calls + fin_calls,
                 "minutes": r(load_json(QWEN_RUN / "manifest.json")["runtime_seconds"] / 60, 1),
                 "note": f"{fin_calls - 4} finalization calls rejected before 4 were accepted"})
    for run, what in ((PILOT_FAILED, "frontier pilot, measurement parser v1"),
                      (PILOT, "frontier pilot after parser fix"),
                      (SOL_RUN, "GPT-Sol-designed studies run with Qwen subjects")):
        cells = [c for p in sorted((run / "studies").glob("*/cells.jsonl")) for c in load_jsonl(p)]
        calls = sum(len(load_jsonl(p)) for p in (run / "studies").glob("*/raw_calls.jsonl"))
        note = ""
        if run == PILOT_FAILED:
            note = "every interaction finished; all 4 measurements rejected by the v1 parser"
        if run == SOL_RUN:
            note = "1 interview measurement failed 3 times; recovered by a separate logged call"
        rows.append({"case_study": "automated-social-science", "run": run.name, "what": what,
                     "units": f"{len(cells)} cells", "succeeded": sum(c["status"] == "complete" for c in cells),
                     "failed": sum(c["status"] != "complete" for c in cells), "model_calls": calls,
                     "minutes": r(sum(c["runtime_seconds"] for c in cells) / 60, 1), "note": note})
    for run, what in (("lasagna-cola-mlx-12", "first Qwen persona panel"),
                      ("lasagna-cola-mlx-12-unanchored", "same panel after the midpoint fix")):
        resp = load_json(PARROT / run / "responses.json")
        ratings = [v for x in resp for k, v in x["answers"].items() if isinstance(v, int)]
        rows.append({"case_study": "expected-parrot", "run": run, "what": what, "units": f"{len(resp)} personas",
                     "succeeded": sum(x.get("error") is None for x in resp),
                     "failed": sum(x.get("error") is not None for x in resp), "model_calls": len(resp),
                     "minutes": "", "note": f"{sum(v == 3 for v in ratings)} of {len(ratings)} ratings at the midpoint 3"})
    res = load_json(PARROT / "expected-parrot-qwen-replication-v2" / "results.json")
    n_calls = sum(len(p["raw_outputs"]) for c in PARROT_CONDITIONS for p in res[c]["responses"])
    rows.append({"case_study": "expected-parrot", "run": "expected-parrot-qwen-replication-v2",
                 "what": "EP personas under three elicitation conditions", "units": "3 x 12 personas",
                 "succeeded": sum(len(res[c]["responses"]) for c in PARROT_CONDITIONS), "failed": 0,
                 "model_calls": n_calls, "minutes": "", "note": "v1 directory holds the first two conditions, identical answers"})
    return rows


def scientist_lever_sets() -> list[dict]:
    """Which levers each scientist chose for the same four scenario seeds."""
    paper = {}
    for path in sorted(VENDOR.glob("*.json")):
        scm = json.loads(json.loads(path.read_text(encoding="utf-8")))["args"]
        scenario = scm["scenario_description"]
        paper[scenario] = scm["variables"][1:]  # first variable is the outcome
    qwen = {}
    for path in sorted((CANONICAL / "designs").glob("*.json")):
        d = load_json(path)
        qwen[d["scenario"]] = [c["name"] for c in d["causes"]]
    sol = {s["scenario"]: [f["name"] for f in s["factors"]]
           for s in load_json(SOL_RUN / "design_bundle.json")["studies"]}
    labels = {"two people bargaining over a mug": "mug bargaining",
              "a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud":
                  "bail setting",
              "A person is interviewing for a job as a lawyer": "lawyer interview",
              "3 bidders participating in an auction for a piece of art starting at fifty dollars": "art auction"}
    rows = []
    for scenario, label in labels.items():
        concepts = SHARED_CONCEPTS[label]
        rows.append({
            "scenario": label,
            "manning_et_al_gpt4": "; ".join(paper[scenario]),
            "qwen": "; ".join(qwen[scenario]),
            "gpt_sol": "; ".join(sol[scenario]),
            "levers_total": len(paper[scenario]) + len(qwen[scenario]) + len(sol[scenario]),
            "concepts_named_by_all_three": sum(all(v is not None for v in names) for names in concepts.values()),
            "concepts_named_by_two": sum(sum(v is not None for v in names) == 2 for names in concepts.values()),
        })
    return rows


def seed_replays() -> list[dict]:
    """Fixed seeds make a rerun replay the same draw: the variance is hidden, not removed."""
    rows = []
    for study in ["frontier-mug-bargaining", "frontier-lawyer-interview", "frontier-art-auction",
                  "frontier-bail-setting"]:
        pilot = load_jsonl(PILOT / "studies" / study / "cells.jsonl")[0]
        full = next(c for c in load_jsonl(SOL_RUN / "studies" / study / "cells.jsonl")
                    if c["cell_id"] == pilot["cell_id"])
        same = sum(a["call_hash"] == b["call_hash"] for a, b in zip(pilot["transcript"], full["transcript"]))
        rows.append({"comparison": f"pilot vs full run, {pilot['cell_id']}",
                     "units": "subject turns", "identical": same, "total": len(pilot["transcript"]),
                     "note": "design for this study was edited between runs" if same == 0 else ""})
    v1 = load_json(PARROT / "expected-parrot-qwen-replication" / "results.json")
    v2 = load_json(PARROT / "expected-parrot-qwen-replication-v2" / "results.json")
    for cond in list(PARROT_CONDITIONS)[:2]:
        a, b = v1[cond]["responses"], v2[cond]["responses"]
        rows.append({"comparison": f"Expected Parrot replication v1 vs v2, {cond}", "units": "persona raw outputs",
                     "identical": sum(x["raw_outputs"] == y["raw_outputs"] for x, y in zip(a, b)),
                     "total": len(a), "note": ""})
    return rows


# ----------------------------------------------------------------------------- repeated seeds (Nov 2026)
#
# The runs above drew every cell and every scientist design once. In November 2026 we reran the
# original code with new seeds (case-studies/run_repeated_seeds.py): ten more scientist designs per
# scenario, four more draws of every Qwen-designed and GPT-Sol-designed cell, and four more
# Expected Parrot panels per elicitation condition. Draw 0 below is always the original run.
# Only run directories with a provenance.json (written when a run finished) are read.

REPEAT_DATE = "2026-11-11"
SCIENTIST_REPEATS = OUT_ASSR / f"scientist-design-repeats-qwen3-8b-{REPEAT_DATE}"
SCENARIO_LABELS = {"two people bargaining over a mug": "mug bargaining",
                   "a judge is setting bail for a criminal defendant who committed 50,000 dollars in tax fraud":
                       "bail setting",
                   "A person is interviewing for a job as a lawyer": "lawyer interview",
                   "3 bidders participating in an auction for a piece of art starting at fifty dollars":
                       "art auction"}


def finished(paths) -> list[Path]:
    return sorted(p for p in paths if (p / "provenance.json").exists())


def qwen_subject_runs() -> list[Path]:
    return [QWEN_RUN] + finished(OUT_ASSR.glob(f"generated-experiments-qwen3-8b-{REPEAT_DATE}-seed-*"))


def sol_subject_runs() -> list[Path]:
    return [SOL_RUN] + finished(OUT_ASSR.glob(f"frontier-planned-qwen-subjects-{REPEAT_DATE}-seed-*"))


def scientist_runs() -> list[Path]:
    return [CANONICAL] + finished(SCIENTIST_REPEATS.glob("base-seed-*"))


def parrot_runs() -> list[Path]:
    return [PARROT / "expected-parrot-qwen-replication-v2"] + finished(
        PARROT.glob(f"expected-parrot-qwen-replication-{REPEAT_DATE}-seed-*"))


def run_seed(run: Path) -> str:
    if (run / "provenance.json").exists():
        return str(load_json(run / "provenance.json")["seed"])
    if (run / "manifest.json").exists() and "base_seed" in load_json(run / "manifest.json"):
        return str(load_json(run / "manifest.json")["base_seed"])
    if (run / "prompt_config_snapshot.json").exists():
        return str(load_json(run / "prompt_config_snapshot.json")["base_seed"])
    if (run / "results.json").exists():
        return str(load_json(run / "results.json")["seed"])
    return str(load_json(run / "manifest.json")["seeds_used"][0])


def normalise(name: str) -> str:
    """Deterministic lever-name normalisation: lowercase, curly quotes straightened, spaces collapsed."""
    return " ".join(name.replace("\u2019", "'").lower().strip().split())


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def mean_pairwise(sets: list, fn):
    pairs = list(itertools.combinations(range(len(sets)), 2))
    return mean([fn(sets[i], sets[j]) for i, j in pairs])


# Hand-coded (not computed), written after reading every lever name Qwen produced across the
# eleven design draws: normalised lever name -> concept. Names not listed are their own concept.
# Used only for the "concept" columns of section 2; the name-based columns need no coding.
# Merges are deliberately conservative: only names that describe the same manipulated quantity.
QWEN_DRAW_CONCEPTS: dict[str, dict[str, str]] = {
    "mug bargaining": {
        "initial offer": "opening offer",
        "buyer's budget": "buyer's budget / willingness to pay",
        "bob's maximum willingness to pay": "buyer's budget / willingness to pay",
        "buyer's maximum willingness to pay": "buyer's budget / willingness to pay",
        "buyer's budget constraint": "buyer's budget / willingness to pay",
        "alice's reservation price": "seller's reservation price",
        "seller's reservation price": "seller's reservation price",
        "mug quality": "mug's value / quality",
        "mug's true value": "mug's value / quality",
        "mug's perceived value": "mug's value / quality",
    },
    "bail setting": {
        "severity of tax fraud": "offense severity",
        "severity of crime": "offense severity",
        "defendant's criminal history": "criminal history",
        "prior criminal record": "criminal history",
        "evidence of financial stability": "defendant's finances",
        "evidence of financial capability": "defendant's finances",
        "defendant's financial status": "defendant's finances",
        "flight risk": "flight risk",
        "defendant's flight risk": "flight risk",
        "public safety": "public safety",
        "public safety concerns": "public safety",
    },
    "lawyer interview": {
        "communication clarity": "communication",
        "communication skills": "communication",
    },
    "art auction": {
        "bidder risk tolerance": "bidder risk tolerance",
        "bidder's risk tolerance": "bidder risk tolerance",
        "bidding strategy": "bidding strategy",
        "bidder strategy": "bidding strategy",
        "bidder b's aggressiveness": "bidding strategy",
        "bidder experience": "bidder experience",
        "previous auction experience": "bidder experience",
        "bidder's confidence in art value": "bidder confidence",
        "bidder confidence": "bidder confidence",
        "bidder a's confidence": "bidder confidence",
        "market demand signal": "market information",
        "market observation": "market information",
        "bidder's budget": "bidder budget",
        "bidder c's budget limit": "bidder budget",
    },
}


def concept_of(scenario: str, name: str) -> str:
    return QWEN_DRAW_CONCEPTS[scenario].get(normalise(name), normalise(name))


def scientist_draws() -> list[dict]:
    """One row per (scientist draw, scenario): the accepted design's levers, or the failure."""
    rows = []
    for draw, run in enumerate(scientist_runs()):
        report = load_json(run / "validation_report.json")
        for s in report["scenarios"]:
            label = SCENARIO_LABELS[s["scenario"]]
            design = load_json(run / s["design_file"]) if s["valid"] else None
            causes = design["causes"] if design else []
            rows.append({
                "draw": draw, "base_seed": run_seed(run), "scenario": label,
                "attempts": len(s["attempts"]),
                "attempt_statuses": "; ".join(a["status"] for a in s["attempts"]),
                "valid": s["valid"],
                "levers": "; ".join(c["name"] for c in causes),
                "predicted_directions": "; ".join(c["predicted_direction"] for c in causes),
                "concepts": "; ".join(concept_of(label, c["name"]) for c in causes),
                "outcome": design["outcome"]["name"] if design else "",
            })
    return rows


def scientist_repeat_summary(draws: list[dict]) -> list[dict]:
    rows = []
    for label in SCENARIO_LABELS.values():
        mine = [d for d in draws if d["scenario"] == label]
        valid = [d for d in mine if d["valid"]]
        names = [frozenset(normalise(x) for x in d["levers"].split("; ")) for d in valid]
        concepts = [frozenset(d["concepts"].split("; ")) for d in valid]
        pairs = len(names) * (len(names) - 1) // 2
        modal = max(names.count(s) for s in set(names))
        original = concepts[0] if valid and valid[0]["draw"] == 0 else frozenset()
        rows.append({
            "scenario": label, "draws": len(mine), "valid_designs": len(valid),
            "valid_first_attempt": sum(d["valid"] and d["attempts"] == 1 for d in mine),
            "invalid_attempts": sum(d["attempt_statuses"].count("malformed_json") +
                                    d["attempt_statuses"].count("schema_invalid") +
                                    d["attempt_statuses"].count("provider_error") for d in mine),
            "distinct_lever_sets_names": len(set(names)),
            "identical_name_set_pairs": sum(a == b for a, b in itertools.combinations(names, 2)),
            "pairs": pairs,
            "modal_name_set_draws": modal,
            "new_draws_with_original_name_set": sum(n == names[0] for n in names[1:]) if valid[0]["draw"] == 0 else "",
            "mean_jaccard_names": r(mean_pairwise(names, jaccard), 3),
            "distinct_lever_sets_concepts": len(set(concepts)),
            "identical_concept_set_pairs": sum(a == b for a, b in itertools.combinations(concepts, 2)),
            "mean_jaccard_concepts": r(mean_pairwise(concepts, jaccard), 3),
            "distinct_names": len(set().union(*names)),
            "distinct_concepts": len(set().union(*concepts)),
            "distinct_outcome_names": len({normalise(d["outcome"]) for d in valid}),
            "original_concepts_redrawn": "; ".join(
                f"{c} {sum(c in s for s in concepts[1:])}/{len(concepts) - 1}" for c in sorted(original)),
        })
    return rows


def scientist_lever_frequency(draws: list[dict]) -> list[dict]:
    rows = []
    for label in SCENARIO_LABELS.values():
        valid = [d for d in draws if d["scenario"] == label and d["valid"]]
        seen = defaultdict(lambda: {"draws": set(), "names": set(), "directions": defaultdict(int)})
        for d in valid:
            for name, direction in zip(d["levers"].split("; "), d["predicted_directions"].split("; ")):
                c = seen[concept_of(label, name)]
                c["draws"].add(d["draw"])
                c["names"].add(normalise(name))
                c["directions"][direction] += 1
        for concept, c in sorted(seen.items(), key=lambda kv: (-len(kv[1]["draws"]), kv[0])):
            rows.append({"scenario": label, "concept": concept, "draws_with_lever": len(c["draws"]),
                         "valid_draws": len(valid), "in_original_draw": 0 in c["draws"],
                         "names_used": "; ".join(sorted(c["names"])),
                         "predicted_directions": "; ".join(f"{k} {v}" for k, v in sorted(c["directions"].items()))})
    return rows


def verdict_label(lv: dict, kind: str):
    """Qwen levers: does Qwen's own predicted direction hold? GPT-Sol levers: sign of high - low."""
    if kind == "qwen":
        return lv["prediction_holds"]
    return sign(lv["high_minus_low"])


def repeat_levers(kind: str) -> list[dict]:
    """One row per (lever, subject draw), the same statistics as tables 2-3 for every draw."""
    runs = qwen_subject_runs() if kind == "qwen" else sol_subject_runs()
    rows = []
    for draw, run in enumerate(runs):
        levers = qwen_scientist_levers(run) if kind == "qwen" else sol_scientist_levers(run)
        if kind == "qwen":
            n_cells = {s[3:].replace("-", " "): sum(bool(c.get("measurement_valid"))
                                                     for c in load_jsonl(run / "studies" / s / "cells.jsonl"))
                       for s in QWEN_STUDIES}
        else:
            n_cells = {label: sum(bool(c.get("derived")) for c in sol_cells(study, run))
                       for study, (label, _) in SOL_STUDIES.items()}
        for lv in levers:
            rows.append({"pipeline": kind, "draw": draw, "base_seed": run_seed(run), "study": lv["study"],
                         "lever": lv["lever"], "predicted": lv.get("predicted", ""),
                         "measured_cells": n_cells[lv["study"]],
                         "mean_low": lv["mean_low"], "mean_mid": lv["mean_mid"], "mean_high": lv["mean_high"],
                         "high_minus_low": lv["high_minus_low"], "verdict": verdict_label(lv, kind),
                         "ols_beta": lv["ols_beta"], "hc1_se": lv["hc1_se"], "abs_t": lv["abs_t"],
                         "cells_to_reverse": lv["cells_to_reverse"]})
    return rows


def frag_class(k) -> str:
    if k == "zero effect":
        return "zero effect"
    return "1-2 cells" if k in (1, 2) else ("3 cells" if k == 3 else "survives 3")


def lever_stability(per_draw: list[dict], kind: str) -> list[dict]:
    """Per lever: does the verdict, sign and |t| > 2 repeat across independent subject draws?"""
    out = []
    keys = list(dict.fromkeys((x["study"], x["lever"]) for x in per_draw))
    for study, lever in keys:
        draws = sorted((x for x in per_draw if (x["study"], x["lever"]) == (study, lever)), key=lambda x: x["draw"])
        first, new = draws[0], draws[1:]
        verdicts = [d["verdict"] for d in draws]
        betas = [d["ols_beta"] for d in draws]
        row = {"pipeline": kind, "study": study, "lever": lever, "draws": len(draws),
               "original_verdict": first["verdict"],
               "original_cells_to_reverse": first["cells_to_reverse"],
               "original_fragility": frag_class(first["cells_to_reverse"]),
               "new_draws_same_verdict": sum(d["verdict"] == first["verdict"] for d in new),
               "new_draws": len(new)}
        if kind == "qwen":
            row["draws_prediction_holds"] = sum(bool(v) for v in verdicts)
            row.update(draws_positive="", draws_negative="", draws_zero="")
        else:
            row["draws_prediction_holds"] = ""
            row["draws_positive"] = sum(v == 1 for v in verdicts)
            row["draws_negative"] = sum(v == -1 for v in verdicts)
            row["draws_zero"] = sum(v == 0 for v in verdicts)
        # pooled verdict: the same verdict rule applied to level means averaged over all draws
        pooled = [mean([d[f"mean_{lvl}"] for d in draws]) for lvl in ("low", "mid", "high")]
        pv = qwen_verdict(first["predicted"], pooled) if kind == "qwen" else sign(round(pooled[2] - pooled[0], 9))
        row.update({
            "pooled_high_minus_low": r(pooled[2] - pooled[0]),
            "pooled_verdict": pv,
            "draws_matching_pooled_verdict": sum(d["verdict"] == pv for d in draws),
            "draws_beta_positive": sum(b > 0 for b in betas),
            "draws_beta_negative": sum(b < 0 for b in betas),
            "draws_abs_t_above_2": sum(d["abs_t"] is not None and d["abs_t"] > 2 for d in draws),
            "contrast_by_draw": "; ".join(fmt(d["high_minus_low"]) for d in draws),
            "abs_t_by_draw": "; ".join(fmt(d["abs_t"]) for d in draws),
        })
        out.append(row)
    return out


def stability_summary(qwen_stab: list[dict], sol_stab: list[dict], qwen_draws: list[dict]) -> list[dict]:
    rows = []
    for label, stab in (("Qwen scientist + Qwen subjects", qwen_stab), ("GPT-Sol scientist + Qwen subjects", sol_stab)):
        scored = [s for s in stab if s["original_fragility"] != "zero effect"]
        fragile = [s for s in scored if s["original_fragility"] == "1-2 cells"]
        three = [s for s in scored if s["original_fragility"] == "3 cells"]
        robust = [s for s in scored if s["original_fragility"] == "survives 3"]
        flips = lambda group: sum(s["new_draws_same_verdict"] < s["new_draws"] for s in group)  # noqa: E731
        n = stab[0]["draws"] if stab else 0
        rows.append({
            "pipeline": label, "levers": len(stab), "draws_per_cell": n,
            "verdict_same_in_all_draws": sum(s["new_draws_same_verdict"] == s["new_draws"] for s in stab),
            "verdict_flips_in_some_new_draw": sum(s["new_draws_same_verdict"] < s["new_draws"] for s in stab),
            "fragile_1_2_cells": len(fragile), "fragile_1_2_cells_that_flip": flips(fragile),
            "fragile_3_cells": len(three), "fragile_3_cells_that_flip": flips(three),
            "robust": len(robust), "robust_that_flip": flips(robust),
            "original_differs_from_pooled": sum(s["original_verdict"] != s["pooled_verdict"] for s in stab),
            "single_draws_matching_pooled": sum(s["draws_matching_pooled_verdict"] for s in stab),
            "single_draws": sum(s["draws"] for s in stab),
            "abs_t_above_2_in_every_draw": sum(s["draws_abs_t_above_2"] == s["draws"] for s in stab),
            "abs_t_above_2_in_no_draw": sum(s["draws_abs_t_above_2"] == 0 for s in stab),
            "abs_t_above_2_in_some_draws": sum(0 < s["draws_abs_t_above_2"] < s["draws"] for s in stab),
            "beta_sign_same_in_every_draw": sum(s["draws_beta_positive"] in (0, s["draws"]) and
                                                s["draws_beta_negative"] in (0, s["draws"]) for s in stab),
        })
    held = defaultdict(int)
    for x in qwen_draws:
        held[x["draw"]] += bool(x["verdict"])
    rows[0]["predictions_held_by_draw"] = "; ".join(str(held[d]) for d in sorted(held))
    rows[1]["predictions_held_by_draw"] = "(not scored)"
    return rows


def cell_redraw_identity() -> list[dict]:
    """How often does a new seed reproduce the original cell's transcript or outcome exactly?"""
    rows = []
    for kind, runs in (("Qwen designs", qwen_subject_runs()), ("GPT-Sol designs", sol_subject_runs())):
        studies = QWEN_STUDIES if kind == "Qwen designs" else list(SOL_STUDIES)
        for study in studies:
            def load(run):
                if kind == "Qwen designs":
                    return {c["cell_id"]: ([t["text"] for t in c["transcript"]],
                                           (c.get("measurement") or {}).get("value") if c.get("measurement_valid") else None)
                            for c in load_jsonl(run / "studies" / study / "cells.jsonl")}
                outcome = SOL_STUDIES[study][1]
                return {c["cell_id"]: ([t["text"] for t in c["transcript"]], (c.get("derived") or {}).get(outcome))
                        for c in sol_cells(study, run)}
            orig = load(runs[0])
            same_t = same_v = n = 0
            for run in runs[1:]:
                new = load(run)
                for cid, (text, value) in orig.items():
                    n += 1
                    same_t += new[cid][0] == text
                    same_v += value is not None and new[cid][1] == value
            label = study[3:].replace("-", " ") if kind == "Qwen designs" else SOL_STUDIES[study][0]
            rows.append({"designs": kind, "study": label, "cell_draw_pairs": n,
                         "identical_transcript": same_t, "identical_outcome": same_v,
                         "share_identical_transcript": r(same_t / n, 3) if n else None,
                         "share_identical_outcome": r(same_v / n, 3) if n else None})
    return rows


def parrot_seed_agreement() -> list[dict]:
    """Same personas, same items, same prompt, different seed; versus a prompt change."""
    runs = parrot_runs()
    res = [load_json(run / "results.json") for run in runs]
    keys = list(PARROT_CONDITIONS)

    def answers(i, cond):
        return {x["name"]: x["answers"] for x in res[i][cond]["responses"]}

    def compare(i, ci, j, cj):
        a, b = answers(i, ci), answers(j, cj)
        diffs = [a[n][k] - b[n][k] for n in a for k in RATINGS]
        return sum(d == 0 for d in diffs), sum(abs(d) >= 2 for d in diffs), len(diffs)

    rows = []
    seed_pairs = list(itertools.combinations(range(len(runs)), 2))
    for cond in keys:
        stats = [compare(i, cond, j, cond) for i, j in seed_pairs]
        rows.append({"comparison": f"same prompt, different seed: {PARROT_CONDITIONS[cond]}",
                     "pairs": len(stats), "ratings_per_pair": stats[0][2],
                     "identical_mean": r(mean([s[0] for s in stats]), 1),
                     "identical_min": min(s[0] for s in stats), "identical_max": max(s[0] for s in stats),
                     "differ_by_2_mean": r(mean([s[1] for s in stats]), 1)})
    for ci, cj, label in ((keys[1], keys[2], "persona in user prompt vs system prompt (separate calls)"),
                          (keys[0], keys[1], "one call with full context vs separate calls")):
        stats = [compare(i, ci, j, cj) for i in range(len(runs)) for j in range(len(runs))]
        rows.append({"comparison": f"prompt changed, any seeds: {label}",
                     "pairs": len(stats), "ratings_per_pair": stats[0][2],
                     "identical_mean": r(mean([s[0] for s in stats]), 1),
                     "identical_min": min(s[0] for s in stats), "identical_max": max(s[0] for s in stats),
                     "differ_by_2_mean": r(mean([s[1] for s in stats]), 1)})
    for cond in keys:
        per_item = [[answers(i, cond)[n][k] for i in range(len(runs))]
                    for n in answers(0, cond) for k in RATINGS]
        rows.append({"comparison": f"all {len(runs)} seeds give the same rating: {PARROT_CONDITIONS[cond]}",
                     "pairs": "", "ratings_per_pair": len(per_item),
                     "identical_mean": sum(len(set(v)) == 1 for v in per_item), "identical_min": "",
                     "identical_max": "", "differ_by_2_mean": sum(max(v) - min(v) >= 2 for v in per_item)})
    return rows


def parrot_seed_segments() -> list[dict]:
    rows = []
    for run in parrot_runs():
        res = load_json(run / "results.json")
        for cond, label in PARROT_CONDITIONS.items():
            seg = res[cond]["summary"]["by_segment"]
            pol = res[cond]["summary"]["polarization"]
            rows.append({"seed": res["seed"], "condition": label,
                         "appeal_adventurous": seg["adventurous_foodie"]["overall_appeal"],
                         "appeal_mainstream": seg["mainstream_drinker"]["overall_appeal"],
                         "appeal_skeptical": seg["skeptical_health_conscious"]["overall_appeal"],
                         "appeal_range": pol["appeal_segment_range"],
                         "purchase_adventurous": seg["adventurous_foodie"]["purchase_intent"],
                         "purchase_mainstream": seg["mainstream_drinker"]["purchase_intent"],
                         "purchase_skeptical": seg["skeptical_health_conscious"]["purchase_intent"],
                         "purchase_range": pol["purchase_segment_range"]})
    return rows


def repeat_ledger() -> list[dict]:
    """The repeated-seed runs, with runtimes from provenance.json."""
    rows = []
    for run in scientist_runs()[1:]:
        rep = load_json(run / "validation_report.json")
        prov = load_json(run / "provenance.json")
        rows.append({"stage": "Qwen scientist designs", "run": str(run.relative_to(OUT_ASSR)), "seed": prov["seed"],
                     "units": f"{len(rep['scenarios'])} scenarios", "succeeded": rep["summary"]["valid_designs"],
                     "failed": len(rep["scenarios"]) - rep["summary"]["valid_designs"],
                     "model_calls": rep["summary"]["total_attempts"],
                     "minutes": r(prov["runtime_seconds"] / 60, 1)})
    for kind, runs in (("Qwen-designed cells", qwen_subject_runs()[1:]), ("GPT-Sol-designed cells", sol_subject_runs()[1:])):
        for run in runs:
            prov = load_json(run / "provenance.json")
            cells = [c for p in sorted((run / "studies").glob("*/cells.jsonl")) for c in load_jsonl(p)]
            ok = [c for c in cells if (c.get("measurement_valid") if kind.startswith("Qwen") else c["status"] == "complete")]
            calls = sum(len(load_jsonl(p)) for p in (run / "studies").glob("*/raw_calls.jsonl"))
            rows.append({"stage": kind, "run": run.name, "seed": prov["seed"], "units": f"{len(cells)} cells",
                         "succeeded": len(ok), "failed": len(cells) - len(ok), "model_calls": calls,
                         "minutes": r(prov["runtime_seconds"] / 60, 1)})
    for run in parrot_runs()[1:]:
        prov = load_json(run / "provenance.json")
        res = load_json(run / "results.json")
        rows.append({"stage": "Expected Parrot panels", "run": run.name, "seed": prov["seed"],
                     "units": "3 x 12 personas",
                     "succeeded": sum(len(res[c]["responses"]) for c in PARROT_CONDITIONS), "failed": 0,
                     "model_calls": sum(len(p["raw_outputs"]) for c in PARROT_CONDITIONS for p in res[c]["responses"]),
                     "minutes": r(prov["runtime_seconds"] / 60, 1)})
    return rows


# ----------------------------------------------------------------------------- case study 2: Expected Parrot

def parrot_segments() -> list[dict]:
    rows = []
    for run, label in (("lasagna-cola-mlx-12", "minimal personas, first prompt (anchored)"),
                       ("lasagna-cola-mlx-12-unanchored", "minimal personas, midpoint fix (original baseline)")):
        resp = load_json(PARROT / run / "responses.json")
        seg = defaultdict(list)
        for x in resp:
            seg[x["segment"]].append(x["answers"])
        row = {"condition": label}
        for key, name in (("appeal", "appeal"), ("purchase_intent", "purchase")):
            means = {s: mean([a[key] for a in seg[s]]) for s in SEGMENTS}
            row.update({f"{name}_{s.split('_')[0]}": r(means[s], 2) for s in SEGMENTS})
            row[f"{name}_range"] = r(max(means.values()) - min(means.values()), 2)
        rows.append(row)
    res = load_json(PARROT / "expected-parrot-qwen-replication-v2" / "results.json")
    for cond, label in PARROT_CONDITIONS.items():
        seg = defaultdict(list)
        for x in res[cond]["responses"]:
            seg[x["segment"]].append(x["answers"])
        row = {"condition": label}
        for key, name in (("overall_appeal", "appeal"), ("purchase_intent", "purchase")):
            means = {s: mean([a[key] for a in seg[s]]) for s in SEGMENTS}
            row.update({f"{name}_{s.split('_')[0]}": r(means[s], 2) for s in SEGMENTS})
            row[f"{name}_range"] = r(max(means.values()) - min(means.values()), 2)
        rows.append(row)
    target = res["public_expected_parrot_target"]["by_segment"]
    row = {"condition": "public Expected Parrot run (cited, Claude + image)"}
    for key, name in (("overall_appeal", "appeal"), ("purchase_intent", "purchase")):
        means = {s: target[s][key] for s in SEGMENTS}
        row.update({f"{name}_{s.split('_')[0]}": means[s] for s in SEGMENTS})
        row[f"{name}_range"] = r(max(means.values()) - min(means.values()), 2)
    rows.append(row)
    return rows


def parrot_overall() -> list[dict]:
    res = load_json(PARROT / "expected-parrot-qwen-replication-v2" / "results.json")
    cited = {"overall_appeal": 2.25, "concept_clarity": 3.83, "uniqueness": 3.00, "purchase_intent": 3.00}
    rows = []
    for key in RATINGS:
        vals = [x["answers"][key] for x in res["expected_parrot_like_separate_context_only_q1"]["responses"]]
        rows.append({"measure": key, "qwen_separate_calls": r(mean(vals), 2),
                     "expected_parrot_cited": cited[key]})
    return rows


def parrot_agreement() -> list[dict]:
    """Same 12 personas, same items: how often does a persona give the same rating twice?"""
    res = load_json(PARROT / "expected-parrot-qwen-replication-v2" / "results.json")
    answers = {c: {x["name"]: x["answers"] for x in res[c]["responses"]} for c in PARROT_CONDITIONS}
    names = list(answers["rich_personas_joint_full_context"])
    keys = list(PARROT_CONDITIONS)
    pairs = [(keys[1], keys[2], "separate calls: persona in user prompt vs system prompt"),
             (keys[0], keys[1], "one call with full context vs separate calls")]
    rows = []
    for a, b, label in pairs:
        diffs = [answers[a][n][k] - answers[b][n][k] for n in names for k in RATINGS]
        flips = [n for n in names
                 if (answers[a][n]["purchase_intent"] >= 4) != (answers[b][n]["purchase_intent"] >= 4)]
        rows.append({"comparison": label, "ratings_compared": len(diffs),
                     "identical": sum(d == 0 for d in diffs),
                     "differ_by_2_or_more": sum(abs(d) >= 2 for d in diffs),
                     "purchase_top2_flips": len(flips), "personas_flipping": "; ".join(flips)})
    return rows


# ----------------------------------------------------------------------------- output

def fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:,.4g}" if abs(x) < 1000 else f"{x:,.0f}"
    return str(x)


def frag_text(v) -> str:
    return str(v)


def svg_fragility(qwen: list[dict], sol: list[dict]) -> str:
    """One horizontal bar per lever: how many single-draw cells must go to reverse it."""
    levers = [("Qwen", lv) for lv in qwen] + [("GPT-Sol", lv) for lv in sol]
    w, left, bar_h, gap, top = 920, 370, 14, 4, 70
    h = top + len(levers) * (bar_h + gap) + 70
    unit = (w - left - 140) / 4
    colors = {1: "#c0392b", 2: "#e67e22", 3: "#f1c40f", SURVIVES: "#2e86c1", "zero effect": "#9aa5b1"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="11">',
             f'<rect width="{w}" height="{h}" fill="white"/>',
             '<text x="20" y="26" font-size="16" font-weight="bold">How many single-draw cells must be '
             'deleted to reverse each theorised lever?</text>',
             '<text x="20" y="46" font-size="12" fill="#444">27 cells per study, one model draw each. '
             'Exhaustive search over deleting 1, 2 or 3 cells.</text>']
    for i, (who, lv) in enumerate(levers):
        y = top + i * (bar_h + gap)
        k = lv["cells_to_reverse"]
        length = 4 if k == SURVIVES else (0.15 if k == "zero effect" else k)
        label = f'{who} · {lv["study"]} · {lv["lever"]}'
        parts.append(f'<text x="{left - 8}" y="{y + 11}" text-anchor="end">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{length * unit:.1f}" height="{bar_h}" '
                     f'fill="{colors[k]}"/>')
        text = {SURVIVES: "survives 3 deletions", "zero effect": "zero effect"}.get(
            k, f"{k} cell" + ("s" if k != 1 else ""))
        parts.append(f'<text x="{left + length * unit + 6:.1f}" y="{y + 11}">{text}</text>')
    y = h - 40
    parts.append(f'<text x="20" y="{y}" font-size="11" fill="#555">Qwen levers: reversal of the '
                 'prediction verdict. GPT-Sol levers: reversal of the sign of the high-minus-low '
                 'contrast on the primary outcome.</text>')
    parts.append(f'<text x="20" y="{y + 16}" font-size="11" fill="#555">Red = one deleted draw reverses '
                 'the lever. Blue = survives every deletion of up to three cells.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def svg_seed_verdicts(qwen_draws: list[dict], sol_draws: list[dict]) -> str:
    """One row per lever, one square per independent draw of its 27 cells: does the verdict repeat?"""
    rows = defaultdict(list)
    for who, draws in (("Qwen", qwen_draws), ("GPT-Sol", sol_draws)):
        for d in draws:
            rows[(who, d["study"], d["lever"])].append(d)
    keys = list(rows)
    n = max(len(v) for v in rows.values())
    w, left, sq, gap, top = 980, 370, 14, 4, 92
    col = sq + 8
    h = top + len(keys) * (sq + gap) + 96
    frag_x = left + n * col + 24
    pale = {"same": "#aed6f1", "flip": "#f5b7b1", "ref": "#d5d8dc"}
    dark = {"same": "#2e86c1", "flip": "#c0392b", "ref": "#566573"}
    frag_col = {1: "#c0392b", 2: "#e67e22", 3: "#b7950b", SURVIVES: "#2e86c1", "zero effect": "#7f8c8d"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="11">',
             f'<rect width="{w}" height="{h}" fill="white"/>',
             '<text x="20" y="26" font-size="16" font-weight="bold">Does a theorised lever give the same '
             f'answer on {n} independent draws of its 27 cells?</text>',
             '<text x="20" y="46" font-size="12" fill="#444">Same designs, same prompts, new seeds for every '
             'subject and measurement call. Draw 1 is the original run.</text>']
    for j in range(n):
        parts.append(f'<text x="{left + j * col + sq / 2:.1f}" y="{top - 8}" text-anchor="middle" '
                     f'fill="#444">{j + 1}</text>')
    parts.append(f'<text x="{left + (n * col) / 2:.1f}" y="{top - 24}" text-anchor="middle" '
                 'fill="#444">draw</text>')
    parts.append(f'<text x="{frag_x}" y="{top - 8}" fill="#444">original deletion test</text>')
    for i, key in enumerate(keys):
        who, study, lever = key
        y = top + i * (sq + gap)
        draws = sorted(rows[key], key=lambda d: d["draw"])
        ref = draws[0]["verdict"]
        parts.append(f'<text x="{left - 8}" y="{y + 11}" text-anchor="end">{who} · {study} · {lever}</text>')
        for d in draws:
            kind = "ref" if d["draw"] == 0 else ("same" if d["verdict"] == ref else "flip")
            strong = d["abs_t"] is not None and d["abs_t"] > 2
            fill = (dark if strong else pale)[kind]
            parts.append(f'<rect x="{left + d["draw"] * col}" y="{y}" width="{sq}" height="{sq}" fill="{fill}"/>')
            glyph = ("✓" if d["verdict"] else "✗") if who == "Qwen" else {1: "+", -1: "−", 0: "0"}[d["verdict"]]
            parts.append(f'<text x="{left + d["draw"] * col + sq / 2:.1f}" y="{y + 11}" text-anchor="middle" '
                         f'font-size="10" fill="{"white" if strong else "#222"}">{glyph}</text>')
        k = draws[0]["cells_to_reverse"]
        text = {SURVIVES: "survives 3 deletions", "zero effect": "zero effect"}.get(
            k, f"reversed by {k} cell" + ("s" if k != 1 else ""))
        flips = sum(d["verdict"] != ref for d in draws[1:])
        parts.append(f'<text x="{frag_x}" y="{y + 11}" fill="{frag_col[k]}">{text}</text>')
        parts.append(f'<text x="{frag_x + 140}" y="{y + 11}" fill="#333">'
                     f'{"same verdict in every draw" if flips == 0 else f"{flips} of {len(draws) - 1} new draws differ"}'
                     '</text>')
    y = h - 72
    legend = [("ref", True, "original draw, |t| > 2"), ("ref", False, "original draw, |t| ≤ 2"),
              ("same", True, "same verdict, |t| > 2"), ("same", False, "same verdict, |t| ≤ 2"),
              ("flip", True, "different verdict, |t| > 2"), ("flip", False, "different verdict, |t| ≤ 2")]
    for j, (kind, strong, label) in enumerate(legend):
        x = 20 + (j % 3) * 230
        yy = y + (j // 3) * 18
        parts.append(f'<rect x="{x}" y="{yy - 10}" width="12" height="12" fill="{(dark if strong else pale)[kind]}"/>')
        parts.append(f'<text x="{x + 18}" y="{yy}">{label}</text>')
    parts.append(f'<text x="20" y="{y + 42}" font-size="11" fill="#555">Verdict: Qwen levers, ✓ if '
                 "Qwen's own predicted direction holds; GPT-Sol levers, sign of the high-minus-low contrast "
                 'on the primary outcome.</text>')
    parts.append(f'<text x="20" y="{y + 58}" font-size="11" fill="#555">|t| = |OLS coefficient| / HC1 '
                 'robust SE from outcome ~ 1 + level indices, fitted separately in each draw.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def markdown_repeats(qstab, sstab, summary, sci_draws, sci_summary, sci_freq, p_agree, p_segs, redraw) -> list[str]:
    q, s = summary
    L = ["## 1. Repeated subject draws: does each lever's verdict come back?", "",
         f"The original runs drew each experiment cell once. We reran every cell of both designs "
         f"{q['draws_per_cell'] - 1} more times with new seeds (same designs, prompts, temperatures and "
         f"parsers), so each lever now has {q['draws_per_cell']} independent estimates, each from its own "
         "27 cells. A lever *differs* in a draw when its verdict is not the original one: for Qwen's levers, "
         "whether Qwen's predicted direction holds; for GPT-Sol's, the sign of the high-minus-low contrast. "
         "*Original deletion test* is the fragility class from section 4. *Pooled* applies the same verdict "
         "rule to the level means averaged over all draws, our best estimate of what this model does on "
         "average.", "",
         "| | Qwen designs, Qwen subjects | GPT-Sol designs, Qwen subjects |", "|---|---:|---:|"]
    spec = [("levers theorised", "levers"), ("independent draws of each 27-cell study", "draws_per_cell"),
            ("same verdict in every draw", "verdict_same_in_all_draws"),
            ("verdict differs in at least one new draw", "verdict_flips_in_some_new_draw"),
            ("original verdict differs from the pooled verdict", "original_differs_from_pooled"),
            ("deletion-fragile in the original (1-2 cells): levers", "fragile_1_2_cells"),
            ("… of which differ in a new draw", "fragile_1_2_cells_that_flip"),
            ("reversed by 3 deletions in the original: levers", "fragile_3_cells"),
            ("… of which differ in a new draw", "fragile_3_cells_that_flip"),
            ("survived 3 deletions in the original: levers", "robust"),
            ("… of which differ in a new draw", "robust_that_flip"),
            ("\\|t\\| > 2 in every draw", "abs_t_above_2_in_every_draw"),
            ("\\|t\\| > 2 in some draws but not others", "abs_t_above_2_in_some_draws"),
            ("\\|t\\| > 2 in no draw", "abs_t_above_2_in_no_draw"),
            ("OLS coefficient has the same sign in every draw", "beta_sign_same_in_every_draw")]
    for label, key in spec:
        L.append(f"| {label} | {q[key]} | {s[key]} |")
    L.append(f"| single-draw verdicts matching the pooled verdict | {q['single_draws_matching_pooled']} of "
             f"{q['single_draws']} | {s['single_draws_matching_pooled']} of {s['single_draws']} |")
    L += ["", f"Qwen's predictions holding, by draw (original first): **{q['predictions_held_by_draw']}** "
          f"(of {q['levers']}).", "", "![verdicts across seeds](lever_verdicts_across_seeds.svg)", "",
          "Levers Qwen theorised (verdict = Qwen's predicted direction holds):", "",
          "| study | lever | original deletion test | prediction holds in | new draws same as original | "
          "pooled: holds? | high − low by draw | \\|t\\| by draw |", "|---|---|---|---:|---:|---|---|---|"]
    for x in qstab:
        L.append(f"| {x['study']} | {x['lever']} | {x['original_cells_to_reverse']} | "
                 f"{x['draws_prediction_holds']} of {x['draws']} | {x['new_draws_same_verdict']} of {x['new_draws']} | "
                 f"{fmt(x['pooled_verdict'])} | {x['contrast_by_draw']} | {x['abs_t_by_draw']} |")
    L += ["", "Levers GPT-Sol theorised (verdict = sign of high − low on the primary outcome):", "",
          "| study | lever | original deletion test | + / − / 0 draws | new draws same as original | "
          "pooled sign | high − low by draw | \\|t\\| by draw |", "|---|---|---|---|---:|---:|---|---|"]
    for x in sstab:
        L.append(f"| {x['study']} | {x['lever']} | {x['original_cells_to_reverse']} | "
                 f"{x['draws_positive']} / {x['draws_negative']} / {x['draws_zero']} | "
                 f"{x['new_draws_same_verdict']} of {x['new_draws']} | {'+−0'[[1, -1, 0].index(x['pooled_verdict'])]} | "
                 f"{x['contrast_by_draw']} | {x['abs_t_by_draw']} |")
    n_sci = sci_summary[0]["draws"]
    L += ["", "How much does a new seed change a cell at all? Each cell's new draws compared with its original "
          "draw (primary outcome as above; Qwen designs: the measured value):", "",
          "| designs | study | cell × new-draw pairs | identical transcript | identical outcome |",
          "|---|---|---:|---:|---:|"]
    for x in redraw:
        L.append(f"| {x['designs']} | {x['study']} | {x['cell_draw_pairs']} | {x['identical_transcript']} "
                 f"({fmt(x['share_identical_transcript'])}) | {x['identical_outcome']} "
                 f"({fmt(x['share_identical_outcome'])}) |")
    L += ["", "Per-draw statistics for every lever: `seed_repeat_levers_by_draw.csv`.", "",
          "## 2. Repeated scientist draws: which levers does Qwen propose?", "",
          f"Qwen's design step reran {n_sci - 1} more times per scenario with new seeds (same prompt, "
          f"temperature 0.3, up to three attempts for invalid output, as in the original), giving {n_sci} "
          "draws including the original. *Names* compares lever names after lowercasing and collapsing "
          "whitespace; *concepts* applies a hand-coded map written after reading all the names "
          "(`QWEN_DRAW_CONCEPTS` in the script). Jaccard = |A ∩ B| / |A ∪ B| for two draws' lever sets, "
          "averaged over all pairs.", "",
          "| scenario | draws | valid | invalid attempts | distinct sets (names) | most common set drawn | new draws repeating the original set | identical pairs (names) | "
          "mean Jaccard (names) | distinct sets (concepts) | identical pairs (concepts) | mean Jaccard (concepts) | "
          "distinct concepts | distinct outcome names | original levers redrawn |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in sci_summary:
        L.append(f"| {x['scenario']} | {x['draws']} | {x['valid_designs']} | {x['invalid_attempts']} | "
                 f"{x['distinct_lever_sets_names']} | {x['modal_name_set_draws']} times | "
                 f"{x['new_draws_with_original_name_set']} of {x['draws'] - 1} | "
                 f"{x['identical_name_set_pairs']} of {x['pairs']} | "
                 f"{x['mean_jaccard_names']} | {x['distinct_lever_sets_concepts']} | "
                 f"{x['identical_concept_set_pairs']} of {x['pairs']} | {x['mean_jaccard_concepts']} | "
                 f"{x['distinct_concepts']} | {x['distinct_outcome_names']} | {x['original_concepts_redrawn']} |")
    orig = [x for x in sci_freq if x["in_original_draw"]]
    redrawn = sum(x["draws_with_lever"] - 1 for x in orig)
    L += ["", f"Across the {len(orig)} levers of the original Qwen design, a new draw proposed the same concept "
          f"{redrawn} of {len(orig) * (n_sci - 1)} times; "
          f"{sum(x['new_draws_with_original_name_set'] for x in sci_summary)} of {len(sci_summary) * (n_sci - 1)} new "
          "draws repeated the original lever set exactly. Levers drawn with both an *increase* and a *decrease* "
          "prediction: " + ("; ".join(f"{x['scenario']}: {x['concept']} ({x['predicted_directions']})" for x in sci_freq
                                     if "increase" in x["predicted_directions"] and "decrease" in x["predicted_directions"])
                            or "none") + ".",
          "", "How often each lever concept appears:", "",
          "| scenario | concept | draws with it | in original draw | names used | predicted direction |",
          "|---|---|---:|---|---|---|"]
    for x in sci_freq:
        L.append(f"| {x['scenario']} | {x['concept']} | {x['draws_with_lever']} of {x['valid_draws']} | "
                 f"{fmt(x['in_original_draw'])} | {x['names_used']} | {x['predicted_directions']} |")
    L += ["", "Every draw's levers and outcome: `scientist_repeat_draws.csv`.", "",
          "## 3. Expected Parrot: seed noise versus a prompt change", "",
          "The replication script reran with new seeds and the three prompts unchanged. *Same prompt, "
          "different seed* compares one condition across every pair of seeds. *Prompt changed* compares two "
          "conditions across every combination of seeds; in the original single run this was the only "
          "comparison available, and it mixed the prompt change with a seed change. 48 ratings = 12 personas "
          "× 4 items. In the last three rows the columns count persona-items instead: how many of the 48 got "
          "the same rating from every seed, and how many spanned 2 or more points across seeds.", "",
          "| comparison | pairs | ratings | identical (mean) | min | max | differ by ≥ 2 (mean) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for x in p_agree:
        L.append(f"| {x['comparison']} | {x['pairs']} | {x['ratings_per_pair']} | {x['identical_mean']} | "
                 f"{x['identical_min']} | {x['identical_max']} | {x['differ_by_2_mean']} |")
    L += ["", "Segment means by seed (the original run is seed 42):", "",
          "| seed | condition | appeal: adventurous / mainstream / health | appeal range | "
          "purchase: adventurous / mainstream / health | purchase range |", "|---:|---|---|---:|---|---:|"]
    for x in sorted(p_segs, key=lambda x: (list(PARROT_CONDITIONS.values()).index(x["condition"]), x["seed"])):
        L.append(f"| {x['seed']} | {x['condition']} | {x['appeal_adventurous']} / {x['appeal_mainstream']} / "
                 f"{x['appeal_skeptical']} | {x['appeal_range']} | {x['purchase_adventurous']} / "
                 f"{x['purchase_mainstream']} / {x['purchase_skeptical']} | {x['purchase_range']} |")
    L.append("")
    return L


def markdown(qwen, sol, summary, sol_desc, ledger, sets, replays, segs, overall, agree, rep_qwen_draws,
             rep_sol_draws, rep_qwen_stab, rep_sol_stab, rep_summary, sci_draws, sci_summary, sci_freq,
             parrot_seed_agree, parrot_seed_segs, rep_ledger, cell_redraw) -> str:
    L = ["# Results", "",
         "Generated by `python3 analysis/make_results.py` from the saved runs under `case-studies/`. "
         "Do not edit by hand. Nothing here is human data. Sections 1-3 compare repeated draws made with "
         "new seeds in November 2026; sections 4-11 describe the original October 2026 runs, in which every "
         "cell and every scientist design was drawn once.", ""]
    L += markdown_repeats(rep_qwen_stab, rep_sol_stab, rep_summary, sci_draws, sci_summary, sci_freq,
                          parrot_seed_agree, parrot_seed_segs, cell_redraw)
    L += ["## 4. How fragile is each theorised lever? (original single draws)", "",
         "Each experiment cell was run once. *Cells to reverse* = the smallest number of the 27 cells "
         "whose deletion changes the lever's verdict (exhaustive search up to 3). For Qwen's levers the "
         "verdict is whether Qwen's own prediction holds; for GPT-Sol's levers it is the sign of the "
         "high-minus-low contrast (a change to exactly zero counts). `|t|` = |OLS "
         "coefficient| / HC1 robust SE from `outcome ~ 1 + level indices`.", "",
         "| pipeline | levers | reversed by 1 cell | by 2 | by 3 | survive 3 deletions | zero effect | "
         "|t| < 2 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        L.append(f"| {s['pipeline']} | {s['levers']} | {s['reversed_by_1_cell']} | {s['reversed_by_2_cells']} | "
                 f"{s['reversed_by_3_cells']} | {s['survive_3_deletions']} | {s['zero_effect']} | "
                 f"{s['abs_t_below_2']} |")
    L += ["", "![fragility](lever_fragility.svg)", "",
          "## 5. Levers Qwen theorised, tested on Qwen's own subjects", "",
          "| study | lever | Qwen predicted | low / mid / high mean | high − low | prediction holds | "
          "cells to reverse | OLS β | HC1 SE | \\|t\\| |", "|---|---|---|---|---:|---|---:|---:|---:|---:|"]
    for x in qwen:
        L.append(f"| {x['study']} | {x['lever']} | {x['predicted']} | {fmt(x['mean_low'])} / {fmt(x['mean_mid'])}"
                 f" / {fmt(x['mean_high'])} | {fmt(x['high_minus_low'])} | {fmt(x['prediction_holds'])} | "
                 f"{frag_text(x['cells_to_reverse'])} | {fmt(x['ols_beta'])} | {fmt(x['hc1_se'])} | "
                 f"{x['abs_t']} |")
    held = sum(x["prediction_holds"] for x in qwen)
    L += ["", f"Predictions holding: **{held} of {len(qwen)}**.", "",
          "## 6. Levers GPT-Sol theorised, tested on Qwen subjects", "",
          "Primary outcome per study as in the original regression slide (mug: deal 0/1; bail: $; "
          "interview: 0–100 rating; auction: final price $). Ordered coding of nominal factors "
          "(outside option, procedure, information regime) is a shorthand.", "",
          "| study | lever | low / mid / high mean | OLS β | HC1 SE | \\|t\\| | cells to reverse sign |",
          "|---|---|---|---:|---:|---:|---:|"]
    for x in sol:
        L.append(f"| {x['study']} | {x['lever']} | {fmt(x['mean_low'])} / {fmt(x['mean_mid'])} / "
                 f"{fmt(x['mean_high'])} | {fmt(x['ols_beta'])} | {fmt(x['hc1_se'])} | {x['abs_t']} | "
                 f"{frag_text(x['cells_to_reverse'])} |")
    L += ["", "Descriptive statistics quoted in the frontier write-up:", "",
          "| study | statistic | value |", "|---|---|---:|"]
    L += [f"| {x['study']} | {x['statistic']} | {fmt(x['value'])} |" for x in sol_desc]
    L += ["", "## 7. Same scenario, three scientists, different levers", "",
          "Lever names are read from the data files. The overlap counts use a hand-coded concept map "
          "(`SHARED_CONCEPTS` in the script). GPT-Sol had seen summaries of the paper, so its overlap "
          "with Manning et al. is not independent.", "",
          "| scenario | Manning et al. (GPT-4 + authors) | Qwen | GPT-Sol | levers | concepts named by all three |"
          " by exactly two |", "|---|---|---|---|---:|---:|---:|"]
    for x in sets:
        L.append(f"| {x['scenario']} | {x['manning_et_al_gpt4']} | {x['qwen']} | {x['gpt_sol']} | "
                 f"{x['levers_total']} | {x['concepts_named_by_all_three']} | {x['concepts_named_by_two']} |")
    L += ["", "## 8. Expected Parrot: segment means by elicitation", "",
          "| condition | appeal: adventurous / mainstream / health | appeal range | "
          "purchase: adventurous / mainstream / health | purchase range |", "|---|---|---:|---|---:|"]
    for x in segs:
        L.append(f"| {x['condition']} | {x['appeal_adventurous']} / {x['appeal_mainstream']} / "
                 f"{x['appeal_skeptical']} | {x['appeal_range']} | {x['purchase_adventurous']} / "
                 f"{x['purchase_mainstream']} / {x['purchase_skeptical']} | {x['purchase_range']} |")
    L += ["", "Overall means, separate-call condition vs the public run:", "",
          "| measure | Qwen separate calls | Expected Parrot (cited) |", "|---|---:|---:|"]
    L += [f"| {x['measure']} | {x['qwen_separate_calls']} | {x['expected_parrot_cited']} |" for x in overall]
    L += ["", "## 9. Expected Parrot: does a persona give the same answer twice?", "",
          "Same 12 personas and the same four 1–5 items. The first pair differs in where the persona sits "
          "(user vs system prompt, plus one line asking for a decisive answer) and in the seed; the second "
          "in whether all questions share one call with full context.", "",
          "| comparison | ratings | identical | differ by ≥ 2 | personas crossing purchase ≥ 4 | who |",
          "|---|---:|---:|---:|---:|---|"]
    for x in agree:
        L.append(f"| {x['comparison']} | {x['ratings_compared']} | {x['identical']} | {x['differ_by_2_or_more']} | "
                 f"{x['purchase_top2_flips']} | {x['personas_flipping']} |")
    L += ["", "## 10. Fixed seeds replay the same draw", "",
          "| comparison | units | identical | total | note |", "|---|---|---:|---:|---|"]
    L += [f"| {x['comparison']} | {x['units']} | {x['identical']} | {x['total']} | {x['note']} |" for x in replays]
    L += ["", "## 11. Every saved run, including failures", "",
          "Runtime is recorded model time (manifest or summed cell runtimes) where the run saved it.", "",
          "| case study | run | what | units | succeeded | failed | model calls | minutes | note |",
          "|---|---|---|---|---:|---:|---:|---:|---|"]
    L += [f"| {x['case_study']} | `{x['run']}` | {x['what']} | {x['units']} | {x['succeeded']} | {x['failed']} | "
          f"{x['model_calls']} | {x['minutes']} | {x['note']} |" for x in ledger]
    L += ["", "The repeated-seed runs (November 2026; minutes from each run's `provenance.json`):", "",
          "| stage | run | seed | units | succeeded | failed | model calls | minutes |",
          "|---|---|---:|---|---:|---:|---:|---:|"]
    L += [f"| {x['stage']} | `{x['run']}` | {x['seed']} | {x['units']} | {x['succeeded']} | {x['failed']} | "
          f"{x['model_calls']} | {x['minutes']} |" for x in rep_ledger]
    total = sum(x["minutes"] for x in rep_ledger)
    L += ["", f"Total recorded model time for the repeated-seed runs: {total:.0f} minutes ({total / 60:.1f} hours)."]
    return "\n".join(L) + "\n"


def build() -> dict:
    qwen, sol = qwen_scientist_levers(), sol_scientist_levers()
    return {
        "qwen": qwen, "sol": sol, "summary": fragility_summary(qwen, sol),
        "sol_desc": sol_study_summary(), "ledger": run_ledger(), "sets": scientist_lever_sets(),
        "replays": seed_replays(), "segs": parrot_segments(), "overall": parrot_overall(),
        "agree": parrot_agreement(), **build_repeats(),
    }


def build_repeats() -> dict:
    rq, rs = repeat_levers("qwen"), repeat_levers("sol")
    qstab, sstab = lever_stability(rq, "qwen"), lever_stability(rs, "sol")
    draws = scientist_draws()
    return {"rep_qwen_draws": rq, "rep_sol_draws": rs, "rep_qwen_stab": qstab, "rep_sol_stab": sstab,
            "rep_summary": stability_summary(qstab, sstab, rq), "sci_draws": draws,
            "sci_summary": scientist_repeat_summary(draws), "sci_freq": scientist_lever_frequency(draws),
            "parrot_seed_agree": parrot_seed_agreement(), "parrot_seed_segs": parrot_seed_segments(),
            "rep_ledger": repeat_ledger(), "cell_redraw": cell_redraw_identity()}


def main() -> None:
    t = build()
    write_csv("lever_fragility_summary.csv", t["summary"])
    write_csv("qwen_scientist_levers.csv", t["qwen"])
    write_csv("sol_scientist_levers.csv", t["sol"])
    write_csv("sol_study_summary.csv", t["sol_desc"])
    write_csv("scientist_lever_sets.csv", t["sets"])
    write_csv("seed_replays.csv", t["replays"])
    write_csv("parrot_segment_means.csv", t["segs"])
    write_csv("parrot_overall_means.csv", t["overall"])
    write_csv("parrot_answer_agreement.csv", t["agree"])
    write_csv("run_ledger.csv", t["ledger"])
    write_csv("seed_repeat_summary.csv", t["rep_summary"])
    write_csv("seed_repeat_lever_stability.csv", t["rep_qwen_stab"] + t["rep_sol_stab"])
    write_csv("seed_repeat_levers_by_draw.csv", t["rep_qwen_draws"] + t["rep_sol_draws"])
    write_csv("scientist_repeat_draws.csv", t["sci_draws"])
    write_csv("scientist_repeat_summary.csv", t["sci_summary"])
    write_csv("scientist_repeat_lever_frequency.csv", t["sci_freq"])
    write_csv("parrot_seed_agreement.csv", t["parrot_seed_agree"])
    write_csv("parrot_seed_segment_means.csv", t["parrot_seed_segs"])
    write_csv("repeat_run_ledger.csv", t["rep_ledger"])
    write_csv("seed_repeat_cell_identity.csv", t["cell_redraw"])
    (OUT / "lever_fragility.svg").write_text(svg_fragility(t["qwen"], t["sol"]), encoding="utf-8")
    (OUT / "lever_verdicts_across_seeds.svg").write_text(
        svg_seed_verdicts(t["rep_qwen_draws"], t["rep_sol_draws"]), encoding="utf-8")
    (OUT / "RESULTS.md").write_text(markdown(**t), encoding="utf-8")
    print(f"wrote results to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
