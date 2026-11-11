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
# Used only for the overlap count in table 6. Edit here if you disagree with a coding.
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

def qwen_scientist_levers() -> list[dict]:
    """All-Qwen pipeline: Qwen theorised the levers, played the subjects and measured outcomes."""
    out = []
    for study in QWEN_STUDIES:
        design = load_json(QWEN_RUN / "studies" / study / "finalized_design.json")["source_design"]
        cells = load_jsonl(QWEN_RUN / "studies" / study / "cells.jsonl")
        valid = [c for c in cells if c.get("measurement_valid") and c.get("measurement") is not None]
        rows = [(c["indices"], float(c["measurement"]["value"])) for c in valid]
        x = [[1.0] + [float(i) for i in idx] for idx, _ in rows]
        beta, se, _ = ols_hc1(x, [v for _, v in rows])
        for j, cause in enumerate(design["causes"]):
            direction = cause["predicted_direction"]

            def verdict(m, direction=direction):
                if direction == "increase":
                    return m[2] - m[0] > 0
                if direction == "decrease":
                    return m[2] - m[0] < 0
                return not (min(m[0], m[2]) <= m[1] <= max(m[0], m[2]))  # non-monotonic

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


def sol_cells(study: str) -> list[dict]:
    """Frontier run cells, with the one separately recovered interview measurement merged in."""
    cells = load_jsonl(SOL_RUN / "studies" / study / "cells.jsonl")
    rec_path = SOL_RUN / "studies" / study / "measurement_recoveries.jsonl"
    recovered = {x["cell_id"]: x for x in load_jsonl(rec_path)} if rec_path.exists() else {}
    for c in cells:
        if c["status"] != "complete" and c["cell_id"] in recovered:
            c["derived"] = dict(recovered[c["cell_id"]]["measurement"]["outcomes"])
            c["recovered"] = True
    return cells


def sol_scientist_levers() -> list[dict]:
    """GPT-Sol theorised the levers (five per study); Qwen played the subjects and measured."""
    bundle = {s["study_id"]: s for s in load_json(SOL_RUN / "design_bundle.json")["studies"]}
    out = []
    for study, (label, outcome) in SOL_STUDIES.items():
        cells = sol_cells(study)
        rows = [(c["indices"], float(c["derived"][outcome])) for c in cells]
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


def markdown(qwen, sol, summary, sol_desc, ledger, sets, replays, segs, overall, agree) -> str:
    L = ["# Results", "",
         "Generated by `python3 analysis/make_results.py` from the saved runs under `case-studies/`. "
         "Do not edit by hand. Every model output here is one stochastic draw; nothing is human data.", "",
         "## 1. How fragile is each theorised lever?", "",
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
          "## 2. Levers Qwen theorised, tested on Qwen's own subjects", "",
          "| study | lever | Qwen predicted | low / mid / high mean | high − low | prediction holds | "
          "cells to reverse | OLS β | HC1 SE | \\|t\\| |", "|---|---|---|---|---:|---|---:|---:|---:|---:|"]
    for x in qwen:
        L.append(f"| {x['study']} | {x['lever']} | {x['predicted']} | {fmt(x['mean_low'])} / {fmt(x['mean_mid'])}"
                 f" / {fmt(x['mean_high'])} | {fmt(x['high_minus_low'])} | {fmt(x['prediction_holds'])} | "
                 f"{frag_text(x['cells_to_reverse'])} | {fmt(x['ols_beta'])} | {fmt(x['hc1_se'])} | "
                 f"{x['abs_t']} |")
    held = sum(x["prediction_holds"] for x in qwen)
    L += ["", f"Predictions holding: **{held} of {len(qwen)}**.", "",
          "## 3. Levers GPT-Sol theorised, tested on Qwen subjects", "",
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
    L += ["", "## 4. Same scenario, three scientists, different levers", "",
          "Lever names are read from the data files. The overlap counts use a hand-coded concept map "
          "(`SHARED_CONCEPTS` in the script). GPT-Sol had seen summaries of the paper, so its overlap "
          "with Manning et al. is not independent.", "",
          "| scenario | Manning et al. (GPT-4 + authors) | Qwen | GPT-Sol | levers | concepts named by all three |"
          " by exactly two |", "|---|---|---|---|---:|---:|---:|"]
    for x in sets:
        L.append(f"| {x['scenario']} | {x['manning_et_al_gpt4']} | {x['qwen']} | {x['gpt_sol']} | "
                 f"{x['levers_total']} | {x['concepts_named_by_all_three']} | {x['concepts_named_by_two']} |")
    L += ["", "## 5. Expected Parrot: segment means by elicitation", "",
          "| condition | appeal: adventurous / mainstream / health | appeal range | "
          "purchase: adventurous / mainstream / health | purchase range |", "|---|---|---:|---|---:|"]
    for x in segs:
        L.append(f"| {x['condition']} | {x['appeal_adventurous']} / {x['appeal_mainstream']} / "
                 f"{x['appeal_skeptical']} | {x['appeal_range']} | {x['purchase_adventurous']} / "
                 f"{x['purchase_mainstream']} / {x['purchase_skeptical']} | {x['purchase_range']} |")
    L += ["", "Overall means, separate-call condition vs the public run:", "",
          "| measure | Qwen separate calls | Expected Parrot (cited) |", "|---|---:|---:|"]
    L += [f"| {x['measure']} | {x['qwen_separate_calls']} | {x['expected_parrot_cited']} |" for x in overall]
    L += ["", "## 6. Expected Parrot: does a persona give the same answer twice?", "",
          "Same 12 personas and the same four 1–5 items. The first pair differs in where the persona sits "
          "(user vs system prompt, plus one line asking for a decisive answer) and in the seed; the second "
          "in whether all questions share one call with full context.", "",
          "| comparison | ratings | identical | differ by ≥ 2 | personas crossing purchase ≥ 4 | who |",
          "|---|---:|---:|---:|---:|---|"]
    for x in agree:
        L.append(f"| {x['comparison']} | {x['ratings_compared']} | {x['identical']} | {x['differ_by_2_or_more']} | "
                 f"{x['purchase_top2_flips']} | {x['personas_flipping']} |")
    L += ["", "## 7. Fixed seeds replay the same draw", "",
          "| comparison | units | identical | total | note |", "|---|---|---:|---:|---|"]
    L += [f"| {x['comparison']} | {x['units']} | {x['identical']} | {x['total']} | {x['note']} |" for x in replays]
    L += ["", "## 8. Every saved run, including failures", "",
          "Runtime is recorded model time (manifest or summed cell runtimes) where the run saved it.", "",
          "| case study | run | what | units | succeeded | failed | model calls | minutes | note |",
          "|---|---|---|---|---:|---:|---:|---:|---|"]
    L += [f"| {x['case_study']} | `{x['run']}` | {x['what']} | {x['units']} | {x['succeeded']} | {x['failed']} | "
          f"{x['model_calls']} | {x['minutes']} | {x['note']} |" for x in ledger]
    return "\n".join(L) + "\n"


def build() -> dict:
    qwen, sol = qwen_scientist_levers(), sol_scientist_levers()
    return {
        "qwen": qwen, "sol": sol, "summary": fragility_summary(qwen, sol),
        "sol_desc": sol_study_summary(), "ledger": run_ledger(), "sets": scientist_lever_sets(),
        "replays": seed_replays(), "segs": parrot_segments(), "overall": parrot_overall(),
        "agree": parrot_agreement(),
    }


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
    (OUT / "lever_fragility.svg").write_text(svg_fragility(t["qwen"], t["sol"]), encoding="utf-8")
    (OUT / "RESULTS.md").write_text(markdown(**t), encoding="utf-8")
    print(f"wrote results to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
