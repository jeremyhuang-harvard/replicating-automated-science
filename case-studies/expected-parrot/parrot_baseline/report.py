"""Self-contained HTML report: inline CSS only, charts built from styled divs, every dynamic value escaped."""

from __future__ import annotations

from html import escape
from typing import Any

from .schemas import Response, StudyManifest

LABEL = "Simulated hypotheses, not human evidence"

_CSS = """
:root{color-scheme:light dark;--bg:#fcfcfb;--panel:#f3f2ef;--ink:#0b0b0b;--ink2:#52514e;--grid:#d9d8d3;--warn-bg:#fff3cd;--warn-ink:#4a3500;--warn-edge:#b07d00;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948;
--r1:#c6dcf7;--r2:#8fbbee;--r3:#5a9be3;--r4:#2a78d6;--r5:#1b4f95;--t1:#0b0b0b;--t2:#0b0b0b;--t3:#0b0b0b;--t4:#fff;--t5:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--panel:#262624;--ink:#fff;--ink2:#c3c2b7;--grid:#44433f;--warn-bg:#3a2e08;--warn-ink:#ffe9a8;--warn-edge:#eda100;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--r1:#1f3a5f;--r2:#2b5a96;--r3:#3987e5;--r4:#6aa5ec;--r5:#a6c8f4;--t1:#fff;--t2:#fff;--t3:#fff;--t4:#0b0b0b;--t5:#0b0b0b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main,header .in{max-width:980px;margin:0 auto;padding:0 20px}header{padding:24px 0 8px}h1{margin:.2em 0;font-size:1.7rem}h2{margin-top:2.2em;font-size:1.25rem;border-bottom:1px solid var(--grid);padding-bottom:.3em}h3{font-size:1rem;margin:1.4em 0 .4em}
.banner{background:var(--warn-bg);color:var(--warn-ink);border:2px solid var(--warn-edge);border-radius:8px;padding:12px 16px;font-weight:700;font-size:1.05rem}
.banner small{display:block;font-weight:400;margin-top:4px}
.card{background:var(--panel);border-radius:8px;padding:12px 16px}dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0}dt{color:var(--ink2)}dd{margin:0;overflow-wrap:anywhere}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--ink2);font-size:.9rem;margin:6px 0}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.row{display:grid;grid-template-columns:190px 1fr 48px;align-items:center;gap:10px;margin:5px 0}.row .lbl{font-size:.9rem;color:var(--ink2);overflow-wrap:anywhere}.row .val{font-variant-numeric:tabular-nums;font-weight:600}
.track{height:18px;background:linear-gradient(to right,var(--grid) 1px,transparent 1px) 0 0/20% 100%;border-right:1px solid var(--grid);border-left:1px solid var(--grid)}
.bar{height:100%;border-radius:0 4px 4px 0;min-width:2px}.axis{display:grid;grid-template-columns:190px 1fr 48px;gap:10px;font-size:.75rem;color:var(--ink2)}.axis div div{display:flex}.axis span{flex:1;text-align:right}
.stack{display:flex;gap:2px;height:22px}.seg{display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:600;border-radius:3px;min-width:0}
.quote{border-left:4px solid var(--grid);padding:2px 12px;margin:8px 0}.quote .who{color:var(--ink2);font-size:.85rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid var(--grid);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--ink2)}td.n{text-align:right;font-variant-numeric:tabular-nums}
details{margin:8px 0}summary{cursor:pointer;color:var(--ink2)}li{margin:4px 0}footer{margin:3em 0;color:var(--ink2);font-size:.85rem}
@media (max-width:640px){.row,.axis{grid-template-columns:110px 1fr 40px}}
"""

_METHODOLOGY = [
    "A fixed panel of synthetic personas is defined by explicit product attitudes (novelty seeking, brand loyalty, health concern, price sensitivity), grouped into segments.",
    "Each persona receives one self-contained prompt containing the complete concept and every survey question, and returns one JSON object with all answers. No answer depends on conversational memory.",
    "Numeric items use a 1-5 scale. Means and rating distributions are computed over valid responses only; failed responses are listed as warnings.",
    "Representative quotes are the open-text answers of the respondent whose first rating is closest to their segment mean. They illustrate, they do not prove.",
    "The run is reproducible from manifest.json: engine, model, seed, concept, panel, questions and exact prompts are recorded.",
]


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


def _seg_color(index: int) -> str:
    return f"var(--s{index % 8 + 1})"


def _axis() -> str:
    ticks = "".join(f"<span>{i}</span>" for i in range(1, 6))
    return f'<div class="axis"><span></span><div><div>{ticks}</div></div><span></span></div>'


def _mean_bars(manifest: StudyManifest, summary: dict[str, Any], segments: list[str]) -> str:
    out = ['<div class="legend">' + "".join(
        f'<span><i class="sw" style="background:{_seg_color(i)}"></i>{_e(s)}</span>' for i, s in enumerate(segments)
    ) + '<span><i class="sw" style="background:var(--ink2)"></i>overall</span></div>']
    for q in (q for q in manifest.questions if q.kind == "rating"):
        out.append(f"<h3>{_e(q.question_id)}: {_e(q.text)}</h3>")
        rows = [("overall", summary["means"]["overall"].get(q.question_id), "var(--ink2)")]
        rows += [(s, summary["means"]["by_segment"].get(s, {}).get(q.question_id), _seg_color(i)) for i, s in enumerate(segments)]
        for label, value, color in rows:
            if value is None:
                out.append(f'<div class="row"><span class="lbl">{_e(label)}</span><span class="lbl">no valid responses</span><span></span></div>')
                continue
            out.append(
                f'<div class="row"><span class="lbl">{_e(label)}</span><div class="track"><div class="bar" '
                f'style="width:{value / 5 * 100:.1f}%;background:{color}"></div></div><span class="val">{value:.2f}</span></div>'
            )
        out.append(_axis())
    return "\n".join(out)


def _distributions(manifest: StudyManifest, summary: dict[str, Any], segments: list[str]) -> str:
    legend = "".join(f'<span><i class="sw" style="background:var(--r{v})"></i>{v}</span>' for v in range(1, 6))
    out = [f'<div class="legend">Rating (count of respondents): {legend}</div>']
    dists = summary["distributions"]
    for q in (q for q in manifest.questions if q.kind == "rating"):
        out.append(f"<h3>{_e(q.question_id)}</h3>")
        groups = [("overall", dists["overall"].get(q.question_id))]
        groups += [(s, dists["by_segment"].get(s, {}).get(q.question_id)) for s in segments]
        for label, dist in groups:
            total = sum(dist.values()) if dist else 0
            if not total:
                continue
            cells = "".join(
                f'<div class="seg" style="flex:{n} 1 0;background:var(--r{v});color:var(--t{v})" title="{v}: {n}">{n}</div>'
                for v, n in ((int(k), n) for k, n in dist.items()) if n
            )
            out.append(f'<div class="row"><span class="lbl">{_e(label)}</span><div class="stack">{cells}</div><span class="val">n={total}</span></div>')
    return "\n".join(out)


def _quotes(manifest: StudyManifest, summary: dict[str, Any]) -> str:
    out = []
    for q in (q for q in manifest.questions if q.kind == "text"):
        out.append(f"<h3>{_e(q.text)}</h3>")
        for seg, by_q in summary["quotes"].items():
            quote = by_q.get(q.question_id)
            if quote:
                out.append(f'<div class="quote">&ldquo;{_e(quote["text"])}&rdquo;<div class="who">{_e(seg)} &middot; {_e(quote["persona_id"])}</div></div>')
    return "\n".join(out)


def _table(headers: list[str], rows: list[list[Any]], numeric_from: int = 99) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td class="{"n" if i >= numeric_from else ""}">{_e(c)}</td>' for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_report(manifest: StudyManifest, responses: list[Response], summary: dict[str, Any]) -> str:
    segments = list(dict.fromkeys(p.segment for p in manifest.panel))
    rating_ids = [q.question_id for q in manifest.questions if q.kind == "rating"]
    means_rows = [["overall", summary["counts"]["ok"]] + [f'{summary["means"]["overall"].get(q, float("nan")):.2f}' for q in rating_ids]]
    for seg in segments:
        m = summary["means"]["by_segment"].get(seg)
        means_rows.append([seg, summary["counts"]["by_segment"][seg]] + [f"{m[q]:.2f}" if m else "n/a" for q in rating_ids])
    by_id = {r.persona_id: r for r in responses}
    panel_rows = []
    for p in manifest.panel:
        r = by_id.get(p.persona_id)
        panel_rows.append([p.persona_id, p.segment, p.description] + [(r.answers.get(q, "") if r and r.ok else "failed") for q in rating_ids])
    warnings = "".join(f"<li>{_e(w)}</li>" for w in summary["warnings"]) or "<li>None.</li>"
    limitations = "".join(f"<li>{_e(x)}</li>" for x in summary["limitations"])
    methodology = "".join(f"<li>{_e(x)}</li>" for x in _METHODOLOGY)
    c = summary["counts"]
    meta = [("Engine", manifest.engine), ("Model", manifest.model), ("Created (UTC)", manifest.created_at), ("Seed", manifest.seed),
            ("Study ID", manifest.study_id), ("Panel", f'{c["panel"]} personas, {c["ok"]} valid, {c["failed"]} failed')]
    meta_html = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in meta)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(manifest.concept.title)} &mdash; simulated concept test</title><style>{_CSS}</style></head>
<body><header><div class="in">
<div class="banner" role="note">&#9888; {_e(LABEL.upper())}<small>Synthetic personas answered these questions. Use the results to generate hypotheses to test with real people, not to make decisions.</small></div>
<h1>{_e(manifest.concept.title)}</h1></div></header>
<main>
<div class="card"><strong>Concept</strong><p>{_e(manifest.concept.description)}</p><dl>{meta_html}</dl></div>
<h2>Mean ratings by segment</h2>
{_mean_bars(manifest, summary, segments)}
<details><summary>Table view</summary>{_table(["Group", "n"] + rating_ids, means_rows, 1)}</details>
<h2>Rating distributions</h2>
{_distributions(manifest, summary, segments)}
<h2>Representative quotes</h2>
{_quotes(manifest, summary)}
<h2>Panel</h2>
<details><summary>{c["panel"]} personas and their ratings</summary>{_table(["ID", "Segment", "Attitude description"] + rating_ids, panel_rows, 3)}</details>
<h2>Methodology</h2><ul>{methodology}</ul>
<h2>Warnings</h2><ul>{warnings}</ul>
<h2>Limitations</h2><ul>{limitations}</ul>
</main>
<footer><div style="max-width:980px;margin:0 auto;padding:0 20px">{_e(LABEL)}. Generated locally by parrot_baseline.</div></footer>
</body></html>
"""
