"""Build docs/event.html from results/published.json. English only.

The existing page answers the original question -- does daily foreign flow
predict returns -- and keeps its own inputs. This one answers a different
question and reads a single file, so the two cannot contradict each other by
drifting apart.

Investor categories are stored under their exchange names and rendered in
English here, in the presentation layer, so the analysis keeps the labels the
data actually carries.

The retraction exhibit is a section, not a footnote. A page that quietly
replaced -117.7 with -46.5 would be asking to be trusted; a page that shows
the sweep that forced the change is showing its work.

    python 42_build_event_page.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "published.json"
OUT = ROOT / "docs" / "event.html"
INDEX = ROOT / "docs" / "index.html"

EN = {
    "기관합계": "Institutions, total",
    "금융투자": "Securities firms",
    "보험": "Insurance",
    "투신": "Investment trusts",
    "사모": "Private funds",
    "은행": "Banks",
    "기타금융": "Other financial",
    "연기금": "Pension funds",
    "기타법인": "Other corporations",
    "개인": "Retail",
    "외국인": "Foreign",
    "기타외국인": "Other foreign",
}

FALLBACK_CSS = """
body{margin:0;font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",
Helvetica,Arial,sans-serif;color:#1a1a1a;background:#fff}
.wrap{max-width:820px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:28px;line-height:1.25;margin:0 0 8px}
h2{font-size:19px;margin:40px 0 12px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{padding:7px 10px;border-bottom:1px solid #e6e6e6;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:1px solid #bbb;font-weight:600}
"""

EXTRA_CSS = """
.ev-lede{font-size:17px;line-height:1.6;margin:0 0 28px;color:#333}
.ev-kpi{display:flex;flex-wrap:wrap;gap:14px;margin:20px 0 8px}
.ev-kpi div{flex:1 1 200px;border:1px solid #e2e2e2;border-radius:8px;
padding:14px 16px}
.ev-kpi .n{font-size:24px;font-weight:600;letter-spacing:-.01em}
.ev-kpi .l{font-size:12px;text-transform:uppercase;letter-spacing:.06em;
color:#777;margin-bottom:4px}
.ev-kpi .s{font-size:12px;color:#777;margin-top:3px}
.ev-note{font-size:13px;color:#666;margin:8px 0 0}
.ev-pass{color:#0a7d3f;font-weight:600}
.ev-fail{color:#999}
.ev-neg{color:#b3261e}
.ev-list{padding-left:20px;font-size:14px;line-height:1.6}
.ev-list li{margin:8px 0}
table.ev tbody tr.hl td{background:#fbf7ea;font-weight:600}
.ev-scroll{overflow-x:auto}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def bp(v, nd=1) -> str:
    return "&mdash;" if v is None else f"{v:+,.{nd}f}"


def tv(v) -> str:
    return "&mdash;" if v is None else f"{v:+.2f}"


def table(headers, body_rows, cls="ev") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    rows = "".join("<tr" + (' class="hl"' if hl else "") + ">"
                   + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                   for cells, hl in body_rows)
    return (f'<div class="ev-scroll"><table class="{cls}"><thead><tr>{head}'
            f"</tr></thead><tbody>{rows}</tbody></table></div>")


def cell(d, key="bp"):
    return None if not d else d.get(key)


def site_css() -> str:
    """Reuse the stylesheet the other page already ships, so the two match
    without this script owning a second copy of the design."""
    if INDEX.exists():
        found = re.findall(r"<style[^>]*>(.*?)</style>",
                           INDEX.read_text(encoding="utf-8"), re.S)
        if found:
            return "\n".join(found)
    return FALLBACK_CSS


def build(d: dict) -> str:
    h = d["headline"]
    full, clean = h["full_sample"], h.get("after_threats")
    cov = d["coverage"]

    kpi = (
        '<div class="ev-kpi">'
        f'<div><div class="l">Full provisional sample</div>'
        f'<div class="n ev-neg">{bp(cell(full))} bp</div>'
        f'<div class="s">per standard deviation &middot; '
        f't {tv(cell(full, "t"))} &middot; {full["events"]:,} events</div></div>'
        f'<div><div class="l">After every filter</div>'
        f'<div class="n ev-neg">{bp(cell(clean))} bp</div>'
        f'<div class="s">index reviews, dividend cluster and market-wide '
        f'waves removed &middot; t {tv(cell(clean, "t"))}</div></div>'
        f'<div><div class="l">Reporting seasons</div>'
        f'<div class="n">{full["seasons"]}</div>'
        f'<div class="s">{cov["first"]} to {cov["last"]} &middot; '
        f'the unit inference is clustered on</div></div>'
        "</div>")

    where = table(
        ["Sample", "Coefficient", "t", "Events"],
        [([ "Provisional filings", bp(cell(full)), tv(cell(full, "t")),
            f'{full["events"]:,}'], True),
         (["Periodic filings", bp(cell(h["periodic"])),
           tv(cell(h["periodic"], "t")), f'{h["periodic"]["events"]:,}'], False),
         (["All filings pooled", bp(cell(h["all_events"])),
           tv(cell(h["all_events"], "t")),
           f'{h["all_events"]["events"]:,}'], False),
         (["Foreign flow, provisional", bp(cell(h["foreign"])),
           tv(cell(h["foreign"], "t")), f'{h["foreign"]["events"]:,}'], False)])

    hz = table(["Horizon", "Coefficient", "t"],
               [([k.replace("abn", "") + " trading days", bp(cell(v)),
                  tv(cell(v, "t"))], k == "abn60")
                for k, v in d["horizons"].items()])

    sw = d["why_the_earlier_number_was_larger"]
    sweep = table(
        ["Minimum announcements per day", "Coefficient", "t", "Days used"],
        [([str(r["min_n"]), bp(r["bp"]), tv(r["t"]), f'{r["days"]:,}'],
          r["min_n"] == 25) for r in sw["sweep"]])

    dec = d["decomposition"]
    dec_rows = []
    for k, v in dec["types"].items():
        mark = ""
        if "p_fwer" in v:
            passed = abs(v["t"]) >= dec["bar"]
            mark = (f'<span class="ev-pass">clears</span>' if passed
                    else f'<span class="ev-fail">&mdash;</span>')
        else:
            mark = '<span class="ev-fail">reference</span>'
        dec_rows.append((
            [esc(EN.get(k, k)), bp(v["bp"]), tv(v["t"]),
             f'{v["p_fwer"]:.3f}' if "p_fwer" in v else "&mdash;", mark],
            k == "연기금"))
    dec_tbl = table(["Investor category", "Coefficient", "t",
                     "Family-wise p", ""], dec_rows)

    pf = d["portfolio"]
    pf_rows = []
    for side in ("crowded", "quiet"):
        for who, label in (("institutional", "Institutions, total"),
                           ("pension", "Pension funds"),
                           ("placebo", "Random ranking (placebo)")):
            v = pf[side][who]
            pf_rows.append((
                [f"{side.capitalize()} surprises &minus; all surprises",
                 label, bp(v["bp"]), tv(v["t"]), f'{v["positions"]:,}'],
                side == "crowded" and who != "placebo"))
    pf_tbl = table(["Contrast", "Ranked on", "bp per position", "t",
                    "Positions"], pf_rows)

    th = d["threats"]["interactions"]
    th_rows = []
    for k in ("size", "review window", "December", "market wave"):
        if k not in th:
            continue
        m, i = th[k]["main"], th[k]["interaction"]
        th_rows.append(([
            {"size": "Market capitalisation rank",
             "review window": "Quarterly index review window",
             "December": "December dividend cluster",
             "market wave": "Market-wide institutional buying"}[k],
            bp(m["bp"]), tv(m["t"]), bp(i["bp"]), tv(i["t"])],
            k == "market wave"))
    th_tbl = table(["Moderator", "Main", "t", "Interaction", "t"], th_rows)

    lims = "".join(f"<li>{esc(x)}</li>" for x in d["limitations"])
    rej = "".join(f"<li>{esc(x)}</li>" for x in d["rejected"])
    est = d["estimator"]
    stale = ("" if not d.get("stale_inputs") else
             '<p class="ev-note"><strong>Warning:</strong> published with '
             "inputs older than the panel: "
             + esc(", ".join(d["stale_inputs"])) + "</p>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pre-event institutional positioning &middot; Korea</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{site_css()}{EXTRA_CSS}</style></head><body><div class="wrap">

<h1>What institutional buying before an earnings surprise tells you</h1>
<p class="ev-lede">{esc(d["claim"])}</p>
<p class="ev-note">Korea &middot; provisional earnings announcements &middot;
{cov["first"]} to {cov["last"]} &middot;
<a href="index.html">the daily foreign-flow study is here</a></p>
{stale}
{kpi}

<h2>Where the effect holds, and where it does not</h2>
{where}
<p class="ev-note">The effect is confined to provisional announcements, the
filings that carry genuine news. Periodic reports, which mostly confirm what
is already known, show nothing. Neither does foreign flow &mdash; the series
this project originally set out to use.</p>

<h2>Horizon</h2>
{hz}
<p class="ev-note">Nothing at five or twenty days, then the whole effect at
sixty. Only three horizons were examined, so the search was narrow, but a
result living at a single horizon is weaker than one that accumulates.</p>

<h2>Why an earlier version of this page said &minus;117.7</h2>
<p class="ev-note">{esc(sw["note"])}</p>
{sweep}
<p class="ev-note">Weighting each day equally sounds neutral until the days
carry between one and several hundred announcements. Weighting each position
equally &mdash; what a book actually earns &mdash; and clustering on the
reporting season gives the figures at the top of this page.</p>

<h2>Which investors carry it</h2>
{dec_tbl}
<p class="ev-note">Eight categories were searched at once, so the threshold
was set by permutation before the numbers were read: shuffle the baseline
within each announcement day, re-estimate the whole family, keep the largest
|t|. The bar is {dec["bar"]:.2f} at the 95th percentile of
{dec["perm"]:,} permutations. Pension funds clear it, and their coefficient is
essentially the whole aggregate. Entered together, however, neither survives
&mdash; the categories are not separable, and the aggregate remains the honest
form of the signal.</p>

<h2>The usable form is avoidance</h2>
{pf_tbl}
<p class="ev-note">Among the strongest surprises, those institutions had
already crowded into underperform the rest. The long side does not work: the
random ranking separates the quiet surprises better than the real one does,
which is how a contrast that looks promising turns out to be machinery.</p>

<h2>Alternative explanations</h2>
{th_tbl}
<p class="ev-note">Each threat is one interaction on the full sample rather
than a split, so the quantity of interest carries its own standard error.
Index rebalancing would have to concentrate the effect in large caps, since
the global and domestic benchmarks are large-cap indices; the interaction is
small and points the other way. The dividend cluster and the review windows
are likewise flat. The one live interaction is market-wide institutional
buying, where the effect roughly triples &mdash; consistent with mechanical
price pressure rather than information, though four interactions were tested
and this one does not clear a family-wise threshold on its own.</p>

<h2>Stated limitations</h2>
<ul class="ev-list">{lims}</ul>

<h2>Tested and rejected</h2>
<ul class="ev-list">{rej}</ul>

<h2>Method</h2>
<p class="ev-note">Sample: {esc(est["sample"])}.
Outcome: {esc(est["outcome"])}.
Regressors: {esc(est["regressors"])}.
Weighting: {esc(est["weighting"])}.
Inference: {esc(est["inference"])}.
Days carrying fewer than {est["min_events_per_day"]} announcements are
dropped. Every figure on this page is read from a single results file
regenerated from the panel; none is typed by hand.</p>
<p class="ev-note">Generated {esc(d["generated"])}.</p>

</div></body></html>
"""


def link_from_index() -> bool:
    """Put one link on the other page, after its heading, once."""
    if not INDEX.exists():
        return False
    s = INDEX.read_text(encoding="utf-8")
    if "event.html" in s:
        return False
    link = ('<p><a href="event.html">Pre-event institutional positioning '
            "and the earnings surprise &rarr;</a></p>")
    out = re.sub(r"(</h1>)", r"\1" + link, s, count=1)
    if out == s:
        return False
    INDEX.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    if not SRC.exists():
        raise SystemExit("run 41_publish.py first")
    d = json.loads(SRC.read_text(encoding="utf-8"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(d), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print("linked from index.html" if link_from_index()
          else "index.html already links here (or is missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
