"""python scripts/run_all.py  ->  results/*.json|csv, figures/*.png, demo/decision_screen.html"""
import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decisiontrace import config as C
from decisiontrace import engine as E
from decisiontrace import evalharness as H
from decisiontrace import scenarios
from decisiontrace.analysis import build_experiment, margin_vs_best_other
from decisiontrace.ask import GroundedResponder
from decisiontrace.ui import PRESETS, render_page, render_scenario

R, F = ROOT / "results", ROOT / "figures"
for d in (R, F, ROOT / "demo"):
    d.mkdir(exist_ok=True)
NAVY, TEAL, ORANGE, GREY, RED, GOLD = "#1F2A44", "#1B7F79", "#D9622B", "#9AA3B2", "#C0392B", "#C98A00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- scenarios
exps = {n: build_experiment(scenarios.get(n)) for n in scenarios.SCENARIOS}
scen_rows = []
for n, ex in exps.items():
    d = ex.decision
    scen_rows.append({"scenario": n, "title": d["title"], "state": d["state"], "expected": "/".join(sorted(scenarios.SCENARIO_EXPECTED[n])),
                      "state_ok": d["state"] in scenarios.SCENARIO_EXPECTED[n], "winner": d["winner"], "lead_pp": round(d["lead_pp"], 2),
                      "win_rate_pct": d["win_rate_pct"], "stability_pct": round(d["stability"]["pct"], 1), "confidence": d["confidence"],
                      "headline": d["headline"], "headline_issues": "; ".join(H.headline_hygiene(ex))})
scen = pd.DataFrame(scen_rows)
scen.to_csv(R / "scenario_states.csv", index=False)

# ---------------------------------------------------------------- evaluation
rows = []
for n, ex in exps.items():
    for r in H.evaluate_responder(ex, GroundedResponder(ex)):
        rows.append({"scenario": n, **r})
ev = pd.DataFrame(rows)
ev.to_csv(R / "eval_grounded_answers.csv", index=False)
base = exps["base"]
bl = pd.DataFrame(H.evaluate_responder(base, H.UngroundedBaseline(base)))
bl.to_csv(R / "eval_baseline_answers.csv", index=False)
mut, fp = H.mutation_test(list(exps.values()))
adv = H.adversarial_recall(base)

dims = H.DIMS + ["intent", "all"]
rate = lambda df: {k: float(df[f"p_{k}"].mean()) for k in dims}
metrics = {
    "scenarios": scen_rows,
    "grounded": {"n_answers": int(len(ev)), "rates": rate(ev),
                 "by_scenario": {n: rate(g) for n, g in ev.groupby("scenario")},
                 "by_category": {c: rate(g) for c, g in ev.groupby("cat")}},
    "baseline": {"n_answers": int(len(bl)), "rates": rate(bl)},
    "mutation": {m: v for m, v in mut.items()}, "mutation_false_positives": fp,
    "adversarial": adv, "n_questions": len(H.QUESTIONS),
    "base": {"state": base.decision["state"], "headline": base.decision["headline"], "lead_pp": base.decision["lead_pp"],
             "win_rate_pct": base.decision["win_rate_pct"], "stability_pct": base.decision["stability"]["pct"],
             "most_sensitive": base.decision["most_sensitive"], "n_evidence_rows": len(base.ledger.all()),
             "flips": [{k: f[k] for k in ("assumption", "side", "value", "unit", "within_tested_range")} for f in base.decision["flips"]],
             "segments": {k: [{kk: r[kk] for kk in ("segment", "winner", "contribution_pp", "weight_pct")} for r in v] for k, v in base.decision["segments"].items() if k != "industry"}},
}
(R / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

# ---------------------------------------------------------------- figures
# 1. enterprise-representation flip curve + sensitivity tornado
ex = base
pop, spec, a0 = ex.pop, ex.spec, ex.a0
W = ex.decision["winner"]; wi = C.OPTIONS.index(W)
xs = np.linspace(0.03, 0.55, 90)
ms = [margin_vs_best_other(E.expected_shares(pop, spec, {**a0, "enterprise_share": x}), wi) * 100 for x in xs]
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
a = ax[0]
a.axhline(0, c=NAVY, lw=.8)
a.plot(xs * 100, ms, c=TEAL, lw=2)
lo, hi = a0["enterprise_share"] * .85 * 100, a0["enterprise_share"] * 1.15 * 100
a.axvspan(lo, hi, color="#E8ECF4", label="tested range (+/-15%)")
a.axvline(a0["enterprise_share"] * 100, c=GREY, ls="--", lw=.8)
fl = [f for f in ex.decision["flips"] if f["assumption"] == "enterprise_share"]
if fl:
    a.axvline(fl[0]["value"], c=RED, ls=":", lw=1.4)
    a.annotate(f"{W} stops leading\nbelow {fl[0]['value']:.1f}%", (fl[0]["value"], 2), xytext=(fl[0]["value"] + 3, -6), fontsize=7, color=RED,
               arrowprops=dict(arrowstyle="-", color=RED))
a.set_xlabel("Enterprise share of the weighted population (%)"); a.set_ylabel(f"{W}'s lead over the best other option (pp)")
a.legend(frameon=False, fontsize=7, loc="upper left"); a.set_title("How much enterprise representation does the result need?", fontsize=8.5, loc="left")
b = ax[1]
t = list(reversed(ex.decision["sensitivity"]))
b.barh([x["assumption"].replace("_", " ") for x in t], [x["swing_pp"] for x in t], color=[ORANGE if x["flips_in_range"] else TEAL for x in t])
b.set_xlabel("Swing in lead across the tested range (pp)"); b.set_title("Which assumption moves the result most?", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(F / "fig_sensitivity.png", dpi=200); plt.close(fig)

# 2. same pipeline, different evidence -> different presentation
STATE_COL = {"ROBUST": "#2E8B57", "MODERATE": GOLD, "LOW": ORANGE, "CONTESTED": "#7B4FA3", "UNSTABLE": RED, "INSUFFICIENT_EVIDENCE": RED, "STALE": "#6B7280"}
fig, ax = plt.subplots(figsize=(8.6, 3.1))
y = np.arange(len(scen))[::-1]
ax.barh(y, scen.lead_pp, color=[STATE_COL[s] for s in scen.state], height=.62)
for yi, r in zip(y, scen.itertuples()):
    ax.text(max(r.lead_pp, 0) + .3, yi, f"{r.state.replace('_', ' ')}  ({r.stability_pct:.0f}% of assumption sets, {r.win_rate_pct:.0f}% of runs)", va="center", fontsize=7.2, color=NAVY)
ax.set_yticks(y); ax.set_yticklabels(scen.title); ax.set_xlim(0, 24)
ax.set_xlabel("Headline lead of the top option over the runner-up (pp): what a naive dashboard would show")
ax.set_title("Same lead, different evidence: what DecisionTrace says instead", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(F / "fig_presentation_states.png", dpi=200); plt.close(fig)

# 3. evaluation
fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.9))
dn = ["citation", "numeric", "observed", "uncertainty", "overreach"]
x = np.arange(len(dn)); w = .38
g, bsl = metrics["grounded"]["rates"], metrics["baseline"]["rates"]
ax[0].bar(x - w / 2, [g[k] for k in dn], w, color=TEAL, label=f"grounded responder ({len(ev)} answers, 8 scenarios)")
ax[0].bar(x + w / 2, [bsl[k] for k in dn], w, color=GREY, label=f"scripted ungrounded baseline ({len(bl)})")
ax[0].set_xticks(x); ax[0].set_xticklabels(["citation\naccuracy", "numerical\nfidelity", "observed vs\ngenerated", "uncertainty\npreserved", "no evidence\noverreach"], fontsize=7)
ax[0].set_ylim(0, 1.18); ax[0].legend(frameon=False, fontsize=6.6, loc="upper center"); ax[0].set_title("Answer quality by dimension", fontsize=8.5, loc="left")
labels = ["caught by mutation test\n(patterns the checkers\nwere written for)", "caught on hand-written\nsubtle overreach\n(unseen phrasing)"]
mrate = sum(v["caught"] for v in mut.values()) / sum(v["applied"] for v in mut.values())
ax[1].bar([0, 1], [mrate, adv["caught"] / adv["overreach_cases"]], color=[TEAL, ORANGE], width=.5)
for i, (v, n_) in enumerate([(mrate, sum(v["applied"] for v in mut.values())), (adv["caught"] / adv["overreach_cases"], adv["overreach_cases"])]):
    ax[1].text(i, v + .03, f"{v:.0%}  (n={n_})", ha="center", fontsize=7.5)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(labels, fontsize=7); ax[1].set_ylim(0, 1.2)
ax[1].set_title("Do the checkers themselves generalize?", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(F / "fig_evaluation.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- demo page + screenshots
names = list(scenarios.SCENARIOS); titles = {n: scenarios.SCENARIOS[n].title for n in names}
blocks = []
for n in names:
    e_ = exps[n]; r_ = GroundedResponder(e_)
    blocks.append(render_scenario(e_, n, qa=[(q, r_.answer(q)) for q in PRESETS], live=False, display="block" if n == "base" else "none"))
(ROOT / "demo" / "decision_screen.html").write_text(render_page(blocks, names, titles, "base", live=False))
if shutil.which("wkhtmltoimage"):
    for n in ("base", "near_tie", "missing_segment", "contradictory"):
        e_ = exps[n]; r_ = GroundedResponder(e_)
        tmp = ROOT / "demo" / f"_tmp_{n}.html"
        tmp.write_text(render_page([render_scenario(e_, n, qa=[(q, r_.answer(q)) for q in PRESETS], live=False)], names, titles, n, live=False))
        subprocess.run(["wkhtmltoimage", "--quiet", "--width", "1120", str(tmp), str(F / f"screen_{n}.png")], capture_output=True)
        tmp.unlink()

print(scen[["scenario", "state", "state_ok", "lead_pp", "stability_pct", "win_rate_pct"]].to_string(index=False))
print("grounded", {k: round(v, 3) for k, v in metrics["grounded"]["rates"].items()})
print("baseline", {k: round(v, 3) for k, v in metrics["baseline"]["rates"].items()})
print("mutation", {m: f"{v['caught']}/{v['applied']}" for m, v in mut.items()}, "fp", fp)
print("adversarial", adv["caught"], "/", adv["overreach_cases"], "clean flagged", adv["false_positives"], "/", adv["clean_cases"])
