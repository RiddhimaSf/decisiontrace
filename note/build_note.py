"""Builds note/DecisionTrace_note.pdf (4 pages). Every number is read from the experiment, never typed.

    python note/build_note.py          (needs: pip install -r requirements.txt, reportlab, pillow, pypdf)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisiontrace import config as C                      # noqa: E402
from decisiontrace import engine as E                      # noqa: E402
from decisiontrace import scenarios                        # noqa: E402
from decisiontrace.analysis import build_experiment        # noqa: E402
from decisiontrace.ask import GroundedResponder            # noqa: E402

OUT = ROOT / "note"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
REPO = "github.com/RiddhimaSf/decisiontrace"

# ------------------------------------------------------------------ fonts & palette
FD = Path("/usr/share/fonts/truetype/crosextra")
if not (FD / "Carlito-Regular.ttf").exists():
    sys.exit("Rebuilding the PDF needs the Carlito font (Linux package fonts-crosextra-carlito) at " + str(FD) + ". The finished PDF is already committed: note/DecisionTrace_note.pdf")
MONO = Path("/usr/share/fonts/truetype/dejavu")
for n, f in [("Carlito", "Carlito-Regular.ttf"), ("Carlito-Bold", "Carlito-Bold.ttf"), ("Carlito-Italic", "Carlito-Italic.ttf"), ("Carlito-BoldItalic", "Carlito-BoldItalic.ttf")]:
    pdfmetrics.registerFont(TTFont(n, str(FD / f)))
    fm.fontManager.addfont(str(FD / f))
pdfmetrics.registerFont(TTFont("Mono", str(MONO / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("Carlito", normal="Carlito", bold="Carlito-Bold", italic="Carlito-Italic", boldItalic="Carlito-BoldItalic")
plt.rcParams.update({"font.family": ["Carlito", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9AA3B2", "axes.labelcolor": "#1F2A44", "xtick.color": "#1F2A44", "ytick.color": "#1F2A44"})

INK, TEAL, ORG, GREY, LIGHT, MUTED = "#1F2A44", "#1B7F79", "#D9622B", "#9AA3B2", "#EEF1F6", "#5B657A"
RED, PURPLE, SLATE = "#C0392B", "#7B4FA3", "#6B7280"
OPT = {"A": ORG, "B": TEAL, "C": GREY}
CI = colors.HexColor
W = letter[0] - 2 * 0.62 * inch

# ------------------------------------------------------------------ data (everything below is computed)
ex = build_experiment(scenarios.get("base"))
d, L, pop, spec, a0 = ex.decision, ex.ledger, ex.pop, ex.spec, ex.a0
sh = {s["option"]: s["mean"] for s in d["shares"]}
lead = d["lead_pp"]
seg = {}
for dim in ("company_size",):
    for r in d["segments"][dim]:
        seg[r["segment"]] = {"w": r["weight_pct"], **{o: L.get(f"segment.{dim}={r['segment']}.share.{o}")["value"] for o in "ABC"}}
smb = seg["SMB"]
smb_ahead = smb["A"] - smb["B"]
sens = {t["assumption"]: t for t in d["sensitivity"]}
flip_rel = next(f for f in d["flips"] if f["assumption"] == "reliability_sensitivity")
nominal_rel = a0["reliability_sensitivity"]
scen = pd.read_csv(ROOT / "results" / "scenario_states.csv").set_index("scenario")
M = json.loads((ROOT / "results" / "metrics.json").read_text())
n_ans = M["grounded"]["n_answers"]
mut_ok = sum(v["caught"] for v in M["mutation"].values()); mut_n = sum(v["applied"] for v in M["mutation"].values())
adv = M["adversarial"]
contra = build_experiment(scenarios.get("contradictory")).decision
contra_smb = next(x for x in contra["disagreement"]["segments"] if x["value"] == "SMB")
noise_x = scenarios.SCENARIOS["high_variance"].run_noise_sd / scenarios.SCENARIOS["base"].run_noise_sd
stab = d["stability"]
n_stab = stab["n_tested"]; k_stab = round(stab["pct"] / 100 * n_stab)
answer = GroundedResponder(ex).answer("Did real customers prefer B?").text
answer_short = " ".join(answer.split(". ")[:2]).rstrip(".") + "."
chain = L.chain(L.get("overall.lead")["id"])
cls_counts = pd.Series([r["cls"] for r in L.all()]).value_counts().to_dict()


# ------------------------------------------------------------------ figure 1: fixture + who prefers what
def fig_fixture(path):
    fig = plt.figure(figsize=(7.3, 2.75), dpi=300)
    ax = fig.add_axes([0.0, 0.10, 0.25, 0.78]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_aspect("equal")
    order = ["B"] * round(sh["B"]) + ["A"] * round(sh["A"]) + ["C"] * (100 - round(sh["B"]) - round(sh["A"]))
    for i, o in enumerate(order):
        r, c = divmod(i, 10)
        ax.add_patch(FancyBboxPatch((c + 0.08, 9 - r + 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.12", fc=OPT[o], ec="none"))
    fig.text(0.0, 0.93, f"{pop['alive'].sum():,} synthetic agents", fontsize=11, weight="bold", color=INK)
    fig.text(0.0, 0.015, "each square ≈ 80 agents", fontsize=8.5, color=MUTED)
    bx = fig.add_axes([0.47, 0.17, 0.52, 0.66])
    rows = [("Overall", None, {o: sh[o] for o in "ABC"})] + [(f"{k}  ({v['w']:.0f}%)", k, {o: v[o] for o in "ABC"}) for k, v in seg.items()]
    for i, (lab, _, vals) in enumerate(rows[::-1]):
        left = 0
        top = max(vals, key=vals.get)
        for o in "ABC":
            bx.barh(i, vals[o], left=left, color=OPT[o], height=0.68, edgecolor="white", linewidth=1.2)
            if vals[o] >= 9:
                bx.text(left + vals[o] / 2, i, f"{vals[o]:.0f}", ha="center", va="center", color="white", fontsize=9.5, weight="bold" if o == top else "normal")
            left += vals[o]
        bx.text(-1.5, i, lab, ha="right", va="center", fontsize=10, weight="bold" if i == len(rows) - 1 else "normal", color=INK)
    bx.set_xlim(0, 100); bx.set_ylim(-0.6, len(rows) - 0.4); bx.axis("off")
    fig.text(0.47, 0.93, "Who prefers what (%)", fontsize=11, weight="bold", color=INK)
    for k, (o, lab) in enumerate([("A", "A  Lower price"), ("B", "B  Better reliability"), ("C", "C  Easier setup")]):
        fig.text(0.47 + k * 0.175, 0.015, "■", color=OPT[o], fontsize=10, family="DejaVu Sans"); fig.text(0.492 + k * 0.175, 0.015, lab, fontsize=9, color=INK)
    fig.savefig(path, dpi=300); plt.close(fig)


# ------------------------------------------------------------------ figure 2: THE chart, from the same 20 seeded runs
def flip_curve():
    xs = np.round(np.linspace(0.36, 1.08, 37), 4)
    M_ = np.zeros((len(xs), spec.n_runs))
    for i, x in enumerate(xs):
        a = dict(a0); a["reliability_sensitivity"] = float(x)
        for r in range(spec.n_runs):
            s = E.simulate_run(pop, spec, a, np.random.default_rng(spec.seed * 1000 + r))["overall"] * 100
            M_[i, r] = s[1] - max(s[0], s[2])
    return xs, M_


def fig_flip(path, xs, M_):
    mean, p10, p90 = M_.mean(1), np.percentile(M_, 10, axis=1), np.percentile(M_, 90, axis=1)
    i0 = int(np.argmin(np.abs(xs - nominal_rel)))
    cross = next(i for i in range(len(xs) - 1) if mean[i] <= 0 < mean[i + 1])
    xf = xs[cross] + (0 - mean[cross]) * (xs[cross + 1] - xs[cross]) / (mean[cross + 1] - mean[cross])
    fig, ax = plt.subplots(figsize=(7.3, 3.75), dpi=300)
    lo_t, hi_t = C.ASSUMPTIONS["reliability_sensitivity"]["tested"]
    ax.axvspan(lo_t, hi_t, color=LIGHT, zorder=0)
    ax.text(lo_t + 0.008, -11.2, "tested range (±20%)", ha="left", va="bottom", fontsize=9, color=MUTED)
    ax.axvspan(xs[0], xf, ymin=0, ymax=1, color="#FBE9DF", zorder=0)
    ax.axhline(0, color=INK, lw=1.1, zorder=1)
    ax.fill_between(xs, p10, p90, color=TEAL, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, mean, color=TEAL, lw=3, zorder=3)
    ax.axvline(nominal_rel, color=INK, ls=(0, (3, 3)), lw=1, zorder=1)
    ax.plot([nominal_rel], [mean[i0]], "o", color=INK, ms=9, zorder=5, mec="white", mew=1.8)
    ax.plot([xf], [0], "o", color=ORG, ms=11, zorder=5, mec="white", mew=1.8)
    ax.annotate(f"Nominal assumption: {nominal_rel:.2f}\nB leads by {mean[i0]:+.1f}pp", (nominal_rel, mean[i0]), xytext=(nominal_rel + 0.07, -6.0),
                fontsize=11, color=INK, weight="bold", va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
    ax.annotate(f"Flip point: {flip_rel['value']:.2f}\n{abs(nominal_rel - flip_rel['value']) / nominal_rel:.0%} below nominal", (xf, 0), xytext=(0.375, 15.5), ha="left", fontsize=11, color=ORG, weight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=ORG, lw=0.9))
    ax.text(xs[0] + 0.012, 36.5, "B is no longer the preferred option", fontsize=10, color="#8A3509", va="top")
    ax.text(xs[-1] - 0.01, 2.2, "shaded band: 10th–90th percentile\nof 20 simulated runs", ha="right", va="bottom", fontsize=8.5, color=TEAL)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-12, 38)
    ax.set_xlabel("Reliability sensitivity (how strongly reliability need drives choosing B)", fontsize=10.5)
    ax.set_ylabel("B's lead over the best other option (pp)", fontsize=10.5)
    ax.grid(axis="y", color="#E3E7EE", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return float(xf), float(mean[i0])


def fig_stability(path):
    fig = plt.figure(figsize=(7.3, 2.25), dpi=300)
    fig.text(0.0, 0.93, "Decision stability", fontsize=11.5, weight="bold", color=INK)
    fig.text(0.0, 0.73, f"{stab['pct']:.1f}%", fontsize=30, weight="bold", color=TEAL, va="center")
    fig.text(0.0, 0.575, f"of {n_stab} tested assumption sets keep B preferred", fontsize=10, color=INK)
    ax = fig.add_axes([0.0, 0.40, 0.40, 0.11]); ax.axis("off"); ax.set_xlim(0, n_stab); ax.set_ylim(0, 1)
    ax.barh(0.5, k_stab, color=TEAL, height=1, edgecolor="none"); ax.barh(0.5, n_stab - k_stab, left=k_stab, color=ORG, height=1, edgecolor="none")
    for yy, col, txt in [(0.285, TEAL, f"B preferred: {k_stab}"), (0.185, ORG, f"another option preferred: {n_stab - k_stab}")]:
        fig.add_artist(Rectangle((0.0, yy - 0.012), 0.014, 0.05, transform=fig.transFigure, color=col))
        fig.text(0.024, yy, txt, fontsize=9.5, color=col, weight="bold", va="baseline")
    fig.text(0.0, 0.045, "Latin-hypercube samples; each assumption within its tested range.", fontsize=8.3, color=MUTED, va="bottom")
    bx = fig.add_axes([0.62, 0.17, 0.37, 0.64])
    names = {"reliability_sensitivity": "Reliability sensitivity", "price_sensitivity_scale": "Price sensitivity", "technical_shift": "Technical-buyer mix",
             "enterprise_share": "Enterprise share", "temperature": "Choice randomness", "setup_sensitivity": "Setup sensitivity"}
    order = list(sens)[::-1]
    bx.barh([names[k] for k in order], [sens[k]["swing_pp"] for k in order], color=[ORG if sens[k]["flips_in_range"] else TEAL for k in order], height=0.62)
    for i, k in enumerate(order):
        bx.text(sens[k]["swing_pp"] + 0.5, i, f"{sens[k]['swing_pp']:.1f}", va="center", fontsize=9.5, color=INK, weight="bold" if sens[k]["flips_in_range"] else "normal")
    bx.set_xlim(0, 33); bx.tick_params(axis="y", length=0, labelsize=10); bx.tick_params(axis="x", labelsize=8.5); bx.spines["left"].set_visible(False)
    bx.set_xlabel("swing in B's lead across the tested range (pp)", fontsize=9)
    fig.text(0.62, 0.93, "What matters most?", fontsize=11.5, weight="bold", color=INK)
    fig.savefig(path, dpi=300); plt.close(fig)


def fig_architecture(path):
    fig = plt.figure(figsize=(7.3, 0.62), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")
    boxes = [("Simulation", "synthetic agents choose", GREY), ("Evidence ledger", "input · generated · derived", INK), ("Deterministic analysis", "stability · segments · flips", INK), ("Decision presentation", "changes wording with evidence", TEAL)]
    w, gap = 21.5, 4.67
    for i, (t, sub, col) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.2), w, 7.6, boxstyle="round,pad=0,rounding_size=0.8", fc=col if col != GREY else "#D5DAE3", ec="none"))
        ax.text(x + w / 2, 6.0, t, ha="center", va="center", fontsize=9.5, weight="bold", color="white" if col != GREY else INK)
        ax.text(x + w / 2, 3.0, sub, ha="center", va="center", fontsize=7.6, color="#E8ECF4" if col != GREY else MUTED)
        if i < 3:
            ax.annotate("", (x + w + gap - 0.4, 5.0), (x + w + 0.4, 5.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    fig.savefig(path, dpi=300); plt.close(fig)


def fig_classes(path):
    fig = plt.figure(figsize=(7.3, 1.35), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 14); ax.axis("off")
    items = [("INPUT", "assumptions and\nconfiguration", f"{cls_counts.get('INPUT', 0)} ledger rows", MUTED, False),
             ("GENERATED", "simulated agent\nresponses", f"{cls_counts.get('GENERATED', 0)} ledger rows", ORG, False),
             ("DERIVED", "stability, segment effects,\nflip thresholds", f"{cls_counts.get('DERIVED', 0)} ledger rows", TEAL, False),
             ("OBSERVED", "real customer\noutcomes", "none supplied", "#2E8B57", True)]
    w, gap = 21.5, 4.67
    for i, (t, sub, cnt, col, dashed) in enumerate(items):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.8), w, 12.4, boxstyle="round,pad=0,rounding_size=0.9", fc="white" if dashed else col, ec=col, lw=1.6, ls=(0, (4, 3)) if dashed else "-"))
        tc = col if dashed else "white"
        ax.text(x + w / 2, 11.0, t, ha="center", va="center", fontsize=10.5, weight="bold", color=tc)
        if dashed:
            ax.text(x + w / 2, 6.9, "∅", ha="center", va="center", fontsize=24, color=col, family="DejaVu Sans")
        else:
            ax.text(x + w / 2, 6.9, sub, ha="center", va="center", fontsize=8.8, color="white", linespacing=1.15)
        ax.text(x + w / 2, 2.7, cnt, ha="center", va="center", fontsize=8.5, color=tc, style="italic")
        if i < 3:
            ax.annotate("", (x + w + gap - 0.4, 7.0), (x + w + 0.4, 7.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1, ls=(0, (2, 2)) if i == 2 else "-"))
    fig.savefig(path, dpi=300); plt.close(fig)


def banner(name, out):
    im = PILImage.open(ROOT / "figures" / f"screen_{name}.png").convert("RGB").crop((20, 86, 1100, 172))
    im.save(out)


# ------------------------------------------------------------------ build figures
xs, M_ = flip_curve()
xf, nominal_lead = fig_flip(FIG / "flip.png", xs, M_)
fig_fixture(FIG / "fixture.png"); fig_stability(FIG / "stability.png"); fig_architecture(FIG / "arch.png"); fig_classes(FIG / "classes.png")
banner("base", FIG / "banner_base.png"); banner("near_tie", FIG / "banner_tie.png")
assert abs(nominal_lead - lead) < 0.05, (nominal_lead, lead)           # the chart's nominal point IS the headline number

# ------------------------------------------------------------------ styles
def S(**k):
    base = dict(fontName="Carlito", textColor=CI(INK))
    base.update(k)
    return ParagraphStyle("s", **base)

H1 = S(fontName="Carlito-Bold", fontSize=27, leading=29.5, spaceAfter=3)
H2 = S(fontName="Carlito-Bold", fontSize=25, leading=27.5, spaceAfter=4)
KICK = S(fontName="Carlito-Bold", fontSize=9.5, leading=12, textColor=CI(ORG))
SUB = S(fontSize=12.5, leading=16, textColor=CI(MUTED))
BODY = S(fontSize=11, leading=15)
SMALL = S(fontSize=8.8, leading=11.5, textColor=CI(MUTED))
LEAD = S(fontSize=12, leading=16.5)
CARDT = S(fontName="Carlito-Bold", fontSize=9, leading=11, textColor=CI(MUTED))
BIG = lambda col, sz=34: S(fontName="Carlito-Bold", fontSize=sz, leading=sz + 9, textColor=CI(col))


def P(t, st=BODY):
    return Paragraph(t, st)


def img(path, width):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def box(rows, widths, bg=LIGHT, pad=10, border=None, extra=()):
    t = Table(rows, colWidths=widths)
    st = [("BACKGROUND", (0, 0), (-1, -1), CI(bg)), ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
          ("TOPPADDING", (0, 0), (-1, -1), pad - 2), ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 2), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if border:
        st.append(("BOX", (0, 0), (-1, -1), 1.2, CI(border)))
    t.setStyle(TableStyle(st + list(extra)))
    return t


def chrome(c, doc):
    c.saveState()
    pw, ph = letter
    m = 0.62 * inch
    c.setFont("Carlito-Bold", 9); c.setFillColor(CI(INK)); c.drawString(m, ph - 0.42 * inch, "DecisionTrace")
    c.setFont("Carlito", 9); c.setFillColor(CI(MUTED)); c.drawRightString(pw - m, ph - 0.42 * inch, "When should you trust a simulated decision?")
    c.setStrokeColor(CI("#DCE1E8")); c.setLineWidth(0.6); c.line(m, ph - 0.50 * inch, pw - m, ph - 0.50 * inch)
    c.setFont("Carlito", 8.5); c.setFillColor(CI(MUTED))
    c.drawString(m, 0.38 * inch, f"Synthetic experiment. No real customer data.  ·  {REPO}")
    c.drawRightString(pw - m, 0.38 * inch, f"{doc.page} / 4")
    c.restoreState()


def spc(h):
    return Spacer(1, h)


story = []
# ================================================================== PAGE 1
story += [P("DECISIONTRACE  ·  RIDDHIMA  ·  OCTOBER 2026", KICK), spc(4),
          P("When should you trust a simulated decision?", H1),
          P("Evidence, uncertainty, and decision stability for AI simulations", SUB), spc(9),
          P("A simulation says <b>Option B is likely to outperform Option A.</b> A decision-maker&rsquo;s next questions are: "
            "<b>Why? How certain? Which populations drove it? What does it depend on? What would flip it?</b>", LEAD), spc(8),
          img(FIG / "fixture.png", W), spc(6)]
cap = S(fontSize=10, textColor=CI(MUTED))
res = Table([[P(f"{sh['B']:.1f}%", BIG(TEAL, 38)), P(f"{sh['A']:.1f}%", BIG(ORG, 38)), P(f"{sh['C']:.1f}%", BIG(MUTED, 38)), P(f"+{lead:.1f}pp", BIG(TEAL, 38))],
             [P("B  Better reliability", cap), P("A  Lower price", cap), P("C  Easier setup", cap), P("B leads A", S(fontName="Carlito-Bold", fontSize=10, textColor=CI(TEAL)))]],
            colWidths=[W * 0.25] * 4)
res.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                         ("LINEABOVE", (0, 0), (-1, 0), 1.2, CI(INK)), ("LINEAFTER", (2, 0), (2, 1), 0.8, CI("#DCE1E8")), ("LEFTPADDING", (3, 0), (3, 1), 16)]))
story += [res, spc(14), P("The headline hides two problems.", S(fontName="Carlito-Bold", fontSize=18, leading=21, textColor=CI(ORG))), spc(7)]
gw = (W - 12) / 2
p1 = [P(f"{smb_ahead:.1f}pp", BIG(ORG, 40)),
      P(f"<b>SMB is {seg['SMB']['w']:.0f}% of the population, and prefers A.</b> There, A leads B by {smb_ahead:.1f}pp "
        f"({smb['A']:.1f}% vs {smb['B']:.1f}%). The aggregate &ldquo;B wins&rdquo; is carried by Mid-market and Enterprise.", S(fontSize=11, leading=14.5))]
p2 = [P(f"{flip_rel['value']:.2f}", BIG(ORG, 40)),
      P(f"<b>B&rsquo;s lead flips when reliability sensitivity falls below {flip_rel['value']:.2f}.</b> The nominal assumption is {nominal_rel:.2f}. "
        f"That is one unvalidated input, and a {abs(nominal_rel - flip_rel['value']) / nominal_rel:.0%} change.", S(fontSize=11, leading=14.5))]
two = Table([[p1, "", p2]], colWidths=[gw, 12, gw])
two.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), CI("#FDEEE4")), ("BACKGROUND", (2, 0), (2, 0), CI("#FDEEE4")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                         ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                         ("LEFTPADDING", (1, 0), (1, 0), 0), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
story += [two, spc(14), img(FIG / "arch.png", W), spc(9),
          box([[P("<b>Scope.</b> A synthetic test fixture. Not a model of real buyers. No real customer data. An independent experiment, not any company&rsquo;s models, data or methodology.", S(fontSize=9.5, leading=12.5))]], [W], bg="#F4F6FA", pad=10),
          PageBreak()]

# ================================================================== PAGE 2
story += [P("What would flip the decision?", H2),
          P("Hold everything else at nominal and move one assumption: how much reliability matters to the simulated buyers.", LEAD), spc(8),
          img(FIG / "flip.png", W),
          P(f"Each point is the mean of the same {spec.n_runs} seeded runs used for the headline, so the nominal point matches page 1. The deterministic analysis in the repo puts the flip at {flip_rel['value']:.3f}.", SMALL), spc(10),
          img(FIG / "stability.png", W), spc(10)]
take = box([[P("The decision rests mostly on an unvalidated simulation input, <u>not</u> population mix.", S(fontName="Carlito-Bold", fontSize=20, leading=23, textColor=colors.white))],
            [P(f"Reliability sensitivity moves B&rsquo;s lead by {sens['reliability_sensitivity']['swing_pp']:.0f}pp across its tested range. Enterprise share moves it {sens['enterprise_share']['swing_pp']:.1f}pp. "
               "A simulation can produce a decisive-looking answer while the decision itself stays fragile to assumptions nobody has validated. That is a product problem.", S(fontSize=11, leading=15, textColor=CI("#E3E8F2")))]],
           [W], bg=INK, pad=16)
story += [take, PageBreak()]

# ================================================================== PAGE 3
card_defs = [
    ("MISSING SEGMENT", "INSUFFICIENT EVIDENCE", RED, "No Enterprise agents in the population.", f"{scen.loc['missing_segment','winner']} would win instead."),
    ("HIGH VARIANCE", "UNSTABLE", RED, f"Simulator noise ×{noise_x:.0f}.", f"B is ahead in only {scen.loc['high_variance','win_rate_pct']:.0f}% of runs. No robust preference."),
    ("SUBGROUP CONFLICT", "CONTESTED", PURPLE, f"Aggregate favors B by {scen.loc['contradictory','lead_pp']:.1f}pp.", f"SMB ({contra_smb['weight_pct']:.0f}% of the population) prefers A by a wide margin."),
    ("STALE SIMULATION", "OUT OF DATE", SLATE, "The run used input v1; the current input is v2.", "Re-run before using."),
]


def card(label, state, col, l1, l2):
    t = Table([[P(label, CARDT)], [P(state, S(fontName="Carlito-Bold", fontSize=15, leading=18, textColor=CI(col)))], [P(l1, S(fontSize=11, leading=14))], [P(l2, S(fontSize=11, leading=14, textColor=CI(MUTED)))]], colWidths=[(W - 12) / 2])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CI("#F4F6FA")), ("LINEABOVE", (0, 0), (-1, 0), 3, CI(col)), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 10)]))
    return t


grid = Table([[card(*card_defs[0]), "", card(*card_defs[1])], [spc(9), "", ""], [card(*card_defs[2]), "", card(*card_defs[3])]], colWidths=[(W - 12) / 2, 12, (W - 12) / 2])
grid.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
small = [("POPULATION IMBALANCE", "INSUFFICIENT EVIDENCE", f"Enterprise is 25% of the result from 169 agents."),
         ("INSUFFICIENT SAMPLE", "INSUFFICIENT EVIDENCE", f"Only 300 agents (minimum 1,000)."),
         ("NEAR TIE", "UNSTABLE", f"B ahead by {scen.loc['near_tie','lead_pp']:.1f}pp, in {scen.loc['near_tie','win_rate_pct']:.0f}% of runs.")]
srow = Table([[[P(a, CARDT), P(f"<b>{b}</b>", S(fontSize=10, leading=12, textColor=CI(RED))), P(c, S(fontSize=9.5, leading=12))] for a, b, c in small]], colWidths=[W / 3] * 3)
srow.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, CI("#DCE1E8")), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story += [P("When should the product<br/>stop declaring a winner?", H2),
          P(f"Seven controlled degradations of the same experiment. Each changes what the product is willing to say, even where the headline lead barely moves.", LEAD), spc(10),
          grid, spc(9), srow, spc(10),
          P("Weak-evidence states never claim a leader.", S(fontName="Carlito-Bold", fontSize=23, leading=26, textColor=CI(TEAL))), spc(8)]
naive = box([[[P("A NAIVE PRODUCT SAYS", CARDT), P("B wins.", S(fontName="Carlito-Bold", fontSize=40, leading=46, textColor=CI(MUTED)))],
               P("Same run. Same data.<br/>No uncertainty, no dissent, no assumptions shown.", S(fontSize=11, leading=15, textColor=CI(MUTED)))]], [W * 0.40, W * 0.60], bg="#F4F6FA", pad=12)
dtsays = Table([[P("DECISIONTRACE SAYS, SAME RUN", CARDT)], [img(FIG / "banner_base.png", W)],
                [P(f"<b>The aggregate result masks significant population disagreement.</b> SMB prefers A, {smb_ahead:.1f}pp ahead of B. <b>Most sensitive assumption:</b> reliability sensitivity.", S(fontSize=10.5, leading=13.5))]], colWidths=[W])
dtsays.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
weak = Table([[P(f"AND WHEN THE EVIDENCE WEAKENS: A NEAR TIE, B AHEAD BY ONLY {scen.loc['near_tie','lead_pp']:.1f}pp", CARDT)], [img(FIG / "banner_tie.png", W)]], colWidths=[W])
weak.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
cmp_ = [naive, spc(8), dtsays, spc(8), weak]
story += cmp_ + [PageBreak()]

# ================================================================== PAGE 4
import re as _re
col_of = {"DERIVED": TEAL, "GENERATED": ORG, "INPUT": MUTED}
layers = [("overall.lead", "DERIVED")]
for cls_ in ("DERIVED", "GENERATED", "INPUT"):
    keys_ = [c["key"] for c in chain if c["cls"] == cls_]
    if keys_:
        layers.append((", ".join(sorted(keys_)), cls_))
chain_txt = "  ←  ".join(f"<font name='Mono' size='7.4'>{k}</font> <font size='8' color='{col_of[c]}'><b>{c}</b></font>" for k, c in layers)
sent = _re.split(r"(?<=\.)\s+", answer.strip())
ans_html = f"<b>{sent[0]}</b> " + sent[1].replace("[EV-0005]", "<font name='Mono' size='8' color='#145E59'>[EV-0005]</font>")
qa = box([[P("A USER ASKS", CARDT), P("THE SYSTEM ANSWERS", CARDT)],
          [P("<b>Did real customers prefer B?</b>", S(fontSize=15, leading=18)),
           P(ans_html, S(fontSize=11, leading=14.5))]],
         [W * 0.34, W * 0.66], bg="#F4F6FA", pad=12)
stat_t = lambda big, col, sub, small_: [P(big, BIG(col, 27)), P(sub, S(fontSize=10.5, leading=13.5)), P(small_, SMALL)]
evs = Table([[stat_t(f"{n_ans}/{n_ans}", TEAL, "grounded answers pass all five checks", "citation · numerical fidelity · observed vs generated · uncertainty · overreach"),
              stat_t(f"{mut_ok:,}/{mut_n:,}", TEAL, "deliberately corrupted answers caught", "dropped citations, altered numbers, added certainty, invented causes"),
              stat_t(f"{adv['caught']}/{adv['overreach_cases']}", ORG, "subtle, novel overreach answers caught", f"{adv['false_positives']} of {adv['clean_cases']} clean paraphrases flagged. High precision, low recall.")]],
            colWidths=[W / 3] * 3)
evs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 14), ("LINEABOVE", (0, 0), (-1, 0), 1.2, CI(INK)), ("TOPPADDING", (0, 0), (-1, -1), 8)]))
nxt = [("Real observed outcomes", "Compare the simulation with reality."), ("LLM judge + human audit", "Catch semantic overreach the lexicons miss."),
       ("Calibration", "Does reported stability match actual decision robustness?"), ("Other population models", "How much does DecisionTrace itself depend on the simulator?")]
nx = Table([[P(f"<b>{a}</b><br/>{b}", S(fontSize=10, leading=12.5)) for a, b in nxt[:2]], [P(f"<b>{a}</b><br/>{b}", S(fontSize=10, leading=12.5)) for a, b in nxt[2:]]], colWidths=[W / 2] * 2)
nx.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story += [P("Don&rsquo;t let explanation become evidence.", H2),
          P("A follow-up answer can only cite the ledger, and the ledger knows what kind of evidence each row is.", LEAD), spc(8), qa, spc(10),
          img(FIG / "classes.png", W), spc(5),
          P("Every number in an answer links to a ledger row and its provenance chain. The overall lead, for example, traces back through: " + chain_txt + ".", SMALL), spc(12),
          P("Then I tried to break the evaluator.", S(fontName="Carlito-Bold", fontSize=18, leading=21, textColor=CI(ORG))), spc(5), evs, spc(4),
          P("The responder and the checks were built together, so the 100% measures consistency, not generalization. The corrupted answers use the phrases the checks were written for. The seven subtle ones do not.", SMALL), spc(8),
          box([[P("<b>Provenance can constrain what an explanation is allowed to claim.</b> But deterministic semantic guardrails do not generalize well enough on their own.", S(fontSize=13, leading=16.5, textColor=colors.white))]], [W], bg=INK, pad=14), spc(10),
          P("What I&rsquo;d test next", S(fontName="Carlito-Bold", fontSize=13, leading=16)), nx, spc(6),
          P("Independent synthetic experiment. Not any company&rsquo;s models, data, APIs or methodology.", S(fontSize=9, textColor=CI(MUTED)))]

doc = BaseDocTemplate(str(OUT / "DecisionTrace_note.pdf"), pagesize=letter, leftMargin=0.62 * inch, rightMargin=0.62 * inch, topMargin=0.72 * inch, bottomMargin=0.6 * inch,
                      title="DecisionTrace: When should you trust a simulated decision?", author="Riddhima", subject="Evidence, uncertainty, and decision stability for AI simulations")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=chrome)])
doc.build(story)
from pypdf import PdfReader                                           # noqa: E402
print("pages:", len(PdfReader(str(OUT / "DecisionTrace_note.pdf")).pages), "| flip x:", round(xf, 3), "| nominal lead:", round(nominal_lead, 2), "| smb ahead:", round(smb_ahead, 1))
