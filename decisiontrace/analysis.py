"""From simulation output to an inspectable decision.

Everything here is deterministic code. Nothing in this module calls an LLM.
Pipeline:  runs -> ledger (INPUT/GENERATED/DERIVED) -> segments -> perturbations -> flip thresholds
           -> decision state + headline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from . import config as C
from . import engine as E
from .ledger import Ledger

PP = 100.0


def fmt(v, unit: str) -> str:
    """Canonical display of an evidence value. The scorer and the responder both use this."""
    if v is None:
        return "n/a"
    if unit == "pct":
        return f"{v:.1f}%"
    if unit == "pp":
        return f"{v:+.1f}pp"
    if unit == "ppabs":
        return f"{v:.1f}pp"
    if unit == "count":
        return f"{int(v):,}"
    if unit in ("x", "coef", "ratio", "share", "rel"):
        return f"{v:.2f}"
    return str(v)


@dataclass
class Experiment:
    spec: E.Spec
    ledger: Ledger
    decision: dict
    pop: dict
    a0: dict


# ------------------------------------------------------------------ perturbation machinery
def lhs(n, d, rng):
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(cut[:, j])
    return cut


def margin_vs_best_other(shares, winner_idx):
    other = np.delete(shares, winner_idx)
    return float(shares[winner_idx] - other.max())


def perturbed_samples(spec, n, rng):
    names = [k for k in C.ASSUMPTION_NAMES if not (k == "enterprise_share" and ("company_size", "Enterprise") in spec.drop_segments)]
    u = lhs(n, len(names), rng)
    out = []
    for row in u:
        a = spec.nominal()
        for j, k in enumerate(names):
            lo, hi = C.ASSUMPTIONS[k]["tested"]
            if k == "enterprise_share":
                lo, hi = spec.assumptions[k] * 0.85, spec.assumptions[k] * 1.15
            a[k] = lo + row[j] * (hi - lo)
        out.append(a)
    return out, names


def tested_range(spec, k):
    lo, hi = C.ASSUMPTIONS[k]["tested"]
    if k == "enterprise_share":
        return spec.assumptions[k] * 0.85, spec.assumptions[k] * 1.15
    nom = C.ASSUMPTIONS[k]["nominal"]
    if k in spec.assumptions and spec.assumptions[k] != nom:        # scenario shifted the nominal
        return lo + (spec.assumptions[k] - nom), hi + (spec.assumptions[k] - nom)
    return lo, hi


def flip_search(pop, spec, winner_idx, k, grid=161):
    """Nearest value of assumption k (others at nominal) on each side where the nominal winner stops winning."""
    a0 = spec.nominal()
    plo, phi = C.ASSUMPTIONS[k]["phys"]
    xs = np.linspace(plo, phi, grid)
    def m(x):
        a = dict(a0); a[k] = float(x)
        return margin_vs_best_other(E.expected_shares(pop, spec, a), winner_idx)
    ms = np.array([m(x) for x in xs])
    nom = a0[k]
    out = {"below": None, "above": None}
    i0 = int(np.argmin(np.abs(xs - nom)))
    for side, rng_idx in (("below", range(i0, 0, -1)), ("above", range(i0, grid - 1))):
        for i in rng_idx:
            j = i - 1 if side == "below" else i + 1
            if ms[i] > 0 >= ms[j]:
                lo_x, hi_x = (xs[j], xs[i]) if side == "below" else (xs[i], xs[j])
                for _ in range(40):
                    mid = (lo_x + hi_x) / 2
                    if side == "below":
                        if m(mid) > 0: hi_x = mid
                        else: lo_x = mid
                    else:
                        if m(mid) > 0: lo_x = mid
                        else: hi_x = mid
                out[side] = float((lo_x + hi_x) / 2)
                break
    return out


# ------------------------------------------------------------------ build
def build_experiment(spec: E.Spec, ledger: Ledger | None = None, exp_id: str | None = None) -> Experiment:
    pop = E.generate_population(spec)
    a0 = spec.nominal()
    L = ledger or Ledger()
    exp_id = exp_id or spec.name
    L.start(exp_id, spec.title, spec.question, spec.run_input_version, spec.current_input_version,
            json.dumps({k: v for k, v in asdict(spec).items() if k != "assumptions"}, default=str))
    rng = np.random.default_rng(spec.seed + 1)

    # ---------------- INPUT evidence
    for k, d in C.ASSUMPTIONS.items():
        lo, hi = tested_range(spec, k)
        L.add_assumption(k, a0[k], lo, hi, d["unit"], d["desc"])
    i_agents = L.add("input.n_agents", "INPUT", int(pop["alive"].sum()), "count", note="simulated agents in the population")
    i_runs = L.add("input.n_runs", "INPUT", spec.n_runs, "count", note="repeated simulation runs")
    i_opts = L.add("input.options", "INPUT", {o: C.OPTION_LABELS[o] for o in C.OPTIONS}, "", note="messages being compared")
    i_ver = L.add("input.version", "INPUT", {"run": spec.run_input_version, "current": spec.current_input_version}, "",
                  note="input data version the run was produced from vs the current version")
    i_obs = L.add("input.observed_outcome_data", "OBSERVED", False, "",
                  note="No observed customer outcome data was supplied. Every preference figure is simulated.")
    i_mix = L.add("input.population_mix", "INPUT", dict(zip(C.SEGMENTS["company_size"], spec.size_mix)), "share")
    ass_ids = {}
    for k in C.ASSUMPTION_NAMES:
        lo, hi = tested_range(spec, k)
        ass_ids[k] = L.add(f"input.assumption.{k}", "INPUT", a0[k], C.ASSUMPTIONS[k]["unit"], ci=(lo, hi),
                           note=C.ASSUMPTIONS[k]["desc"], details={"tested_lo": lo, "tested_hi": hi})

    # ---------------- GENERATED evidence: repeated runs
    runs = [E.simulate_run(pop, spec, a0, np.random.default_rng(spec.seed * 1000 + r)) for r in range(spec.n_runs)]
    overall = np.array([r["overall"] for r in runs]) * PP                      # (runs, 3)
    g_over = L.add("generated.runs.overall", "GENERATED", overall.round(3).tolist(), "pct",
                   depends_on=[i_agents, i_runs], assumptions=list(ass_ids.values()),
                   note="overall option shares in each simulated run", details={"columns": list(C.OPTIONS)})
    profile = E.segment_profile(pop, spec, a0)
    seg_runs, g_seg = {}, {}
    for key in E.segment_masks(pop):
        arr = np.array([r["segments"][key] for r in runs]) * PP
        seg_runs[key] = arr
        g_seg[key] = L.add(f"generated.segment.{key[0]}={key[1]}.runs", "GENERATED", np.nan_to_num(arr).round(3).tolist(), "pct",
                           depends_on=[i_agents, i_runs], assumptions=list(ass_ids.values()),
                           details={"columns": list(C.OPTIONS), "n_agents": profile[key]["n"]})

    # ---------------- DERIVED: overall
    mean = overall.mean(axis=0); sd = overall.std(axis=0, ddof=1) if spec.n_runs > 1 else np.zeros(3)
    order = np.argsort(-mean)
    w_i, r_i = int(order[0]), int(order[1])
    W, R = C.OPTIONS[w_i], C.OPTIONS[r_i]
    share_ids = {}
    for j, o in enumerate(C.OPTIONS):
        share_ids[o] = L.add(f"overall.share.{o}", "DERIVED", float(mean[j]), "pct",
                             ci=(float(mean[j] - 1.96 * sd[j]), float(mean[j] + 1.96 * sd[j])),
                             depends_on=[g_over], assumptions=list(ass_ids.values()),
                             note=f"mean simulated preference for {o} across {spec.n_runs} runs (band = +/-1.96 sd across runs)")
    margins = overall[:, w_i] - np.delete(overall, w_i, axis=1).max(axis=1)
    lead = float(mean[w_i] - mean[r_i])
    win_rate = float((margins > 0).mean()) * 100
    p05 = float(np.percentile(margins, 5))
    e_lead = L.add("overall.lead", "DERIVED", lead, "pp", ci=(float(margins.mean() - 1.96 * margins.std(ddof=1)), float(margins.mean() + 1.96 * margins.std(ddof=1))) if spec.n_runs > 1 else (None, None),
                   depends_on=[share_ids[W], share_ids[R]], details={"winner": W, "runner_up": R},
                   note=f"lead of {W} over {R}")
    e_win = L.add("overall.win_rate", "DERIVED", win_rate, "pct", depends_on=[g_over, i_runs],
                  details={"winner": W}, note=f"share of runs in which {W} had the highest preference")
    e_p05 = L.add("overall.margin_p05", "DERIVED", p05, "pp", depends_on=[g_over], details={"winner": W},
                  note="5th-percentile lead across runs (worst plausible run)")
    L.add("overall.winner", "DERIVED", W, "", depends_on=[share_ids[W]], details={"runner_up": R})

    # ---------------- DERIVED: segments (exact decomposition of the lead within each partition)
    seg_info = {}
    for key, arr in seg_runs.items():
        prof = profile[key]
        m = np.nanmean(arr, axis=0)
        if np.isnan(m).any() or prof["n"] == 0:
            continue
        segsd = np.nanstd(arr, axis=0, ddof=1) if spec.n_runs > 1 else np.zeros(3)
        so = np.argsort(-m)
        seg_win, seg_lead = C.OPTIONS[so[0]], float(m[so[0]] - m[so[1]])
        seg_margin = float(m[w_i] - m[r_i])
        contrib = prof["weight"] * seg_margin
        ids = {}
        for j, o in enumerate(C.OPTIONS):
            ids[o] = L.add(f"segment.{key[0]}={key[1]}.share.{o}", "DERIVED", float(m[j]), "pct",
                           ci=(float(m[j] - 1.96 * segsd[j]), float(m[j] + 1.96 * segsd[j])),
                           depends_on=[g_seg[key]], note=f"mean simulated preference for {o} within {key[1]}")
        e_n = L.add(f"segment.{key[0]}={key[1]}.n", "INPUT", prof["n"], "count", depends_on=[i_agents],
                    note=f"simulated agents in {key[1]}")
        e_wt = L.add(f"segment.{key[0]}={key[1]}.weight", "DERIVED", prof["weight"] * 100, "pct", depends_on=[i_agents],
                     assumptions=[ass_ids["enterprise_share"]], note=f"weighted share of the population in {key[1]}")
        e_mg = L.add(f"segment.{key[0]}={key[1]}.margin", "DERIVED", seg_margin * 1.0, "pp", depends_on=[ids[W], ids[R]],
                     details={"winner": W, "runner_up": R}, note=f"{W} minus {R} preference within {key[1]}")
        e_ct = L.add(f"segment.{key[0]}={key[1]}.contribution", "DERIVED", contrib, "pp", depends_on=[e_wt, e_mg],
                     details={"winner": W, "runner_up": R}, note=f"{key[1]}'s contribution to the overall {W}-{R} lead (weight x margin)")
        L.add(f"segment.{key[0]}={key[1]}.winner", "DERIVED", seg_win, "", depends_on=list(ids.values()), details={"lead_pp": seg_lead})
        drv = {}
        for attr in ("reliability_need", "price_sensitivity", "setup_friction", "technical_sophistication", "switching_cost"):
            v, ov = prof[attr], profile["__overall__"][attr]
            drv[attr] = L.add(f"segment.{key[0]}={key[1]}.driver.{attr}", "DERIVED", v, "ratio",
                              depends_on=[i_agents], assumptions=list(ass_ids.values()),
                              details={"overall": ov, "diff": (v - ov) if v is not None else None},
                              note=f"mean modeled {attr.replace('_', ' ')} of {key[1]} agents (overall {ov:.2f})")
        seg_info[key] = {"weight": prof["weight"], "n": prof["n"], "mean": m, "winner": seg_win, "lead": seg_lead,
                         "margin_pp": seg_margin, "contribution_pp": contrib, "ids": {"contribution": e_ct, "margin": e_mg, "n": e_n, "weight": e_wt}}
    # check: contributions decompose the lead exactly within each partition
    for dim in C.SEGMENTS:
        cs = [v["contribution_pp"] for k, v in seg_info.items() if k[0] == dim]
        if cs and len(cs) == len(C.SEGMENTS[dim]):
            assert abs(sum(cs) - lead) < 1e-6, (dim, sum(cs), lead)

    # ---------------- disagreement
    dis = []
    for key, v in seg_info.items():
        if v["weight"] >= C.DISAGREE_MIN_WEIGHT and v["n"] >= C.MIN_SEGMENT_N and v["winner"] != W and v["lead"] >= C.DISAGREE_MIN_MARGIN_PP:
            dis.append({"segment": f"{key[0]}={key[1]}", "dim": key[0], "value": key[1], "winner": v["winner"],
                        "lead_pp": v["lead"], "weight_pct": v["weight"] * 100,
                        "ahead_of_winner_pp": float(v["mean"][C.OPTIONS.index(v["winner"])] - v["mean"][w_i])})
    dis_weight_size = sum(d["weight_pct"] for d in dis if d["dim"] == "company_size" and d["lead_pp"] >= C.DISAGREE_STRONG_MARGIN_PP) / 100
    contested = dis_weight_size > 0.5
    e_dis = L.add("disagreement.flag", "DERIVED", bool(dis), "", depends_on=[v["ids"]["margin"] for v in seg_info.values()],
                  details={"segments": dis, "weight_of_dissenting_company_sizes": dis_weight_size},
                  note="segments (>=10% of population, >=3pp lead) whose preferred option differs from the overall winner")

    # ---------------- perturbation analysis: stability, tornado, flip thresholds, stress tests
    samples, names = perturbed_samples(spec, spec.n_lhs, np.random.default_rng(spec.seed + 2))
    sm = np.array([margin_vs_best_other(E.expected_shares(pop, spec, a), w_i) for a in samples])
    stability = float((sm > 0).mean()) * 100
    st_label = next((lab for thr, lab in C.STABILITY_BANDS if stability / 100 >= thr), "UNSTABLE")
    g_pert = L.add("generated.perturbation_grid", "GENERATED", {"n_samples": len(samples), "factors": names}, "",
                   depends_on=[i_agents] + [ass_ids[k] for k in names], assumptions=[ass_ids[k] for k in names],
                   note=f"{len(samples)} Latin-hypercube assumption sets over the tested ranges, noise-free expected shares")
    e_stab = L.add("sens.stability", "DERIVED", stability, "pct", depends_on=[g_pert], details={"winner": W, "n_tested": len(samples), "label": st_label},
                   assumptions=[ass_ids[k] for k in names],
                   note=f"share of tested assumption sets under which {W} remains the preferred option")
    L.add("sens.stability_label", "DERIVED", st_label, "", depends_on=[e_stab])
    L.add("sens.margin_quantiles", "DERIVED", {"p05": float(np.percentile(sm, 5)), "p50": float(np.percentile(sm, 50)), "p95": float(np.percentile(sm, 95))},
          "pp", depends_on=[g_pert])

    nominal_margin = margin_vs_best_other(E.expected_shares(pop, spec, a0), w_i)
    tornado, flips = [], []
    for k in names:
        lo, hi = tested_range(spec, k)
        mlo = margin_vs_best_other(E.expected_shares(pop, spec, {**a0, k: lo}), w_i)
        mhi = margin_vs_best_other(E.expected_shares(pop, spec, {**a0, k: hi}), w_i)
        swing = abs(mhi - mlo)
        idx = swing / max(abs(nominal_margin), 1e-9)
        fl = flip_search(pop, spec, w_i, k)
        within = {s: (fl[s] is not None and lo <= fl[s] <= hi) for s in fl}
        ev = L.add(f"sens.{k}.swing", "DERIVED", swing * PP, "ppabs", depends_on=[g_pert, ass_ids[k]], assumptions=[ass_ids[k]],
                   details={"margin_at_low_pp": mlo * PP, "margin_at_high_pp": mhi * PP, "index": idx, "tested": [lo, hi]},
                   note=f"change in {W}'s lead across the tested range of {k}")
        tornado.append({"assumption": k, "swing_pp": swing * PP, "margin_lo_pp": mlo * PP, "margin_hi_pp": mhi * PP, "index": idx, "evidence": ev,
                        "flips_in_range": (mlo > 0) != (mhi > 0)})
        for side in ("below", "above"):
            if fl[side] is not None:
                unit = "pct" if k == "enterprise_share" else C.ASSUMPTIONS[k]["unit"]
                val = fl[side] * (100 if k == "enterprise_share" else 1)
                fe = L.add(f"flip.{k}.{side}", "DERIVED", val, unit, depends_on=[g_pert, ass_ids[k]], assumptions=[ass_ids[k]],
                           details={"within_tested_range": bool(within[side]), "nominal": a0[k], "tested": [lo, hi], "winner": W, "side": side},
                           note=f"{W} stops being the preferred option when {k} goes {side} this value (others at nominal)")
                flips.append({"assumption": k, "side": side, "value": val, "unit": unit, "within_tested_range": bool(within[side]), "evidence": fe})
    tornado.sort(key=lambda t: -t["index"])
    most = tornado[0]["assumption"]
    e_most = L.add("sens.most_sensitive", "DERIVED", most, "", depends_on=[t["evidence"] for t in tornado],
                   details={"index": tornado[0]["index"]}, note="assumption whose tested range moves the lead the most relative to the lead itself")

    stress = []
    for dim, val in [("company_size", s) for s in C.SEGMENTS["company_size"]] + [("buyer_type", s) for s in C.SEGMENTS["buyer_type"]]:
        if (dim, val) in spec.drop_segments:
            continue
        sp2 = E.Spec(**{**{k: getattr(spec, k) for k in spec.__dataclass_fields__}, "drop_segments": tuple(spec.drop_segments) + ((dim, val),)})
        try:
            sh = E.expected_shares(pop_with_drop(pop, sp2), sp2, a0)
        except Exception:
            continue
        o = np.argsort(-sh)
        item = {"dropped": f"{dim}={val}", "winner": C.OPTIONS[o[0]], "lead_pp": float((sh[o[0]] - sh[o[1]]) * PP), "same_winner": C.OPTIONS[o[0]] == W}
        stress.append(item)
        L.add(f"stress.drop.{dim}={val}", "DERIVED", item["winner"], "", depends_on=[g_pert],
              details={"lead_pp": item["lead_pp"], "same_winner": item["same_winner"]},
              note=f"preferred option if {val} agents were missing from the population")

    # ---------------- decision state (the product's honesty policy)
    stale = spec.run_input_version != spec.current_input_version
    n_agents = int(pop["alive"].sum())
    coverage = []                       # the headline would lean on agents we do not have
    for dim, val in spec.drop_segments:
        coverage.append(f"no {val} agents in the population")
    for key, v in seg_info.items():
        if key[0] == "company_size" and v["n"] < C.MIN_SEGMENT_N and v["weight"] >= C.DISAGREE_MIN_WEIGHT:
            coverage.append(f"{key[1]} is {v['weight'] * 100:.0f}% of the weighted result but only {v['n']} agents (minimum {C.MIN_SEGMENT_N})")
    ci_half = 1.96 * margins.std(ddof=1) if spec.n_runs > 1 else 0.0
    unstable_reasons = []
    if p05 <= 0:
        unstable_reasons.append(f"{W} is not ahead in every plausible run (worst-case gap {p05:+.1f}pp)")
    if win_rate / 100 < C.WIN_RATE_MIN:
        unstable_reasons.append(f"{W} is ahead in only {win_rate:.0f}% of runs")
    if stability / 100 < 0.5:
        unstable_reasons.append(f"{W} survives only {stability:.0f}% of tested assumption sets")
    if lead < C.NEAR_TIE_PP:
        unstable_reasons.append(f"near tie: the gap is {lead:.1f}pp")
    if stale:
        state = "STALE"
    elif n_agents < C.MIN_AGENTS or coverage:
        state = "INSUFFICIENT_EVIDENCE"
    elif unstable_reasons:
        state = "UNSTABLE"
    elif contested:
        state = "CONTESTED"
    elif st_label == "HIGH" and win_rate / 100 >= C.ROBUST_WIN_RATE:
        state = "ROBUST"
    elif st_label in ("HIGH", "MODERATE"):
        state = "MODERATE"
    else:
        state = "LOW"
    conf = {"ROBUST": "High", "MODERATE": "Moderate", "LOW": "Low", "CONTESTED": "Low",
            "UNSTABLE": "Insufficient", "INSUFFICIENT_EVIDENCE": "Insufficient", "STALE": "Insufficient"}[state]

    warnings = []
    for key, v in seg_info.items():
        if key[0] == "company_size" and v["n"] < C.MIN_SEGMENT_N:
            warnings.append(f"{key[1]} is underrepresented (n={v['n']} agents; minimum {C.MIN_SEGMENT_N})")
    for dim, val in spec.drop_segments:
        warnings.append(f"No {val} agents in the population: the simulation says nothing about {val}")
    flips_inside = [f for f in flips if f["within_tested_range"]]

    ev_state = L.add("decision.state", "DERIVED", state, "", depends_on=[e_win, e_p05, e_stab, e_lead, i_agents, i_ver, e_dis],
                     details={"unstable_reasons": unstable_reasons, "coverage_issues": coverage, "n_agents": n_agents, "contested": contested, "stale": stale, "confidence": conf},
                     note="presentation state chosen by the honesty policy (STALE > INSUFFICIENT > UNSTABLE > CONTESTED > stability bands)")
    L.add("decision.confidence", "DERIVED", conf, "", depends_on=[ev_state])
    L.con.commit()

    decision = _decision_view(spec, L, W, R, mean, sd, lead, win_rate, p05, stability, st_label, state, conf, unstable_reasons,
                              seg_info, dis, contested, tornado, most, flips, stress, warnings, n_agents, flips_inside, coverage,
                              {"i_agents": i_agents, "i_runs": i_runs, "i_obs": i_obs, "e_stab": e_stab, "e_lead": e_lead, "e_win": e_win,
                               "e_most": e_most, "e_dis": e_dis, "share_ids": share_ids, "state": ev_state})
    return Experiment(spec, L, decision, pop, a0)


def pop_with_drop(pop, sp2):
    p2 = dict(pop)
    p2["alive"] = E._alive_mask(sp2, pop["size"], pop["tech"], pop["industry"])
    if p2["alive"].sum() == 0:
        raise ValueError("empty population")
    return p2


# ------------------------------------------------------------------ view model for the UI / API
def _decision_view(spec, L, W, R, mean, sd, lead, win_rate, p05, stability, st_label, state, conf, reasons, seg_info, dis, contested,
                   tornado, most, flips, stress, warnings, n_agents, flips_inside, coverage, ids):
    lab = C.OPTION_LABELS
    def seg_rows(dim):
        rows = []
        for v in C.SEGMENTS[dim]:
            k = (dim, v)
            if k in seg_info:
                s = seg_info[k]
                rows.append({"segment": v, "dim": dim, "weight_pct": s["weight"] * 100, "n": s["n"], "winner": s["winner"],
                             "winner_lead_pp": s["lead"], "contribution_pp": s["contribution_pp"], "margin_pp": s["margin_pp"],
                             "evidence": s["ids"]["contribution"]})
            else:
                rows.append({"segment": v, "dim": dim, "weight_pct": 0, "n": 0, "winner": None, "winner_lead_pp": None,
                             "contribution_pp": None, "margin_pp": None, "evidence": None})
        return rows
    headline = {
        "ROBUST": f"{W} ({lab[W]}) leads {R} by {lead:.1f}pp in the simulation.",
        "MODERATE": f"{W} ({lab[W]}) leads {R} by {lead:.1f}pp in the simulation, but the lead depends on assumptions.",
        "LOW": f"{W} ({lab[W]}) leads {R} by {lead:.1f}pp in the simulation, but the lead is fragile.",
        "CONTESTED": f"{W} ({lab[W]}) leads overall by {lead:.1f}pp, but the aggregate masks strong segment disagreement.",
        "UNSTABLE": "Result unstable: insufficient evidence for a robust preference.",
        "INSUFFICIENT_EVIDENCE": (f"Insufficient evidence: only {n_agents:,} simulated agents (minimum {C.MIN_AGENTS:,})."
                                  if n_agents < C.MIN_AGENTS
                                  else (f"Insufficient evidence: {coverage[0]}." if coverage else "Insufficient evidence.")),
        "STALE": f"Out of date: this run used input {spec.run_input_version}; the current input is {spec.current_input_version}. Re-run before using.",
    }[state]
    flip_text = []
    for f in sorted(flips, key=lambda f: (not f["within_tested_range"], f["assumption"])):
        v = fmt(f["value"], f["unit"])
        side = "falls below" if f["side"] == "below" else "rises above"
        scope = "inside the tested range" if f["within_tested_range"] else "outside the tested range"
        flip_text.append({"text": f"If {f['assumption'].replace('_', ' ')} {side} {v}, {W} is no longer preferred ({scope}).",
                          "assumption": f["assumption"], "side": f["side"], "value": f["value"], "unit": f["unit"],
                          "within_tested_range": f["within_tested_range"], "evidence": f["evidence"]})
    checks = [
        {"ok": True, "text": f"{n_agents:,} simulated agents", "evidence": ids["i_agents"]},
        {"ok": spec.n_runs >= 10, "text": f"{spec.n_runs} repeated runs", "evidence": ids["i_runs"]},
        {"ok": None, "text": "Generated evidence (agent responses), not observation", "evidence": ids["share_ids"][W]},
        {"ok": False, "text": "No observed customer outcome data", "evidence": ids["i_obs"]},
    ]
    return {
        "scenario": spec.name, "title": spec.title, "question": spec.question, "note": spec.note,
        "state": state, "headline": headline, "confidence": conf, "unstable_reasons": reasons, "coverage_issues": coverage,
        "winner": W, "winner_label": lab[W], "runner_up": R,
        "shares": [{"option": o, "label": lab[o], "mean": float(mean[j]), "lo": float(mean[j] - 1.96 * sd[j]), "hi": float(mean[j] + 1.96 * sd[j]),
                    "evidence": ids["share_ids"][o]} for j, o in enumerate(C.OPTIONS)],
        "lead_pp": lead, "win_rate_pct": win_rate, "margin_p05_pp": p05,
        "stability": {"pct": stability, "label": st_label, "n_tested": spec.n_lhs, "evidence": ids["e_stab"]},
        "segments": {d: seg_rows(d) for d in ("company_size", "buyer_type", "industry")},
        "disagreement": {"flag": bool(dis), "contested": contested, "segments": dis, "evidence": ids["e_dis"]},
        "sensitivity": [{k: v for k, v in t.items()} for t in tornado], "most_sensitive": most,
        "flips": flip_text, "stress": stress, "warnings": warnings, "checks": checks,
        "assumptions": L.assumptions(), "state_evidence": ids["state"],
    }
