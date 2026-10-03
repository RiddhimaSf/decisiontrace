"""Follow-up interrogation of a completed simulation.

The responder sees ONLY the evidence ledger (structured rows). It never recomputes anything and it never
invents a 'why'. Every number in an answer is formatted from a cited ledger row; every causal statement is
phrased as an association between modeled attributes and simulated choices; questions about real-world
behavior are refused.

`GroundedResponder` is deterministic. `llm_answer` lets an LLM *phrase* an answer from the same evidence,
but the reply is rejected (and the grounded answer used instead) unless it passes the same checks the
evaluation harness applies.  No LLM computes a result anywhere in this project.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from . import config as C
from .analysis import fmt

SEG_PATTERNS = [
    ("buyer_type", "Nontechnical", r"non[- ]?technical"),
    ("buyer_type", "Technical", r"\btechnical\b|\btechnically\b"),
    ("company_size", "SMB", r"\bsmbs?\b|small (business|compan)"),
    ("company_size", "Mid-market", r"mid[- ]?market|mid[- ]?size|\bmid\b"),
    ("company_size", "Enterprise", r"enterprise|large (compan|business)"),
    ("industry", "Healthcare", r"healthcare|health care"),
    ("industry", "Software", r"\bsoftware\b"),
    ("industry", "Finance", r"\bfinance|financial"),
    ("industry", "Retail", r"\bretail"),
    ("industry", "Manufacturing", r"manufactur"),
]
PARAM_PATTERNS = [
    ("price_sensitivity_scale", r"price"),
    ("reliability_sensitivity", r"reliab"),
    ("setup_sensitivity", r"setup|set-up|onboarding"),
    ("enterprise_share", r"enterprise (share|representation|weight|mix)|representation|population mix|share of enterprise"),
    ("technical_shift", r"sophistication|technical (distribution|mix|skill)"),
    ("temperature", r"temperature|stochastic|randomness|noise"),
]
DRIVERS_BY_OPTION = {"A": ["price_sensitivity"], "B": ["reliability_need", "switching_cost"], "C": ["setup_friction", "technical_sophistication"]}


@dataclass
class Answer:
    text: str
    citations: list = field(default_factory=list)
    intent: str = ""
    refused: bool = False
    source: str = "grounded"


def find_segments(q: str):
    ql = q.lower()
    stripped = re.sub(r"non[- ]?technical", " ", ql)          # so 'non-technical' is not also read as 'technical'
    out = []
    for dim, val, pat in SEG_PATTERNS:
        if re.search(pat, stripped if val == "Technical" else ql):
            out.append((dim, val))
    return out


def find_param(q: str):
    ql = q.lower()
    for name, pat in PARAM_PATTERNS:
        if re.search(pat, ql):
            return name
    return None


def find_option(q: str):
    m = re.search(r"\b(?:message|option)\s+([ABC])\b", q) or re.search(r"\b([ABC])(?:'s)?\s+(?:advantage|lead|win|is|become)", q) \
        or re.search(r"(?:prefer|choose|chose|pick|favor)\s+([ABC])\b", q)
    if m:
        return m.group(1)
    for o, pat in (("A", r"lower price|cheaper"), ("B", r"reliab"), ("C", r"easier setup|ease of setup")):
        if re.search(pat, q.lower()):
            return o
    return None


def classify(q: str) -> str:
    ql = q.lower()
    if re.search(r"revenue|profit|competitor|next year|forecast|churn|\broi\b|conversion|market share|price point|maximi[sz]e|will .* (increase|grow|win the market)", ql):
        return "unsupported"
    if re.search(r"any of this (is |was )?observed|is (any|all) of this (real|observed)|how many (agents|runs)|what evidence|provenance|sample size", ql):
        return "evidence_status"
    if re.search(r"\b(real|actual|actually|observed|validated?|validation|empirical|in practice|real[- ]world|trust that|prove|proof)\b", ql):
        return "real_world"
    if re.search(r"^\s*why\b|\bwhy (did|do|does|is|are|would)\b|what (drove|explains|caused|is behind)|\breason", ql):
        return "why"
    if re.search(r"flip|what would (have to )?change|tipping|threshold|break[- ]?even|at what point|become preferred|stop being", ql):
        return "flip"
    if re.search(r"most (sensitive|decision[- ]sensitive|important)|which assumption|matters most", ql):
        return "most_sensitive"
    if re.search(r"which assumptions|what assumptions|assumptions? (does|do)|inputs? did you assume|what .* assum", ql):
        return "assumptions"
    if re.search(r"how many (agents|runs)|what evidence|any of this observed|provenance|where does .* come from|sample size|evidence (is|supports)", ql):
        return "evidence_status"
    if re.search(r"disagree|hide|masks?|which segments? prefer|differ(ent)? from the overall|something other than", ql):
        return "disagreement"
    if re.search(r"robust|stable|stability|sensitiv|certain|confident|fragile|depend on|what happens if|survive", ql):
        return "robustness"
    if re.search(r"contribut|drove the|which segments? (drove|contributed)|which (buyer type|segment) contributed", ql):
        return "segment_contribution"
    if find_segments(q):
        return "segment_pref"
    return "best_option"


class GroundedResponder:
    def __init__(self, exp):
        self.exp, self.L, self.d = exp, exp.ledger, exp.decision

    # ---- helpers --------------------------------------------------------------------------------
    def ev(self, key):
        return self.L.get(key)

    def c(self, row):
        return f"[{row['id']}]"

    def v(self, row, unit=None):
        return fmt(row["value"], unit or row["unit"])

    def caution(self):
        st, d = self.d["state"], self.d
        row = self.ev("decision.state")
        if st == "STALE":
            return f"Caution: this run is out of date (it used input {self.exp.spec.run_input_version}; the current input is {self.exp.spec.current_input_version}), so treat every figure as provisional {self.c(row)}."
        if st == "INSUFFICIENT_EVIDENCE":
            why = d["coverage_issues"][0] if d["coverage_issues"] else f"only {d['checks'][0]['text']}"
            return f"Caution: the evidence is insufficient ({why}), so these figures are not a basis for a decision {self.c(row)}."
        if st == "UNSTABLE":
            return f"Caution: the result is unstable and does not support a robust preference ({'; '.join(d['unstable_reasons'])}) {self.c(row)}."
        if st == "CONTESTED":
            return f"Caution: the aggregate masks strong segment disagreement, so the overall figure may not describe any single group {self.c(row)}."
        return ""

    def generated_note(self):
        return f"This is generated simulation evidence, not observed customer behavior {self.c(self.ev('input.observed_outcome_data'))}."

    def stability_sentence(self):
        s = self.ev("sens.stability")
        W = self.d["winner"]
        return (f"Decision stability: {s['details']['label']} ({W} remains preferred under {self.v(s)} of "
                f"{s['details']['n_tested']} tested assumption sets) {self.c(s)}.")

    def done(self, intent, parts, refused=False):
        text = " ".join(p for p in parts if p)
        cites = sorted(set(re.findall(r"\[(EV-\d{4})\]", text)))
        return Answer(text, cites, intent, refused)

    # ---- intents --------------------------------------------------------------------------------
    def answer(self, q: str) -> Answer:
        intent = classify(q)
        return getattr(self, f"_{intent}")(q)

    def _unsupported(self, q):
        opts = self.ev("input.options")
        return self.done("unsupported", [
            f"That is not supported by the evidence in this ledger {self.c(opts)}.",
            "The simulation only measures simulated message preference; it contains no data on revenue, conversion, competitors, forecasts or pricing outcomes.",
            f"I can answer questions about preference shares, segment contributions, robustness, flip conditions and assumptions. {self.generated_note()}"], refused=True)

    def _real_world(self, q):
        obs = self.ev("input.observed_outcome_data")
        segs = find_segments(q)
        parts = ["The simulation does not establish that.",
                 f"Every preference figure here is generated by simulated agents, and no observed customer outcome data was supplied {self.c(obs)}."]
        if segs and self.ev(f"segment.{segs[0][0]}={segs[0][1]}.margin"):
            m = self.ev(f"segment.{segs[0][0]}={segs[0][1]}.margin")
            parts.append(f"What the simulation does show is a modeled {self.v(m)} {m['details']['winner']}-{m['details']['runner_up']} margin among simulated {segs[0][1]} agents {self.c(m)}; it cannot say whether real {segs[0][1]} customers would behave this way.")
        else:
            parts.append("Validating these results would require observed customer outcome data, which this experiment does not contain.")
        return self.done("real_world", parts, refused=True)

    def _best_option(self, q):
        d, L = self.d, self.L
        W, R = d["winner"], d["runner_up"]
        sw, sr = self.ev(f"overall.share.{W}"), self.ev(f"overall.share.{R}")
        lead, obs = self.ev("overall.lead"), self.ev("input.observed_outcome_data")
        shares = {o: self.ev(f"overall.share.{o}") for o in C.OPTIONS}
        if d["state"] in ("UNSTABLE", "INSUFFICIENT_EVIDENCE", "STALE"):
            listing = ", ".join(f"{o} {self.v(shares[o])} {self.c(shares[o])}" for o in C.OPTIONS)
            return self.done("best_option", [
                "The simulation does not support a robust preference between the messages.", self.caution(),
                f"For reference only, simulated shares were {listing}; the gap between the top two is {self.v(lead)} {self.c(lead)}.",
                self.stability_sentence(), self.generated_note()])
        parts = [f"In the simulation, {W} ({C.OPTION_LABELS[W]}) has the highest simulated preference at {self.v(sw)} "
                 f"(band {fmt(sw['ci_low'], 'pct')} to {fmt(sw['ci_high'], 'pct')} across runs) {self.c(sw)}, "
                 f"versus {self.v(sr)} for {R} {self.c(sr)}. The gap is {self.v(lead)} {self.c(lead)}.",
                 self.stability_sentence(), self.caution(), self.generated_note()]
        return self.done("best_option", parts)

    def _segment_contribution(self, q):
        d = self.d
        dim = "buyer_type" if any(s[0] == "buyer_type" for s in find_segments(q)) or re.search(r"buyer type", q.lower()) else "company_size"
        W, R = d["winner"], d["runner_up"]
        lead = self.ev("overall.lead")
        rows = [(v, self.ev(f"segment.{dim}={v}.contribution")) for v in C.SEGMENTS[dim]]
        rows = [(v, r) for v, r in rows if r]
        rows.sort(key=lambda x: -x[1]["value"])
        listing = ", ".join(f"{v} {self.v(r)} {self.c(r)}" for v, r in rows)
        top = rows[0]
        parts = [f"Contributions to the overall {W}-{R} lead of {self.v(lead)} {self.c(lead)}, by {dim.replace('_', ' ')} (weight x segment margin): {listing}.",
                 f"{top[0]} contributes most; a negative value means the segment pulls against the overall result.",
                 "Contributions sum exactly to the overall lead within a dimension.",
                 self.stability_sentence(), self.caution(), self.generated_note()]
        return self.done("segment_contribution", parts)

    def _segment_pref(self, q):
        d = self.d
        dim, val = find_segments(q)[0]
        key = f"segment.{dim}={val}"
        n = self.ev(f"{key}.n")
        if n is None:
            mix = self.ev("input.population_mix")
            return self.done("segment_pref", [
                f"The simulation has no {val} agents, so it says nothing about how {val} customers would respond {self.c(self.ev('decision.state'))}.",
                f"Population composition is recorded in the inputs {self.c(mix)}.", self.stability_sentence(), self.caution(), self.generated_note()])
        W, R = d["winner"], d["runner_up"]
        shares = {o: self.ev(f"{key}.share.{o}") for o in C.OPTIONS}
        win = self.ev(f"{key}.winner")["value"]
        sw = shares[win]
        others = ", ".join(f"{o} {self.v(shares[o])} {self.c(shares[o])}" for o in C.OPTIONS if o != win)
        m, wt = self.ev(f"{key}.margin"), self.ev(f"{key}.weight")
        parts = [f"In the simulation, {val} agents (n={self.v(n)} {self.c(n)}, {self.v(wt)} of the weighted population {self.c(wt)}) most often chose {win} at {self.v(sw)} "
                 f"(band {fmt(sw['ci_low'], 'pct')} to {fmt(sw['ci_high'], 'pct')} across runs) {self.c(sw)}; others: {others}.",
                 f"Their {W}-{R} margin is {self.v(m)} {self.c(m)} compared with {self.v(self.ev('overall.lead'))} overall {self.c(self.ev('overall.lead'))}."]
        if n["value"] < C.MIN_SEGMENT_N:
            parts.append(f"This segment is underrepresented ({self.v(n)} agents), so read it with extra caution.")
        parts += [self.stability_sentence(), self.caution(), self.generated_note()]
        return self.done("segment_pref", parts)

    def _why(self, q):
        d, L = self.d, self.L
        segs = find_segments(q)
        if not segs:
            return self._unsupported(q)
        dim, val = segs[0]
        key = f"segment.{dim}={val}"
        n = self.ev(f"{key}.n")
        obs = self.ev("input.observed_outcome_data")
        if n is None:
            return self.done("why", [f"The simulation has no {val} agents, so it cannot say why {val} customers would respond in any way {self.c(self.ev('decision.state'))}.", self.caution(), self.generated_note()], refused=True)
        asked = find_option(q)
        win_row = self.ev(f"{key}.winner")
        win = win_row["value"]
        W, R = d["winner"], d["runner_up"]
        parts = []
        target = asked or win
        if asked and asked != win:
            sh = {o: self.ev(f"{key}.share.{o}") for o in C.OPTIONS}
            parts.append(f"Contrary to the question's premise, {val} agents did not prefer {asked} in the simulation; their most frequent choice was {win} at {self.v(sh[win])} {self.c(sh[win])}, "
                         f"against {self.v(sh[asked])} for {asked} {self.c(sh[asked])}.")
        m = self.ev(f"{key}.margin")
        lead = self.ev("overall.lead")
        parts.append(f"In the simulation, {val} agents show a {self.v(m)} {W}-{R} margin {self.c(m)} versus {self.v(lead)} overall {self.c(lead)}.")
        drv = []
        for attr in DRIVERS_BY_OPTION[target]:
            r = self.ev(f"{key}.driver.{attr}")
            if r and r["value"] is not None:
                drv.append(f"{attr.replace('_', ' ')} averages {fmt(r['value'], 'ratio')} for {val} agents versus {fmt(r['details']['overall'], 'ratio')} overall {self.c(r)}")
        if drv:
            parts.append(f"Modeled attributes associated with a preference for {target} in this segment: " + "; ".join(drv) + ".")
        parts.append("This is an association between modeled agent attributes and simulated choices, not an established cause, "
                     f"and the simulation does not establish why real buyers would choose {target} {self.c(obs)}.")
        parts += [self.stability_sentence(), self.caution()]
        return self.done("why", parts)

    def _robustness(self, q):
        d = self.d
        p = find_param(q)
        W = d["winner"]
        if p is None:
            s, w, p05, nr = self.ev("sens.stability"), self.ev("overall.win_rate"), self.ev("overall.margin_p05"), self.ev("input.n_runs")
            parts = [self.stability_sentence(),
                     f"{W} had the highest simulated preference in {self.v(w)} of the {self.v(nr)} repeated runs {self.c(w)} {self.c(nr)}; the worst-case lead across runs was {self.v(p05)} {self.c(p05)}.",
                     self.caution(), self.generated_note()]
            return self.done("robustness", parts)
        sw = self.ev(f"sens.{p}.swing")
        if sw is None:
            ar = self.ev(f"input.assumption.{p}")
            return self.done("robustness", [f"{p.replace('_', ' ')} was not varied in this experiment, so the simulation says nothing about its effect on the result {self.c(ar)}.",
                                            self.stability_sentence(), self.caution(), self.generated_note()])
        det = sw["details"]
        parts = [f"Varying {p.replace('_', ' ')} across its tested range ({det['tested'][0]:.2f} to {det['tested'][1]:.2f}) moves {W}'s lead between "
                 f"{det['margin_at_low_pp']:+.1f}pp and {det['margin_at_high_pp']:+.1f}pp, a swing of {self.v(sw)} {self.c(sw)}."]
        fl = [self.ev(f"flip.{p}.{s}") for s in ("below", "above")]
        fl = [f for f in fl if f]
        if fl:
            for f in fl:
                inside = "inside" if f["details"]["within_tested_range"] else "outside"
                side = "falls below" if f["details"]["side"] == "below" else "rises above"
                parts.append(f"{W} stops being preferred if {p.replace('_', ' ')} {side} {self.v(f)} ({inside} the tested range) {self.c(f)}.")
        else:
            parts.append(f"No flip was found for {p.replace('_', ' ')} anywhere in its explored range.")
        parts += [self.stability_sentence(), self.caution(), self.generated_note()]
        return self.done("robustness", parts)

    def _flip(self, q):
        d = self.d
        W = d["winner"]
        parts = []
        inside = [f for f in d["flips"] if f["within_tested_range"]]
        outside = [f for f in d["flips"] if not f["within_tested_range"]]
        if inside:
            for f in inside:
                row = self.L.by_id(f["evidence"])
                side = "falls below" if f["side"] == "below" else "rises above"
                parts.append(f"If {f['assumption'].replace('_', ' ')} {side} {self.v(row)} (inside the tested range), {W} is no longer preferred {self.c(row)}.")
        else:
            parts.append("No single assumption flips the result inside its tested range, holding the others at nominal.")
        for f in outside[:3]:
            row = self.L.by_id(f["evidence"])
            side = "falls below" if f["side"] == "below" else "rises above"
            parts.append(f"Outside the tested range: if {f['assumption'].replace('_', ' ')} {side} {self.v(row)}, {W} is no longer preferred {self.c(row)}.")
        for s in d["stress"]:
            if not s["same_winner"]:
                row = self.ev(f"stress.drop.{s['dropped']}")
                parts.append(f"If {s['dropped'].split('=')[1]} agents were missing, {row['value']} would be preferred instead {self.c(row)}.")
        parts += [self.stability_sentence(), self.caution(), self.generated_note()]
        return self.done("flip", parts)

    def _most_sensitive(self, q):
        t = self.d["sensitivity"][:2]
        W = self.d["winner"]
        most = self.ev("sens.most_sensitive")
        r1 = self.L.by_id(t[0]["evidence"])
        parts = [f"The most decision-sensitive assumption is {t[0]['assumption'].replace('_', ' ')} {self.c(most)}: across its tested range it moves {W}'s lead by {self.v(r1)}, "
                 f"{r1['details']['index']:.2f} times the lead itself {self.c(r1)}."]
        if len(t) > 1:
            r2 = self.L.by_id(t[1]["evidence"])
            parts.append(f"Next is {t[1]['assumption'].replace('_', ' ')} with a swing of {self.v(r2)} {self.c(r2)}.")
        parts += [self.stability_sentence(), self.caution(), self.generated_note()]
        return self.done("most_sensitive", parts)

    def _assumptions(self, q):
        parts = ["The result depends on these modeled assumptions, none of which was validated against observed data:"]
        items = []
        for name in C.ASSUMPTION_NAMES:
            r = self.ev(f"input.assumption.{name}")
            items.append(f"{name.replace('_', ' ')} = {fmt(r['value'], r['unit'])} (tested {fmt(r['ci_low'], r['unit'])} to {fmt(r['ci_high'], r['unit'])}) {self.c(r)}")
        parts.append("; ".join(items) + ".")
        parts.append(self.generated_note())
        return self.done("assumptions", parts)

    def _evidence_status(self, q):
        na, nr, obs = self.ev("input.n_agents"), self.ev("input.n_runs"), self.ev("input.observed_outcome_data")
        parts = [f"The simulation used {self.v(na)} simulated agents {self.c(na)} and {self.v(nr)} repeated runs {self.c(nr)}.",
                 "Evidence is classed as INPUT (configuration), GENERATED (simulated agent responses), or DERIVED (deterministic computation over those).",
                 f"No OBSERVED customer outcome data exists in this experiment {self.c(obs)}."]
        if self.d["warnings"]:
            parts.append("Coverage warnings: " + "; ".join(self.d["warnings"]) + ".")
        parts.append(self.caution())
        return self.done("evidence_status", parts)

    def _disagreement(self, q):
        d = self.d
        e = self.ev("disagreement.flag")
        W = d["winner"]
        if not d["disagreement"]["segments"]:
            return self.done("disagreement", [f"No sizeable segment prefers a different option than {W} by a meaningful margin {self.c(e)}.",
                                                self.stability_sentence(), self.caution(), self.generated_note()])
        listing = []
        for s in d["disagreement"]["segments"]:
            if s["dim"] == "industry":
                continue
            r_win = self.ev(f"segment.{s['dim']}={s['value']}.share.{s['winner']}")
            r_ovr = self.ev(f"segment.{s['dim']}={s['value']}.share.{W}")
            listing.append(f"{s['value']} prefers {s['winner']} at {self.v(r_win)} versus {W} at {self.v(r_ovr)} {self.c(r_win)} {self.c(r_ovr)}")
        parts = [f"The aggregate result masks significant population disagreement {self.c(e)}: " + ("; ".join(listing) if listing else "see segment table") + ".",
                 self.stability_sentence(), self.caution(), self.generated_note()]
        return self.done("disagreement", parts)


# ------------------------------------------------------------------ optional LLM phrasing (never computes)
SYSTEM_PROMPT = """You answer questions about a completed AI simulation using ONLY the evidence rows provided.
Rules:
1. Every number must be copied exactly from an evidence row and followed by its id in square brackets, e.g. [EV-0012].
2. Say 'simulated' or 'in the simulation'. Never present these results as what real customers do.
3. Never give a reason ('because', 'due to', 'drives') beyond the modeled attributes in the rows. Describe them as associated, not as causes.
4. Preserve uncertainty: include the decision stability or confidence band whenever you state a result.
5. If the evidence does not support an answer, say 'That is not supported by the evidence' and stop."""


def llm_answer(exp, question: str, grounded: Answer, scorer, client=None, model: str | None = None) -> Answer:
    """Let an LLM phrase the answer from the same evidence rows. Rejected unless it passes the harness checks."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model = model or os.environ.get("DECISIONTRACE_MODEL", "claude-sonnet-5-5")
    rows = [exp.ledger.by_id(i) for i in grounded.citations]
    packet = json.dumps([{k: r[k] for k in ("id", "key", "cls", "value", "unit", "ci_low", "ci_high", "note", "details")} for r in rows], default=str)
    msg = client.messages.create(model=model, max_tokens=500, system=SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": f"Question: {question}\n\nEvidence rows:\n{packet}"}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    cand = Answer(text, sorted(set(re.findall(r"\[(EV-\d{4})\]", text))), grounded.intent, source="llm")
    return cand if scorer(cand) else Answer(grounded.text, grounded.citations, grounded.intent, grounded.refused, "fallback:rejected-by-checks")
