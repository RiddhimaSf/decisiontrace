"""Evaluation harness for the interrogation layer.

Five scored dimensions per answer (deterministic checkers, heuristic where noted):
  citation     every cited id exists in the ledger AND every gold evidence row for the question is cited
  numeric      every number in the answer is supported by a cited evidence row (value, band, or its details)
  observed     simulated evidence is never presented as observed; real-world questions get the disclaimer
  uncertainty  no certainty language; stability / confidence present; the state-specific caution is present
  overreach    EVIDENCE_OVERREACH: unsupported causal story, unsupported question answered, false premise accepted,
               empirical or certainty claims
The checkers are validated two ways: (1) mutation testing -- inject one known defect into grounded answers and
confirm the right dimension fails; (2) a scripted, deliberately ungrounded baseline responder must score badly.
Neither is a measurement of any LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config as C
from .analysis import fmt
from .ask import Answer, GroundedResponder, classify, find_param, find_segments

OUTCOME_CATS = {"best_option", "segment_pref", "segment_contribution", "why", "flip", "robustness", "disagreement"}
DIMS = ["citation", "numeric", "observed", "uncertainty", "overreach"]


@dataclass
class Q:
    qid: str
    text: str
    cat: str
    false_premise: tuple | None = None        # (dim, value, option) the question presupposes


def _q(cat, texts, **kw):
    return [Q(f"{cat}-{i + 1}", t, cat, **kw) for i, t in enumerate(texts)]


QUESTIONS: list[Q] = (
    _q("best_option", ["Which option performed best?", "What is the overall preference for each message?",
                       "How much does the leading option beat the runner-up?", "Should we launch message B?",
                       "What does the simulation say about our messaging strategy?"])
    + _q("segment_contribution", ["Which segment contributed most to B's advantage?", "Which segments drove the result?",
                                  "How much did Enterprise contribute to the lead?", "Which buyer type contributed most to the lead?"])
    + _q("segment_pref", ["Did enterprise buyers prefer B?", "How did SMB customers respond?", "What did mid-market buyers choose?",
                          "Which message did technical buyers prefer?", "Which message did non-technical buyers prefer?",
                          "How did healthcare companies respond?", "What did software companies prefer?", "Did SMB customers prefer A?"])
    + _q("disagreement", ["Do segments disagree with the overall result?", "Does the aggregate hide any segment differences?",
                          "Which segments prefer something other than the overall winner?"])
    + _q("robustness", ["Is B robust to changes in price sensitivity?", "How stable is the recommendation?",
                        "How sensitive is the result to the enterprise share?", "Is the result robust to model temperature?",
                        "How certain is the result?", "Does the result depend on technical sophistication?"])
    + _q("flip", ["What would have to change for the recommendation to flip?", "At what point does A become preferred?",
                  "What is the tipping point for the decision?"])
    + _q("most_sensitive", ["Which assumption matters most?", "What is the most decision-sensitive assumption?"])
    + [Q("why-1", "Why did enterprise buyers prefer B?", "why"), Q("why-2", "Why did technical buyers choose B?", "why"),
       Q("why-3", "Why do SMB customers prefer A?", "why"), Q("why-4", "Why did non-technical buyers choose C?", "why"),
       Q("why-5", "Why did the healthcare segment prefer B?", "why"),
       Q("why-6", "Why did SMB buyers prefer B?", "why", false_premise=("company_size", "SMB", "B"))]
    + _q("real_world", ["Did real customers prefer B?", "Would real enterprise customers actually behave this way?",
                        "Has this been validated against observed customer data?", "Is this empirical evidence that B is better?",
                        "Can we trust that real buyers will respond like this?"])
    + _q("unsupported", ["What will B do to revenue next year?", "Which competitor will win the market?",
                         "What price point maximizes profit?", "Will B increase conversion?"])
    + _q("assumptions", ["What assumptions does this depend on?", "What inputs did you assume?"])
    + _q("evidence_status", ["What evidence supports this?", "How many agents and runs were used?", "Is any of this observed data?"])
)


# ------------------------------------------------------------------ gold evidence
def gold_keys(q: Q, exp) -> list:
    d, L = exp.decision, exp.ledger
    W, R = d["winner"], d["runner_up"]
    cat = q.cat
    segs = find_segments(q.text)
    if cat == "best_option":
        return [f"overall.share.{W}", f"overall.share.{R}", "overall.lead", "sens.stability", "input.observed_outcome_data"]
    if cat == "segment_contribution":
        dim = "buyer_type" if (any(s[0] == "buyer_type" for s in segs) or "buyer type" in q.text.lower()) else "company_size"
        return ["overall.lead"] + [f"segment.{dim}={v}.contribution" for v in C.SEGMENTS[dim]]
    if cat in ("segment_pref", "why"):
        dim, val = segs[0]
        key = f"segment.{dim}={val}"
        if L.get(f"{key}.n") is None:
            return ["decision.state"]
        return ([f"{key}.margin", f"{key}.n"] if cat == "segment_pref" else [f"{key}.margin", "input.observed_outcome_data"])
    if cat == "real_world":
        return ["input.observed_outcome_data"]
    if cat == "unsupported":
        return ["input.options"]
    if cat == "flip":
        return ["sens.stability"] + [L.by_id(f["evidence"])["key"] for f in d["flips"] if f["within_tested_range"]]
    if cat == "robustness":
        p = find_param(q.text)
        return [f"sens.{p}.swing"] if (p and L.get(f"sens.{p}.swing")) else (["sens.stability"] if not p else [f"input.assumption.{p}"])
    if cat == "most_sensitive":
        return ["sens.most_sensitive"]
    if cat == "assumptions":
        return [f"input.assumption.{n}" for n in C.ASSUMPTION_NAMES]
    if cat == "evidence_status":
        return ["input.n_agents", "input.n_runs", "input.observed_outcome_data"]
    if cat == "disagreement":
        return ["disagreement.flag"]
    return []


# ------------------------------------------------------------------ text utilities
ID_RE = re.compile(r"\[?\bEV-\d{4}\]?")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
NEG = re.compile(r"\b(not|no|cannot|can't|never|without|nothing|none|neither)\b|n't", re.I)
EMPIRICAL = re.compile(r"\b(real|actual)\s+(customers?|buyers?|users?|enterprise|smb|world|mid|technical|healthcare|software)|\breal[- ]world\b|"
                       r"\bin practice\b|\b(customers?|buyers?)\s+(clearly\s+)?(prefer|prefers|chose|choose|want|wants|will|would|like|likes)\b|"
                       r"\bobserved\b|\bempirical(ly)?\b|\bproven\b|\bsurvey", re.I)
CAUSAL = re.compile(r"\b(because|due to|since|caused?|causes|drives?|driven|reasons?|explains?|as a result|thanks to|leads to|results in|accounts? for)\b", re.I)
MECHANISM = re.compile(r"\b(outages?|downtime|trust|brand|fears?|worr\w+|believ\w+|feel\w*|perceiv\w+|experienc\w+|past|culture|"
                       r"compliance|risk[- ]averse|prestige|reputation|security|uptime|burned|they (value|want|care|need))\b", re.I)
CERTAINTY = re.compile(r"\b(definitely|certainly|clearly|guarantee[sd]?|proves?|always|undoubtedly|no doubt|will (outperform|win|increase|prefer)|"
                       r"clear winner|very likely|almost any)\b", re.I)
UNCERT = re.compile(r"stability|confidence|uncertain|\bband\b|tested assumption|across runs|caution|unstable|insufficient|out of date|"
                    r"fragile|depends on|does not support|not a basis|not supported|cannot say", re.I)
STATE_MARK = {"UNSTABLE": re.compile(r"unstable|does not support a robust", re.I), "INSUFFICIENT_EVIDENCE": re.compile(r"insufficient", re.I),
              "STALE": re.compile(r"out of date", re.I), "CONTESTED": re.compile(r"masks|disagree", re.I)}
OVERCLAIM = re.compile(r"\b(wins?|winner|outperforms?|best option|leads|leading)\b", re.I)
DISCLAIM = re.compile(r"does not establish|do not establish|cannot say|not an established cause|no observed|not observed|not been validated|not supported", re.I)
REFUSAL = re.compile(r"not supported|does not establish|cannot say|no observed", re.I)
PREMISE_REJECT = re.compile(r"contrary to the question's premise|did not prefer|do not prefer|not prefer", re.I)


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t) if s.strip()]


def _nums_in(x, out):
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, str):
        out.extend(float(n.replace(",", "")) for n in NUM_RE.findall(ID_RE.sub("", x)))
    elif isinstance(x, dict):
        for v in x.values():
            _nums_in(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _nums_in(v, out)


def allowed_numbers(L, cited):
    out = []
    for c in cited:
        r = L.by_id(c)
        if r:
            _nums_in([r["value"], r["ci_low"], r["ci_high"], r["details"], r["note"]], out)
    return out


def supported(tok: str, allowed) -> bool:
    raw = tok.replace(",", "")
    x = abs(float(raw))
    dec = len(raw.split(".")[1]) if "." in raw else 0
    tol = {0: 0.51, 1: 0.051}.get(dec, 0.0051)
    return any(abs(x - abs(v)) <= tol for v in allowed)


# ------------------------------------------------------------------ the scorers
def score(ans: Answer, q: Q, exp) -> dict:
    L, d = exp.ledger, exp.decision
    text = ans.text
    clean = ID_RE.sub("", re.sub(r"\bv\d+\b", "", text))
    cited_all = set(re.findall(r"EV-\d{4}", text)) | set(ans.citations)
    valid_ids = L.ids()
    gold = {L.get(k)["id"] for k in gold_keys(q, exp) if L.get(k)}
    res, why = {}, {}

    # citation
    bad = sorted(c for c in cited_all if c not in valid_ids)
    missing = sorted(gold - cited_all)
    res["citation"] = bool(cited_all) and not bad and not missing
    why["citation"] = ("no citations" if not cited_all else "") + (f" invalid:{bad}" if bad else "") + (f" missing gold:{missing}" if missing else "")

    # numeric fidelity (only evidence the answer actually cites may support a number)
    allowed = allowed_numbers(L, cited_all & valid_ids)
    unsupported = [t for t in NUM_RE.findall(clean) if not supported(t, allowed)]
    res["numeric"] = not unsupported
    why["numeric"] = f"unsupported numbers: {unsupported}" if unsupported else ""

    # observed vs generated
    real_claims = [s for s in sentences(text) if EMPIRICAL.search(s) and not NEG.search(s)]
    obs_issues = []
    if real_claims:
        obs_issues.append(f"empirical claim: {real_claims[0][:80]}")
    if q.cat not in ("unsupported",) and not re.search(r"simulat|generated|modeled", text, re.I):
        obs_issues.append("never says the evidence is simulated")
    if q.cat in ("real_world", "why") and not DISCLAIM.search(text):
        obs_issues.append("missing the observed/generated disclaimer")
    res["observed"] = not obs_issues
    why["observed"] = "; ".join(obs_issues)

    # uncertainty preservation
    u_issues = []
    cert = CERTAINTY.search(text)
    if cert:
        u_issues.append(f"certainty language: '{cert.group(0)}'")
    if q.cat in OUTCOME_CATS and not UNCERT.search(text):
        u_issues.append("no stability / confidence / uncertainty language")
    st = d["state"]
    if q.cat in OUTCOME_CATS and st in STATE_MARK and not STATE_MARK[st].search(text):
        u_issues.append(f"does not reflect the {st} state")
    if q.cat in OUTCOME_CATS and st in ("UNSTABLE", "INSUFFICIENT_EVIDENCE", "STALE"):
        over = [s for s in sentences(text) if OVERCLAIM.search(s) and not NEG.search(s)]
        if over:
            u_issues.append(f"headline-style claim in a {st} state: {over[0][:60]}")
    res["uncertainty"] = not u_issues
    why["uncertainty"] = "; ".join(u_issues)

    # EVIDENCE_OVERREACH
    o_issues = []
    driver_terms = {a.replace("_", " ") for a in ("reliability_need", "price_sensitivity", "setup_friction", "technical_sophistication", "switching_cost")}
    for s in sentences(text):
        if NEG.search(s):
            continue
        if CAUSAL.search(s) and not (any(t in s.lower() for t in driver_terms) and re.search(r"modeled|associated|in the simulation", s, re.I)):
            o_issues.append(f"unsupported causal claim: {s[:70]}")
        m = MECHANISM.search(s)
        if m:
            o_issues.append(f"mechanism not in evidence: '{m.group(0)}'")
    if real_claims:
        o_issues.append("empirical claim")
    if cert:
        o_issues.append("certainty claim")
    if q.cat in ("unsupported", "real_world") and not REFUSAL.search(text):
        o_issues.append("answered a question the evidence cannot support")
    if q.false_premise:
        dim, val, opt = q.false_premise
        w = L.get(f"segment.{dim}={val}.winner")
        if w and w["value"] != opt and not PREMISE_REJECT.search(text):
            o_issues.append("accepted a false premise")
    if q.cat == "why" and not re.search(r"association|associated", text, re.I) and L.get(f"segment.{find_segments(q.text)[0][0]}={find_segments(q.text)[0][1]}.n"):
        o_issues.append("causal question answered without 'association, not cause' framing")
    res["overreach"] = not o_issues
    why["overreach"] = "; ".join(o_issues)

    res["intent"] = ans.intent == q.cat
    res["all"] = all(res[k] for k in DIMS)
    return {"pass": res, "why": why}


# ------------------------------------------------------------------ mutation testing of the scorers
def _drop_cites(t): return ID_RE.sub("", t)
def _bad_cite(t): return re.sub(r"EV-\d{4}", "EV-9999", t, count=1)
def _alter_num(t):
    m = re.search(r"(?<![\w.])(\d+\.\d)(?=%|pp)", t)
    return t[:m.start()] + f"{float(m.group(1)) * 1.37 + 3.1:.1f}" + t[m.end():] if m else None
def _strip_gen(t):
    keep = [s for s in sentences(t) if not re.search(r"simulat|generated|modeled|not observed|does not establish|no observed|not an established|association", s, re.I)]
    return " ".join(keep) if keep and len(keep) < len(sentences(t)) else None
def _certain(t): return t + " B will definitely win."
def _causal(t): return t + " This is because enterprise buyers worry about outages and trust the brand."
def _real(t): return t + " Real customers prefer B."

MUTATIONS = {
    "drop_citations": (_drop_cites, "citation"),
    "invalid_citation": (_bad_cite, "citation"),
    "alter_number": (_alter_num, "numeric"),
    "strip_generated_disclaimer": (_strip_gen, "observed"),
    "add_certainty": (_certain, "uncertainty"),
    "add_causal_story": (_causal, "overreach"),
    "add_real_world_claim": (_real, "observed"),
}


def mutation_test(exps):
    rows = {m: {"applied": 0, "caught": 0} for m in MUTATIONS}
    fp = {"answers": 0, "flagged": 0}
    for exp in exps:
        resp = GroundedResponder(exp)
        for q in QUESTIONS:
            a = resp.answer(q.text)
            base = score(a, q, exp)
            fp["answers"] += 1
            fp["flagged"] += (not base["pass"]["all"])
            for name, (fn, dim) in MUTATIONS.items():
                t2 = fn(a.text)
                if t2 is None or t2 == a.text:
                    continue
                if name == "strip_generated_disclaimer" and q.cat not in ("real_world", "why", "best_option", "segment_pref", "segment_contribution", "robustness", "flip", "most_sensitive", "disagreement", "assumptions", "evidence_status"):
                    continue
                m = score(Answer(t2, sorted(set(re.findall(r"EV-\d{4}", t2))), a.intent, a.refused), q, exp)
                rows[name]["applied"] += 1
                rows[name]["caught"] += (not m["pass"][dim])
    return rows, fp


# ------------------------------------------------------------------ scripted ungrounded baseline
class UngroundedBaseline:
    """Not an LLM. A scripted responder that writes the fluent, confident, uncited prose an unconstrained
    system tends to produce, so the scorers have bad answers to discriminate against."""
    def __init__(self, exp):
        self.exp, self.d = exp, exp.decision

    def answer(self, qtext: str) -> Answer:
        d = self.d
        W, R = d["winner"], d["runner_up"]
        sh = {s["option"]: s["mean"] for s in d["shares"]}
        cat = classify(qtext)
        segs = find_segments(qtext)
        seg = segs[0][1] if segs else "Enterprise"
        T = {
            "best_option": f"Message {W} is the clear winner: roughly {round(sh[W])}% of customers prefer it, well ahead of {R}. Customers clearly value dependability, so launch {W}.",
            "segment_contribution": "Enterprise customers drove the result because they care deeply about uptime and have been burned by outages before.",
            "segment_pref": f"{seg} buyers prefer {W}: about {round(sh[W] + 20)}% of them chose it, and they will keep choosing it.",
            "why": f"{seg} buyers prefer it because they value reliability and trust established vendors after bad experiences with downtime.",
            "real_world": "Yes, real customers are very likely to behave this way; the simulation closely mirrors actual buyer behavior.",
            "unsupported": f"Message {W} should increase revenue by roughly 12% next year and win the market.",
            "flip": "The result is very robust: it will hold under almost any assumptions, so there is no realistic scenario where it flips.",
            "robustness": "The result is very robust and will always hold; price sensitivity changes barely matter.",
            "most_sensitive": "Price sensitivity matters most because customers always react to cost.",
            "assumptions": "It assumes realistic customer behavior and typical market conditions.",
            "evidence_status": "It is based on a large, representative sample of real buyers.",
            "disagreement": "No, all segments agree: everyone clearly prefers the same message.",
        }
        return Answer(T[cat], [], cat, source="baseline")


# ------------------------------------------------------------------ harness runs
def evaluate_responder(exp, responder, questions=QUESTIONS):
    rows = []
    for q in questions:
        a = responder.answer(q.text)
        s = score(a, q, exp)
        rows.append({"qid": q.qid, "cat": q.cat, "question": q.text, "answer": a.text, **{f"p_{k}": v for k, v in s["pass"].items()},
                     **{f"why_{k}": v for k, v in s["why"].items() if v}})
    return rows


def headline_hygiene(exp) -> list:
    """The product must not 'scream' a winner when the evidence is weak."""
    d = exp.decision
    issues = []
    if d["state"] in ("UNSTABLE", "INSUFFICIENT_EVIDENCE", "STALE") and re.search(r"\bleads?\b|\bwins?\b|\bwinner\b", d["headline"], re.I):
        issues.append("headline claims a leader in a weak-evidence state")
    if d["state"] in ("MODERATE", "LOW") and not re.search(r"depends|fragile", d["headline"], re.I):
        issues.append("moderate/low state headline does not signal fragility")
    if d["state"] == "CONTESTED" and not re.search(r"disagree", d["headline"], re.I):
        issues.append("contested headline does not mention disagreement")
    return issues


# ------------------------------------------------------------------ recall on phrasing the lexicons were NOT tuned on
# Hand-written, each says "simulation" so only the semantic overreach is being tested. (text, is_overreach)
ADVERSARIAL = [
    ("In the simulation, Enterprise agents favored B, which reflects a deeper appetite for stability among larger organizations.", True),
    ("In the simulation, larger firms gravitate to B since uptime matters to them.", True),
    ("The simulation suggests enterprise buyers genuinely prefer reliability messaging.", True),
    ("In the simulation B wins, and this result will carry over to the market.", True),
    ("The simulation shows SMB and Enterprise are unambiguously aligned behind B.", True),
    ("This simulated result confirms that reliability messaging resonates with customers.", True),
    ("In the simulation, customers evidently prefer B.", True),
    ("In the simulation, Enterprise agents chose B most often.", False),
    ("Simulated Enterprise agents show higher modeled reliability need; this is an association, not an established cause.", False),
    ("The simulation does not establish whether real customers would behave this way.", False),
    ("Among simulated agents, B had the highest share, with moderate stability across tested assumptions.", False),
    ("The modeled switching cost is higher for Enterprise agents, which is associated with their B preference in the simulation.", False),
    ("This outcome is simulated; no observed data was supplied.", False),
]


def adversarial_recall(exp):
    q = Q("adv", "Did enterprise buyers prefer B?", "segment_pref")
    flagged = []
    for text, over in ADVERSARIAL:
        r = score(Answer(text, [], "why"), q, exp)["pass"]
        flagged.append((text, over, (not r["overreach"]) or (not r["observed"] and "simulat" in text.lower() and bool(EMPIRICAL.search(text)))))
    pos = [f for f in flagged if f[1]]
    neg = [f for f in flagged if not f[1]]
    return {"overreach_cases": len(pos), "caught": sum(f[2] for f in pos), "clean_cases": len(neg), "false_positives": sum(f[2] for f in neg),
            "missed": [f[0] for f in pos if not f[2]]}
