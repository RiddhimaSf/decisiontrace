import json
import re

import numpy as np
import pytest

from decisiontrace import config as C
from decisiontrace import engine as E
from decisiontrace import evalharness as H
from decisiontrace import scenarios
from decisiontrace.analysis import build_experiment, margin_vs_best_other
from decisiontrace.ask import Answer, GroundedResponder, classify, find_segments, llm_answer


# ------------------------------------------------------------------ ledger & provenance
def test_segment_contributions_decompose_the_lead_exactly(exps):
    for ex in exps.values():
        d = ex.decision
        for dim in ("company_size", "buyer_type"):
            cs = [r["contribution_pp"] for r in d["segments"][dim] if r["contribution_pp"] is not None]
            if len(cs) == len(C.SEGMENTS[dim]):
                assert sum(cs) == pytest.approx(d["lead_pp"], abs=1e-6)


def test_every_derived_row_traces_back_to_generated_and_input_and_never_to_observed(exps):
    L = exps["base"].ledger
    for key in ("overall.share.B", "overall.lead", "sens.stability", "segment.company_size=Enterprise.contribution"):
        row = L.get(key)
        classes = L.classes_in_chain(row["id"])
        assert {"DERIVED", "INPUT"} <= classes and "OBSERVED" not in classes
    assert "GENERATED" in L.classes_in_chain(L.get("overall.share.B")["id"])


def test_the_ledger_records_the_absence_of_observed_data(exps):
    row = exps["base"].ledger.get("input.observed_outcome_data")
    assert row["cls"] == "OBSERVED" and row["value"] is False


def test_every_depends_on_id_exists(exps):
    L = exps["base"].ledger
    ids = L.ids()
    for r in L.all():
        assert set(r["depends_on"]) <= ids


def test_pipeline_is_deterministic():
    a = build_experiment(scenarios.get("base")).decision
    b = build_experiment(scenarios.get("base")).decision
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ------------------------------------------------------------------ flip analysis
def test_flip_thresholds_are_real_sign_changes(exps):
    ex = exps["base"]
    wi = C.OPTIONS.index(ex.decision["winner"])
    for f in ex.decision["flips"]:
        k, v = f["assumption"], f["value"] / (100 if f["assumption"] == "enterprise_share" else 1)
        eps = 0.02 * abs(v) + 1e-3
        inside = v + eps if f["side"] == "below" else v - eps          # still on the nominal side
        outside = v - eps if f["side"] == "below" else v + eps
        assert margin_vs_best_other(E.expected_shares(ex.pop, ex.spec, {**ex.a0, k: inside}), wi) > 0
        assert margin_vs_best_other(E.expected_shares(ex.pop, ex.spec, {**ex.a0, k: outside}), wi) <= 0


def test_in_range_flag_matches_the_tested_range(exps):
    ex = exps["base"]
    for f in ex.decision["flips"]:
        row = ex.ledger.by_id(f["evidence"])
        lo, hi = row["details"]["tested"]
        v = f["value"] / (100 if f["assumption"] == "enterprise_share" else 1)
        assert f["within_tested_range"] == (lo <= v <= hi)


def test_stability_counts_samples_where_the_winner_survives(exps):
    ex = exps["base"]
    assert 0 <= ex.decision["stability"]["pct"] <= 100 and ex.decision["stability"]["n_tested"] == C.N_LHS


# ------------------------------------------------------------------ the product changes what it says
@pytest.mark.parametrize("name", list(scenarios.SCENARIO_EXPECTED))
def test_scenario_reaches_its_expected_state(exps, name):
    assert exps[name].decision["state"] in scenarios.SCENARIO_EXPECTED[name]
    assert H.headline_hygiene(exps[name]) == []


def test_weak_evidence_states_never_claim_a_leader(exps):
    for ex in exps.values():
        d = ex.decision
        if d["state"] in ("UNSTABLE", "INSUFFICIENT_EVIDENCE", "STALE"):
            assert not re.search(r"\bleads?\b|\bwins?\b", d["headline"]) and d["confidence"] == "Insufficient"


def test_missing_segment_is_reported_as_missing_not_estimated(exps):
    d = exps["missing_segment"].decision
    assert any("Enterprise" in w for w in d["warnings"])
    assert all(r["n"] == 0 for r in d["segments"]["company_size"] if r["segment"] == "Enterprise")


# ------------------------------------------------------------------ interrogation
def test_intents():
    assert classify("Did real customers prefer B?") == "real_world"
    assert classify("What will B do to revenue next year?") == "unsupported"
    assert classify("Why did enterprise buyers prefer B?") == "why"
    assert classify("Is any of this observed data?") == "evidence_status"
    assert find_segments("non-technical buyers") == [("buyer_type", "Nontechnical")]


def test_real_world_and_unsupported_questions_are_refused(exps):
    r = GroundedResponder(exps["base"])
    for q in ("Did real customers prefer B?", "Would real enterprise customers actually behave this way?", "What will B do to revenue next year?"):
        a = r.answer(q)
        assert a.refused and re.search(r"not supported|does not establish", a.text)


def test_false_premise_is_corrected(exps):
    a = GroundedResponder(exps["base"]).answer("Why did SMB buyers prefer B?")
    assert "Contrary to the question's premise" in a.text and "A" in a.text


def test_why_answers_are_associations_not_causes(exps):
    a = GroundedResponder(exps["base"]).answer("Why did enterprise buyers prefer B?")
    assert "not an established cause" in a.text and not re.search(r"\bbecause\b", a.text, re.I)


def test_all_cited_ids_exist_and_numbers_are_supported(exps):
    for ex in exps.values():
        r = GroundedResponder(ex)
        for q in H.QUESTIONS:
            a = r.answer(q.text)
            s = H.score(a, q, ex)
            assert s["pass"]["citation"] and s["pass"]["numeric"], (q.text, s["why"])


def test_grounded_responder_passes_every_dimension_in_every_scenario(exps):
    for name, ex in exps.items():
        rows = H.evaluate_responder(ex, GroundedResponder(ex))
        bad = [(r["qid"], {k: r[f"why_{k}"] for k in H.DIMS if f"why_{k}" in r}) for r in rows if not (r["p_all"] and r["p_intent"])]
        assert not bad, (name, bad[:3])


def test_unstable_answers_do_not_scream_a_winner(exps):
    for n in ("near_tie", "high_variance", "stale"):
        a = GroundedResponder(exps[n]).answer("Which option performed best?")
        assert not re.search(r"\bwins?\b|\bleads\b|clear winner", a.text, re.I)
        assert re.search(r"does not support a robust preference|out of date", a.text)


# ------------------------------------------------------------------ the evaluation harness itself
def test_question_bank_size_and_coverage():
    assert len(H.QUESTIONS) >= 40
    assert {q.cat for q in H.QUESTIONS} >= {"best_option", "segment_pref", "why", "real_world", "unsupported", "flip", "robustness"}


def test_ungrounded_baseline_scores_badly(exps):
    ex = exps["base"]
    rows = H.evaluate_responder(ex, H.UngroundedBaseline(ex))
    assert sum(r["p_all"] for r in rows) == 0
    assert np.mean([r["p_overreach"] for r in rows]) < 0.2


def test_every_injected_defect_is_caught_and_clean_answers_are_not_flagged(exps):
    mut, fp = H.mutation_test(list(exps.values()))
    for name, v in mut.items():
        assert v["applied"] > 100 and v["caught"] == v["applied"], name
    assert fp["flagged"] == 0


def test_checkers_have_high_precision_but_limited_recall_on_unseen_phrasing(exps):
    """Documented limitation: lexicon checkers do not generalize. This test pins the honest numbers."""
    adv = H.adversarial_recall(exps["base"])
    assert adv["false_positives"] == 0
    assert adv["caught"] < adv["overreach_cases"]


# ------------------------------------------------------------------ LLM adapter never overrides the guardrails
class _FakeClient:
    def __init__(self, text):
        self.messages = self
        self._t = text

    def create(self, **kw):
        class B:  # minimal anthropic-like response
            type = "text"
        b = B(); b.text = self._t

        class M: content = [b]
        return M()


def test_llm_answer_with_invented_evidence_falls_back_to_grounded(exps):
    ex = exps["base"]
    q = next(q for q in H.QUESTIONS if q.qid == "why-1")
    g = GroundedResponder(ex).answer(q.text)
    bad = _FakeClient("Enterprise buyers prefer B because they worry about outages. Real customers will definitely choose it.")
    out = llm_answer(ex, q.text, g, lambda a: H.score(a, q, ex)["pass"]["all"], client=bad)
    assert out.source.startswith("fallback") and out.text == g.text


def test_llm_answer_that_passes_the_checks_is_accepted(exps):
    ex = exps["base"]
    q = next(q for q in H.QUESTIONS if q.qid == "real_world-1")
    g = GroundedResponder(ex).answer(q.text)
    out = llm_answer(ex, q.text, g, lambda a: H.score(a, q, ex)["pass"]["all"], client=_FakeClient(g.text))
    assert out.source == "llm"


# ------------------------------------------------------------------ API & UI
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from decisiontrace.api import app
    return TestClient(app)


def test_api_scenarios_and_decision(client):
    s = client.get("/api/scenarios").json()
    assert {x["scenario"] for x in s} == set(scenarios.SCENARIOS)
    d = client.get("/api/decision/near_tie").json()
    assert d["state"] == "UNSTABLE" and "unstable" in d["headline"].lower()


def test_api_ask_returns_grounded_answer_with_guardrails(client):
    r = client.post("/api/ask", json={"scenario": "base", "question": "Did real customers prefer B?"}).json()
    assert r["refused"] and r["guardrails"]["observed"] and r["guardrails"]["overreach"]
    assert all(c["id"].startswith("EV-") for c in r["citations"])


def test_api_evidence_includes_provenance_chain(client):
    d = client.get("/api/decision/base").json()
    ev = client.get(f"/api/evidence/base/{d['stability']['evidence']}").json()
    assert ev["cls"] == "DERIVED" and ev["chain"] and "GENERATED" in ev["classes_in_chain"]
    assert client.get("/api/evidence/base/EV-9999").status_code == 404


def test_page_renders_and_changes_with_the_scenario(client):
    base = client.get("/?scenario=base").text
    tie = client.get("/?scenario=near_tie").text
    assert "leads A by" in base and "Ask the simulation" in base
    assert "Result unstable" in tie and "leads A by" not in tie
