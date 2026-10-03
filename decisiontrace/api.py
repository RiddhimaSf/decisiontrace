"""FastAPI surface:  uvicorn decisiontrace.api:app --reload      then open http://localhost:8000

GET  /                                  decision screen (?scenario=base|imbalance|...)
GET  /api/scenarios                     scenarios with their current presentation state
GET  /api/decision/{scenario}           the decision view-model (JSON)
GET  /api/evidence/{scenario}/{id}      one ledger row with its provenance chain
POST /api/ask {scenario, question}      grounded answer + citations + guardrail results
"""
from __future__ import annotations

import re
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import scenarios
from .analysis import build_experiment
from .ask import GroundedResponder
from .evalharness import Q, score
from .ui import evidence_display, render_page, render_scenario

app = FastAPI(title="DecisionTrace", version="0.1")


@lru_cache(maxsize=None)
def experiment(name: str):
    if name not in scenarios.SCENARIOS:
        raise HTTPException(404, f"unknown scenario {name!r}")
    return build_experiment(scenarios.get(name))


class AskRequest(BaseModel):
    scenario: str = "base"
    question: str


@app.get("/api/scenarios")
def list_scenarios():
    out = []
    for n in scenarios.SCENARIOS:
        d = experiment(n).decision
        out.append({"scenario": n, "title": d["title"], "state": d["state"], "headline": d["headline"], "confidence": d["confidence"]})
    return out


@app.get("/api/decision/{scenario}")
def decision(scenario: str):
    return experiment(scenario).decision


@app.get("/api/evidence/{scenario}/{evid}")
def evidence(scenario: str, evid: str):
    L = experiment(scenario).ledger
    row = L.by_id(evid)
    if row is None:
        raise HTTPException(404, "no such evidence id")
    return {**row, "display": evidence_display(row), "chain": L.chain(evid), "classes_in_chain": sorted(L.classes_in_chain(evid))}


@app.post("/api/ask")
def ask(req: AskRequest):
    ex = experiment(req.scenario)
    ans = GroundedResponder(ex).answer(req.question)
    g = score(ans, Q("live", req.question, ans.intent), ex)["pass"]
    L = ex.ledger
    return {"scenario": req.scenario, "question": req.question, "intent": ans.intent, "text": ans.text, "refused": ans.refused,
            "source": ans.source, "citations": [{"id": c, "key": L.by_id(c)["key"], "cls": L.by_id(c)["cls"]} for c in ans.citations],
            "guardrails": g}


@app.get("/", response_class=HTMLResponse)
def home(scenario: str = "base"):
    names = list(scenarios.SCENARIOS)
    titles = {n: scenarios.SCENARIOS[n].title for n in names}
    ex = experiment(scenario)
    return render_page([render_scenario(ex, scenario, live=True)], names, titles, scenario, live=True)
