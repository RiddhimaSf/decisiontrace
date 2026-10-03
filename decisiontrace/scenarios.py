"""The base experiment plus eight ways to break it. Each broken scenario is a controlled degradation of the
evidence; the product is expected to change what it says (see SCENARIO_EXPECTED)."""
from __future__ import annotations

from . import engine as E

SCENARIOS = {}


def _s(name, title, note, **kw):
    sp = E.Spec(name=name, title=title, note=note)
    a = kw.pop("assumptions", {})
    sp.assumptions.update(a)
    for k, v in kw.items():
        setattr(sp, k, v)
    SCENARIOS[name] = sp
    return sp


_s("base", "Baseline", "Three messages, 8,000 simulated agents, 20 runs.")
_s("imbalance", "Population imbalance",
   "Enterprise is weighted to 25% of the result but only ~2% of generated agents exist.", size_mix=(0.70, 0.28, 0.02))
_s("missing_segment", "Missing segment", "No Enterprise agents were generated.", drop_segments=(("company_size", "Enterprise"),))
_s("high_variance", "High run-to-run variance", "The simulator is noisy: coefficients jitter strongly between runs.", run_noise_sd=0.45)
_s("contradictory", "Contradictory subgroups", "SMB is the majority and strongly prefers A; Enterprise strongly prefers B.",
   size_mix=(0.58, 0.17, 0.25))
_s("stale", "Stale simulation", "Run produced from input v1; inputs have since moved to v2.", run_input_version="v1", current_input_version="v2")
_s("insufficient_sample", "Insufficient sample", "Only 300 simulated agents.", n_agents=300)
_s("near_tie", "Near tie", "Reliability matters slightly less, so B barely leads A.", assumptions={"reliability_sensitivity": 0.655})

# which presentation state the product must reach in each scenario
SCENARIO_EXPECTED = {
    "base": {"MODERATE", "ROBUST", "LOW"},
    "imbalance": {"INSUFFICIENT_EVIDENCE"},
    "missing_segment": {"INSUFFICIENT_EVIDENCE"},
    "high_variance": {"UNSTABLE"},
    "contradictory": {"CONTESTED"},
    "stale": {"STALE"},
    "insufficient_sample": {"INSUFFICIENT_EVIDENCE"},
    "near_tie": {"UNSTABLE"},
}


def get(name: str) -> E.Spec:
    sp = SCENARIOS[name]
    return E.Spec(**{k: (dict(getattr(sp, k)) if k == "assumptions" else getattr(sp, k)) for k in sp.__dataclass_fields__})
