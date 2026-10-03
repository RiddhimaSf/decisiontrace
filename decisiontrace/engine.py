"""The simulation: a TEST FIXTURE, not a model of real buyers.

DecisionTrace's engineering starts *after* this module produces results. The fixture only needs to
produce plausible-looking, segment-structured output with controllable fragility, so the layer
above it (evidence ledger, flip analysis, interrogation, evaluation) has something to work on.

Synthetic agents have attributes (company size, industry, role, technical sophistication, price
sensitivity, switching cost, setup pain). Each agent chooses among three messages with a softmax
over utilities:   A: lower price   B: better reliability   C: easier setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C

SIZES = C.SEGMENTS["company_size"]
INDUSTRIES = C.SEGMENTS["industry"]
BUYER = C.SEGMENTS["buyer_type"]


@dataclass
class Spec:
    name: str = "base"
    title: str = "Baseline experiment"
    question: str = "Which messaging strategy should we launch?"
    n_agents: int = 8000
    seed: int = 42
    n_runs: int = C.N_RUNS
    n_lhs: int = C.N_LHS
    size_mix: tuple = (0.50, 0.25, 0.25)          # SMB, Mid-market, Enterprise (nominal population mix)
    assumptions: dict = field(default_factory=lambda: {k: v["nominal"] for k, v in C.ASSUMPTIONS.items()})
    run_noise_sd: float = 0.04                    # run-to-run jitter of the sensitivity coefficients
    drop_segments: tuple = ()                     # e.g. (("company_size", "Enterprise"),) -> missing from the population
    run_input_version: str = "v2"
    current_input_version: str = "v2"
    note: str = ""
    # fixture knobs (scenario designers can bend these)
    k_price: float = 2.54
    k_rel: float = 5.98
    k_setup: float = 5.01
    b_price: float = -1.15
    b_rel: float = -3.16
    b_setup: float = -2.28
    k_tech_rel: float = 1.7           # technical buyers value reliability more

    def nominal(self) -> dict:
        return dict(self.assumptions)


def _beta(rng, a, b, n):
    return rng.beta(a, b, n)


def generate_population(spec: Spec) -> dict:
    rng = np.random.default_rng(spec.seed)
    n = spec.n_agents
    size = rng.choice(3, n, p=np.array(spec.size_mix) / sum(spec.size_mix))
    industry = rng.choice(5, n, p=[.30, .20, .20, .15, .15])
    role = rng.choice(4, n, p=[.30, .25, .25, .20])                 # eng, it-ops, procurement, exec
    tech = np.where(role < 2, _beta(rng, 5, 2.5, n), _beta(rng, 2, 4, n))
    price = np.select([size == 0, size == 1], [_beta(rng, 5, 3, n), _beta(rng, 3.5, 3.5, n)], _beta(rng, 2.5, 4.5, n))
    p_cur = np.array([[.35, .35, .30], [.20, .40, .40], [.10, .55, .35]])
    u = rng.random(n)
    cum = np.cumsum(p_cur[size], axis=1)
    cur = (u[:, None] > cum).sum(axis=1)                              # 0 none, 1 legacy, 2 competitor
    switch = np.clip(0.15 + 0.30 * (cur == 1) + 0.20 * (cur == 2) + 0.25 * (size / 2) + 0.20 * _beta(rng, 2, 3, n), 0, 1)
    reg = np.isin(industry, [1, 2]).astype(float)                    # healthcare, finance
    rel_need = np.clip(0.30 * switch + 0.30 * (size / 2) + 0.20 * reg + 0.20 * _beta(rng, 2, 2, n), 0, 1)
    setup_noise = _beta(rng, 2, 2, n)
    return {"size": size, "industry": industry, "role": role, "tech": tech, "price": price, "switch": switch,
            "rel_need": rel_need, "setup_noise": setup_noise, "cur": cur,
            "buyer": (tech < C.TECH_THRESHOLD).astype(int),          # 0 Technical, 1 Nontechnical
            "alive": _alive_mask(spec, size, tech, industry)}


def _alive_mask(spec, size, tech, industry):
    alive = np.ones(len(size), bool)
    for dim, val in spec.drop_segments:
        if dim == "company_size":
            alive &= size != SIZES.index(val)
        elif dim == "buyer_type":
            alive &= ((tech < C.TECH_THRESHOLD).astype(int) != BUYER.index(val))
        elif dim == "industry":
            alive &= industry != INDUSTRIES.index(val)
    return alive


def weights(pop: dict, spec: Spec, enterprise_share: float) -> np.ndarray:
    """Post-stratification weights so the weighted population has the requested enterprise share
    (other sizes keep their relative nominal proportions)."""
    size, alive = pop["size"], pop["alive"]
    nonent = np.array(spec.size_mix[:2], float)
    nonent = nonent / nonent.sum()
    target = np.array([(1 - enterprise_share) * nonent[0], (1 - enterprise_share) * nonent[1], enterprise_share])
    emp = np.array([max(1, (alive & (size == k)).sum()) for k in range(3)], float)
    emp = emp / emp.sum()
    # if a size is absent (dropped), redistribute its target mass proportionally across the present sizes
    present = np.array([(alive & (size == k)).sum() > 0 for k in range(3)])
    t = np.where(present, target, 0.0)
    t = t / t.sum()
    w = (t / emp)[size] * alive
    return w


def choice_probs(pop: dict, spec: Spec, a: dict) -> np.ndarray:
    tech = np.clip(pop["tech"] * (1 + a["technical_shift"]), 0, 1)
    pain = 0.65 * (1 - tech) + 0.35 * pop["setup_noise"]
    ua = spec.k_price * a["price_sensitivity_scale"] * pop["price"] + spec.b_price
    ub = spec.k_rel * (a["reliability_sensitivity"] / 0.72) * pop["rel_need"] + spec.k_tech_rel * tech + spec.b_rel
    uc = spec.k_setup * a["setup_sensitivity"] * pain * (1 - 0.35 * tech) + spec.b_setup
    u = np.stack([ua, ub, uc], axis=1) / a["temperature"]
    u -= u.max(axis=1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(axis=1, keepdims=True)


def expected_shares(pop, spec, a) -> np.ndarray:
    """Noise-free preference shares under assumption set `a` (used for perturbation analysis)."""
    w = weights(pop, spec, a["enterprise_share"])
    p = choice_probs(pop, spec, a)
    return (w[:, None] * p).sum(axis=0) / w.sum()


def segment_masks(pop) -> dict:
    out = {}
    for dim, vals in C.SEGMENTS.items():
        key = {"company_size": "size", "buyer_type": "buyer", "industry": "industry"}[dim]
        for i, v in enumerate(vals):
            out[(dim, v)] = (pop[key] == i) & pop["alive"]
    return out


def simulate_run(pop, spec, a, rng, run_noise_sd=None) -> dict:
    """One simulation run: jitter the coefficients (run-level variance), let every agent choose, aggregate."""
    sd = spec.run_noise_sd if run_noise_sd is None else run_noise_sd
    b = dict(a)
    for k in ("price_sensitivity_scale", "reliability_sensitivity", "setup_sensitivity"):
        b[k] = a[k] * float(np.exp(rng.normal(0, sd)))
    p = choice_probs(pop, spec, b)
    u = rng.random(len(p))
    ch = (u[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, 2)
    w = weights(pop, spec, a["enterprise_share"])
    onehot = np.eye(3)[ch]
    overall = (w[:, None] * onehot).sum(axis=0) / w.sum()
    seg = {}
    for key, m in segment_masks(pop).items():
        ws = w * m
        seg[key] = ((ws[:, None] * onehot).sum(axis=0) / ws.sum()) if ws.sum() > 0 else np.full(3, np.nan)
    return {"overall": overall, "segments": seg}


def segment_profile(pop, spec, a) -> dict:
    """Deterministic segment descriptives under the nominal weighting: weight, n, mean drivers."""
    w = weights(pop, spec, a["enterprise_share"])
    tech = np.clip(pop["tech"] * (1 + a["technical_shift"]), 0, 1)
    pain = 0.65 * (1 - tech) + 0.35 * pop["setup_noise"]
    attrs = {"reliability_need": pop["rel_need"], "price_sensitivity": pop["price"],
             "setup_friction": pain, "technical_sophistication": tech, "switching_cost": pop["switch"]}
    out = {"__overall__": {"weight": 1.0, "n": int(pop["alive"].sum()),
                           **{k: float((w * v).sum() / w.sum()) for k, v in attrs.items()}}}
    for key, m in segment_masks(pop).items():
        ws = w * m
        out[key] = {"weight": float(ws.sum() / w.sum()), "n": int(m.sum()),
                    **{k: (float((ws * v).sum() / ws.sum()) if ws.sum() > 0 else None) for k, v in attrs.items()}}
    return out
