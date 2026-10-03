"""Every threshold and assumption range lives here so it can be reviewed and challenged."""

OPTIONS = ("A", "B", "C")
OPTION_LABELS = {"A": "Lower price", "B": "Better reliability", "C": "Easier setup"}

SEGMENTS = {
    "company_size": ["SMB", "Mid-market", "Enterprise"],
    "buyer_type": ["Technical", "Nontechnical"],        # Technical = technical_sophistication >= TECH_THRESHOLD
    "industry": ["Software", "Healthcare", "Finance", "Retail", "Manufacturing"],
}
TECH_THRESHOLD = 0.6

# ---- when does the product change what it says?
MIN_AGENTS = 1000              # below this: INSUFFICIENT_EVIDENCE
MIN_SEGMENT_N = 200            # below this a segment is flagged as underrepresented
NEAR_TIE_PP = 3.0              # mean lead below this => UNSTABLE (near tie)
WIN_RATE_MIN = 0.80            # share of repeated runs in which the winner must lead
STABILITY_BANDS = [(0.90, "HIGH"), (0.70, "MODERATE"), (0.50, "LOW")]   # else UNSTABLE
ROBUST_WIN_RATE = 0.95
DISAGREE_MIN_WEIGHT = 0.10     # a segment must be >=10% of the population ...
DISAGREE_STRONG_MARGIN_PP = 10.0   # a dissenting segment this far ahead counts toward CONTESTED
DISAGREE_MIN_MARGIN_PP = 3.0   # ... and prefer another option by >=3pp to count as disagreement

# ---- the assumptions the decision depends on (tested range is what we perturb; phys is the outer bound for flip search)
ASSUMPTIONS = {
    "price_sensitivity_scale":  {"nominal": 1.0,  "tested": (0.8, 1.2),     "phys": (0.2, 2.5), "unit": "x",
                                 "desc": "scales how strongly price sensitivity drives preference for A"},
    "reliability_sensitivity":  {"nominal": 0.72, "tested": (0.576, 0.864), "phys": (0.2, 1.5), "unit": "coef",
                                 "desc": "how strongly reliability need drives preference for B"},
    "setup_sensitivity":        {"nominal": 1.0,  "tested": (0.8, 1.2),     "phys": (0.2, 2.5), "unit": "x",
                                 "desc": "scales how strongly setup friction drives preference for C"},
    "enterprise_share":         {"nominal": 0.25, "tested": (0.2125, 0.2875), "phys": (0.02, 0.60), "unit": "share",
                                 "desc": "enterprise representation in the weighted population (+/-15% relative)"},
    "technical_shift":          {"nominal": 0.0,  "tested": (-0.2, 0.2),    "phys": (-0.6, 0.6), "unit": "rel",
                                 "desc": "relative shift of the technical-sophistication distribution"},
    "temperature":              {"nominal": 1.0,  "tested": (0.8, 1.25),    "phys": (0.3, 3.0), "unit": "x",
                                 "desc": "choice stochasticity (softmax temperature) of simulated agents"},
}
ASSUMPTION_NAMES = list(ASSUMPTIONS)

N_RUNS = 20
N_LHS = 400
QUEUE_TOP = 5
