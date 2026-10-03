"""Export a single self-contained HTML file with every scenario and pre-computed Q&A (works offline, no server)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decisiontrace import scenarios
from decisiontrace.analysis import build_experiment
from decisiontrace.ask import GroundedResponder
from decisiontrace.ui import PRESETS, render_page, render_scenario

out = Path(__file__).resolve().parents[1] / "demo" / "decision_screen.html"
names = list(scenarios.SCENARIOS)
titles = {n: scenarios.SCENARIOS[n].title for n in names}
blocks = []
for n in names:
    ex = build_experiment(scenarios.get(n))
    r = GroundedResponder(ex)
    blocks.append(render_scenario(ex, n, qa=[(q, r.answer(q)) for q in PRESETS], live=False, display="block" if n == "base" else "none"))
out.write_text(render_page(blocks, names, titles, "base", live=False))
print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB")
