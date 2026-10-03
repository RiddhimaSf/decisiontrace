import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def exps():
    from decisiontrace import scenarios
    from decisiontrace.analysis import build_experiment
    return {n: build_experiment(scenarios.get(n)) for n in scenarios.SCENARIOS}
