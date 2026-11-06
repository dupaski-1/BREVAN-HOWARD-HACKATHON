"""python -m pytest test_agents.py  (or: python test_agents.py)"""
import numpy as np

from agents import ReferenceAgent, RegimeTiltAgent, RobustAgent
from core import LABELS, SIGNALS, Observation, is_legal, project
from sim import run
from backtest import SCENARIOS


def test_projection_always_legal():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        assert is_legal(project(rng.uniform(-30, 120, 5)))


def test_garbage_inputs_never_break_the_mandate():
    a = RobustAgent()
    for sig in ({k: None for k in SIGNALS}, {k: 1e9 for k in SIGNALS}, {k: float("nan") for k in SIGNALS}):
        d = a.decide(Observation(signals=sig, staleness={k: 99 for k in SIGNALS}))
        assert is_legal([d.units[l] for l in LABELS])


def test_missing_data_is_not_calm():
    a = RobustAgent()
    good = {k: 0.3 for k in SIGNALS}
    for _ in range(30):
        calm = a.decide(Observation(good, {k: 0 for k in SIGNALS}))
    dark = a.decide(Observation({k: None for k in SIGNALS}, {k: 6 for k in SIGNALS}))
    assert dark.confidence < 0.4 and dark.escalate and calm.confidence > dark.confidence


def test_mandate_rules_hold_in_every_scenario():
    for sc in SCENARIOS:
        for cls in (ReferenceAgent, RegimeTiltAgent, RobustAgent):
            run(cls(), sc, 7)  # sim.run asserts legality every period


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
