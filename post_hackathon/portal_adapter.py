"""Glue between RobustAgent and the portal's agent.py.

UNVERIFIED: the editor only exposed a fragment of the portal's StatePacket
(state.quant.values, state.constraints). Check the real field names in the
starter file and adjust the three getattr lines marked CHECK. Everything else
is the tested core.
"""
from agents import RobustAgent
from core import Observation, SIGNALS

_agent = RobustAgent()


def to_observation(state) -> Observation:
    values = dict(getattr(state.quant, "values", {}))              # CHECK
    stale = dict(getattr(state.quant, "staleness", {}))            # CHECK
    rets = getattr(state, "returns", None)                         # CHECK
    return Observation(
        signals={k: values.get(k) for k in SIGNALS}, staleness=stale, returns=rets,
    )


def decide(state):
    d = _agent.decide(to_observation(state))
    return d.units, d.confidence, d.escalate, d.evidence
