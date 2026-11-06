"""Shared building blocks for the Uncertain World Simulator agents.

Mandate 2.1.0 as shown on the portal: units are % of capital.
  A developed equity 0-60, B emerging equity 0-20, C US 10y 15-55,
  D investment grade credit 0-30, E oil 0-10, A+B 20-60, cash <= 30,
  no leverage, mandate lost at a 20 % drawdown, volatility target 7 %.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt

import numpy as np

LABELS = ("A", "B", "C", "D", "E")
LOW = np.array([0.0, 0.0, 15.0, 0.0, 0.0])
HIGH = np.array([60.0, 20.0, 55.0, 30.0, 10.0])
EQ_MIN, EQ_MAX = 20.0, 60.0
TOTAL, MIN_INVESTED = 100.0, 70.0  # cash at most 30
COST_BPS = np.array([5.0, 15.0, 2.0, 10.0, 5.0])  # one way, per traded notional
REFERENCE = np.array([35.0, 10.0, 35.0, 15.0, 5.0])
VOL_TARGET = 7.0  # annual %, from the portal
DD_LIMIT = 20.0
REGISTERED_THRESHOLD = 0.40  # the escalation threshold shown on the portal

SIGNALS = (
    "correlation level", "dispersion", "growth", "inflation", "liquidity",
    "market stress", "momentum", "positioning", "volatility",
)

# Prior annual vols (%) and correlations. These are assumptions, replaced by a
# rolling estimate as soon as asset returns are observed.
PRIOR_VOL = np.array([16.0, 22.0, 7.0, 6.0, 30.0])
PRIOR_CORR = np.array([
    [1.0, 0.75, 0.0, 0.35, 0.2],
    [0.75, 1.0, 0.0, 0.3, 0.2],
    [0.0, 0.0, 1.0, 0.5, -0.1],
    [0.35, 0.3, 0.5, 1.0, 0.1],
    [0.2, 0.2, -0.1, 0.1, 1.0],
])
PRIOR_COV = PRIOR_CORR * np.outer(PRIOR_VOL, PRIOR_VOL)


def _move(x: np.ndarray, idx: list[int], delta: float) -> None:
    """Add delta (can be negative) over idx in proportion to the room left."""
    cap = (HIGH[idx] - x[idx]) if delta > 0 else (x[idx] - LOW[idx])
    total = cap.sum()
    if total <= 1e-12:
        return
    amount = min(abs(delta), total)
    x[idx] += np.sign(delta) * amount * cap / total


def project(order) -> np.ndarray:
    """Closest legal book. The portal does this too, but doing it ourselves
    means our own turnover and cost estimates match what really trades."""
    x = np.clip(np.nan_to_num(np.asarray(order, float), nan=0.0), LOW, HIGH)
    for _ in range(60):
        eq = x[0] + x[1]
        if eq < EQ_MIN - 1e-9:
            _move(x, [0, 1], EQ_MIN - eq)
            continue
        if eq > EQ_MAX + 1e-9:
            _move(x, [0, 1], EQ_MAX - eq)
            continue
        s = x.sum()
        if s > TOTAL + 1e-9:
            before = x.copy()
            _move(x, [2, 3, 4], TOTAL - s)
            if np.allclose(before, x):
                _move(x, [0, 1], TOTAL - s)
            continue
        if s < MIN_INVESTED - 1e-9:
            before = x.copy()
            _move(x, [2, 3], MIN_INVESTED - s)
            if np.allclose(before, x):
                _move(x, [0, 1, 4], MIN_INVESTED - s)
            continue
        break
    return x


def is_legal(x) -> bool:
    x = np.asarray(x, float)
    return bool(
        np.all(x >= LOW - 1e-6) and np.all(x <= HIGH + 1e-6)
        and EQ_MIN - 1e-6 <= x[0] + x[1] <= EQ_MAX + 1e-6
        and MIN_INVESTED - 1e-6 <= x.sum() <= TOTAL + 1e-6
    )


def portfolio_vol(units, cov=PRIOR_COV) -> float:
    w = np.asarray(units, float) / 100.0
    return float(sqrt(max(w @ cov @ w, 0.0)))


class Zscorer:
    """Online z-scores that are usable from the first few observations."""

    def __init__(self, halflife: int = 52):
        self.alpha = 1 - 0.5 ** (1 / halflife)
        self.mean: dict[str, float] = {}
        self.var: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def update(self, key: str, x: float) -> None:
        n = self.n.get(key, 0)
        if n == 0:
            self.mean[key], self.var[key] = x, 0.25
        else:
            a = max(self.alpha, 1.0 / (n + 1))
            d = x - self.mean[key]
            self.mean[key] += a * d
            self.var[key] = (1 - a) * (self.var[key] + a * d * d)
        self.n[key] = n + 1

    def z(self, key: str, x: float) -> float:
        if self.n.get(key, 0) < 4:
            return x / 0.5
        return (x - self.mean[key]) / sqrt(max(self.var[key], 1e-4))


def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + exp(-max(-30.0, min(30.0, v))))


@dataclass
class Observation:
    """What an agent sees in one period."""
    signals: dict[str, float | None]
    staleness: dict[str, int] = field(default_factory=dict)  # periods old
    returns: dict[str, float] | None = None  # delayed class returns, % over cash
    pnl: float | None = None  # last week's P&L in % of capital, if known
    texts: list[str] = field(default_factory=list)  # deliberately not trusted


@dataclass
class Decision:
    units: dict[str, float]
    confidence: float
    escalate: bool
    evidence: list[str]
    regime_probs: dict[str, float] = field(default_factory=dict)
    notes: str = ""
