"""Three agents, from simplest to most defensive.

ReferenceAgent   holds the reference book, scaled to the volatility target.
RegimeTiltAgent  reads growth and inflation, tilts by regime, pays costs only
                 when a position is far from its target.
RobustAgent      adds a drawdown guard, staleness aware confidence, and a
                 human escalation rule. This is the one to submit.

All share one decision pipeline in QuantAgent so the differences are flags.
"""
from __future__ import annotations

from math import exp, log, sqrt

import numpy as np

from core import (
    COST_BPS, DD_LIMIT, LABELS, PRIOR_COV, REFERENCE, REGISTERED_THRESHOLD,
    SIGNALS, VOL_TARGET, Decision, Observation, Zscorer, portfolio_vol,
    project, sigmoid,
)

REGIMES = ("goldilocks", "reflation", "stagflation", "deflation")

# Prior tilt tables, units A..E. Textbook macro priors: CALIBRATE THEM on the
# portal's "By regime" sheet (Data tab) before trusting them. Each row already
# respects the mandate once projected.
TILT = {
    "goldilocks": np.array([45.0, 12.0, 20.0, 15.0, 3.0]),
    "reflation":  np.array([40.0, 12.0, 15.0, 15.0, 8.0]),
    "stagflation": np.array([25.0, 0.0, 15.0, 20.0, 10.0]),
    "deflation":  np.array([20.0, 0.0, 55.0, 15.0, 0.0]),
}
# Where to hide when the book is falling. Bonds are NOT automatically safe:
# in stagflation both bonds and equities fell together in 2022.
SAFE = {
    "goldilocks": np.array([20.0, 0.0, 35.0, 15.0, 0.0]),
    "reflation":  np.array([20.0, 0.0, 25.0, 15.0, 10.0]),
    "stagflation": np.array([20.0, 0.0, 15.0, 25.0, 10.0]),
    "deflation":  np.array([20.0, 0.0, 50.0, 0.0, 0.0]),
}
# No trade band in units: wider for dearer classes (B 15 bps, D 10 bps).
BAND = np.array([3.0, 6.0, 2.5, 5.0, 3.0])


class QuantAgent:
    use_tilt = True
    use_bands = True
    use_guard = True
    use_confidence = True
    name = "quant"

    def __init__(self):
        self.z = Zscorer()
        self.last_good: dict[str, float] = {}
        self.held = REFERENCE.copy()
        self.target = REFERENCE.copy()
        self.probs = np.full(4, 0.25)
        self.stab = 1.0
        self.nav, self.peak = 100.0, 100.0
        self.rets: list[np.ndarray] = []
        self.pnl_hist: list[float] = []
        self.t = 0

    # ---- inputs -------------------------------------------------------
    def _effective_z(self, obs: Observation):
        """z-score of every signal, shrunk to 0 (no view) as it goes stale.
        A missing reading is never treated as a calm reading."""
        zs, weights = {}, {}
        for name in SIGNALS:
            v = obs.signals.get(name)
            age = obs.staleness.get(name, 0)
            if v is not None:
                self.last_good[name] = float(v)
                self.z.update(name, float(v))
                w = exp(-age / 6.0)
            else:
                age = max(age, 1)
                w = 0.0
            if name in self.last_good:
                decay = w if v is not None else exp(-age / 6.0) * 0.5
                zs[name] = self.z.z(name, self.last_good[name]) * decay
            else:
                zs[name] = 0.0
            weights[name] = w
        return zs, weights

    def _regime(self, zs):
        growth = 0.6 * zs["growth"] + 0.2 * zs["momentum"] - 0.2 * zs["market stress"]
        pg = sigmoid(1.6 * growth)
        pi = sigmoid(1.6 * zs["inflation"])
        raw = np.array([pg * (1 - pi), pg * pi, (1 - pg) * pi, (1 - pg) * (1 - pi)])
        raw = 0.85 * raw + 0.15 * 0.25  # keep a floor of humility
        probs = 0.7 * raw + 0.3 * self.probs  # smoothing: avoid flip-flopping
        probs /= probs.sum()
        tv = 0.5 * np.abs(probs - self.probs).sum()  # how fast the regime view moved
        self.stab = 0.8 * self.stab + 0.2 * (1 - min(1.0, 10 * tv))
        self.probs = probs
        return probs

    def _covariance(self):
        if len(self.rets) < 26:
            return PRIOR_COV
        r = np.array(self.rets[-104:]) * np.sqrt(52)
        sample = np.cov(r.T)
        lam = min(0.7, 26 / len(r))  # shrink to the prior while data is short
        return lam * PRIOR_COV + (1 - lam) * sample

    # ---- the decision ---------------------------------------------------
    def decide(self, obs: Observation) -> Decision:
        self.t += 1
        evidence: list[str] = []
        if obs.returns:
            self.rets.append(np.array([obs.returns.get(l, 0.0) for l in LABELS]))
        if obs.pnl is not None:
            self.nav *= 1 + obs.pnl / 100.0
            self.peak = max(self.peak, self.nav)
            self.pnl_hist.append(obs.pnl)
        dd = 100 * (1 - self.nav / self.peak)  # positive number, % below peak

        zs, weights = self._effective_z(obs)
        coverage = float(np.mean(list(weights.values())))
        probs = self._regime(zs)
        evidence.append(
            "regime odds " + ", ".join(f"{r} {p:.0%}" for r, p in zip(REGIMES, probs))
        )
        evidence.append(f"data coverage {coverage:.0%} (fresh readings only)")

        # surprise: a week far outside the book's own recent volatility
        surprise = 1.0
        if len(self.pnl_hist) > 20:
            sd = float(np.std(self.pnl_hist[-60:-1])) or 1e-6
            if abs(self.pnl_hist[-1]) > 3.5 * sd:
                surprise = 0.6
                evidence.append("last week was a >3.5 sigma move for this book")

        clarity = 1 - (-(probs * np.log(probs)).sum() / log(4))
        confidence = coverage ** 2 * (0.55 + 0.30 * clarity + 0.15 * self.stab) * surprise
        if coverage < 0.4:
            confidence *= 0.5
            evidence.append("most signals stale or missing: cutting confidence")
        confidence = float(np.clip(confidence, 0.02, 0.95))

        # 1) target book
        if self.use_tilt:
            blend = sum(p * TILT[r] for p, r in zip(probs, REGIMES))
            safe = sum(p * SAFE[r] for p, r in zip(probs, REGIMES))
            trust = confidence
            raw_target = REFERENCE + trust * (blend - REFERENCE)
        else:
            raw_target, safe = REFERENCE.copy(), REFERENCE.copy()

        # 2) drawdown guard: de-risk linearly between 6 % and 14 % below peak
        cut = 0.0
        if self.use_guard:
            cut = float(np.clip((dd - 6.0) / (14.0 - 6.0), 0, 1))
            if cut > 0:
                raw_target = (1 - cut) * raw_target + cut * safe
                evidence.append(f"drawdown {dd:.1f} %: moved {cut:.0%} toward the defensive book")
        target = project(raw_target)

        # 3) volatility target (annual %), using shrunk covariance
        vol = portfolio_vol(target, self._covariance())
        scale = min(1.0, VOL_TARGET / vol) if vol > 0 else 1.0
        if self.use_guard and scale < 1.0:
            target = project(target * scale)
            evidence.append(f"scaled risk by {scale:.2f} to hold the {VOL_TARGET:.0f} % vol target")

        # 4) pay costs only when the gap is big; trade fast in a storm
        urgent = cut > 0.3 or zs["market stress"] > 2.0 or surprise < 1
        if self.use_bands:
            self.target = target if urgent else 0.7 * self.target + 0.3 * target
            band = BAND * (0.25 if urgent else 1.0)
            new = self.held.copy()
            gap = self.target - self.held
            trade = np.abs(gap) > band
            new[trade] = self.target[trade]
            book = project(new)
        else:
            book = target
        self.held = book

        escalate = False
        if self.use_confidence:
            escalate = confidence < REGISTERED_THRESHOLD or dd > 0.75 * DD_LIMIT
            if dd > 0.75 * DD_LIMIT:
                evidence.append(f"drawdown {dd:.1f} % is within 5 points of losing the mandate")
        else:
            confidence = 0.6  # constant, like the frozen agent: scores zero on responsiveness

        return Decision(
            units={l: float(u) for l, u in zip(LABELS, book)},
            confidence=confidence,
            escalate=escalate,
            evidence=evidence,
            regime_probs=dict(zip(REGIMES, map(float, probs))),
            notes=f"dd={dd:.1f} cut={cut:.2f} vol={vol:.1f}",
        )


class ReferenceAgent(QuantAgent):
    name = "reference"
    use_tilt = False
    use_bands = False
    use_guard = False
    use_confidence = False


class RegimeTiltAgent(QuantAgent):
    name = "regime_tilt"
    use_guard = False
    use_confidence = False


class RobustAgent(QuantAgent):
    name = "robust"
