"""A synthetic stand-in for the portal's markets worlds, for offline tests.

It is NOT the organisers' simulator. It mimics the same ideas: 4 hidden macro
regimes that switch unannounced, signals that lead the data, late and missing
readings, an abrupt break, a data outage and a rare stress. Use it to check
that an agent behaves; use the portal's xlsx export for real calibration.
"""
from __future__ import annotations

import numpy as np

from core import LABELS, SIGNALS, Observation

REGIMES = ("goldilocks", "reflation", "stagflation", "deflation")
# Annual excess return (%) per class A..E in each regime. Stock/bond
# correlation is positive in stagflation, the 2022 pattern.
MU = np.array([
    [9, 10, 2, 3, 0],
    [7, 9, -3, 1, 10],
    [-6, -8, -5, -3, 8],
    [-4, -6, 6, 2, -10],
], float) * 1.5
VOL = np.array([[10, 15, 5, 4, 20], [11, 16, 6, 4, 22], [14, 20, 7, 6, 25], [15, 22, 5, 6, 25]], float)
CORR_NORMAL = np.array([[1, .75, -.2, .3, .2], [.75, 1, -.1, .3, .2], [-.2, -.1, 1, .5, -.1], [.3, .3, .5, 1, .1], [.2, .2, -.1, .1, 1]])
CORR_STAG = np.array([[1, .8, .5, .6, .2], [.8, 1, .5, .6, .2], [.5, .5, 1, .7, -.1], [.6, .6, .7, 1, .1], [.2, .2, -.1, .1, 1]])
LAG = {"growth": 2, "inflation": 5}


def make_world(scenario: str, seed: int, T: int = 416):
    rng = np.random.default_rng(seed)
    reg = np.zeros(T, int)
    r = rng.integers(4)
    for t in range(T):
        if rng.random() > 0.985:
            r = rng.integers(4)
        reg[t] = r
    if scenario == "abrupt":
        reg[T // 2:] = (reg[T // 2 - 1] + 2) % 4
    if scenario == "stagflation_long":
        reg[T // 3: T // 3 + 100] = 2
    stress = np.ones(T)
    if scenario == "rare_stress":
        stress[300:312] = 3.0
    rets = np.zeros((T, 5))
    for t in range(T):
        corr = CORR_STAG if reg[t] == 2 else CORR_NORMAL
        sd = VOL[reg[t]] * stress[t] / np.sqrt(52)
        cov = corr * np.outer(sd, sd)
        rets[t] = rng.multivariate_normal(MU[reg[t]] / 52, cov)
    g = np.where(np.isin(reg, [0, 1]), 0.6, -0.6)
    i = np.where(np.isin(reg, [1, 2]), 0.6, -0.6)
    sig = {k: np.zeros(T) for k in SIGNALS}
    sig["growth"] = np.r_[np.full(LAG["growth"], g[0]), g[:-LAG["growth"]]] + rng.normal(0, 0.45, T)
    sig["inflation"] = np.r_[np.full(LAG["inflation"], i[0]), i[:-LAG["inflation"]]] + rng.normal(0, 0.45, T)
    sig["market stress"] = -0.4 * g + rng.normal(0, 0.5, T)
    mom = np.convolve(rets[:, 0], np.ones(13) / 13, "same")
    sig["momentum"] = np.r_[0, mom[:-1]] * 15 + rng.normal(0, 0.4, T)
    sig["volatility"] = 0.3 * (stress - 1) + rng.normal(0, 0.4, T)
    for k in ("correlation level", "dispersion", "liquidity", "positioning"):
        sig[k] = rng.normal(0, 0.5, T)
    miss = rng.random((T, len(SIGNALS))) < 0.04
    if scenario == "outage":
        miss[200:215, :] = True
    return reg, rets, sig, miss, stress


def run(agent, scenario: str, seed: int, T: int = 416):
    from core import COST_BPS, DD_LIMIT, is_legal
    reg, rets, sig, miss, stress = make_world(scenario, seed, T)
    last = {k: None for k in SIGNALS}
    age = {k: 0 for k in SIGNALS}
    held = np.array([35.0, 10, 35, 15, 5])
    nav, peak, maxdd, lost = 100.0, 100.0, 0.0, False
    pnl_prev, costs, turn = None, 0.0, 0.0
    confs, escs, rows = [], [], []
    for t in range(T):
        for j, k in enumerate(SIGNALS):
            if miss[t, j]:
                age[k] += 1
            else:
                last[k], age[k] = sig[k][t], 0
        obs_sig = {k: (None if miss[t, j] else float(sig[k][t])) for j, k in enumerate(SIGNALS)}
        obs = Observation(
            signals=obs_sig, staleness=dict(age),
            returns=None if t == 0 else dict(zip(LABELS, rets[t - 1])), pnl=pnl_prev,
        )
        d = agent.decide(obs)
        x = np.array([d.units[l] for l in LABELS])
        assert is_legal(x), (t, x)
        trade = np.abs(x - held)
        c = float((trade / 100 * COST_BPS / 100).sum())  # % of capital
        pnl = float((x / 100 * rets[t]).sum()) - c
        nav *= 1 + pnl / 100
        peak = max(peak, nav)
        dd = 100 * (1 - nav / peak)
        maxdd = max(maxdd, dd)
        if dd >= DD_LIMIT:
            lost = True
        costs += c
        turn += trade.sum() / 100
        held, pnl_prev = x, pnl
        confs.append(d.confidence)
        escs.append(d.escalate)
        rows.append(reg[t])
    yrs = T / 52
    return dict(
        ret=((nav / 100) ** (1 / yrs) - 1) * 100, maxdd=maxdd, lost=lost,
        turnover=turn / yrs, costs=costs / yrs, esc=float(np.mean(escs)),
        conf_sd=float(np.std(confs)), conf=float(np.mean(confs)),
    )
