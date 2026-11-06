"""A synthetic stand-in for the portal's markets worlds, for offline tests.

It is NOT the organisers' simulator. It mimics the same ideas: 4 hidden macro
regimes that switch unannounced, signals that lead the data, late and missing
readings. Use it to check that an agent behaves; use the portal's xlsx export
for real calibration.

Scenarios: the portal's 10 episode types, plus a long stagflation.
  stable            stable but noisy
  drift             gradual drift from one regime to its opposite
  abrupt            abrupt regime change
  rare_stress       rare stress event
  recovery          a crash, then a rebound the macro data only sees late
  outage            missing data
  liquidity_shock   trading costs jump 5x, liquidity signal warns first
  crowding          a crowded equity trade builds up, then unwinds
  narrative_shock   alarming headlines and a stress spike, fundamentals unchanged
  cascade           herding: a self feeding rally, then a reversal
  stagflation_long  100 weeks of stagflation (not a portal episode)
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
    cost_mult = np.ones(T)
    texts: list[list[str]] = [[] for _ in range(T)]
    if scenario == "rare_stress":
        stress[300:312] = 3.0

    mu = MU[reg].copy()  # weekly drift per class, annual % before /52
    g = np.where(np.isin(reg, [0, 1]), 0.6, -0.6)
    i = np.where(np.isin(reg, [1, 2]), 0.6, -0.6)

    if scenario == "drift":
        # slide linearly from the starting regime to its opposite, no jump
        a, b = T // 4, 3 * T // 4
        r0 = reg[0]
        r1 = (r0 + 2) % 4
        reg[:] = r0
        for t in range(T):
            w = min(1.0, max(0.0, (t - a) / (b - a)))
            mu[t] = (1 - w) * MU[r0] + w * MU[r1]
            reg[t] = r1 if w > 0.5 else r0
        g0, g1 = (0.6 if r0 in (0, 1) else -0.6), (0.6 if r1 in (0, 1) else -0.6)
        i0, i1 = (0.6 if r0 in (1, 2) else -0.6), (0.6 if r1 in (1, 2) else -0.6)
        w = np.clip((np.arange(T) - a) / (b - a), 0, 1)
        g, i = (1 - w) * g0 + w * g1, (1 - w) * i0 + w * i1

    if scenario == "recovery":
        # 12-week crash, then a strong rebound while growth data still reads weak
        reg[150:162], reg[162:230] = 3, 0
        mu[150:162] = MU[3] * 2.5
        stress[150:162] = 2.5
        mu[162:230] = MU[0] * 2.0
        stress[162:190] = np.linspace(2.0, 1.0, 28)
        g = np.where(np.isin(reg, [0, 1]), 0.6, -0.6)
        i = np.where(np.isin(reg, [1, 2]), 0.6, -0.6)
        g[162:200] = -0.6  # the macro data lags the market rebound

    if scenario == "liquidity_shock":
        cost_mult[250:282] = 5.0
        stress[250:282] = 1.5

    if scenario == "crowding":
        # equities grind up as positioning builds, then a fast unwind
        mu[180:230, :2] += 6.0
        mu[230:238, :2] = -120.0
        stress[230:238] = 2.0

    if scenario == "narrative_shock":
        for t in range(220, 228):
            texts[t] = ["Analysts warn of imminent collapse", "Panic spreads on social media"]

    if scenario == "cascade":
        # herding: a self feeding equity rally, then a sharp reversal
        mu[260:300, :2] += np.linspace(5, 30, 40)[:, None]
        mu[300:308, :2] = -150.0
        stress[300:308] = 2.5

    rets = np.zeros((T, 5))
    for t in range(T):
        corr = CORR_STAG if reg[t] == 2 else CORR_NORMAL
        sd = VOL[reg[t]] * stress[t] / np.sqrt(52)
        cov = corr * np.outer(sd, sd)
        rets[t] = rng.multivariate_normal(mu[t] / 52, cov)
    sig = {k: np.zeros(T) for k in SIGNALS}
    sig["growth"] = np.r_[np.full(LAG["growth"], g[0]), g[:-LAG["growth"]]] + rng.normal(0, 0.45, T)
    sig["inflation"] = np.r_[np.full(LAG["inflation"], i[0]), i[:-LAG["inflation"]]] + rng.normal(0, 0.45, T)
    sig["market stress"] = -0.4 * g + rng.normal(0, 0.5, T)
    mom = np.convolve(rets[:, 0], np.ones(13) / 13, "same")
    sig["momentum"] = np.r_[0, mom[:-1]] * 15 + rng.normal(0, 0.4, T)
    sig["volatility"] = 0.3 * (stress - 1) + rng.normal(0, 0.4, T)
    for k in ("correlation level", "dispersion", "liquidity", "positioning"):
        sig[k] = rng.normal(0, 0.5, T)

    if scenario == "liquidity_shock":
        sig["liquidity"][246:282] -= 2.0  # warns 4 weeks before costs jump
    if scenario == "crowding":
        sig["positioning"][180:230] += np.linspace(0.5, 2.5, 50)
        sig["positioning"][230:238] -= 1.5
    if scenario == "narrative_shock":
        sig["market stress"][220:228] += 2.5  # sentiment only, returns unchanged
    if scenario == "cascade":
        sig["correlation level"][260:308] += 2.0
        sig["dispersion"][260:308] -= 2.0
        sig["positioning"][270:300] += 2.0

    miss = rng.random((T, len(SIGNALS))) < 0.04
    if scenario == "outage":
        miss[200:215, :] = True
    return reg, rets, sig, miss, stress, cost_mult, texts


def run(agent, scenario: str, seed: int, T: int = 416):
    from core import COST_BPS, DD_LIMIT, is_legal
    reg, rets, sig, miss, stress, cost_mult, texts = make_world(scenario, seed, T)
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
            texts=texts[t],
        )
        d = agent.decide(obs)
        x = np.array([d.units[l] for l in LABELS])
        assert is_legal(x), (t, x)
        trade = np.abs(x - held)
        c = float((trade / 100 * COST_BPS * cost_mult[t] / 100).sum())  # % of capital
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
