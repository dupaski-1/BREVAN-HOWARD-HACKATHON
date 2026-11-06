"""Part 2's starter: the reference book inside the mandate, with a place for each module.

Your part 1 agent runs unchanged in part 2, in another world: five asset
classes instead of strategies, a week per period, and the mandate's rules
(the mandate's page, at the head of part 2 in the steps tab). This file is
the other way to start part 2. It holds the reference allocation every week,
inside every band, and marks where each module's idea goes. It validates no
module by itself, and held as it is it loses mandates: that is what the
modules are for.

One unit is 1 % of capital and the 100 units are all of it: what you do not
put in a class is cash.
"""

from __future__ import annotations

from collections import deque
from math import exp, isfinite, sqrt

from uws.agent.base import BaseAgent
from uws.contracts.action import Action, ScenarioBranch, ScenarioUncertainty, Uncertainty
from uws.contracts.config import DevContext
from uws.contracts.constants import STRATEGY_LABELS, TOTAL_RISK_UNITS
from uws.contracts.feedback import Feedback
from uws.contracts.state import StatePacket

# The mandate, as its page states it. Units are % of capital.
CLASSES = {
    "A": "Developed equity",
    "B": "Emerging equity",
    "C": "US 10-year Treasury",
    "D": "Investment-grade credit",
    "E": "Oil (WTI futures)",
}
REFERENCE = {"A": 35.0, "B": 10.0, "C": 35.0, "D": 15.0, "E": 5.0}
"""The reference allocation: the yardstick of every module and of the real world."""
BANDS = {"A": (0.0, 60.0), "B": (0.0, 20.0), "C": (15.0, 55.0), "D": (0.0, 30.0), "E": (0.0, 10.0)}
"""Each class's band. A and B together stay within EQUITY_BAND."""
EQUITY_BAND = (20.0, 60.0)
CASH_MAX = 30.0
"""At most 30 units in cash: at least 70 invested."""
BURN_IN_WEEKS = 104
"""Two years of warm-up: the mandate is measured from the week after."""

REGIMES = ("goldilocks", "inflation", "recession", "stagflation")
"""Module 23's four regimes. Name your scenario branches exactly so."""

ESCALATION_THRESHOLD = 0.40
CONFIDENCE = 0.6
HISTORY_WEEKS = 52
"""Weeks of returns kept for you: a year."""

## Module 2
DEFENSIVE = {"A":20.0,"B":0.0, "C":50.0,"D":0.0,"E":0.0}
DEFENSIVE_NO_BONDS = {"A":20.0,"B":0.0, "C":50.0,"D":0.0,"E":0.0}  # credit+oil version lost 2 mandates at 18:44
DD_START,DD_LIMIT = 5.0,13.0
RECENT_WEEKS = 26 #half a year according to module.

# Module 4, macro
TILT_UNITS = 8.0
MOM_WEEKS = 13
CUT_MARGIN = 0.10
"""How far the vol factor or the drawdown share must fall before a cut trades at once."""
RISK_BUDGET = {"A": 0.33, "B": 0.24, "C": 0.19, "D": 0.20, "E": 0.04}
"""Share of the risk each class carries: little for oil, whose futures roll drags its return."""
REGIME_PRIOR = {
    "goldilocks":  {"A": 1.0,  "B": 1.0,  "C": 0.0,  "D": 0.5,  "E": 0.0},
    "inflation":   {"A": 0.0,  "B": 0.5,  "C": -1.0, "D": -0.5, "E": 1.0},
    "recession":   {"A": -1.0, "B": -1.0, "C": 1.0,  "D": -0.5, "E": -1.0},
    "stagflation": {"A": -1.0, "B": -0.5, "C": -1.0, "D": -1.0, "E": 1.0},
}
"""Textbook direction of each class by regime (module 3: the check worlds assume the textbook)."""
REGIME_WEIGHT = 0.7  # unused now: see W_LEARNT, W_PRIOR, W_MOM
W_LEARNT, W_PRIOR, W_MOM = 0.8, 0.2, 0.1
"""Shares of the tilt: learnt expected returns by regime, textbook priors (both sized by confidence), and momentum."""
LEARNT_REF = 0.10
"""A weekly expected return of 0.10 of a week's volatility above the average earns the full tilt."""
MARKET_SHARE = 0.4
"""Share of the regime chances taken from market-led odds rather than the macro view."""
"""Share of the tilt from the regime view; the rest from 13-week momentum."""
SURE = 0.6
G_CUT = -1.00
MIN_WEEKS = 52
NAMED = {
    (False, False): "goldilocks",
    (False, True): "inflation",
    (True, False): "recession",
    (True, True): "stagflation"
}

class MyAgent(BaseAgent):
    """The reference book, held, with a marked place for each module."""

    name = "my-team-agent"

    def __init__(self) -> None:
        # What the modules need, kept up to date for you.
        self.returns: dict[str, deque[float]] = {
            label: deque(maxlen=HISTORY_WEEKS) for label in STRATEGY_LABELS
        }
        """Each class's recent weekly returns (0.004 is +0.4 %), oldest first."""
        self.pnl = 0.0
        """Net P&L since the warm-up ended, in % of capital."""
        self.peak = 0.0
        """Its highest point so far."""
        self.book = dict.fromkeys(STRATEGY_LABELS, 0.0)
        """What you hold, in units: nothing before week 0."""
        self.last_view: dict[str, float] | None = None
        """Last week's view of the regime (`_regime_view`), None before the first week."""
        self.signals: dict[str, float] = {}
        """The latest reading of each signal (`state.quant.values`), the last one seen when one
        is missing: `self.signals.get("volatility", 0.0)`."""
        self.z: dict[str, float] = {}
        self.named_weeks: dict[str, int] = {k: 0 for k in REGIMES}
        self.regime_weeks: dict[str, int] = {k: 0 for k in REGIMES}
        self.regime_sum: dict[str, dict[str, float]] = {
            k: dict.fromkeys(STRATEGY_LABELS, 0.0) for k in REGIMES
        }
        self.all_weeks = 0
        self.all_sum: dict[str, float] = dict.fromkeys(STRATEGY_LABELS, 0.0)

        self.vol_history: deque[float] = deque(maxlen=156)
        """The 'volatility' signal, week by week, over the last 3 years: its median is the usual level."""
        # M3 macro
        self.all_weeks = 0
        self.all_sum:dict[str, float] = dict.fromkeys(STRATEGY_LABELS, 0.0)

    def warmup(self, dev_context: DevContext) -> None:
        """Called once, before week 0."""
        dev_context.escalation.declare(ESCALATION_THRESHOLD)

    def on_feedback(self, feedback: Feedback) -> None:
        """A closed week's result, a week late."""
        if feedback.for_period == BURN_IN_WEEKS:
            self.pnl, self.peak = 0.0, 0.0
        self.pnl += feedback.realized_pnl_net
        self.peak = max(self.peak, self.pnl)

    @property
    def drawdown(self) -> float:
        """How far the P&L is below its highest point, in % of capital. 20 loses the mandate."""
        return self.peak - self.pnl

    def decide(self, state: StatePacket) -> Action:
        """This week's book: the budget, the view, the tilt, the drawdown cushion, the trade.

        The cushion (module 22) comes after the tilt (module 24), so it keeps the
        last word: a tilt never puts back the risk the cushion has cut.
        """
        self._remember(state)
        target = self._risk_budget(state)  # module 21
        view = self._regime_view(state)  # module 23
        expected = self._regime_returns(state)  # module 23
        target = self._tilt(target, view, expected)  # module 24
        target = self._keep_the_mandate(target)  # module 22
        order = self._trade(target)  # module 25
        action = self._action(order, view, state, expected)
        self.last_view = view
        return action

    def _risk_budget(self, state: StatePacket) -> dict[str, float]:
        labels = tuple(STRATEGY_LABELS)
        rows= [
            row for row in zip(*(self.returns[label] for label in labels))
            if all(isfinite(value) for value in row)
        ]
        if len(rows)< 26:
            return dict(REFERENCE)
        n=len(rows)
        means = [sum(row[i] for row in rows)/n for i in range(5)]
        cov = [
            [
                sum(
                    (row[i]-means[i])*(row[j]-means[j])
                    for row in rows
                )/(n-1)
                for j in range(5)
            ]
            for i in range(5)
        ]
        if not all(isfinite(value) for row in cov for value in row):
            return dict(REFERENCE)

        trace = sum (cov[i][i] for i in range(5))
        if trace <= 1e-16:
            return dict(REFERENCE)

        ridge = max(trace*1e-8/5, 1e-12)
        for i in range(5):
            cov[i][i]+=ridge

        #Solve the positive equal-risk
        x=[1.0/sqrt(cov[i][i]) for i in range(5)]
        budget = [RISK_BUDGET[label] for label in labels]  # was 1/5 each

        for _ in range(100):
            for i in range(5):
                a=cov[i][i]
                c= sum(
                    cov[i][j]*x[j]
                    for j in range(5) if j!=i
                )
                root = sqrt(c*c+4*a*budget[i])
                x[i]=(
                    2*budget[i]/(root+c)
                    if c>=0
                    else (root - c)/(2*a)
                )   
            
            contributions = [
                x[i]*sum(cov[i][j]*x[j] for j in range(5))
                for i in range(5)
            ]
            variance = sum(contributions)
            if max(
                abs(contributions[k]/variance - budget[k])
                for k in range(5)
            ) < 1e-6:
                break
        total = sum(x)
        weights = [value/total for value in x]
        variance = sum(
            weights[i]*cov[i][j]*weights[j]
            for i in range(5) for j in range(5)
        )
        annual_vol=sqrt(52*variance)
        scale=min(1.0,0.07/annual_vol)

        target={
            label: 100*scale*weights[i]
            for i, label in enumerate(labels)
        }
        return inside_bands(target)
            
            

    def _keep_the_mandate(self, target: dict[str, float]) -> dict[str, float]:
        """Module 22, keep the mandate: cut risk as the drawdown or the volatility grows.
        `self.drawdown` is counted for you from the first scored week.
        """
        # 1. drawdown share: 1 below S, 0 from L, linear in between
        s_dd = min(1.0, max(0.0,(DD_LIMIT-self.drawdown) / (DD_LIMIT-DD_START)))

        # 2. volatility share: cut when volatility is above its usual (median) level
        s_vol = 1.0
        vol = self.signals.get("volatility")
        if vol is not None and vol > 0:
            self.vol_history.append(vol)
            if len(self.vol_history) >= RECENT_WEEKS:
                usual = sorted(self.vol_history)[len(self.vol_history) // 2]
                s_vol = min(1.0, usual / vol)
        
        # volatility: shrink every class by the same factor, so each keeps its share the risk

        total = sum(target.values())
        factor = max(s_vol,70.0/total) if total > 0 else 1.0
        target = {k: v* factor for k, v in target.items()}
        self.share = min(s_dd,factor)
        self._execution_controls=(s_dd,factor)

        # drawdown: rotate toward the defensive book
        defensive = DEFENSIVE_NO_BONDS if self._bonds_move_with_equities() else DEFENSIVE
        return {k: s_dd * target[k]+(1-s_dd)*defensive[k] for k in target}

        #s = min(s_dd, s_vol)

        # 3. if Treasuries move with equities, park the cut in cash instead.
        #defensive = DEFENSIVE_NO_BONDS if self._bonds_move_with_equities() else DEFENSIVE
        #return {k: s*target[k]+(1-s)*defensive[k] for k in target}

    def _bonds_move_with_equities(self)-> bool:
        a = list(self.returns["A"])[-RECENT_WEEKS:]
        c = list(self.returns["C"])[-RECENT_WEEKS:]
        if len(a)<10 or len(a) != len(c):
            return False
        ma,mc = sum(a)/len(a), sum(c)/len(c)
        return sum((x-ma)*(y-mc)for x,y in zip(a,c)) > 0

    def _zscore(self, key: str, value: float) -> float | None:
        n, mean, m2 = self.signal_stats.get(key, (0, 0.0, 0.0))
        if n < 2 or m2 <= 0.0:
            return None
        return (value - mean) / (m2 / (n - 1)) ** 0.5
    
    def _record_signal(self, key: str, value: float) -> None:
        n, mean, m2 = self.signal_stats.get(key, (0, 0.0, 0.0))
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
        self.signal_stats[key] = [n, mean, m2]
    
    def _named_regime(self):
        def z(key: str) -> float | None:
            return self._zscore(key, self.smooth[key]) if key in self.smooth else None
        
        z_growth, z_infl, z_vol = z("growth"), z("inflation"), z("volatility")
        if z_growth is None or z_infl is None:
            return None
        if z_vol is None or GROWTH_RULE == "growth":
            contraction_score = -z_growth
        elif GROWTH_RULE == "or":
            contraction_score = max(-z_growth, z_vol)
        elif GROWTH_RULE == "sum":
            contraction_score = z_vol - z_growth
        else:
            contraction_score = min(-z_growth, z_vol)
        return NAMED[(contraction_score > 0.0, z_infl > 0.0)]
    
    def _base_rates(self):
        total = sum(self.regime_counts.values())
        return {regime: count / total for regime, count in self.regime_counts.items()}

    def _regime_view(self, state: StatePacket) -> dict[str, float]:
        """Module 23, read the macro regime: your probability of each, this week.

        Here even odds: no view.
        """
        self.z.update({k: v for k, v in state.quant.values.items() if v is not None})
        contracting = self.z.get("growth", 0.0) < G_CUT
        rising = self.z.get("inflation", 0.0) > 0.0
        if contracting:
            named = "stagflation" if rising else "recession"
        else:
            named = "inflation" if rising else "goldilocks"
        self.named_weeks[named] += 1
        total = sum(self.named_weeks.values())
        dist = min(abs(self.z.get("growth", 0.0) - G_CUT), abs(self.z.get("inflation", 0.0)))
        sure = 0.35 + 0.55 * min(1.0, dist / 1.5)
        return {k: sure * (k == named) + (1 - sure) * self.named_weeks[k] / total for k in REGIMES}
    

    def _regime_returns(self, state: StatePacket) -> dict[str, dict[str, float]]:
        lagged = state.quant.strategy_returns_lagged
        if all(lagged.get(label) is not None for label in STRATEGY_LABELS):
            self.all_weeks += 1
            for label in STRATEGY_LABELS:
                self.all_sum[label] += lagged[label]
            if self.last_view is not None:
                last_named = max(self.last_view, key=self.last_view.get)
                self.regime_weeks[last_named] += 1
                for label in STRATEGY_LABELS:
                    self.regime_sum[last_named][label] += lagged[label]
            
        if self.all_weeks == 0:
            return {k: dict.fromkeys(STRATEGY_LABELS, 0.0) for k in REGIMES}
        overall = {label: self.all_sum[label] / self.all_weeks for label in STRATEGY_LABELS}
        expected = {}
        for k in REGIMES:
            n = self.regime_weeks[k]
            w = n / (n + 26.0)
            expected[k] = {
                label: w * (self.regime_sum[k][label] / n if n else 0.0) + (1 - w) * overall[label]
                for label in STRATEGY_LABELS
            }
        return expected

    def _regime_odds(self) -> dict[str, float]:
        """Market-led regime odds: contraction from market volatility (moves first) and growth, inflation from its reading."""
        def sig(x: float) -> float:
            return 1.0 / (1.0 + exp(-max(-30.0, min(30.0, x))))
        z_vol = 0.0
        hist = list(self.vol_history)
        vol = self.signals.get("volatility")
        if vol is not None and len(hist) >= RECENT_WEEKS:
            m_ = sum(hist) / len(hist)
            sd = sqrt(sum((v - m_) ** 2 for v in hist) / (len(hist) - 1))
            if sd > 0:
                z_vol = (vol - m_) / sd
        z_growth = self.z.get("growth", 0.0) or 0.0
        z_infl = self.z.get("inflation", 0.0) or 0.0
        p_c = sig(1.5 * (0.6 * z_vol - 0.4 * z_growth) - 1.0)
        p_i = sig(1.5 * z_infl)
        return {
            "goldilocks": (1 - p_c) * (1 - p_i),
            "inflation": (1 - p_c) * p_i,
            "recession": p_c * (1 - p_i),
            "stagflation": p_c * p_i,
        }

    def _tilt(
        self,
        target: dict[str, float],
        view: dict[str, float],
        expected: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Module 24: lean on what the agent learnt (expected returns by regime, weighted by view), per unit of risk, sized by confidence."""
        try:
            if not expected or not view:
                return target
            vols, mom = {}, {}
            for label in STRATEGY_LABELS:
                r = list(self.returns[label])[-26:]
                if len(r) < 26:
                    return target
                mean = sum(r) / len(r)
                v = sqrt(sum((x - mean) ** 2 for x in r) / (len(r) - 1))
                if not isfinite(v) or v <= 0:
                    return target
                vols[label] = v
                mom[label] = (sum(r[-MOM_WEEKS:]) / MOM_WEEKS) / v
            # the view, nudged by market-led odds (markets move before the data)
            odds = self._regime_odds()
            p = {q: (1 - MARKET_SHARE) * max(0.0, view.get(q, 0.0)) + MARKET_SHARE * odds[q] for q in REGIMES}
            tot = sum(p.values()) or 1.0
            p = {q: p[q] / tot for q in REGIMES}
            conf = max(0.0, (max(p.values()) - 0.25) / 0.75)
            def centred(d):
                mu = sum(d.values()) / len(d)
                return {l: d[l] - mu for l in d}
            # learnt: expected return per unit of risk, centred, scaled by a fixed factor
            learnt = centred({l: sum(p[q] * expected[q][l] for q in REGIMES) / vols[l] for l in STRATEGY_LABELS})
            learnt = {l: max(-1.0, min(1.0, learnt[l] / LEARNT_REF)) for l in STRATEGY_LABELS}
            prior = centred({l: sum(p[q] * REGIME_PRIOR[q][l] for q in REGIMES) for l in STRATEGY_LABELS})
            mom = centred(mom)
            mgap = max(abs(x) for x in mom.values()) or 1.0
            mom = {l: mom[l] / mgap for l in STRATEGY_LABELS}
            delta = {
                l: TILT_UNITS * (conf * (W_LEARNT * learnt[l] + W_PRIOR * prior[l]) + W_MOM * mom[l])
                for l in STRATEGY_LABELS
            }
            return inside_bands({l: target[l] + delta[l] for l in STRATEGY_LABELS})
        except Exception:
            return target

    def _trade(self, target: dict[str, float]) -> dict[str, float]:
        target= inside_bands(target)
        controls=self._execution_controls
        # a real cut only: the controls fell CUT_MARGIN below where they stood at the last cut or since recovered,
        # not a hair from the noisy weekly volatility reading
        ref=getattr(self,"_cut_reference",controls)
        tightened=any(c<r-CUT_MARGIN for c,r in zip(controls,ref))
        self._cut_reference=controls if tightened else tuple(max(c,r) for c,r in zip(controls,ref))
        self._last_execution_controls=controls
        
        urgent = tightened or getattr(self, "_execution_pending",False)
        urgent = urgent or sum(self.book.values())< TOTAL_RISK_UNITS-CASH_MAX-1e-8
        self._execution_urgent=urgent
        if urgent:
            return target
      
        bands={"A":1.5, "B":3.0, "C":1.0, "D":2.0, "E":1.5}
        order={
            k:target[k]
            if abs(target[k]-self.book[k]) >bands[k]
            else self.book[k]
            for k in STRATEGY_LABELS
        }


        return inside_bands(order)



    def _remember(self, state: StatePacket) -> None:
        for label in STRATEGY_LABELS:
            value = state.quant.strategy_returns_lagged.get(label)
            if value is not None:
                self.returns[label].append(value)
        self.signals.update({k: v for k, v in state.quant.values.items() if v is not None})

    def _action(
        self,
        order: dict[str, float],
        view: dict[str, float],
        state: StatePacket,
        expected: dict[str, dict[str, float]] | None = None,
    ) -> Action:
        """The order as an Action: inside the bands and the capital, the rest in cash.

        Also within the turnover the world allows in a period
        (`state.constraints`), and with the view stated as scenarios: one
        branch per regime, its weight your probability (`view`), its returns
        your expected returns in that regime (`expected`, from `_regime_returns`).
        """
        units = inside_bands(order)
        invested = sum(units.values())
        if invested > TOTAL_RISK_UNITS:
            units = {label: value * TOTAL_RISK_UNITS / invested for label, value in units.items()}
        traded = sum(abs(units[label] - self.book[label]) for label in STRATEGY_LABELS)
        limit = state.constraints.max_turnover_per_period
        self._execution_pending = (
            getattr(self,"_execution_urgent",False) and traded >limit
        )
        if traded > limit:
            share = limit / traded
            units = {
                label: self.book[label] + (units[label] - self.book[label]) * share
                for label in STRATEGY_LABELS
            }
        self.book = units
        total = sum(view.values())
        branches = [
            ScenarioBranch(
                name=regime,
                weight=view[regime] / total,
                returns=expected[regime] if expected else dict.fromkeys(STRATEGY_LABELS, 0.0),
            )
            for regime in REGIMES
        ]
        escalate = CONFIDENCE < ESCALATION_THRESHOLD
        return Action(
            allocations=units,
            reserve=max(0.0, TOTAL_RISK_UNITS - sum(units.values())),
            escalate=escalate,
            escalate_reason=(
                f"stated confidence {CONFIDENCE:.2f} is below the registered "
                f"{ESCALATION_THRESHOLD:.2f}"
                if escalate
                else None
            ),
            uncertainty=Uncertainty(
                kind="scenario",
                payload=ScenarioUncertainty(scenarios=branches, confidence=CONFIDENCE),
                escalation_threshold=ESCALATION_THRESHOLD,
            ),
            evidence=[],
        )


def inside_bands(order: dict[str, float]) -> dict[str, float]:
    '''Closest allocation'''
    try:
        desired={label:float(order.get(label,0.0)) for label in BANDS}
    except (TypeError, ValueError):
        return dict(REFERENCE)
    if not all(isfinite(value) for value in desired.values()):
        return dict(REFERENCE)
    def shifted(shift: float) ->dict[str,float]:
        units = {
            label: min(high, max(low, desired[label]+shift))
            for label, (low,high) in BANDS.items()
        }
        equity=units["A"]+ units["B"]
        equity_target=min(EQUITY_BAND[1],max(EQUITY_BAND[0],equity))
        if equity != equity_target:
            low = max(BANDS["A"][0],equity_target - BANDS["B"][1])
            high=min(BANDS["A"][1],equity_target - BANDS["B"][0])
            a=(desired["A"]-desired["B"]+equity_target)/2
            units["A"]=min(high,max(low,a))
            units["B"]=equity_target-units["A"]
        return units

    units = shifted(0.0)
    invested = sum(units.values())
    minimum = TOTAL_RISK_UNITS-CASH_MAX
    if minimum <= invested <= TOTAL_RISK_UNITS:
        return units
    
    target=minimum if invested < minimum else TOTAL_RISK_UNITS
    low=min(BANDS[label][0]-desired[label] for label in BANDS) -1
    high= max(BANDS[label][1]-desired[label] for label in BANDS) + 1
    for _ in range(60):
        middle = (low+high)/2
        if sum(shifted(middle).values()) < target:
            low=middle
        else:
            high=middle
    return shifted(high if target ==minimum else low)
