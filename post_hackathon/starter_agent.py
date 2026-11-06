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
        """Module 21, budget risk, not capital: the book's core.

        Here the reference, whose equities carry most of the book's risk.
        """
        return dict(REFERENCE)

    def _keep_the_mandate(self, target: dict[str, float]) -> dict[str, float]:
        """Module 22, keep the mandate: cut risk as the drawdown or the volatility grows.

        `self.drawdown` is counted for you from the first scored week. Here nothing is cut.
        """
        return target

    def _regime_view(self, state: StatePacket) -> dict[str, float]:
        """Module 23, read the macro regime: your probability of each, this week.

        Here even odds: no view.
        """
        return {regime: 1.0 / len(REGIMES) for regime in REGIMES}

    def _regime_returns(self, state: StatePacket) -> dict[str, dict[str, float]]:
        """Module 23, read the macro regime: your expected returns, per regime and class.

        For each regime, the weekly return you expect of each class there,
        learnt from your own weeks: `state.quant.strategy_returns_lagged` is
        last week's return, so it belongs to the regime of last week's view,
        `self.last_view` (None the first week). Each regime's go in its
        branch's `returns`, which the module checks. Here zero: nothing learnt.
        """
        return {regime: dict.fromkeys(STRATEGY_LABELS, 0.0) for regime in REGIMES}

    def _tilt(
        self,
        target: dict[str, float],
        view: dict[str, float],
        expected: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Module 24, tilt by regime, as far as you trust the view.

        Your expected return of each class under your view: each regime's
        `expected` returns (module 23) weighted by its probability in `view`.
        Lean the target towards the classes above their average. Here no tilt.
        """
        return target

    def _trade(self, target: dict[str, float]) -> dict[str, float]:
        """Module 25, implement under costs: trade only when it pays.

        Here straight to the target, every week. `self.book` is what you hold.
        """
        return dict(target)

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
    """Each class clipped to its band.

    The platform moves an order still outside the rules (the equities
    together, cash) inside them before it trades, and counts the weeks it had
    to: the final portfolio's checklist reads that count.
    """
    return {
        label: min(max(float(order.get(label, 0.0)), low), high)
        for label, (low, high) in BANDS.items()
    }
