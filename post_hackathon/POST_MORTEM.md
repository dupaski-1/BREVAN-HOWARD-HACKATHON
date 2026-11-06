# Uncertain World Simulator: post mortem for team Neyman

Source: every tab of the cockpit, read on 6 Nov 2026 after the deadline. The frozen agent.py was only partly readable, so statements about its code are inferred from its behaviour.

## Where you ended up

At the 17:21 leaderboard snapshot (frozen at 17:00) Neyman ranked 2 of 15 with +3.97 % a year, 0 of 4 mandates lost and 3 of 5 modules. The final agent (last scored around 18:00 UTC) looks different: +3.53 % a year, 3 of 4 mandates kept, 4 of 5 modules validated (20 of 25 module points).

The leaderboard rule matters here. A team that loses one mandate ranks behind every team that kept all four. The last scored agent lost the 2018 to 2026 window (week 211, 21 Oct 2022, worst drawdown 21.4 %). If that agent is the one judged, the rank 2 position was very likely given up. Verify against the final ranking, which the portal says is decided after the deadline.

## What went wrong, in order of cost

1. The 2022 loss. Your book held 45 % in US 10 year Treasuries during the fall that lost the mandate, and that class fell 20.0 % over cash, costing 9.6 % of capital. Developed equity (19 %) and credit (8 %) fell too, so cash at 22 % was the only part that helped. Bonds and equities fell together because inflation was the driver. The regime tilt leaned into duration when the regime called for the opposite.
2. Constant confidence. The episode tab states plainly that the agent reported the same confidence every period, so responsiveness and difficulty awareness score zero. Escalations were 0. The 15 point uncertainty block was probably the cheapest place to gain points.
3. Module 5 missed by 0.1 point of return. Turnover (1.60x), costs (0.100 %) and mandate checks passed, and the return after costs was 3.6 % against 3.7 % needed. Development world 1 passed at 3.75 %, so the margin is thin and the hidden worlds decided it. Cash at 22 % in a stress is a likely drag.
4. A close call in development world 5: drawdown 19.2 % against a limit of 20 %.
5. Part 1 ended at 12 of 13 steps validated.

## What went right

Modules 1 to 4 validated (16:17, 16:21, 17:18, 18:39). Real mandate returns were positive in all four windows, and turnover of 1.44x and costs of 0.089 % a year were well inside the limits. The reference book lost 3 of 4 mandates, with a worst drawdown of 36.8 %, so your risk control worked in three windows out of four.

## What the new agents change

RobustAgent differs from the frozen one in four ways. It treats a stale or missing signal as a reason to hold less of a view, never as a calm reading. It sizes the tilt by confidence, so it falls back to the reference book when unsure. It moves toward a defensive book as the drawdown grows from 6 % to 14 %, and that book keeps credit and oil rather than assuming Treasuries are safe. Its confidence drops sharply when data goes dark and it escalates below the registered 0.40 threshold, or when it is within 5 points of the mandate limit.

## Offline results (synthetic world, 20 seeds, 5 scenarios)

| agent | return % a year | mandates lost | turnover | escalation rate |
| --- | --- | --- | --- | --- |
| reference | 0.91 | 51 % | 0.00 | 0 % |
| regime_tilt | 1.26 | 43 % | 0.19 | 0 % |
| robust | 1.24 | 25 % | 1.07 | 21 % |

Read this carefully. The simulator is my own approximation, so these numbers say the guard and the confidence logic behave as designed, and say nothing about the portal's score. The returns are lower than the real ones because the synthetic equity premium is small. Escalation at 21 % is higher than ideal and comes mostly from deep drawdowns in this harsh world. In a total data outage confidence fell to about 0.02 and recovered afterwards.

## Next steps if the portal reopens or you run a similar contest

1. Calibrate the tilt tables on the portal's "By regime" and "Returns" sheets, especially the stagflation row.
2. Backtest on the 2018 to 2026 data first, since it is the window that decided the result.
3. Check the escalation rate on the 10 development scenarios, then tune the 0.40 threshold.
4. Confirm the field names in portal_adapter.py against the starter file.
