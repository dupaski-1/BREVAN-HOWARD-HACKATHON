"""python backtest.py : compare the three agents on synthetic worlds."""
import sys
import numpy as np

from agents import ReferenceAgent, RegimeTiltAgent, RobustAgent
from sim import run

SCENARIOS = ["stable", "abrupt", "outage", "rare_stress", "stagflation_long"]
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 12)


def main():
    print(f"{'scenario':18}{'agent':13}{'ret%/yr':>8}{'maxDD%':>8}{'lost':>6}{'turn':>6}{'cost%':>7}{'esc':>6}{'confSD':>8}")
    tot = {}
    for sc in SCENARIOS:
        for cls in (ReferenceAgent, RegimeTiltAgent, RobustAgent):
            res = [run(cls(), sc, 1000 + s) for s in SEEDS]
            m = {k: np.mean([r[k] for r in res]) for k in res[0]}
            tot.setdefault(cls.name, []).append(m)
            print(f"{sc:18}{cls.name:13}{m['ret']:8.2f}{max(r['maxdd'] for r in res):8.1f}"
                  f"{sum(r['lost'] for r in res):>4}/{len(res):<2}{m['turnover']:6.2f}{m['costs']:7.3f}{m['esc']:6.2f}{m['conf_sd']:8.3f}")
    print("\nmean over scenarios")
    for n, ms in tot.items():
        print(f"{n:13} ret {np.mean([m['ret'] for m in ms]):.2f}  turnover {np.mean([m['turnover'] for m in ms]):.2f}"
              f"  lost-rate {np.mean([m['lost'] for m in ms]):.2f}  escalation {np.mean([m['esc'] for m in ms]):.2f}")


if __name__ == "__main__":
    main()
