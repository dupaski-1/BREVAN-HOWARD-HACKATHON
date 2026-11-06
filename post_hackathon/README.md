# UWS agents

Three agents for the Uncertain World Simulator, plus an offline test world.

    python backtest.py 20      # compare the agents on 11 scenarios: the portal's 10 episodes plus a long stagflation
    python test_agents.py      # legality and behaviour checks

Files: core.py (mandate, projection, z-scores), agents.py (the 3 agents),
sim.py (synthetic world, not the organisers' one), backtest.py,
portal_adapter.py (unverified glue to the portal), POST_MORTEM.md.

Before relying on the numbers: the tilt tables in agents.py are macro priors.
Calibrate them on the "By regime" and "Returns" sheets of the portal's Data
export, and check the field names in portal_adapter.py.

NOTE: NONE OF THIS WAS SCORED
