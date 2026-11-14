import sys
import numpy as np
import libsumo as traci
import env as E

ROUTE = sys.argv[1] if len(sys.argv) > 1 else "results/route_sim.json"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
SEEDS = range(101, 111)              # screening seeds, not used for any result

def near_red(veh):
    nxt = traci.vehicle.getNextTLS(veh)
    if not nxt:
        return False
    n = nxt[0]
    dist, state = (n.dist, n.state) if hasattr(n, "dist") else (n[2], n[3])
    return dist < 250 and state in "rRyYu"

env = E.AmbulanceEnv(route_file=ROUTE, scale=SCALE, decision_s=1)
for name, a in (("fixed", 1), ("always_preempt", 0)):
    trips, reds, others = [], [], []
    for seed in SEEDS:
        env.reset(seed)
        done, red, other = False, 0, 0
        while not done:
            obs, state, r, done, info = env.step([a] * env.n)
            if E.AMB in traci.vehicle.getIDList() and traci.vehicle.getSpeed(E.AMB) < 1.0:
                if near_red(E.AMB):
                    red += 1
                else:
                    other += 1
        if info["ambulance_s"] is not None:
            trips.append(info["ambulance_s"])
            reds.append(red)
            others.append(other)
    print("DIAG %s | x%s | finished %d/10 | trip %.0f s | waiting at red %.0f s | stopped elsewhere %.0f s"
          % (name, SCALE, len(trips), np.mean(trips), np.mean(reds), np.mean(others)))
env.close()