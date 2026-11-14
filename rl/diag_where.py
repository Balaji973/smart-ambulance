import sys
from collections import defaultdict
import libsumo as traci
import env as E

ROUTE = sys.argv[1] if len(sys.argv) > 1 else "results/route_sim.json"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
SEEDS = range(101, 111)              # screening seeds, not used for any result

def next_red(veh):
    nxt = traci.vehicle.getNextTLS(veh)
    if not nxt:
        return None
    n = nxt[0]
    tid, dist, state = (n.id, n.dist, n.state) if hasattr(n, "id") else (n[0], n[2], n[3])
    if dist < 250 and state in "rRyYu":
        return tid
    return None

env = E.AmbulanceEnv(route_file=ROUTE, scale=SCALE, decision_s=1)
for name, a in (("fixed", 1), ("always_preempt", 0)):
    per = defaultdict(float)
    finished = 0
    controlled = set()
    for seed in SEEDS:
        env.reset(seed)
        controlled = {x["tls"] for x in env.agents}
        done = False
        local = defaultdict(int)
        while not done:
            obs, state, r, done, info = env.step([a] * env.n)
            if E.AMB in traci.vehicle.getIDList() and traci.vehicle.getSpeed(E.AMB) < 1.0:
                tid = next_red(E.AMB)
                if tid is not None:
                    local[tid] += 1
        if info["ambulance_s"] is not None:
            finished += 1
            for k, v in local.items():
                per[k] += v
    for k in sorted(per, key=lambda k: -per[k]):
        print("WHERE %s | signal %s | %s | waiting at red %.1f s per run"
              % (name, k, "CONTROLLED" if k in controlled else "NOT controlled", per[k] / max(finished, 1)))
env.close()