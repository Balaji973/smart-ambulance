import numpy as np
import libsumo as traci
from env import AmbulanceEnv, AMB

def next_tls(veh):
    nxt = traci.vehicle.getNextTLS(veh)
    if not nxt:
        return None
    n = nxt[0]
    if hasattr(n, "dist"):
        return n.dist, n.state
    return n[2], n[3]

def measure(scale, seeds):
    env = AmbulanceEnv(scale=scale, decision_s=1)
    rows = []
    for seed in seeds:
        env.reset(seed)
        done, red, green_q, other = False, 0, 0, 0
        while not done:
            obs, state, r, done, info = env.step([1] * env.n)
            if AMB in traci.vehicle.getIDList() and traci.vehicle.getSpeed(AMB) < 1.0:
                t = next_tls(AMB)
                if t is not None and t[0] < 60:
                    if t[1] in "rRyYu":
                        red += 1
                    else:
                        green_q += 1
                else:
                    other += 1
        trip = info["ambulance_s"] or 0
        rows.append([trip, red, green_q, other])
    env.close()
    return np.mean(rows, axis=0)

for scale in (1, 3, 5):
    trip, red, gq, oth = measure(scale, range(101, 111))
    print("SCALE", scale, "| trip %.0f s | stopped at red %.0f s (%.1f%%) | stopped at green signal (queue) %.0f s | stopped elsewhere %.0f s"
          % (trip, red, 100 * red / trip, gq, oth))