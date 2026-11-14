import sys, json, random
import numpy as np
import libsumo as traci
import sumolib
import env as E

folder = sys.argv[1]                                   # e.g. sim or sim_south
SCALES = [float(x) for x in sys.argv[2:]] or [3.0, 5.0]
E.CFG = folder + "/osm.sumocfg"
E.NET = folder + "/osm.net.xml.gz"
SEEDS = range(101, 111)                                # screening seeds only

net = sumolib.net.readNet(E.NET)
n_tls = len([x for x in net.getNodes() if x.getType().startswith("traffic_light")])

def best_route():
    random.seed(7)
    edges = [e for e in net.getEdges() if e.allows("passenger")]
    best, key = None, (0, 0)
    for _ in range(6000):
        a, b = random.sample(edges, 2)
        path, cost = net.getShortestPath(a, b, vClass="passenger")
        if not path:
            continue
        length = sum(e.getLength() for e in path)
        if not 2500 <= length <= 5000:
            continue
        crossed = sum(1 for e in path[:-1] if e.getToNode().getType().startswith("traffic_light"))
        if (crossed, length) > key:
            best, key = path, (crossed, length)
    return best, key

def near_red(veh):
    nxt = traci.vehicle.getNextTLS(veh)
    if not nxt:
        return False
    n = nxt[0]
    dist, state = (n.dist, n.state) if hasattr(n, "dist") else (n[2], n[3])
    return dist < 250 and state in "rRyYu"

def measure(route_file, scale):
    env = E.AmbulanceEnv(route_file=route_file, scale=scale, decision_s=1)
    trips, reds = [], []
    for seed in SEEDS:
        env.reset(seed)
        done, red = False, 0
        while not done:
            obs, state, r, done, info = env.step([1] * env.n)
            if (E.AMB in traci.vehicle.getIDList() and traci.vehicle.getSpeed(E.AMB) < 1.0
                    and near_red(E.AMB)):
                red += 1
        ok = info["ambulance_s"] is not None
        trips.append(info["ambulance_s"] if ok else float("nan"))
        reds.append(red if ok else float("nan"))
    n_agents = env.n
    env.close()
    return n_agents, np.array(trips), np.array(reds)

route, key = best_route()
if route is None:
    raise SystemExit("NO ROUTE of 2.5-5 km found in " + folder)
rf = "results/route_" + folder.replace("/", "_").replace("\\", "_") + ".json"
json.dump([e.getID() for e in route], open(rf, "w"))
for scale in SCALES:
    n_agents, trips, reds = measure(rf, scale)
    finished = int(np.sum(~np.isnan(trips)))
    head = "SCREEN %s | signals on map %d | route %.0f m | signals crossed %d | agents %d | x%s | finished %d/10" % (
        folder, n_tls, key[1], key[0], n_agents, scale, finished)
    if finished == 0:
        print(head + " | no finished runs")
    else:
        share = 100 * np.nansum(reds) / np.nansum(trips)
        print(head + " | trip %.0f s | waiting at red within 250 m %.0f s = %.1f%% of trip"
              % (np.nanmean(trips), np.nanmean(reds), share))