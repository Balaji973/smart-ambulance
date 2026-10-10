import csv, json, random
import traci, sumolib

CFG = "sim/osm.sumocfg"
NET = "sim/osm.net.xml.gz"
net = sumolib.net.readNet(NET)

def pick_route():
    random.seed(42)
    edges = [e.getID() for e in net.getEdges()
             if e.allows("passenger") and not e.getID().startswith(":")]
    best, best_len = None, 0
    for _ in range(300):
        a, b = random.sample(edges, 2)
        r = traci.simulation.findRoute(a, b)
        if not r.edges:
            continue
        n = sum(1 for e in r.edges
                if net.getEdge(e).getToNode().getType().startswith("traffic_light"))
        if 3 <= n <= 5 and r.length > best_len:
            best, best_len = r.edges, r.length
    return best

def run(seed, edges=None):
    traci.start(["sumo", "-c", CFG, "--seed", str(seed), "--no-warnings"])
    for _ in range(100):
        traci.simulationStep()          # let traffic build up first
    if edges is None:
        edges = pick_route()
        if edges is None:
            traci.close()
            raise SystemExit("No route with 3-5 signals found. Try a different map area.")
    traci.route.add("amb_route", edges)
    traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance")
    traci.vehicletype.setColor("ambulance", (255, 0, 0, 255))
    traci.vehicle.add("ambulance_1", "amb_route", typeID="ambulance")
    start = traci.simulation.getTime()
    travel = None
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if "ambulance_1" in traci.simulation.getArrivedIDList():
            travel = traci.simulation.getTime() - start
            break
    traci.close()
    return travel, edges

rows = []
route_edges = None
for seed in range(1, 6):
    t, route_edges = run(seed, route_edges)
    print("seed", seed, "ambulance travel time (s):", t)
    rows.append([seed, t])

with open("results/route.json", "w") as f:
    json.dump(route_edges, f)

with open("results/baseline_fixed.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "travel_time_s"])
    w.writerows(rows)