import csv, json, os, statistics
import traci, sumolib

CFG = "sim/osm.sumocfg"
NET = "sim/osm.net.xml.gz"
ROUTE_FILE = "results/route_v2.json"   # change to results/route.json for the first route
SCALE = 5                              # traffic multiplier
SEEDS = range(1, 11)
ACT_DIST = 200                         # metres before a junction when preemption starts
HOLD_S = 20                            # seconds the ambulance's green phase is held

net = sumolib.net.readNet(NET)
ROUTE = json.load(open(ROUTE_FILE))

def build_plan():
    """For each signal the ambulance drives through: (tls_id, link indices it uses, route distance to the stop line)."""
    cum, d = {}, 0.0
    for eid in ROUTE:
        d += net.getEdge(eid).getLength()
        cum[eid] = d
    plan = []
    for tls in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls)
        idxs, hit = [], None
        for idx, group in enumerate(links):
            for (inl, outl, via) in group:
                ie = inl.rsplit("_", 1)[0]
                oe = outl.rsplit("_", 1)[0]
                if ie in cum and oe in ROUTE and ROUTE.index(oe) == ROUTE.index(ie) + 1:
                    idxs.append(idx)
                    hit = ie
        if hit:
            plan.append((tls, idxs, cum[hit]))
    return plan

def green_phase(tls, idxs):
    """Index of a normal phase in this signal's program that is green for all the ambulance's links."""
    current = traci.trafficlight.getProgram(tls)
    for logic in traci.trafficlight.getAllProgramLogics(tls):
        if logic.programID == current:
            for i, ph in enumerate(logic.phases):
                if all(ph.state[j] in "Gg" for j in idxs):
                    return i
    return None

def run(seed, mode):
    traci.start(["sumo", "-c", CFG, "--seed", str(seed), "--no-warnings", "--scale", str(SCALE)])
    for _ in range(100):
        traci.simulationStep()
    traci.route.add("amb_route", ROUTE)
    traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance")
    traci.vehicletype.setColor("ambulance", (255, 0, 0, 255))
    traci.vehicle.add("ambulance_1", "amb_route", typeID="ambulance")

    plan = build_plan() if mode == "greenwave" else []
    if mode == "greenwave" and seed == SEEDS[0]:
        print("signals controlled on route:", len(plan))
    last = {}
    start = traci.simulation.getTime()
    stopped_s = 0
    travel = None
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        now = traci.simulation.getTime()
        ids = traci.vehicle.getIDList()
        if "ambulance_1" in ids:
            pos = traci.vehicle.getDistance("ambulance_1")
            for tls, idxs, d in plan:
                gap = d - pos
                if 0 <= gap <= ACT_DIST and now - last.get(tls, -999) >= HOLD_S:
                    last[tls] = now
                    st = traci.trafficlight.getRedYellowGreenState(tls)
                    if not all(st[j] in "Gg" for j in idxs):
                        p = green_phase(tls, idxs)
                        if p is not None:
                            traci.trafficlight.setPhase(tls, p)
                            traci.trafficlight.setPhaseDuration(tls, HOLD_S)
        stopped_s += sum(1 for v in ids
                         if v != "ambulance_1" and traci.vehicle.getSpeed(v) < 0.1)
        if "ambulance_1" in traci.simulation.getArrivedIDList():
            travel = now - start
            break
        if now - start > 1200:
            break
    traci.close()
    return travel, stopped_s

rows = []
for mode in ("fixed", "greenwave"):
    for seed in SEEDS:
        t, s = run(seed, mode)
        print(mode, "seed", seed, "| ambulance (s):", t, "| stopped-vehicle-seconds:", s)
        rows.append([mode, seed, t, s])

tag = os.path.splitext(os.path.basename(ROUTE_FILE))[0]
with open(f"results/baselines_scale{SCALE}_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "seed", "ambulance_s", "stopped_vehicle_s"])
    w.writerows(rows)

print("\nSUMMARY (route:", tag, "| traffic x" + str(SCALE) + "| seeds:", len(SEEDS), ")")
for mode in ("fixed", "greenwave"):
    a = [r[2] for r in rows if r[0] == mode and r[2] is not None]
    c = [r[3] for r in rows if r[0] == mode]
    failed = sum(1 for r in rows if r[0] == mode and r[2] is None)
    if a:
        print(mode, "| ambulance:", round(statistics.mean(a), 1), "s | cross-traffic stopped:",
              round(statistics.mean(c)), "veh-s | runs that did not finish:", failed)