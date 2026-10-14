import json
import traci, sumolib

net = sumolib.net.readNet("sim/osm.net.xml.gz")
ROUTE = json.load(open("results/route.json"))

traci.start(["sumo", "-c", "sim/osm.sumocfg", "--seed", "1", "--no-warnings"])
for _ in range(100):
    traci.simulationStep()
traci.route.add("amb_route", ROUTE)
traci.vehicle.add("ambulance_1", "amb_route")
t_add = traci.simulation.getTime()
t_in = None
t_end = t_add
slow = {}
loss = 0.0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    now = traci.simulation.getTime()
    t_end = now
    if "ambulance_1" in traci.simulation.getArrivedIDList():
        break
    if "ambulance_1" not in traci.vehicle.getIDList():
        continue
    if t_in is None:
        t_in = now
    loss = traci.vehicle.getTimeLoss("ambulance_1")
    if traci.vehicle.getSpeed("ambulance_1") < 1.0:
        road = traci.vehicle.getRoadID("ambulance_1")
        slow[road] = slow.get(road, 0) + 1
traci.close()

print("total trip (s):", t_end - t_add)
print("waited to enter the road (s):", (t_in - t_add) if t_in else "never entered")
print("time loss vs free-flow (s):", round(loss))
print("seconds below 1 m/s, by road:")
for road, secs in sorted(slow.items(), key=lambda x: -x[1])[:8]:
    if road.startswith(":"):
        kind = "inside a junction"
    elif net.getEdge(road).getToNode().getType().startswith("traffic_light"):
        kind = "road ending at a signal"
    else:
        kind = "road with no signal at its end"
    print("  ", road, secs, "s -", kind)