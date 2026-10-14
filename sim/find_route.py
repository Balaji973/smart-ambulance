import json, random
import sumolib

net = sumolib.net.readNet("sim/osm.net.xml.gz")
edges = [e for e in net.getEdges() if e.allows("passenger")]
random.seed(7)

def crossed(path):
    # signals the ambulance actually drives through (the last edge's end doesn't count)
    return sum(1 for e in path[:-1] if e.getToNode().getType().startswith("traffic_light"))

best, best_key = None, (0, 0)
for _ in range(4000):
    a, b = random.sample(edges, 2)
    path, cost = net.getShortestPath(a, b, vClass="passenger")
    if not path:
        continue
    length = sum(e.getLength() for e in path)
    if length < 1500:
        continue
    key = (crossed(path), length)
    if key > best_key:
        best, best_key = path, key

if best is None:
    raise SystemExit("No route found")
ids = [e.getID() for e in best]
print("signals crossed:", best_key[0], "| length m:", round(best_key[1]), "| edges:", len(ids))
json.dump(ids, open("results/route_v2.json", "w"))