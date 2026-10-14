import pandas as pd
from env import AmbulanceEnv

env = AmbulanceEnv()
rows = []
for mode, a in (("always_hold", 1), ("always_preempt", 0)):
    for seed in range(1, 11):
        obs, state = env.reset(seed)
        done = False
        while not done:
            obs, state, r, done, info = env.step([a] * env.n)
        rows.append([mode, seed, info["ambulance_s"], info["stopped_s"]])
env.close()

q = pd.read_csv("results/qmix_eval.csv")
for _, r in q.iterrows():
    rows.append(["qmix", int(r["seed"]), r["ambulance_s"], r["stopped_vehicle_s"]])

gw = pd.read_csv("results/baselines_scale5_route_v2.csv")
for _, r in gw[gw["mode"] == "greenwave"].iterrows():
    rows.append(["greenwave_baseline", int(r["seed"]), r["ambulance_s"], r["stopped_vehicle_s"]])

d = pd.DataFrame(rows, columns=["method", "seed", "ambulance_s", "stopped_vehicle_s"])
d["stopped_per_s"] = d.stopped_vehicle_s / d.ambulance_s
d.to_csv("results/final_comparison.csv", index=False)

print("\nAMBULANCE TIME (s) PER SEED")
print(d.pivot(index="seed", columns="method", values="ambulance_s").round(0))
print("\nAVERAGES")
print(d.groupby("method")[["ambulance_s", "stopped_vehicle_s", "stopped_per_s"]].mean().round(2))