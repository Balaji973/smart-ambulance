import torch, pandas as pd
from env import AmbulanceEnv
from qmix import AgentNet, act

SEEDS = range(11, 41)   # 30 fresh seeds (training used seeds 1000+)
env = AmbulanceEnv()

def load(path):
    a = AgentNet()
    a.load_state_dict(torch.load(path)["agent"])
    a.eval()
    return a

policies = {
    "fixed": lambda obs: [1] * env.n,
    "always_preempt": lambda obs: [0] * env.n,
}
for name, path in (("qmix_v3", "rl/qmix_v3.pt"), ("qmix_v4", "rl/qmix_v4.pt")):
    ag = load(path)
    policies[name] = (lambda ag: (lambda obs: act(ag, obs, 0.0)))(ag)

rows = []
for name, pol in policies.items():
    for seed in SEEDS:
        obs, state = env.reset(seed)
        done = False
        while not done:
            obs, state, r, done, info = env.step(pol(obs))
        rows.append([name, seed, info["ambulance_s"], info["stopped_s"]])
env.close()

d = pd.DataFrame(rows, columns=["method", "seed", "ambulance_s", "stopped_vehicle_s"])
d["stopped_per_s"] = d.stopped_vehicle_s / d.ambulance_s
d.to_csv("results/eval_30seeds.csv", index=False)

print("\nRESULTS OVER 30 FRESH SEEDS")
print(d.groupby("method")[["ambulance_s", "stopped_vehicle_s", "stopped_per_s"]].mean().round(2))
p = d.pivot(index="seed", columns="method", values="ambulance_s")
print("\nSEEDS (OF 30) WHERE EACH METHOD BEATS FIXED TIMING:")
for m in ["always_preempt", "qmix_v3", "qmix_v4"]:
    print(m, int((p[m] < p["fixed"]).sum()))