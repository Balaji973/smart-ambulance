import csv, random, sys
import numpy as np
import torch, torch.nn as nn
from env2 import AmbulanceEnv2 as AmbulanceEnv

N, OBS, NA = 3, 9, 2
STATE = N * OBS
BEST, LAST = "rl/qmix2_best.pt", "rl/qmix2_last.pt"
VAL_SEEDS = range(41, 61)      # validation seeds: only used to choose the checkpoint
TEST_SEEDS = range(71, 101)    # final test seeds: run the test once

def with_id(obs, n=N):
    eye = torch.eye(n).unsqueeze(0).expand(obs.shape[0], n, n)
    return torch.cat([obs, eye], dim=-1)

class AgentNet(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS + N, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, NA))
    def forward(self, x):
        return self.net(x)

class Mixer(nn.Module):
    def __init__(self, emb=32):
        super().__init__()
        self.emb = emb
        self.w1 = nn.Linear(STATE, N * emb)
        self.b1 = nn.Linear(STATE, emb)
        self.w2 = nn.Linear(STATE, emb)
        self.b2 = nn.Sequential(nn.Linear(STATE, emb), nn.ReLU(), nn.Linear(emb, 1))
    def forward(self, qs, state):
        B = qs.shape[0]
        w1 = torch.abs(self.w1(state)).view(B, N, self.emb)
        b1 = self.b1(state).view(B, 1, self.emb)
        h = nn.functional.elu(torch.bmm(qs.view(B, 1, N), w1) + b1)
        w2 = torch.abs(self.w2(state)).view(B, self.emb, 1)
        b2 = self.b2(state).view(B, 1, 1)
        return (torch.bmm(h, w2) + b2).view(B)

class Buffer:
    def __init__(self, cap=40000):
        self.cap, self.data, self.i = cap, [], 0
    def add(self, tr):
        if len(self.data) < self.cap:
            self.data.append(tr)
        else:
            self.data[self.i] = tr
        self.i = (self.i + 1) % self.cap
    def sample(self, B):
        o, s, a, r, o2, s2, d = zip(*random.sample(self.data, B))
        f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
        return f(o), f(s), torch.tensor(np.array(a), dtype=torch.long), f(r), f(o2), f(s2), f(d)

def act(agent, obs, eps):
    with torch.no_grad():
        q = agent(with_id(torch.tensor(obs).unsqueeze(0)))[0]
    greedy = q.argmax(dim=-1).tolist()
    return [random.randrange(NA) if random.random() < eps else g for g in greedy]

def update(agent, mixer, t_agent, t_mixer, opt, buf, B=64, gamma=0.99):
    o, s, a, r, o2, s2, d = buf.sample(B)
    q_taken = agent(with_id(o)).gather(2, a.unsqueeze(-1)).squeeze(-1)
    qtot = mixer(q_taken, s)
    with torch.no_grad():
        best = agent(with_id(o2)).argmax(dim=-1, keepdim=True)
        q2 = t_agent(with_id(o2)).gather(2, best).squeeze(-1)
        target = r + gamma * (1 - d) * t_mixer(q2, s2)
    loss = ((qtot - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(agent.parameters()) + list(mixer.parameters()), 10)
    opt.step()
    return loss.item()

def run_episode(env, seed, policy):
    obs, state = env.reset(seed)
    done = False
    while not done:
        obs, state, r, done, info = env.step(policy(obs))
    amb = info["ambulance_s"] if info["ambulance_s"] is not None else float(env.max_time)
    return amb, info["stopped_s"]

def mean_result(env, seeds, policy):
    res = [run_episode(env, s, policy) for s in seeds]
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))

def train(episodes):
    env = AmbulanceEnv()
    agent, mixer = AgentNet(), Mixer()
    t_agent, t_mixer = AgentNet(), Mixer()
    t_agent.load_state_dict(agent.state_dict())
    t_mixer.load_state_dict(mixer.state_dict())
    opt = torch.optim.Adam(list(agent.parameters()) + list(mixer.parameters()), lr=5e-4)
    buf = Buffer()

    for name, a in (("fixed", 1), ("always_preempt", 0)):
        amb, stp = mean_result(env, VAL_SEEDS, lambda o, a=a: [a] * env.n)
        print("REF validation", name, "| ambulance", round(amb, 1), "| stopped", round(stp), flush=True)

    f = open("results/train_log2.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "seed", "epsilon", "ambulance_s", "stopped_veh_s", "return", "loss"])
    best = float("inf")
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        obs, state = env.reset(1000 + ep)
        assert env.n == N, "expected %d agents, got %d" % (N, env.n)
        done, total, loss = False, 0.0, float("nan")
        while not done:
            acts = act(agent, obs, eps)
            obs2, state2, r, done, info = env.step(acts)
            buf.add((obs, state, acts, r * 0.1, obs2, state2, float(done)))
            obs, state, total = obs2, state2, total + r
        if len(buf.data) >= 500:
            for _ in range(200):
                loss = update(agent, mixer, t_agent, t_mixer, opt, buf)
            if ep % 2 == 1:
                t_agent.load_state_dict(agent.state_dict())
                t_mixer.load_state_dict(mixer.state_dict())
        w.writerow([ep, 1000 + ep, round(eps, 3), info["ambulance_s"], info["stopped_s"], round(total, 2), loss])
        f.flush()
        if ep % 50 == 49:
            v_amb, v_stop = mean_result(env, VAL_SEEDS, lambda o: act(agent, o, 0.0))
            mark = ""
            if v_amb < best:
                best = v_amb
                torch.save({"agent": agent.state_dict(), "mixer": mixer.state_dict()}, BEST)
                mark = "  <- new best"
            print("VAL after ep", ep + 1, "| ambulance", round(v_amb, 1), "| stopped", round(v_stop), mark, flush=True)
    torch.save({"agent": agent.state_dict(), "mixer": mixer.state_dict()}, LAST)
    env.close()

def test():
    import pandas as pd
    env = AmbulanceEnv()
    agent = AgentNet()
    agent.load_state_dict(torch.load(BEST)["agent"])
    agent.eval()
    pols = {"fixed": lambda o: [1] * env.n,
            "always_preempt": lambda o: [0] * env.n,
            "qmix2_best": lambda o: act(agent, o, 0.0)}
    rows = []
    for name, pol in pols.items():
        for s in TEST_SEEDS:
            amb, stp = run_episode(env, s, pol)
            rows.append([name, s, amb, stp])
    env.close()
    d = pd.DataFrame(rows, columns=["method", "seed", "ambulance_s", "stopped_vehicle_s"])
    d["stopped_per_s"] = d.stopped_vehicle_s / d.ambulance_s
    d.to_csv("results/test_qmix2.csv", index=False)
    print("\nFINAL TEST, 30 SEEDS (71-100)")
    print(d.groupby("method")[["ambulance_s", "stopped_vehicle_s", "stopped_per_s"]].mean().round(2))
    p = d.pivot(index="seed", columns="method", values="ambulance_s")
    for m in ["always_preempt", "qmix2_best"]:
        print(m, "beats fixed on", int((p[m] < p["fixed"]).sum()), "of 30 seeds")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        train(int(sys.argv[1]) if len(sys.argv) > 1 else 3)