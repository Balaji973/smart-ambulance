import csv, random, sys
import numpy as np
import torch, torch.nn as nn
from env import AmbulanceEnv

N, OBS, NA = 3, 6, 2          # agents, observation size, actions
STATE = N * OBS
CKPT = "rl/qmix.pt"

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
    def __init__(self, cap=30000):
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

def train(episodes):
    env = AmbulanceEnv()
    agent, mixer = AgentNet(), Mixer()
    t_agent, t_mixer = AgentNet(), Mixer()
    t_agent.load_state_dict(agent.state_dict())
    t_mixer.load_state_dict(mixer.state_dict())
    opt = torch.optim.Adam(list(agent.parameters()) + list(mixer.parameters()), lr=5e-4)
    buf = Buffer()
    f = open("results/train_log.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "seed", "epsilon", "ambulance_s", "stopped_veh_s", "return", "loss"])
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        seed = 1000 + ep                       # training seeds never overlap the evaluation seeds 1-10
        obs, state = env.reset(seed)
        assert env.n == N, "expected 3 agents, got %d" % env.n
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
        print("ep", ep, "| eps", round(eps, 2), "| ambulance_s", info["ambulance_s"],
              "| stopped", info["stopped_s"], "| return", round(total, 1), "| loss", round(loss, 4))
        w.writerow([ep, seed, round(eps, 3), info["ambulance_s"], info["stopped_s"], round(total, 2), loss])
        f.flush()
        if ep % 10 == 9 or ep == episodes - 1:
            torch.save({"agent": agent.state_dict(), "mixer": mixer.state_dict()}, CKPT)
    env.close()

def evaluate():
    env = AmbulanceEnv()
    agent = AgentNet()
    agent.load_state_dict(torch.load(CKPT)["agent"])
    rows = []
    for seed in range(1, 11):
        obs, state = env.reset(seed)
        done = False
        while not done:
            obs, state, r, done, info = env.step(act(agent, obs, 0.0))
        print("qmix seed", seed, "| ambulance_s:", info["ambulance_s"], "| stopped:", info["stopped_s"])
        rows.append([seed, info["ambulance_s"], info["stopped_s"]])
    env.close()
    with open("results/qmix_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "ambulance_s", "stopped_vehicle_s"])
        w.writerows(rows)
    a = [r[1] for r in rows if r[1] is not None]
    print("\nQMIX (greedy, seeds 1-10): ambulance", round(sum(a) / len(a), 1),
          "s | cross-traffic stopped", round(sum(r[2] for r in rows) / len(rows)), "veh-s")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        train(int(sys.argv[1]) if len(sys.argv) > 1 else 3)