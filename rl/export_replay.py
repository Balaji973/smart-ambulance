import json
import torch
import env as envmod
from env import AmbulanceEnv
from qmix import AgentNet, act, CKPT

env = AmbulanceEnv()
agent = AgentNet()
agent.load_state_dict(torch.load(CKPT)["agent"])
agent.eval()
total = sum(env.net.getEdge(e).getLength() for e in env.route)

def code(obs):
    out = ""
    for i, a in enumerate(env.agents):
        gap = a["d"] - env.pos
        if gap < 0:
            out += "P"
        elif obs[i][4] > 0.5:
            out += "G"
        elif gap <= 300:
            out += "A"
        else:
            out += "R"
    return out

def run(seed, policy):
    obs, state = env.reset(seed)
    t0 = envmod.traci.simulation.getTime()
    frames = [[0.0, 0.0, 0.0, code(obs)]]
    done = False
    while not done:
        obs, state, r, done, info = env.step(policy(obs))
        t = envmod.traci.simulation.getTime() - t0
        frames.append([round(t, 1), round(env.pos, 1), round(env.spd, 1), code(obs)])
    frames.append([info["ambulance_s"], round(total, 1), 0.0, "P" * env.n])
    return {"time": info["ambulance_s"], "stopped": info["stopped_s"], "frames": frames}

out = {"total_m": round(total, 1), "seeds": {}}
for seed in range(1, 11):
    fixed = run(seed, lambda o: [1] * env.n)
    qmix = run(seed, lambda o: act(agent, o, 0.0))
    out["seeds"][str(seed)] = {"fixed": fixed, "qmix": qmix}
    print("seed", seed, "| fixed", fixed["time"], "| qmix", qmix["time"])
env.close()
json.dump(out, open("web/replay.json", "w"))
print("saved web/replay.json")