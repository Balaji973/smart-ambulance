import json
import numpy as np
import libsumo as traci
import sumolib

CFG = "sim/osm.sumocfg"
NET = "sim/osm.net.xml.gz"
AMB = "ambulance_1"
VREF = 13.0   # m/s, roughly free-flow speed on this route

class AmbulanceEnv:
    """One agent per signal on the ambulance's route.
    Action per agent: 0 = preempt (request green for the ambulance), 1 = hold (leave the normal program alone)."""

    def __init__(self, route_file="results/route_v2.json", scale=5, decision_s=2,
                 hold_s=20, act_dist=200, amb_w=1.0, cross_w=0.5, max_time=1200):
        self.net = sumolib.net.readNet(NET)
        self.route = json.load(open(route_file))
        self.scale, self.decision_s, self.hold_s = scale, decision_s, hold_s
        self.act_dist, self.amb_w, self.cross_w = act_dist, amb_w, cross_w
        self.max_time = max_time
        self.running = False
        self.n, self.obs_dim, self.n_actions = 3, 6, 2

    def _green_phase(self, tls, idxs):
        current = traci.trafficlight.getProgram(tls)
        for logic in traci.trafficlight.getAllProgramLogics(tls):
            if logic.programID == current:
                for i, ph in enumerate(logic.phases):
                    if all(ph.state[j] in "Gg" for j in idxs):
                        return i
        return None

    def _build_agents(self):
        cum, d = {}, 0.0
        for eid in self.route:
            d += self.net.getEdge(eid).getLength()
            cum[eid] = d
        agents = []
        for tls in traci.trafficlight.getIDList():
            links = traci.trafficlight.getControlledLinks(tls)
            idxs, hit = [], None
            incoming = set()
            for idx, group in enumerate(links):
                for (inl, outl, via) in group:
                    ie, oe = inl.rsplit("_", 1)[0], outl.rsplit("_", 1)[0]
                    incoming.add(ie)
                    if ie in cum and oe in self.route and self.route.index(oe) == self.route.index(ie) + 1:
                        idxs.append(idx)
                        hit = ie
            if hit:
                agents.append({"tls": tls, "idxs": idxs, "d": cum[hit], "amb_edge": hit,
                               "other_edges": sorted(incoming - {hit}),
                               "phase": self._green_phase(tls, idxs)})
        agents.sort(key=lambda a: a["d"])
        return agents

    def reset(self, seed):
        if self.running:
            traci.close()
        traci.start(["sumo", "-c", CFG, "--seed", str(seed), "--no-warnings",
                     "--scale", str(self.scale)])
        self.running = True
        for _ in range(100):
            traci.simulationStep()
        traci.route.add("amb_route", self.route)
        traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance")
        traci.vehicletype.setColor("ambulance", (255, 0, 0, 255))
        traci.vehicle.add(AMB, "amb_route", typeID="ambulance")
        self.agents = self._build_agents()
        self.n = len(self.agents)
        self.cross_edges = sorted({e for a in self.agents for e in a["other_edges"]})
        self.start = traci.simulation.getTime()
        self.last_switch, self.stopped_total = {}, 0
        self.travel, self.pos, self.spd = None, 0.0, 0.0
        return self._obs(), self._state()

    def close(self):
        if self.running:
            traci.close()
            self.running = False

    def _obs(self):
        now = traci.simulation.getTime()
        if AMB in traci.vehicle.getIDList():
            self.pos = traci.vehicle.getDistance(AMB)
            self.spd = traci.vehicle.getSpeed(AMB)
        rows = []
        for a in self.agents:
            gap = a["d"] - self.pos
            q_amb = traci.edge.getLastStepHaltingNumber(a["amb_edge"])
            q_oth = sum(traci.edge.getLastStepHaltingNumber(e) for e in a["other_edges"])
            st = traci.trafficlight.getRedYellowGreenState(a["tls"])
            green = 1.0 if all(st[j] in "Gg" for j in a["idxs"]) else 0.0
            since = min(now - self.last_switch.get(a["tls"], -999), 60) / 60
            rows.append([np.clip(gap, -50, 500) / 500, self.spd / 20,
                         min(q_amb, 20) / 20, min(q_oth, 40) / 40, green, since])
        return np.array(rows, dtype=np.float32)

    def _state(self):
        return self._obs().flatten()

    def step(self, actions):
        now = traci.simulation.getTime()
        for a, action in zip(self.agents, actions):
            gap = a["d"] - self.pos
            if (action == 0 and a["phase"] is not None and 0 <= gap <= self.act_dist
                    and now - self.last_switch.get(a["tls"], -999) >= self.hold_s):
                st = traci.trafficlight.getRedYellowGreenState(a["tls"])
                if not all(st[j] in "Gg" for j in a["idxs"]):
                    traci.trafficlight.setPhase(a["tls"], a["phase"])
                    traci.trafficlight.setPhaseDuration(a["tls"], self.hold_s)
                self.last_switch[a["tls"]] = now

        stopped, steps, done, deficit = 0, 0, False, 0.0
        for _ in range(self.decision_s):
            traci.simulationStep()
            steps += 1
            ids = traci.vehicle.getIDList()
            stopped += sum(1 for v in ids if v != AMB and traci.vehicle.getSpeed(v) < 0.1)
            t = traci.simulation.getTime()
            if AMB in traci.simulation.getArrivedIDList():
                self.travel, done = t - self.start, True
                break
            if t - self.start > self.max_time:
                done = True
                break
            if AMB in ids:
                deficit += 1.0 - min(traci.vehicle.getSpeed(AMB) / VREF, 1.0)
            else:
                deficit += 1.0
        self.stopped_total += stopped
        local = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.cross_edges)
        reward = -self.amb_w * deficit / steps - self.cross_w * min(local, 60) / 30.0
        info = {"ambulance_s": self.travel, "stopped_s": self.stopped_total}
        return self._obs(), self._state(), reward, done, info


if __name__ == "__main__":
    env = AmbulanceEnv()
    policies = {"always_hold (same as fixed timing)": 1, "always_preempt": 0}
    for name, a in policies.items():
        amb, stp = [], []
        for seed in range(1, 11):
            obs, state = env.reset(seed)
            done = False
            while not done:
                obs, state, r, done, info = env.step([a] * env.n)
            amb.append(info["ambulance_s"])
            stp.append(info["stopped_s"])
            print(name, "seed", seed, "| ambulance_s:", info["ambulance_s"], "| stopped:", info["stopped_s"])
        print("==>", name, "AVERAGE ambulance", round(sum(amb) / len(amb), 1),
              "s | stopped", round(sum(stp) / len(stp)), "veh-s")
    env.close()