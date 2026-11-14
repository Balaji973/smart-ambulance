import numpy as np
import libsumo as traci
from env import AmbulanceEnv, AMB

class AmbulanceEnv2(AmbulanceEnv):
    """Same environment as before, but each agent sees 9 numbers instead of 6."""

    def _obs(self):
        now = traci.simulation.getTime()
        wait = 0.0
        if AMB in traci.vehicle.getIDList():
            self.pos = traci.vehicle.getDistance(AMB)
            self.spd = traci.vehicle.getSpeed(AMB)
            wait = traci.vehicle.getWaitingTime(AMB)
        rows = []
        for a in self.agents:
            gap = a["d"] - self.pos
            q_amb = traci.edge.getLastStepHaltingNumber(a["amb_edge"])
            q_oth = sum(traci.edge.getLastStepHaltingNumber(e) for e in a["other_edges"])
            n_edge = traci.edge.getLastStepVehicleNumber(a["amb_edge"])
            st = traci.trafficlight.getRedYellowGreenState(a["tls"])
            green = 1.0 if all(st[j] in "Gg" for j in a["idxs"]) else 0.0
            since = min(now - self.last_switch.get(a["tls"], -999), 60) / 60
            to_switch = min(max(traci.trafficlight.getNextSwitch(a["tls"]) - now, 0), 60) / 60
            rows.append([np.clip(gap, -50, 500) / 500, self.spd / 20,
                         min(q_amb, 20) / 20, min(q_oth, 40) / 40, green, since,
                         to_switch, min(wait, 60) / 60, min(n_edge, 30) / 30])
        return np.array(rows, dtype=np.float32)