# Smart Ambulance: Cooperative Multi-Agent RL (QMIX) for Edge-Cloud Emergency Signal Coordination



## Introduction

Ambulances in Chennai lose a large part of their journey time at traffic signals. During peak hours the average response time is **20–40 minutes**, against the globally recommended **8-minute "golden window"**. In a cardiac arrest, every minute of delay lowers the patient's chance of survival by **7–10%**.

Today's solutions each have a weakness:

- **Manual clearing by traffic police** cannot coordinate several junctions at once.
- **Fixed green-wave systems** help the ambulance but block cross-traffic completely.
- **Centralised smart signals** depend on one server, which is a single point of failure and adds latency.

## Proposed Solution

This project treats **every traffic signal as an intelligent agent** trained with **QMIX**, a cooperative multi-agent reinforcement learning algorithm. The agents learn to cooperate with their neighbours, so each signal acts for the good of the whole corridor and not just its own junction.

Each signal follows a **four-state, distance-based protocol**:

| State | Trigger | Behaviour |
|---|---|---|
| 🔴 RED | Normal operation | Regular signal cycle |
| 🟠 APPROACHING | Ambulance ~300 m away | Warns drivers in advance |
| 🟢 GREEN | Ambulance ~200 m away | Signal forced green |
| ⚪ PASSED | Ambulance has cleared the junction | Resumes normal cycle |

Intersections within **500 m** of the ambulance also coordinate with each other, so congestion does not build up around its path.

## Architecture (Edge-Cloud)

- **Cloud:** QMIX agents are trained in Python (PyTorch) on the **SUMO** traffic simulator.
- **Edge:** The trained model is exported to **TensorFlow.js** and runs at each intersection, with inference in under 10 ms and no central server.
- **Data and visualisation:** OpenStreetMap (road network), Overpass API (signal and hospital locations), OSRM (routing) and Leaflet.js (map), all free with no API keys.

## Objectives and Targets

- Reduce ambulance travel time by **25–35%** compared with fixed-timing signals
- Reduce cross-traffic delay by **60%** compared with blanket green waves
- Build a decentralised system with no single point of failure
- Use real Chennai road and signal data instead of an abstract grid

## Tech Stack

`Python 3.12` · `PyTorch` · `SUMO` · `TensorFlow.js` · `Leaflet.js` · `OpenStreetMap` · `Overpass API` · `OSRM`

## Evaluation

The QMIX model will be compared against two baselines, **fixed-timing signals** and **rule-based green waves**. The comparison measures ambulance response time and cross-traffic delay.

## Sustainable Development Goals

SDG 3 (Good Health and Well-Being) · SDG 9 (Industry, Innovation and Infrastructure) · SDG 11 (Sustainable Cities and Communities)



> ⚠️ **Status:** Under development. This is a simulation-only academic prototype, and the performance figures above are project targets, not final results.
