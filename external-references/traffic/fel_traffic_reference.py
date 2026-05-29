#!/usr/bin/env python3
"""Dependency-free external Future Event List reference for traffic DES specs.

This model is intentionally compact: cars are events moving across lane
resources, signals gate lane changes, and lane capacities cap occupancy. It
consumes the same `des/model-spec/v1` JSON that the TypeScript registry runs,
and the shared source/sink trip JSON used by comparison runners, so validators
can compare internal traffic-control stats against an external event-list
implementation without requiring SUMO or SimPy.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


TrafficNetwork = Dict[str, Any]
Trip = Dict[str, Any]


@dataclass
class Car:
    car_id: int
    source_id: str
    destination_sink_id: str
    route: List[str]
    route_index: int
    created_at: float
    seq: int = 0
    distance_m: float = 0.0


class TrafficFelReference:
    def __init__(self, params: Dict[str, Any], network: TrafficNetwork, scheduled_trips: List[Trip]) -> None:
        self.params = params
        self.network = network
        self.scheduled_trips = sorted(scheduled_trips, key=lambda t: (float(t["departSec"]), str(t["sourceId"]), str(t["destinationSinkId"])))
        self.nodes = {n["id"]: n for n in network["nodes"]}
        self.lanes = {l["id"]: l for l in network["lanes"]}
        self.signals = {s["nodeId"]: s for s in network.get("signals", [])}
        self.sources = {s["id"]: s for s in network["sources"]}
        self.sinks = {s["id"]: s for s in network["sinks"]}
        self.routes: Dict[str, List[str]] = {}
        self.precompute_routes()

        self.events: List[Tuple[float, int, int, str, Dict[str, Any]]] = []
        self.event_seq = 0
        self.next_car_id = 1
        self.active: Dict[int, Car] = {}
        self.lane_occupancy = {lane_id: 0 for lane_id in self.lanes}
        self.max_lane_occupancy = {lane_id: 0 for lane_id in self.lanes}
        self.lane_occupancy_area = {lane_id: 0.0 for lane_id in self.lanes}
        self.last_time = 0.0
        self.generated_demand = 0
        self.entered = 0
        self.exited = 0
        self.dropped = 0
        self.max_active = 0
        self.travel_times: List[float] = []
        self.completed_distances: List[float] = []
        self.time_series: List[Dict[str, Any]] = []
        self.next_sample_at = 0.0

    def run(self) -> Dict[str, Any]:
        duration = float(self.params["durationSec"])
        for trip in self.scheduled_trips:
            depart = float(trip["departSec"])
            if depart < duration - 1e-9:
                self.schedule(depart, 0, "depart", trip)

        sample_every = float(self.params.get("sampleEverySec", max(float(self.params.get("dtSec", 1.0)), 5.0)))
        self.next_sample_at = 0.0

        while self.events:
            time_sec, _priority, _seq, kind, data = heapq.heappop(self.events)
            if time_sec > duration + 1e-9:
                break
            self.accrue(time_sec)
            while self.next_sample_at <= time_sec + 1e-9 and self.next_sample_at <= duration + 1e-9:
                self.sample(self.next_sample_at)
                self.next_sample_at += sample_every
            if kind == "depart":
                self.handle_depart(time_sec, data)
            elif kind == "lane_exit":
                self.handle_lane_exit(time_sec, int(data["carId"]), int(data["seq"]))

        self.accrue(duration)
        while self.next_sample_at <= duration + 1e-9:
            self.sample(self.next_sample_at)
            self.next_sample_at += sample_every
        return self.result(duration)

    def schedule(self, time_sec: float, priority: int, kind: str, data: Dict[str, Any]) -> None:
        self.event_seq += 1
        heapq.heappush(self.events, (time_sec, priority, self.event_seq, kind, data))

    def accrue(self, time_sec: float) -> None:
        dt = max(0.0, time_sec - self.last_time)
        if dt <= 0:
            self.last_time = max(self.last_time, time_sec)
            return
        for lane_id, occupancy in self.lane_occupancy.items():
            self.lane_occupancy_area[lane_id] += occupancy * dt
        self.last_time = time_sec

    def sample(self, time_sec: float) -> None:
        self.time_series.append({
            "timeSec": time_sec,
            "entered": self.entered,
            "exited": self.exited,
            "dropped": self.dropped,
            "active": len(self.active),
            "laneOccupancy": dict(self.lane_occupancy),
        })

    def handle_depart(self, time_sec: float, trip: Trip) -> None:
        self.generated_demand += 1
        max_cars = int(self.params.get("maxCars", 10**9))
        if len(self.active) >= max_cars:
            self.dropped += 1
            return
        source_id = str(trip["sourceId"])
        sink_id = str(trip["destinationSinkId"])
        route = list(trip.get("route") or self.routes.get(f"{source_id}->{sink_id}") or [])
        if not route:
            self.dropped += 1
            return
        first_lane = route[0]
        if not self.can_enter_lane(first_lane):
            self.dropped += 1
            return
        car = Car(
            car_id=self.next_car_id,
            source_id=source_id,
            destination_sink_id=sink_id,
            route=route,
            route_index=0,
            created_at=time_sec,
        )
        self.next_car_id += 1
        self.active[car.car_id] = car
        self.entered += 1
        self.enter_lane(first_lane)
        self.max_active = max(self.max_active, len(self.active))
        self.schedule_lane_exit(time_sec, car)

    def handle_lane_exit(self, time_sec: float, car_id: int, event_seq: int) -> None:
        car = self.active.get(car_id)
        if car is None or car.seq != event_seq:
            return
        lane_id = car.route[car.route_index]
        next_lane_id = car.route[car.route_index + 1] if car.route_index + 1 < len(car.route) else None
        if next_lane_id is None:
            self.leave_lane(lane_id)
            lane = self.lanes[lane_id]
            car.distance_m += float(lane["lengthM"])
            self.exited += 1
            self.travel_times.append(time_sec - car.created_at)
            self.completed_distances.append(car.distance_m)
            del self.active[car_id]
            return

        if not self.signal_allows(lane_id, time_sec):
            self.retry_lane_exit(time_sec, car, self.next_green_time(lane_id, time_sec))
            return
        if not self.can_enter_lane(next_lane_id):
            self.retry_lane_exit(time_sec, car, time_sec + self.retry_step())
            return

        self.leave_lane(lane_id)
        car.distance_m += float(self.lanes[lane_id]["lengthM"])
        car.route_index += 1
        self.enter_lane(next_lane_id)
        self.schedule_lane_exit(time_sec, car)

    def retry_lane_exit(self, time_sec: float, car: Car, retry_at: float) -> None:
        car.seq += 1
        self.schedule(max(time_sec + 1e-6, retry_at), 2, "lane_exit", {"carId": car.car_id, "seq": car.seq})

    def schedule_lane_exit(self, time_sec: float, car: Car) -> None:
        lane = self.lanes[car.route[car.route_index]]
        travel = float(lane["lengthM"]) / max(1e-9, float(lane["speedLimitMps"]))
        car.seq += 1
        self.schedule(time_sec + travel, 2, "lane_exit", {"carId": car.car_id, "seq": car.seq})

    def enter_lane(self, lane_id: str) -> None:
        self.lane_occupancy[lane_id] += 1
        self.max_lane_occupancy[lane_id] = max(self.max_lane_occupancy[lane_id], self.lane_occupancy[lane_id])

    def leave_lane(self, lane_id: str) -> None:
        self.lane_occupancy[lane_id] = max(0, self.lane_occupancy[lane_id] - 1)

    def can_enter_lane(self, lane_id: str) -> bool:
        return self.lane_occupancy[lane_id] < self.lane_capacity(lane_id)

    def lane_capacity(self, lane_id: str) -> int:
        lane = self.lanes[lane_id]
        if "capacity" in lane:
            return int(lane["capacity"])
        vehicle_space = float(self.params.get("carLengthM", 4.8)) + float(self.params.get("minGapM", 2.5))
        return max(1, int(math.floor(float(lane["lengthM"]) / max(1e-9, vehicle_space))))

    def retry_step(self) -> float:
        return max(0.1, min(1.0, float(self.params.get("dtSec", 1.0))))

    def signal_allows(self, incoming_lane_id: str, time_sec: float) -> bool:
        lane = self.lanes[incoming_lane_id]
        node = self.nodes.get(lane["to"])
        if node is None or node.get("kind") != "intersection":
            return True
        signal = self.signals.get(node["id"])
        if not signal:
            return True
        return incoming_lane_id in current_signal_phase(signal, time_sec)["greenLanes"]

    def next_green_time(self, incoming_lane_id: str, time_sec: float) -> float:
        lane = self.lanes[incoming_lane_id]
        signal = self.signals.get(lane["to"])
        if not signal:
            return time_sec
        phases = signal["phases"]
        cycle = sum(float(p["durationSec"]) for p in phases)
        if cycle <= 0:
            return time_sec
        offset = float(signal.get("offsetSec", 0.0))
        cycle_index = math.floor((time_sec + offset) / cycle)
        for k in range(4):
            start = (cycle_index + k) * cycle - offset
            cursor = 0.0
            for phase in phases:
                phase_start = start + cursor
                phase_end = phase_start + float(phase["durationSec"])
                cursor += float(phase["durationSec"])
                if phase_end < time_sec - 1e-9:
                    continue
                if incoming_lane_id in phase["greenLanes"]:
                    return max(time_sec, phase_start)
        return time_sec + self.retry_step()

    def precompute_routes(self) -> None:
        for source in self.network["sources"]:
            sink_ids = source.get("destinationSinkIds") or [s["id"] for s in self.network["sinks"]]
            for sink_id in sink_ids:
                sink = self.sinks.get(sink_id)
                if not sink:
                    continue
                route = shortest_lane_path(self.network, source["nodeId"], sink["nodeId"])
                if route:
                    self.routes[f"{source['id']}->{sink_id}"] = route

    def result(self, duration: float) -> Dict[str, Any]:
        sorted_travel = sorted(self.travel_times)
        total_distance = sum(self.completed_distances)
        total_travel = sum(self.travel_times)
        lane_stats = []
        for lane_id, lane in self.lanes.items():
            lane_stats.append({
                "id": lane_id,
                "from": lane["from"],
                "to": lane["to"],
                "capacity": self.lane_capacity(lane_id),
                "finalOccupancy": self.lane_occupancy[lane_id],
                "maxOccupancy": self.max_lane_occupancy[lane_id],
                "avgOccupancy": self.lane_occupancy_area[lane_id] / max(1e-9, duration),
            })
        return {
            "generatedDemand": self.generated_demand,
            "departed": self.entered,
            "arrived": self.exited,
            "entered": self.entered,
            "exited": self.exited,
            "dropped": self.dropped,
            "activeAtEnd": len(self.active),
            "maxActiveCars": self.max_active,
            "completionRatio": self.exited / max(1, self.entered),
            "meanTravelTimeSec": mean(sorted_travel),
            "p95TravelTimeSec": percentile(sorted_travel, 0.95),
            "meanSpeedMps": total_distance / max(1e-9, total_travel),
            "meanWaitingTimeSec": 0.0,
            "collisionCount": 0,
            "eventCount": self.event_seq,
            "laneStats": lane_stats,
            "timeSeries": self.time_series,
            "notes": [
                "future-event-list traffic baseline",
                "scheduled departures originate from traffic source entities and terminate at sink entities",
            ],
        }


def current_signal_phase(signal: Dict[str, Any], time_sec: float) -> Dict[str, Any]:
    cycle = sum(float(p["durationSec"]) for p in signal["phases"])
    t = ((time_sec + float(signal.get("offsetSec", 0.0))) % cycle + cycle) % cycle
    for phase in signal["phases"]:
        if t < float(phase["durationSec"]):
            return phase
        t -= float(phase["durationSec"])
    return signal["phases"][-1]


def shortest_lane_path(network: TrafficNetwork, source_node_id: str, sink_node_id: str) -> List[str]:
    nodes = [n["id"] for n in network["nodes"]]
    dist = {node_id: float("inf") for node_id in nodes}
    prev_lane: Dict[str, str] = {}
    prev_node: Dict[str, str] = {}
    pending = set(nodes)
    dist[source_node_id] = 0.0
    while pending:
        current = min(pending, key=lambda n: dist.get(n, float("inf")))
        if not math.isfinite(dist[current]):
            break
        pending.remove(current)
        if current == sink_node_id:
            break
        for lane in [l for l in network["lanes"] if l["from"] == current]:
            alt = dist[current] + float(lane["lengthM"])
            if alt < dist.get(lane["to"], float("inf")):
                dist[lane["to"]] = alt
                prev_lane[lane["to"]] = lane["id"]
                prev_node[lane["to"]] = current
    if not math.isfinite(dist.get(sink_node_id, float("inf"))):
        return []
    route: List[str] = []
    cur = sink_node_id
    while cur != source_node_id:
        lane = prev_lane.get(cur)
        parent = prev_node.get(cur)
        if lane is None or parent is None:
            return []
        route.append(lane)
        cur = parent
    route.reverse()
    return route


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(sorted_xs: List[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, math.floor(p * (len(sorted_xs) - 1)))
    return sorted_xs[idx]


def build_five_intersection_network() -> TrafficNetwork:
    nodes = [
        {"id": "W", "kind": "source", "x": 0, "y": 1},
        {"id": "S0", "kind": "source", "x": 1, "y": 2},
        {"id": "N2", "kind": "source", "x": 3, "y": 0},
        {"id": "I0", "kind": "intersection", "x": 1, "y": 1},
        {"id": "I1", "kind": "intersection", "x": 2, "y": 1},
        {"id": "I2", "kind": "intersection", "x": 3, "y": 1},
        {"id": "I3", "kind": "intersection", "x": 4, "y": 1},
        {"id": "I4", "kind": "intersection", "x": 5, "y": 1},
        {"id": "E", "kind": "sink", "x": 6, "y": 1},
        {"id": "N1", "kind": "sink", "x": 2, "y": 0},
        {"id": "S4", "kind": "sink", "x": 5, "y": 2},
    ]

    def lane(lane_id: str, from_node: str, to_node: str, length_m: float = 120.0) -> Dict[str, Any]:
        return {"id": lane_id, "from": from_node, "to": to_node, "lengthM": length_m, "speedLimitMps": 13.5}

    lanes = [
        lane("W-I0", "W", "I0", 90),
        lane("S0-I0", "S0", "I0", 85),
        lane("I0-I1", "I0", "I1"),
        lane("I1-I2", "I1", "I2"),
        lane("N2-I2", "N2", "I2", 90),
        lane("I2-I3", "I2", "I3"),
        lane("I3-I4", "I3", "I4"),
        lane("I4-E", "I4", "E", 100),
        lane("I1-N1", "I1", "N1", 80),
        lane("I4-S4", "I4", "S4", 85),
    ]

    def phase(name: str, green_lanes: List[str], duration_sec: float) -> Dict[str, Any]:
        return {"name": name, "greenLanes": green_lanes, "durationSec": duration_sec}

    signals = [
        {"nodeId": "I0", "phases": [phase("main", ["W-I0"], 28), phase("side", ["S0-I0"], 16)]},
        {"nodeId": "I1", "phases": [phase("main", ["I0-I1"], 30)]},
        {"nodeId": "I2", "phases": [phase("main", ["I1-I2"], 26), phase("side", ["N2-I2"], 18)], "offsetSec": 5},
        {"nodeId": "I3", "phases": [phase("main", ["I2-I3"], 30)]},
        {"nodeId": "I4", "phases": [phase("main", ["I3-I4"], 26)]},
    ]
    sources = [
        {"id": "west", "nodeId": "W", "ratePerMin": 18, "destinationSinkIds": ["east", "north1", "south4"]},
        {"id": "south0", "nodeId": "S0", "ratePerMin": 7, "destinationSinkIds": ["east", "north1", "south4"]},
        {"id": "north2", "nodeId": "N2", "ratePerMin": 8, "destinationSinkIds": ["east", "south4"]},
    ]
    sinks = [
        {"id": "east", "nodeId": "E"},
        {"id": "north1", "nodeId": "N1"},
        {"id": "south4", "nodeId": "S4"},
    ]
    return {"nodes": nodes, "lanes": lanes, "signals": signals, "sources": sources, "sinks": sinks}


def generated_schedule_from_rates(params: Dict[str, Any], network: TrafficNetwork) -> List[Trip]:
    duration = float(params["durationSec"])
    dt = float(params.get("dtSec", 1.0))
    mult = float(params.get("spawnRateMultiplier", 1.0))
    accum = {s["id"]: 0.0 for s in network["sources"]}
    trips: List[Trip] = []
    t = 0.0
    while t < duration - 1e-9:
        for source in network["sources"]:
            sink_ids = source.get("destinationSinkIds") or [s["id"] for s in network["sinks"]]
            accum[source["id"]] += float(source.get("ratePerMin", 0.0)) * mult * dt / 60.0
            count = int(math.floor(accum[source["id"]]))
            accum[source["id"]] -= count
            for k in range(count):
                sink_id = sink_ids[(len(trips) + k) % len(sink_ids)]
                trips.append({"departSec": t, "sourceId": source["id"], "destinationSinkId": sink_id})
        t += dt
    return trips


def load_problem(path: str) -> Tuple[str, Dict[str, Any], TrafficNetwork, List[Trip]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("$schema") == "des/model-spec/v1":
        model = raw.get("model", "traffic-flow")
        params = dict(raw.get("parameters", {}))
        network = params.get("network") or build_five_intersection_network()
        scheduled = list(params.get("scheduledTrips") or [])
        if not scheduled:
            scheduled = generated_schedule_from_rates(params, network)
        return model, params, network, scheduled
    params = dict(raw.get("params", raw.get("parameters", raw)))
    network = raw.get("network") or params.get("network") or build_five_intersection_network()
    scheduled = list(raw.get("trips") or raw.get("scheduledTrips") or params.get("scheduledTrips") or raw.get("demand") or [])
    if scheduled and "vehicles" in scheduled[0]:
        scheduled = expand_demand_rows(scheduled)
    if not scheduled:
        scheduled = generated_schedule_from_rates(params, network)
    return raw.get("model", "traffic-flow"), params, network, scheduled


def expand_demand_rows(rows: List[Dict[str, Any]]) -> List[Trip]:
    out: List[Trip] = []
    for row in rows:
        vehicles = int(row.get("vehicles", 0))
        begin = float(row.get("beginSec", 0.0))
        end = float(row.get("endSec", begin))
        for i in range(vehicles):
            depart = begin if vehicles <= 1 else begin + (end - begin) * i / vehicles
            out.append({
                "departSec": depart,
                "sourceId": row["sourceId"],
                "destinationSinkId": row["sinkId"],
                "route": row.get("route"),
            })
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, help="Traffic DES model-spec JSON or normalized traffic problem")
    parser.add_argument("--out", default="out/external/traffic/fel-reference.json")
    args = parser.parse_args()

    start = time.time()
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
    try:
        model, params, network, scheduled = load_problem(args.problem)
        result = TrafficFelReference(params, network, scheduled).run()
        payload = {
            "kernel": "python-traffic-fel-reference",
            "model": model,
            "status": "ok",
            "elapsedMs": (time.time() - start) * 1000.0,
            "input": {
                "scheduledTrips": len(scheduled),
                "nodes": len(network["nodes"]),
                "lanes": len(network["lanes"]),
                "signals": len(network.get("signals", [])),
            },
            "result": result,
        }
    except Exception as exc:
        payload = {
            "kernel": "python-traffic-fel-reference",
            "status": "error",
            "message": str(exc),
            "elapsedMs": (time.time() - start) * 1000.0,
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    if payload.get("status") == "ok":
        result = payload["result"]
        print(
            "traffic FEL reference: "
            f"entered={result['entered']} exited={result['exited']} "
            f"dropped={result['dropped']} meanTravel={result['meanTravelTimeSec']:.3f}"
        )
    else:
        print(f"traffic FEL reference error: {payload.get('message')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
