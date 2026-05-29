#!/usr/bin/env python3
"""Ciw traffic reference for shared source/sink trip JSON.

The Ciw model groups trips by exact lane route. Each route becomes one
infinite-server queueing node with deterministic sequential arrivals and
deterministic sequential service times equal to free-flow lane travel plus
signal waiting. That keeps the shared scheduled-trip input intact while using
Ciw's independent queueing-network FEL engine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        import ciw  # type: ignore
    except Exception as exc:
        write_json(out, {
            "kernel": "ciw-traffic-reference",
            "status": "unavailable",
            "message": f"Ciw is not installed for this Python interpreter: {exc}",
            "elapsedMs": (time.time() - start) * 1000.0,
        })
        return 0

    try:
        problem = read_json(Path(args.problem))
        result = run_ciw(problem, ciw)
        payload = {
            "kernel": "ciw-traffic-reference",
            "status": "ok",
            "elapsedMs": (time.time() - start) * 1000.0,
            "result": result,
        }
    except Exception as exc:
        payload = {
            "kernel": "ciw-traffic-reference",
            "status": "error",
            "message": str(exc),
            "elapsedMs": (time.time() - start) * 1000.0,
        }
    write_json(out, payload)
    return 0


def run_ciw(problem: Dict[str, Any], ciw: Any) -> Dict[str, Any]:
    network = problem["network"]
    params = problem.get("params", {})
    duration = float(params.get("durationSec", problem.get("durationSec", 0)))
    lane_by_id = {lane["id"]: lane for lane in network.get("lanes", [])}
    trips = normalize_trips(problem)
    validate_trips(network, trips, lane_by_id, duration)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for trip in trips:
        key = " ".join(str(lane_id) for lane_id in trip["route"])
        grouped.setdefault(key, []).append(trip)

    route_keys = sorted(grouped)
    if not route_keys:
        return empty_result()

    class SequenceDistribution(ciw.dists.Distribution):  # type: ignore
        def __init__(self, sequence: List[float]) -> None:
            self.sequence = list(sequence)
            self.index = 0

        def sample(self, t: float | None = None, ind: Any = None) -> float:
            if self.index >= len(self.sequence):
                return 1e12
            value = self.sequence[self.index]
            self.index += 1
            return max(0.0, float(value))

    arrival_distributions = []
    service_distributions = []
    route_distances: List[float] = []
    for key in route_keys:
        route_trips = sorted(grouped[key], key=lambda t: (float(t["departSec"]), str(t.get("id", ""))))
        departures = [float(t["departSec"]) for t in route_trips]
        interarrivals = [departures[0]]
        for i in range(1, len(departures)):
            interarrivals.append(max(0.0, departures[i] - departures[i - 1]))
        interarrivals.append(1e12)
        route = list(route_trips[0]["route"])
        service_times = [route_duration(network, lane_by_id, route, float(t["departSec"])) for t in route_trips]
        service_times.append(0.0)
        arrival_distributions.append(SequenceDistribution(interarrivals))
        service_distributions.append(SequenceDistribution(service_times))
        route_distances.append(route_distance(lane_by_id, route))

    routing = [[0.0 for _ in route_keys] for _ in route_keys]
    network_model = ciw.create_network(
        arrival_distributions=arrival_distributions,
        service_distributions=service_distributions,
        routing=routing,
        number_of_servers=[float("inf") for _ in route_keys],
    )
    ciw.seed(0)
    simulation = ciw.Simulation(network_model)
    simulation.simulate_until_max_time(duration + 1e-9)
    records = simulation.get_all_records()

    departed = 0
    arrived = 0
    durations: List[float] = []
    speeds: List[float] = []
    waits: List[float] = []
    intervals: List[Tuple[float, float]] = []
    for rec in records:
        arrival = float(rec.arrival_date)
        exit_date = float(rec.exit_date) if rec.exit_date is not None else math.inf
        if arrival <= duration + 1e-9:
            departed += 1
            intervals.append((arrival, min(exit_date, duration)))
        if exit_date <= duration + 1e-9:
            arrived += 1
            travel_time = exit_date - arrival
            route_index = int(rec.node) - 1
            durations.append(travel_time)
            speeds.append(route_distances[route_index] / max(1e-9, travel_time))
            waits.append(max(0.0, travel_time - route_distances[route_index] / 13.5))

    return {
        "generatedDemand": len(trips),
        "departed": departed,
        "arrived": arrived,
        "activeAtEnd": max(0, departed - arrived),
        "maxActive": max_active(intervals),
        "meanTravelTimeSec": mean(durations),
        "meanSpeedMps": mean(speeds),
        "meanWaitingTimeSec": mean(waits),
        "collisionCount": 0,
        "notes": [
            "Ciw queueing-network future-event simulation",
            "each unique scheduled source/sink route is an infinite-server Ciw node with sequential deterministic arrivals and service times",
        ],
    }


def empty_result() -> Dict[str, Any]:
    return {
        "generatedDemand": 0,
        "departed": 0,
        "arrived": 0,
        "activeAtEnd": 0,
        "maxActive": 0,
        "meanTravelTimeSec": 0.0,
        "meanSpeedMps": 0.0,
        "meanWaitingTimeSec": 0.0,
        "collisionCount": 0,
        "notes": ["no scheduled trips"],
    }


def normalize_trips(problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    trips = problem.get("trips") or problem.get("scheduledTrips") or []
    if trips:
        return [dict(trip) for trip in trips]
    out: List[Dict[str, Any]] = []
    for demand in problem.get("demand", []):
        vehicles = int(demand.get("vehicles", 0))
        begin = float(demand.get("beginSec", 0))
        end = float(demand.get("endSec", begin))
        step = (end - begin) / max(1, vehicles)
        for i in range(vehicles):
            out.append({
                "id": f"{demand['id']}-{i + 1}",
                "departSec": begin + i * step,
                "sourceId": demand.get("sourceId"),
                "destinationSinkId": demand.get("sinkId"),
                "route": list(demand["route"]),
            })
    return out


def validate_trips(network: Dict[str, Any], trips: List[Dict[str, Any]], lane_by_id: Dict[str, Dict[str, Any]], duration: float) -> None:
    source_by_id = {source["id"]: source for source in network.get("sources", [])}
    sink_by_id = {sink["id"]: sink for sink in network.get("sinks", [])}
    for trip in trips:
        depart = float(trip["departSec"])
        if depart < 0 or depart > duration + 1e-9:
            raise ValueError(f"trip {trip.get('id')} departSec outside simulation duration: {depart}")
        source = source_by_id.get(trip.get("sourceId"))
        sink = sink_by_id.get(trip.get("destinationSinkId"))
        if source is None:
            raise ValueError(f"trip {trip.get('id')} references unknown source {trip.get('sourceId')}")
        if sink is None:
            raise ValueError(f"trip {trip.get('id')} references unknown sink {trip.get('destinationSinkId')}")
        allowed = source.get("destinationSinkIds") or list(sink_by_id.keys())
        if trip["destinationSinkId"] not in allowed:
            raise ValueError(f"trip {trip.get('id')} uses sink not allowed by source {source['id']}")
        route = trip.get("route") or []
        if not route:
            raise ValueError(f"trip {trip.get('id')} has empty route")
        current = source["nodeId"]
        for lane_id in route:
            lane = lane_by_id.get(lane_id)
            if lane is None:
                raise ValueError(f"trip {trip.get('id')} references unknown lane {lane_id}")
            if lane["from"] != current:
                raise ValueError(f"trip {trip.get('id')} route is not contiguous at lane {lane_id}")
            current = lane["to"]
        if current != sink["nodeId"]:
            raise ValueError(f"trip {trip.get('id')} route ends at {current}, expected sink node {sink['nodeId']}")


def route_duration(network: Dict[str, Any], lane_by_id: Dict[str, Dict[str, Any]], route: List[str], depart_sec: float) -> float:
    now = depart_sec
    for i, lane_id in enumerate(route):
        lane = lane_by_id[lane_id]
        now += lane_travel_sec(lane)
        if i < len(route) - 1:
            now += signal_wait_sec(network, lane_id, now)
    return max(0.0, now - depart_sec)


def route_distance(lane_by_id: Dict[str, Dict[str, Any]], route: List[str]) -> float:
    return sum(float(lane_by_id[lane_id]["lengthM"]) for lane_id in route)


def lane_travel_sec(lane: Dict[str, Any]) -> float:
    return float(lane["lengthM"]) / max(1e-9, float(lane["speedLimitMps"]))


def signal_wait_sec(network: Dict[str, Any], incoming_lane_id: str, at_sec: float) -> float:
    incoming_lane = next((lane for lane in network.get("lanes", []) if lane["id"] == incoming_lane_id), None)
    if incoming_lane is None:
        return 0.0
    signal = next((sig for sig in network.get("signals", []) if sig["nodeId"] == incoming_lane["to"]), None)
    if signal is None:
        return 0.0
    phases = signal.get("phases", [])
    cycle = sum(float(phase["durationSec"]) for phase in phases)
    if cycle <= 0:
        return 0.0
    local = (at_sec + float(signal.get("offsetSec", 0))) % cycle
    waits = []
    cursor = 0.0
    for phase in phases:
        start = cursor
        end = cursor + float(phase["durationSec"])
        if incoming_lane_id in phase.get("greenLanes", []):
            if start <= local < end:
                return 0.0
            waits.append(start - local if start >= local else cycle - local + start)
        cursor = end
    return max(0.0, min(waits)) if waits else 0.0


def max_active(intervals: List[Tuple[float, float]]) -> int:
    events: List[Tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    active = 0
    best = 0
    for _, delta in sorted(events, key=lambda x: (x[0], x[1])):
        active += delta
        best = max(best, active)
    return best


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(str(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def mean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else 0.0


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
