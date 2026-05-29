#!/usr/bin/env python3
"""SimPy traffic reference for shared source/sink trip JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        import simpy  # type: ignore
    except Exception as exc:
        write_json(out, {
            "kernel": "simpy-traffic-reference",
            "status": "unavailable",
            "message": f"SimPy is not installed for this Python interpreter: {exc}",
            "elapsedMs": (time.time() - start) * 1000.0,
        })
        return 0

    try:
        problem = read_json(Path(args.problem))
        result = run_simpy(problem, simpy)
        payload = {
            "kernel": "simpy-traffic-reference",
            "status": "ok",
            "elapsedMs": (time.time() - start) * 1000.0,
            "result": result,
        }
    except Exception as exc:
        payload = {
            "kernel": "simpy-traffic-reference",
            "status": "error",
            "message": str(exc),
            "elapsedMs": (time.time() - start) * 1000.0,
        }
    write_json(out, payload)
    return 0


def run_simpy(problem: Dict[str, Any], simpy: Any) -> Dict[str, Any]:
    network = problem["network"]
    params = problem.get("params", {})
    duration = float(params.get("durationSec", problem.get("durationSec", 0)))
    lane_by_id = {lane["id"]: lane for lane in network.get("lanes", [])}
    trips = normalize_trips(problem)
    validate_trips(network, trips, lane_by_id, duration)

    env = simpy.Environment()
    active: Set[str] = set()
    counters = {
        "departed": 0,
        "arrived": 0,
        "maxActive": 0,
    }
    durations: List[float] = []
    speeds: List[float] = []
    waits: List[float] = []

    def vehicle(trip: Dict[str, Any]) -> Iterable[Any]:
        depart = float(trip["departSec"])
        if depart > env.now:
            yield env.timeout(depart - env.now)
        vehicle_id = str(trip.get("id") or f"trip-{counters['departed'] + 1}")
        active.add(vehicle_id)
        counters["departed"] += 1
        counters["maxActive"] = max(counters["maxActive"], len(active))
        wait_sec = 0.0
        distance_m = 0.0
        route = list(trip["route"])
        for i, lane_id in enumerate(route):
            lane = lane_by_id[lane_id]
            yield env.timeout(lane_travel_sec(lane))
            distance_m += float(lane["lengthM"])
            if i < len(route) - 1:
                signal_wait = signal_wait_sec(network, lane_id, env.now)
                wait_sec += signal_wait
                if signal_wait > 0:
                    yield env.timeout(signal_wait)
        active.discard(vehicle_id)
        counters["arrived"] += 1
        travel_time = env.now - depart
        durations.append(travel_time)
        speeds.append(distance_m / max(1e-9, travel_time))
        waits.append(wait_sec)

    for trip in trips:
        env.process(vehicle(trip))
    env.run(until=duration + 1e-9)

    return {
        "generatedDemand": len(trips),
        "departed": counters["departed"],
        "arrived": counters["arrived"],
        "activeAtEnd": len(active),
        "maxActive": counters["maxActive"],
        "meanTravelTimeSec": mean(durations),
        "meanSpeedMps": mean(speeds),
        "meanWaitingTimeSec": mean(waits),
        "collisionCount": 0,
        "notes": [
            "SimPy process-oriented future-event simulation",
            "each scheduled source/sink trip is a SimPy process with timeout events for departures, lane travel, and signal waits",
        ],
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
