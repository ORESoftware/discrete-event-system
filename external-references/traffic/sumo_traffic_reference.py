#!/usr/bin/env python3
"""Optional SUMO cross-check for DES traffic-flow models.

This is source-only glue. It generates SUMO XML files from a normalized traffic
problem, then calls SUMO and netconvert from the host environment. No simulator
binary is vendored in this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir")
    parser.add_argument("--sumo-bin")
    parser.add_argument("--netconvert-bin")
    parser.add_argument("--collision-action", default="warn")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        problem = read_json(Path(args.problem))
    except Exception as exc:
        write_json(out, payload("error", f"could not read problem JSON: {exc}"))
        return 0

    sumo_bin = find_binary(args.sumo_bin, "SUMO_BIN", "sumo")
    netconvert_bin = find_binary(args.netconvert_bin, "SUMO_NETCONVERT_BIN", "netconvert")
    if sumo_bin is None or netconvert_bin is None:
        missing = []
        if sumo_bin is None:
            missing.append("SUMO_BIN/sumo")
        if netconvert_bin is None:
            missing.append("SUMO_NETCONVERT_BIN/netconvert")
        write_json(out, payload(
            "unavailable",
            "SUMO external simulator is not installed or not on PATH; set "
            + " and ".join(missing)
            + " to enable this validator.",
            simulator={"sumo": sumo_bin, "netconvert": netconvert_bin},
        ))
        return 0

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="des-sumo-traffic-"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        files = write_sumo_inputs(problem, workdir, args.collision_action)
        net = run_command([
            netconvert_bin,
            "--node-files", str(files["nodes"]),
            "--edge-files", str(files["edges"]),
            "--output-file", str(files["net"]),
            "--no-turnarounds", "true",
            "--junctions.join", "false",
        ], workdir)
        if net.returncode != 0:
            write_json(out, payload("error", "netconvert failed", commands=[command_payload(net)]))
            return 0

        sim = run_command([
            sumo_bin,
            "-c", str(files["config"]),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ], workdir)
        if sim.returncode != 0:
            write_json(out, payload("error", "sumo failed", commands=[command_payload(net), command_payload(sim)]))
            return 0

        result = parse_sumo_outputs(problem, files, sim.stderr)
        write_json(out, payload(
            "ok",
            "SUMO simulation completed.",
            result=result,
            simulator={"sumo": sumo_bin, "netconvert": netconvert_bin},
            commands=[command_payload(net), command_payload(sim)],
            workdir=str(workdir),
        ))
        return 0
    except Exception as exc:
        write_json(out, payload("error", f"SUMO adapter failed: {exc}", workdir=str(workdir)))
        return 0


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def payload(status: str, message: str, **extra: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "module": "traffic-sumo-reference",
        "status": status,
        "message": message,
    }
    data.update(extra)
    return data


def find_binary(explicit: Optional[str], env_var: str, fallback: str) -> Optional[str]:
    value = explicit or os.environ.get(env_var) or fallback
    if os.path.isabs(value) or os.sep in value:
        return value if os.path.exists(value) else None
    found = shutil.which(value)
    if found:
        return found
    for base in [Path(site.getuserbase()) / "bin", Path.home() / "Library" / "Python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "bin"]:
        candidate = base / value
        if candidate.exists():
            return str(candidate)
    return None


def write_sumo_inputs(problem: Dict[str, Any], workdir: Path, collision_action: str) -> Dict[str, Path]:
    network = problem["network"]
    params = problem.get("params", {})
    duration = float(params.get("durationSec", problem.get("durationSec", 120)))
    dt = float(params.get("dtSec", 1))
    lane_width = float(params.get("laneWidthM", 3.7))
    files = {
        "nodes": workdir / "des.nodes.nod.xml",
        "edges": workdir / "des.edges.edg.xml",
        "routes": workdir / "des.routes.rou.xml",
        "net": workdir / "des.net.xml",
        "config": workdir / "des.sumocfg",
        "tripinfo": workdir / "tripinfo.xml",
        "summary": workdir / "summary.xml",
    }

    signal_nodes = {s.get("nodeId") for s in network.get("signals", [])}
    node_lines = ['<nodes>']
    for node in network.get("nodes", []):
        x = float(node["x"]) * 120.0
        y = -float(node["y"]) * 120.0
        node_type = "traffic_light" if node["id"] in signal_nodes else "priority"
        node_lines.append(
            f'  <node id="{esc(node["id"])}" x="{x:.6f}" y="{y:.6f}" type="{node_type}"/>'
        )
    node_lines.append("</nodes>")
    files["nodes"].write_text("\n".join(node_lines) + "\n", encoding="utf-8")

    edge_lines = ['<edges>']
    for lane in network.get("lanes", []):
        edge_lines.append(
            f'  <edge id="{esc(lane["id"])}" from="{esc(lane["from"])}" to="{esc(lane["to"])}" '
            f'priority="1" numLanes="1" width="{lane_width:.6f}" '
            f'length="{float(lane["lengthM"]):.6f}" speed="{float(lane["speedLimitMps"]):.6f}"/>'
        )
    edge_lines.append("</edges>")
    files["edges"].write_text("\n".join(edge_lines) + "\n", encoding="utf-8")

    car_length = float(params.get("carLengthM", 4.8))
    min_gap = float(params.get("minGapM", 2.5))
    accel = float(params.get("maxAccelMps2", 2.2))
    decel = float(params.get("maxDecelMps2", 4.0))
    max_speed = max(float(lane.get("speedLimitMps", 13.5)) for lane in network.get("lanes", [{"speedLimitMps": 13.5}]))
    route_lines = [
        '<routes>',
        f'  <vType id="des_car" length="{car_length:.6f}" minGap="{min_gap:.6f}" '
        f'accel="{accel:.6f}" decel="{decel:.6f}" maxSpeed="{max_speed:.6f}" sigma="0.5"/>',
    ]
    trips = problem.get("trips") or problem.get("scheduledTrips") or []
    if trips:
        route_ids: Dict[str, str] = {}
        for trip in trips:
            key = " ".join(str(edge) for edge in trip["route"])
            if key not in route_ids:
                route_ids[key] = "route_" + safe_id(str(len(route_ids) + 1))
                route_lines.append(f'  <route id="{route_ids[key]}" edges="{esc(key)}"/>')
        for i, trip in enumerate(sorted(trips, key=lambda t: (float(t.get("departSec", 0)), str(t.get("id", ""))))):
            key = " ".join(str(edge) for edge in trip["route"])
            vehicle_id = str(trip.get("id") or f"trip-{i + 1}")
            depart = float(trip.get("departSec", 0))
            route_lines.append(
                f'  <vehicle id="{esc(vehicle_id)}" type="des_car" route="{route_ids[key]}" '
                f'depart="{depart:.6f}" departLane="best" departSpeed="max" departPos="base"/>'
            )
    else:
        for demand in problem.get("demand", []):
            route_id = "route_" + safe_id(str(demand["id"]))
            edges = " ".join(esc(edge) for edge in demand["route"])
            route_lines.append(f'  <route id="{route_id}" edges="{edges}"/>')
            vehicles = int(demand.get("vehicles", 0))
            begin = float(demand.get("beginSec", 0))
            end = float(demand.get("endSec", duration))
            if vehicles > 0:
                route_lines.append(
                    f'  <flow id="{esc(str(demand["id"]))}" type="des_car" route="{route_id}" '
                    f'begin="{begin:.6f}" end="{end:.6f}" number="{vehicles}" '
                    f'departLane="best" departSpeed="max" departPos="base"/>'
                )
    route_lines.append("</routes>")
    files["routes"].write_text("\n".join(route_lines) + "\n", encoding="utf-8")

    files["config"].write_text(f"""<configuration>
  <input>
    <net-file value="{esc(str(files["net"]))}"/>
    <route-files value="{esc(str(files["routes"]))}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="{duration:.6f}"/>
    <step-length value="{dt:.6f}"/>
  </time>
  <processing>
    <collision.action value="{esc(collision_action)}"/>
    <collision.check-junctions value="true"/>
  </processing>
  <output>
    <tripinfo-output value="{esc(str(files["tripinfo"]))}"/>
    <summary-output value="{esc(str(files["summary"]))}"/>
  </output>
</configuration>
""", encoding="utf-8")
    return files


def parse_sumo_outputs(problem: Dict[str, Any], files: Dict[str, Path], stderr: str) -> Dict[str, Any]:
    trips = []
    if files["tripinfo"].exists():
        root = ET.parse(files["tripinfo"]).getroot()
        trips = list(root.iter("tripinfo"))
    durations = [float(t.get("duration", "0")) for t in trips]
    lengths = [float(t.get("routeLength", "0")) for t in trips]
    waiting = [float(t.get("waitingTime", "0")) for t in trips]
    speeds = [lengths[i] / durations[i] for i in range(len(trips)) if durations[i] > 0]
    summary = parse_summary(files["summary"])
    trips = problem.get("trips") or problem.get("scheduledTrips") or []
    generated = len(trips) if trips else sum(int(d.get("vehicles", 0)) for d in problem.get("demand", []))
    departed = summary.get("inserted") or summary.get("departed") or generated
    arrived = summary.get("arrived") or summary.get("ended") or len(trips)
    return {
        "generatedDemand": generated,
        "departed": int(departed),
        "arrived": int(arrived),
        "activeAtEnd": max(0, int(departed) - int(arrived)),
        "meanTravelTimeSec": mean(durations),
        "meanSpeedMps": mean(speeds),
        "meanWaitingTimeSec": mean(waiting),
        "collisionCount": int(summary.get("collisions") or collision_count(stderr)),
        "summary": summary,
    }


def parse_summary(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    out: Dict[str, int] = {}
    for step in root.iter("step"):
        for key in ("loaded", "inserted", "departed", "running", "waiting", "ended", "arrived", "collisions"):
            raw = step.get(key)
            if raw is None:
                continue
            try:
                out[key] = max(out.get(key, 0), int(float(raw)))
            except ValueError:
                pass
    return out


def collision_count(text: str) -> int:
    return len(re.findall(r"\bcollision\b", text, flags=re.IGNORECASE))


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else 0.0


def run_command(args: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=120)


def command_payload(proc: subprocess.CompletedProcess[str]) -> Dict[str, Any]:
    return {
        "args": proc.args,
        "status": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def esc(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


if __name__ == "__main__":
    sys.exit(main())
