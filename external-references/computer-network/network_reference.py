#!/usr/bin/env python3
"""Dependency-free external reference for the computer-network DES.

This is intentionally source-only: it vendors no solver binary and imports only
Python's standard library.  The TypeScript validator invokes it through the
sanctioned external-program runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Packet = Dict[str, Any]


def protocol_profile(protocol: Optional[str]) -> Dict[str, Any]:
    p = protocol or "raw"
    if p == "http":
        return {"protocol": "http", "overheadBytes": 640, "startupDelayMs": 40}
    if p == "tcp":
        return {"protocol": "tcp", "overheadBytes": 40, "startupDelayMs": 20}
    if p == "udp":
        return {"protocol": "udp", "overheadBytes": 28, "startupDelayMs": 0}
    return {"protocol": "raw", "overheadBytes": 0, "startupDelayMs": 0}


def effective_packet_size(flow: Dict[str, Any]) -> int:
    return int(flow["packetSizeBytes"]) + int(protocol_profile(flow.get("protocol"))["overheadBytes"])


def default_forwarding_rate(kind: str) -> float:
    return {"host": 1000.0, "switch": 10000.0, "router": 5000.0}[kind]


def default_node_queue_limit(kind: str) -> int:
    return {"host": 128, "switch": 512, "router": 256}[kind]


def mb(bytes_count: float) -> float:
    return bytes_count / 1_000_000.0


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def percentile(sorted_xs: List[float], p: float) -> float:
    if not sorted_xs:
        return float("nan")
    idx = min(len(sorted_xs) - 1, math.floor(p * (len(sorted_xs) - 1)))
    return sorted_xs[idx]


def normalize_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    links: List[Dict[str, Any]] = []
    ids = set()
    for link in problem["links"]:
        base = dict(link)
        base["bidirectional"] = False
        links.append(base)
        ids.add(base["id"])
        if link.get("bidirectional"):
            reverse_id = f"{link['id']}:rev"
            i = 2
            while reverse_id in ids:
                reverse_id = f"{link['id']}:rev{i}"
                i += 1
            ids.add(reverse_id)
            rev = dict(link)
            rev["id"] = reverse_id
            rev["from"], rev["to"] = link["to"], link["from"]
            rev["bidirectional"] = False
            links.append(rev)
    out = dict(problem)
    out["links"] = links
    out["routingMetric"] = problem.get("routingMetric", "latency")
    return out


def link_weight(link: Dict[str, Any], metric: str) -> float:
    if metric == "hop":
        return 1.0
    if metric == "cost":
        return float(link.get("costPerMb", 0.0))
    return float(link["latencyMs"]) + (1500.0 * 8.0 / (float(link["bandwidthMbps"]) * 1e6) * 1000.0)


@dataclass
class Node:
    spec: Dict[str, Any]
    queue_limit: int = field(init=False)
    forwarding_rate: float = field(init=False)
    queue: List[Tuple[Packet, float]] = field(default_factory=list)
    forwarding_credit: float = 0.0
    queue_area: float = 0.0
    max_queue: int = 0
    total_queue_delay_ms: float = 0.0
    max_queue_delay_ms: float = 0.0
    processed_from_queue: int = 0
    received_packets: int = 0
    forwarded_packets: int = 0
    delivered_packets: int = 0
    dropped_packets: int = 0

    def __post_init__(self) -> None:
        self.queue_limit = int(self.spec.get("queueLimitPackets", default_node_queue_limit(self.spec["kind"])))
        self.forwarding_rate = float(self.spec.get("forwardingRatePps", default_forwarding_rate(self.spec["kind"])))

    @property
    def node_id(self) -> str:
        return self.spec["id"]

    def can_accept(self, reserved: int = 0) -> bool:
        return len(self.queue) + reserved < self.queue_limit

    def receive(self, packet: Packet, time_ms: float) -> bool:
        if not self.can_accept():
            return False
        packet["currentNodeId"] = self.node_id
        packet.pop("currentLinkId", None)
        self.queue.append((packet, time_ms))
        self.received_packets += 1
        self.max_queue = max(self.max_queue, len(self.queue))
        return True

    def record_queue(self, dt_ms: float) -> None:
        self.queue_area += len(self.queue) * dt_ms
        self.max_queue = max(self.max_queue, len(self.queue))

    def stats(self, total_ms: float) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "kind": self.spec["kind"],
            "forwardingRatePps": self.forwarding_rate,
            "queueLimitPackets": self.queue_limit,
            "receivedPackets": self.received_packets,
            "forwardedPackets": self.forwarded_packets,
            "deliveredPackets": self.delivered_packets,
            "droppedPackets": self.dropped_packets,
            "finalQueue": len(self.queue),
            "maxQueue": self.max_queue,
            "avgQueue": self.queue_area / max(1.0, total_ms),
            "meanQueueDelayMs": self.total_queue_delay_ms / max(1, self.processed_from_queue),
            "maxQueueDelayMs": self.max_queue_delay_ms,
        }


@dataclass
class Link:
    spec: Dict[str, Any]
    queue_limit: int = field(init=False)
    cost_per_mb: float = field(init=False)
    scheduled: List[Tuple[Packet, float]] = field(default_factory=list)
    available_at_ms: float = 0.0
    occupancy_area: float = 0.0
    max_in_flight: int = 0
    enqueued_packets: int = 0
    delivered_packets: int = 0
    dropped_packets: int = 0
    transmitted_bytes: int = 0
    total_serialization_ms: float = 0.0
    total_queue_delay_ms: float = 0.0
    max_queue_delay_ms: float = 0.0
    total_time_on_link_ms: float = 0.0
    max_time_on_link_ms: float = 0.0
    total_cost: float = 0.0

    def __post_init__(self) -> None:
        self.queue_limit = int(self.spec.get("queueLimitPackets", 64))
        self.cost_per_mb = float(self.spec.get("costPerMb", 0.0))

    @property
    def link_id(self) -> str:
        return self.spec["id"]

    def can_accept(self) -> bool:
        return len(self.scheduled) < self.queue_limit

    def serialization_ms(self, packet: Packet) -> float:
        return float(packet["sizeBytes"]) * 8.0 / (float(self.spec["bandwidthMbps"]) * 1e6) * 1000.0

    def enqueue(self, packet: Packet, time_ms: float) -> None:
        serialization_ms = self.serialization_ms(packet)
        start_at_ms = max(time_ms, self.available_at_ms)
        queue_delay_ms = max(0.0, start_at_ms - time_ms)
        deliver_at_ms = start_at_ms + serialization_ms + float(self.spec["latencyMs"])
        time_on_link_ms = queue_delay_ms + serialization_ms + float(self.spec["latencyMs"])
        self.available_at_ms = start_at_ms + serialization_ms
        packet_cost = mb(float(packet["sizeBytes"])) * self.cost_per_mb
        packet["cost"] += packet_cost
        packet.pop("currentNodeId", None)
        packet["currentLinkId"] = self.link_id
        packet["hops"].append(self.spec["to"])
        self.enqueued_packets += 1
        self.transmitted_bytes += int(packet["sizeBytes"])
        self.total_serialization_ms += serialization_ms
        self.total_queue_delay_ms += queue_delay_ms
        self.max_queue_delay_ms = max(self.max_queue_delay_ms, queue_delay_ms)
        self.total_time_on_link_ms += time_on_link_ms
        self.max_time_on_link_ms = max(self.max_time_on_link_ms, time_on_link_ms)
        self.total_cost += packet_cost
        self.scheduled.append((packet, deliver_at_ms))
        self.max_in_flight = max(self.max_in_flight, len(self.scheduled))

    def release_arrivals(self, time_ms: float) -> List[Packet]:
        ready: List[Packet] = []
        keep: List[Tuple[Packet, float]] = []
        for packet, deliver_at_ms in self.scheduled:
            if deliver_at_ms <= time_ms + 1e-9:
                ready.append(packet)
                self.delivered_packets += 1
            else:
                keep.append((packet, deliver_at_ms))
        self.scheduled = keep
        return ready

    def step_occupancy(self, dt_ms: float) -> None:
        self.occupancy_area += len(self.scheduled) * dt_ms
        self.max_in_flight = max(self.max_in_flight, len(self.scheduled))

    def stats(self, total_ms: float) -> Dict[str, Any]:
        simulated_sec = max(1e-9, total_ms / 1000.0)
        return {
            "id": self.link_id,
            "from": self.spec["from"],
            "to": self.spec["to"],
            "bandwidthMbps": float(self.spec["bandwidthMbps"]),
            "latencyMs": float(self.spec["latencyMs"]),
            "costPerMb": self.cost_per_mb,
            "queueLimitPackets": self.queue_limit,
            "enqueuedPackets": self.enqueued_packets,
            "deliveredPackets": self.delivered_packets,
            "droppedPackets": self.dropped_packets,
            "transmittedBytes": self.transmitted_bytes,
            "throughputMbps": self.transmitted_bytes * 8.0 / simulated_sec / 1e6,
            "utilization": min(1.0, self.total_serialization_ms / max(1e-9, total_ms)),
            "finalInFlight": len(self.scheduled),
            "maxInFlight": self.max_in_flight,
            "avgInFlight": self.occupancy_area / max(1.0, total_ms),
            "meanQueueDelayMs": self.total_queue_delay_ms / max(1, self.enqueued_packets),
            "maxQueueDelayMs": self.max_queue_delay_ms,
            "meanTimeOnLinkMs": self.total_time_on_link_ms / max(1, self.enqueued_packets),
            "maxTimeOnLinkMs": self.max_time_on_link_ms,
            "totalCost": self.total_cost,
        }


class ComputerNetworkReference:
    def __init__(self, problem: Dict[str, Any]) -> None:
        self.problem = normalize_problem(problem)
        self.nodes: Dict[str, Node] = {n["id"]: Node(n) for n in self.problem["nodes"]}
        self.links: Dict[str, Link] = {l["id"]: Link(l) for l in self.problem["links"]}
        self.outgoing: Dict[str, List[Link]] = {}
        for link in self.links.values():
            self.outgoing.setdefault(link.spec["from"], []).append(link)
        self.flows = [{"spec": f, "pending": 0.0, "generated": 0, "droppedAtSource": 0} for f in self.problem["flows"]]
        self.delivered: List[Packet] = []
        self.dropped: List[Packet] = []
        self.time_series: List[Dict[str, Any]] = []
        self.invariant_violations: List[str] = []
        self.route_cache: Dict[str, Optional[Link]] = {}
        self.next_packet_id = 1
        self.time_ms = 0.0
        self.max_active_packets = 0
        self.next_sample_at_ms = 0.0

    def has_work(self) -> bool:
        drain_until = float(self.problem["durationMs"]) + float(self.problem.get("drainAfterSourcesMs", 1000))
        return self.time_ms < float(self.problem["durationMs"]) or (
            self.active_packets() > 0 and self.time_ms < drain_until
        )

    def run(self) -> Dict[str, Any]:
        dt = float(self.problem["dtMs"])
        max_ticks = math.ceil((float(self.problem["durationMs"]) + float(self.problem.get("drainAfterSourcesMs", 1000))) / dt) + 5
        ticks = 0
        while ticks < max_ticks and self.has_work():
            self.step()
            ticks += 1
        return self.build_result()

    def step(self) -> None:
        dt = float(self.problem["dtMs"])
        self.release_link_arrivals()
        if self.time_ms < float(self.problem["durationMs"]):
            self.generate_flow_packets()
        for node in self.nodes.values():
            self.step_node(node)
        for link in self.links.values():
            link.step_occupancy(dt)
        self.record_invariants()
        self.record_stats()
        self.time_ms += dt

    def step_node(self, node: Node) -> None:
        dt = float(self.problem["dtMs"])
        node.forwarding_credit += node.forwarding_rate * dt / 1000.0
        budget = math.floor(node.forwarding_credit + 1e-12)
        while budget > 0 and node.queue:
            packet, enqueued_at = node.queue.pop(0)
            queue_delay_ms = max(0.0, self.time_ms - enqueued_at)
            node.total_queue_delay_ms += queue_delay_ms
            node.max_queue_delay_ms = max(node.max_queue_delay_ms, queue_delay_ms)
            node.processed_from_queue += 1
            node.forwarding_credit -= 1.0
            budget -= 1
            if packet["destination"] == node.node_id:
                node.delivered_packets += 1
                self.deliver(packet, node)
                continue
            if len(packet["hops"]) - 1 >= packet["ttlHops"]:
                node.dropped_packets += 1
                self.drop(packet, "ttl-expired", node.node_id)
                continue
            link = self.next_link(node.node_id, packet["destination"])
            if link is None:
                node.dropped_packets += 1
                self.drop(packet, "no-route", node.node_id)
                continue
            if not link.can_accept():
                node.dropped_packets += 1
                link.dropped_packets += 1
                self.drop(packet, "link-queue-overflow", link.link_id)
                continue
            node.forwarded_packets += 1
            link.enqueue(packet, self.time_ms)
        node.record_queue(dt)

    def release_link_arrivals(self) -> None:
        for link in self.links.values():
            for packet in link.release_arrivals(self.time_ms):
                node = self.nodes.get(link.spec["to"])
                if node is None:
                    self.drop(packet, "no-route", link.spec["to"])
                    continue
                if not node.receive(packet, self.time_ms):
                    node.dropped_packets += 1
                    self.drop(packet, "node-queue-overflow", node.node_id)

    def generate_flow_packets(self) -> None:
        for flow in self.flows:
            spec = flow["spec"]
            profile = protocol_profile(spec.get("protocol"))
            flow_start_ms = float(spec.get("startMs", 0.0)) + float(profile["startupDelayMs"])
            if self.time_ms < flow_start_ms or self.time_ms > float(spec.get("endMs", self.problem["durationMs"])):
                continue
            flow["pending"] += float(spec["ratePps"]) * float(self.problem["dtMs"]) / 1000.0
            while flow["pending"] >= 1.0 - 1e-12:
                if "maxPackets" in spec and flow["generated"] >= int(spec["maxPackets"]):
                    flow["pending"] = 0.0
                    break
                packet = {
                    "packetId": self.next_packet_id,
                    "flowId": spec["id"],
                    "protocol": profile["protocol"],
                    "source": spec["source"],
                    "destination": spec["destination"],
                    "payloadBytes": int(spec["packetSizeBytes"]),
                    "sizeBytes": effective_packet_size(spec),
                    "createdAtMs": self.time_ms,
                    "ttlHops": int(spec.get("ttlHops", max(8, len(self.problem["nodes"]) * 4))),
                    "hops": [spec["source"]],
                    "cost": 0.0,
                    "currentNodeId": spec["source"],
                }
                self.next_packet_id += 1
                flow["generated"] += 1
                flow["pending"] -= 1.0
                source = self.nodes[spec["source"]]
                if self.active_packets() >= float(self.problem.get("maxPacketsInSystem", float("inf"))):
                    flow["droppedAtSource"] += 1
                    source.dropped_packets += 1
                    self.drop(packet, "max-packets-in-system", source.node_id)
                    continue
                if not source.receive(packet, self.time_ms):
                    flow["droppedAtSource"] += 1
                    source.dropped_packets += 1
                    self.drop(packet, "node-queue-overflow", source.node_id)

    def deliver(self, packet: Packet, node: Node) -> None:
        packet["deliveredAtMs"] = self.time_ms
        packet["currentNodeId"] = node.node_id
        packet.pop("currentLinkId", None)
        self.delivered.append(packet)

    def drop(self, packet: Packet, reason: str, station_id: str) -> None:
        packet["droppedAtMs"] = self.time_ms
        packet["dropReason"] = reason
        packet["currentNodeId"] = station_id
        packet.pop("currentLinkId", None)
        self.dropped.append(packet)

    def active_packets(self) -> int:
        return sum(len(n.queue) for n in self.nodes.values()) + sum(len(l.scheduled) for l in self.links.values())

    def generated_packets(self) -> int:
        return sum(int(f["generated"]) for f in self.flows)

    def next_link(self, source: str, destination: str) -> Optional[Link]:
        key = f"{source}->{destination}"
        if key not in self.route_cache:
            self.route_cache[key] = self.shortest_next_link(source, destination)
        return self.route_cache[key]

    def shortest_next_link(self, source: str, destination: str) -> Optional[Link]:
        if source == destination:
            return None
        node_order = [n["id"] for n in self.problem["nodes"]]
        dist = {node_id: float("inf") for node_id in node_order}
        prev_node: Dict[str, str] = {}
        prev_link: Dict[str, Link] = {}
        unsettled = list(node_order)
        dist[source] = 0.0
        metric = self.problem.get("routingMetric", "latency")
        while unsettled:
            best_i = -1
            best = float("inf")
            for i, node_id in enumerate(unsettled):
                d = dist.get(node_id, float("inf"))
                if d < best:
                    best = d
                    best_i = i
            if best_i < 0 or not math.isfinite(best):
                break
            u = unsettled.pop(best_i)
            if u == destination:
                break
            for link in self.outgoing.get(u, []):
                v = link.spec["to"]
                if v not in unsettled:
                    continue
                nd = best + link_weight(link.spec, metric)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev_node[v] = u
                    prev_link[v] = link
        if destination not in prev_link:
            return None
        cur = destination
        first = prev_link[cur]
        while prev_node.get(cur, source) != source:
            cur = prev_node[cur]
            first = prev_link[cur]
        return first

    def record_invariants(self) -> None:
        for node in self.nodes.values():
            if len(node.queue) > node.queue_limit:
                self.invariant_violations.append(f"{node.node_id}: node queue {len(node.queue)} > {node.queue_limit}")
        for link in self.links.values():
            if len(link.scheduled) > link.queue_limit:
                self.invariant_violations.append(f"{link.link_id}: link queue {len(link.scheduled)} > {link.queue_limit}")

    def record_stats(self) -> None:
        active = self.active_packets()
        self.max_active_packets = max(self.max_active_packets, active)
        sample_every = float(self.problem.get("sampleEveryMs", max(float(self.problem["dtMs"]), 100.0)))
        if self.time_ms + 1e-9 < self.next_sample_at_ms:
            return
        elapsed_ms = max(1.0, self.time_ms + float(self.problem["dtMs"]))
        self.time_series.append({
            "tMs": self.time_ms,
            "generatedPackets": self.generated_packets(),
            "deliveredPackets": len(self.delivered),
            "droppedPackets": len(self.dropped),
            "activePackets": active,
            "nodeQueues": {node.node_id: len(node.queue) for node in self.nodes.values()},
            "linkInFlight": {link.link_id: len(link.scheduled) for link in self.links.values()},
            "linkUtilization": {link.link_id: link.stats(elapsed_ms)["utilization"] for link in self.links.values()},
        })
        self.next_sample_at_ms += sample_every

    def build_result(self) -> Dict[str, Any]:
        latencies = sorted([(p.get("deliveredAtMs", self.time_ms) - p["createdAtMs"]) for p in self.delivered])
        total_delivered_bytes = sum(int(p["sizeBytes"]) for p in self.delivered)
        total_delivered_payload_bytes = sum(int(p["payloadBytes"]) for p in self.delivered)
        total_generated_bytes = sum(int(f["generated"]) * effective_packet_size(f["spec"]) for f in self.flows)
        total_cost = sum(float(p["cost"]) for p in self.delivered) + sum(float(p["cost"]) for p in self.dropped)
        simulated_sec = max(1e-9, float(self.problem["durationMs"]) / 1000.0)
        node_stats = [n.stats(self.time_ms) for n in self.nodes.values()]
        link_stats = [l.stats(self.time_ms) for l in self.links.values()]
        return {
            "generatedPackets": self.generated_packets(),
            "deliveredPackets": len(self.delivered),
            "droppedPackets": len(self.dropped),
            "activePackets": self.active_packets(),
            "maxActivePackets": self.max_active_packets,
            "deliveryRatio": len(self.delivered) / max(1, self.generated_packets()),
            "offeredLoadMbps": total_generated_bytes * 8.0 / simulated_sec / 1e6,
            "throughputMbps": total_delivered_bytes * 8.0 / simulated_sec / 1e6,
            "goodputMbps": total_delivered_payload_bytes * 8.0 / simulated_sec / 1e6,
            "meanLatencyMs": mean(latencies),
            "p95LatencyMs": percentile(latencies, 0.95),
            "totalCost": total_cost,
            "totalSimulatedMs": self.time_ms,
            "routingMetric": self.problem.get("routingMetric", "latency"),
            "flowStats": self.build_flow_stats(),
            "nodeStats": node_stats,
            "linkStats": link_stats,
            "bottlenecks": identify_bottlenecks(node_stats, link_stats),
            "timeSeries": self.time_series,
            "deliveredPacketsTrace": [snapshot(p) for p in self.delivered[:200]],
            "droppedPacketsTrace": [snapshot(p) for p in self.dropped[:200]],
            "invariantViolations": self.invariant_violations,
        }

    def build_flow_stats(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        simulated_sec = max(1e-9, float(self.problem["durationMs"]) / 1000.0)
        for flow in self.flows:
            spec = flow["spec"]
            protocol = protocol_profile(spec.get("protocol"))["protocol"]
            delivered = [p for p in self.delivered if p["flowId"] == spec["id"]]
            dropped = [p for p in self.dropped if p["flowId"] == spec["id"]]
            latencies = sorted([(p.get("deliveredAtMs", self.time_ms) - p["createdAtMs"]) for p in delivered])
            delivered_bytes = sum(int(p["sizeBytes"]) for p in delivered)
            delivered_payload_bytes = sum(int(p["payloadBytes"]) for p in delivered)
            total_cost = sum(float(p["cost"]) for p in delivered) + sum(float(p["cost"]) for p in dropped)
            out.append({
                "id": spec["id"],
                "protocol": protocol,
                "source": spec["source"],
                "destination": spec["destination"],
                "generatedPackets": int(flow["generated"]),
                "deliveredPackets": len(delivered),
                "droppedPackets": len(dropped),
                "deliveryRatio": len(delivered) / max(1, int(flow["generated"])),
                "generatedBytes": int(flow["generated"]) * effective_packet_size(spec),
                "deliveredBytes": delivered_bytes,
                "offeredLoadMbps": int(flow["generated"]) * effective_packet_size(spec) * 8.0 / simulated_sec / 1e6,
                "throughputMbps": delivered_bytes * 8.0 / simulated_sec / 1e6,
                "goodputMbps": delivered_payload_bytes * 8.0 / simulated_sec / 1e6,
                "meanLatencyMs": mean(latencies),
                "p95LatencyMs": percentile(latencies, 0.95),
                "meanTimeInSystemMs": mean(latencies),
                "p95TimeInSystemMs": percentile(latencies, 0.95),
                "totalCost": total_cost,
                "meanCostPerDeliveredPacket": total_cost / max(1, len(delivered)),
            })
        return out


def snapshot(packet: Packet) -> Dict[str, Any]:
    keys = [
        "packetId", "flowId", "protocol", "source", "destination", "payloadBytes",
        "sizeBytes", "createdAtMs", "deliveredAtMs", "droppedAtMs", "dropReason",
        "currentNodeId", "currentLinkId", "hops", "cost",
    ]
    return {k: packet[k] for k in keys if k in packet}


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    return value


def identify_bottlenecks(node_stats: List[Dict[str, Any]], link_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for link in link_stats:
        queue_pressure = float(link["avgInFlight"]) / max(1.0, float(link["queueLimitPackets"]))
        delay_pressure = min(1.0, float(link["meanQueueDelayMs"]) / 1000.0)
        drop_pressure = min(1.0, float(link["droppedPackets"]) / max(1.0, float(link["enqueuedPackets"] + link["droppedPackets"])))
        score = float(link["utilization"]) + queue_pressure + delay_pressure + drop_pressure
        reports.append({
            "id": link["id"],
            "kind": "link",
            "score": score,
            "reason": bottleneck_reason(link["utilization"], link["avgInFlight"], link["maxInFlight"], link["droppedPackets"], link["meanQueueDelayMs"]),
            "utilization": link["utilization"],
            "avgQueue": link["avgInFlight"],
            "maxQueue": link["maxInFlight"],
            "droppedPackets": link["droppedPackets"],
            "meanQueueDelayMs": link["meanQueueDelayMs"],
        })
    for node in node_stats:
        queue_pressure = float(node["avgQueue"]) / max(1.0, float(node["queueLimitPackets"]))
        delay_pressure = min(1.0, float(node["meanQueueDelayMs"]) / 1000.0)
        drop_pressure = min(1.0, float(node["droppedPackets"]) / max(1.0, float(node["receivedPackets"] + node["droppedPackets"])))
        service_pressure = 0.25 if node["forwardedPackets"] > 0 and node["avgQueue"] > 0 else 0.0
        score = queue_pressure + delay_pressure + drop_pressure + service_pressure
        reports.append({
            "id": node["id"],
            "kind": "node",
            "score": score,
            "reason": bottleneck_reason(None, node["avgQueue"], node["maxQueue"], node["droppedPackets"], node["meanQueueDelayMs"]),
            "avgQueue": node["avgQueue"],
            "maxQueue": node["maxQueue"],
            "droppedPackets": node["droppedPackets"],
            "meanQueueDelayMs": node["meanQueueDelayMs"],
        })
    reports = [r for r in reports if r["score"] > 0 or r["droppedPackets"] > 0 or r["maxQueue"] > 0]
    reports.sort(key=lambda r: r["score"], reverse=True)
    return reports[:8]


def bottleneck_reason(utilization: Optional[float], avg_queue: float, max_queue: float, dropped: int, mean_queue_delay_ms: float) -> str:
    if dropped > 0:
        return "drops observed"
    if utilization is not None and utilization >= 0.95:
        return "saturated link"
    if mean_queue_delay_ms >= 10:
        return "queueing delay"
    if avg_queue >= 1 or max_queue >= 10:
        return "queue buildup"
    if utilization is not None and utilization >= 0.75:
        return "high utilization"
    return "capacity pressure"


def builtin_problem(name: str) -> Dict[str, Any]:
    if name in ("small-enterprise", "default"):
        return {
            "nodes": [
                {"id": "client-a", "kind": "host", "forwardingRatePps": 2000, "queueLimitPackets": 256},
                {"id": "client-b", "kind": "host", "forwardingRatePps": 2000, "queueLimitPackets": 256},
                {"id": "edge-1", "kind": "router", "forwardingRatePps": 6000, "queueLimitPackets": 512},
                {"id": "core-1", "kind": "router", "forwardingRatePps": 8000, "queueLimitPackets": 512},
                {"id": "server", "kind": "host", "forwardingRatePps": 4000, "queueLimitPackets": 512},
            ],
            "links": [
                {"id": "client-a-edge", "from": "client-a", "to": "edge-1", "bandwidthMbps": 100, "latencyMs": 1, "costPerMb": 0.001, "queueLimitPackets": 128, "bidirectional": True},
                {"id": "client-b-edge", "from": "client-b", "to": "edge-1", "bandwidthMbps": 50, "latencyMs": 2, "costPerMb": 0.001, "queueLimitPackets": 128, "bidirectional": True},
                {"id": "edge-core", "from": "edge-1", "to": "core-1", "bandwidthMbps": 25, "latencyMs": 8, "costPerMb": 0.004, "queueLimitPackets": 96, "bidirectional": True},
                {"id": "core-server", "from": "core-1", "to": "server", "bandwidthMbps": 100, "latencyMs": 3, "costPerMb": 0.002, "queueLimitPackets": 128, "bidirectional": True},
            ],
            "flows": [
                {"id": "a-to-server", "source": "client-a", "destination": "server", "protocol": "http", "ratePps": 650, "packetSizeBytes": 1200, "maxPackets": 650},
                {"id": "b-to-server", "source": "client-b", "destination": "server", "protocol": "tcp", "ratePps": 300, "packetSizeBytes": 1000, "maxPackets": 300},
            ],
            "durationMs": 1000,
            "dtMs": 1,
            "routingMetric": "latency",
            "drainAfterSourcesMs": 1500,
            "maxPacketsInSystem": 5000,
            "sampleEveryMs": 100,
        }
    if name in ("bottleneck-lab", "bottleneck"):
        return {
            "nodes": [
                {"id": "web-client", "kind": "host", "forwardingRatePps": 6000, "queueLimitPackets": 512},
                {"id": "telemetry-client", "kind": "host", "forwardingRatePps": 6000, "queueLimitPackets": 512},
                {"id": "edge", "kind": "switch", "forwardingRatePps": 12000, "queueLimitPackets": 1024},
                {"id": "wan-router", "kind": "router", "forwardingRatePps": 9000, "queueLimitPackets": 1024},
                {"id": "api-server", "kind": "host", "forwardingRatePps": 9000, "queueLimitPackets": 1024},
            ],
            "links": [
                {"id": "web-edge", "from": "web-client", "to": "edge", "bandwidthMbps": 100, "latencyMs": 1, "costPerMb": 0.001, "queueLimitPackets": 256, "bidirectional": True},
                {"id": "telemetry-edge", "from": "telemetry-client", "to": "edge", "bandwidthMbps": 100, "latencyMs": 1, "costPerMb": 0.001, "queueLimitPackets": 256, "bidirectional": True},
                {"id": "edge-wan", "from": "edge", "to": "wan-router", "bandwidthMbps": 5, "latencyMs": 25, "costPerMb": 0.010, "queueLimitPackets": 96, "bidirectional": True},
                {"id": "wan-api", "from": "wan-router", "to": "api-server", "bandwidthMbps": 100, "latencyMs": 4, "costPerMb": 0.002, "queueLimitPackets": 256, "bidirectional": True},
            ],
            "flows": [
                {"id": "http-api", "source": "web-client", "destination": "api-server", "protocol": "http", "ratePps": 900, "packetSizeBytes": 1100, "maxPackets": 1800},
                {"id": "udp-telemetry", "source": "telemetry-client", "destination": "api-server", "protocol": "udp", "ratePps": 700, "packetSizeBytes": 900, "maxPackets": 1400},
                {"id": "tcp-bulk", "source": "web-client", "destination": "api-server", "protocol": "tcp", "ratePps": 350, "packetSizeBytes": 1400, "maxPackets": 700},
            ],
            "durationMs": 2000,
            "dtMs": 1,
            "routingMetric": "latency",
            "drainAfterSourcesMs": 4000,
            "maxPacketsInSystem": 10000,
            "sampleEveryMs": 100,
        }
    raise ValueError(f"unknown builtin {name!r}")


def load_problem(path: Optional[str], builtin: str) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get("model") == "computer-network":
            params = raw.get("parameters", {})
            if "problem" in params:
                return params["problem"]
            return builtin_problem(params.get("builtin", builtin))
        return raw
    return builtin_problem(builtin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", help="ComputerNetworkProblem JSON or full des/model-spec/v1 JSON")
    parser.add_argument("--builtin", default="bottleneck-lab", choices=["small-enterprise", "default", "bottleneck-lab", "bottleneck"])
    parser.add_argument("--out", default="out/external/computer-network/reference.json")
    args = parser.parse_args()

    start = time.time()
    problem = load_problem(args.problem, args.builtin)
    result = ComputerNetworkReference(problem).run()
    elapsed_ms = (time.time() - start) * 1000.0
    payload = {
        "kernel": "python-computer-network-reference",
        "elapsedMs": elapsed_ms,
        "result": result,
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)
    print(f"wrote {args.out}")
    print(
        "computer-network reference: "
        f"generated={result['generatedPackets']} delivered={result['deliveredPackets']} "
        f"dropped={result['droppedPackets']} top={result['bottlenecks'][0]['kind']}:{result['bottlenecks'][0]['id'] if result['bottlenecks'] else 'none'}"
    )


if __name__ == "__main__":
    main()
