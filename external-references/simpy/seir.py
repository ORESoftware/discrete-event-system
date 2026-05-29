#!/usr/bin/env python3
"""SimPy reference implementation of our SEIR-with-hospitalization model.

Process-oriented FEL semantics: every patient is a SimPy Process whose
lifetime is a sequence of `env.timeout(uniform(a, b))` waits with branching
decisions in between. SimPy's scheduler maintains the global future-event
list internally, so this is a straight FEL kernel with per-entity exit
clocks - the same semantic class as our FEL-individual / PI / Gillespie
kernels, just from a peer-reviewed off-the-shelf library.

Output schema matches `RunResult` in src/des/runners/types.ts.
"""

import argparse
import json
import os
import random
import sys
import time

import simpy

DEFAULT_CONFIG = {
    "horizonDays": 1200,
    "phase1Days":  800,
    "sourceCap":   500,
    "arrivalsInterarrival": [0.7, 1.3],
    "residence": {
        "S":   [0.20, 0.40], "E":   [0.20, 0.40], "I-P": [0.20, 0.40],
        "I-A": [0.20, 0.40], "I-S": [0.20, 0.40], "I-H": [0.20, 0.40],
        "R":   [1.50, 2.50], "D":   [0.10, 0.30],
    },
    "probabilities": {
        "asymptomaticShare":           0.40,
        "hospitalizationGivenSymptom": 0.20,
        "caseFatalityGivenHospital":   0.12,
    },
}

COMPARTMENT_ORDER = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R"]
MATRIX_ROWS = ["__source__", "S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D"]
MATRIX_COLS = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D", "main-sink"]


def run_simulation(seed: int, config=DEFAULT_CONFIG):
    rng = random.Random(seed)
    env = simpy.Environment()

    # State maintained as plain dicts to keep the running-mean math obvious.
    pops      = {c: 0 for c in COMPARTMENT_ORDER}
    pop_sums  = {c: 0.0 for c in COMPARTMENT_ORDER}
    peak      = {c: 0 for c in COMPARTMENT_ORDER}
    transitions = {}
    counters  = {"created": 0, "absorbed": 0, "last_t": 0.0}

    def record_transition(frm: str, to: str):
        row = transitions.setdefault(frm, {})
        row[to] = row.get(to, 0) + 1

    def tick_pop_integral(now: float):
        dt = now - counters["last_t"]
        if dt <= 0:
            return
        for c in COMPARTMENT_ORDER:
            pop_sums[c] += pops[c] * dt
        counters["last_t"] = now

    def add_to(c: str, now: float):
        tick_pop_integral(now)
        pops[c] += 1
        if pops[c] > peak[c]:
            peak[c] = pops[c]

    def remove_from(c: str, now: float):
        tick_pop_integral(now)
        pops[c] -= 1

    def draw_uniform(a: float, b: float) -> float:
        return a + rng.random() * (b - a)

    def patient(env, eid: int):
        # Source -> S.
        prev = "__source__"
        current = "S"
        record_transition(prev, current)
        add_to(current, env.now)

        while True:
            # Stay at the current compartment for U(a, b) days.
            a, b = config["residence"][current]
            yield env.timeout(draw_uniform(a, b))

            # Decide where to go next based on the topology.
            p = config["probabilities"]
            if current == "S":
                next_ = "E"
            elif current == "E":
                next_ = "I-P"
            elif current == "I-P":
                next_ = "I-A" if rng.random() < p["asymptomaticShare"] else "I-S"
            elif current == "I-A":
                next_ = "R"
            elif current == "I-S":
                next_ = "I-H" if rng.random() < p["hospitalizationGivenSymptom"] else "R"
            elif current == "I-H":
                next_ = "D" if rng.random() < p["caseFatalityGivenHospital"] else "R"
            elif current == "R":
                next_ = "S"
            elif current == "D":
                # D is a brief residence before main-sink, mirroring the
                # framework topology so transition counts and splits line up.
                record_transition("D", "main-sink")
                remove_from_d = current
                # leave D
                # Do not add to compartment dict (D not tracked).
                counters["absorbed"] += 1
                return
            else:
                return

            now = env.now
            remove_from(current, now)
            record_transition(current, next_)

            if next_ in COMPARTMENT_ORDER:
                add_to(next_, now)
                current = next_
                continue

            if next_ == "D":
                # Move into D for a brief residence; we don't count D in pops.
                a, b = config["residence"]["D"]
                yield env.timeout(draw_uniform(a, b))
                record_transition("D", "main-sink")
                counters["absorbed"] += 1
                return

            return

    def source(env):
        while True:
            a, b = config["arrivalsInterarrival"]
            yield env.timeout(draw_uniform(a, b))
            if env.now >= config["phase1Days"]:
                return
            if counters["created"] >= config["sourceCap"]:
                return
            counters["created"] += 1
            env.process(patient(env, counters["created"]))

    env.process(source(env))
    env.run(until=config["horizonDays"])
    tick_pop_integral(config["horizonDays"])  # close out the integral

    counts = {}
    splits = {}
    for r in MATRIX_ROWS:
        row = transitions.get(r, {})
        total = sum(row.get(c, 0) for c in MATRIX_COLS)
        counts[r] = {c: row.get(c, 0) for c in MATRIX_COLS}
        splits[r] = {c: (row.get(c, 0) / total) if total > 0 else 0
                     for c in MATRIX_COLS}

    time_avg = {c: pop_sums[c] / config["horizonDays"] for c in COMPARTMENT_ORDER}
    final_pop = dict(pops)

    return {
        "kernel":             "simpy",
        "seed":               seed,
        "totals":             {"created":  counters["created"],
                               "absorbed": counters["absorbed"]},
        "finalPopulations":   final_pop,
        "transitionCounts":   counts,
        "splitProbs":         splits,
        "timeAvgPopulations": time_avg,
        "peakPopulations":    peak,
    }


def main():
    parser = argparse.ArgumentParser(description="SimPy SEIR reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  required=True,
                        help="Path to write JSON result")
    args = parser.parse_args()

    t0 = time.time()
    result = run_simulation(args.seed)
    result["elapsedMs"] = (time.time() - t0) * 1000.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"simpy seed={args.seed}  -> {args.out}  ({result['elapsedMs']:.1f} ms)")


if __name__ == "__main__":
    main()
