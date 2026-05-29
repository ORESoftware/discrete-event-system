#!/usr/bin/env python3
"""Ciw reference implementation of our SEIR-with-hospitalization model.

Ciw is a queueing-network DES library. Each compartment is a node with an
arrival process, a service-time distribution, and number_of_servers. Routing
between nodes is a probability matrix. Internally Ciw maintains a global
event list and per-individual records, so this is a process-oriented FEL
kernel from a library with a different API and authorship from SimPy.

We use number_of_servers=inf to get M/M/inf semantics (every individual
served concurrently with its own clock), matching FEL-individual / PI /
Gillespie / SimPy.

Output schema matches `RunResult` in src/des/runners/types.ts.
"""

import argparse
import json
import os
import time

import ciw

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

NODE_ORDER = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D"]
COMPARTMENT_ORDER = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R"]
MATRIX_ROWS = ["__source__", "S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D"]
MATRIX_COLS = ["S", "E", "I-P", "I-A", "I-S", "I-H", "R", "D", "main-sink"]


def run_simulation(seed: int, config=DEFAULT_CONFIG):
    p = config["probabilities"]
    res = config["residence"]
    arr = config["arrivalsInterarrival"]

    arrival_distributions = [
        ciw.dists.Uniform(arr[0], arr[1]),  # S
        None,                               # E
        None,                               # I-P
        None,                               # I-A
        None,                               # I-S
        None,                               # I-H
        None,                               # R
        None,                               # D
    ]
    service_distributions = [
        ciw.dists.Uniform(*res["S"]),
        ciw.dists.Uniform(*res["E"]),
        ciw.dists.Uniform(*res["I-P"]),
        ciw.dists.Uniform(*res["I-A"]),
        ciw.dists.Uniform(*res["I-S"]),
        ciw.dists.Uniform(*res["I-H"]),
        ciw.dists.Uniform(*res["R"]),
        ciw.dists.Uniform(*res["D"]),
    ]

    # Routing matrix: row i = "P(go from node i to node j)". Probabilities
    # sum to <= 1; the leftover is the probability of leaving the system
    # (exit, which we treat as "main-sink"). All values must be floats.
    asymp = p["asymptomaticShare"]
    hosp  = p["hospitalizationGivenSymptom"]
    cfr   = p["caseFatalityGivenHospital"]

    R = [
        # to:  S    E    I-P  I-A  I-S  I-H  R    D
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                  # S -> E
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],                  # E -> I-P
        [0.0, 0.0, 0.0, asymp, 1.0 - asymp, 0.0, 0.0, 0.0],        # I-P -> I-A | I-S
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],                  # I-A -> R
        [0.0, 0.0, 0.0, 0.0, 0.0, hosp, 1.0 - hosp, 0.0],          # I-S -> I-H | R
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - cfr, cfr],            # I-H -> R | D
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                  # R -> S
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                  # D -> exit (main-sink)
    ]

    number_of_servers = [float("inf")] * len(NODE_ORDER)

    N = ciw.create_network(
        arrival_distributions=arrival_distributions,
        service_distributions=service_distributions,
        routing=R,
        number_of_servers=number_of_servers,
    )

    ciw.seed(seed)
    sim = ciw.Simulation(N)
    sim.simulate_until_max_time(config["horizonDays"])

    records = sim.get_all_records()

    # Reconstruct transition counts and time-averaged populations from the
    # per-record log. Each Record has node (1-indexed), arrival_date,
    # exit_date, destination (next node, or -1 for exit).
    transitions = {}
    pop_sums = {c: 0.0 for c in COMPARTMENT_ORDER}
    horizon = config["horizonDays"]

    by_individual = {}
    for r in records:
        by_individual.setdefault(r.id_number, []).append(r)

    sourceCap = config["sourceCap"]
    created_total = len(by_individual)
    if created_total > sourceCap:
        # Trim to first sourceCap arrivals to mirror the framework's source
        # cutoff. In practice with mean iat=1.0 day and cap=500 the source
        # naturally finishes well before phase1=800 anyway.
        keep_ids = sorted(by_individual.keys())[:sourceCap]
        by_individual = {k: by_individual[k] for k in keep_ids}
        created_total = sourceCap

    absorbed = 0
    for iid, recs in by_individual.items():
        recs.sort(key=lambda r: r.arrival_date)
        prev_node = "__source__"
        for r in recs:
            node_name = NODE_ORDER[r.node - 1]
            transitions.setdefault(prev_node, {})[node_name] = (
                transitions.setdefault(prev_node, {}).get(node_name, 0) + 1
            )
            if node_name in COMPARTMENT_ORDER:
                a = max(0.0, r.arrival_date)
                e = min(horizon,
                        r.exit_date if r.exit_date and r.exit_date > 0
                        else horizon)
                if e > a:
                    pop_sums[node_name] += e - a
            prev_node = node_name
        last = recs[-1]
        if last.destination == -1:  # exited the system after this service
            transitions.setdefault(prev_node, {})["main-sink"] = (
                transitions.setdefault(prev_node, {}).get("main-sink", 0) + 1
            )
            absorbed += 1

    counts = {}
    splits = {}
    for r in MATRIX_ROWS:
        row = transitions.get(r, {})
        total = sum(row.get(c, 0) for c in MATRIX_COLS)
        counts[r] = {c: row.get(c, 0) for c in MATRIX_COLS}
        splits[r] = {c: (row.get(c, 0) / total) if total > 0 else 0
                     for c in MATRIX_COLS}

    time_avg = {c: pop_sums[c] / horizon for c in COMPARTMENT_ORDER}

    final_pop = {c: 0 for c in COMPARTMENT_ORDER}
    for iid, recs in by_individual.items():
        for r in recs:
            node_name = NODE_ORDER[r.node - 1]
            if node_name not in COMPARTMENT_ORDER:
                continue
            if r.arrival_date <= horizon and (r.exit_date is None
                                              or r.exit_date > horizon
                                              or r.exit_date < 0):
                final_pop[node_name] += 1

    peak = {c: 0 for c in COMPARTMENT_ORDER}  # not directly observable

    return {
        "kernel":             "ciw",
        "seed":               seed,
        "totals":             {"created":  created_total, "absorbed": absorbed},
        "finalPopulations":   final_pop,
        "transitionCounts":   counts,
        "splitProbs":         splits,
        "timeAvgPopulations": time_avg,
        "peakPopulations":    peak,
    }


def main():
    parser = argparse.ArgumentParser(description="Ciw SEIR reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  required=True)
    args = parser.parse_args()

    t0 = time.time()
    result = run_simulation(args.seed)
    result["elapsedMs"] = (time.time() - t0) * 1000.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"ciw   seed={args.seed}  -> {args.out}  ({result['elapsedMs']:.1f} ms)")


if __name__ == "__main__":
    main()
