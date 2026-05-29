#!/usr/bin/env python3
"""External FEL-style reference wrapper for the computer-network DES.

The underlying source-only packet reference mirrors the internal ticked DES,
including explicit future arrivals for link deliveries. This wrapper gives the
generic external-FEL validator a stable module id and payload shape while
reusing the existing audited packet-network reference implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from network_reference import ComputerNetworkReference, json_safe, load_problem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", help="ComputerNetworkProblem JSON or full des/model-spec/v1 JSON")
    parser.add_argument("--builtin", default="bottleneck-lab", choices=["small-enterprise", "default", "bottleneck-lab", "bottleneck"])
    parser.add_argument("--out", default="out/external/computer-network-fel/reference.json")
    args = parser.parse_args()

    start = time.time()
    problem = load_problem(args.problem, args.builtin)
    result = ComputerNetworkReference(problem).run()
    elapsed_ms = (time.time() - start) * 1000.0
    payload = {
        "kernel": "python-computer-network-fel-reference",
        "semantic": "packet-network future event list with periodic source/service ticks matching the internal DES contract",
        "elapsedMs": elapsed_ms,
        "result": result,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    top = result["bottlenecks"][0] if result["bottlenecks"] else None
    top_label = f"{top['kind']}:{top['id']}" if top else "none"
    print(
        "computer-network FEL reference: "
        f"generated={result['generatedPackets']} delivered={result['deliveredPackets']} "
        f"dropped={result['droppedPackets']} top={top_label}"
    )


if __name__ == "__main__":
    main()
