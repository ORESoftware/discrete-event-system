#!/usr/bin/env python3
"""Source-only external reference solver for small IP/MIP instances.

No solver binary is vendored here.  The TypeScript side invokes this file
through the sanctioned external-program registry.

Solvers:
  - brute-force: dependency-free exact enumeration for bounded integer models.
  - scipy-milp: optional scipy.optimize.milp bridge when SciPy is installed.
  - auto: try scipy-milp, then fall back to brute-force when possible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def extract_problem(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept raw IPMIPProblem or a des/model-spec/v1 wrapper."""
    if payload.get("$schema") == "des/model-spec/v1":
        params = payload.get("parameters", {})
        if "raw" in params:
            return params["raw"]
        if "knapsack" in params:
            k = params["knapsack"]
            values = list(k["values"])
            weights = list(k["weights"])
            return {
                "sense": "max",
                "c": values,
                "A": [weights],
                "b": [k["capacity"]],
                "integerVars": [True for _ in values],
                "ub": [1 for _ in values],
                "varNames": [f"item_{i}" for i in range(len(values))],
                "conNames": ["capacity"],
            }
        raise ValueError("model spec must provide parameters.raw or parameters.knapsack")
    return payload


def finite_ub(problem: Dict[str, Any], j: int) -> Optional[float]:
    ub = problem.get("ub")
    if ub is None:
        return None
    val = ub[j]
    if val is None:
        return None
    val = float(val)
    if math.isfinite(val):
        return val
    return None


def objective(problem: Dict[str, Any], x: List[float]) -> float:
    return sum(float(c) * float(v) for c, v in zip(problem["c"], x))


def feasible(problem: Dict[str, Any], x: List[float], tol: float = 1e-9) -> bool:
    for j, v in enumerate(x):
        if v < -tol:
            return False
        ub = finite_ub(problem, j)
        if ub is not None and v > ub + tol:
            return False
        if problem["integerVars"][j] and abs(v - round(v)) > tol:
            return False
    for row, rhs in zip(problem["A"], problem["b"]):
        lhs = sum(float(a) * float(v) for a, v in zip(row, x))
        if lhs > float(rhs) + tol:
            return False
    return True


def solve_bruteforce(problem: Dict[str, Any], max_enumerations: int) -> Dict[str, Any]:
    n = len(problem["c"])
    if any(not bool(v) for v in problem["integerVars"]):
        return {
            "status": "unavailable",
            "solver": "python-bruteforce",
            "message": "brute-force reference requires all variables integer",
        }
    domains: List[range] = []
    total = 1
    for j in range(n):
        ub = finite_ub(problem, j)
        if ub is None:
            return {
                "status": "unavailable",
                "solver": "python-bruteforce",
                "message": f"variable {j} has no finite upper bound",
            }
        hi = math.floor(ub + 1e-9)
        if hi < 0:
            return {"status": "infeasible", "solver": "python-bruteforce", "x": [], "objective": None}
        domains.append(range(0, hi + 1))
        total *= hi + 1
        if total > max_enumerations:
            return {
                "status": "unavailable",
                "solver": "python-bruteforce",
                "message": f"enumeration size {total} exceeds max {max_enumerations}",
            }

    best_x: Optional[List[float]] = None
    best_z: Optional[float] = None
    sense = problem["sense"]
    checked = 0
    for xs in itertools.product(*domains):
        checked += 1
        x = [float(v) for v in xs]
        if not feasible(problem, x):
            continue
        z = objective(problem, x)
        if best_z is None or (sense == "max" and z > best_z + 1e-12) or (sense == "min" and z < best_z - 1e-12):
            best_z = z
            best_x = x

    if best_x is None:
        return {
            "status": "infeasible",
            "solver": "python-bruteforce",
            "x": [],
            "objective": None,
            "enumerated": checked,
        }
    return {
        "status": "optimal",
        "solver": "python-bruteforce",
        "x": best_x,
        "objective": best_z,
        "enumerated": checked,
    }


def solve_scipy_milp(problem: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as exc:
        return {
            "status": "unavailable",
            "solver": "scipy-milp",
            "message": f"scipy.optimize.milp unavailable: {type(exc).__name__}: {exc}",
        }

    c = np.array(problem["c"], dtype=float)
    if problem["sense"] == "max":
        c_work = -c
    else:
        c_work = c
    A = np.array(problem["A"], dtype=float)
    b = np.array(problem["b"], dtype=float)
    ub = []
    for j in range(len(c)):
        val = finite_ub(problem, j)
        ub.append(np.inf if val is None else val)
    constraints = LinearConstraint(A, -np.inf * np.ones(len(b)), b)
    integrality = np.array([1 if v else 0 for v in problem["integerVars"]], dtype=int)
    res = milp(c=c_work, integrality=integrality, bounds=Bounds(np.zeros(len(c)), np.array(ub)), constraints=constraints)
    if not res.success:
        status = "infeasible" if res.status == 2 else "unbounded" if res.status == 3 else "numerical-error"
        return {"status": status, "solver": "scipy-milp", "x": [], "objective": None, "message": res.message}
    obj = float(np.dot(c, res.x))
    return {
        "status": "optimal",
        "solver": "scipy-milp",
        "x": [float(v) for v in res.x.tolist()],
        "objective": obj,
        "message": res.message,
        "nit": getattr(res, "nit", None),
    }


def solve(problem: Dict[str, Any], solver: str, max_enumerations: int) -> Dict[str, Any]:
    if solver == "brute-force":
        return solve_bruteforce(problem, max_enumerations)
    if solver == "scipy-milp":
        return solve_scipy_milp(problem)
    if solver == "auto":
        scipy = solve_scipy_milp(problem)
        if scipy["status"] != "unavailable":
            return scipy
        brute = solve_bruteforce(problem, max_enumerations)
        if brute["status"] == "unavailable":
            brute["scipyMessage"] = scipy.get("message")
        return brute
    raise ValueError(f"unknown solver {solver}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, help="Path to raw IPMIPProblem JSON or des/model-spec/v1 wrapper.")
    ap.add_argument("--out", required=True, help="Where to write reference JSON.")
    ap.add_argument("--solver", default="auto", choices=["auto", "brute-force", "scipy-milp"])
    ap.add_argument("--max-enumerations", type=int, default=1_000_000)
    args = ap.parse_args()

    t0 = time.time()
    problem = extract_problem(json.loads(Path(args.problem).read_text()))
    result = solve(problem, args.solver, args.max_enumerations)
    payload = {
        "reference": "external-references/ip-mip/ip_mip_reference.py",
        "requestedSolver": args.solver,
        "elapsedMs": round((time.time() - t0) * 1000, 3),
        "problem": problem,
        "result": result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(json.dumps({"status": result.get("status"), "solver": result.get("solver"), "out": str(out)}))


if __name__ == "__main__":
    main()
