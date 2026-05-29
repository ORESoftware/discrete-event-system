#!/usr/bin/env python3
"""scipy + numpy + sympy reference implementation of the SEIR-with-
hospitalization model.

Three independent verifications in one script:

  1. SYMBOLIC closed-form steady state.
     sympy solves the 7-equation linear system A @ N* = -b for the symbolic
     N*, then asserts that N_S* simplifies to mu_S * lambda / q, where
     q = (1 - p_a) * p_h * p_d. This proves the algebra in MATH.md is
     correct.

  2. NUMERICAL closed-form steady state.
     numpy.linalg.solve(-A, b) computes the open-system fixed point
     directly. Should match (1) to machine precision.

  3. NUMERICAL forward-Euler difference equation.
     N(t + dt) = N(t) + dt * (A @ N(t) + b(t)).
     Independent reimplementation of difference-runner.ts.

  4. NUMERICAL ODE.
     scipy.integrate.solve_ivp(method='LSODA', rtol=1e-8, atol=1e-10).
     Adaptive, stiffness-aware solver - much higher precision than our
     fixed-step RK4 in ode-runner.ts.

How to run:

    cd courses/hdm-fall-2022/des
    python3 -m venv .venv-external
    source .venv-external/bin/activate
    pip install -r external-references/scipy-ode/requirements.txt
    python3 external-references/scipy-ode/seir.py \
            --out out/external/scipy-ode/standard.json

The default config matches DEFAULT_CONFIG in src/des/runners/types.ts so
the JSON drops straight into validate-with-externals.ts as a column.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


# Mirror DEFAULT_CONFIG in src/des/runners/types.ts
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


def _mu(config):
    """Mean residence times."""
    res = config["residence"]
    return {
        "arrival": (config["arrivalsInterarrival"][0] + config["arrivalsInterarrival"][1]) / 2,
        "S":   (res["S"][0]   + res["S"][1])   / 2,
        "E":   (res["E"][0]   + res["E"][1])   / 2,
        "I-P": (res["I-P"][0] + res["I-P"][1]) / 2,
        "I-A": (res["I-A"][0] + res["I-A"][1]) / 2,
        "I-S": (res["I-S"][0] + res["I-S"][1]) / 2,
        "I-H": (res["I-H"][0] + res["I-H"][1]) / 2,
        "R":   (res["R"][0]   + res["R"][1])   / 2,
        "D":   (res["D"][0]   + res["D"][1])   / 2,
    }


def build_matrix(config):
    """7x7 transition matrix A and constant source vector b for the alive
    compartments [S, E, I-P, I-A, I-S, I-H, R]."""
    mu = _mu(config)
    p  = config["probabilities"]
    p_a, p_h, p_d = p["asymptomaticShare"], p["hospitalizationGivenSymptom"], p["caseFatalityGivenHospital"]

    A = np.zeros((7, 7))
    A[0, 0] = -1 / mu["S"];   A[0, 6] =  1 / mu["R"]                    # S
    A[1, 0] =  1 / mu["S"];   A[1, 1] = -1 / mu["E"]                    # E
    A[2, 1] =  1 / mu["E"];   A[2, 2] = -1 / mu["I-P"]                  # I-P
    A[3, 2] =  p_a       / mu["I-P"]; A[3, 3] = -1 / mu["I-A"]          # I-A
    A[4, 2] = (1 - p_a)  / mu["I-P"]; A[4, 4] = -1 / mu["I-S"]          # I-S
    A[5, 4] =  p_h       / mu["I-S"]; A[5, 5] = -1 / mu["I-H"]          # I-H
    A[6, 3] =  1 / mu["I-A"]
    A[6, 4] = (1 - p_h)  / mu["I-S"]
    A[6, 5] = (1 - p_d)  / mu["I-H"]
    A[6, 6] = -1 / mu["R"]                                              # R

    return A, mu, p


def lambda_src(t: float, C: float, config) -> float:
    mu = _mu(config)
    return 1 / mu["arrival"] if (C < config["sourceCap"] and t < config["phase1Days"]) else 0.0


# -----------------------------------------------------------------------------
# 1. Symbolic verification with sympy
#
# Strategy: substitute the *claimed* closed-form  N*_c = mu_c * f_c  with
# f_S = lambda / q  into the 7 fixed-point equations, then sympy.simplify each
# residual. If all 7 simplify to 0, the claim is proven. This is
# substantially faster than sp.solve() on the symbolic linear system, which
# searches an exponential space and hangs on systems this size.
# -----------------------------------------------------------------------------
def symbolic_verification():
    lam, mu_S, mu_E, mu_IP, mu_IA, mu_IS, mu_IH, mu_R = sp.symbols(
        "lambda mu_S mu_E mu_IP mu_IA mu_IS mu_IH mu_R", positive=True)
    p_a, p_h, p_d = sp.symbols("p_a p_h p_d", positive=True)

    q   = (1 - p_a) * p_h * p_d
    f_S = lam / q

    # Claim: N*_c = mu_c * f_c, where the f_c's are determined by the topology.
    NS  = mu_S  * f_S
    NE  = mu_E  * f_S
    NIP = mu_IP * f_S
    NIA = mu_IA * p_a       * f_S
    NIS = mu_IS * (1 - p_a) * f_S
    NIH = mu_IH * p_h       * (1 - p_a) * f_S
    NR  = mu_R  * (1 - q)   * f_S

    # Residuals of the seven fixed-point equations. All must simplify to 0.
    residuals = [
        lam + NR / mu_R - NS / mu_S,
        NS / mu_S - NE / mu_E,
        NE / mu_E - NIP / mu_IP,
        p_a       * NIP / mu_IP - NIA / mu_IA,
        (1 - p_a) * NIP / mu_IP - NIS / mu_IS,
        p_h       * NIS / mu_IS - NIH / mu_IH,
        NIA / mu_IA + (1 - p_h) * NIS / mu_IS + (1 - p_d) * NIH / mu_IH - NR / mu_R,
    ]
    simplified = [sp.simplify(r) for r in residuals]
    all_zero   = all(r == 0 for r in simplified)

    labels = ["S-balance", "E-balance", "IP-balance", "IA-balance",
              "IS-balance", "IH-balance", "R-balance"]
    return {
        "fS_equals_lambda_over_q":  all_zero,
        "qDefinition":              "(1 - p_a) * p_h * p_d",
        "fS_formula":               "lambda / q",
        "fixedPointResiduals":      {l: str(r) for l, r in zip(labels, simplified)},
    }


# -----------------------------------------------------------------------------
# 2. Numerical closed-form steady state (open system, lambda const)
# -----------------------------------------------------------------------------
def closed_form_steady_state(config):
    open_cfg = {**config, "sourceCap": float("inf"), "phase1Days": float("inf")}
    A, mu, p = build_matrix(open_cfg)
    lam = 1 / mu["arrival"]
    b = np.array([lam, 0, 0, 0, 0, 0, 0])
    N_star = np.linalg.solve(-A, b)
    q   = (1 - p["asymptomaticShare"]) * p["hospitalizationGivenSymptom"] * p["caseFatalityGivenHospital"]
    return dict(zip(COMPARTMENT_ORDER, N_star.tolist())), {
        "lambda": lam, "q": q, "f_S": lam / q,
        "totalAlive": float(N_star.sum()),
    }


# -----------------------------------------------------------------------------
# 3. Forward-Euler difference equation (numpy reimplementation of difference-runner.ts)
# -----------------------------------------------------------------------------
def difference_solve(config, dt: float):
    A, mu, p = build_matrix(config)
    lam = 1 / mu["arrival"]
    p_d = p["caseFatalityGivenHospital"]

    n_steps = max(1, int(round(config["horizonDays"] / dt)))
    N = np.zeros(7)
    C = 0.0
    deaths = 0.0
    pop_sums = np.zeros(7)
    peak     = np.zeros(7)
    diverged = False

    for i in range(n_steps):
        pop_sums += N * dt
        t   = i * dt
        src = lam if (C < config["sourceCap"] and t < config["phase1Days"]) else 0.0
        b_t = np.array([src, 0, 0, 0, 0, 0, 0])

        dN  = A @ N + b_t
        deaths += N[5] * p_d / mu["I-H"] * dt
        N += dt * dN
        C += dt * src
        peak = np.maximum(peak, N)

        if not np.all(np.isfinite(N)):
            diverged = True
            break

    return {
        "diverged":            diverged,
        "finalPopulations":    dict(zip(COMPARTMENT_ORDER, N.tolist())),
        "timeAvgPopulations":  dict(zip(COMPARTMENT_ORDER, (pop_sums / config["horizonDays"]).tolist())),
        "peakPopulations":     dict(zip(COMPARTMENT_ORDER, peak.tolist())),
        "totals": {"created": float(C), "absorbed": float(deaths)},
        "dt": dt,
    }


# -----------------------------------------------------------------------------
# 4. ODE via scipy.integrate.solve_ivp (LSODA, adaptive, stiffness-aware)
#
# The source emits at rate lambda until it has emitted sourceCap entities
# (deterministically t_off = sourceCap / lambda for the ODE) or t reaches
# phase1Days, whichever is first. Adaptive solvers struggle with this
# discontinuity, so we integrate in two SMOOTH pieces:
#
#   phase 1  [0, t_off]   : src = lambda  (constant)
#   phase 2  [t_off,  T]  : src = 0       (drain phase)
#
# and pass the final state of phase 1 as the initial state of phase 2.
# -----------------------------------------------------------------------------
def ode_solve(config):
    A, mu, p = build_matrix(config)
    lam = 1 / mu["arrival"]
    p_d = p["caseFatalityGivenHospital"]
    T   = config["horizonDays"]

    if np.isfinite(config["sourceCap"]) and np.isfinite(config["phase1Days"]):
        t_off = min(config["phase1Days"], config["sourceCap"] / lam)
    elif np.isfinite(config["sourceCap"]):
        t_off = config["sourceCap"] / lam
    elif np.isfinite(config["phase1Days"]):
        t_off = config["phase1Days"]
    else:
        t_off = T  # source runs the entire horizon (open system)
    t_off = float(min(t_off, T))

    def make_rhs(src_value):
        def rhs(t, y):
            N = y[:7]
            b_t = np.array([src_value, 0, 0, 0, 0, 0, 0])
            dN  = A @ N + b_t
            dC  = src_value
            dD  = N[5] * p_d / mu["I-H"]
            return np.concatenate([dN, [dC, dD]])
        return rhs

    integrate_kwargs = dict(method="LSODA", rtol=1e-8, atol=1e-10)
    pieces = []

    if t_off > 0:
        # phase 1: constant source
        t_eval_1 = np.linspace(0, t_off, int(t_off) + 1)
        sol1 = solve_ivp(make_rhs(lam), [0, t_off], np.zeros(9),
                         t_eval=t_eval_1, **integrate_kwargs)
        if not sol1.success:
            raise RuntimeError(f"LSODA phase-1 failed: {sol1.message}")
        pieces.append(sol1)

    if t_off < T:
        # phase 2: drain
        y_mid = pieces[-1].y[:, -1] if pieces else np.zeros(9)
        t_eval_2 = np.linspace(t_off, T, int(T - t_off) + 1)
        sol2 = solve_ivp(make_rhs(0.0), [t_off, T], y_mid,
                         t_eval=t_eval_2, **integrate_kwargs)
        if not sol2.success:
            raise RuntimeError(f"LSODA phase-2 failed: {sol2.message}")
        # drop the duplicate boundary sample
        if pieces:
            sol2.t = sol2.t[1:]
            sol2.y = sol2.y[:, 1:]
        pieces.append(sol2)

    t_all = np.concatenate([p.t for p in pieces])
    y_all = np.concatenate([p.y for p in pieces], axis=1)

    pops    = y_all[:7, :]
    Nfinal  = pops[:, -1]
    Cfinal  = y_all[7, -1]
    Dfinal  = y_all[8, -1]
    trap    = getattr(np, "trapezoid", np.trapz)
    timeAvg = trap(pops, t_all, axis=1) / T
    peak    = pops.max(axis=1)

    return {
        "finalPopulations":   dict(zip(COMPARTMENT_ORDER, Nfinal.tolist())),
        "timeAvgPopulations": dict(zip(COMPARTMENT_ORDER, timeAvg.tolist())),
        "peakPopulations":    dict(zip(COMPARTMENT_ORDER, peak.tolist())),
        "totals":             {"created": float(Cfinal), "absorbed": float(Dfinal)},
    }


# -----------------------------------------------------------------------------
# Pack everything into a RunResult-shaped JSON
# -----------------------------------------------------------------------------
def build_result(config, dt: float):
    t0 = time.time()

    sym  = symbolic_verification()
    cf_pops, cf_meta = closed_form_steady_state(config)
    diff = difference_solve(config, dt)
    ode  = ode_solve(config)

    elapsed = (time.time() - t0) * 1000.0

    p = config["probabilities"]
    splits = {
        "__source__": {"S": 1.0},
        "S":           {"E": 1.0},
        "E":           {"I-P": 1.0},
        "I-P":         {"I-A": p["asymptomaticShare"],
                        "I-S": 1 - p["asymptomaticShare"]},
        "I-A":         {"R": 1.0},
        "I-S":         {"R": 1 - p["hospitalizationGivenSymptom"],
                        "I-H": p["hospitalizationGivenSymptom"]},
        "I-H":         {"R": 1 - p["caseFatalityGivenHospital"],
                        "D": p["caseFatalityGivenHospital"]},
        "R":           {"S": 1.0},
        "D":           {"main-sink": 1.0},
    }
    counts = {r: {c: splits.get(r, {}).get(c, 0) for c in MATRIX_COLS}
              for r in MATRIX_ROWS}
    splits_out = {r: {c: splits.get(r, {}).get(c, 0) for c in MATRIX_COLS}
                  for r in MATRIX_ROWS}

    # The "main" RunResult uses the LSODA ODE result (highest precision).
    return {
        "kernel":             "scipy-ode",
        "seed":               0,
        "totals":             ode["totals"],
        "finalPopulations":   ode["finalPopulations"],
        "transitionCounts":   counts,
        "splitProbs":         splits_out,
        "timeAvgPopulations": ode["timeAvgPopulations"],
        "peakPopulations":    ode["peakPopulations"],
        "elapsedMs":          elapsed,
        # Extra deterministic checks - these are not used by the standard
        # validate-with-externals.ts comparison columns but are very handy
        # diagnostically.
        "_extras": {
            "symbolic":             sym,
            "closedFormSteadyState": cf_pops,
            "closedFormMeta":        cf_meta,
            "differenceEquation":    diff,
            "configDt":              dt,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="scipy/numpy/sympy SEIR reference")
    parser.add_argument("--seed", type=int, default=0,
                        help="ignored (this kernel is deterministic) but kept for parity")
    parser.add_argument("--out",  required=True, help="Path to write JSON result")
    parser.add_argument("--dt",   type=float, default=0.05,
                        help="step size for the difference-equation check")
    args = parser.parse_args()

    result = build_result(DEFAULT_CONFIG, args.dt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    sym = result["_extras"]["symbolic"]
    print(f"scipy-ode      -> {args.out}  ({result['elapsedMs']:.1f} ms)")
    print(f"  symbolic check f_S = lambda/q : "
          f"{'CONFIRMED' if sym['fS_equals_lambda_over_q'] else 'FAILED'}")
    print(f"  diff-eq diverged at dt={args.dt}: {result['_extras']['differenceEquation']['diverged']}")


if __name__ == "__main__":
    main()
