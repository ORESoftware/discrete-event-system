#!/usr/bin/env python3
"""
External reference for the two-disease co-infection model.

Reads the framework's run from `out/two-disease-framework.json` (params and
ensemble means) and implements the same model two ways:

  (1) DETERMINISTIC mean-field ODE solved with scipy.solve_ivp(LSODA).
      Compares against the framework's ENSEMBLE MEAN trajectory.

  (2) STOCHASTIC Gillespie SSA. Same per-person rates the framework uses.
      Compares its ensemble mean to the framework's ensemble mean.

Outputs `out/external/two-disease/python.json` with both reference traces
and final-state aggregates.

ODE (mean-field):
    dS/dt   = -β_A · S · (A + AB) / N - β_B · S · (B + AB) / N
    dA/dt   =  β_A · S · (A + AB) / N - β_B · A · (B + AB) / N - γ_A · A
    dB/dt   =  β_B · S · (B + AB) / N - β_A · B · (A + AB) / N - γ_B · B
    dAB/dt  =  β_B · A · (B + AB) / N + β_A · B · (A + AB) / N - γ_AB · AB
    dR/dt   = γ_A·(1−p_d_A)·A + γ_B·(1−p_d_B)·B + γ_AB·(1−p_d_AB)·AB
    dD/dt   = γ_A·p_d_A·A     + γ_B·p_d_B·B     + γ_AB·p_d_AB·AB

where N = S + A + B + AB + R (alive denominator).
"""

import json
import os
import pathlib
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IN_PATH = ROOT / "out" / "two-disease-framework.json"
OUT_DIR = ROOT / "out" / "external" / "two-disease"
OUT_PATH = OUT_DIR / "python.json"


def two_disease_ode(t, y, params):
    S, A, B, AB, R, D = y
    N = S + A + B + AB + R
    if N <= 0:
        return [0.0] * 6
    bA = params["beta_A"]; bB = params["beta_B"]
    gA = params["gamma_A"]; gB = params["gamma_B"]; gAB = params["gamma_AB"]
    pdA = params["p_death_A"]; pdB = params["p_death_B"]; pdAB = params["p_death_AB"]
    iA = (A + AB) / N
    iB = (B + AB) / N
    dS = -bA * S * iA - bB * S * iB
    dA = bA * S * iA - bB * A * iB - gA * A
    dB = bB * S * iB - bA * B * iA - gB * B
    dAB = bB * A * iB + bA * B * iA - gAB * AB
    dR = gA * (1 - pdA) * A + gB * (1 - pdB) * B + gAB * (1 - pdAB) * AB
    dD = gA *      pdA  * A + gB *      pdB  * B + gAB *      pdAB  * AB
    return [dS, dA, dB, dAB, dR, dD]


def run_ode(params, t_eval):
    y0 = [
        params["N"] - params["initialA"] - params["initialB"] - params["initialAB"],
        params["initialA"], params["initialB"], params["initialAB"], 0.0, 0.0,
    ]
    sol = solve_ivp(
        fun=lambda t, y: two_disease_ode(t, y, params),
        t_span=(0.0, params["simT"]),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-9, atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"LSODA failed: {sol.message}")
    return {
        "t":  sol.t.tolist(),
        "S":  sol.y[0].tolist(),
        "A":  sol.y[1].tolist(),
        "B":  sol.y[2].tolist(),
        "AB": sol.y[3].tolist(),
        "R":  sol.y[4].tolist(),
        "D":  sol.y[5].tolist(),
    }


def run_gillespie(params, seed, t_grid):
    """
    Per-person Gillespie SSA. Each person has a state in {S, A, B, AB, R, D}.
    Reactions and rates (per the model spec):
       S → A   : β_A · I_A / N      per S person
       S → B   : β_B · I_B / N      per S person
       A → AB  : β_B · I_B / N      per A person
       B → AB  : β_A · I_A / N      per B person
       A → R   : γ_A · (1−p_d_A)    per A person
       A → D   : γ_A · p_d_A        per A person
       B → R   : γ_B · (1−p_d_B)    per B person
       B → D   : γ_B · p_d_B        per B person
       AB → R  : γ_AB · (1−p_d_AB)  per AB person
       AB → D  : γ_AB · p_d_AB      per AB person

    Total rate Λ at any moment is sum-over-reactions of (per-person rate × #persons).
    Inter-event time ~ Exp(Λ); next reaction chosen with prob proportional to
    its contribution. We record (S, A, B, AB, R, D) at every t_grid point.
    """
    rng = np.random.default_rng(seed)
    N = params["N"]
    counts = {
        "S":  N - params["initialA"] - params["initialB"] - params["initialAB"],
        "A":  params["initialA"], "B":  params["initialB"], "AB": params["initialAB"],
        "R":  0, "D":  0,
    }
    bA = params["beta_A"]; bB = params["beta_B"]
    gA = params["gamma_A"]; gB = params["gamma_B"]; gAB = params["gamma_AB"]
    pdA = params["p_death_A"]; pdB = params["p_death_B"]; pdAB = params["p_death_AB"]
    t = 0.0
    grid_idx = 0
    out = {"t": [], "S": [], "A": [], "B": [], "AB": [], "R": [], "D": []}
    def record_at(time):
        nonlocal grid_idx
        while grid_idx < len(t_grid) and t_grid[grid_idx] <= time:
            out["t"].append(t_grid[grid_idx])
            for k in ["S", "A", "B", "AB", "R", "D"]:
                out[k].append(counts[k])
            grid_idx += 1
    record_at(t)
    sim_T = params["simT"]
    while t < sim_T and grid_idx < len(t_grid):
        alive = counts["S"] + counts["A"] + counts["B"] + counts["AB"] + counts["R"]
        if alive == 0:
            break
        I_A = (counts["A"] + counts["AB"]) / alive
        I_B = (counts["B"] + counts["AB"]) / alive
        rates = [
            ("S", "A",  counts["S"]  * bA * I_A),
            ("S", "B",  counts["S"]  * bB * I_B),
            ("A", "AB", counts["A"]  * bB * I_B),
            ("B", "AB", counts["B"]  * bA * I_A),
            ("A", "R",  counts["A"]  * gA * (1 - pdA)),
            ("A", "D",  counts["A"]  * gA * pdA),
            ("B", "R",  counts["B"]  * gB * (1 - pdB)),
            ("B", "D",  counts["B"]  * gB * pdB),
            ("AB","R",  counts["AB"] * gAB * (1 - pdAB)),
            ("AB","D",  counts["AB"] * gAB * pdAB),
        ]
        Lambda = sum(r for (_, _, r) in rates)
        if Lambda <= 0:
            break
        dt = rng.exponential(1.0 / Lambda)
        if t + dt > sim_T:
            t = sim_T; break
        t += dt
        u = rng.uniform(0, Lambda)
        cum = 0.0
        chosen = None
        for r in rates:
            cum += r[2]
            if u <= cum:
                chosen = r; break
        if chosen is None:
            chosen = rates[-1]
        src, dst, _ = chosen
        counts[src] -= 1
        counts[dst] += 1
        record_at(t)
    # Pad remaining grid points with the final state.
    while grid_idx < len(t_grid):
        out["t"].append(t_grid[grid_idx])
        for k in ["S", "A", "B", "AB", "R", "D"]:
            out[k].append(counts[k])
        grid_idx += 1
    return out


def main():
    if not IN_PATH.exists():
        print(f"[two-disease] missing {IN_PATH}; run main-two-disease.js first")
        return 1
    with IN_PATH.open() as f:
        framework = json.load(f)
    params = framework["params"]
    t_grid = framework["meanTrace"]["t"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[two-disease] ODE LSODA  ...", end=" ", flush=True)
    ode = run_ode(params, t_grid)
    print("done")

    reps = int(framework.get("reps", 30))
    print(f"[two-disease] Gillespie SSA ({reps} reps)  ...", end=" ", flush=True)
    ssa_traces = []
    for r in range(reps):
        ssa_traces.append(run_gillespie(params, params["seed"] + 1000 + r, t_grid))
    # Mean trajectory.
    T = len(t_grid)
    ssa_mean = {"t": list(t_grid), "S": [0.0]*T, "A": [0.0]*T, "B": [0.0]*T,
                "AB": [0.0]*T, "R": [0.0]*T, "D": [0.0]*T}
    for tr in ssa_traces:
        for k in ["S", "A", "B", "AB", "R", "D"]:
            for i in range(T):
                ssa_mean[k][i] += tr[k][i] / reps
    final_D = [tr["D"][-1] for tr in ssa_traces]
    final_R = [tr["R"][-1] for tr in ssa_traces]
    print("done")

    out = {
        "tool": "scipy LSODA + Gillespie SSA",
        "params": params,
        "ode": ode,
        "ssa_mean": ssa_mean,
        "ssa_final_D_mean": float(np.mean(final_D)),
        "ssa_final_D_std":  float(np.std(final_D, ddof=1) if len(final_D) > 1 else 0.0),
        "ssa_final_R_mean": float(np.mean(final_R)),
        "ssa_final_R_std":  float(np.std(final_R, ddof=1) if len(final_R) > 1 else 0.0),
        "ssa_reps": reps,
    }
    with OUT_PATH.open("w") as f:
        json.dump(out, f)
    print(f"[two-disease] wrote {OUT_PATH}")

    # Print final state for sanity.
    print(f"[two-disease] ODE final D = {ode['D'][-1]:.2f}, R = {ode['R'][-1]:.2f}")
    print(f"[two-disease] SSA final D = {np.mean(final_D):.2f} ± {np.std(final_D, ddof=1):.2f}")
    print(f"[two-disease] SSA final R = {np.mean(final_R):.2f} ± {np.std(final_R, ddof=1):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
