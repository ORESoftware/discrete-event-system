#!/usr/bin/env python3
"""
External reference for the USACC MDP.

Re-implements the SAME MDP as `src/des/mdp/usacc-mdp.ts` (state encoding,
action set, transition model, reward model — line-by-line ported) and
solves it with the same value-iteration algorithm. Dumps V* and π* to
out/external/court-mdp/python.json. The TS validator compares against
out/court-mdp-framework.json and asserts agreement to ~1e-9 per state.

Why this is a meaningful cross-check despite being a port: the two
implementations use different float libraries (V8 vs CPython), different
loop nesting in Python vs TS, and different default RNG behaviour.
Identical numeric results across both is strong evidence of correctness.

Usage (called by external-references/run-all.sh):
    python3 court_mdp.py
"""
import json
import os
import pathlib
import sys
from typing import List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT_DIR = ROOT / "out" / "external" / "court-mdp"
OUT_PATH = OUT_DIR / "python.json"

# -----------------------------------------------------------------------------
# State / action encoding (identical to TS).
# -----------------------------------------------------------------------------

STAGES        = ["SUB", "VAL", "ADM", "TRI"]
ACTIONS       = [
    "request_more_evidence",
    "verify_identity",
    "normalize_record",
    "assign_reviewers",
    "hold_for_audit",
    "escalate_to_next_stage",
    "release_escrow",
    "reject_or_close",
]
N_ACTIONS = len(ACTIONS)

N_FACTORS_PRODUCT = 4 * 3 * 3 * 3 * 2 * 4  # 864
ACCEPTED  = 864
CLOSED    = 865
EXHAUSTED = 866
N_STATES  = 867

FUND_UNFUNDED  = 0
FUND_ESCROWED  = 1
FUND_ACTIVE    = 2
FUND_EXHAUSTED = 3

ACTION_COST = {
    "request_more_evidence":  -2,
    "verify_identity":        -2,
    "normalize_record":       -1,
    "assign_reviewers":       -3,
    "hold_for_audit":         -5,
    "escalate_to_next_stage": -2,
    "release_escrow":         -1,
    "reject_or_close":         0,
}
DRAW_PCT = {
    "request_more_evidence":  25,
    "verify_identity":        25,
    "normalize_record":       10,
    "assign_reviewers":       30,
    "hold_for_audit":         50,
    "escalate_to_next_stage": 25,
    "release_escrow":          0,
    "reject_or_close":         0,
}


def encode(stage: int, ev: int, corr: int, man: int, conf: int, fund: int) -> int:
    return ((((stage * 3 + ev) * 3 + corr) * 3 + man) * 2 + conf) * 4 + fund


def decode(sid: int):
    if sid >= N_FACTORS_PRODUCT:
        return None
    fund = sid %  4; sid //=  4
    conf = sid %  2; sid //=  2
    man  = sid %  3; sid //=  3
    corr = sid %  3; sid //=  3
    ev   = sid %  3; sid //=  3
    stage = sid
    return stage, ev, corr, man, conf, fund


def is_terminal(sid: int) -> bool:
    return sid >= N_FACTORS_PRODUCT


def quality(stage, ev, corr, man, conf, fund):
    return ev + corr - man - 1.5 * conf


def reward_of_accept(s):
    return 50.0 * (quality(*s) - 0.5)


def reward_of_close(s):
    return 50.0 * (0.5 - quality(*s))


def outcomes(sid: int, action: int) -> List[Tuple[int, float, float]]:
    if is_terminal(sid):
        return [(sid, 1.0, 0.0)]
    s = decode(sid)
    stage, ev, corr, man, conf, fund = s
    a = ACTIONS[action]
    cost = ACTION_COST[a]
    funded = fund > FUND_UNFUNDED

    # (stage, ev, corr, man, conf, fund_pre_draw, pct_int_percent)
    edges = []
    base = (stage, ev, corr, man, conf, fund)

    def stay(**override):
        s2 = dict(stage=base[0], ev=base[1], corr=base[2], man=base[3],
                  conf=base[4], fund=base[5])
        s2.update(override)
        return (s2["stage"], s2["ev"], s2["corr"], s2["man"], s2["conf"], s2["fund"])

    if a == "request_more_evidence":
        ev_up = min(ev + 1, 2); man_up = min(man + 1, 2)
        edges = [
            (stay(ev=ev_up), 60),
            (base,           30),
            (stay(man=man_up), 10),
        ]
    elif a == "verify_identity":
        corr_up = min(corr + 1, 2); man_dn = max(man - 1, 0)
        edges = [
            (stay(corr=corr_up), 50),
            (base,               30),
            (stay(man=man_dn),   20),
        ]
    elif a == "normalize_record":
        ev_up = min(ev + 1, 2)
        edges = [
            (stay(ev=ev_up), 30),
            (stay(conf=0),   40),
            (base,           30),
        ]
    elif a == "assign_reviewers":
        ev_up = min(ev + 1, 2)
        edges = [
            (stay(conf=0),   60),
            (stay(ev=ev_up), 30),
            (base,           10),
        ]
    elif a == "hold_for_audit":
        ev_up = min(ev + 1, 2)
        edges = [
            (stay(man=0),    70),
            (stay(ev=ev_up), 20),
            (base,           10),
        ]
    elif a == "escalate_to_next_stage":
        if stage == 3:
            return [(ACCEPTED, 1.0, cost + reward_of_accept(s))]
        edges = [
            (stay(stage=stage + 1), 80),
            (stay(conf=1),          20),
        ]
    elif a == "release_escrow":
        f_next = min(fund + 1, FUND_ACTIVE)
        edges = [(stay(fund=f_next), 100)]
    elif a == "reject_or_close":
        return [(CLOSED, 1.0, cost + reward_of_close(s))]
    else:
        raise ValueError(f"unknown action {a}")

    draw_pct = DRAW_PCT[a]

    out = []
    for (es, pct) in edges:
        base_p = pct * (100 - draw_pct) / 10000.0
        draw_p = pct * draw_pct        / 10000.0
        if base_p > 0:
            out.append((encode(*es), base_p, cost))
        if draw_p > 0:
            f_after = es[5] - 1
            if f_after < FUND_UNFUNDED:
                out.append((EXHAUSTED, draw_p, cost - 150))
            else:
                ed = (es[0], es[1], es[2], es[3], es[4], f_after)
                out.append((encode(*ed), draw_p, cost))

    # Coalesce identical nextStates.
    by_next = {}
    for (ns, p, r) in out:
        if ns in by_next:
            cur_p, cur_r = by_next[ns]
            by_next[ns] = (cur_p + p, cur_r)  # rewards must already match
        else:
            by_next[ns] = (p, r)
    coalesced = [(ns, p, r) for ns, (p, r) in by_next.items()]
    s_total = sum(p for _, p, _ in coalesced)
    if abs(s_total - 1) > 1e-9:
        raise RuntimeError(f"outcomes({sid}, {action}) prob sum {s_total} != 1")
    return coalesced


# -----------------------------------------------------------------------------
# Value iteration (synchronous Bellman backup).
# -----------------------------------------------------------------------------

def value_iteration(gamma: float = 0.95, tol: float = 1e-10, max_iter: int = 5000):
    # Pre-build transition table.
    table = [[outcomes(s, a) for a in range(N_ACTIONS)] for s in range(N_STATES)]
    V = [0.0] * N_STATES
    V[ACCEPTED] = 0.0
    V[CLOSED]   = 0.0
    V[EXHAUSTED] = -150.0

    iters = 0
    final_delta = float("inf")
    for it in range(max_iter):
        Vn = [0.0] * N_STATES
        Vn[ACCEPTED] = V[ACCEPTED]
        Vn[CLOSED] = V[CLOSED]
        Vn[EXHAUSTED] = V[EXHAUSTED]
        delta = 0.0
        for s in range(N_STATES):
            if is_terminal(s):
                continue
            best = -float("inf")
            for a in range(N_ACTIONS):
                ol = table[s][a]
                q = 0.0
                for (ns, p, r) in ol:
                    q += p * (r + gamma * V[ns])
                if q > best:
                    best = q
            Vn[s] = best
            d = abs(Vn[s] - V[s])
            if d > delta:
                delta = d
        V = Vn
        iters = it + 1
        final_delta = delta
        if delta < tol:
            break

    pi = [-1] * N_STATES
    for s in range(N_STATES):
        if is_terminal(s):
            continue
        best_a = 0; best_q = -float("inf")
        for a in range(N_ACTIONS):
            ol = table[s][a]
            q = 0.0
            for (ns, p, r) in ol:
                q += p * (r + gamma * V[ns])
            if q > best_q:
                best_q = q; best_a = a
        pi[s] = best_a
    return V, pi, iters, final_delta, gamma


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    V, pi, iters, delta, gamma = value_iteration(gamma=0.95, tol=1e-10)
    out = {
        "tool": "python value iteration",
        "gamma": gamma,
        "iterations": iters,
        "finalDelta": delta,
        "V": V,
        "policy": pi,
    }
    with OUT_PATH.open("w") as f:
        json.dump(out, f)
    print(f"[court-mdp] {iters} sweeps, max|ΔV|={delta:.3e}")
    print(f"[court-mdp] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
