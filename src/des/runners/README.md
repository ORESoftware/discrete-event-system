# Validation runners

Reusable simulation kernels and the drivers that compare them. There are two
families of runners in this folder:

  1. **SEIR runners** — every kernel takes the same `SimConfig`
     (`./types.ts`) and emits the same `RunResult`, so they're plug-
     compatible and Welch-comparable. Used to verify the engine on the
     epidemic model.
  2. **Per-simulation validators** — one driver per top-level `main-X.ts`
     simulation (`convolution`, `backpropagation`, `electric-circuit`,
     `elevator`). Each reads the framework's JSON output and the
     corresponding external reference's JSON output and reports
     deviation. See the bottom of this file for those.

## Kernels

| Kernel                | File                            | Semantics |
|-----------------------|---------------------------------|-----------|
| Framework (3-queue)   | `framework-runner.ts`           | Fixed-step DES; per-station processor with input/processing/output queues. |
| FEL-fifo (M/M/1)      | `fel-runner.ts` (`service: 'fifo'`)       | Classical FEL kernel, single-server FIFO at each station. |
| FEL-individual (M/M/inf) | `fel-runner.ts` (`service: 'individual'`) | Classical FEL kernel, per-individual exit clock at each station. |
| PerIndividual         | `per-individual-runner.ts`      | Fixed-step DES with the new `PerIndividualProcessor`: single queue, per-entity remaining-time clocks (M/M/inf semantic). |
| Gillespie SSA         | `gillespie-runner.ts`           | Compartment-level event-driven stochastic SSA (direct method). |
| ODE RK4               | `ode-runner.ts`                 | Deterministic mean-field RK4 integration of the SEIR ODE. |
| Difference equation   | `difference-runner.ts`          | Forward-Euler difference equation (discrete-time analog of the ODE). Also exports `analyticalSteadyState()` and `maxStableStep()`. |

## Drivers

| Driver                      | What it does |
|-----------------------------|--------------|
| `replicate.ts`              | N=30 reps of framework vs FEL-fifo, Welch t-tests on splits and populations. Demonstrates the granularity bias of the original framework processor. |
| `stepsize-sweep.ts`         | Sweeps the framework over `stepSize` ∈ {1.0, 0.5, 0.1, 0.05} vs FEL-fifo. Emits ASCII plot + CSV + SVG showing the gap shrinking as step size goes to zero. |
| `per-individual-vs-fel.ts`  | Verifies `PerIndividualProcessor` matches FEL-individual (M/M/inf) within Welch noise. Also runs a convergence sweep showing PI -> FEL agreement as `stepSize` -> 0. |
| `validate-references.ts`    | Four-way validation: PI, FEL-individual, Gillespie SSA, ODE RK4 all on the same model. |
| `validate-with-externals.ts`| Same as above but also folds in JSON results from external Python tools (SimPy, Ciw) under `out/external/`. See `../../../external-references/README.md`. |
| `validate-external-fel-models.ts` | Cross-checks non-epidemic DES examples by writing one JSON spec per scenario, running it internally, and passing the same JSON to source-only external FEL references. |
| `steady-state.ts`           | Mathematical verification: closed-form analytical $N^*$ vs forward-Euler difference equation, ODE RK4, Gillespie SSA, and FEL-individual run as open systems. Also demonstrates the forward-Euler stability bound $\Delta t < 2\,\min_c \mu_c$. See `MATH.md` for the derivation. |

## Running

After `npm run build`:

```bash
# In-repo only
N=30 node dist/des/runners/replicate.js
node dist/des/runners/stepsize-sweep.js
N=30 node dist/des/runners/per-individual-vs-fel.js
N=20 STEPSIZE=0.05 node dist/des/runners/validate-references.js

# Bring in external libraries (SimPy, Ciw) for second-opinion validation
bash ../../external-references/run-all.sh   # produces out/external/<tool>/*.json
N=20 STEPSIZE=0.05 node dist/des/runners/validate-with-externals.js

# Feed identical JSON specs to internal DES and source-only external FEL models.
node dist/des/runners/validate-external-fel-models.js

# Verify the model mathematically: closed-form steady state vs every kernel.
N=5 HORIZON=10000 node dist/des/runners/steady-state.js
```

Environment variables honored by most drivers: `N` (number of reps),
`STEPSIZE` (step size for fixed-step kernels). All drivers seed PRNGs
deterministically per-rep so runs are reproducible.

## Output

| Driver                       | Artifacts |
|------------------------------|-----------|
| `replicate.ts`               | stdout only |
| `stepsize-sweep.ts`          | `out/stepsize-sweep.csv`, `out/stepsize-sweep.svg`, stdout |
| `per-individual-vs-fel.ts`   | stdout only |
| `validate-references.ts`     | stdout only |
| `validate-with-externals.ts` | stdout only (reads `out/external/*/*.json`) |
| `validate-external-fel-models.ts` | `out/external-fel/<scenario>/input.json`, external JSON payloads, stdout |
| `steady-state.ts`            | stdout only |

## Math reference

`MATH.md` next to this file derives the continuous-time ODE, the forward-Euler
difference equation, and the closed-form steady-state populations from
scratch, including the stability bound on $\Delta t$ and the mean-lifespan
derivation. `difference-runner.ts` and `steady-state.ts` are the executable
companions to that document.

### External difference-equation / ODE references

For independent verification of the math, `external-references/` ships
scripts that implement the same model in three other tools:

| Tool      | Lang   | Solver         | Verifies |
|-----------|--------|----------------|----------|
| scipy-ode | Python | LSODA + sympy + numpy.linalg | symbolic $f_S = \lambda/q$, numerical closed form, difference equation, ODE |
| octave    | Octave | lsode + `\`    | closed form via mldivide, difference equation, ODE |
| r-desolve | R      | deSolve::lsoda + `solve()` | closed form, difference equation, ODE |

To run any/all of them:

```bash
# Default: pick up python3, octave, Rscript from $PATH; skip what's missing.
bash external-references/run-all.sh

# Override interpreter locations via env var:
PYTHON_BIN=/opt/py311/bin/python OCTAVE_BIN=octave-cli RSCRIPT_BIN=Rscript-4.3 \
  bash external-references/run-all.sh

# Then read their JSON results back into the comparison.
node dist/des/runners/validate-with-externals.js
```

Tools whose interpreter isn't on `$PATH` (or whose Python/R packages are
missing) are skipped with a friendly message — the script never fails.
See `external-references/README.md` for full env-var docs.

## Queue correctness

The framework's `EntityProcessor` and `PerIndividualProcessor` both store
pending entities in `@oresoftware/linked-queue`. If the queue had any
FIFO bias, every result computed here would inherit it. The tests in
`../test/queue-bias-test.ts` verify the queue is bias-free across the
operations the framework actually uses:

| Test | Verifies |
|------|----------|
| T1   | 100k pure enqueue/dequeue preserves FIFO order |
| T2   | Random interleaved enqueue/dequeue under load preserves FIFO order |
| T3   | `remove(k)` does not perturb the relative order of any other items |
| T4   | `getRandomKey()` is uniform (chi-square test, $\alpha = 0.001$, df=99) |
| T5   | Equal enqueue+dequeue leaves no leftover entries (no Map leak) |
| T6   | head/tail/size/lookup invariants hold throughout 200k random ops |
| T7   | for-of and `reverseIterator()` yield insertion order |
| T8   | `addToFront` produces correct LIFO front |

Run with `node dist/des/test/queue-bias-test.js`. All 21 assertions pass
(latest: chi-square = 98.43 vs critical 148.2 at df=99 — textbook uniform).

## Per-simulation validators

In addition to the SEIR runners above, the repo ships four standalone
simulations under `src/des/main-*.ts`. Each has a one-shot validator
that compares its output to a corresponding external reference (Python
or scipy/SimPy/numpy). The validators live in this folder.

| Simulation        | Framework entry point                 | Validator                              | External reference                                  |
|-------------------|---------------------------------------|----------------------------------------|-----------------------------------------------------|
| Convolution       | `dist/des/main-convolution.js`        | `validate-convolution.js`              | `external-references/convolution/conv.py` (numpy)   |
| Backpropagation   | `dist/des/main-backpropagation.js`    | `validate-backpropagation.js`          | `external-references/backpropagation/bp.py` (naive py) |
| Electric circuit  | `dist/des/main-electric-circuit.js`   | `validate-electric-circuit.js`         | `external-references/electric-circuit/circuit.py` (analytical + scipy LSODA) |
| Elevator          | `dist/des/main-elevator.js`           | `validate-elevator.js`                 | `external-references/elevator/elevator.py` (SimPy)  |
| Court MDP (USACC) | `dist/des/main-court-mdp.js`          | `validate-court-mdp.js`                | `external-references/court-mdp/court_mdp.py` (Python value iteration) |
| Two-disease       | `dist/des/main-two-disease.js`        | `validate-two-disease.js`              | `external-references/two-disease/two_disease.py` (LSODA + Gillespie SSA) |
| Computer network  | `dist/des/main-computer-network.js`   | `validate-computer-network.js`         | `external-references/computer-network/network_reference.py` (source-only Python) |
| IP/MIP solver graph | `dist/des/main-ip-mip-des.js`       | `validate-ip-mip-external.js`          | `external-references/ip-mip/ip_mip_reference.py` (source-only Python, optional SciPy MILP) |
| External FEL suite | JSON specs written by validator      | `validate-external-fel-models.js`      | `external-references/computer-network/network_fel_reference.py`, `external-references/traffic/fel_traffic_reference.py` |

Typical workflow for any of them:

```bash
npm run build
node dist/des/main-<sim>.js                       # writes out/<sim>-framework.json
bash external-references/run-all.sh               # writes out/external/<sim>/*.json
node dist/des/runners/validate-<sim>.js           # reports max-abs-error / aggregates
```

`validate-ip-mip-external.ts` writes small bounded IP/MIP scenario files
itself and invokes the registered `ip-mip-reference` solver module
directly, so it does not require `run-all.sh`.

Each validator's expected agreement bound:

| Simulation        | Bound                                                                  | Achieved (current run) |
|-------------------|------------------------------------------------------------------------|------------------------|
| Convolution       | max-abs error < 1e-12 vs `numpy.convolve`                              | 1.1e-16 (~ 1 ULP)      |
| Backpropagation   | max-abs error < 1e-12 on every weight after 10 000 SGD steps            | 3.6e-15 (~16 ULPs)     |
| Electric circuit  | forward-Euler order ≈ 1.0 (theoretical); err < 5e-3 vs scipy at dt=1e-3| order 1.01, 1.85e-3    |
| Elevator          | aggregate (mean wait / travel / total) within 10% of SimPy             | mean wait Δ ≈ 0.22 s   |
| Court MDP         | max V* diff = 0 vs Python; identical π* on all 864 states               | bit-exact match        |
| Two-disease       | ∫populations within tiered tolerances vs LSODA + SSA; Welch p > 0.01 on final D | Welch p = 0.606, all tolerances pass |
| IP/MIP solver graph | objective match vs external exact enumeration; both incumbents feasible | 15/15 checks pass      |

## MDP / control systems

Two MDP-driven control system demonstrations live alongside the SEIR
runners:

### USACC court MDP

`main-court-mdp.ts` models the [US Anti-Corruption Court Project's MDP
spec](https://oresoftware.github.io/us-anti-corruption-court-project/mdp)
as a 4-stage station graph. Cases (moving entities) carry a fully-
observable state vector across 864 product states; each station applies
a `Policy` (interface in `main-court-mdp.ts`) to choose one of 8 actions;
the action induces a stochastic transition. Four policies ship:

  * `RejectAllPolicy`         — closes every case immediately (baseline).
  * `AlwaysEscalatePolicy`    — pushes every case to trial.
  * `NaiveThresholdPolicy`    — hand-tuned per-factor heuristic.
  * `OptimalPolicy`           — `argmax` action from value iteration.

Value iteration is implemented in `mdp/value-iteration.ts` and runs in
~25 ms (64 sweeps to ‖ΔV‖∞ ≤ 1e-9). The Python reference at
`external-references/court-mdp/court_mdp.py` runs the same value
iteration; `validate-court-mdp.ts` confirms bit-exact V* and identical
π* on all 864 states.

Running:

```bash
npm run build
node dist/des/main-court-mdp.js                # solves MDP, runs all 4 policies
bash external-references/run-all.sh            # runs Python value iteration
node dist/des/runners/validate-court-mdp.js    # asserts V* / π* match
```

Latest run on `CASES=5000 SEED=42` shows the optimal policy beating
the naive threshold by 2.3× on mean reward (140.6 vs 61.6), validating
that the framework can host a non-trivial control problem.

### Elevator dispatch MDP

The elevator simulation now ships **three** dispatch policies:

  * `uncoordinated`      — every car runs SCAN/LOOK independently.
  * `coordinated`        — each car queries a `Coordinator` before
    picking its next target, and skips any (floor, direction) pair
    already claimed by another car. Implements the user-stated rules
    "if a car is already going to a floor, don't send a second" and
    "skip a floor going up so other cars can stop at that floor" as
    one rule: myopic minimisation of expected redundant stops.
  * `coordinated-pickup` — same as coordinated, plus opportunistic
    mid-flight pit-stops: a moving elevator can pick up an unclaimed
    floor call along its path, **iff** it is not full and the call
    direction matches its current direction. Requires the
    `Coordinator` to track which elevator claimed each call (not just
    whether one did) so the new pit-stop can claim safely.

`compare-elevator-dispatch.ts` sweeps seeds × arrival-rates and reports
the gap. Latest sweep (5 seeds, λ ∈ {0.1, 0.2, 0.3, 0.4}/s, 1-hour sims,
12 floors), `coordinated-pickup` vs `uncoordinated`:

```
  λ=0.10:  meanWait −9.6%   p95Wait −13.4%   meanTotal  −5.7%
  λ=0.20:  meanWait −33.4%  p95Wait −38.0%   meanTotal −22.0%
  λ=0.30:  meanWait −47.5%  p95Wait −51.0%   meanTotal −34.5%
  λ=0.40:  meanWait −51.2%  p95Wait −55.4%   meanTotal −41.4%
```

(The plain `coordinated` mode falls between these and uncoordinated.)
The benefit grows with load: at heavy load, opportunistic pickups cut
the worst-case (p95) wait by ~55%. The framework's SimPy aggregate
match (within 10%) is preserved by the uncoordinated baseline; the
coordinated and coordinated-pickup modes are strict improvements on top.

The elevator architecture is also stress-tested by
`src/des/test/elevator-invariants-test.ts`, which checks 7 invariant
families (conservation, capacity, position bounds, state-machine
consistency, timestamp monotonicity, floor-queue direction, coordinator
exclusivity) every tick across 45 configurations. PASS on 45/45.

Running:

```bash
npm run build
node dist/des/main-elevator.js                                 # both modes side by side
SEEDS=1,2,3,4,5 LAMBDAS=0.1,0.2,0.3,0.4 SIM_T=3600 \
  node dist/des/runners/compare-elevator-dispatch.js           # full sweep
```
