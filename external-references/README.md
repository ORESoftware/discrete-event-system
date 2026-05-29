# External reference kernels

Off-the-shelf libraries running the same SEIR-with-hospitalization model as
our in-repo kernels (framework, FEL-fifo, FEL-individual, PerIndividual,
Gillespie SSA, ODE RK4, difference equation). Their only purpose is
independent corroboration: if a peer-reviewed library agrees with our
results, that's a strong second opinion that our model and our kernels
are correct.

Each tool writes JSON files into `out/external/<tool>/<seed-or-name>.json`
matching the schema our TypeScript drivers consume:

```jsonc
{
  "kernel":            "simpy" | "ciw" | "scipy-ode" | "octave" | "r-desolve" | ...,
  "seed":              42,
  "totals":            { "created": 500, "absorbed": 491 },
  "finalPopulations":  { "S": 0, "E": 1, ... },
  "transitionCounts":  { "S": { "E": 100, ... }, ... },
  "splitProbs":        { "S": { "E": 1.0, ... }, ... },
  "timeAvgPopulations":{ "S": 8.5, "E": 8.6, ... },
  "peakPopulations":   { "S": 50, ... },
  "elapsedMs":         1234.5,

  // Optional: deterministic tools include extra fields used for
  // mathematical verification rather than the standard column comparison.
  "_extras": {
    "symbolic":              { "fS_equals_lambda_over_q": true, ... },
    "closedFormSteadyState": { "S": 20.833, ... },
    "differenceEquation":    { "dt": 0.05, "diverged": false, ... }
  }
}
```

The TS driver `validate-with-externals.ts` reads them and adds each tool
as a new column to the comparison table alongside PI, FEL, Gillespie,
and ODE.

## Sanctioned external module system

External solvers and validators are registered in TypeScript as
`ExternalProgramModule` entries. A module declares:

- a stable module id
- a source file under `external-references/`
- the interpreter env var and PATH fallback, such as `PYTHON_BIN` /
  `python3`
- an argv builder for allowed parameters and output paths

The common runner in `src/des/runners/external-program.ts` verifies that
the source file is inside `external-references/` and invokes the
interpreter with `spawnSync(command, argv, {shell: false})`. This gives us
a sanctioned way to call external programs without checking interpreter
binaries, solver executables, virtualenvs, or generated outputs into git.

```bash
npm run external-modules
npm run external-neural-network
npm run external-computer-network
npm run external-computer-network-fel
npm run external-traffic-fel -- --problem=out/external-fel/smart-traffic-signalized-corridor/input.json
npm run external-traffic-fel -- --problem=out/external-fel-comparison/traffic-shared-input.json
npm run external-traffic-simpy -- --problem=out/external-fel-comparison/traffic-shared-input.json
npm run external-traffic-ciw -- --problem=out/external-fel-comparison/traffic-shared-input.json
npm run external-traffic-sumo -- --problem=out/external/traffic/smart-traffic-sumo-problem.json
npm run compare-external-fel
npm run validate-computer-network
npm run validate-smart-traffic-external
npm run validate-external-fel-models
npm run validate-neural-network
npm run validate-ip-mip-external
```

The generic CLI is available as `npm run external-module -- <module-id>
[--key=value ...]`. For example, after a validator or demo has written a
problem JSON:

```bash
npm run external-module -- ip-mip-reference \
  --problem=out/external/ip-mip/knapsack-4item-problem.json \
  --out=out/external/ip-mip/knapsack-4item-cli-reference.json \
  --solver=brute-force \
  --maxEnumerations=2000000
```

To add another external solver, put its source wrapper under
`external-references/<tool>/`, register it in
`src/des/runners/external-modules.ts`, and consume it from a validator
through `runExternalModule(<id>, params)`.

## Tools shipped

### Stochastic (M reps each — one JSON per seed)

| Tool       | Language | Install                                  | What it implements |
|------------|----------|------------------------------------------|--------------------|
| **SimPy**  | Python   | `pip install -r simpy/requirements.txt`  | Process-oriented FEL with `env.timeout()` for residence draws. The de-facto Python DES library. |
| **Ciw**    | Python   | `pip install -r ciw/requirements.txt`    | Queueing-network DES with `M/M/inf` servers and a routing matrix. |

### Deterministic (one run each — one JSON named `standard.json`)

| Tool        | Language | Install                                       | What it implements |
|-------------|----------|-----------------------------------------------|--------------------|
| **scipy-ode** | Python (numpy + scipy + sympy) | `pip install -r scipy-ode/requirements.txt` | Three checks in one script: symbolic verification of $f_S = \lambda/q$ via sympy, closed-form $N^*$ via `numpy.linalg.solve(-A, b)`, forward-Euler difference equation, and ODE via `scipy.integrate.solve_ivp(method='LSODA', rtol=1e-8, atol=1e-10)`. |
| **octave**    | GNU Octave | `apt install octave` / `brew install octave` | Closed-form via `\` (mldivide), forward-Euler difference equation, ODE via `lsode`. Uses different solver internals from scipy. |
| **r-desolve** | R + deSolve + jsonlite | `Rscript -e 'install.packages(c("deSolve","jsonlite"))'` | Closed-form via `solve(-A, b)`, forward-Euler difference equation, ODE via `deSolve::lsoda`. R is the dominant language for epidemiological modeling, deSolve is its standard ODE library. |

The deterministic tools serve as second-opinion verification of the math
in `src/des/runners/MATH.md` and the in-repo `difference-runner.ts` and
`ode-runner.ts`. Each independently confirms:

- $N^*_c = \mu_c \cdot f_c$ with $f_S = \lambda / q$ (closed form)
- The difference equation diverges iff $\Delta t > 2 \min_c \mu_c$
- LSODA / lsoda / lsode all agree with our internal RK4 to several decimal places

### Source-only module solvers/validators

| Tool | Language | Install | What it implements |
|------|----------|---------|--------------------|
| **computer-network** | Python standard library only | none beyond `python3` | Source-only reference simulator for packet topology, queueing, link serialization, drops, and bottleneck metrics. `validate-computer-network.ts` invokes it without a shell through the external module system. |
| **computer-network/fel** | Python standard library only | none beyond `python3` | FEL-style packet-network wrapper that consumes the same `computer-network` JSON model spec as the internal registry. Used by `validate-external-fel-models.ts`. |
| **neural-network** | Python standard library only | none beyond `python3` | Source-only reference for the neural demos: XOR MLP training, corridor value iteration, and RK4 neural ODE decay. The validator invokes it without a shell through `src/des/runners/external-program.ts`. |
| **ip-mip** | Python standard library; optional SciPy | none beyond `python3`; optional `scipy` for `scipy.optimize.milp` | Source-only IP/MIP reference solver. It brute-force enumerates small bounded all-integer models and can delegate to SciPy MILP when installed. `validate-ip-mip-external.ts` invokes it through the external module system. |
| **traffic/fel** | Python standard library only | none beyond `python3` | Source-only Future Event List traffic reference. It reads both `traffic-flow` / `smart-traffic-flow` model specs and shared source/sink scheduled-trip JSON, schedules cars across lane resources, gates movement by signals when present, and reports aggregate traffic stats for comparison. |
| **traffic/simpy** | Python + SimPy | `pip install -r simpy/requirements.txt` | Optional process-oriented SimPy traffic reference. Each scheduled source/sink trip is a SimPy process with timeout events for departure, lane travel, and signal waits. |
| **traffic/ciw** | Python + Ciw | `pip install -r ciw/requirements.txt` | Optional Ciw queueing-network traffic reference. Each unique scheduled source/sink route is modeled as an infinite-server Ciw node with deterministic sequential arrivals and service times. |
| **traffic/sumo** | Python wrapper plus external SUMO | `sumo` and `netconvert` on PATH, or `SUMO_BIN` / `SUMO_NETCONVERT_BIN` | Optional black-box traffic simulator cross-check. The source wrapper writes SUMO XML from a normalized smart-traffic baseline and parses tripinfo/summary metrics. If SUMO is absent, it returns `unavailable` JSON instead of failing. |

## Quick start

```bash
cd courses/hdm-fall-2022/des

# 1. Install whichever externals you have an interpreter for. The script
#    skips any tool whose interpreter or runtime package is missing.
python3 -m venv .venv-external
source .venv-external/bin/activate
pip install -r external-references/simpy/requirements.txt
pip install -r external-references/ciw/requirements.txt
pip install -r external-references/scipy-ode/requirements.txt

# Optional: GNU Octave
brew install octave                                    # macOS
sudo apt install octave                                # Debian/Ubuntu

# Optional: R + deSolve + jsonlite
Rscript -e 'install.packages(c("deSolve", "jsonlite"))'

# 2. Run every available tool. Skip messages will appear for any that
#    aren't installed.
bash external-references/run-all.sh

# 3. Compare with the in-repo kernels.
node dist/des/runners/validate-with-externals.js
node dist/des/runners/validate-neural-network.js
node dist/des/runners/validate-external-fel-models.js
npm run validate-ip-mip-external
```

## Environment variables

`run-all.sh` discovers each interpreter via an env var defaulting to the
bare command name on `$PATH`. Override per-machine without editing the
script:

| Variable    | Default      | Used by               |
|-------------|--------------|-----------------------|
| `PYTHON_BIN`  | `python3`  | simpy, ciw, scipy-ode, neural-network, computer-network, computer-network/fel, traffic/fel, ip-mip |
| `SUMO_BIN`    | `sumo`     | traffic/sumo optional validator |
| `SUMO_NETCONVERT_BIN` | `netconvert` | traffic/sumo optional validator |
| `OCTAVE_BIN`  | `octave`   | octave                |
| `RSCRIPT_BIN` | `Rscript`  | r-desolve             |
| `N`           | `10`       | Number of stochastic reps per Python tool |
| `BASE_SEED`   | `100000`   | Stochastic reps use seeds `BASE_SEED .. BASE_SEED + N - 1` |

Examples:

```bash
# Use a specific Python interpreter
PYTHON_BIN=/opt/py311/bin/python bash external-references/run-all.sh

# Use Octave's CLI build instead of the GUI launcher
OCTAVE_BIN=octave-cli bash external-references/run-all.sh

# Bigger stochastic runs
N=30 BASE_SEED=200000 bash external-references/run-all.sh
```

## Layout

```
external-references/
  README.md             this file
  run-all.sh            runs every available tool, with env-var executable paths
  simpy/        seir.py + requirements.txt    (stochastic, N reps)
  ciw/          seir.py + requirements.txt    (stochastic, N reps)
  scipy-ode/    seir.py + requirements.txt    (deterministic, single run)
  octave/       seir.m                        (deterministic, single run)
  r-desolve/    seir.R                        (deterministic, single run)
  neural-network/ nn_reference.py             (deterministic, source-only)
  computer-network/ network_reference.py      (deterministic, source-only)
  computer-network/ network_fel_reference.py  (external FEL comparison)
  ip-mip/       ip_mip_reference.py           (solver/validator, source-only)
  traffic/      fel_traffic_reference.py      (source-only FEL comparison)
  traffic/      simpy_traffic_reference.py    (optional SimPy comparison)
  traffic/      ciw_traffic_reference.py      (optional Ciw comparison)
  traffic/      sumo_traffic_reference.py     (optional SUMO wrapper)
out/external/
  simpy/<seed>.json
  ciw/<seed>.json
  scipy-ode/standard.json
  octave/standard.json
  r-desolve/standard.json
  neural-network/reference.json
  computer-network/reference.json
  computer-network-fel/fel-reference.json
  traffic-fel/fel-reference.json
  ip-mip/<scenario>-problem.json
  ip-mip/<scenario>-reference.json
  traffic/smart-traffic-sumo-problem.json
  traffic/smart-traffic-sumo-reference.json
```

The `out/` directory is gitignored — only the *scripts* live in the repo,
never the binaries or simulation outputs.
