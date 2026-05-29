#!/usr/bin/env bash
# =============================================================================
# Run every available external reference kernel and dump JSON results to
# out/external/<tool>/<seed-or-name>.json.
#
# Tools fall into two classes:
#   stochastic    SimPy, Ciw       -> N reps each (env: N, BASE_SEED)
#   deterministic scipy-ode,       -> 1 run each, file named "standard.json"
#                 octave, r-desolve
#
# Each interpreter is found via an env var (default: bare command on $PATH):
#
#     PYTHON_BIN=python3        (override: PYTHON_BIN=/opt/py311/bin/python)
#     OCTAVE_BIN=octave         (override: OCTAVE_BIN=octave-cli)
#     RSCRIPT_BIN=Rscript       (override: RSCRIPT_BIN=Rscript-4.3)
#
# Tools whose interpreter is not on $PATH (or whose Python module / R
# package is not importable) are SKIPPED with a friendly message; the
# script does not fail. Run from courses/hdm-fall-2022/des or invoke with
# absolute path - it cd's to its own parent automatically.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# Stochastic-rep counts. Override with `N=30 BASE_SEED=200000 ./run-all.sh`.
N="${N:-10}"
BASE_SEED="${BASE_SEED:-100000}"

# Interpreter locations. Default to the bare command name; bash will resolve
# via $PATH at exec time. Override per-machine with the env vars above.
: "${PYTHON_BIN:=python3}"
: "${OCTAVE_BIN:=octave}"
: "${RSCRIPT_BIN:=Rscript}"

OUT_ROOT="out/external"
mkdir -p "$OUT_ROOT"

have_command () { command -v "$1" >/dev/null 2>&1; }
have_python_module () { "$PYTHON_BIN" -c "import $1" 2>/dev/null; }
have_r_package      () { "$RSCRIPT_BIN" -e "suppressMessages(library($1))" 2>/dev/null; }

# -----------------------------------------------------------------------------
# Stochastic kernels: N reps, deterministic per-seed.
# -----------------------------------------------------------------------------
run_stochastic_python () {
  local tool=$1 script=$2 module=$3
  if ! have_command "$PYTHON_BIN" || ! have_python_module "$module"; then
    echo "skipping $tool ($PYTHON_BIN -m $module not importable; pip install -r external-references/$tool/requirements.txt)" >&2
    return 0
  fi
  echo "=== $tool : $N stochastic reps ==="
  mkdir -p "$OUT_ROOT/$tool"
  for i in $(seq 0 $((N - 1))); do
    local seed=$((BASE_SEED + i))
    "$PYTHON_BIN" "$script" --seed "$seed" --out "$OUT_ROOT/$tool/$seed.json"
  done
}

# -----------------------------------------------------------------------------
# Deterministic kernels: one run, file named standard.json.
# -----------------------------------------------------------------------------
run_deterministic_python () {
  local tool=$1 script=$2 module=$3
  if ! have_command "$PYTHON_BIN" || ! have_python_module "$module"; then
    echo "skipping $tool ($PYTHON_BIN -m $module not importable; pip install -r external-references/$tool/requirements.txt)" >&2
    return 0
  fi
  echo "=== $tool : 1 deterministic run ==="
  mkdir -p "$OUT_ROOT/$tool"
  "$PYTHON_BIN" "$script" --out "$OUT_ROOT/$tool/standard.json"
}

run_deterministic_octave () {
  local tool=$1 script=$2
  if ! have_command "$OCTAVE_BIN"; then
    echo "skipping $tool ($OCTAVE_BIN not on \$PATH; install GNU Octave or set OCTAVE_BIN)" >&2
    return 0
  fi
  echo "=== $tool : 1 deterministic run ==="
  mkdir -p "$OUT_ROOT/$tool"
  "$OCTAVE_BIN" --no-gui --quiet "$script" -- --out "$OUT_ROOT/$tool/standard.json"
}

run_deterministic_r () {
  local tool=$1 script=$2
  if ! have_command "$RSCRIPT_BIN"; then
    echo "skipping $tool ($RSCRIPT_BIN not on \$PATH; install R or set RSCRIPT_BIN)" >&2
    return 0
  fi
  if ! have_r_package deSolve || ! have_r_package jsonlite; then
    echo "skipping $tool (R packages 'deSolve' and/or 'jsonlite' missing; run: $RSCRIPT_BIN -e 'install.packages(c(\"deSolve\", \"jsonlite\"))')" >&2
    return 0
  fi
  echo "=== $tool : 1 deterministic run ==="
  mkdir -p "$OUT_ROOT/$tool"
  "$RSCRIPT_BIN" "$script" --out "$OUT_ROOT/$tool/standard.json"
}

run_stochastic_python    simpy     external-references/simpy/seir.py     simpy
run_stochastic_python    ciw       external-references/ciw/seir.py       ciw
run_deterministic_python scipy-ode external-references/scipy-ode/seir.py scipy
run_deterministic_octave octave    external-references/octave/seir.m
run_deterministic_r      r-desolve external-references/r-desolve/seir.R

# -----------------------------------------------------------------------------
# Other simulations (convolution, backpropagation, electric circuit, elevator).
# These read their input from out/<sim>-framework.json (produced by the
# corresponding `node dist/des/main-<sim>.js` invocation) and write to
# out/external/<sim>/<tool>.json. They expect the framework run to have
# happened first; missing inputs are reported as a friendly skip.
# -----------------------------------------------------------------------------
run_pullbased_python () {
  local tool=$1 script=$2 module=$3 input=$4
  if ! have_command "$PYTHON_BIN" || ! have_python_module "$module"; then
    echo "skipping $tool ($PYTHON_BIN -m $module not importable; pip install -r external-references/$tool/requirements.txt)" >&2
    return 0
  fi
  if [ ! -f "$input" ]; then
    echo "skipping $tool (input $input not present; run: node dist/des/main-${tool}.js)" >&2
    return 0
  fi
  echo "=== $tool : 1 deterministic run ==="
  "$PYTHON_BIN" "$script"
}

run_pullbased_python convolution     external-references/convolution/conv.py         numpy out/convolution-framework.json
run_pullbased_python backpropagation external-references/backpropagation/bp.py       json  out/backprop-framework.json
run_pullbased_python electric-circuit external-references/electric-circuit/circuit.py scipy out/electric-circuit-framework.json
run_pullbased_python elevator        external-references/elevator/elevator.py        simpy out/elevator-framework.json
run_pullbased_python two-disease     external-references/two-disease/two_disease.py  scipy out/two-disease-framework.json

# court-mdp doesn't need an input file (the value iteration is independent
# of any framework run), but we route it through run_pullbased_python with
# a stable input path so it composes the same way. The "input" is the
# framework dump that contains the configuration the Python script
# implicitly reproduces.
run_pullbased_python court-mdp       external-references/court-mdp/court_mdp.py      json  out/court-mdp-framework.json

# Dependency-free external references. These require only a Python interpreter,
# no pip-installed module. The TypeScript validator can also invoke this script
# directly through src/des/runners/external-program.ts.
run_plain_python () {
  local tool=$1 script=$2
  if ! have_command "$PYTHON_BIN"; then
    echo "skipping $tool ($PYTHON_BIN not on \$PATH; set PYTHON_BIN)" >&2
    return 0
  fi
  echo "=== $tool : 1 deterministic run ==="
  mkdir -p "$OUT_ROOT/$tool"
  "$PYTHON_BIN" "$script" --out "$OUT_ROOT/$tool/reference.json"
}

run_plain_python computer-network external-references/computer-network/network_reference.py
run_plain_python neural-network  external-references/neural-network/nn_reference.py

echo
echo "done. compare with one or more of:"
echo "  node dist/des/runners/validate-with-externals.js"
echo "  node dist/des/runners/validate-convolution.js"
echo "  node dist/des/runners/validate-backpropagation.js"
echo "  node dist/des/runners/validate-electric-circuit.js"
echo "  node dist/des/runners/validate-elevator.js"
echo "  node dist/des/runners/validate-court-mdp.js"
echo "  node dist/des/runners/validate-two-disease.js"
echo "  node dist/des/runners/validate-computer-network.js"
echo "  node dist/des/runners/validate-neural-network.js"
