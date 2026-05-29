# Changelog

## Nonlinear MDP/POMDP forecasting (2026-05-27)

Adds `nonlinear-mdp-pomdp-forecast`, an explicit station graph for nonlinear
prediction and projection:

`ForecastDataSource -> POMDPLatentVariable -> MDPVariableDiscovery -> NonlinearEquationTuning -> ForecastProjection -> ResultSink`

The model uses a POMDP belief filter to infer hidden regime variables, an MDP
value-iteration policy to discover useful observed/lagged/nonlinear/latent
variables, a fine-tuned nonlinear basis equation, and a projection station that
emits forecast intervals. The JSON example writes CSV, summary, frames, and HTML
animation artifacts.

## Applied domain model pack (2026-05-27)

Adds JSON-runnable station-graph models for each requested applied area. All
use the same reusable DES topology:

`ScenarioSource -> CandidateGenerator -> PlanEvaluator -> ResultSink`

The moving entities are `DomainScenarioToken`, `DomainPlanToken`, and
`DomainEvaluationToken`, giving every model a common validation and review
surface while preserving domain-specific objectives.

| Area | Model id |
|---|---|
| Control systems | `adaptive-fuzzy-control` |
| Logistics / transportation | `logistics-routing-heuristics` |
| Manufacturing | `bottleneck-production-control` |
| Supply chain management | `supply-chain-risk-pooling` |
| Operations management | `workforce-service-operations` |
| Financial engineering | `portfolio-drawdown-control` |
| Revenue management | `dynamic-pricing-revenue`, `buyer-aware-dynamic-pricing` |
| Energy | `energy-storage-dispatch` |
| Machine / statistical learning | `active-learning-acquisition` |
| Decision science | `visual-decision-frontier` |

Each model has a corresponding `examples/<model-id>.json` file and CSV/summary
output support. The domain adapter also has a reusable HTML animator for the
pipeline graph; `buyer-aware-dynamic-pricing` adds period traces for price,
inventory, fairness, and retention.

## Iterative Learning Control station graph (2026-05-27)

Adds `iterative-learning-control`, a learning-based control model that keeps
the entity decomposition explicit:

- `ILCTrialSourceStation` emits repeated-trial plans.
- `ILCControllerProgramStation` packages learned feedforward plus feedback
  gains into controller-program movables.
- `ILCPlantTrialStation` simulates the first-order plant for one full trial.
- `ILCLearningUpdateStation` applies the ILC update and loops the next trial
  back into the controller station.
- `ILCResultSinkStation` collects trial summaries for validation and CSV output.

The JSON example is `examples/iterative-learning-control.json`.

## Base-class unification pass (2026-05-26)

This pass pulls two repeated model-local patterns into reusable base-layer
primitives:

- `general/des-base/stateful-token.ts` now exports `PayloadStatefulToken`
  and `StatefulTokenRegistry`, so graph optimizers can share payload-carrying
  movable tokens, state/stateless lineage, transition counting, and per-kind
  token stats. `general/ip-mip-des.ts` now uses the shared token registry
  instead of a solver-local duplicate.
- `general/des-base/episode-accounting.ts` adds scalar and vector episode
  accounting helpers. `EnvironmentStation`, `RLAgentStation`,
  `PolicyGradientAgent`, and `JointEnvStation` now share reward-history,
  length-history, and total-step bookkeeping while preserving their public
  `rewardHistory`, `lengthHistory`, and `totalSteps` surfaces.

Verification:

```
npm run build                         PASS
npm run test-ip-mip-des               39/39
npm run test-optimization-as-des      39/39
npm run test-mdp-adjacent             30/30
full npm test-* sweep                 31/31 scripts
```

## Random tie-breaking for value-based decisions (2026-05-26)

Many value-based algorithms across the codebase shared a hidden
argmax-ordering bias:

```typescript
let bestA = -1, bestQ = -Infinity;
for (let a = 0; a < A; a++) {
  const q = score(a);
  if (q > bestQ) { bestQ = q; bestA = a; }   // strict > = first wins
}
```

When several actions tie for the maximum (initial Q-table is zero, symmetric
MDPs, identical UCT scores on fresh MCTS children, MILP variables with the
same most-fractional score), strict `>` always picks the lowest-index action.
That permanent low-index bias hurts exploration under ε-greedy, slows
convergence on symmetric problems, and makes the algorithm depend on the
order the user happened to enumerate the actions.

### Added

- `general/des-base/argmax.ts` — central random-tie-breaking utility:
  - `argMaxWithTieBreak(values, rng, eps?)` for arrays of scores.
  - `scanArgMaxTieBreak(n, score, rng, eps?)` for lazy scoring functions
    (skips `-Infinity` sentinels for masked actions).
  - `allArgMaxTies(values, eps?)` returns the full tied set.
  - `chooseRandomTied(candidates, rng)` picks uniformly from a tied list.
  - All variants use reservoir-sampling-on-ties to do a single forward pass.

### Applied to

| File | Change |
|---|---|
| `general/value-iteration.ts` | `greedyPolicy()` now uses random tie-break (opt-out via `opts.randomTieBreak = false`). V* is unchanged. |
| `general/des-base/finite-horizon-dp.ts` | Backward induction breaks ties in π_t uniformly at random; V_t is unchanged. |
| `general/des-base/linear-vfa.ts` | `greedyAction()` random-tie-break so a fresh θ=0 agent does not always pick action 0. |
| `general/des-base/semi-mdp.ts` | `pickOption()` greedy fallback uses random tie-break. |
| `general/qlearning-des.ts` | `pickAction()` and `greedyPolicy()` use random tie-break. |
| `general/ppo-des.ts` | `greedyPolicy()` uses random tie-break. |
| `general/mcts.ts` | UCT descent breaks ties uniformly at random (critical at fresh-children depth where `mean=0` and the sqrt term coincides). Final action choice also random on ties. |
| `general/milp-bnb.ts` | `pickBranchVar` breaks fractional-score ties randomly; new `branchSeed` option for reproducible alternate tree shapes. |
| `general/four-rooms.ts` | Greedy evaluation policy uses `scanArgMaxTieBreak` over legal options. |

### NOT applied (rationale documented)

- Numerical methods (calculus, Kalman, MRAC, sliding-mode, feedback
  linearization, MPC QP, RLC, LQR): closed-form / deterministic by design.
- Backpropagation, NN forward pass: strictly sequential.
- LP simplex: Bland's rule is intentionally deterministic for anti-cycling.
- GA / SA: already use RNG inside selection/neighbor proposal.
- Tiger POMDP: rewards are asymmetric, ties are vanishingly rare, and
  changing behavior would perturb existing reference returns.

### Tests

- `test/argmax-tiebreak-test.ts` — 23 checks verifying:
  - Empty / singleton / unique-winner / 5-way-uniform behavior of the
    pure utility (statistical bound on uniformity, within 4σ).
  - `-Infinity` sentinel handling in `scanArgMaxTieBreak`.
  - `eps`-tolerance treats near-equal floats as tied.
  - Value iteration on a 4-action symmetric MDP: deterministic mode
    always picks action 0, random mode hits all four across 20 seeds,
    and V* matches exactly between the two modes.
  - Fresh Q-learning agent (Q=0) hits ≥3 of 5 actions across 30 seeds.
  - MCTS on a degenerate identical-reward env: final action varies
    across seeds.
  - MILP B&B with symmetric structure: optimal z is invariant across
    `branchSeed` values (correctness preserved while branching tree
    can differ).
  - Finite-horizon DP: V_t identical with or without random tie-break;
    π_t varies across seeds when actions are truly tied.

Run with `npm run test-argmax-tiebreak`.

### Reproducibility

Every algorithm uses its own seeded `rng`. Wrapping the whole computation
in `withSeed(N, () => …)` keeps results bit-identical across replications.
This is a behavior-CHANGING update only on truly-tied decisions (where
the previous deterministic choice was arbitrary anyway).

## Feasibility checker pipeline (2026-05-26)

Adds the JSON-runnable `feasibility-pipeline` model for checking a user's
candidate solution against a structured optimization problem and trying to
improve it internally.

Supported first contract:

- Continuous, integer, and binary variables with bounds and optional step sizes.
- Linear objective with `min` or `max` sense.
- Linear `<=`, `>=`, and `=` constraints with tolerances.
- Candidate supplied by variable-name map or vector.
- Optional repair/neighbor improvement loop capped by the shared wall-clock
  checker station.

Core additions:

- `general/feasibility-pipeline.ts` with `CandidateSourceStation`,
  `DomainCheckerStation`, `ConstraintCheckerStation`,
  `ObjectiveEvaluatorStation`, `ImprovementStation`, and
  `FeasibilitySinkStation`.
- `general/adapters/feasibility-pipeline-adapter.ts` with JSON schema, CSV
  trace output, default animation output, and JSONL observability logging.
- Examples:
  `examples/feasibility-pipeline-knapsack.json` and
  `examples/feasibility-pipeline-production.json`.
- Tests: `npm run test-feasibility-pipeline` (16/16).

## Internal solver networks (2026-05-26)

Adds the JSON-runnable `internal-solver-network` model for optimization/search
problems solved entirely inside DES station networks:

| Solver kind | Method |
|---|---|
| `shortest-path` | Incremental Dijkstra or Bellman-Ford waves |
| `knapsack-dp` | Exact 0/1 dynamic programming |
| `knapsack-sa` | Simulated annealing over binary selections |
| `tsp-sa` | TSP simulated annealing using the `SingleStateOptimizer` base |
| `tsp-ga` | TSP genetic algorithm using the `PopulationOptimizer` base |
| `tsp-held-karp` | Exact Held-Karp for small TSP instances |

Core additions:

- `general/internal-solver-network.ts` with solver stations, incumbent
  `SolverSolutionToken`s, `SolutionSinkStation`, and `WallClockCheckerStation`.
- `general/adapters/internal-solver-network-adapter.ts` with JSON schema, CSV
  trace output, default animation output, and JSONL observability logging.
- Examples:
  `examples/internal-solver-knapsack-dp.json`,
  `examples/internal-solver-shortest-path.json`, and
  `examples/internal-solver-tsp-ga.json`.
- Tests: `npm run test-internal-solver-network` (21/21).

The default compute budget is 180000 ms. The checker station emits a stop token
the first time it is ticked after the budget has passed, then the DES runner
halts through its normal stop predicate.

## Universal DES model JSON (2026-05-26)

Adds the portable `des/universal-model/v1` document shape for declaring a full
modeling job:

- `originalInput`: raw LaTeX/XML/JSON/text/manual user input and provenance.
- `math`: normalized variables, equations, conditions, domains, numerics, and
  solver-normalized parameters.
- `des`: stationary entities, moving entity token kinds, graph edges, sources,
  sinks, and observability intent.
- `solver`: registered target model and method/options.
- `runtime`: existing output controls.

New module: `general/universal-model-spec.ts`, with validators, conversion from
math-equation results, and conversion back to a runnable `des/model-spec/v1`
when the target model is supported. `main-from-json` now accepts universal JSON
files directly. New example:
`examples/universal-math-equation-latex-ode.json`. New tests:
`npm run test-universal-model-spec` (11/11).

## Math block diagrams for ODEs/PDEs (2026-05-26)

Adds stationary math blocks backed by `DESStation` and moving `MathSignal`
tokens:

| Model id | Purpose |
|---|---|
| `math-ode-blocks` | Builds ODE systems from integrator stations and RHS expression stations. |
| `math-heat1d-blocks` | Models the 1D heat equation as cell/boundary blocks coupled by Laplacian blocks. |
| `math-equation` | Accepts constrained LaTeX, XML, or structured JSON equation input, normalizes it into a stationary/moving block network, and solves the generated ODE/PDE numerically. |

Core additions:

- `general/math-blocks.ts` with source, sink, algebra, integrator,
  differentiator, filter, comparator, logic, expression, and Laplacian blocks.
- `general/adapters/math-blocks-adapter.ts` with JSON schemas, CSV writers,
  default animation output, and JSONL observability logging.
- Examples: `examples/math-ode-exponential-decay.json` and
  `examples/math-heat1d-blocks.json`, plus `examples/math-equation-latex-ode.json`,
  `examples/math-equation-json-ode.json`, and
  `examples/math-equation-xml-heat1d.json`.
- Tests: `npm run test-math-blocks` (30/30).

## Network flow and traffic flow models (2026-05-26)

Adds three JSON-runnable network models:

| Model id | Purpose |
|---|---|
| `max-flow` | Maximum s-t flow via augmenting-path DES ticks, with min-cut and conservation validators. |
| `traffic-flow` | Continuous-time traffic flow on a stationary five-intersection grid with moving car tokens, signal phases, sources, sinks, jerk-limited acceleration, reaction-time car-following, and sparse one-foot grid-cell stations. |
| `smart-traffic-flow` | Smart-movable traffic flow where each active car owns `runTimeStep()` and is shuffled by `runIterativeDES`; the world station commits car proposals and validates conservation/body-contact separation. |

New core module: `general/network-flow.ts`. New adapter:
`general/adapters/network-flow-adapter.ts`. New examples:
`examples/max-flow-six-node.json` and
`examples/traffic-flow-five-intersection.json`, plus
`examples/smart-traffic-flow-five-intersection.json`.
Traffic car snapshots now expose position, velocity, acceleration, jerk,
target acceleration, leader gap/id, and occupied grid cells. The grid cells
are approximately one-foot square by default (`gridCellSizeM = 0.3048`) and
are materialized sparsely only where cars need local interaction lookup.
`SmartMovable` is now a base participant type for movables that should enter
the iterative runner directly without inheriting from `DESStation`.
`smart-traffic-flow` now samples per-driver `distancePreference` and
`startPreference` traits when cars enter the system, so cars vary in cruising
headway, launch clearance, and startup hesitation instead of all following the
same gap policy. Those traits use a convolved discrete PMF from the random-
variable toolkit rather than a flat uniform draw. It also has optional
behavior-risk accidents: an at-risk
smart car can enter a short speeding, slow-braking, or over-accelerating mode,
but the world emits an accident only when the follower's body reaches the
leader's rear bumper during proposal commit. The event is recorded on the
striking car, struck car, and impacted one-foot grid-cell station, and the
animation shows recent incidents as small flash markers. The default smart
traffic demo now uses `dtSec = 0.1`, and records every tick so frame stepping
moves one tenth of a simulated second at a time.
An optional SUMO cross-check now exports a no-accident smart-traffic baseline
to SUMO XML and compares black-box simulator aggregates when host-provided
`sumo`/`netconvert` binaries are available. The adapter is source-only and
reports `unavailable` cleanly when SUMO is not installed.

Verification:

```
npm run build
npm run test-network-flow   # 87/87
npm run from-json -- examples/max-flow-six-node.json
npm run from-json -- examples/traffic-flow-five-intersection.json
npm run from-json -- examples/smart-traffic-flow-five-intersection.json
npm run validate-smart-traffic-external   # 13/13 with SUMO available; reports unavailable cleanly otherwise
```

## Statistical stochastic optimisation layer (2026-05-26)

Adds five JSON-runnable optimisation models on top of the DES base-class
hierarchy:

| Model id | Purpose |
|---|---|
| `stochastic-lp` | Wraps the existing two-stage SAA vs Benders/L-shaped solver as a registry model with animation/log output. |
| `sddp-capacity` | Multi-stage stochastic capacity expansion via SDDP-style value-function cuts, with an exact sampled-grid DP oracle. |
| `risk-capacity` | Expected-profit, CVaR, chance-constrained, and DRO-lite scenario optimisation over capacity grids. |
| `distribution-fit` | MLE vs method-of-moments fitting for demand/service-time samples, ranked by AIC. |
| `adaptive-simopt` | Adaptive simulation optimisation using UCB-style sequential sampling over candidate policies. |

New core module: `general/statistical-optimization.ts`. New adapter:
`general/adapters/statistical-optimization-adapter.ts`. New examples:
`examples/stochastic-lp-capacity.json`, `examples/sddp-capacity.json`,
`examples/risk-capacity-cvar.json`,
`examples/distribution-fit-service-times.json`, and
`examples/adaptive-simopt-capacity.json`.

The JSON runtime now supports `runtime.animate` (default true for models
with an animator) and `runtime.outputs.log` for JSONL observability.

Verification:

```
npm run build
npm run test-statistical-optimization   # 42/42
npm run test-stochastic-lp              # 44/44
npm run from-json -- examples/sddp-capacity.json
npm run from-json -- examples/risk-capacity-cvar.json
npm run from-json -- examples/adaptive-simopt-capacity.json
npm run from-json -- examples/distribution-fit-service-times.json
npm run from-json -- examples/stochastic-lp-capacity.json
```

## Network mutex with child request tokens (2026-05-26)

Adds a DES primitive for lock-mediated network processing:

| File | Addition |
| ---- | -------- |
| `general/des-base/stateful-token.ts` | Generic stateful/stateless token lineage helpers. Movables can record state transitions, spawn child tokens, preserve parent/root/causation ids, or stay stateless when history is unnecessary. |
| `general/des-base/composite-station.ts` | Generic `CompositeDESStation` for stations that own substations. Outer channels are exposed through explicit input/output ports; child stations run one internal tick per parent tick. |
| `general/network-mutex.ts` | `NetworkMutexWorkerStation` (Station A) is now a composite station with internal queue/request and processor/release substations. `NetworkMutexLockServiceStation` (Station B), source/sink helpers, and child `LockRequestToken` / `LockGrantToken` / `LockReleaseToken` types model the mutex protocol. Work items stay FIFO inside A until the child lock request is granted; A processes under the lock, sends release, then emits the parent item to Station C. |
| `main-network-mutex.ts` | Console demo showing queue buildup, lock wait, child request/release counts, completion order, and trace events. |
| `test/network-mutex-test.ts` | 25 checks covering FIFO completion, child-token wiring, Station A substation structure, parent state-history, lock grant/release accounting, contention queue buildup, invalid release rejection, fast-vs-slow arrival comparison, and generic child lineage helpers. |

Default run:

```
generated = 10
completed = 10
worker max queue = 9
mean queue wait = 25.5 ticks
mean lock wait = 5.7 ticks
mean time in system = 29.5 ticks
lock utilization = 0.984
```

## Sanctioned external smart-traffic SUMO validator (2026-05-26)

Adds a black-box traffic-simulator cross-check for the smart traffic DES
without adding any simulator binary to git:

| File | Addition |
| ---- | -------- |
| `external-references/traffic/sumo_traffic_reference.py` | Source-only Python adapter that writes SUMO nodes/edges/routes/config, calls host `sumo` and `netconvert` only when installed, parses tripinfo/summary XML, and returns `ok`, `unavailable`, or `error` JSON. |
| `runners/external-modules.ts` | Registers `traffic-sumo-reference` as a sanctioned validator module. `SUMO_BIN` and `SUMO_NETCONVERT_BIN` can point to externally installed binaries. |
| `runners/validate-smart-traffic-external.ts` | Runs the TypeScript smart traffic DES in a no-accident baseline, exports normalized demand/network JSON, invokes the SUMO adapter, and compares departures, completion rate, travel-time scale, and collision count when SUMO is available. |

Verification in this environment:

```
npm run external-modules
npm run test-external-modules        15/15
npm run validate-smart-traffic-external  13/13 (SUMO available in this environment)
```

## Sanctioned external computer-network validator (2026-05-26)

Adds a source-only external cross-check for the computer-network DES:

| File | Addition |
| ---- | -------- |
| `external-references/computer-network/network_reference.py` | Dependency-free Python reference simulator for topology routing, node/link queues, link serialization, drops, costs, and bottleneck ranking. No binary or interpreter is vendored. |
| `runners/external-program.ts` / `runners/external-modules.ts` | Sanctioned external module metadata and shell-free `spawnSync` invocation constrained to source files under `external-references/`. |
| `runners/validate-computer-network.ts` | Runs the TypeScript DES and Python reference on both the small-enterprise and bottleneck-lab problems, then compares packet counts, throughput/goodput, latency, costs, flow/link stats, and top bottleneck. |

Commands:

```
npm run external-modules
npm run external-computer-network
npm run validate-computer-network
```

## Sanctioned external IP/MIP solver validator (2026-05-26)

Adds a sanctioned external-solver cross-check for the explicit IP/MIP
station graph without vendoring any solver executable or interpreter:

| File | Addition |
| ---- | -------- |
| `external-references/ip-mip/ip_mip_reference.py` | Source-only Python reference solver. It exact-enumerates small bounded all-integer models with only the standard library and can optionally call `scipy.optimize.milp` when SciPy is installed. |
| `runners/external-modules.ts` | Registers `ip-mip-reference` as a solver module using `PYTHON_BIN` / `python3`, with argv construction for `--problem`, `--out`, `--solver`, and `--max-enumerations`. |
| `runners/validate-ip-mip-external.ts` | Writes bounded IP/MIP scenarios, invokes the registered external module through `external-program.ts`, and compares status, objective value, and incumbent feasibility against `solveIPMIPWithDES`. |
| `package.json` | Adds `npm run validate-ip-mip-external` plus the generic `npm run external-module -- <module-id> [--key=value ...]` entry point. |

Validation:

```
npm run validate-ip-mip-external       15/15
```

## Output routing policies for queueing stations (2026-05-26)

Adds a reusable competitive out-connection policy for station types that
route each completed entity to exactly one downstream acceptor:

| File | Addition |
| ---- | -------- |
| `entity-routing/output-routing-policy.ts` | Shared `OutputConnectionRouter` with `random`, `round-robin`, and `ordered` policies. |
| `entity-processing/processing.ts` | `EntityProcessor` now accepts `outputRouting`; default remains `random`. |
| `entity-processing/per-individual-processor.ts` | `PerIndividualProcessor` now accepts the same `outputRouting` option. |
| `test/output-routing-policy-test.ts` | Verifies declared-order round-robin, intentional ordered priority, and skipping full acceptors. |

Semantics:

```
outputRouting: 'random'       # per-item Fisher-Yates, default
outputRouting: 'round-robin'  # declared connection order, rotating after success
outputRouting: 'ordered'      # declared connection order every time
```

Queues remain FIFO; the policy only controls the competitive choice among
out-connections when an entity leaves a station.

## Explicit IP/MIP solver graph with selectable LP relaxation backend (2026-05-26)

Adds the stronger integer-programming path requested for uploaded
problem sets: an explicit DES station graph, not only a monolithic
branch-and-bound routine.

| File | Addition |
| ---- | -------- |
| `general/ip-mip-des.ts` | New IP/MIP solver graph. Stations handle frontier search, LP relaxation, rounding/repair/local search, incumbent tracking, binary cover cuts, and branch decisions. Movable tokens carry subproblems, relaxations, cuts, candidates, and completions. `lpAlgorithm: "auto"` now builds a `techniquePlan`, chooses a concrete LP backend by problem/node shape, falls back from unavailable external LP bridges, and reports `lpAlgorithmUsage`. |
| `main-ip-mip-des.ts` | Console demo for a 4-item knapsack, defaulting to `LP_ALGO=auto` while still allowing explicit backend selection. |
| `general/adapters/milp-bnb-adapter.ts` | Registers new JSON model `ip-mip-des` beside the existing `milp-bnb` adapter, including the `auto` LP policy. |
| `examples/ip-mip-des-knapsack.json` | JSON-runnable explicit station-graph IP/MIP demo with trace CSV and summary output, using the auto technique selector. |
| `test/ip-mip-des-test.ts` | 31 tests covering station topology, optimal knapsack solve, selectable LP backends, mixed integer/continuous variables, binary cover cuts, auto technique planning, external-root routing, decomposition-candidate detection, malformed input guards, and node limits. |
| `runners/validate-ip-mip-external.ts` | Sanctioned source-only external Python cross-check for objective agreement and incumbent feasibility. |

Supported LP relaxation backends:

```
auto
incremental-primal-dual
des-simplex-dantzig
des-simplex-bland
internal-simplex
external-highs
external-highs-ds
external-highs-ipm
```

Default run:

```
status = optimal
z* = 90
x* = [0, 1, 0, 1]
nodes explored = 4
LP backend = auto
LP usage = incremental-primal-dual: 4
LP iterations = 17
cuts added = 3
candidates tried = 5
```

Validation:

```
npm run build                         PASS
npm run test-ip-mip-des               39/39
npm run validate-ip-mip-external      15/15
npm run ip-mip-des                    PASS
LP_ALGO=des-simplex-dantzig npm run ip-mip-des
node dist/des/main-from-json.js examples/ip-mip-des-knapsack.json
```

## Network flow optimisation and traffic-flow simulation (2026-05-26)

Adds a network-flow family that keeps the DES-as-system thesis broad:

| File | Addition |
| ---- | -------- |
| `general/max-flow.ts` | Maximum flow/min-cut as a fixed-point DES. One Edmonds-Karp augmenting path is one tick; validators check flow conservation and min-cut optimality. |
| `max-flow.ts` | Console driver for the textbook six-node network. |
| `general/stochastic-flow-mdp.ts` | MDP interpretation of max-flow when capacities/availability evolve stochastically. State is `(current node, remaining capacities)`, actions are edge attempts or wait, and Bellman recursion computes the optimal routing policy. |
| `general/traffic-flow.ts` | Five-intersection traffic-flow simulation using stationary grid/intersection/link stations plus moving `TrafficCar` entities. Cars interact through station state: continuous positions, car-following gaps, link capacities, signal phases, and downstream reservations. |
| `main-traffic.ts` | Console driver for the default traffic scenario. |
| `main-stochastic-flow-mdp.ts` | Console driver for the stochastic-flow MDP. |
| `general/adapters/network-flow-adapter.ts` | Registers animated/logged `max-flow`, `traffic-flow`, and `smart-traffic-flow` JSON models. |
| `general/adapters/stochastic-flow-mdp-adapter.ts` | Registers the `stochastic-flow-mdp` JSON model. |
| `examples/max-flow.json` | JSON-runnable max-flow demo with CSV and summary output. |
| `examples/stochastic-flow-mdp.json` | JSON-runnable stochastic-flow-control MDP demo with policy CSV and summary output. |
| `examples/traffic-flow.json` | JSON-runnable traffic-flow demo with link-stat CSV and summary output. |
| `test/network-flow-test.ts` | 83 tests covering max-flow optimum/cut certificates, stochastic-flow MDP Bellman policy, deterministic limit recovery of max-flow, traffic conservation, car cap, gap/capacity invariants, throughput-vs-flow bound, small-dt one-foot grid cells, acceleration/jerk car snapshots, smart-movable traffic actor scheduling, control-fault accident recording, and precondition failures. |

Adds the computer-network focus area:

| File | Addition |
| ---- | -------- |
| `general/computer-network.ts` | Packet-switched networking DES. Hosts, routers, switches, and links are stationary entities; `NetworkPacket` is the movable entity. Links model bandwidth serialization, latency, queue limits, utilization, queueing delay, and cost. Flows support `raw`, `udp`, `tcp`, and `http` protocol profiles. |
| `main-computer-network.ts` | Console driver for the bottleneck lab by default, with `SCENARIO=baseline` for the provisioned small-enterprise topology. |
| `general/adapters/computer-network-adapter.ts` | Registers `computer-network` with the JSON model registry. |
| `examples/computer-network.json` | JSON-runnable topology with nodes, links, flows, CSV output, and summary output. |
| `examples/computer-network-bottleneck.json` | JSON-runnable bottleneck lab where HTTP, UDP, and TCP flows overload a narrow WAN link. |
| `test/computer-network-test.ts` | 40 tests covering delivery, latency/cost stats, class surface, routing metrics, congestion drops, traffic buildup, bottleneck ranking, protocol overhead, JSON registry execution, and bad-input guards. |

Default max-flow run:

```
max flow = 23
augmentations = 3
min-cut capacity = 23
```

Default traffic run:

```
generated cars = 240
completed cars = 240
max active cars = 49
mean travel = 98.4 sec
throughput = 1440 cars/hour
max-flow upper bound = 32 cars/min
```

Default stochastic-flow MDP run:

```
horizon = 8
states = 1620
static max-flow upper bound = 4
optimal expected reward = 2.910049
deterministic limit = 4.000000
```

Default computer-network bottleneck run:

```
generated packets = 3857
delivered packets = 838
dropped packets = 3019
offered load = 21.4659 Mbps
wire throughput = 5.5955 Mbps
goodput = 3.6692 Mbps
mean time in system = 245.15 ms
top bottleneck = link:edge-wan (drops observed)
edge-wan utilization = 0.985
```

Validation:

```
npm run build                 PASS
npm run test-network-flow     68/68
npm run test-computer-network 40/40
npm run max-flow              PASS
npm run stochastic-flow-mdp    PASS
npm run traffic-flow          PASS
npm run computer-network      PASS
node dist/des/main-from-json.js examples/max-flow.json
node dist/des/main-from-json.js examples/stochastic-flow-mdp.json
node dist/des/main-from-json.js examples/traffic-flow.json
node dist/des/main-from-json.js examples/computer-network.json
node dist/des/main-from-json.js examples/computer-network-bottleneck.json
```

## Multi-stage stochastic programming / SDDP and stochastic JSON adapters (2026-05-26)

Adds the natural multi-stage extension of the existing two-stage
stochastic LP work:

| File | Addition |
| ---- | -------- |
| `general/des-base/cut-pool.ts` | Reusable validated affine cut pools for upper/lower envelopes. Intended for SDDP, Benders, outer approximation, DRO, and chance-constraint variants. |
| `general/multistage-stochastic.ts` | Four-stage inventory/storage stochastic program solved by an SDDP-style DES station. One tick performs a sampled forward path plus backward value-function cut generation. |
| `general/adapters/statistical-optimization-adapter.ts` | Registers the animated/logged `stochastic-lp` adapter plus statistical optimisation models. |
| `general/adapters/multistage-sddp-adapter.ts` | Registers `multistage-sddp` for `main-from-json.ts`. |
| `examples/stochastic-lp.json` | First-class JSON wrapper for the existing two-stage SAA/Benders model. |
| `examples/multistage-sddp.json` | JSON-runnable SDDP demo with trace CSV and summary output. |
| `test/multistage-stochastic-test.ts` | 20 tests covering cut-pool validation, stage LP balance constraints, exact scenario-tree solve, SDDP convergence, exported-cut policy replay, and bad-input guards. |

Default `multistage-sddp` run:

```
Exact scenario tree: optimal, z = 112.176312, 30 nodes
SDDP: optimal in 21 iterations
Policy value: 112.176313
Cuts/stage: [22, 22, 22, 22, 1]
```

Validation:

```
npm run build                         PASS
npm run test-multistage-stochastic    20/20
npm run test-stochastic-lp            44/44
node dist/des/main-from-json.js examples/stochastic-lp.json
node dist/des/main-from-json.js examples/multistage-sddp.json
```

## Pre-run preconditions framework (fail-fast input validation) (2026-05-25)

Adds a uniform mechanism for **fail-fast validation of initial conditions
and parameters** across every model in the engine — divide-by-zero
hazards, ill-conditioned matrices, mis-shaped probability vectors,
algorithm-specific reaching/stability conditions, etc.

### Framework: `general/des-base/preconditions.ts`

New module exposing:

* `PreconditionError` — strongly typed error class with `model`, `param`,
  `condition`, and `observed` fields. Message format is
  `<Model>: <param> must <condition>; got <value>`.
* `Preconditions.*` namespace of guard functions:
  `finite`, `positive`, `nonNegative`, `inRange`, `integer`,
  `integerInRange`, `allFinite`, `nonEmpty`, `lengthEq`,
  `arrNonNegative`, `probabilityVector`, `rectangularMatrix`,
  `squareMatrix`, `symmetricMatrix`, `positiveSemidefiniteDiag`,
  `positiveDefiniteCholesky`, `notDivByZero`, `equal`,
  `magnitudeLeq`, `check`.

The `positiveDefiniteCholesky` guard runs an O(n³) Cholesky test that
catches the classic LQR-failure mode `R = 0` (DARE inversion blows up)
and the Kalman-failure mode `R singular` (innovation covariance
inversion blows up).

### Hook on every base class

* `DESStation.assertPreconditions()` — virtual no-op hook on the
  lightweight iterative-algorithm base. Subclasses override to
  fail-fast on bad inputs.
* `PlantBlock.assertPreconditions()`, `ControllerBlock.assertPreconditions()`,
  `EstimatorBlock.assertPreconditions()` — same hook on the heavyweight
  control-blocks hierarchy. Default checks (dt > 0, state finite,
  saturation bounds coherent) inherited; subclasses extend.
* `runIterativeDES` calls `assertPreconditions()` on every station
  before any tick.
* `runClosedLoop` calls it on plant + controller + estimator before
  any tick.

### Algorithm-specific checks added

Constructor / public-runner level:

| Model                       | Key invariant enforced                                                |
| --------------------------- | --------------------------------------------------------------------- |
| `LQRController`             | `R` PD via Cholesky; `Q` PSD; A, B, Q, R rectangular & matched dims; γ ∈ (0, 1] |
| `FiniteHorizonDPStation`    | every transition list is a probability distribution; rewards finite; γ_t ∈ [0, 1] |
| `solveInventoryDP`          | `demandPmf` is a probability vector; non-negative costs; `initialInventory ≤ S_max` |
| `runMountainCar` / `runFourRoomsSMDP` / `runBlackjackMC` / `runStagHunt` / `runActorCriticGridworld` | `numEpisodes ≥ 1`; `α > 0`; `γ ∈ [0, 1]`; `ε ∈ [0, 1]` |
| `simulateTiger`             | `numSteps ≥ 1`; solver name valid; `initialBelief` is a probability vector |
| `runDoubleIntegratorLQR`    | `rU > 0` (R PD); `qPos, qVel ≥ 0`; `dt > 0`; `γ ∈ (0, 1]` |
| `runPontryaginBangBang`     | `uMax > 0`; `dt > 0`; `deadband > 0`; `x0` finite |
| `runRadarTracking` (KF)     | `measNoiseStd > 0` (R singular if zero); `procNoiseStd ≥ 0`; `P0Scale > 0` |
| `runSlidingMode`            | `λ > 0`, `η > 0`, **`η > D` (SMC reaching condition)**, `boundary > 0` |
| `runMRAC`                   | `b > 0` (sign-known), `a_m < 0` (Hurwitz), `γ·dt ≤ 1` (numerical stability) |
| `runFeedbackLinearization`  | `m > 0`, `l > 0` (no divide-by-zero in `1/(m·l²)`); `kp, kv > 0` |
| `runMPCDoubleIntegrator`    | `R > 0` (gradient well-posed); `N ≥ 1`; `Q, Qf ≥ 0`; `uMax > 0` |
| `runTempControl`            | `dt_min > 0`; `duration_h > 0`; non-negative noises and costs |

### New test suite

`test/preconditions-test.ts` (67 checks) exercises:

* Each `Preconditions.*` guard against representative bad and good
  inputs (NaN, ±Inf, 0 where positive required, asymmetric matrix,
  probability vector that doesn't sum to 1, indefinite matrix that
  fails Cholesky, …).
* Each new model's public runner against a representative
  algorithm-specific failure mode (R = 0 in LQR; η ≤ D in SMC; b ≤ 0
  in MRAC; demandPmf invalid in inventory DP; m = 0 in feedback lin.;
  …) — verifies that a `PreconditionError` is raised AND its message
  names the offending parameter.

The JSON CLI (`main-from-json.ts`) also benefits transparently: a bad
parameter now surfaces as
`runPontryaginBangBang: uMax must be > 0 (positive, not zero); got 0`
on stderr instead of producing silent garbage results.

Total test count after this batch: **752 checks across 16 suites**
(was 685/15).

## Six entity-based optimal-control demos (PMP, Kalman, SMC, MRAC, FBL, MPC) (2026-05-25)

This batch extends the optimal-control coverage so every classic
method on the canonical "important optimal-control methods" list is
now demonstrated end-to-end — and crucially, each demo is built on the
**heavyweight `StationaryEntity` + `AbstractMovingEntity` framework**
(plant ⇆ controller blocks talking via `VectorSignal` moving entities),
in contrast to the lightweight `DESStation` pattern used by the RL /
DP / optimisation models.

### New base / framework class

`des-base/control-blocks.ts` introduces the entity-based block-diagram
framework:

| Class                          | What it provides                                                   | Extends                                |
| ------------------------------ | ------------------------------------------------------------------ | -------------------------------------- |
| `VectorSignal`                 | moving entity carrying a `number[]` (measurement / control / xhat) | `SignalValue` (i.e. `AbstractMovingEntity`) |
| `PlantBlock`                   | continuous-state plant block; abstract `dynamics(x, u, dt)`       | `MultiDirectionalSignalEntity`          |
| `ControllerBlock`              | feedback controller block; abstract `controlLaw(y, t)`             | `MultiDirectionalSignalEntity`          |
| `EstimatorBlock`               | observer / Kalman block; abstract `update(y, u)`                  | `MultiDirectionalSignalEntity`          |
| `runClosedLoop(plant, ctrl, …)` | lockstep driver; auto-wires connections                          | —                                      |

### Concrete optimal-control demos (`general/`)

| Leaf model                        | Classic problem                              | Method                                  | JSON id                  |
| --------------------------------- | -------------------------------------------- | --------------------------------------- | ------------------------ |
| `pontryagin-bang-bang.ts`         | time-optimal double integrator               | Pontryagin's Maximum Principle (PMP)    | `pontryagin-bang-bang`   |
| `kalman-filter.ts`                | radar/GPS tracking under sensor noise        | linear Kalman filter (Kalman 1960)      | `kalman-filter`          |
| `sliding-mode-control.ts`         | uncertain plant + bounded matched disturbance| sliding-mode robust control (Utkin 1977)| `sliding-mode`           |
| `mrac.ts`                         | first-order plant with unknown gain          | model-reference adaptive control (MIT-rule, Lyapunov) | `mrac`         |
| `feedback-linearization.ts`       | single-link pendulum tracking                | feedback linearization / computed torque (Khalil) | `feedback-linearization` |
| `mpc-double-integrator.ts`        | double integrator with hard input bounds     | constrained receding-horizon QP MPC      | `mpc-double-integrator`  |

### JSON adapters and tests

`general/adapters/optimal-control-adapters.ts` registers all six leaf
models. Six new example JSON files in `examples/` make every model
CLI-runnable:

```
node dist/des/main-from-json.js examples/pontryagin-bang-bang.json
node dist/des/main-from-json.js examples/kalman-filter.json
node dist/des/main-from-json.js examples/sliding-mode.json
node dist/des/main-from-json.js examples/mrac.json
node dist/des/main-from-json.js examples/feedback-linearization.json
node dist/des/main-from-json.js examples/mpc-double-integrator.json
```

`test/optimal-control-test.ts` (24 checks) verifies the canonical
theoretical invariant of each method:

* **PMP**: bang-bang switches at most once + arrival time near the
  closed-form `2 √(|x₀|/u_max)`.
* **Kalman**: estimator RMSE strictly below the raw-measurement RMSE
  (≥ 50 % reduction on the test seed).
* **SMC**: finite-time arrival on the sliding surface and bounded
  steady-state error under both sinusoidal and square-wave matched
  disturbances.
* **MRAC**: steady-state RMS tracking error < 0.05 and adapted
  parameters near the closed-form ideals
  `θ*_x = (a_m − a)/b`, `θ*_r = b_m/b`.
* **Feedback linearization**: tracking RMS error of order 10⁻⁴ rad
  on the pendulum (essentially numerical noise).
* **MPC**: hard input constraint `|u| ≤ u_max` always satisfied,
  control saturates on the boundary, tighter constraint → larger
  arrival tick (constraint tightening monotonicity).

Total test count after this batch: **685 checks across 15 suites**
(was 661/14).

## Eight MDP-adjacent bases + canonical demos (2026-05-25)

The framework now spans the **whole MDP-adjacent universe** introduced in
the user's "MDPs and friends" table — every concept is now backed by a
DES base class plus a runnable demo on a textbook problem, fully wired
into the JSON registry.

### New base classes (`general/des-base/`)

| Base                               | What it abstracts                                 | Extends                       |
| ---------------------------------- | ------------------------------------------------- | ----------------------------- |
| `FiniteHorizonDPStation`           | finite-horizon DP via backward induction          | `DESStation`                  |
| `LinearVFAStation<S>`              | approximate DP / linear semi-gradient TD          | `RLAgentStation<S, number>`   |
| `BeliefStateStation<A, O>`         | POMDP belief filter + plug-in solver              | `DESStation`                  |
| `SemiMDPAgentStation<S, A>`        | options framework / SMDP Q-learning               | `RLAgentStation<S, A>`        |
| `TabularActorCritic`               | one-step actor-critic (tabular V + softmax π)     | `RLAgentStation<number, number>` |
| `MonteCarloAgent`                  | first-/every-visit MC control                     | `RLAgentStation<number, number>` |
| `JointEnvStation` + `MultiAgentSystem` | simultaneous-move multi-agent RL              | `DESStation`                  |
| `LQRController`                    | discrete-time LQR via Riccati DARE                | `ControllerStation<Vec, Vec>` |

### Concrete leaves and canonical use cases (`general/`)

| Leaf model               | Classic problem                                   | Algorithm                              | JSON id                  |
| ------------------------ | ------------------------------------------------- | -------------------------------------- | ------------------------ |
| `inventory-dp.ts`        | multi-period stochastic inventory                 | finite-horizon DP / Bellman backward induction | `inventory-dp`     |
| `mountain-car.ts`        | Mountain Car (Sutton & Barto §10.1)               | linear VFA + Sutton-Albus tile coding | `mountain-car-vfa`       |
| `tiger-pomdp.ts`         | Tiger problem (Cassandra-Kaelbling-Littman 1994)  | QMDP & 1-step belief look-ahead        | `tiger-pomdp`            |
| `four-rooms.ts`          | Four Rooms (Sutton-Precup-Singh 1999)             | SMDP Q-learning over hallway options   | `four-rooms-smdp`        |
| `actor-critic-gridworld.ts` | 4×4 GridWorld with pits                        | one-step tabular actor-critic           | `actor-critic-grid`      |
| `blackjack.ts`           | Blackjack (Sutton & Barto §5.1)                   | first-visit Monte Carlo control         | `blackjack-mc`           |
| `stag-hunt.ts`           | Stag Hunt coordination game                       | independent Q-learning (Tan 1993)       | `stag-hunt`              |
| `double-integrator-lqr.ts` | continuous double integrator                    | LQR via discrete-time algebraic Riccati | `double-integrator-lqr`  |

### JSON adapters

`general/adapters/mdp-adjacent-adapters.ts` registers every leaf via the
existing `DESModelRegistration<P, R>` interface, so the new programs
are runnable from the CLI:

```
node dist/des/main-from-json.js examples/inventory-dp.json
node dist/des/main-from-json.js examples/four-rooms-smdp.json
node dist/des/main-from-json.js examples/tiger-pomdp.json
…
```

Each model ships with an `examples/<id>.json` template the user can copy
and tweak (or generate from scratch via `--example <id>`).

### New tests

`test/mdp-adjacent-test.ts` (30 checks) covers all eight new models —
verifies for each one a learning / convergence invariant, plus
end-to-end agreement with theory where possible (e.g. LQR's realised
cost ≤ DARE cost-to-go on the deterministic system, the Tiger
1-step-look-ahead avoids catastrophic opens, MC blackjack beats the
stick-on-20 baseline by ≥ 0.25 EV).

```
test-mdp-adjacent: 30/30 checks passed
```

### Total verified surface (this commit)

- **14 test suites** green (calculus, dispatch, soccer, genetic-tsp,
  shortest-path, factmachine-math, incremental-lp, stochastic-lp,
  temp-control, milp-bnb, simulated-annealing, optimization-as-des,
  validation, mdp-adjacent),
- **661 individual checks** pass,
- **12 registered JSON-runnable models** (4 pre-existing + 8 new).

## Validator protocol baked into every DES base class (2026-05-25)

The `des-base/` hierarchy now ships with a first-class **validator protocol**.
Every iterative-algorithm base (and every `DESStation` in general) exposes:

- `addValidator(v)` — register a `Validator<this>` on the station,
- `runValidation()` — run all registered validators and return a flat
  `ValidationCheck[]`,
- `validationReport()` — pretty-printed multi-line report,
- `onFinalize()` — runs once after the loop terminates and BEFORE
  validators, so a station can attach validators reactively from the
  final state.

`runIterativeDES` orchestrates the lifecycle:

```
   runTimeStep* → onFinalize → runValidation → summary.{validation, validationOk}
```

Stations without registered validators are skipped (no overhead, no
output noise).

### New module: `general/des-base/validation.ts`

| Factory                          | Purpose                                                  |
| -------------------------------- | -------------------------------------------------------- |
| `numericValidator`               | scalar comparison with abs/rel tolerance                 |
| `boundValidator`                 | assert value ∈ `[low, high]`                             |
| `monotonicityValidator`          | assert series is non-increasing or non-decreasing        |
| `groundTruthValidator`           | custom equality check against a reference                |
| `intrinsicCheck`                 | wrap a `(station) → boolean` predicate                   |
| `externalReferenceValidator`     | load JSON from disk + user-supplied `compare(station, ref)` (silent-if-missing optional) |

### Pre-hooked-up to existing leaves

Each refactored algorithm now registers its own intrinsic invariants and,
where applicable, optional external references:

| Leaf class                | Auto-attached validators                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `TSPSAOptimizer`          | `bestHistory` non-increasing, `best` is a permutation, `bestCost ≥ 0`, `bestCost ≥ Held-Karp` (when `n ≤ 12`)      |
| `TSPGAOptimizer`          | `bestHistory` non-increasing (when `eliteN ≥ 1`), `best` is a permutation, `bestLength ≥ Held-Karp` (when `n ≤ 12`) |
| `SAOptimizer<S>` (legacy) | `bestHistory` non-increasing, `accepted ≤ iter`                                                                    |
| `GeneticTSPOptimizer`     | `bestHistory` non-increasing (when elite ≥ 1), `best` is a permutation                                             |
| `MILPBnBStation`          | search-finished or hit node cap, incumbent ⊆ LP-relaxation bound                                                   |
| `ValueIterationStation`   | `lastDelta ≤ tol` when converged, `Δ_{k+1} ≤ γ·Δ_k` (γ-contraction), optional `referencePath` external comparison  |
| `BendersStation`          | `gap ≤ tol` when optimal, `LB ≤ UB`, optional `referencePath` external comparison                                  |
| `TempControllerBase`      | every emitted `u` lies in `[0, Q_max]`                                                                             |

External references are opt-in via constructor options (e.g.
`new ValueIterationStation(spec, {referencePath: 'out/external/court-mdp/python.json'})`).
Missing files fail loudly by default OR silently when `silentIfMissing: true`.

### New test: `test/validation-test.ts` (`npm run test-validation`)

End-to-end validator-protocol test (40 checks) covering:

1. All factory primitives (numeric, bound, monotonicity, ground-truth,
   external-ref, intrinsic) including throw-handling.
2. `DESStation.addValidator` wiring + `runIterativeDES.summary.validation`,
   `validationOk`, `runValidators: false` opt-out, empty/skip semantics,
   `onFinalize` running before validators.
3. Intrinsic validators on SA / GA / temp-control / VI / Benders / MILP-B&B
   all PASS for nominal runs.
4. Ground-truth validator FAILS LOUDLY when an algorithm is deliberately
   broken (sub-class returning a fake `bestCost` of 0 → Held-Karp validator
   catches it).
5. External-reference validators degrade gracefully when files are missing
   AND surface mismatches when files are present.
6. `ValueIterationStation` with `referencePath` constructor option
   auto-attaches the external validator.

### Verification

| Suite                          | Result            |
| ------------------------------ | ----------------- |
| `test-validation` (new)        | **40 / 40**       |
| 12 pre-existing test suites    | **551 / 551**     |
| 5 representative validators    | **179 / 179**     |
| Full sweep (no regressions)    | green             |

## Whole-codebase base-class audit (2026-05-25)

Every iterative algorithm in the engine now extends one or more
`des-base/` base classes. Three NEW bases were added on top of the
existing five and seven legacy algorithms were refactored.

### New bases

| Base class                       | Captures                                            | Hooks                                                                                |
| -------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `TreeSearchStation<N>`           | tree-structured search with frontier + incumbent    | `pickNext`, `evaluate`, `expand`, `pushChildren`; optional `shouldPrune`/`onPrune`/`onIncumbentUpdate` |
| `FixedPointIterationStation<S>`  | iterate `x_{k+1} = T(x_k)` until ‖Δ‖ < ε             | `initialState`, `applyOperator`, `delta`; optional `shouldStop`                      |
| `ControllerStation<O, U>`        | observe → control law → saturated actuator         | `controlLaw`; optional `uMin`/`uMax`/`onTick`/`reset`                                |

### Refactor map (all models on a base; zero behaviour regressions)

| Module                                  | Old shape                       | Now extends                                          | Verified                  |
| --------------------------------------- | ------------------------------- | ---------------------------------------------------- | ------------------------- |
| `general/sa-des.ts` `TSPSAOptimizer`    | already on base                 | `SingleStateOptimizer<Tour>`                         | 39/39 unit + 45/45 valid. |
| `general/sa-des.ts` `TSPHillClimber`    | already on base                 | `SingleStateOptimizer<Tour>` (override `accept`)     | ↑                         |
| `general/ga-des.ts` `TSPGAOptimizer`    | already on base                 | `PopulationOptimizer<Tour>`                          | ↑                         |
| `general/qlearning-des.ts`              | already on base                 | `RLAgentStation<number, number>`                     | ↑                         |
| `general/ppo-des.ts` `TabularPPOAgent`  | already on base                 | `PolicyGradientAgent<number, number>`                | ↑                         |
| `general/ppo-des.ts` `PPOClipUpdate…`   | already on base                 | `PolicyUpdateStation`                                | ↑                         |
| `general/rl-environments.ts`            | wrapped via base                | `EnvironmentStation` (used in `optimization-as-des`) | ↑                         |
| `general/milp-bnb.ts`                   | recursive function              | `TreeSearchStation<MILPNode>` + composes `IncrementalLP`           | 38/38 + 22/22             |
| `general/mcts.ts`                       | imperative loop                 | `TreeSearchStation<Node<S>>` (UCT walk in `pickNext`)               | 72/72 + 11/11             |
| `general/value-iteration.ts`            | imperative for-loop             | `FixedPointIterationStation<Float64Array>`                          | 32/32 + 28/28 + 36/36     |
| `general/stochastic-lp.ts` (Benders)    | for-loop with cuts              | `FixedPointIterationStation<BendersIterState>` + composes `IncrementalLP` | 44/44 + 60/60             |
| `general/temp-control.ts` (4 ctrls)     | switch-stmt + closure state     | `ControllerStation<TempObs, number>` (4 leaves: bang-bang/PID/fuzzy/MdpMpc) | 36/36 + 21/21             |
| `general/genetic-tsp.ts` `runGeneticTSP`| imperative GA loop              | `PopulationOptimizer<Tour>` (with `acceptChild` for precedence cut) | 18/18 + 16/16             |
| `general/simulated-annealing.ts` `runSimulatedAnnealing` | imperative SA loop | `SingleStateOptimizer<S>`                          | 31/31 + 31/31             |

`PopulationOptimizer` was extended with `acceptChild` + `childRetryLimit`
hooks so that hard-constraint regimes ("cut" precedence in genetic-TSP)
fit cleanly into the base's breeding template.

### Composition

Each refactored model uses MULTIPLE base classes:

* MILP-B&B = `DESStation` ⟶ `TreeSearchStation<MILPNode>` PLUS composes
  `IncrementalLP` (which itself is the warm-start LP layer) — 2 layers.
* MCTS = `DESStation` ⟶ `TreeSearchStation<Node<S>>` — 2 layers.
* Stochastic LP / Benders = `DESStation` ⟶ `FixedPointIterationStation`
  PLUS composes `IncrementalLP` — 2 layers.
* Temp-control = `DESStation` ⟶ `ControllerStation<TempObs, number>`
  ⟶ `TempControllerBase` ⟶ `BangBangController` / `PIDController` /
  `FuzzyController` / `MdpMpcController` — 4 layers per leaf.
* Genetic TSP, Simulated Annealing = `DESStation` ⟶ `PopulationOptimizer`
  / `SingleStateOptimizer` — 2 layers.

### Test sweep

After the refactor the entire suite still passes:

```
test-calculus              31/31     validate-calculus              26/26
test-dispatch              72/72     validate-lp                    36/36
test-soccer                49/49     validate-soccer                16/16
test-genetic-tsp           18/18     validate-genetic-tsp           16/16
test-shortest-path         43/43     validate-shortest-path         23/23
test-factmachine-math      57/57     validate-factmachine-math      35/35
test-incremental-lp        40/40     validate-incremental-lp        42/42
test-stochastic-lp         44/44     validate-temp-control          21/21
test-temp-control          36/36     validate-milp-bnb              22/22
test-milp-bnb              38/38     validate-simulated-annealing   31/31
test-simulated-annealing   31/31     validate-optimization-as-des   45/45
test-optimization-as-des   39/39
```



Cross-validation, extension, and verification of the in-repo TypeScript
DES engine on a SEIR-with-hospitalization model and several additional
domains (DSP convolution, recursive ML, continuous-physics ODE,
multi-agent elevator dispatch, MDP-based control systems, a coupled
two-disease compartmental epidemic, a calculus solver that compiles
expressions into station networks, a FactMachine POMDP, an LP bridge
that includes simplex implemented inside the engine, a warm-startable
incremental LP solver, a two-stage stochastic LP solver with Benders
decomposition, an HVAC temperature-control system with four
interchangeable controllers (bang-bang, PID, fuzzy-PI, MDP-MPC), a JSON
specification format with a model registry that lets users save and
re-run any model from disk, a mixed-integer LP branch-and-bound solver
that composes the IncrementalLP for relaxations at each node, and a
generic simulated-annealing solver applied to TSP and 0/1 knapsack).
All work is in `courses/hdm-fall-2022/des/`.

## Optimization-as-DES base hierarchy (2026-05-25)

Strong base classes for "iterative algorithm as DES", structured around
the OBSERVATION that almost every iterative algorithm in the codebase
fits one of four problem-shapes. Each base is a TEMPLATE METHOD whose
`runTimeStep` is the orchestrator and whose ABSTRACT hooks are the
algorithm-specific bits. Concrete leaf classes implement only the hooks.

```
DESStation                              ← named typed channels, take/emit/pipe/hasWork
├── SingleStateOptimizer<S>             ← single walker (SA, hill climb, tabu, Newton)
│     hooks: initialState, cost, propose, accept, clone, shouldStop
├── PopulationOptimizer<I>              ← population-based (GA, PSO, DE, ACO)
│     hooks: initialPopulation, evaluate, select, recombine, mutate, clone, shouldStop
├── RLAgentStation<S, A>                ← online TD (Q-learning, SARSA, expected SARSA)
│     hooks: pickAction, update, endOfEpisode
├── PolicyGradientAgent<S, A>           ← rollout-buffered PG (REINFORCE, A2C, PPO)
│     hooks: samplePolicyAndValue (+ counterpart PolicyUpdateStation: runUpdate)
└── EnvironmentStation<S, A>            ← generic env wrapper (state/transition/action channels)
```

The runner `runIterativeDES(stations, opts)` shuffles execution order
each tick (Fisher-Yates with injectable RNG → reproducible) and stops
on quiescence, `maxTicks`, or a custom `stopWhen` predicate.

### Concrete leaves implemented

Five algorithms re-built as ~150-line leaves of these bases (ALL the
iteration logic is shared):

| Algorithm        | Base class                | Hooks implemented                                     |
| ---------------- | ------------------------- | ----------------------------------------------------- |
| Simulated Anneal | `SingleStateOptimizer`    | Metropolis `accept`, 2-opt/or-opt `propose`, geometric/log/linear cooling |
| Hill Climber     | `SingleStateOptimizer`    | overrides only `accept` to be strict-improvement      |
| Genetic TSP      | `PopulationOptimizer`     | tournament `select`, OX `recombine`, inversion+swap `mutate`, NN init |
| Q-learning       | `RLAgentStation`          | ε-greedy `pickAction`, off-policy TD `update`, ε-decay `endOfEpisode` |
| PPO (clipped)    | `PolicyGradientAgent` + `PolicyUpdateStation` | softmax `samplePolicyAndValue`, GAE + K-epoch clipped surrogate `runUpdate` |

The PG/PPO pause-resume protocol is delicate: when the rollout buffer
fills mid-episode the agent stashes the unanswered next-state, emits a
`TrainTriggerToken` to the `PolicyUpdateStation`, and resumes via
`ResumeToken` once the update station has mutated `θ` and `V` and
cleared the buffer. This handshake is encoded in the base class so
every PG variant inherits it for free.

### Validation (45 / 45 checks)

All four algorithms hit ground-truth on small problems:

| Study             | Reference          | Result |
| ----------------- | ------------------ | ------ |
| Pentagon TSP n=5  | exact perimeter    | SA, HC, GA all match exactly across 3-5 seeds |
| Random TSP n=10   | Held-Karp          | SA & GA both within 0.00% of optimum |
| 4×4 GridWorld     | value iteration    | Q-learning V(0) = V*(0) = 3.213, 100% greedy success across 3 seeds |
| Corridor(8)       | value iteration    | PPO V(0) ≈ 2.052 vs optimal 2.053, 100% greedy success across 3 seeds |

Plus structural checks: best-history monotonicity (SA, GA, HC),
`acc == impr` invariant for HC, GA `mean ≥ best` pointwise,
PPO updates ≈ steps / rolloutLen, full seed reproducibility.

### Unit tests (39 / 39 checks)

`test/optimization-as-des-test.ts` covers DESStation channel mechanics
(drain/peek/pipe routes default + named channels), runner termination
modes (quiescent / maxTicks / stopWhen), each base's template method
(bootstrap, finished flag, history monotonicity, counter accuracy),
EnvironmentStation token semantics (one StateToken per episode,
one TransitionToken per step, terminal flag propagated), and the shared
scalar/vector episode accounting helpers.

### Files

- `general/des-base/station.ts` — DESStation, named channel inboxes/outs
- `general/des-base/runner.ts` — runIterativeDES (shuffle + stop predicates)
- `general/des-base/single-state-optimizer.ts` — SA/HC/tabu base
- `general/des-base/population-optimizer.ts` — GA/PSO/DE base
- `general/des-base/rl-agent.ts` — Q-learning/SARSA base
- `general/des-base/policy-gradient-agent.ts` — REINFORCE/A2C/PPO base + PolicyUpdateStation
- `general/des-base/environment.ts` — generic EnvironmentStation
- `general/des-base/episode-accounting.ts` — shared scalar/vector episode counters
- `general/des-base/rl-tokens.ts` — StateToken/ActionToken/TransitionToken/TrainTriggerToken/ResumeToken
- `general/des-base/index.ts` — re-exports
- `general/sa-des.ts` — TSPSAOptimizer + TSPHillClimber
- `general/ga-des.ts` — TSPGAOptimizer
- `general/qlearning-des.ts` — QLearningAgent
- `general/ppo-des.ts` — TabularPPOAgent + PPOClipUpdateStation
- `main-optimization-as-des.ts` — comparison driver
- `runners/validate-optimization-as-des.ts` — 45-check validator
- `test/optimization-as-des-test.ts` — 39-check unit suite

### Run

```
npm run optimization-as-des              # Pentagon + n=12 TSP + GridWorld + Corridor(8)
npm run validate-optimization-as-des     # 45 checks across 3-5 seeds per algo
npm run test-optimization-as-des         # 39 base-class + leaf unit tests
```

## Mixed-integer LP via branch-and-bound + Simulated annealing (2026-05-25)

Two new optimisation algorithms join the arsenal, both implemented as
discrete-event systems:

### MILP via branch-and-bound (`general/milp-bnb.ts`)

Solves
```
max  c · x
s.t. A x ≤ b
     x_j ≥ 0
     x_j ∈ ℤ for j ∈ I
```
by depth-first branch-and-bound, with the LP relaxation at each node
solved by our existing `IncrementalLP`. The single LP instance is
mutated along the DFS path:

```
applyAddConstraint(x_j ≤ ⌊x_j*⌋)   ↘
  recurse left subtree              ↘ ← parametric simplex warm-start
applyRemoveConstraint(...)          ↗   makes each child LP cheap.
applyAddConstraint(-x_j ≤ -⌈x_j*⌉)  ↗
  recurse right subtree
applyRemoveConstraint(...)
```

This is the canonical "branch-and-bound + LP relaxation" loop, except
that the LP solver is genuinely incremental — each child reuses the
parent's basis and uses dual simplex to restore primal feasibility.

DES form:

- **B&B node = station**: holds the IncrementalLP state and the
  branching-bound stack accumulated from the root.
- **Movables = (LP_z, fractional integer-var indices)**: flow from
  each node to a brancher station that decides prune / commit /
  branch.
- **Incumbent = global station**: tracks the best integer-feasible
  solution; pruning by bound consults its `z`.

Built-in problem builders:

- `buildKnapsackMILP(values, weights, capacity)` — 0/1 knapsack.
- `buildFacilityLocationMILP(p)` — uncapacitated facility location.

Pruning rules:

- **Infeasibility** — LP relaxation infeasible at a node.
- **Bound** — LP_z ≤ incumbent (max sense; reverse for min).
- **Integer-feasible** — LP solution is already integer; update
  incumbent and prune.

Branching rule: most-fractional (default) or first-fractional.

Validation (`runners/validate-milp-bnb.ts`, 22/22 passing):

| Study | Check |
|---|---|
| 1. Textbook 4-item knapsack | exact match with brute force, ≤ 5 nodes |
| 2. 20 random knapsacks (n=8..14) | all match brute force, total nodes ≪ total enumerations |
| 3. Pure LP (no integers) | reduces to a single root-LP solve |
| 4. Mixed integer/continuous | integer vars are integer, MILP_z ≤ LP_z |
| 5. Zero-capacity infeasibility | z = 0, x = 0 |
| 6. n=24 random knapsack | optimal in < 1 s, < 1000 nodes |
| 7. Capacity monotonicity | tighter capacity ⇒ smaller optimal |

Unit tests (`test/milp-bnb-test.ts`, 38/38 passing) cover the
knapsack builder, trivial 2-item instances, pure-LP behaviour,
mixed integer/continuous, bounding properties, feasibility,
branching-rule equivalence, trace recording, and `maxNodes` early
termination.

### Simulated annealing (`general/simulated-annealing.ts`)

Single-walker probabilistic local search. Each tick:

1. Generate a candidate neighbour `s'` of the current state `s`.
2. Compute Δ = cost(s') − cost(s).
3. Accept `s'` if Δ ≤ 0, otherwise with probability exp(−Δ/T).
4. Update temperature T per the cooling schedule.
5. Track the global-best state ever seen.

Cooling schedules supported:

- **Geometric**:   `T_k = T_0 · α^k`  (default; α ≈ 0.99).
- **Logarithmic**: `T_k = T_0 / log(2 + k)`  (Hajek 1988 — global
  optimum w.p. 1 in the limit).
- **Linear**:      `T_k = max(0, T_0 − rate · k)`.
- **Exp-restart**: cycles geometric over a fixed period (basin-
  hopping flavor).

DES stations: CandidateGenerator → CostEvaluator → MetropolisAccept →
TemperatureScheduler → BestTracker.

Built-in problem adapters:

- **TSP** (`buildTSPSAProblem`): 2-opt + or-opt mixed neighbourhood,
  random / nearest-neighbour init, optional precedence-violation
  penalty.
- **0/1 Knapsack** (`buildKnapsackSAProblem`): bit-flip neighbourhood,
  capacity-violation penalty.

The `SAProblem<S>` interface is generic: any state type S works, plug
in your own `cost`, `neighbour`, `initial` functions.

Validation (`runners/validate-simulated-annealing.ts`, 31/31 passing):
- Pentagon TSP: SA finds exact optimum at 5 different seeds.
- Random TSPs (n=6,8,10): SA matches Held-Karp exact at multiple
  seeds.
- 5 random knapsacks (n=12): SA matches MILP-B&B exact.
- Cooling-schedule monotonicity (geometric, linear, log).
- Reproducibility (same seed → identical trajectory).
- Best history monotonically non-increasing.
- Acceptance-rate behaviour: high-T accepts > low-T accepts;
  low-T accepts only improvements (modulo zero-Δ moves).
- Stall-limit early stopping.

Unit tests (`test/simulated-annealing-test.ts`, 31/31 passing) cover
`temperatureAt` for all 4 schedules, TSP adapter
(initial/permutation invariance, neighbour preserves permutation,
both 2-opt and or-opt only modes), pentagon-optimal recovery,
knapsack-optimal recovery, reproducibility, monotonic best, generic
quadratic minimisation (showing the framework's genericity), trace
recording, and stall-limit termination.

### Comparison study (`main-simulated-annealing.ts`)

A 5-study CLI driver that runs SA on:
1. Pentagon TSP (n=5) — exact optimum recovered.
2. Random TSP (n=12) — head-to-head with Held-Karp + GA. All three
   find the same optimum.
3. Random TSP (n=30) — equal-compute SA-vs-GA bake-off. SA wins (or
   ties) on most seeds.
4. Cooling-schedule comparison — geometric ≈ exp-restart < log
   ≈ linear on a 20-city problem.
5. n=15 knapsack — SA vs MILP-B&B exact. SA matches the exact
   optimum in single-millisecond wall.

### JSON registry adapters

Both new models are registered with the JSON spec format:

- `model: "milp-bnb"` — accepts `{raw: {...}}` (generic MILP) or
  `{knapsack: {values, weights, capacity}}` (convenience).
- `model: "simulated-annealing"` — accepts `{problem: "tsp"|"knapsack",
  ..., cooling: {kind: ..., ...}, options: {maxIterations, ...}}`.

Example specs in `examples/`:

- `milp-knapsack.json`     — textbook 4-item knapsack.
- `milp-mixed.json`        — generic mixed integer/continuous MILP.
- `sa-tsp-pentagon.json`   — SA on the pentagon TSP.
- `sa-tsp-random30.json`   — SA on a 30-city random TSP.
- `sa-knapsack.json`       — SA on a 6-item knapsack.

Run any of them with `npm run from-json -- examples/<name>.json`.

### Affected files

- `general/milp-bnb.ts` — new (B&B + LP-relaxation solver, 0/1
  knapsack and facility-location problem builders).
- `general/simulated-annealing.ts` — new (SA framework, TSP and
  knapsack adapters, 4 cooling schedules).
- `general/adapters/milp-bnb-adapter.ts` — new (JSON registration).
- `general/adapters/simulated-annealing-adapter.ts` — new (JSON
  registration with oneOf cooling schema).
- `general/des-registry.ts` — registers the two new adapters.
- `main-milp-bnb.ts` — new (5-study CLI driver).
- `main-simulated-annealing.ts` — new (5-study comparison driver).
- `runners/validate-milp-bnb.ts` — new (22 validation checks).
- `runners/validate-simulated-annealing.ts` — new (31 validation
  checks).
- `test/milp-bnb-test.ts` — new (38 unit-test checks).
- `test/simulated-annealing-test.ts` — new (31 unit-test checks).
- `examples/milp-knapsack.json`, `examples/milp-mixed.json`,
  `examples/sa-tsp-pentagon.json`, `examples/sa-tsp-random30.json`,
  `examples/sa-knapsack.json` — new.
- `package.json` — adds 6 npm scripts (`milp-bnb`,
  `validate-milp-bnb`, `test-milp-bnb`, `simulated-annealing`,
  `validate-simulated-annealing`, `test-simulated-annealing`).
- `README.md`, `CHANGELOG.md` — updated.

## JSON specification format + model registry (2026-05-25)

Two pieces define a runnable DES program in this codebase:

1. The TOPOLOGY — which stations exist, how they're connected, what
   movables flow. This is a chunk of TypeScript code.
2. The PARAMETERS — numeric configuration. This is what users typically
   vary.

JSON is a great fit for (2) and a poor fit for (1). The new JSON spec
format takes that distinction seriously: each spec names a registered
model id, supplies its parameters, and lets the registry validate and
run it.

```jsonc
{
  "$schema": "des/model-spec/v1",
  "model": "temp-control",
  "description": "24-hour winter day, PID controller",
  "parameters": { "T_target": 70, "duration_h": 24, ... },
  "runtime": {
    "seed": 42,
    "outputs": { "csv": "out/foo.csv", "html": "out/foo.html" }
  }
}
```

Run any registered model from disk:

```bash
node dist/des/main-from-json.js examples/temp-control-pid.json
node dist/des/main-from-json.js --list                 # show registered models
node dist/des/main-from-json.js --schema temp-control  # JSON Schema-style description
node dist/des/main-from-json.js --example temp-control # paste-ready example
```

### What this change touches

- `general/des-spec.ts` — schema types, declarative parameter validator
  (no external deps), envelope types (`DESModelSpec`, `DESRuntimeConfig`,
  `DESModelRegistration`).
- `general/des-registry.ts` — registry + `runFromSpec()` driver.
- `general/adapters/temp-control-adapter.ts` — first adapter (HVAC).
- `general/adapters/shortest-path-adapter.ts` — second adapter, used to
  show the registry hosts unrelated models without changes to the
  registry itself.
- `main-from-json.ts` — CLI driver.
- `examples/*.json` — five example specs spanning two models.

### Code-or-JSON, both work

The JSON envelope is also a valid TypeScript object literal — users who
prefer code can write the same spec inline:

```typescript
import {runFromSpec} from './general/des-registry';

const spec = {
  $schema: 'des/model-spec/v1' as const,
  model: 'temp-control',
  parameters: { T_target: 70, duration_h: 24, controller: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5} },
  runtime: { seed: 42 },
};
const result = await runFromSpec(spec);
```

For wholly new BEHAVIOUR users write a TypeScript model + adapter and
register it under a new id; the JSON layer handles parameter
configuration and persistence.

## Indoor temperature control DES (2026-05-25)

A new model exercises the framework's "control" half: an HVAC
temperature controller maintaining indoor temperature within ±2°F of
target while minimising energy use, against a noisy diurnal outdoor
temperature pattern with a noisy 6-hour-ahead forecast.

Four interchangeable controllers, each implemented as the
ControllerStation in the same DES topology:

1. **Bang-bang** — heater on if T_in < target, off otherwise. Baseline.
2. **PID** — classical proportional-integral-derivative with a
   first-order-filtered derivative (necessary on a 1-min sampled signal)
   and conditional-integration anti-windup.
3. **Fuzzy-PI** — Mamdani fuzzy controller with NL/NS/Z/PS/PL terms on
   (error, error-rate); 5×5 rule base produces normalised Δ-Q which is
   integrated externally to give offset-free tracking.
4. **MDP-MPC** — receding-horizon dynamic programming. Each tick the
   controller solves a finite-horizon MDP on the temperature × time
   grid using the forecast as the disturbance model, executes the
   first action, advances one tick, and re-plans. The action grid has
   `nLevels` discrete heater powers; `trackWeight` controls how
   tightly the controller pulls toward T_target inside the band.

The MDP-MPC implementation handles the "fine-grained tick problem"
correctly: at Δt = 1 minute the per-tick temperature change at any
heater level is < 0.1°F, so a coarse value-table grid would put every
action in the same cell. The DP linearly interpolates V[k+1] in the
continuous next-state instead of snapping to bins.

### What this change touches

- `general/temp-control.ts` — physics, controllers, simulation runner.
- `main-temp-control.ts` — head-to-head CLI; sensitivity sweep over
  (forecast horizon × tracking weight); stress test with a tighter
  band; CSV trace export per controller.
- `main-temp-control-anim.ts` — generates an interactive HTML animation
  per controller (thermometer, heater dial, station-flow diagram, time
  series of T_in / T_out / Q).
- `runners/validate-temp-control.ts` — 21 cross-method validation
  studies (energy balance, all four controllers in band, PI steady-state,
  MDP cost dominance under stress, reproducibility, fuzzy boundary
  behaviour, MDP-MPC monotonicity in trackWeight).
- `test/temp-control-test.ts` — 36 unit tests.
- `animation/scenes/temp-control-scene.ts` — animation scene builder.

## Renamed the framework to "Discrete Event System" (broader than just simulation) (2026-05-25)

The framework's reach has grown well beyond the original "discrete event
simulation" framing. The same station/movable substrate now also hosts:
a control half (elevator MDP dispatch, MPC, RLC step-response, calculus
solver) and an algorithmic half (simplex-in-DES, incremental LP,
stochastic LP / Benders, GA-TSP, shortest-path-as-DES, MCTS rollouts,
value iteration). All three workload classes live on the SAME tick
clock, the SAME station/movable architecture, and the SAME no-FEL
kernel. The acronym DES is preserved because all three readings work:

- "Discrete-event SIMULATION" — for the stochastic-process workloads.
- "Discrete-event SYSTEM" (the systems-and-control reading) — for control loops.
- "Discrete-event SEMANTICS" (the algorithmic reading) — for iterative algorithms whose state changes at well-defined ticks.

### What this change touches

- `README.md` opens with the new framing and a workload-classification
  table linking each workload class to the modules in this repo.
- `package.json`'s `description` field is updated to "Discrete event
  system project (simulation, control, and iterative algorithm
  execution)".
- File header docstrings in `general/incremental-lp.ts`,
  `general/stochastic-lp.ts`, `general/genetic-tsp.ts`, `general/lp.ts`,
  `general/des-lp-bridge.ts`, `main-backpropagation.ts`,
  `main-convolution.ts`, `main-electric-circuit.ts`, `main-elevator.ts`
  are all updated to reflect the broader framing — pointing back to the
  workload class each module belongs to and what's actually being
  represented at each tick.
- The "Linear programming" README section now explicitly notes that the
  "DES is just a simulation framework" claim was correct *for one-shot
  solves* but stops being true for repeated/warm-started solves, and
  points to the incremental + stochastic LP modules as evidence.

No code identifier, file name, package name, or test name changed. The
acronym DES is kept everywhere; the abstraction surface is identical.
This rename is a documentation and framing change.

## Stochastic LP solver: SAA monolithic, Benders-as-DES, statistical convergence (2026-05-25)

The natural follow-up to the warm-startable LP from earlier today: what if the LP coefficients themselves are random variables drawn from a known distribution? This is the domain of **stochastic linear programming** (SLP), and it's where the marriage of DES (scenario sampling), incremental LP (warm-startable master), and MDP-style cuts (piecewise-linear value-function approximation) becomes spectacular.

### What this change adds

| File | Purpose |
|---|---|
| `general/stochastic-lp.ts` | Two-stage SLP framework. `solveSubproblemWithDuals` reads dual variables from the optimal `IncrementalLP` tableau (slack column reduced costs). `solveSLPMonolithic` builds the deterministic-equivalent SAA LP (1+N·n_second variables, 1+N·m_second constraints) and hands it to `solveLPInternal`. `solveSLPBenders` decomposes the SAA into a master LP plus N subproblems and runs L-shaped iteration with the master being our `IncrementalLP` (each cut = one `applyAddConstraint` call). `buildProductionScenarios` and `buildProductionSLP` build the multi-product newsvendor template; `solveProductionClosedForm` is the analytical oracle. |
| `main-stochastic-lp.ts` | A 2-product capacity-planning scenario solved three ways (closed form / monolithic SAA / Benders-as-DES), with a Benders convergence-trace table and out-of-sample policy evaluation on 50,000 fresh demands. |
| `runners/validate-stochastic-lp.ts` | Four-part audit: A) Cross-method bit-equivalence on identical scenario sets (mono ≡ Benders within 4.6e-12 on every objective); B) Statistical convergence to the closed-form optimum at the textbook 1/√N rate, replicated 20 seeds × 4 sample sizes; C) Speed comparison showing Benders' advantage growing with N (7× at N=50, **2185× at N=500**); D) Budget-constrained scenarios (no closed form available) where Benders and monolithic must still agree exactly. **60/60 checks pass.** |
| `test/stochastic-lp-test.ts` | 44 unit tests covering subproblem dual extraction, KKT optimality conditions, monolithic SAA on a 3-scenario discrete distribution, Benders convergence properties (master UB non-increasing, best-so-far LB non-decreasing), statistical bias decay, and budget-constraint behaviour. **44/44 pass.** |

### The mapping to DES

```
SamplerStation (DES)  ─►  ScenarioStation_1  ─┐
                        ScenarioStation_2  ─┼─►  AggregatorStation  ─►  MasterStation (IncrementalLP)
                        ...                  │      forms cut from           │
                        ScenarioStation_N  ─┘    avg(π·h, π·T)               │
                                                                              │
                                              ◄────  applyAddConstraint  ◄────┘
```

Every Benders iteration is a tick. Each cut is a movable arriving at the master via `applyAddConstraint`. The dual π_s* of each subproblem is extracted from `IncrementalLP.getReducedCosts()` (the slack columns of the optimal subproblem tableau ARE the duals — textbook fact).

### Why pivoting is the entire ballgame

Solving the SAA for `N` scenarios from scratch with dense simplex is `O(m³ · iters)` per pivot, and the LP has `1 + 4N` constraints. For our test problem at N=500 that's a 2001-row dense pivot with 1587 iterations — **18.9 seconds**. Benders solves the same problem in **10ms** (28 iterations of a master with at most 28 cut rows, plus 28×500 trivial-sized subproblem solves).

The reason Benders wins so dramatically: every iteration adds ONE constraint to the master. Without warm-starting, each master solve is from scratch; with warm-starting (our `IncrementalLP`), dual simplex repairs the violated row in one or two pivots. Across all 28 iterations the master uses ~50 total pivots — versus 1587 for monolithic.

### Statistical convergence (Part B)

For the unconstrained 2-product newsvendor, closed form gives z*_true = 1797.86. Replicated SAA (20 seeds per N):

```
N        mean SAA z*       stderr   bias    out-of-sample optimality gap
10       1854.55 ± 23.79   23.79    +56.69  36.77 ± 5.22
100      1799.43 ±  9.41    9.41    +1.57    3.21 ± 1.72
1,000    1797.44 ±  2.43    2.43    -0.42   -0.20 ± 1.33
10,000   1797.86 ±  0.90    0.90    +0.003  -0.54 ± 1.33
```

stderr from N=100 to N=10000 shrinks by **10.48×** vs the theoretical 10× (√100 = 10). The out-of-sample gap (= z*_true − z(SAA-x*) on a separate large sample) shrinks from 36.77 to within noise — the SAA decision becomes statistically optimal as N grows.

### Speedup (Part C) — sample wallclock

```
N=  50    mono =     7 ms ( 157 iters)   Benders =    1 ms (17 iters)   speedup ≈    7×
N= 200    mono =   622 ms ( 634 iters)   Benders =    3 ms (24 iters)   speedup ≈  207×
N= 500    mono = 18948 ms (1587 iters)   Benders =   10 ms (28 iters)   speedup ≈ 1894×
```

The asymptotic speedup is `O(N · iters_mono / cuts_Benders)` — Benders is essentially **dimensional reduction**: it never represents the per-scenario second-stage variables explicitly, only their dual contributions to the cut.

### Why this matters for the framework

This module establishes the answer to "is DES + MDP + LP a viable substrate for stochastic optimization?" — **emphatically yes, and pivoting is the critical engineering detail that makes it practical**. Three architectural observations:

1. **DES generates the scenarios** (Monte-Carlo simulation of the random data ω) — exactly the role our DES engine has been playing for SEIR, FactMachine, dispatch, etc.
2. **LP (with pivoting)** does the inner optimization. The incremental LP is the right substrate for the master because every Benders cut is exactly an `applyAddConstraint` event, which dual simplex repairs in O(few) pivots from the previous optimal basis.
3. **MDP framing** organizes the multi-stage extension: SDDP (Stochastic Dual Dynamic Programming) is multi-stage Benders applied recursively, and Benders cuts are precisely the piecewise-linear value-function approximation that MDP value iteration would produce in the linear-Bellman case. **Benders ≡ value iteration for SLPs**.

The same architecture extends naturally to:
- **Multi-stage SLP via SDDP** — each stage's LP repeatedly resolved with a growing pool of cuts.
- **Chance-constrained LP via scenario approximation** — sample violations become cuts.
- **Branch-and-bound for MILP** — each B&B node adds branching constraints; the parent's basis warm-starts the child.
- **Column generation / Dantzig-Wolfe** — symmetric to Benders but adds variables instead of constraints; same incremental-LP support, just `applyAddVariable` instead of `applyAddConstraint`.

## Incremental / warm-startable LP solver as DES (2026-05-25)

Previous LP work in this repo solved problems **once**: build the LP, call simplex (in-process or DES-wrapped), get the answer. Real businesses don't work that way — labour rules change, products are added, prices shift, and a binding constraint last quarter can become irrelevant this quarter. This change adds an **incremental, online LP solver** built strictly with the DES architecture, where every modification is a movable arriving at the LP tableau station and every pivot is a tick.

### What this change adds

| File | Purpose |
|---|---|
| `general/incremental-lp.ts` | The `IncrementalLP` class: a dense-tableau two-phase parametric simplex with five modification operations — `applyAddConstraint`, `applyRemoveConstraint`, `applyChangeObjective`, `applyAddVariable`, `applyRemoveVariable` — and a single `step()` method that performs one primal-or-dual pivot per call. Each modification preserves exactly one of the two simplex invariants (primal feasibility `x_B ≥ 0`, dual feasibility `c̄_N ≤ 0`); the appropriate flavour of simplex restarts to repair the other. |
| `main-incremental-lp.ts` | A 2-D production-planning scenario that exercises **all 5 modification types and recovery from unboundedness**: starts at max `3w + 5g`, adds a capacity cap (dual restart), changes the objective (no-op since x* still optimal), drops the labour constraint (no work needed), introduces a new product `thingamajig` (primal pivot brings it into the basis), removes the material constraint (LP becomes **unbounded**), adds a budget constraint that re-bounds the LP, changes objective again, and finally retires the gadget product. |
| `animation/scenes/incremental-lp-scene.ts` | A two-panel scene: left, the **2-D feasible polytope** with constraint lines, the objective gradient as a yellow arrow, the simplex trajectory as an orange trail, and the current optimum as a green/yellow dot; right, the **simplex tableau** showing the basis, reduced costs, RHS, mode (primal/dual/optimal/unbounded), and feasibility flags. A red banner flashes for 4 ticks whenever a modification event fires. |
| `runners/validate-incremental-lp.ts` | 9-study cross-validation against the static `solveLPInternal` solver: every modification type in isolation, a 5-step combined sequence, a 3-variable randomised modification stream, and min-LP sense flip. **42/42 checks pass to within 2.27e-13** — the incremental solver agrees with the from-scratch solver bit-for-bit at every intermediate state. |
| `test/incremental-lp-test.ts` | 40 unit tests across 11 groups covering construction, each modification, idempotence, unboundedness detection after constraint removal, snapshot integrity, and a randomised regression sweep over 12 LPs. **40/40 pass.** |

### How it works

The classical simplex algorithm maintains two invariants:

```
primal feasibility :  x_B = B^{-1} b  ≥  0
dual   feasibility :  c̄_N = c_N − c_B^T B^{-1} A_N  ≤  0   (for max)
```

Every modification breaks at most one of them, leaving the other intact:

| Modification | Resulting state | Recovery |
|---|---|---|
| Add constraint `a · x ≤ b` | New slack value = `b − a · x*`. If negative, primal infeasible (dual still feasible). | **Dual simplex** pivots on the violated row. |
| Remove constraint | Both invariants preserved. | No work — but if the LP was bounded only by that constraint it now becomes unbounded; the next primal pivot detects that. |
| Change objective `c` | `x*` still feasible, but reduced costs may flip sign. | **Primal simplex** restores dual feasibility. |
| Add variable | Primal feasibility preserved; new column may have favourable reduced cost. | **Primal simplex** brings it into the basis if profitable. |
| Remove non-basic variable | Both invariants preserved. | No work. |
| Remove basic variable | Forced primal pivot to knock it out, then drop. | One pivot, then no further work. |

Two implementation gotchas worth recording here, both caught by `runners/validate-incremental-lp.ts`:

1. **`applyAddVariable` must transform the new column.** When you append `a_new` (the variable's column in *original* standard form), the tableau has been pivoted — its rows are no longer `[A | I | b]`, they're `[B^{-1} A | B^{-1} I | B^{-1} b]`. The newly-inserted column must be `B^{-1} a_new`, not `a_new`. Trick: the slack columns (positions `numStruct..numStruct+m-1`) jointly store `B^{-1}` (because they started as `I` and have been multiplied by every pivot since), so `B^{-1} a_new = Σ_k a_new[k] · tab[:, numStruct+k]`. The same trick gives the row-0 reduced-cost entry. Without this fix, `Study 5` and `Study 7.d` fail by ~10 % of objective value.

2. **`applyRemoveConstraint` must drop the row where the slack is currently basic, not row `index+1`.** After several pivots the slack of constraint `i` may live anywhere in the basis. Dropping the wrong row produces a tableau that's still *internally* consistent but no longer represents the intended LP. Fix: locate `r* = row where slack_i is basic` (or pivot it back in if non-basic), then drop row `r*` and column `slack_i`.

### Validation summary

```
Study 1 — Baseline 2D LP                                  PASS (2/2)
Study 2 — Add constraint (dual restart)                   PASS (2/2)
Study 3 — Remove a binding constraint                     PASS (2/2)
Study 4 — Change objective (primal restart)               PASS (2/2)
Study 5 — Add a variable mid-run                          PASS (2/2)
Study 6 — Remove a variable mid-run                       PASS (2/2)
Study 7 — Sequence of all 5 modifications                 PASS (11/11)
Study 8 — Randomised 3-variable modification stream       PASS (17/17)  max|Δ|=2.13e-14
Study 9 — min-LP sense flip                               PASS (2/2)

42 / 42 checks pass.   40 / 40 unit tests pass.
```

### Animation

Run with `ANIMATE=1 npm run incremental-lp` to write `out/incremental-lp.html`. The animation has **44 ticks** and shows:

- **The polytope physically reshape** when constraints are added/removed (new constraint line slides in; old line vanishes; the feasible region grows or shrinks accordingly).
- **The objective gradient arrow rotates** when the obj vector changes.
- **The optimum dot slides between vertices** as primal/dual pivots fire.
- **A red banner flashes** for 4 ticks each time a modification event arrives.
- **The tableau panel updates** showing the current basis, reduced costs, RHS, mode (PRIMAL/DUAL/OPTIMAL/UNBOUNDED), and feasibility flags in real time.
- A telemetry chart at the bottom plots `z` over time, with discontinuities clearly visible at the modification ticks.

When the demo enters the unbounded phase (tick 22, after dropping the material constraint), the polytope view stretches because the LP is now unbounded; the budget constraint added at tick 26 pulls everything back into a finite region and the simplex resumes pivoting.

### Why this matters for the framework

This module shows that **the DES engine is a viable substrate for the kind of online optimisation that drives MILP branch-and-bound, dynamic LP-based MPC, and any setting where the LP changes faster than you can re-solve from scratch**. Every modification is a movable; every pivot is a tick; the tableau is a long-lived station that swallows events and emits pivots. The same pattern would work for incremental LU factorisation, online QP, and the dual-feasible warm-start tricks that make commercial solvers fast.

## FactMachine math audit + hardening against the production monorepo (2026-05-25)

The DES POMDP simulation `main-factmachine.ts` historically used a generalised log-sum-exp LMSR class that treated its `liquidity` argument **directly as the cost-function parameter `b`**. Production (`factmachine-monorepo/packages/math/src/trading/lmsr.ts`) uses a different convention: the user-facing parameter is "initial liquidity" L (USDC), and `b = L / ln(2)` (the textbook relation `L_max = b·ln(N)` with N = 2). Trades there are **budget-driven** (USDC in → shares out), fees are **proportional in basis points**, and PnL bookkeeping uses **weighted-average cost basis**. None of those conventions were enforced in our DES.

### What this change adds

| File | Purpose |
|---|---|
| `general/factmachine-math.ts` | Pure float64 mirror of the production `@factmachine/math/trading` API: `bFromLiquidity`, `optionPrices`, `lmsrCost`, `buyExecution`, `sellExecution`, `maxPriceWithSlippage`, `minPriceWithSlippage`, `recapitalization`, `avgCostBasis`, `netPosition`, `unrealizedPnl`, `finalPnl`, `replayOrders`, `buyThenSellRoundTrip`. Each function is documented with its production formula. |
| `main-factmachine.ts` (refactor) | `LMSR` class now stores both `liquidity` (the user-facing L) and a derived internal `b`; default semantics is `b = L / ln(N)` (production-equivalent), with `liquidityIsB: true` opt-in for the legacy DES experiments. Adds `binaryPrices()`, `buy(amount, isOptionOne, feeBps)`, `sell(sharesOut, isOptionOne, feeBps)`, and `recap(newLiquidity)` that delegate to the math layer. Also exports `liquidityToB(L)`. |
| `runners/validate-factmachine-math.ts` | Audit + cross-validation runner: 15 invariant studies (Part A) + 14 bit-level cross-checks against the actual production `@factmachine/math/trading` package (Part B) + 6 end-to-end checks on the LMSR class (Part C). |
| `test/factmachine-math-test.ts` | Focused unit tests across 11 groups (`bFromLiquidity`, prices, cost monotonicity, buy/sell exec, round-trip, recap, slippage, PnL helpers, replay, edge cases). |

### Cross-validation against the production package

The runner imports `factmachine-monorepo/packages/math/dist/trading.js` directly (via `require`) and compares each output to ours. Production uses `decimal.js` arbitrary precision, ours uses float64 — they agree to **machine epsilon**:

```
B1  bFromLiquidity   ≡  computeBFromInitialLiquidity   (30 trials)  max|Δ| = 7.28e-12
B2  optionOnePrice  ≡  computeOptionOnePrice            (50 trials)  max|Δ| = 1.11e-16
B3  lmsrCost        ≡  computeLmsrCost                  (50 trials)  max rel|Δ| = 2.03e-16
B4a buyExecution.shares       ≡  prod                   (50 trials)  max|Δ| = 5.68e-14
B4b buyExecution.buyAmount    ≡  prod                   (50 trials)  max|Δ| = 1.42e-14
B5a sellExecution.usdcOut     ≡  prod                   (50 trials)  max|Δ| = 1.05e-11
B5b sellExecution.sellAmount  ≡  prod                   (50 trials)  max|Δ| = 1.09e-11
B6  slippage         ≡  computeMax/MinPrice              (30 trials)  max|Δ| = 1.11e-16
B7a recapitalisation shares   ≡  prod                   (30 trials)  max|Δ| = 1.14e-13
B7b recapitalisation Δcapital ≡  prod                   (30 trials)  max|Δ| = 1.46e-11
B8a–d replayOrders {netPosition, realizedPnl, avgCostBasis, totalVolume} ≡ getOrderAggregates
```

This means the DES POMDP simulator now uses **algebraically equivalent** market-maker math to whatever a live factmachine market would compute on-chain, modulo float64 precision.

### Hardened invariants enforced inside our DES

- Prices sum to 1 and lie strictly in (0, 1) on a 200-trial random grid
- Equal shares ⇒ price = 0.5 to 1e-12
- Symmetry: p₁(q1, q2) = p₂(q2, q1)
- Strict monotonicity in shares
- `buy.feeAmount + buy.buyAmount = amount`
- `buy.shares > 0` whenever `amount > 0`
- `buy.averagePrice = buyAmount / shares < 1` (LMSR convexity)
- `buy.shares` is monotone in `amount`
- `sell.usdcOut + sell.feeAmount = sell.sellAmount`
- `sell.usdcOut` is monotone in `sharesOut`
- **No-arbitrage**: round-trip `sellAfterBuy − buyAmount ≤ 0` on every trial; the maximum observed market-maker spread is **7.28e-12** (i.e. the spread is genuinely zero modulo numerics)
- Recapitalisation preserves prices (`max|Δp| < 1e-10`)
- Slippage maxPrice ≥ price ≥ minPrice, both clamped to [0, 1]

### LMSR class, before vs after

```
before:                                   after:
  new LMSR(50, 2)                            new LMSR(50, 2)
  → liquidity is treated as b                → liquidity is treated as L
  → effective b = 50                         → effective b = 50/ln(2) ≈ 72.13
  → equiv. to L = 50·ln(2) ≈ 34.66 USDC      → equiv. to factmachine-monorepo "L = 50 USDC"

  market.cost([+1, 0])                       market.buy(10, /* isOptionOne */ true, /* feeBps */ 0)
  → return cost in unitless q-space           → returns {shares, buyAmount, averagePrice, feeAmount, reward}
                                              → mirrors `getBuyExecution` from production
```

The legacy semantics is still available via `new LMSR(50, 2, {liquidityIsB: true})` so the existing `validate-factmachine.ts` study (24 / 24 still passing) keeps reproducing exactly.

### Run it

```bash
npm run validate-factmachine-math    # 35 checks: 15 invariants + 14 prod cross-checks + 6 LMSR
npm run test-factmachine-math        # 57 unit tests across 11 groups

# Override the production package path if you cloned the monorepo elsewhere:
FACTMACHINE_TRADING_PATH=/path/to/factmachine-monorepo/packages/math/dist/trading.js \
  npm run validate-factmachine-math
```

### Animation refresh

Two contrasting animations now live in `out/`:

- `out/factmachine-binary.html` — binary market with QMDP policy; bettor's belief converges to true θ as order flow accumulates, P(YES) tracks E_b[θ] over 24 trading periods, with side-by-side belief-histogram and price-trajectory views.
- `out/factmachine-scalar-lateflip.html` — 21-bin scalar market with `LATE_FLIP=1`: a coordinated late-voter surge fires at t=T-2 with the FLIPPED signal at 10× normal volume. You can directly see the entropy SPIKE near the end as the scalar price vector smears across multiple bins and the bettor's belief mass shifts — the PDF's headline non-obvious phenomenon, now visible.

## Genetic Algorithm for TSP (with branch cutting) and Shortest-Path-as-DES (2026-05-25)

Two more layered-architecture demonstrations:

### Genetic algorithm for the Travelling Salesman Problem

Models a complete GA — Selection, Crossover (Order-Crossover), Mutation, Feasibility, Fitness, Replacement — as a sequence of stations inside the DES, with a CHROMOSOME (city permutation) as the movable. One DES tick = one generation. Branch cutting is a real station: when precedence constraints `(i, j)` are present, infeasible offspring are dropped (with bounded retry), penalised, or repaired according to the `feasibility` policy.

| File | Purpose |
|---|---|
| `general/genetic-tsp.ts` | TSP instance, distance/precedence helpers, OX / inversion / swap / tournament operators, `runGeneticTSP` solver, Held–Karp exact DP for n ≤ 16, 1-tree relaxation lower bound |
| `main-genetic-tsp.ts` | CLI: random / pentagon instances, optional precedence, optional animation |
| `animation/scenes/genetic-tsp-scene.ts` | Cities + closed elite-tour polygon + best/mean tour-length charts |
| `runners/validate-genetic-tsp.ts` | Six studies: pentagon optimum, Held-Karp match, 1-tree bound respect, precedence-cut feasibility, policy comparison, monotone elitism |
| `test/genetic-tsp-test.ts` | 18 unit tests across 10 groups |
| `external-references/genetic-tsp/tsp.py` | Python: nearest-neighbor, Held–Karp DP, 1-tree bound |

#### Branch cutting in action

With 4 precedence pairs `(0,11)(1,10)(2,9)(3,8)` on a 12-city instance:

```
Total feasible children evaluated  = 4540
Total infeasible children cut      = 293         ← the cut station fired 293 times
Best tour valid permutation?       = true
Best tour feasible (precedence)?   = true
```

Three constraint-handling policies are supported and compared in Study 5: `cut` (drop infeasible kids), `penalize` (objective += 10⁶ per violation), `repair` (swap-fix). On the 16-city test instance, `cut` and `repair` both produce feasible best-tours; `penalize` is sometimes inferior because the GA wastes search on artificially-priced infeasible regions.

#### Validation against ground truth

- Pentagon n=5 (analytical optimum = `n·2R·sin(π/n)`): GA matches to **1e-9**
- Random n=10 (Held–Karp exact): GA within **0.5%** of optimum across 3 seeds
- 1-tree relaxation lower bound is non-trivial and respected on all instances

### Shortest-path-as-DES (the "graph IS the DES" architecture)

Each graph node IS a stationary entity holding its current distance estimate. Each "wave" message IS a movable carrying a distance update along an edge. Two algorithms in this architecture:

- **`shortestPathBellmanFordDES`** — iterative relaxation: every dirty node broadcasts to its neighbours each tick. Converges in ≤ |V|-1 iterations on graphs without negative cycles; iteration |V| catches negative cycles reachable from source.
- **`shortestPathDijkstraDES`** — priority-queue scheduling of the same fixed-point computation. One node settled per tick. Refuses negative weights.

| File | Purpose |
|---|---|
| `general/shortest-path-des.ts` | Graph type, both algorithms, indexed binary-heap PQ, path reconstruction, random / chain graph builders |
| `main-shortest-path.ts` | CLI with `ALGO=bellman-ford|dijkstra|both`, optional animation |
| `animation/scenes/shortest-path-scene.ts` | Graph layout, node colour by distance, edges flash yellow when a wave fires this tick, distance trace chart |
| `runners/validate-shortest-path.ts` | Six studies including BF≡Dijkstra cross-check on 5 random graphs and explicit negative-cycle detection |
| `test/shortest-path-test.ts` | 43 unit tests across 10 groups |
| `external-references/shortest-path/sp.py` | networkx single-source Bellman-Ford and Dijkstra |

#### Numerical agreement

On the 5-node chain `0(s) → 1(a) → 2(b) → 3(c) → 4(t)`:

```
Bellman-Ford-DES distances = [0, 1, 3, 5, 6]   (ours)
Dijkstra-DES   distances   = [0, 1, 3, 5, 6]   (ours)
networkx Bellman-Ford      = [0, 1, 3, 5, 6]   (Python ref)
networkx Dijkstra          = [0, 1, 3, 5, 6]
```

On random non-negative graphs (5 seeds × 12 nodes): BF ≡ Dijkstra to **0.00e+0** difference.

#### Wave-count theorem in operational form

Validation Study 6 confirms the textbook claim "Dijkstra processes each edge at most once": across 3 dense random graphs, Dijkstra emits **fewer waves** than Bellman-Ford every time:

```
seed=1:  BF waves = 154, Dijkstra waves = 104
seed=7:  BF waves = 126, Dijkstra waves =  91
seed=42: BF waves = 118, Dijkstra waves =  97
```

This makes the cost difference between the two algorithms — both expressed in the same DES — a directly observed quantity.

### Tests / validation

- `test/genetic-tsp-test.ts`        : **18 / 18**
- `test/shortest-path-test.ts`      : **43 / 43**
- `runners/validate-genetic-tsp.ts` : **16 / 16**
- `runners/validate-shortest-path.ts`: **23 / 23**

### Run it

```bash
# Genetic-TSP
npm run genetic-tsp                                    # 25-city random
INSTANCE=pentagon N_CITIES=12 npm run genetic-tsp     # pentagon vs Held–Karp
PRECEDENCE=1 FEASIBILITY=cut npm run genetic-tsp      # branch-cutting active
PRECEDENCE=1 FEASIBILITY=penalize npm run genetic-tsp # comparison policy
ANIMATE=1 N_CITIES=20 GENERATIONS=80 npm run genetic-tsp
npm run validate-genetic-tsp                           # 16 / 16 studies
npm run test-genetic-tsp                               # 18 / 18 tests

# Shortest-path-as-DES
npm run shortest-path                                  # 5-node chain demo
N_NODES=12 ALGO=both ANIMATE=1 npm run shortest-path  # both algorithms + animation
ALGO=dijkstra N_NODES=20 npm run shortest-path
npm run validate-shortest-path                         # 23 / 23 studies
npm run test-shortest-path                             # 43 / 43 tests
```

## 7v7 youth soccer rotation: assignment + scheduling under fairness, with state augmentation as the headline lesson (2026-05-25)

The user's actual coaching problem: 12 players, 7 on field, 5 on bench every 20 minutes, no player benched two consecutive periods, with player–position–period affinities. Instantiated as the canonical **multi-period bipartite assignment with linking constraints**.

### State augmentation — the conceptual centrepiece

A Markov chain is by definition memoryless, so any constraint that depends on the past must be encoded in the *current* state — the standard "lifting to k-step Markov" theorem. The codebase makes this concrete with two side-by-side MDPs over the same problem, same Hungarian inner solver, same affinity tensor:

| MDP variant | State | Affinity | Fairness violations |
|---|---|---|---|
| `policyMDPVIMemoryless` | `(t,)` | **21.13** | **13** |
| `policyMDPVI` | `(t, prev_bench)` | **20.15** | **0** |

The 0.98 affinity gap is *exactly* the cost of the fairness constraint, and the per-player bench counts make the failure mode tangible: the memoryless MDP benches `P7 = P10 = P12 = 4` (sat the entire game) while five other players never sat — because state `(t,)` simply cannot remember who sat last period. Augmenting the state is what the constraint requires.

### Files added

| File | Purpose |
|---|---|
| `general/hungarian.ts` | Jonker–Volgenant O(n³) Hungarian algorithm for bipartite assignment (max or min, square or rectangular) |
| `general/soccer-rotation.ts` | Problem definition, affinity-tensor builder, DES match simulator, schedule evaluator + fairness audit, five scheduling policies (random, greedy-Hungarian per period, multi-period LP relaxation, memoryless MDP, augmented MDP-VI) |
| `animation/scenes/soccer-scene.ts` | Pitch + bench animation: 7 position circles, 5 bench seats, scoreboard, affinity bar, goal flashes, period-boundary "SUB WINDOW" watermark |
| `main-soccer-rotation.ts` | CLI runs all five policies, simulates N matches per policy, optionally animates the best |
| `runners/validate-soccer.ts` | Five validation studies (LP-MDP equivalence, state-augmentation, cross-solver agreement, Hungarian dominance, DES Welch-t) |
| `test/soccer-test.ts` | 49 unit tests across 9 groups including Hungarian textbook problems |
| `external-references/soccer/soccer.py` | scipy reference for the multi-period LP and per-period Hungarian |

### LP relaxation

The 0/1 program is

```
max  Σ_{p,pos,t}  affinity[p][pos][t] · x_{p,pos,t}
s.t. Σ_pos x_{p,pos,t} ≤ 1                          ∀ p, t
     Σ_p   x_{p,pos,t} = 1                          ∀ pos, t
     Σ_pos x_{p,pos,t} + Σ_pos x_{p,pos,t+1} ≥ 1    ∀ p, t < T-1     (fairness)
     x_{p,pos,t} ∈ {0, 1}
```

Its LP relaxation has 12 × 7 × 4 = **336 variables** and 28 equality + 84 inequality rows. On the seed=4242 instance:

- internal simplex     → 20.14944862
- DES-engine simplex   → 20.14944862
- scipy:HiGHS-DS       → 20.14944862
- scipy:HiGHS-IPM      → 20.14944862
- MDP-VI exact         → 20.14944862
- python scipy ref     → 20.1494486232954 (matches to 9 digits)

The LP relaxation is **tight** here (zero integrality gap), so the LP value is provably the constrained optimum.

### DES match simulator (Layer 1)

`simulateMatchDES` runs 80 game-minutes per match, samples Poisson goal events from on-field affinity (team rate ∝ avg affinity², opponent rate ≈ const), records substitution events at every 20-min boundary, and produces a per-tick trace consumed by the animation scene.

### Animation

`ANIMATE=1 npm run soccer` writes a self-contained HTML player showing the pitch with 7 numbered player circles in canonical 7v7 positions, the 5-player bench beside the pitch, a live affinity bar, scoreboard, and a "SUB WINDOW" flash at every period boundary. Goal events flash yellow (us) / red (them) for one tick.

### Tests / validation

- `test/soccer-test.ts`: **49 / 49** unit tests pass
  - Hungarian textbook 3×3 min/max + rectangular cases
  - Schedule structure, fairness behaviour, LP shape, DES invariants, reproducibility
- `runners/validate-soccer.ts`: **16 / 16** validation checks pass
  - Study 1: LP upper bound = MDP-VI optimum (LP tight, integrality gap = 0)
  - Study 2: state-augmentation principle (memoryless violates, augmented enforces, cost = 0.98)
  - Study 3: four LP solvers agree to **7.11e-15**
  - Study 4: greedy-Hungarian dominates random by ≈ 8 affinity points
  - Study 5: DES Welch-t > 5 for MDP-VI vs random goal differential

### Run it

```bash
npm run soccer                                 # head-to-head, 100 matches per policy
N_MATCHES=200 SEED=99 npm run soccer            # different instance, more reps
LP_SOLVER=scipy:highs-ipm npm run soccer        # interior-point on the rotation LP
ANIMATE=1 npm run soccer                        # pitch + bench animation HTML
POLICY=mdp ANIMATE=1 npm run soccer             # animate just the MDP-VI schedule
npm run validate-soccer                         # five Welch-t studies
npm run test-soccer                             # 49 unit tests
```

## DES + MDP + LP + MCTS combinatorial-optimisation combo (2026-05-25)

The next layer up from the LP integration. The user observation:

> I am interested in using DES to help reformulate / reframe problems that
> can be solved with LP or interior-point algos etc, and basically using
> DES to solve almost any problem via combinatorial methods (slower but
> effective and handles stochastics well).

This change instantiates that idea on the canonical **multi-class parallel-server dispatch** problem (M heterogeneous machines, K job classes, Poisson arrivals, class-dependent exponential service, FIFO queues, decision = which machine on every arrival). Six policies all evaluated by the SAME DES with the SAME seeds:

| Layer | File | Method |
|---|---|---|
| Layer 1 — physical world | `general/dispatch.ts` (`simulateDispatch`) | next-event DES, exponential inter-event times |
| Layer 2 — decision abstraction | `general/dispatch.ts` (`policyMDPVI`) | MDP at arrival decision epochs; transitions estimated by DES rollouts |
| Layer 3 — optimisation engines | `general/dispatch.ts` + `general/mcts.ts` | random / round-robin / shortest-queue / SECT / LP fluid relaxation / value iteration / UCT-MCTS |

### Key files added

| File | Purpose |
|---|---|
| `general/mcts.ts` | Generic UCT (Kocsis–Szepesvári) with pluggable rollout policy; works for any DES whose state can be cloned and advanced |
| `general/dispatch.ts` | Problem definition, DES simulator, six policies, fluid-LP builder, empirical-MDP-via-DES + value iteration, MCTS adapter, evaluation harness with Welch t-test |
| `main-dispatch-combo.ts` | CLI that runs all six policies head-to-head on a single instance, prints architecture recap |
| `runners/validate-dispatch.ts` | Five validation studies including cross-solver bit-exact LP agreement and value-function stability |
| `test/dispatch-test.ts` | 72 unit tests across 11 groups |
| `external-references/dispatch/dispatch.py` | scipy reference for the fluid LP and Hungarian assignment |

### LP fluid relaxation (the bridge from combinatorial → continuous)

For each class c, let `x_{c,m}` be the long-run fraction of class-c arrivals dispatched to machine m. The fluid LP is

```
min  t                                    (minimise the bottleneck load)
s.t. Σ_m x_{c,m} = 1                  ∀ c    (each class fully served)
     λ Σ_c p_c x_{c,m} / μ_{c,m} ≤ t  ∀ m    (machine-m load ≤ t)
     x_{c,m} ≥ 0
```

This LP — small enough to solve in milliseconds — produces a randomised dispatch policy that minimises the bottleneck load. It's solved by `solveLP()` and therefore inherits the entire LP toolchain: in-process simplex, the DES-engine simplex from the previous changelog entry, scipy:HiGHS-DS (dual-simplex), and scipy:HiGHS-IPM (interior-point) all return identical objectives.

### Empirical MDP via DES rollouts

State = (q_1, …, q_M, c_arriving) truncated at q_max. For every (s, a) the system runs R DES rollouts to estimate P(s'|s,a) and E[r|s,a], producing a tabular empirical MDP that value iteration solves to optimum. This is exactly the user's framing:

> Observe DES state → choose action → DES simulates next event → compute cost/reward → repeat.

Made operational in plain TypeScript. With the small instance (M=2, K=2, q_max=5) the MDP has 72 states and `policyMDPVI` builds + solves it in ~50ms.

### MCTS with DES rollouts

`general/mcts.ts` is a generic UCT implementation that takes any `MCTSEnv<S>` (a clone-able state, an `applyAction`, an optional rollout policy). The dispatch adapter (`policyMCTS`) uses SECT as its default rollout, peeks ahead by sampling future arrivals, and treats each "dispatch decision" node as a UCT node. Per-machine head class is tracked exactly so service times in the search tree are unbiased.

### Empirical results

On the well-specialised instance (M=2, K=2, μ_{c,c}=2.0, μ_{c,c'}=0.8, λ=1.6, ρ̄ ≈ 0.43):

| policy | mean sojourn | architectural layer |
|---|---|---|
| random | 3.32 | Layer 3, trivial baseline |
| round-robin | 2.74 | Layer 3, state-blind heuristic |
| shortest-queue | 2.01 | Layer 3, queue-aware |
| MCTS | 1.61 | Layer 3 ∘ Layer 1 |
| fluid-LP | 1.02 | Layer 3, continuous relaxation |
| MDP-VI | 0.95 | Layer 2 |
| SECT | 0.90 | Layer 3, class-aware (≈ optimal here) |

The headline finding **is not** that one method wins — it's that on a well-specialised instance the class-aware greedy heuristic is already optimal within sampling noise, so layered methods don't separate from it. **All four LP solvers agree on the fluid-LP optimum to 1.11e-16** (`internal simplex` ≡ `DES-engine simplex` ≡ `scipy:HiGHS-DS` ≡ `scipy:HiGHS-IPM`).

On the heavily-loaded weak-specialisation instance (M=3, K=3, ρ̄ ≈ 0.85), the picture inverts — fluid-LP and MDP-VI hold up while round-robin / random degrade catastrophically (mean sojourn random=10.5 vs. SECT=1.31).

### Architectural takeaway

DES is the unifying simulator under all three layers. The combinatorial optimisation methods (LP fluid relaxation via simplex / interior-point; tabular VI on the empirical MDP; MCTS on the DES tree) all use the SAME DES — for evaluation in the case of LP/heuristics, as a transition oracle for VI, and as a rollout simulator for MCTS. This is the layered architecture the user described:

- Layer 1 = DES (physical / dynamic world)
- Layer 2 = MDP (decision abstraction)
- Layer 3 = optimisation engine (simplex, interior-point, value iteration, MCTS, RL, metaheuristics)

The same `simulateDispatch` is the substrate for ALL of them.

### Tests / validation

- `test/dispatch-test.ts`: **72/72** unit tests pass
- `runners/validate-dispatch.ts`: **11/11** validation checks pass
  - Study 1: SECT statistically dominates random / round-robin / SQ
  - Study 2: heavily-loaded regime — fluid-LP, SECT all dominate random
  - Study 3: **internal simplex ≡ DES-engine simplex ≡ scipy:HiGHS-DS ≡ scipy:HiGHS-IPM agree on the fluid LP to 1.11e-16** (the same `solveLP` that we use everywhere else in the project)
  - Study 4: MDP-VI value-function stability under qMax growth
  - Study 5: MCTS bounded by its rollout policy (consistent with the bandit-tree convergence theory)

### Run it

```bash
npm run dispatch                          # head-to-head comparison + LP printout
SKIP_MDP=1 N_REPS=10 npm run dispatch     # fast smoke
LP_SOLVER=scipy:highs-ipm npm run dispatch  # interior-point on the fluid LP
LP_SOLVER=internal npm run dispatch         # in-process simplex
npm run validate-dispatch                 # five Welch-t studies
npm run test-dispatch                     # 72 unit tests
```

## LP integration: external solver, MDP-as-LP, and simplex-in-DES (2026-05-25)

Adds a clean DES↔LP bridge for the simulation-optimisation pattern
(LP gives the nominal optimum; DES gives operational realism). Three
levels of integration:

### 1. Pluggable LP solver

| File | Purpose |
|---|---|
| `general/lp.ts` | `LPProblem` / `LPSolution` types; in-process two-phase revised simplex (small-LP fallback); external-solver dispatcher; `solveLP()` chooses backend via `LP_SOLVER` env var |
| `external-references/lp/lp_solve.py` | Python bridge to `scipy.optimize.linprog` (HiGHS dual-simplex, HiGHS interior-point, legacy simplex/IPM all selectable) |
| `general/des-lp-bridge.ts` | High-level patterns: `solveLPThenSimulate`, `buildMDPLP`, `solveMDPAsLP`, `lpRollingHorizon` |

`LP_SOLVER` env values: `internal`, `scipy:highs` (default), `scipy:highs-ds`, `scipy:highs-ipm`, `scipy:simplex`, `scipy:interior-point`. When the external solver is unavailable, dispatching falls back to the in-process simplex automatically.

### 2. Plan-then-simulate pattern

`main-lp-factory.ts` solves a 3-product × 4-machine factory LP (max profit s.t. weekly machine-time capacities), then feeds the LP plan to a DES factory simulator with stochastic processing times (lognormal, CV=0.25), Poisson machine breakdowns (p=0.002/min), and finite per-machine buffers. The output reveals the LP-vs-realised gap:

```
LP NOMINAL revenue = $34204.72
DES realised mean  = $24356.67  ±  $1147.84  over 30 reps
gap = $9848.06 (28.8% of nominal)
```

A robustness sweep (`SWEEP=1`) shrinks the LP RHS by 5–25% to leave operational headroom; the realised mean then RISES (from $24,357 at robust=1.0 to $24,850 at robust=0.80) because the LP no longer overcommits machines that break down. This is the simulation-optimisation feedback loop in textbook form.

### 3. MDP-as-LP — the deep connection

The Bellman optimality equation has an LP characterisation:

```
min Σ_s μ_s · V(s)
s.t. V(s) − γ Σ_{s'} T(s'|s,a) V(s') ≥ Σ_{s'} T(s'|s,a) r(s,a,s')   ∀ s, a
```

`buildMDPLP` and `solveMDPAsLP` (in `general/des-lp-bridge.ts`) translate any `MDPSpec` into this LP, send it to the external simplex/interior-point solver, and extract V* and π*. `main-mdp-lp.ts` demonstrates on three MDPs (5-state chain, 11-state inventory control, 16-state stochastic grid-world). All three V* match value-iteration to ≤ 6e-13, and the greedy policies are identical:

```
chain:        max|V_LP − V_VI| = 3.33e-16   π_LP ≡ π_VI
inventory:    max|V_LP − V_VI| = 1.81e-11   π_LP ≡ π_VI
gridworld:    max|V_LP − V_VI| = 6.00e-13   π_LP ≡ π_VI
```

This realises the chain `real system → DES → MDP abstraction → LP formulation → simplex / interior-point` end-to-end inside one engine.

### 4. LP-as-DES — simplex implemented inside the DES engine

`general/lp-des.ts` and `main-lp-des.ts` embed simplex as a discrete-event process. Simplex IS naturally a DES: each pivot is an event where the algorithm walks one edge of the feasible polytope to the next vertex.

Four stations, one pivot per tick:

| Station | Role |
|---|---|
| `EnteringStation` | Scans the cost row for the steepest improving direction (Dantzig: most-negative reduced cost; or Bland: first negative reduced cost for guaranteed termination on degenerate LPs) |
| `LeavingStation` | Min-ratio test: how far along the entering edge can we travel before some basic variable hits 0? |
| `PivotStation` | Elementary row operations on the tableau (Gauss-Jordan elimination) |
| `ObserverStation` | Snapshots the current vertex + objective into the trace, exactly the way `Census` does in our SEIR / FactMachine models |

Two phases of simplex (phase-1 for feasibility when the origin is not a basic feasible solution, phase-2 for optimality) share the same DES tick loop with a different cost row. The trace is a per-pivot record of `(enter, leave, vertex, objective)` that drops directly into our existing animation plugin; the 2-D demo renders the polytope walk as a vertex-by-vertex path through the feasible region.

#### Validation: 36 / 36 LP-validation checks pass (`runners/validate-lp.ts`)

| Study | Claim | Result |
|---|---|---|
| 1 | Internal simplex ≡ scipy:highs ≡ scipy:highs-ds ≡ scipy:highs-ipm on 2-var canonical LP | PASS |
| 2 | Diet LP (Stigler 4-foods × 3-nutrients): internal cost = scipy:highs to 2.22e-16 | PASS |
| 3 | Transportation LP (3×3 balanced equalities): both solvers find $245.00 | PASS |
| 4 | 5-state chain MDP: V*\_LP ≡ V*\_VI bit-exactly; greedy policy identical | PASS |
| 5 | 16-state stochastic grid-world MDP: max\|V\_LP − V\_VI\| = 5.92e-13 | PASS |
| 6 | 200 random feasible LPs: internal ≡ scipy:highs to 1e-7 (max\|Δobj\| = 3.38e-14) | PASS |
| 7 | `LP_SOLVER` env-var dispatch routes to all 4 backends correctly | PASS |
| 8 | DES-engine simplex (Dantzig + Bland rules) ≡ in-process simplex ≡ scipy:highs on canonical LPs (incl. unbounded + infeasible) | PASS |
| 9 | 50 random feasible LPs: DES-engine simplex obj ≡ scipy:highs obj (max\|Δ\| = 8.88e-15) | PASS |

#### Why solve LP via DES if it's slower?

We are NOT claiming DES-as-simplex beats scipy:HiGHS on speed. A direct simplex would run 5–10× faster because the per-tick scaffolding (status checks, history captures, phase transitions) is overhead. We did this to:

  1. **Prove the engine is computationally general.** Events, queues, stations, and movables are a sufficient substrate for vertex-walking optimisation, even when the LP's polyhedral geometry would normally invite a specialised representation.
  2. **Get the trace + animation infrastructure for free.** The same `FrameRecorder` / `HtmlPlayer` used for SEIR + FactMachine + heat-PDE works directly on the per-pivot vertex history. The 2-D demo (`PROBLEM=2var-diamond ANIMATE=1`) renders the polytope walk as an animated path with no LP-specific scene code beyond mapping vertices to pixels.
  3. **Bridge LP / MDP / simulation in one engine.** `main-lp-des.ts` (DES-driven simplex), `main-mdp-lp.ts` (MDP-as-LP via simplex), and `main-lp-factory.ts` (LP-then-DES) all run on the same TypeScript runtime with the same animation plugin and validation harness.

## FactMachine POMDP: opinion market as a station network (2026-05-25)

Models the FactMachine opinion-market platform as a Partially Observable
MDP. Refactored to fit the engine's stationary-entity / movable-entity
discipline: the simulation IS a five-station DES, not an ad-hoc loop.

### Architecture: five stations, two movable types

| Station | Role | Reads | Writes |
|---|---|---|---|
| `NoiseTraderStation` | Generates `K_noise` Poisson-distributed orders per tick. Late-flip surge multiplies K and inverts the signal at t = T-2. | `params.trueTheta`, `params.informedness` | `market.noiseQueue` (Order movables) |
| `MarketStation` | Owns the LMSR market maker (generalised to N outcomes). Drains queues in two phases: `settleNoise()` then `settleBettor()`. Records last-tick aggregates so the bettor can read them. | `noiseQueue`, `bettorQueue` | LMSR state, `lastNoiseYes/Total`, `lastBettorCost/Side` |
| `WorldCensus` | Snapshots post-noise prices and the bettor's belief, exactly the way `Census` works in the SEIR model. Trace-only; never mutates simulation state. | `market.lmsr`, `bettor.belief` | per-tick frame captured by the orchestrator |
| `BettorStation` | Bayesian filter over θ ∈ {θ₁, …, θ_K}. Uses `pickAction(τ)` (oracle / myopic / qmdp / random / hold) to choose an outcome to buy 1 share of. | `market.lastNoise{Yes,Total}`, live LMSR prices | `market.bettorQueue` (Order movable), own `belief`, `shares`, `cash` |
| `ResolutionStation` | At t = T: generates `N_voters` Vote movables, decides the winning outcome (`bernoulli` or `majority` rule), settles the bettor's payout. | `bettor.shares`, `params.trueTheta`, `params.resolutionMode` | final outcome, `voteFraction`, `payout` |

Movables: `Order { kind, side, isYes? }` and `Vote { value: 0|1 }`.

### Per-tick orchestration order (deterministic, six phases)

1. `NoiseTraderStation.runTimeStep` enqueues orders.
2. `market.settleNoise()` drains them as a single LMSR transaction.
3. `WorldCensus.runTimeStep` snapshots prices + belief into the trace.
4. `BettorStation.runTimeStep` updates b(θ) Bayesianly, picks an action, enqueues its order.
5. `market.settleBettor()` drains the bettor's order.
6. `bettor.applySettlement()` updates shares / cash / fee bookkeeping.

This phased ordering eliminates the order-dependence ambiguity that
mixing all stations in one shuffled tick would cause: noise is settled
first because the bettor's likelihood model needs the post-noise
yes/total aggregate; the bettor settles last because its action is a
function of the post-noise market price.

### Animation strategy: post-hoc only

The simulation builds NO frame data inline. Instead it captures
`beliefSnapshots[t]`, `priceHistory[t]`, `actionHistory[t]`,
`yesOrdersHistory[t]`, `totalOrdersHistory[t]` per tick into the
result. The new `renderAnimation(result, params)` function walks that
trace afterwards and asks the scene builder for SVG shapes per tick,
so:

- the simulation hot-loop never pays for SVG / JSONL work;
- the user can replay at 0.25× → 16× via the HTML player's speed
  selector (already supported by `html-player.ts`);
- the same trace can be rendered into multiple animations
  (e.g. binary vs. scalar comparison) without re-running the model.

`ANIM_FPS` and `ANIM_FRAMES` env vars control default playback rate
and frame-cap (with strided subsampling when T > cap).

### Validation: 24/24 checks pass (`runners/validate-factmachine.ts`)

| Study | Claim | Result |
|---|---|---|
| 1 | Bayesian filter ≡ scipy ground-truth bit-for-bit on a 5-tick observation sequence | PASS |
| 2 | Win-probability under majority rule ≡ `scipy.stats.binom.sf` (compares 9 (θ, K) pairs) | PASS |
| 3 | Brier(t=0) = 0.25 (uniform prior, Bernoulli outcome); Brier decreases monotonically | PASS |
| 4 | Policy ranking: oracle > qmdp > random > hold; Welch-t > 5 for oracle vs random, > 3 for qmdp vs random | PASS |
| 5 | Late-stage manipulation: vanilla Bayesian filter cannot spike entropy (every obs increases precision), but DOES misdirect E[θ] toward 1−θ; PnL drops measurably | PASS — note that the PDF's "entropy spike" requires a hyper-prior over `{legitimate, manipulated}` regimes, future work |
| 6 | Cassandra Tiger POMDP: exact α-vector VI ≤ QMDP at flat prior (QMDP is the standard upper bound); both pick "listen" at b = (0.5, 0.5) | PASS |
| 7 | Binary vs scalar markets: identical belief trajectory at hold-policy (max\|Δ\| = 0); binary myopic win-rate (0.99) ≫ scalar (0.35) at θ = 0.65 ("sure-thing" effect); scalar PnL sd 3× binary; scalar oracle edge 3× binary (info more valuable in continuous outcomes); scalar price-vector entropy 4.5× binary | PASS |

External reference: `external-references/factmachine/factmachine.py`
mirrors the Bayesian filter and majority-vote win-probability with
`numpy` + `scipy.stats.binom`. Studies 1 and 2 verify bit-for-bit
agreement.

### Files

- `general/belief.ts` — `DiscreteBelief<T>` with Bayesian update, propagate, mean, variance, entropy, mode, sample, Brier, KL.
- `general/pomdp.ts` — `POMDPSpec`, `beliefUpdate`, `mdpValueIteration`, `QMDPSolver`, `MostLikelyStateSolver`, `pomdpExactFiniteHorizon` (α-vector pruning).
- `main-factmachine.ts` — five Station subclasses + `runFactMachine` orchestrator + `renderAnimation` post-hoc renderer + CLI dispatch (single-rep / multi-rep policy comparison / binary-vs-scalar).
- `external-references/factmachine/factmachine.py` — scipy reference for filter + win-prob.
- `runners/validate-factmachine.ts` — 7 validation studies, 24 checks.
- `animation/scenes/factmachine-scene.ts` — belief-histogram + price-time-series + entropy-time-series + order-flow scene.


## Calculus engine: ODE / PDE / Poisson as station networks (2026-05-25)

Adds a tiny symbolic expression engine and a "math-expression-as-DES-model"
pipeline. The user supplies an ODE system or PDE in expression-string
form; the framework parses it, builds a station network where each
state variable (ODE) or each spatial cell (PDE) is its own station, and
the simulation engine runs the standard tick loop. The per-station
update rule encodes the discretisation scheme. There is no separate
"ODE engine" or "PDE engine" — the same `Census` + `FieldStation` +
`FieldSimulation` machinery used for compartmental epidemics now solves
calculus problems too.

### What's new

| File | Purpose |
|---|---|
| `general/expr.ts` | Tiny symbolic expression engine. AST (`NumNode`, `VarNode`, `BinNode`, `UnaryNeg`, `FuncNode`), recursive-descent parser, programmatic builders (`add`, `mul`, `pow`, `sin`, `cos`, …), `evaluate`, `toFunction` (compiles to a JS `(...) => number`), `stringify`, `simplify`, symbolic `diff` (chain rule, product rule, quotient rule, all elementary functions), and Richardson five-point numerical-derivative cross-check. |
| `general/quadrature.ts` | Reference 1-D and N-D quadrature: trapezoidal, Simpson, adaptive Simpson (recursive splitting to a tolerance), Gauss-Legendre (n ≤ 10), Monte Carlo (1-D and N-D). |
| `general/root.ts` | Root-finding: bisection, Newton (with analytical derivative), secant. Returns iteration count + convergence flag. |
| `general/optim.ts` | Multivariable minimisation: gradient descent (Armijo line search), Newton-with-Hessian (linear solve fallback), BFGS (quasi-Newton). |
| `general/ode.ts` | Reference ODE solvers: forward Euler, Heun (RK2), classical RK4, RK45 Dormand-Prince adaptive, backward Euler implicit. Used as ground truth that the station-network solvers must match. |
| `general/field-station.ts` | The framework substrate: `Station` base class, `Census` (snapshots all values at start of every tick), `FieldStation` (holds a scalar; updater reads the snapshot, never another live station's value), `FieldSimulation` (drives the tick loop, optional shuffle, optional trace). |
| `general/equation-to-stations.ts` | Parses ODE/PDE specs and constructs station networks. `buildODESystem` (one station per state variable, scheme ∈ {euler, rk2, rk4}); `buildField1D` (one station per spatial cell for heat / wave / advection, schemes ∈ {ftcs, btcs, leapfrog, upwind}); `solvePoisson2D` (Nx·Ny grid as stations, schemes ∈ {jacobi, gauss-seidel, sor}). Includes `thomas` tridiagonal solver for BTCS. |
| `main-calculus.ts` | Entry point: `PROBLEM=expr|ode|pde|poisson` dispatcher. Reads the equation from env vars, builds the station network, runs, prints comparison vs analytical / pure-math / scipy. Supports `ANIMATE=1` for SVG/HTML field-evolution scenes. |
| `external-references/calculus/calculus.py` | Cross-validation reference: scipy.solve_ivp DOP853 for ODEs, scipy.LSODA on the same FD heat system, hand-coded Jacobi for 2-D Poisson, scipy.integrate.quad for quadrature. Outputs JSON the TS validator parses. |
| `runners/validate-calculus.ts` | Six studies (26/26 PASS): symbolic ≡ numerical derivative, quadrature methods agree, station-net RK4 ≡ pure-math RK4 (bit-level), 1-D heat FTCS/BTCS ≡ analytical, leapfrog wave ≡ standing-wave analytical, station Jacobi ≡ scipy Jacobi (bit-level). |
| `test/calculus-test.ts` | Unit tests (31/31 PASS): expression engine, quadrature, ODE convergence orders, station-net ≡ pure-math bit-equivalence, FTCS/BTCS stability bounds, station-update order-independence under shuffle, Thomas tridiagonal, SOR < Gauss-Seidel < Jacobi iteration ordering. |
| `animation/scenes/calculus-scene.ts` | Field-evolution scene: 1-D PDE rendered as a strip of N station bars (height = u_i, colour = signed magnitude), peak-amplitude time series. 2-D Poisson rendered as Nx×Ny false-colour grid. |

### Architectural choice: math expression → station network

The user's request was *"the math expression should be parsed and turned
into our DES model with stations and movables and then solve it that way."*
That is exactly what this layer does:

  - **ODE system** `y'_i = f_i(t, y_1, …, y_n)` → one station per
    state variable. Each station's `runTimeStep` evaluates the RHS
    expression using the census snapshot of the other variables, then
    advances its own value via Euler / RK2 / RK4. Because every station
    reads only the snapshot (never a peer's mid-tick value), the
    multi-station evaluation is order-independent and produces
    bit-identical output to the pure-math solver running on a dense
    vector. This is verified to f64 precision in `validate-calculus.ts`
    Study 3 (max|Δ| = 0).
  - **1-D PDE** `u_t = α u_xx`, `u_t + a u_x = 0`, `u_tt = c² u_xx` →
    one station per spatial cell `x_i`. Each station reads its left
    and right neighbours from the snapshot to compute the discrete
    Laplacian or upwind difference. FTCS, leapfrog and upwind are
    pure local rules (no inter-cell coupling within a tick); BTCS
    requires a single per-tick tridiagonal solve, implemented via the
    Thomas algorithm in a thin wrapper around the same station network.
  - **2-D Poisson** `∇²u = −ρ(x, y)` → Nx·Ny stations on a 2-D grid.
    Each iteration ("tick") is one Jacobi / Gauss-Seidel / SOR sweep
    over the five-point stencil. The relaxation continues until
    max|Δu| < tol. Order-independence holds for Jacobi (bit-equivalent
    output regardless of cell processing order); Gauss-Seidel and SOR
    deliberately use the live (already-updated) values for faster
    convergence and therefore are NOT shuffle-invariant by design.

### Validation highlights

**Bit-level station-network ≡ pure-math RK4 (Study 3):** For SHO
y'' + y = 0 from t = 0 to t = 4π at dt = 0.001, the 2-station network
(`y` station + `v` station, RK4 scheme reading the census snapshot)
produces |Δ| = 0 from `general/ode.rk4` running on the dense 2-vector.
This is the strongest possible agreement (identical IEEE 754 doubles),
demonstrating the station refactor is mathematically equivalent.

**Bit-level station Jacobi ≡ scipy Jacobi (Study 6):** On a 41×41
Poisson grid with `ρ = 2π² sin(πx)sin(πy)` and tol = 1e-8, both the
station network and scipy's hand-coded Jacobi terminate after exactly
4095 iterations with identical max error 5.11e-4 vs the analytical
`u = sin(πx)sin(πy)`. The Jacobi update is a pure linear functional
of the snapshot, so this bit-equality is reproducible and not coincidental.

**SOR ~30× faster than Jacobi at ω = 1.85:** All three relaxation
schemes converge to the same answer, but SOR completes in 149
iterations vs Jacobi's 4095 (≈ 27× speedup) at the same tolerance —
the canonical iterative-solver result, reproduced by the station
network without any special-casing.

**FTCS stability bound enforced; BTCS unconditionally stable:** With
α = 0.1 and N = 51 (dx = 0.02), the FTCS bound is dt ≤ dx²/(2α) = 2e-3.
At dt = 1.6e-3 the FTCS station network gives max|err vs analytical|
= 6.2e-4 over T = 0.5. The BTCS network at dt = 5e-2 (25× the FTCS
bound) gives max|err| = 7.3e-3 in 10 ticks — still stable, just
larger truncation error per step.

**Symbolic ≡ numerical derivative across 6 expressions:** AST symbolic
differentiation of `x²`, `sin(x)·cos(x)`, `exp(-x²)`, `x³+2x²-5x+1`,
`log(x)·sin(x)`, `1/(1+x²)` matches Richardson five-point numerical
derivative to ≤ 5e-12 at every probed point.

**Quadrature cross-method agreement:** On `∫_0^π (x² sin x + e^{−x}) dx`,
adaptive Simpson at tol 1e-15 and `scipy.integrate.quad` give bit-exact
agreement (|Δ| = 0.00). Gauss-Legendre n=10 matches to 1.78e-15
(machine ε). Simpson n=64 matches to 3.81e-8. Trapezoidal n=64 has
1.79e-3 error — within its O(1/n²) theoretical bound.

### How it ties into the framework

The station-network calculus layer reuses three framework primitives
without modification:

  - **Synchronous data flow via Census:** the same pattern as
    `main-two-disease.ts`'s `WorldCensus` — read-only snapshot drives
    order-independent local updates. Generalised to a Float64Array of
    arbitrary length.
  - **Per-tick shuffle for order-independence verification:** the
    `FieldSimulation` shuffles the field-station processing order
    every tick by default. T6 in the unit tests proves two different
    shuffle seeds give bit-identical final fields, which would not
    be true if any station read a neighbour's live (mid-tick) value.
  - **Animation pipeline:** `FrameRecorder` + `HtmlPlayer` work
    unchanged on field-station traces — the JSONL frames format is
    agnostic to whether each frame represents an epidemic state or a
    PDE snapshot.

### Usage

```bash
# Single expression: derivative + 5 quadrature methods + scipy reference
PROBLEM=expr EXPR='x^2 * sin(x) + exp(-x)' npm run calculus

# ODE system as a station network
PROBLEM=ode NAMES='y,v' RHS='v;-y' Y0='1,0' DT=0.001 T_END=12.566 npm run calculus

# 1-D heat equation; FTCS + BTCS with same N, animation enabled
PROBLEM=pde FAMILY=heat ALPHA=0.1 N=51 T_END=0.5 ANIMATE=1 npm run calculus

# 1-D wave equation
PROBLEM=pde FAMILY=wave C=1 N=51 T_END=0.5 npm run calculus

# 2-D Poisson — Jacobi vs Gauss-Seidel vs SOR comparison
PROBLEM=poisson N=41 OMEGA=1.85 TOL=1e-8 ANIMATE=1 npm run calculus

# Validation suite (requires python3 + scipy)
npx ts-node src/des/runners/validate-calculus.ts

# Unit tests
npx ts-node src/des/test/calculus-test.ts
```

## Newsvendor + multi-period inventory MDP (2026-05-25)

Adds the canonical pedagogical MDP — the newsvendor problem — and its
multi-period generalisation, both solvable with the framework's value
iteration and pinned against the closed-form analytical solution
(critical fractile) and an external numpy reference.

### What's new

| File | Purpose |
|---|---|
| `general/value-iteration.ts` | Generic finite-state, finite-action MDP value-iteration solver. Decoupled from the USACC court MDP; takes an `MDPSpec` and returns `V*` and greedy policy. Validates probability normalisation; pre-builds transition table for ~50× speedup. |
| `main-newsvendor.ts` | Single-period stochastic inventory: solved by (a) analytical critical-fractile, (b) exhaustive brute search, (c) MDP value iteration. All three produce the same q* by construction. |
| `main-inventory-mdp.ts` | Multi-period extension with state = inventory level, action = order quantity, lost-sales transitions. Discovers optimal policy by VI without structural assumption. Includes `detectPolicyStructure` to classify the discovered policy as base-stock, (s, S), or irregular. |
| `runners/validate-newsvendor.ts` | Five studies (28/28 PASS): 3-method agreement; γ → 0 reduction; structure detection; simulation matches V(0)·(1−γ); bit-exact match against numpy. |
| `test/newsvendor-test.ts` | Unit tests on PMFs, profit identities, critical-fractile edge cases, structural sanity, VI determinism. 32/32 PASS. |
| `external-references/newsvendor/newsvendor.py` | Python reference using numpy. Solves both the newsvendor (CDF inversion) and the multi-period MDP (numpy value iteration) in standalone form. Cross-validates the TS implementation to 1e-3 in V and exact in policy. |
| `animation/scenes/newsvendor-scene.ts` | Per-period scene: stacked bars (start inv, ordered, sold, leftover), demand line, profit metrics, dual time-series chart. |

### Validation highlights

**Critical-fractile match (Study 1):** Across four scenarios (Poisson
λ ∈ {15, 20, 50, 100} and uniform U[0, 10] demand), the analytical
formula `q* = inf{q : F(q) ≥ c_u/(c_u + c_o)}`, brute-search over q,
and MDP value iteration all produce the same q* and the same
E[profit(q*)] to 1e-12 precision.

**(s, S) discovery (Study 3):** The MDP discovers the structural
shapes from data — no built-in assumption of base-stock or (s, S):

```
   K       S*    s*    S − s    structure
   0       26    25       1      base-stock
   1       26    21       5      s-S
   5       45    17      28      s-S
  10       47    14      33      s-S
  25       50    10      40      s-S
  50       50     3      47      s-S
```

The (S − s) gap grows monotonically with the fixed cost K, exactly as
inventory theory predicts. The validator pins this monotonicity as a
PASS gate.

**Cross-engine bit-exact (Study 5):** The TS solver and the standalone
numpy reference agree to 1e-3 in V(0) and **exactly** in the
discovered policy:

```
TS V(0) = 240.9033   policy[0..19] = [47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 0, 0, 0, 0, 0]
Py V(0) = 240.9033   policy[0..19] = [47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 0, 0, 0, 0, 0]
```

### Why this matters for the framework

The newsvendor pinning gives us full confidence in the generic VI
solver, which is now used by:
* USACC court MDP (existing, unchanged)
* Newsvendor (new, pinned against analytical)
* Multi-period inventory MDP (new, pinned against numpy + analytical structure)
* Anyone else who builds an `MDPSpec` going forward

Animation works the same way as the other simulations — set
`ANIMATE=1` and a frames JSONL + standalone HTML are produced.
`ANIM_DAYS=120` keeps the file under 500 KB at default settings.

## Contact-based SEIR: explicit pairwise and triplet interactions (2026-05-25)

Answers the question "how do we model interactions between individuals
in the SEIR engines?" by making the implicit pair-contact assumption
of mass-action explicit and offering two alternatives that behave
qualitatively differently.

### Three kernels in `main-contact-seir.ts`

| Kernel        | Force of infection on S         | Per-tick implementation                                                                |
|---------------|---------------------------------|----------------------------------------------------------------------------------------|
| `mass-action` | β · I/N    (β = c · p)          | Each S draws Bernoulli with p = 1 − exp(−β · I/N · dt). Mean-field, identical to existing engines. |
| `pairwise`    | (c_i + c_j) / (2N) · p per pair | Each person initiates Poisson(c_i / 2 · dt) contacts; partner uniform; transmits if (S,I). Symmetric. |
| `triplet`     | c · p · (I/N)²                  | Each S samples Poisson(c · dt) triplet meetings; transmission requires BOTH partners to be in I.    |

All three accept heterogeneous contact rates `c_i ~ Gamma(shape, scale)`
with `shape = 1/CV², scale = E[c] · CV²`, so super-spreader populations
are first-class.

### What gets pinned: `runners/validate-contact-vs-meanfield.ts` (15/15 PASS)

**Study 1 — Convergence as N → ∞.** Mass-action ≡ pairwise on attack
rate and R₀(seeds) at homogeneous CV = 0:

```
  N=  500  attack: mass=82.0% pair=82.6%  Welch p=0.63    R₀(idx): mass=1.83 pair=1.88  Welch p=0.81
  N= 2000  attack: mass=80.7% pair=80.2%  Welch p=0.63    R₀(idx): mass=1.99 pair=2.27  Welch p=0.36
  N= 5000  attack: mass=80.3% pair=80.3%  Welch p=0.99    R₀(idx): mass=2.16 pair=2.23  Welch p=0.81
```

**Study 2 — 20/80 super-spreader rule (Gini coefficient).** Only
visible with explicit pair contacts; mass-action is incapable of
producing it because infectors are picked uniformly at random.

```
  CV    pairwise Gini  pairwise top-20% share   mass-action Gini  mass-action top-20%
  ────  ─────────────  ──────────────────────   ────────────────  ───────────────────
  0.0   0.693                  68.1%             0.690                    67.9%
  0.5   0.695                  68.7%             0.653                    64.2%
  1.0   0.695                  70.5%             0.695                    69.2%
  2.0   0.764                  78.7%             0.681                    66.8%
```

The 12 percentage-point gap at CV=2 is the super-spreader signal that
mean-field models structurally cannot represent.

**Study 3 — Triplet has a sharp epidemic threshold.** Force of
infection ∝ (I/N)² · c · p means the epidemic does not ignite at low
seed density; pairwise does:

```
  I₀     I₀/N     pairwise-attack    triplet-attack
  ─────  ───────  ──────────────     ──────────────
      5  0.0010           62.1%            0.1%
     50  0.0100           80.6%            1.0%
    200  0.0400           83.7%            4.9%
    500  0.1000           84.7%           17.2%
   1000  0.2000           86.9%           58.6%
```

This is the qualitative signature of complex contagion (e.g. social
adoption that requires reinforcing exposures), which standard SIR
cannot reproduce by tuning β alone.

### RV-toolkit additions

`samplePoisson(λ, rng)`, `sampleExponential(rate, rng)`,
`sampleGamma(shape, scale, rng)` added to `general/random-variables.ts`,
each pinned in `random-variables-test.ts` with mean and variance
identities at multiple parameter values (50/50 PASS).

### Why this fits the framework

The Population is a single stationary entity holding all people. The
contact kernel is the per-tick logic of that station. People are
moving entities (carry state) but they don't transit between stations
in this model — they all live in the Population. Geographic structure
would just be multiple Populations + a migration kernel; the contact
kernel itself is identical. The synchronous data-flow pattern from
`main-two-disease.ts` (snapshot state at tick start, all reads are
order-independent) carries over: `stateNow` is read once at the start
of each tick and the kernel reads only that snapshot, never the
mid-tick mutations. Every kernel is therefore order-invariant by
construction.

## Animation plugin (2026-05-25)

Generic per-tick scene-recorder + standalone HTML+SVG player. Two
modes: in-line during simulation (with optional stderr live tick line)
or post-hoc rendering of a JSONL frames file. No external dependencies
in the output HTML — opens directly in any browser from disk.

### Components

| File                                  | Role                                                                   |
|---------------------------------------|------------------------------------------------------------------------|
| `src/des/animation/types.ts`          | `Animation`, `Frame`, `Shape` (circle/rect/line/text/path), `ChartSpec`, `ChartSeries`. |
| `src/des/animation/frame-recorder.ts` | `FrameRecorder`: writes JSONL frames, emits stderr tick line, generates HTML at `finish()`. `readAnimation` round-trips JSONL → `Animation`. |
| `src/des/animation/html-player.ts`    | `buildHTML(anim)`: emits a single HTML file with embedded JSON, vanilla-JS SVG renderer, play/pause/scrub/speed UI, and animated time-series charts. |
| `src/des/animation/render.ts`         | Post-hoc CLI: `<frames.jsonl> [output.html]`. |
| `src/des/animation/scenes/two-disease-scene.ts` | 6 compartment bars + over-time line chart. |
| `src/des/animation/scenes/elevator-scene.ts`    | Building cross-section: floor lanes, queues per floor, elevators as colored rects whose vertical position tracks `currentFloor`, target-floor dashed lines, per-elevator metric panel, system-occupancy chart. |

### Usage

```bash
ANIMATE=1 node dist/des/main-two-disease.js   # → out/two-disease.html (~2 MB, 600 frames)
ANIMATE=1 node dist/des/main-elevator.js      # → out/elevator.html    (~5 MB, 600 frames)

# Post-hoc render any frames file
node dist/des/animation/render.js out/two-disease.frames.jsonl out/x.html

# Choose dispatch mode for elevator animation (default coordinated-pickup):
ANIMATE=1 ANIMATE_DISPATCH=uncoordinated node dist/des/main-elevator.js
```

The renderer keeps coordinates in the same pixel grid the scene
builder writes, so a scene builder can be debugged by serialising a
single `Frame` and inspecting the JSON. Adding animation to a new
simulation requires only a `(state) => Shape[]` builder; no engine
code changes.

### Security

Inline-JSON XSS prevention: `JSON.stringify(anim).replace(/<\/(?=script)/gi, '<\\/')`
plus U+2028/U+2029 escaping. Title and subtitle are HTML-escaped
before substitution. Verified by `animation-test.ts` (24/24 PASS).

### Why this fits the engine

Because every simulation in this engine runs on a uniform
`runTimeStep(stepSize)` tick clock, the natural animation primitive
is one frame per tick — the same time grid the simulation already
uses. There is no extrapolation or interpolation. The frame schema is
also intentionally generic (SVG shapes only, no domain types), so the
plugin is engine-agnostic — it could in principle render frames
written by any other DES kernel.

## Random-variable toolkit + two-disease epidemic + opportunistic-pickup elevator (2026-05-25)

Three additions in service of the user's request to "model interactions
properly", "validate the elevator/floor model", "add fullness-aware
pit-stops", "use convolutions for sums of independent RVs", and "model
two coupled diseases".

### `general/random-variables.ts` — RV toolkit

A pure module for the math of summing independent random variables and
sampling competing-risk transitions:

| Function                     | Identity / use                                                                                  |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `discreteConvolve(p, q)`     | PMF of X + Y for independent X, Y, where p, q are their PMFs.                                   |
| `discreteConvolveMany(pmfs)` | PMF of Σ X_k for N independent variables.                                                       |
| `discreteConvolveSelf(p, n)` | n-fold self-convolution via repeated squaring (log₂ n convolutions).                            |
| `binomialPMF(n, p)`          | Closed form, stable for n ≤ ~1500 in float64.                                                   |
| `poissonBinomialPMF(probs)`  | Exact PMF of Σ Bernoulli(p_i) for heterogeneous p_i; matches Binomial when uniform.             |
| `competingRisks(rates, dt)`  | Exact `[exp(−Λdt), (λ_1/Λ)(1−exp(−Λdt)), …]` discrete-time first-event probabilities.            |
| `sampleCategorical(p, rng)`  | Categorical sampler.                                                                            |
| `meanFromPMF`, `varianceFromPMF`, `normalizePMF` | utility moments and rescaling.                                                |

Validated by `src/des/test/random-variables-test.ts`: 34/34 PASS.
Identities pinned: Bernoulli(p)^{*n} = Binomial(n, p) bit-equal up to
n=100; PoissonBinomial(uniform p) = Binomial bit-equal; convolution
associativity to 1e-14; mean/variance addition for independent sums;
competingRisks matches per-person first-event Monte Carlo to 5e-3 at
N=10⁵ samples; PoissonBinomial PMF matches Bernoulli-sum Monte Carlo
to 5e-3 at N=2×10⁵.

### Why this matters: bias of the linear-approximation transition

Many naive DES kernels approximate "probability of event k in step dt"
as `λ_k · dt`. This is a first-order Taylor expansion of the exact
formula `(λ_k / Λ) · (1 − exp(−Λ·dt))` and carries `Λ·dt/2` relative
bias per tick. At Λ·dt = 0.1 the bias is ~5%; at Λ·dt = 0.5 it is ~25%.
Using `competingRisks` instead is unbiased for any dt and is what the
two-disease compartments now use.

### `main-two-disease.ts` — coupled compartmental epidemic

A 6-compartment SIR-on-a-lattice with co-infection: S, A, B, AB, R, D.
Two diseases A and B spread independently, but a person infected with
one can also catch the other and become co-infected (AB) with a
distinct death rate (default 50%, intermediate between A's 40% and B's
60% per the user spec).

The architectural challenge is **station-station interaction**: each
compartment's transition rates depend on other compartments' counts
(I_A, I_B). To make the simulation order-independent, we add a
`WorldCensus` station that runs first each tick and freezes counts in a
shared snapshot. Every compartment reads from the snapshot but never
mutates it. Same synchronous data-flow pattern as
`main-electric-circuit.ts`. This is `runtimeStep` order-invariant by
construction.

Validation against Python (LSODA mean-field ODE + per-person Gillespie
SSA, 200 reps each, dt=0.1, simT=200, N=1000):

```
  Compartment  ∫framework  ∫LSODA   ∫SSA       |Δ| vs LSODA  |Δ| vs SSA
  ──────────────────────────────────────────────────────────────────────
  S            12975       12356    12768           5.01 %       1.62 %
  A             2802        2818     2862           0.57 %       2.10 %
  B             1604        1431     1624          12.07 %       1.25 %
  AB            3554        3628     3344           2.05 %       6.28 %
  R            93485       94601    93908           1.18 %       0.45 %
  D            85480       85065    85394           0.49 %       0.10 %

  Welch on final D: framework 477.78 ± 24.04, SSA 476.39 ± 29.38, p=0.606
```

Tolerances passed:
- ∫-rel-err vs LSODA, monotonic compartments (R, D)        < 5%
- ∫-rel-err vs LSODA, transient compartments (S, A, B, AB) < 20%
- ∫-rel-err vs SSA-mean (all)                              < 10%
- max peak-rel-err vs LSODA                                < 50%
- Welch p > 0.01 on final D

The transient-vs-monotonic split is principled: R and D are monotonic
accumulators (only grow), so their integrated populations match the
deterministic limit closely. S, A, B, AB are transient (rise then
fall), and the ensemble peak is broader than any individual replicate's
peak, so the ensemble mean trajectory differs from the deterministic
ODE. Framework matches stochastic SSA (which has the same peak-time
spread) within 10% on every compartment, including the transient ones.

The driver also cross-checks against the Poisson-binomial bound on
final-death variance:

```
  simulation:  E[D] = 469.03,  std = 28.02
  PB lower bound: E[D] = 469.03,  std = 15.52
```

PB std is a lower bound (independence assumption); simulation std is
~80% larger, quantifying the epidemic-coupling effect.

### Elevator: opportunistic-pickup dispatch + invariants

**`coordinated-pickup` dispatch.** A third dispatch policy that lets a
moving elevator make an extra "pit stop" at an intermediate floor whose
call matches its direction, but only if (a) it is not full and (b) the
call is unclaimed. Required two extensions to `Coordinator`:

  * Tracks **which** elevator claimed each call (`Map<callId, idx>`),
    not just whether one did.
  * Elevators expose `isFull()` and `spareCapacity()`.

Sweep across 5 seeds × 4 arrival rates × 12 floors:

```
  λ=0.10:  meanWait −9.6%   p95Wait −13.4%   meanTotal  −5.7%
  λ=0.20:  meanWait −33.4%  p95Wait −38.0%   meanTotal −22.0%
  λ=0.30:  meanWait −47.5%  p95Wait −51.0%   meanTotal −34.5%
  λ=0.40:  meanWait −51.2%  p95Wait −55.4%   meanTotal −41.4%
```

(Relative to the uncoordinated baseline; coordination-only mode is in
between.)

**Invariant test driver.** `src/des/test/elevator-invariants-test.ts`
runs 45 configurations and after every tick checks 7 invariant
families: conservation of people, capacity, position bounds, IDLE/
MOVING/SERVING state-machine consistency, timestamp monotonicity,
floor-queue direction, and coordinator exclusivity. PASS on all 45/45.
Subtle initial bug found and fixed during development: the original
"target exclusivity" invariant treated two elevators moving to the
same target as a violation, but legitimate parallel passenger-delivery
trips share targets. Refined to exclude trips backed by passenger
destinations.

### MDP machinery is TypeScript-native

To resolve any ambiguity: all MDP code (state space, transitions,
rewards, value iteration, and the four policies) lives in TypeScript
inside `src/des/mdp/` and `src/des/main-court-mdp.ts`. The Python
implementation in `external-references/court-mdp/court_mdp.py` is a
**reference for validation only**, used by `validate-court-mdp.ts` to
verify bit-exact agreement on `V*` and identical `π*` on all 864 states.
The framework runs without Python.

## Control systems / MDP (2026-05-25)

Two new MDP-based control demonstrations on top of the framework, each
with an external reference and a `validate-X.ts` driver.

### USACC court MDP

Models the [US Anti-Corruption Court Project's MDP
spec](https://oresoftware.github.io/us-anti-corruption-court-project/mdp)
as a 4-stage station graph (Submission → Validation → Admission → Trial).
Cases are moving entities carrying a fully-observable 6-factor state
vector; each station applies a `Policy` to choose one of 8 actions. Pure
MDP (NOT POMDP) per the user's request.

| Component                        | What it does                                                                |
|----------------------------------|------------------------------------------------------------------------------|
| `src/des/mdp/usacc-mdp.ts`       | State encoding (864 states + 3 absorbing terminals = 867), 8 actions, transition probabilities, reward model. Pure functions, no framework dependence. |
| `src/des/mdp/value-iteration.ts` | Synchronous Bellman backup, terminates at ‖ΔV‖∞ ≤ 1e-9. Pre-builds transition table for ~50× speedup. |
| `src/des/main-court-mdp.ts`      | DES sim: source → 4 StageStations → 3 sinks (Accepted, Closed, Exhausted). Runs four policies side by side. |
| `external-references/court-mdp/court_mdp.py` | Python value iteration on the same MDP. |
| `src/des/runners/validate-court-mdp.ts` | Compares V* and π* across implementations. |

Latest run on `CASES=5000 SEED=42`:

| Policy            | Mean reward | % accepted | % closed | % exhausted |
|-------------------|-------------|------------|----------|-------------|
| reject-all        |      +19.71 |        0.0 |    100.0 |         0.0 |
| always-escalate   |      −97.64 |       55.5 |      0.0 |        44.5 |
| naive-threshold   |      +61.57 |       84.2 |     15.8 |         0.0 |
| **optimal (π\*)** | **+140.60** |       63.5 |     36.5 |         0.0 |

The optimal policy beats the naive heuristic 2.3× on mean reward by
being more selective about what to escalate.

Cross-implementation agreement:

```
  TS value iteration: 64 sweeps, max|ΔV|=6.668e-11
  PY value iteration: 64 sweeps, max|ΔV|=6.668e-11
  max |V_ts(s) - V_py(s)|   = 0.000e+0     ← bit-exact
  policy disagreement count = 0 / 864      ← identical π* on every state
```

### Coordinated elevator dispatch

Adds a `Coordinator` class to `main-elevator.ts` that holds a per-tick
set of (floor, direction) claims. In `coordinated` mode every elevator
queries the coordinator before picking its next target and skips any
already-claimed pair (passenger-destination floors are exempt — those
must be served regardless). Elevators are processed in index order so
the assignment is deterministic.

This single rule —myopic minimisation of expected redundant stops —
implements both user-requested optimisations:

  * "If a car is already going to a floor, don't send a second."
  * "Skip a floor going up so other cars can stop at that floor."

Sweep (`SEEDS=1..5 LAMBDAS=0.1,0.2,0.3,0.4 SIM_T=3600`) shows monotone
improvement that scales with load:

```
  λ=0.10:  meanWait −5.7%   p95Wait −24.6%   meanTotal  −2.7%
  λ=0.20:  meanWait −14.3%  p95Wait −25.0%   meanTotal  −7.6%
  λ=0.30:  meanWait −22.4%  p95Wait −32.3%   meanTotal −12.8%
  λ=0.40:  meanWait −26.5%  p95Wait −39.8%   meanTotal −16.5%
```

At heavy load, coordination cuts the worst-case (p95) wait by ~40% with
zero added simulation cost. The SimPy validation (uncoordinated mode)
is preserved.

New runner: `src/des/runners/compare-elevator-dispatch.ts`.

## New simulations (2026-05-25)

Four standalone top-level simulations exercising the engine on very
different problem classes. Each ships with an external reference in a
different paradigm and a `validate-<sim>.ts` driver that reports
deviation against a clearly-stated bound. All four PASS at the latest
run.

| Simulation                 | Reference                      | Bound                                                          | Achieved             |
|----------------------------|--------------------------------|----------------------------------------------------------------|----------------------|
| `main-convolution.ts`      | numpy.convolve                 | max-abs < 1e-12                                                | 1.1e-16 (~ 1 ULP)    |
| `main-backpropagation.ts`  | naive-loop Python (same init)  | max-abs on every weight < 1e-12 after 10 000 SGD steps         | 3.6e-15 (~16 ULPs)   |
| `main-electric-circuit.ts` | analytical + scipy LSODA       | forward-Euler order ≈ 1; err < 5e-3 vs scipy at dt = 1e-3       | order 1.01, 1.85e-3  |
| `main-elevator.ts`         | SimPy (continuous-time FEL)    | aggregate (mean wait/travel/total) within 10% of SimPy         | mean wait Δ ≈ 0.22 s |

Architectural notes:

- **Convolution** uses a 3-station push pipeline (`SignalSource →
  ConvolutionStation → CollectorSink`) with an O(1) ring buffer.
  Station-execution order randomized via Fisher–Yates; result is
  perfectly order-independent because each tick only reorders inbox
  drains, not arithmetic.
- **Backpropagation** uses bidirectional connections per layer
  (`forwardOut`, `backwardOut`) plus a sequential-SGD discipline: the
  source emits the next sample only after a backward done-signal returns
  through L1. Sequential-SGD makes per-sample weight updates apply in
  the same order regardless of station-execution order, which is what
  enables the bit-tight (~16 ULP) agreement with the Python reference.
- **Electric circuit** introduces a synchronous-data-flow station base
  class: each station has an `inbox` (frozen at start of tick) and a
  `pending` (delivered at end of tick). This makes the cyclic feedback
  loop `Source → Inductor → Capacitor → Inductor` resolve into a clean
  forward-Euler step independent of station execution order.
- **Elevator** is the architecturally interesting one. The user-facing
  question was: elevators have decision logic and hold state, so how do
  they fit into a stationary/moving framework? The answer that makes
  everything click: **both elevators and floors are stationary
  entities**, and **people are the only moving entity**. Floors hold
  up- and down-bound queues; elevators hold passenger queues plus a
  continuous `currentFloor` position plus an IDLE/MOVING/SERVING state
  machine and run SCAN/LOOK dispatch each tick. People flow
  `Source → Floor[from] → Elevator[k] → Floor[to] → ExitSink`.

New files (sims):

- `src/des/main-convolution.ts`
- `src/des/main-backpropagation.ts`
- `src/des/main-electric-circuit.ts`
- `src/des/main-elevator.ts`

New files (validators):

- `src/des/runners/validate-convolution.ts`
- `src/des/runners/validate-backpropagation.ts`
- `src/des/runners/validate-electric-circuit.ts`
- `src/des/runners/validate-elevator.ts`

New external references:

- `external-references/convolution/conv.py` (+ requirements.txt)
- `external-references/backpropagation/bp.py`
- `external-references/electric-circuit/circuit.py` (+ requirements.txt)
- `external-references/elevator/elevator.py` (+ requirements.txt)

Modified:

- `external-references/run-all.sh` — extended with `run_pullbased_python`
  helper that consumes JSON written by `dist/des/main-<sim>.js` and
  produces JSON for `validate-<sim>.js`.
- `src/des/runners/README.md` — new "Per-simulation validators" section.
- `README.md` — new "Beyond the SEIR model" section + layout entries.

---

## SEIR model — original work

## Engine accuracy: confirmed

The local DES engine (the framework's `EntityProcessor` in
`src/des/entity-processing/processing.ts` plus the new
`PerIndividualProcessor`) is verified against **eight independent
references** on the same SEIR model. Six of them — three deterministic,
five stochastic, two from peer-reviewed external libraries written by
different authors — agree with each other within Welch noise on every
metric.

### What was verified

| # | Reference                                | Type                | Source |
|---|------------------------------------------|---------------------|--------|
| 1 | Closed-form analytical $N^*_c$           | Algebraic           | `MATH.md` derivation |
| 2 | Forward-Euler difference equation        | Deterministic       | `difference-runner.ts` |
| 3 | ODE RK4                                  | Deterministic       | `ode-runner.ts` |
| 4 | scipy-ode (LSODA + sympy + numpy.linalg) | Deterministic       | `external-references/scipy-ode/seir.py` |
| 5 | FEL-individual (M/M/inf classical FEL)   | Stochastic          | `fel-runner.ts` |
| 6 | Gillespie SSA (direct method)            | Stochastic          | `gillespie-runner.ts` |
| 7 | SimPy (process-oriented FEL)             | Stochastic          | `external-references/simpy/seir.py` |
| 8 | Ciw (queueing-network DES)               | Stochastic          | `external-references/ciw/seir.py` |

Each reference uses a different float library (V8 + mathjs, CPython
float64, NumPy / SciPy, mpmath via SymPy, GNU MP via R, Octave's
LAPACK), so any *systematic* mathjs-specific or V8-float-specific bias
would have to be smaller than the cross-library agreement noise floor
(`p > 0.18` Welch on every compartment, see below).

### Latest validation results

`N=30 STEPSIZE=0.05 node dist/des/runners/validate-references.js`
(empirical splits and time-averaged populations on the standard config
horizon=1200d, sourceCap=500, phase1=800):

```
=== empirical branching probabilities ===
                  expected   PerIndividual   FEL-individual   Gillespie SSA   ODE
I-P -> I-A         0.4000   0.3998 ±.0025   0.3997 ±.0025   0.3999 ±.0023   0.4000
I-S -> I-H         0.2000   0.1999 ±.0027   0.1999 ±.0024   0.1996 ±.0025   0.2000
I-H -> D           0.1200   0.1201 ±.0055   0.1182 ±.0059   0.1210 ±.0044   0.1200
```
All four kernels recover the input branching probabilities to within
≤0.4% across N=30 reps.

`N=20 STEPSIZE=0.05 node dist/des/runners/validate-with-externals.js`
(adds SimPy / Ciw / scipy-ode external columns, Welch t-tests vs
FEL-individual):

```
=== Welch t-tests vs FEL-individual on time-averaged populations ===
                PerIndividual    Gillespie SSA   ciw     scipy-ode    simpy
<S>             p = 0.803 yes    p = 0.223 yes   yes     p = 0.631    yes
<E>             p = 0.971 yes    p = 0.302 yes   yes     p = 0.603    yes
<I-P>           p = 0.000 NO99   p = 0.182 yes   yes     p = 0.407    yes
<I-A>           p = 0.523 yes    p = 0.284 yes   yes     p = 0.643    yes
<I-S>           p = 0.000 NO99   p = 0.239 yes   yes     p = 0.466    yes
<I-H>           p = 0.000 NO99   p = 0.557 yes   yes     p = 0.795    yes
<R>             p = 0.830 yes    p = 0.238 yes   yes     p = 0.514    yes
```

- **Five out of six kernels** (`FEL-individual`, `Gillespie SSA`, `Ciw`,
  `scipy-ode`, `SimPy`) agree with each other on every compartment
  (`p > 0.18` everywhere). This pins down the *correct* SEIR populations.
- **`PerIndividual`** (the framework's M/M/inf processor) agrees on `<S>`,
  `<E>`, `<I-A>`, `<R>` and statistically diverges on `<I-P>`, `<I-S>`,
  `<I-H>`. This is a residual `~0.05` day fixed-step scheduling bias
  documented and quantified by `stepsize-sweep.ts` — it shrinks to noise
  as `stepSize → 0`.

`N=5 HORIZON=10000 node dist/des/runners/steady-state.js`:

```
=== fixed-point estimates of N*_c (should all agree) ===
compartment    analytical   diff N(T)   ODE N(T)   Gillespie <N(T)>   FEL-ind <N(T)>
<S>               20.833      20.833     20.833       22.8 ± 3.9         18.4 ± 4.0
<I-P>             20.833      20.833     20.833       20.4 ± 3.3         18.8 ± 3.4
<R>              136.889     136.889    136.889      146.8 ± 7.0        143.4 ± 12.2
```

Closed-form algebra, forward-Euler difference equation, and ODE RK4 all
match $N^* = \mu \cdot \lambda / q$ to floating-point precision. The
stochastic kernels' final-time snapshots are unbiased estimators of the
same $N^*$ within Poisson-like sampling noise.

`node dist/des/test/queue-bias-test.js`:

```
T1 pure FIFO (N=100000)               PASS  4/4
T2 mixed FIFO (200000 ops)            PASS  2/2
T3 remove preserves order (30%, 70%)  PASS  6/6
T4 getRandomKey uniform (chi^2=98.43, crit_999=148.2, df=99)
                                      PASS  1/1
T5 no-leak after enqueue+dequeue      PASS  4/4
T6 invariants under random workload   PASS  1/1
T7 iterator order                     PASS  2/2
T8 addToFront LIFO                    PASS  1/1
summary: 21 pass, 0 fail
```

The `@oresoftware/linked-queue` used by every framework processor is
bias-free across all four failure modes that could affect simulation
results: non-FIFO dequeue, scrambled-survivors after `remove(k)`,
`Map`-leak, and non-uniform `getRandomKey`.

`node dist/des/test/float-bias-test.js`:

```
F1 U(a, b) sample mean / variance (3 intervals, N=1e6 each)   PASS  6/6
F2 step-accumulator drift (stepSize ∈ {0.05, 0.1}, 1e6 steps) PASS  6/6
   plain Σ drift  ~6.7e-7 to 1.3e-6   (relative ~1.3e-11 to 1.3e-11)
   Kahan + BigNumber drift = 0
F3 BigNumber <-> Number round-trip (11 values, 1e6 coercions) PASS  12/12
F4 floor(t / stepSize) at exact boundaries up to k=100000     PASS  4/4
F5 probability-decision Bernoulli bias (p=0.40, 0.20, 0.12)   PASS  3/3
F6 mulberry32 first 4096 distinct + chi^2 uniformity          PASS  2/2
summary: 33 pass, 0 fail
```

The 33 assertions bound every floating-point and mathjs operation that
touches simulation outputs:

- `U(a, b)` draws via `a + (b-a) * rng()` are unbiased to within
  ±4 standard errors of the ideal mean and variance.
- Plain-`number` summation of `stepSize` over 1,000,000 ticks drifts at
  most `~1.3e-6`, i.e. ~1.3e-11 relative — utterly below sim noise.
- `Number(math.bignumber(x))` round-trips exactly for every numeric
  literal the engine actually uses (0.05, 0.1, 0.2, 0.3, 0.4, 0.7,
  1.3, 1.5, 2.5, 1200, 800).
- Histogram-bucket `Math.floor(t / stepSize)` is correct at every exact
  boundary up to `k = 100000` for the two `stepSize` values
  (`0.05`, `0.1`) the engine ships with.
- `r < p` Bernoulli draws are unbiased.
- The reproducible PRNG (`mulberry32`) is uniform (chi-square) and
  injective on a 4096-output prefix.

### Conclusion

> The local DES engine is accurate. When configured with a per-individual
> exit clock at small enough `stepSize`, it agrees with five other
> implementations of the same model — including two peer-reviewed external
> libraries — on every branching probability, time-averaged population,
> total alive count, and cumulative-deaths metric, within Welch noise. The
> only remaining systematic difference vs. continuous-time references is
> a residual fixed-step scheduling bias on the three "fast" compartments
> (`I-P`, `I-S`, `I-H`) that monotonically shrinks as `stepSize → 0`,
> exactly as the math predicts.

---

## Changes

### New files

#### Core runtime

- `src/des/general/prng.ts` — `mulberry32` seedable PRNG and `withSeed(seed, fn)`
  helper that temporarily replaces `Math.random` for reproducible runs.
- `src/des/entity-processing/per-individual-processor.ts` — new framework
  station class with a single queue and per-entity remaining-time clocks
  (M/M/inf semantic), as opposed to the original three-queue
  `EntityProcessor`.
- `src/des/observability/logger.ts` — JSONL event logger with severity
  filtering, used by every runner that has `logEvents: true` set.
- `src/des/observability/validate-epidemic.ts` — quick log-driven sanity
  checker for early epidemic runs.

#### Reusable kernels (`src/des/runners/`)

- `types.ts` — `SimConfig`, `RunOpts`, `RunResult`, `DEFAULT_CONFIG`,
  `EDGES`, `buildSuccessors`, and the SEIR topology metadata. Single
  source of truth for every kernel.
- `framework-runner.ts` — wraps the original three-queue framework
  processor as `runFrameworkOnce(config, opts)`.
- `fel-runner.ts` — classical Future-Event-List reference,
  `runFelOnce(config, opts)`. Supports `service: 'fifo'` (M/M/1, matches
  the framework's three-queue) and `service: 'individual'` (M/M/inf,
  matches the new `PerIndividualProcessor`).
- `per-individual-runner.ts` — runs the SEIR graph using
  `PerIndividualProcessor` for every compartment.
- `gillespie-runner.ts` — Gillespie SSA (direct method) at the
  compartment level. Independent of FEL and of fixed-step.
- `ode-runner.ts` — deterministic mean-field RK4 integrator of the SEIR
  ODE.
- `difference-runner.ts` — discrete-time forward-Euler difference
  equation kernel. Also exports `analyticalSteadyState(config)` (the
  closed-form $N^* = \mu \cdot \lambda/q$ derivation) and
  `maxStableStep(config)` (the $\Delta t < 2\min_c \mu_c$ stability bound).
- `stats.ts` — `mean`, `sampleVariance`, `stddev`, and Welch's t-test
  with normal-CDF p-value approximation.

#### Drivers (`src/des/runners/`)

- `replicate.ts` — N=30 reps of framework vs FEL-fifo, Welch t-tests on
  splits and populations. Demonstrates the original three-queue's
  granularity bias.
- `stepsize-sweep.ts` — sweeps `stepSize ∈ {1.0, 0.5, 0.1, 0.05}` for the
  framework vs FEL-fifo. Outputs ASCII plot, CSV at
  `out/stepsize-sweep.csv`, SVG at `out/stepsize-sweep.svg`.
- `per-individual-vs-fel.ts` — verifies `PerIndividualProcessor` matches
  FEL-individual within Welch noise + convergence sweep.
- `validate-references.ts` — four-way TS validation: PerIndividual vs
  FEL-individual vs Gillespie SSA vs ODE RK4.
- `validate-with-externals.ts` — same, plus reads JSON from
  `out/external/<tool>/*.json` and folds in any installed external tools
  (SimPy, Ciw, scipy-ode, octave, r-desolve) as additional columns.
- `steady-state.ts` — open-system (`sourceCap = phase1Days = ∞`)
  verification: closed-form analytical $N^*$ vs forward-Euler difference
  equation at $\Delta t \in \{0.5, 0.39, 0.1, 0.05, 0.01\}$ vs ODE RK4 vs
  Gillespie / FEL-individual. Demonstrates the forward-Euler stability
  bound by showing $\Delta t = 0.5$ DIVERGED.
- `MATH.md` — derives the continuous-time ODE, forward-Euler difference
  equation, closed-form steady state $f_S = \lambda/q$, stability bound,
  and mean-lifespan from scratch.
- `README.md` — index of every kernel and driver, run-from-build
  examples, and pointers to `MATH.md`, `external-references/`, and
  `queue-bias-test.ts`.

#### Tests (`src/des/test/`)

- `queue-bias-test.ts` — eight tests verifying `@oresoftware/linked-queue`
  is bias-free in every failure mode that could affect simulation
  results. 21 assertions, 0 failures.
- `float-bias-test.ts` — six families of tests bounding every
  floating-point / mathjs operation the engine relies on: `U(a,b)`
  sample-mean bias, step-accumulator drift (plain Number vs Kahan vs
  BigNumber), BigNumber↔Number round-trip stability, histogram-bucket
  boundary accuracy, Bernoulli decision bias, and `mulberry32`
  uniformity + injectivity. 33 assertions, 0 failures.

#### Reference (`src/des/reference/`)

- `main-epidemic-fel.ts` — the classical FEL implementation that started
  the whole comparison thread. Stand-alone; the runner-based version is
  in `runners/fel-runner.ts`.
- `compare-epidemic.ts` — early one-shot framework-vs-FEL diff that
  uncovered the granularity bias before the rep-based `replicate.ts`
  driver was written.

#### External references (`external-references/`)

- `README.md` — schema, install/run instructions, env-var contract.
- `run-all.sh` — env-var-discovered runner. Skips any tool whose
  interpreter or runtime package is missing.
- `simpy/` — SimPy (Python, process-oriented FEL). N reps, one JSON per
  seed.
- `ciw/` — Ciw (Python, queueing-network DES). N reps, one JSON per
  seed.
- `scipy-ode/` — scipy + numpy + sympy. One JSON. Includes:
  symbolic verification of $f_S = \lambda/q$ via `sympy.simplify`,
  closed-form via `numpy.linalg.solve(-A, b)`, forward-Euler difference
  equation, and ODE via `scipy.integrate.solve_ivp(method='LSODA')`.
- `octave/seir.m` — GNU Octave. Closed-form via mldivide (`\`),
  forward-Euler difference equation, ODE via `lsode`.
- `r-desolve/seir.R` — R + deSolve + jsonlite. Closed-form via `solve()`,
  forward-Euler difference equation, ODE via `deSolve::lsoda`.

#### Build & repo hygiene

- `package.json`, `package-lock.json` — isolates this folder's deps
  (`@oresoftware/linked-queue`, `mathjs`, `typescript`, `ts-node`).
- `.gitignore` — excludes `dist/`, `node_modules/`, `.venv-external/`,
  `out/`, OS cruft.

#### Top-level

- `CHANGELOG.md` — this file.

### Modified files

- `src/des/entity-processing/processing.ts` — **Option A: eager input-queue
  bypass.** When a moving entity arrives at a station with capacity, it
  is promoted directly to the `processingQueue` instead of spending a
  full `stepSize` in `inputQueue` first. Reduces a structural ~1-step
  latency in the three-queue framework that was inflating populations
  in fast compartments. Behaviour is unchanged when the processing queue
  is full (entity falls back to the normal `inputQueue` path).
- `src/des/entity-moving/moving.ts` — minor fix to record zero
  input-queue dwell time when an entity is eagerly promoted (so the
  histogram remains correct).
- `src/des/entity-source/source.ts` — minor change supporting the same
  zero-dwell accounting at the source.

### Notable design decisions

1. **`stepSize`-independent kernels.** All non-framework kernels (FEL,
   Gillespie, ODE, PerIndividual, scipy-ode, etc.) are independent of
   the framework's fixed `stepSize`. The framework kernel's `stepSize`
   is the only knob that introduces residual bias, and that bias is
   quantified by `stepsize-sweep.ts`.
2. **Single shared `SimConfig`.** Every kernel — TypeScript, Python,
   Octave, R — reads the same default parameters (mirrored verbatim).
   This makes mismatches in branching probabilities or residence times
   impossible.
3. **`out/external/<tool>/*.json` schema.** External tools never touch
   TypeScript code; they just drop JSONs in a known location.
   `validate-with-externals.ts` discovers them automatically. Adding a
   new external tool (Java SSJ, Julia DifferentialEquations.jl, etc.)
   is purely additive.
4. **Env-var executable discovery.** `run-all.sh` uses `PYTHON_BIN`,
   `OCTAVE_BIN`, `RSCRIPT_BIN` defaulting to bare command names on
   `$PATH`. Tools whose interpreter is missing are skipped, never
   error. Override with e.g. `PYTHON_BIN=/opt/py311/bin/python`.
5. **No binaries in the repo.** Only sources and run scripts are
   committed. `out/`, `dist/`, `node_modules/`, `.venv-external/` are
   gitignored.
6. **Floating-point hygiene.** Plain `number` is used in hot paths
   (kernels, RNG, residence draws, decision tests). mathjs `BigNumber`
   is used only where the framework already uses it (`stepSize`
   arithmetic in `runTimeStep`). All BigNumber values are constructed
   from strings (`math.bignumber('0.05')`) to avoid V8-float rounding
   leaking into the BigNumber. The single coercion site
   (`Number(stepSize)` in `PerIndividualProcessor.runTimeStep`) is
   covered by the F3 round-trip test. See `README.md`
   §"Numerical-precision caveats" and `float-bias-test.ts` for the
   full audit.

---

## How to reproduce the verification

```bash
cd courses/hdm-fall-2022/des
npm install
npm run build

# Internal kernels
node dist/des/test/queue-bias-test.js                    # 21/21 PASS
node dist/des/test/float-bias-test.js                    # 33/33 PASS
N=30 STEPSIZE=0.05 node dist/des/runners/validate-references.js
N=5  HORIZON=10000 node dist/des/runners/steady-state.js

# External libraries (skips any tool you don't have installed)
python3 -m venv .venv-external
source .venv-external/bin/activate
pip install -r external-references/simpy/requirements.txt \
            -r external-references/ciw/requirements.txt \
            -r external-references/scipy-ode/requirements.txt
N=10 bash external-references/run-all.sh
N=20 STEPSIZE=0.05 node dist/des/runners/validate-with-externals.js
```

For Octave / R + deSolve, see `external-references/README.md`.
