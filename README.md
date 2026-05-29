# DES Engine (UTA PhD course project)

A **discrete-event system** framework — DES not as in "discrete-event
*simulation*" only, but as the more general *Discrete Event System*: a
substrate that simulates physical/social processes, runs control loops,
and executes iterative algorithms (linear programming, genetic
algorithms, shortest-path, MCTS, value iteration, Benders/L-shaped,
…) on the same architecture. The kernel is fixed-time-step,
**station-local**, and has **no global future event list (FEL)**.

This README explains *what makes the engine different* and *why those
differences matter*. The acronym DES is used throughout because it
applies to both the original "simulation" framing and the more general
"system" framing — what's stayed constant is the underlying
station/movable architecture and the discrete tick clock.

For mathematical derivations see `src/des/runners/MATH.md`. For accuracy
verification see [`CHANGELOG.md`](CHANGELOG.md).

### Optimal control — entity-based block diagrams for every classic method

For control workloads the natural mental model is the **block diagram**:
a plant communicates with a controller via signal lines. This branch of
the codebase implements that view directly with `StationaryEntity`
blocks and `AbstractMovingEntity` signals — distinct from the
lightweight `DESStation` pattern used by the RL/DP/optimisation models.

`des-base/control-blocks.ts` provides the shared base classes
(`PlantBlock`, `ControllerBlock`, `EstimatorBlock`, `VectorSignal`,
`runClosedLoop`). Concrete demos cover every method on the canonical
"important optimal-control methods" list:

| Optimal-control method                | Base / pattern                        | Concrete demo                              | JSON id                  |
| ------------------------------------- | ------------------------------------- | ------------------------------------------ | ------------------------ |
| PID                                   | `ControllerStation`                   | HVAC heater (`temp-control`)               | `temp-control-pid`       |
| LQR                                   | `LQRController`                       | double integrator                          | `double-integrator-lqr`  |
| MPC (constrained)                     | `ControllerBlock` + projected-grad QP | double integrator with `\|u\|`-bound       | `mpc-double-integrator`  |
| Dynamic programming                   | `FiniteHorizonDPStation`              | multi-period inventory                     | `inventory-dp`           |
| Pontryagin's Maximum Principle (PMP)  | `ControllerBlock`                     | time-optimal double-integrator (bang-bang) | `pontryagin-bang-bang`   |
| Kalman filter (state estimation)      | `EstimatorBlock`                      | radar tracking with noisy sensor           | `kalman-filter`          |
| Robust control (sliding mode)         | `ControllerBlock`                     | uncertain plant + matched disturbance       | `sliding-mode`           |
| H-infinity-style robust control       | `ClosedLoopPlantStation` + adversary  | bounded worst-case disturbance game         | `hinfinity-robust-control` |
| Adaptive control (MRAC)               | `ControllerBlock`                     | unknown-gain plant tracks reference model  | `mrac`                   |
| Learning control (ILC)                | `DESStation` source/station/sink graph | repeated-trial feedforward adaptation      | `iterative-learning-control` |
| Differential games                    | `ClosedLoopPlantStation` + competing policies | pursuit/evasion                       | `pursuit-evasion-game`   |
| Reinforcement learning                | `RLAgentStation`, …                   | Mountain Car / GridWorld / Blackjack / …    | various                  |
| Nonlinear control (feedback lin.)     | `ControllerBlock`                     | pendulum tracking via computed torque       | `feedback-linearization` |

### MDP and friends — base classes for every cell of the table

The base hierarchy now spans every concept in the standard MDP
neighbourhood. Each base ships with a runnable concrete model on a
classic problem and a JSON adapter, so you can drive any of these from
a `.json` file via the unified registry:

| Concept (MDP-adjacent)         | Base class                        | Concrete demo            | JSON id                  |
| ------------------------------ | --------------------------------- | ------------------------ | ------------------------ |
| Bellman / dynamic programming  | `FiniteHorizonDPStation`          | multi-period inventory   | `inventory-dp`           |
| Approximate DP                 | `LinearVFAStation<S>`             | Mountain Car + tile coding | `mountain-car-vfa`     |
| POMDPs                         | `BeliefStateStation<A, O>`        | Tiger problem            | `tiger-pomdp`            |
| Multi-dimensional POMDPs       | `CartesianStateSpace` + `BeliefLookaheadSolver` | 2D grid localization | `grid-localization-pomdp` |
| Semi-MDPs                      | `SemiMDPAgentStation<S, A>`       | Four Rooms + hallway options | `four-rooms-smdp`    |
| Reinforcement learning (TD)    | `RLAgentStation<S, A>` (existing) | GridWorld Q-learning     | (existing)               |
| Policy gradients               | `PolicyGradientAgent` (existing)  | Tabular PPO              | (existing)               |
| Actor-Critic                   | `TabularActorCritic`              | GridWorld                | `actor-critic-grid`      |
| Monte Carlo methods            | `MonteCarloAgent`                 | Blackjack                | `blackjack-mc`           |
| Multi-agent RL                 | `JointEnvStation` + `MultiAgentSystem` | Stag Hunt           | `stag-hunt`              |
| Evolutionary strategies        | `PopulationOptimizer<I>` (existing) | TSP genetic algorithm  | (existing)               |
| Stochastic control             | `LQRController` / `ControllerStation` | Double integrator    | `double-integrator-lqr`  |

Each model is a tiny demo of its idea: ~150 lines of algorithm code,
intrinsic validators auto-attached in the constructor, plus a JSON
adapter that lets you save/share runs as plain JSON.

### Applied domain model pack

`domain-application-models.ts` adds a reusable applied-model pipeline:

`ScenarioSource -> CandidateGenerator -> PlanEvaluator -> ResultSink`

The same source/station/sink/movable graph now covers ten application
areas and eleven JSON-runnable applied models:

| Area | JSON id |
| --- | --- |
| Control systems (adaptive/fuzzy/intelligent) | `adaptive-fuzzy-control` |
| Logistics / transportation | `logistics-routing-heuristics` |
| Manufacturing | `bottleneck-production-control` |
| Supply chain management | `supply-chain-risk-pooling` |
| Operations management | `workforce-service-operations` |
| Financial engineering | `portfolio-drawdown-control` |
| Revenue management | `dynamic-pricing-revenue`, `buyer-aware-dynamic-pricing` |
| Energy / power systems | `energy-storage-dispatch` |
| Machine/statistical learning | `active-learning-acquisition` |
| Decision science + visualization | `visual-decision-frontier` |

### Nonlinear forecasting with MDP/POMDP variable discovery

`nonlinear-mdp-pomdp-forecast` is a JSON-runnable prediction model that
keeps the modeling workflow in the entity framework:

`ForecastDataSource -> POMDPLatentVariable -> MDPVariableDiscovery -> NonlinearEquationTuning -> ForecastProjection -> ResultSink`

The POMDP station infers hidden regime beliefs, the MDP station treats
candidate variables as discovery actions, the equation station fine-tunes a
nonlinear basis expansion, and the projection station rolls the equation
forward with uncertainty bands.

```bash
node dist/des/main-from-json.js examples/nonlinear-mdp-pomdp-forecast.json
```

### Advanced optimization — reusable station patterns

The metaheuristic and combinatorial layer now has framework-level bases
for the "second ten" classic routines. The reusable pieces live in
`des-base/advanced-optimization.ts`, with shared station/movable metadata
in `des-base/model-topology.ts`; the concrete demos are intentionally
small and registry-driven so the same station/token patterns can support
animation and a future visual editor.

| Family | Shared base / token pattern | Concrete demo | JSON id |
| ------ | --------------------------- | ------------- | ------- |
| Particle swarm optimization | `NumericSwarmOptimizerStation` + `OptimizationCandidateToken` | sphere/Rastrigin/Rosenbrock | `particle-swarm` |
| Ant colony optimization | `PheromoneGraphSearchStation` + `GraphWalkToken` | TSP | `ant-colony-tsp` |
| Constraint satisfaction | `ConstraintSatisfactionSearchStation` + `ConstraintAssignmentToken` | map coloring | `map-coloring-csp` |
| SAT / MAX-SAT | `SingleStateOptimizer<boolean[]>` | WalkSAT-style local search | `max-sat-local-search` |
| Semidefinite programming | `UnitVectorRelaxationStation` + Gram matrix state | Max-Cut SDP relaxation | `sdp-maxcut-relaxation` |
| Multi-objective optimization | `ParetoArchiveStation` + `ParetoCandidateToken` | risk/return portfolio front | `pareto-portfolio` |

Together with the existing `simulated-annealing`, GA-backed
`internal-solver-network`, `stochastic-lp`, `policy-gradient-corridor`,
and `actor-critic-grid`, this gives at least two runnable examples in
the major buckets: convex/smooth, combinatorial, metaheuristic,
sequential decision/control, and numerical continuous optimization.

### Neural networks — numerical models moving through DES queues

Neural nets are represented as hybrid DES components: dense layers do the
numeric forward/backward pass, while inputs, labels, predictions, RL
actions, and ODE solve requests move through station-local queues as
tokens. The reusable pieces live in:

| Capability | Class / function | JSON id |
| ---------- | ---------------- | ------- |
| Feed-forward model | `FeedForwardNetwork` | |
| Inference station | `NeuralNetworkStation` | |
| Supervised training station | `SupervisedNeuralNetworkStation` | `neural-xor` |
| Neural Q-learning agent | `NeuralQLearningAgent` | `neural-qlearning-corridor` |
| Neural vector-field ODE solve | `NeuralODESolverStation`, `solveNeuralODE` | `neural-ode-decay` |

This gives neural networks the same station/movable semantics as the
rest of the engine while still reusing the numerical ODE solvers and the
existing MDP/RL environment loop.

The neural JSON examples can also emit standalone HTML animations through
the normal `runtime.outputs.html` hook:

```bash
node dist/des/main-from-json.js examples/neural-xor.json
node dist/des/main-from-json.js examples/neural-qlearning-corridor.json
node dist/des/main-from-json.js examples/neural-ode-decay.json
```

### Collaborative inference — sparse opinion learning

`collaborative-inference` ranks a large option set from overlapping,
subjective partial opinions. A source emits respondent movables, the
survey encoder turns local ratings/rankings into `RatingEvidenceToken`
and `PairwisePreferenceToken` movables, an aggregator station accumulates
evidence, a ranking station applies shrinkage and score fusion, and the
result sink stores the final ranking.

Respondent credibility can be weighted by age-adjusted experience and by
calibrated breadth. Claimed experience is capped by `age - minCredibleAge`
(so a 20-year-old cannot claim 10 credible years if `minCredibleAge=15`),
and a second inference pass can increase weight for respondents who have
used two or more high-scoring items.

Three JSON scenarios are included:

```bash
node dist/des/main-from-json.js examples/collaborative-inference-programming-languages.json
node dist/des/main-from-json.js examples/collaborative-inference-model-validation.json
node dist/des/main-from-json.js examples/collaborative-inference-learning-resources.json
```

## What "DES" means in this codebase

DES = **Discrete Event System**, broader than the textbook "Discrete Event
Simulation". The same engine is used for three distinct workloads, all on
the same station/movable substrate and the same fixed-step tick clock:

| Workload class | What lives at each tick | Examples in this repo |
|---|---|---|
| **Simulation** of stochastic processes | Random arrivals, stochastic state transitions | SEIR, two-disease, FactMachine POMDP, soccer-game realisation, dispatch evaluation, traffic flow |
| **Control** loops (open-loop & closed-loop) | Sensor → controller → actuator updates | Elevator MDP dispatch, RLC step response, MPC rolling-horizon |
| **Iterative algorithms** | One algorithmic step (e.g. one pivot, one generation, one wave) | Simplex-as-DES, **incremental LP**, **stochastic LP / Benders**, max-flow/min-cut, GA-TSP, shortest-path waves, MCTS rollouts, value iteration, calculus solver |

The acronym DES is preserved because every interpretation works:

- *Discrete-event simulation* (the textbook reading) — for the stochastic-process column
- *Discrete-event system* (the systems-and-control reading) — for control loops
- *Discrete-event semantics* (the algorithmic reading) — for iterative algorithms whose state changes at well-defined ticks

The architectural commitments stay constant across all three: stations
are stationary entities holding state; movables flow between them
carrying values; the tick clock advances in fixed steps; there is no
global future event list. The simulation framing was the historical
starting point, but **the engine has always been a system**, and that's
what this codebase has become.

### Programs as source-to-sink streams

The preferred modeling rule is: a program is a stream graph. Sources
emit movable tokens, stationary entities transform or route those tokens,
and sinks record terminal results. In this sense, ordinary free functions
in a program are also stationary entities. If a function does not need a
backlog, processing delay, capacity, or local queue state, model it with
`TransformEntity` / `FunctionEntity` from
`src/des/general/des-base/transform-entity.ts`: the token is transformed
immediately on `take(...)` and emitted downstream.

Use queued `DESStation` subclasses when the function-like step genuinely
has resource contention, waiting time, stochastic service, retained
state, or validation lifecycle. Use movable and smart-movable methods
for behavior that mutates the entity carrying the data. This keeps
simple functions visible in topology and animation without pretending
that every function call creates artificial queueing.

## What this engine is

The system is a graph of two kinds of nodes:

- **Stationary entities** (stations, sources, sinks, decision nodes) live
  at fixed positions in the graph and own internal state plus their own
  random variables. Examples in this codebase:
  - `EntityProcessor` (`src/des/entity-processing/processing.ts`):
    a three-queue station with input / processing / output queues and a
    concurrency limit. M/M/c-style.
  - `PerIndividualProcessor` (`src/des/entity-processing/per-individual-processor.ts`):
    a single-queue station where each entity carries its own remaining-time
    clock. M/M/∞-style.
  - `EntitySource`: emits entities at a configurable inter-arrival
    distribution.
  - `ProbabilityDecisionEntity`: branches an entity to one of N successors
    based on a probability vector.
- **Moving entities** flow between stationary entities. Each one knows
  the stations it has visited and the time it spent at each.

Every step, the runtime calls `runTimeStep(stepSize)` on every node. The
nodes themselves resolve "what should leave me this step?" using their
own RNGs and local state. Order of node execution is randomized by
`fisherYatesShuffle` (a generator-based Fisher–Yates) to make the
simulation order-independent.

When a queueing station must choose exactly one downstream acceptor,
`EntityProcessor` and `PerIndividualProcessor` also support an explicit
`outputRouting` policy:

| Policy | Semantics |
|---|---|
| `random` | Fisher-Yates over out-connections for each routed entity; default bias-removal behavior |
| `round-robin` | declared connection order, rotating after each successful route |
| `ordered` | declared connection order every time; intentional priority/bias |

Queues themselves remain FIFO. The policy only changes the competitive
choice among out-connections once an entity is ready to leave a station.

Network mutex / lock protocols use the same idea with parent and child
movables plus generic substations. `des-base/stateful-token.ts` lets a
movable be stateful or stateless, record its state history, spawn child
tokens with parent/root causality, and share token accounting through a
reusable registry. `des-base/episode-accounting.ts` centralizes scalar and
vector episode reward/length bookkeeping for environment, RL, policy-gradient,
and multi-agent stations. `des-base/composite-station.ts` lets a station own
substations behind explicit input/output ports. In
`general/network-mutex.ts`, Station A is a composite: an internal queue
substation keeps parent work FIFO and spawns child `LockRequestToken`s to
Station B; an internal processor substation starts only after a child
`LockGrantToken` returns, sends a child `LockReleaseToken`, then emits the
parent item to Station C. This models request-spawning-request behavior
without a hidden global scheduler:

```bash
npm run network-mutex
npm run test-network-mutex
```

There is **no global event scheduler.** Nothing in the system says
"entity X arrives at station Y at time t". Each station discovers what
arrives by polling its own input queue at the start of each step.

## How it differs from a classical FEL kernel

A classical DES kernel (SimPy, SSJ, Arena, OMNeT++, AnyLogic) maintains
a single global priority queue of *future events* ordered by event time.
The main loop is:

```
while (FEL non-empty) {
    e = FEL.popMinTime();
    t = e.time;
    e.fire();        // may push new events into FEL
}
```

Time is advanced exactly to the next event. Whoever fires events
schedules future ones — anywhere in the model.

Our engine inverts this:

| Aspect                | Our engine                                       | Classical FEL kernel                              |
|-----------------------|--------------------------------------------------|---------------------------------------------------|
| Time advance          | Fixed step `stepSize`; everyone progresses       | Adaptive; jumps to next scheduled event           |
| Event store           | Distributed across each station's queues         | Single global priority queue                      |
| When does X happen    | Each station decides locally each step           | "Schedule X at time T" globally, anywhere         |
| Order tiebreak        | Fisher–Yates shuffle of node execution order     | Event-time priority + insertion-order tiebreak    |
| Knowledge scope       | Each station only sees its own queues + RNGs     | Anything that holds the FEL handle                |
| Time precision        | Discrete buckets of `stepSize`                   | Continuous (floats; only float precision limits)  |
| Cost per quiet station| O(1) `runTimeStep` no-op per step                | O(0) — quiet stations cost nothing                |
| Cost per busy station | O(queue work × steps)                            | O(events fired)                                   |
| Adding a new station  | Implement `runTimeStep`, plug into graph         | Same                                              |
| Composability         | Sub-graphs are stations themselves               | Same                                              |
| Debuggability         | Step through one tick, inspect every station     | Step through one event at a time                  |

Concrete example for the SEIR-with-hospitalization model in this repo:
a classical FEL would schedule "patient #42 leaves I-S at t = 12.7" the
moment patient 42 enters I-S. Our engine instead has the I-S station
ask, every step, "given my current queue and the U(0.20, 0.40) residence
distribution, who leaves now?" and the answer is computed locally.

The two models give the **same expected populations** at small enough
`stepSize` (proven in `validate-references.js` and `validate-with-externals.js`),
but their internal mechanisms are very different.

## How it differs from other simulation paradigms

### vs Gillespie SSA (continuous-time stochastic, compartment-level)

`gillespie-runner.ts` ships an SSA in this repo for comparison.

| Aspect              | Our engine                          | Gillespie SSA                              |
|---------------------|-------------------------------------|--------------------------------------------|
| State representation| Individual entities                 | Compartment counts only (no entity ID)     |
| Service distribution| Any (Uniform, Exponential, …)       | Exponential only (in the pure direct method)|
| Event resolution    | Per-step batch                      | One reaction at a time                     |
| Time advance        | Fixed `stepSize`                    | Exponentially distributed `dt`             |
| Per-entity history  | Yes — `stationsVisitedCount`, etc.  | No                                         |
| Cost                | O(N entities · stations · steps)    | O(N events) — scales to billions           |
| Best for            | Mid-scale entity-tracking with arbitrary distributions | Large-N or rare-event Markov chains   |

### vs queueing-network DES (Ciw, JMT, Erlang, JINQS)

`external-references/ciw/seir.py` runs Ciw on the same model.

| Aspect              | Our engine                                       | Queueing-network DES                       |
|---------------------|--------------------------------------------------|--------------------------------------------|
| Topology            | Custom code, any directed graph                  | Routing matrix between nodes               |
| Service stages      | Multi-queue per station (input → proc → out)     | Single per-node service                    |
| Branching           | First-class `ProbabilityDecisionEntity` nodes    | Encoded in routing probabilities           |
| Server scheduling   | Concurrency cap (M/M/c) or per-individual (M/M/∞)| Per-node: 1 / c / ∞ servers                |
| Heterogeneous routes| Trivially supported                              | Multiple customer classes                  |

The queueing-network paradigm is a *strict subset* of what this engine
can express, but with a more focused API for queues-of-customers
problems.

### vs ODE / mean-field

`ode-runner.ts` and `difference-runner.ts` ship deterministic
counterparts.

| Aspect            | Our engine                  | ODE / difference equations          |
|-------------------|-----------------------------|-------------------------------------|
| Stochasticity     | Yes (intrinsic noise)       | No (mean field)                     |
| Entity identity   | Yes                         | No (just compartment counts)        |
| Solver            | Time-step + Fisher–Yates    | RK4, LSODA, forward Euler, …        |
| What you get      | Sample paths + distributions| Expected values only                |
| Non-linear feedback| Naturally captured via per-entity decisions | Must be encoded in the rate matrix|

The ODE is the *infinite-population mean-field limit* of the stochastic
engine. As the entity count grows, the engine's expected populations
converge to the ODE solution.

### vs agent-based modelling (NetLogo, Mesa, Repast)

| Aspect              | Our engine                              | ABM                                  |
|---------------------|------------------------------------------|--------------------------------------|
| Agent autonomy      | Limited — entities are passive between stations | Full — agents act each tick          |
| Interaction model   | Through station queues                   | Spatial / network neighbour rules    |
| Topology            | Directed graph of stations               | Often grid or continuous space       |
| Update model        | Stations update; entities are state      | Each agent's `step()` runs           |

The two paradigms can simulate similar problems but with very different
ergonomics. ABM is better when *each entity's autonomous behaviour
matters most*. This engine is better when *what happens at each station
matters most* and entities are mostly conduits.

### vs the new in-engine M/M/∞ variant (`PerIndividualProcessor`)

The framework now ships two kinds of stationary processor:

| | `EntityProcessor`                              | `PerIndividualProcessor`                          |
|-|-----------------------------------------------|---------------------------------------------------|
| Internal queues       | 3: input, processing, output           | 1: items with per-entity remaining-time          |
| Service discipline    | M/M/c (one service draw, c concurrent)| M/M/∞ (each entity has its own clock)             |
| Latency artifact      | ~2 `stepSize` minimum dwell           | ~0.5 `stepSize` average                          |
| Equivalent FEL        | FEL with `service: 'fifo'`            | FEL with `service: 'individual'`                 |
| Mathematical limit    | RK4 ODE with M/M/c rates              | RK4 ODE with M/M/∞ rates                         |
| Best for              | Servers / capacity-limited systems    | Independent residence times (epidemics, etc.)    |

The `M/M/∞` processor is the one that matches most epidemiological models
and was added specifically to remove the structural latency bias of the
three-queue `EntityProcessor`. Both ship in this repo and both are
verified.

## Trade-offs of the station-local fixed-step approach

### Where the framework shines

- **Visual programming intuition.** A station is a "box with input and
  output pipes". This composes naturally with a force-directed graph
  visualiser (see `app/`).
- **Locality.** Each station only knows its own queues and its own RNGs.
  No god-object holding global state.
- **No event-list to debug.** When something goes wrong you step through
  one `runTimeStep` call per node, not one event at a time.
- **Easy hierarchical composition.** A subsystem with internal stations
  can present itself as a single station to its parent — its
  `runTimeStep` just calls its children's.
- **Order-independence is explicit.** Fisher–Yates over the station list
  every step makes "the framework doesn't depend on iteration order" a
  visible architectural property, not an emergent one. Competitive
  out-connection routing can be randomized, round-robin, or intentionally
  ordered per station.

### Where it pays a price

- **Time-discretization bias.** Every fixed-step DES has it — populations
  in the framework's "fast" compartments (mean residence ≪ one step) are
  inflated by the structural latency of the queues. `stepsize-sweep.ts`
  measures this; `PerIndividualProcessor` reduces it; `stepSize → 0`
  removes it. Quantified statistically in `CHANGELOG.md`.
- **Wall-clock cost on quiet networks.** A FEL kernel skips quiet
  intervals — this engine still calls `runTimeStep` on every station at
  every step. For systems with many idle stations, FEL is faster.
- **Less natural for rare events.** If an event happens once per
  10⁶ time units, FEL fires it in O(1); we'd grind through 10⁶ idle
  steps.
- **More memory per entity.** Three queues × stations holds more
  references than a FEL with one event in flight per entity.

In short: this kernel is best for densely-active, mid-scale,
heterogeneous-station simulations where locality and visual composition
matter more than asymptotic accuracy. It is a *bad* fit for very
large-N rare-event Markov chains (use Gillespie or FEL there).

## Beyond the SEIR model

The same engine ships four other top-level simulations that exercise the
framework on very different problem classes, each paired with an external
reference implementation in a different paradigm:

| Simulation                 | Domain                       | External reference                | What it tests on the engine                                           |
|----------------------------|------------------------------|-----------------------------------|-----------------------------------------------------------------------|
| `main-convolution.ts`      | DSP / linear filter          | `numpy.convolve`                  | Token-flow signal pipeline, ring-buffer station                        |
| `main-backpropagation.ts`  | Recursive ML training        | naive-loop Python on same init    | Forward + backward token streams, sequential SGD discipline           |
| `main-electric-circuit.ts` | Continuous physics (ODE)     | analytical + `scipy LSODA`        | Synchronous data-flow with shared mutable state, forward-Euler order  |
| `main-elevator.ts`         | Discrete operations / agents | `SimPy` (continuous-time FEL)     | Stationary cars + floors with people as the only moving entity        |
| `main-court-mdp.ts`        | MDP / control system         | Python value iteration            | Per-station policy lookups, 4-stage station routing, MDP simulation    |
| `main-two-disease.ts`      | Coupled compartmental epi    | `scipy LSODA` + Gillespie SSA     | Cross-station interaction via synchronous WorldCensus + competing risks |
| `main-contact-seir.ts`     | Pair / triplet contact epi   | mass-action SEIR (in-engine)      | Explicit person-to-person transmission, super-spreader Gini, complex contagion |
| `main-newsvendor.ts`       | Single-period stochastic inv | scipy / numpy critical fractile   | Newsvendor q* via three independent methods (analytical, brute, MDP) |
| `main-inventory-mdp.ts`    | Multi-period inventory MDP   | numpy value iteration             | Discovers base-stock and (s, S) policies via VI without structural assumption |

Each has its own validator (`src/des/runners/validate-<sim>.ts`) and a
section in `src/des/runners/README.md` that documents the agreement bound
and the most recent observed error. Bounds:

- **Convolution**: 1.1e-16 max-abs error vs `numpy.convolve` (~ 1 ULP).
- **Backpropagation**: 3.6e-15 max-abs error on every weight vs the
  Python reference after **10 000 SGD steps** with identical initial
  weights. 16 ULPs of accumulated float drift over 10 000 steps.
- **Electric circuit**: forward-Euler convergence order = 1.01
  (textbook); 1.85e-3 vs scipy LSODA at dt = 1e-3; diverges spectacularly
  at dt = 0.5 (above the stability bound).
- **Elevator**: aggregate metrics within 10% of SimPy. Mean wait differs
  by 0.22 s ≈ 0.4 × stepSize; the ~6 s per-person mean diff is caused by
  dispatch-decision sensitivity, not numerical error — once the framework
  and SimPy disagree by a fraction of a tick, the same person can be
  routed to a different elevator and follow a totally different path.

The elevator simulation in particular is interesting: the user-facing
question was "elevator cars need to make decisions and hold data, but
they also move — so how do they fit into a stationary/moving entity
framework?" The clean answer: **both elevators and floors are stationary
entities.** Floors hold up- and down-bound waiting queues. Elevators
hold a passenger queue plus continuous `currentFloor` position plus an
IDLE/MOVING/SERVING state machine and run SCAN/LOOK dispatch each tick.
**People are the only moving entity** in the simulation; they flow
`Source → Floor[from] → Elevator[k] → Floor[to] → ExitSink`.

## Control systems and MDPs

Two MDP-based control demonstrations sit on top of the framework. Both
treat decisions inside the simulation as choices in a Markov decision
process, derive (or hand-code) a policy, and measure the impact on
aggregate outcomes.

**All MDP machinery is implemented in TypeScript inside this engine.**
The state space, transition function, reward function, value-iteration
solver, and the policies that consume `V*`/`π*` all live in
`src/des/mdp/` and `src/des/main-court-mdp.ts`. The Python implementation
in `external-references/court-mdp/court_mdp.py` is a **reference
implementation used only for validation** — `validate-court-mdp.ts`
loads both outputs and verifies that the framework's `V*` and `π*` are
**bit-equal** to Python's. The framework's MDP solver does not depend
on Python in any way at run time.

### USACC court MDP

`main-court-mdp.ts` ports the
[US Anti-Corruption Court Project's MDP spec](https://oresoftware.github.io/us-anti-corruption-court-project/mdp)
into the framework as a 4-stage station graph (Submission → Validation
→ Admission → Trial). Cases are moving entities carrying a fully-
observable 6-factor state vector (864 product states + 3 absorbing
terminals). Each station applies a **policy** to choose one of 8
actions; the action induces a stochastic factor update plus possibly a
stage advance, terminal accept, or terminal close.

**This is the MDP, not POMDP, version** — per the user's request and
per the source page's note that the simplified MDP "is useful for
queue management, reviewer allocation, and funding logic". The visible
state IS the ground truth; no hidden variables.

Four policies ship side by side:

  * `RejectAllPolicy`        — closes every case immediately.
  * `AlwaysEscalatePolicy`   — pushes every case to trial.
  * `NaiveThresholdPolicy`   — hand-tuned per-factor heuristic.
  * `OptimalPolicy`          — argmax action from value iteration.

Value iteration (`src/des/mdp/value-iteration.ts`) computes V* and π*
in ~25 ms (64 sweeps to ‖ΔV‖∞ ≤ 1e-9). The Python reference
(`external-references/court-mdp/court_mdp.py`) runs the same value
iteration; `validate-court-mdp.ts` checks **bit-exact** agreement on V*
and **identical π* on all 864 states**.

Latest run on 5 000 cases:

| Policy            | Mean reward | % accepted | % closed | % exhausted |
|-------------------|-------------|------------|----------|-------------|
| reject-all        |      +19.71 |        0.0 |    100.0 |         0.0 |
| always-escalate   |      −97.64 |       55.5 |      0.0 |        44.5 |
| naive-threshold   |      +61.57 |       84.2 |     15.8 |         0.0 |
| **optimal (π\*)** | **+140.60** |       63.5 |     36.5 |         0.0 |

The optimal policy beats the naive heuristic 2.3× by being more
selective about what to escalate — it accepts fewer cases (63.5% vs
84.2%) but closes more weak ones, dramatically reducing the average
penalty paid for advancing low-quality cases.

### Coordinated elevator dispatch

The elevator simulation now ships **two dispatch policies** and runs
them side by side on the same arrival schedule. The `coordinated`
policy implements one MDP-style rule: each elevator's "pick next
target" decision myopically minimises the expected number of redundant
stops by skipping any (floor, direction) already claimed by another
elevator. This single rule covers both user-stated optimisations:

  * "If a car is already going to a floor, don't send a second."
  * "Skip a floor going up so other cars can stop at that floor."

Sweep across 5 seeds × 4 arrival rates (`SEEDS=1..5
LAMBDAS=0.1,0.2,0.3,0.4 SIM_T=3600`) shows the benefit grows
monotonically with load:

```
  λ=0.10:  meanWait −5.7%   p95Wait −24.6%   meanTotal  −2.7%
  λ=0.20:  meanWait −14.3%  p95Wait −25.0%   meanTotal  −7.6%
  λ=0.30:  meanWait −22.4%  p95Wait −32.3%   meanTotal −12.8%
  λ=0.40:  meanWait −26.5%  p95Wait −39.8%   meanTotal −16.5%
```

At heavy load (λ = 0.4 arrivals/s), coordination cuts the worst-case
(p95) wait by ~40% with no added simulation cost. The SimPy validation
of aggregates (within 10%) is preserved by the uncoordinated baseline;
coordination is a strict improvement.

The driver is `src/des/runners/compare-elevator-dispatch.ts`. The
single-shot `main-elevator.ts` also runs both modes by default and
prints the percentage improvement.

A third dispatch mode, `coordinated-pickup`, adds **opportunistic
mid-flight pickups**: an elevator that is already moving toward a
target may make an extra "pit stop" at an intermediate floor whose
call matches its direction, but only if (a) it is not already full and
(b) the call is unclaimed by another elevator. This required two
extensions to the `Coordinator`:

  * It tracks **which** elevator claimed each call (not just whether one
    did), so a second elevator can decide it is safe to pit-stop.
  * Elevators expose `isFull()` and `spareCapacity()` so the dispatcher
    can decline an opportunistic pickup the elevator cannot honor.

Across the same 5 seeds × 4 arrival rates × 12 floors sweep:

```
  λ=0.10:  meanWait −9.6%   p95Wait −13.4%   meanTotal  −5.7%
  λ=0.20:  meanWait −33.4%  p95Wait −38.0%   meanTotal −22.0%
  λ=0.30:  meanWait −47.5%  p95Wait −51.0%   meanTotal −34.5%
  λ=0.40:  meanWait −51.2%  p95Wait −55.4%   meanTotal −41.4%
```

(All percentages relative to the uncoordinated baseline; see
`compare-elevator-dispatch.ts`.)

### Validating the elevator/floor architecture

Because both elevators and floors are *stationary* and they exchange
state (calls, claims, current position) every tick, the elevator
simulation is the most architecturally interesting in this repo. To
verify it is correct we ship `src/des/test/elevator-invariants-test.ts`,
which runs 45 configurations (varying `nFloors`, `nElevators`,
`capacity`, `arrivalRate`, `dispatchMode`) and after every tick checks
**seven families of invariants**:

  1. *Conservation of people* — every person ever spawned is currently
     in exactly one of: a floor's up/down queue, an elevator's
     passenger queue, or the exit sink.
  2. *Capacity* — `passengers.length ≤ capacity` for every elevator.
  3. *Position bounds* — `1 ≤ currentFloor ≤ nFloors`.
  4. *State-machine consistency* — IDLE elevators have `direction='idle'`
     and zero passengers; MOVING elevators have a non-idle direction;
     SERVING elevators have non-negative service-remaining.
  5. *Timestamp monotonicity* — `boardTime ≥ arrivalTime`,
     `exitTime ≥ boardTime`.
  6. *Floor-queue direction* — anyone in `floor.upQueue` actually wants
     to go up; same for down.
  7. *Coordinator exclusivity* — no two elevators are committed to the
     same `(target, direction)` *for a call*. Two may move to the same
     target if they each have a passenger destination there (passenger
     destinations are not subject to coordinator claims).

All seven invariants pass on all 45 configurations (`PASS  45/45`).
This sanity-checks the cross-station communication pattern and
gives confidence that adding new dispatch policies won't quietly break
correctness.

## Cross-station interaction: the two-disease model

`main-two-disease.ts` is the framework's most demanding stress test of
station-station interaction: a six-compartment model (S, A, B, AB, R, D)
where two diseases A and B can spread independently *and* a person
already infected with one can also catch the other and become co-infected
(AB). Co-infected individuals die at a configurable rate (default 50%,
intermediate between A's 40% and B's 60%).

The interaction is real: a station's transition probabilities depend on
the populations of *other* stations (the prevalences I_A, I_B). In a
naive implementation, the order in which compartments are processed
within a tick would change results — if S runs first, it sees the start-
of-tick prevalences; if S runs last, it sees prevalences updated by A,
B, and AB during this tick. To make the model **order-independent**, we
add an explicit `WorldCensus` station that runs first each tick and
freezes the global counts in a shared object that every compartment
*reads but never mutates*. This is the **synchronous data-flow** pattern
already used in `main-electric-circuit.ts`.

```
  WorldCensus  ──snapshots S, A, B, AB, R, D──┐
       │                                       ├──── all compartments read this snapshot
       ▼                                       │
  Compartment[S, A, B, AB] ──people moving──> Compartment[A, B, AB, R, D]
```

Per-tick transitions in each compartment use the **exact discrete-time
competing-risks** formula (see `general/random-variables.ts`):

```
  P(no event)  = exp(−Λ·dt)         where Λ = Σ λ_k
  P(event k)   = (λ_k / Λ) · (1 − exp(−Λ·dt))
```

This is unbiased for any `dt`, unlike the linear approximation
`P(event k) ≈ λ_k · dt` used by many naive DES kernels (which carries a
`Λ·dt/2` first-order bias per tick).

Validation against Python (200 reps, dt=0.1, simT=200, N=1000):

```
  Compartment  ∫framework  ∫LSODA   ∫SSA      |Δ| vs LSODA  |Δ| vs SSA
  ──────────────────────────────────────────────────────────────────────
  S            12975       12356    12768          5.01 %       1.62 %
  A             2802        2818     2862          0.57 %       2.10 %
  B             1604        1431     1624         12.07 %       1.25 %
  AB            3554        3628     3344          2.05 %       6.28 %
  R            93485       94601    93908          1.18 %       0.45 %
  D            85480       85065    85394          0.49 %       0.10 %

  Welch test on final D: framework 477.78 ± 24.04 (n=200)
                         SSA       476.39 ± 29.38 (n=200)
                         t = 0.516, p = 0.606  →  PASS

  ✓ ∫-rel-err vs LSODA, monotonic (R, D)        < 5%
  ✓ ∫-rel-err vs LSODA, transient (S, A, B, AB) < 20%
  ✓ ∫-rel-err vs SSA-mean (all)                 < 10%
  ✓ Welch p > 0.01 on final D
```

**Why two tolerance tiers vs LSODA?** R and D are monotonic
accumulators — they only grow over the epidemic — so their integrated
populations are dominated by the deterministic limit and the ensemble
variance is small. S, A, B, and AB are transient (rise then fall), so
peak-time variance dominates: the ensemble peak is broader than each
individual replicate's spike, and the deterministic ODE peak does not
equal the mean-of-stochastic-peaks. This is a known property of
nonlinear stochastic systems (Anderson & May 1992, ch.2). The framework
matches **SSA** (also stochastic) within 10% on every compartment,
including the transient ones, because both have the same peak-time
spread.

The driver is `src/des/runners/validate-two-disease.ts`. The Python
reference (`external-references/two-disease/two_disease.py`) implements
both an LSODA mean-field ODE and a per-person Gillespie SSA.

## Math expressions as DES models: ODE / PDE / Poisson on station networks

The framework solves continuous-mathematics problems (ODEs, PDEs,
Poisson, quadrature, root finding) by *building a station network
from the equation* and running it through the same `Census` +
`runTimeStep` loop used for compartmental epidemics. There is no
separate "ODE engine" or "PDE engine"; the equation's discretisation
becomes the per-station update rule.

### Pipeline

1. The user provides the equation as one or more **expression
   strings** (e.g. `RHS='v;-y'` for the harmonic oscillator, or
   `RHO='2*pi^2*sin(pi*x)*sin(pi*y)'` for Poisson source).
2. `general/expr.ts` parses each string into an AST. The same
   engine supports symbolic differentiation, `simplify`, and JS
   compilation via `toFunction`.
3. `general/equation-to-stations.ts` builds a `FieldSimulation`
   whose stations represent:
   - **ODE system:** one station per state variable. The RHS
     expression is evaluated using the census snapshot of all
     other variables.
   - **1-D PDE** (heat / wave / advection): one station per
     spatial cell. Each cell's updater reads its left and right
     neighbours from the snapshot.
   - **2-D Poisson:** one station per `(i, j)` grid cell on an
     Nx × Ny mesh. Each iteration ("tick") is one Jacobi /
     Gauss-Seidel / SOR sweep over the five-point stencil.
4. The standard `runTimeStep` loop drives the simulation; no
   special-case scheduler.

### What's in the box

| File | Role |
|---|---|
| `general/expr.ts` | AST + parser + evaluate + symbolic `diff` + `stringify` + `simplify` + Richardson numerical-derivative cross-check |
| `general/quadrature.ts` | trapezoidal, Simpson, adaptive Simpson, Gauss-Legendre, Monte Carlo (1-D and N-D) |
| `general/root.ts` | bisection, Newton, secant |
| `general/optim.ts` | gradient descent (Armijo), Newton, BFGS |
| `general/ode.ts` | euler, rk2 Heun, rk4, rk45 Dormand-Prince adaptive, backward Euler |
| `general/field-station.ts` | `Station`, `Census`, `FieldStation`, `FieldSimulation` |
| `general/equation-to-stations.ts` | `buildODESystem`, `buildField1D`, `solvePoisson2D`, `thomas` |
| `main-calculus.ts` | dispatcher: `PROBLEM=expr` / `ode` / `pde` / `poisson` |
| `external-references/calculus/calculus.py` | scipy.solve_ivp DOP853, scipy.LSODA on FD heat, scipy.quad, hand-coded Jacobi for 2-D Poisson |
| `runners/validate-calculus.ts` | 6 studies, 26/26 PASS — incl. station-net ≡ pure-math RK4 bit-equality, station Jacobi ≡ scipy Jacobi bit-equality |
| `test/calculus-test.ts` | 31/31 PASS unit tests |
| `animation/scenes/calculus-scene.ts` | 1-D field-strip scene (one bar per station, signed-magnitude colouring), 2-D Poisson false-colour scene |

### Why station-network ≡ pure-math (bit-level)

Two key properties make the station refactor mathematically lossless:

  - The **Census snapshot** is taken at the start of every tick and
    is read-only for the rest of the tick. No station ever reads
    another station's mid-tick value. This is the same invariant
    that makes `main-two-disease.ts` order-independent.
  - The per-station updater encodes the *exact same* discretisation
    formula as the pure-math reference solver — there is no
    rediscretisation, just a structural reorganisation of which
    object owns which scalar.

`validate-calculus.ts` Study 3 verifies this: for the harmonic
oscillator y'' + y = 0 over t ∈ [0, 4π] at dt = 0.001, the station
network's final y(t) is bit-identical to `general/ode.rk4`'s — same
floating-point bits, |Δ| = 0.

### Why station-network ≡ scipy (where the algorithm is identical)

`validate-calculus.ts` Study 6 runs Jacobi iteration on a 41×41
Poisson problem against scipy's hand-coded numpy Jacobi at the same
tolerance. Both terminate at exactly **4095 iterations** with
**identical** max error 5.11e-4 vs the analytical
`u(x, y) = sin(πx)sin(πy)`. Because the Jacobi update is a pure
linear functional of the previous snapshot, this bit-identity is
reproducible.

### Verified scheme behaviour

```
=== STUDY 6: 2-D Poisson, Jacobi / Gauss-Seidel / SOR ===
  PASS    Jacobi pins to sin·sin within 1e-3       (4095 iters, maxErr 5.11e-4)
  PASS    Gauss-Seidel pins to sin·sin within 1e-3 (2161 iters, maxErr 5.13e-4)
  PASS    SOR(ω=1.85) pins to sin·sin within 1e-3  ( 149 iters, maxErr 5.14e-4)
  PASS    Gauss-Seidel ~2× faster than Jacobi
  PASS    SOR(ω=1.85) ~30× faster than Jacobi
  PASS    station Jacobi iter count == scipy Jacobi iter count   (4095 vs 4095)
  PASS    station Jacobi maxErr ≡ scipy Jacobi maxErr             (|Δ|=0.00e+0)
```

Order-independence of *Jacobi* (snapshot-based) is verified in T6 of
`calculus-test.ts`: two different shuffle seeds produce
bit-identical final fields. Gauss-Seidel and SOR deliberately read
live (already-updated) values for faster convergence and so are
shuffle-dependent by design.

### Running

```bash
PROBLEM=expr  EXPR='x^2*sin(x)+exp(-x)' npm run calculus
PROBLEM=ode   NAMES='y,v' RHS='v;-y' Y0='1,0' DT=0.001 T_END=12.566 npm run calculus
PROBLEM=pde   FAMILY=heat ALPHA=0.1 N=51 T_END=0.5 ANIMATE=1 npm run calculus
PROBLEM=pde   FAMILY=wave C=1 N=51 T_END=0.5 npm run calculus
PROBLEM=poisson N=41 OMEGA=1.85 TOL=1e-8 ANIMATE=1 npm run calculus

npm run validate-calculus     # cross-validate against scipy + sympy
npm run test-calculus         # unit tests
```

`ANIMATE=1` writes `out/calculus-heat1d.html` (or `calculus-poisson2d.html`)
— a self-contained file showing each station's value evolving in time
(1-D) or the converged Poisson solution as a false-colour grid (2-D).

### Limitations / honest framing

  - The expression engine supports `+ - * / ^`, unary `-`, and the
    elementary functions `sin / cos / tan / exp / log / sqrt / abs`.
    No symbolic constants (`pi`, `e`); pass them numerically.
  - `buildField1D` heat with `btcs` is implemented for *constant*
    α only (α evaluated at x = 0 of the spatial grid). Variable α
    BTCS would need a per-row tridiagonal that builds the stencil
    from `alphaFn(x_i)` — a 5-line change in the wrapper.
  - 2-D wave / 2-D heat are not yet wired (the Field2D substrate is
    in place; only the per-cell updater is missing).
  - PDE animation files for N > 200 cells × > 500 ticks become
    multi-MB; cap with `ANIM_FRAMES=120` if needed.

## Linear programming: external solver, MDP-as-LP, and simplex-in-DES

This is where the framework starts living up to its broader "discrete
event system" framing. The original DES philosophy assumed simulation:
one-shot LP solves were delegated to an external simplex / interior-point
package because they exploit polyhedral geometry the framework didn't
expose. That stance was correct *for one-shot solves*. As soon as the LP
is solved repeatedly (warm-start after modifications, Benders cuts,
column generation, branch-and-bound, MPC), the right substrate is
exactly a long-lived station that holds the basis and accepts movables
that mutate it — see the **Incremental LP** and **Stochastic LP**
sections below for the concrete realisation. We expose three
integration patterns plus a curiosity (simplex-as-DES) here, and then
build the warm-startable + stochastic variants on top.

### Solver dispatch

`general/lp.ts` defines `LPProblem` / `LPSolution` types and a
`solveLP(p)` function that picks the backend from the `LP_SOLVER` env
var:

```
LP_SOLVER=internal             in-process two-phase revised simplex (small-LP fallback)
LP_SOLVER=scipy:highs          scipy.optimize.linprog method='highs' (DEFAULT)
LP_SOLVER=scipy:highs-ds       HiGHS dual simplex
LP_SOLVER=scipy:highs-ipm      HiGHS interior-point
LP_SOLVER=scipy:simplex        legacy scipy simplex
LP_SOLVER=scipy:interior-point legacy scipy interior-point
```

The external bridge is a thin `child_process.spawnSync` shim onto
`external-references/lp/lp_solve.py`. If scipy/python is unavailable
the dispatcher falls back to the internal solver automatically. The
external solver returns dual variables (shadow prices) and reduced
costs alongside the primal solution.

### Pattern A: plan-then-simulate (factory scheduling)

`main-lp-factory.ts` solves a 3-product × 4-machine LP for the nominal
weekly profit, then feeds the LP plan to a stochastic DES factory
simulator (lognormal processing times, Poisson breakdowns, finite
buffers). The output reveals the gap between the LP's deterministic
optimum and DES's realised throughput — typically 20–30%. A
`SWEEP=1` mode runs a robustness sweep (shrink LP RHS by 5–25%),
demonstrating that a slightly conservative plan often realises HIGHER
expected profit than the naive LP-optimal plan.

```
LP NOMINAL revenue       = $34204.72
DES realised mean ± sd   = $24356.67 ± $1147.84   over 30 reps
gap                      = $9848.06   (28.8% of nominal)
robust-LP @ 0.80 capacity = $24850.00 realised   (+$493 over plan-as-is)
```

### Pattern B: MDP-as-LP

The Bellman optimality equation has an LP characterisation:

```
min Σ_s μ_s · V(s)
s.t. V(s) − γ Σ_{s'} T(s'|s,a) V(s') ≥ Σ_{s'} T(s'|s,a) r(s,a,s')   ∀ s, a
```

`buildMDPLP` and `solveMDPAsLP` (`general/des-lp-bridge.ts`) convert
any `MDPSpec` into this LP and solve it via the external simplex /
interior-point. `main-mdp-lp.ts` runs three MDPs (5-state chain,
11-state inventory control, 16-state stochastic grid-world) and
verifies V*\_LP ≡ V*\_VI to ≤ 6e-13. This realises the chain
**real system → DES → MDP abstraction → LP formulation → simplex /
interior-point** end-to-end inside one engine.

### Pattern C: LP-as-DES — simplex driven by the DES tick loop

`general/lp-des.ts` and `main-lp-des.ts` embed simplex as a
discrete-event process. Simplex IS naturally a DES: each pivot is an
event where the algorithm walks one edge of the feasible polytope to
the next vertex, picking the steepest improving direction.

Four stations, one pivot per tick:

| Station | Role |
|---|---|
| `EnteringStation` | Scan the cost row for the steepest improving direction (Dantzig: most-negative reduced cost; or Bland: first-negative for guaranteed termination on degenerate LPs) |
| `LeavingStation` | Min-ratio test: how far along the entering edge can we travel before some basic var hits 0? |
| `PivotStation` | Elementary row operations on the tableau (Gauss-Jordan elimination) |
| `ObserverStation` | Snapshot the new vertex + objective into the trace, exactly the role `Census` plays in our SEIR / FactMachine models |

Two phases of simplex (phase-1 for feasibility, phase-2 for optimality)
share the same DES tick loop with a different cost row. The pattern
of stations + per-tick orchestration + post-hoc trace replay is
identical to every other simulation in this engine.

#### Why?

We are NOT claiming DES-as-simplex beats scipy:HiGHS on speed — a
direct simplex would run 5–10× faster because the per-tick
scaffolding is overhead. We did this to:

  1. Prove the engine is computationally general. Events, queues,
     stations, and movables suffice for vertex-walking optimisation,
     even when the LP's polyhedral geometry would normally invite a
     specialised representation.
  2. Get the trace + animation infrastructure for free. The same
     `FrameRecorder` / `HtmlPlayer` used for every other simulation
     works directly on the per-pivot vertex history. The 2-D demo
     (`PROBLEM=2var-diamond ANIMATE=1 node dist/des/main-lp-des.js`)
     renders the polytope walk as an animated path with no
     LP-specific scene code beyond mapping vertices to pixels.
  3. Bridge LP / MDP / simulation in one engine. `main-lp-des.ts`
     (DES-driven simplex), `main-mdp-lp.ts` (MDP-as-LP via simplex),
     and `main-lp-factory.ts` (LP-then-DES) all run on the same
     TypeScript runtime, with the same animation plugin, the same
     validation harness, and the same per-tick trace shape.

#### Validation

`runners/validate-lp.ts` runs nine studies, 36 / 36 checks pass:

  - Studies 1–3: in-process simplex ≡ scipy:highs ≡ scipy:highs-ds ≡
    scipy:highs-ipm on canonical LPs (2-var, diet, transportation)
  - Studies 4–5: MDP-as-LP V* ≡ value-iteration V* on chain (max\|Δ\| =
    3.33e-16) and grid-world (max\|Δ\| = 5.92e-13)
  - Study 6: 200 random feasible LPs, internal ≡ scipy:highs (max\|Δ\|
    = 3.38e-14)
  - Study 7: `LP_SOLVER` env-var dispatch routes to all 4 backends
  - Study 8: DES-engine simplex (Dantzig + Bland rules) ≡ in-process
    simplex ≡ scipy:highs on canonical LPs incl. unbounded + infeasible
  - Study 9: 50 random feasible LPs, DES-engine simplex obj ≡
    scipy:highs obj (max\|Δ\| = 8.88e-15)

### Running

```bash
# canonical 2-var LP solved through the DES engine, with animation
PROBLEM=2var-diamond ANIMATE=1 node dist/des/main-lp-des.js

# diet LP (requires phase-1 to find a starting feasible vertex)
PROBLEM=diet PIVOT_RULE=bland node dist/des/main-lp-des.js

# transportation problem
PROBLEM=transport node dist/des/main-lp-des.js

# factory scheduling — LP plan vs DES realised
node dist/des/main-lp-factory.js
SWEEP=1 node dist/des/main-lp-factory.js

# MDP-as-LP — three example MDPs
PROBLEM=inventory LP_SOLVER=scipy:highs-ipm node dist/des/main-mdp-lp.js
PROBLEM=gridworld node dist/des/main-mdp-lp.js
PROBLEM=chain LP_SOLVER=internal node dist/des/main-mdp-lp.js

# full validation
node dist/des/runners/validate-lp.js
```

## DES + MDP + LP + MCTS: combinatorial optimisation as a layered architecture

The next conceptual layer above the LP toolchain. The premise:

> DES is not the optimiser by itself; it is usually the state-transition
> simulator / evaluator / search environment. By defining decision epochs,
> system states, actions, transition dynamics via the simulator, and
> rewards/costs, the DES becomes an MDP simulator.

In this codebase the demonstration is the **multi-class parallel-server dispatch** problem (`general/dispatch.ts`, `main-dispatch-combo.ts`):

- **M** heterogeneous machines, **K** job classes
- arrivals: Poisson(λ); class c with probability p_c
- service: a class-c job on machine m takes Exp(μ_{c,m}) — class-machine interactions are the source of combinatorial structure
- decision at every arrival: which machine?
- objective: minimise long-run mean sojourn time

### The three architectural layers, all backed by the same DES

```
Layer 3 — OPTIMISATION ENGINE
  ├─ heuristics (random / round-robin / shortest-queue / SECT)
  ├─ LP fluid relaxation (simplex / interior-point) — continuous abstraction
  ├─ value iteration on the truncated MDP — exact DP
  └─ Monte Carlo Tree Search (UCT) — search with DES rollouts as oracle

Layer 2 — DECISION ABSTRACTION
  └─ MDP at arrival decision epochs; transition probabilities estimated
     by running the DES R times from each (s, a) pair. This is the
     "DES becomes the MDP simulator" pattern made operational.

Layer 1 — PHYSICAL / DYNAMIC WORLD
  └─ DES simulator (next-event, exponential inter-event times,
     FIFO per-machine queues, class-aware service rates)
```

Every policy is evaluated by the **same** DES with the **same** seeds — head-to-head numerical comparisons are fair.

### LP fluid relaxation (combinatorial → continuous)

For each class c, let `x_{c,m}` ∈ [0, 1] be the long-run fraction of class-c arrivals dispatched to machine m. The fluid LP:

```
min  t                                 (the bottleneck load t = max_m ρ_m)
s.t. Σ_m x_{c,m} = 1                ∀ c    (each class fully served)
     λ Σ_c p_c x_{c,m} / μ_{c,m} ≤ t ∀ m    (machine-m load ≤ t)
     x_{c,m} ≥ 0
```

Solved through the project's pluggable `solveLP()`, so the same instance is solvable by:

- the in-process two-phase revised simplex (`internal`)
- the **DES-engine simplex** from the previous section (each pivot is a discrete event!)
- scipy:HiGHS dual-simplex (`scipy:highs-ds`)
- scipy:HiGHS interior-point (`scipy:highs-ipm`)

Validation Study 3 confirms all four agree on the objective to **1.11e-16** — the fluid-LP pipeline is genuinely solver-agnostic.

The LP optimum `x*_{c,m}` defines a randomised dispatch policy: when class c arrives, sample machine m with probability `x*_{c,m}`. This is the cleanest example in the codebase of using interior-point / simplex INSIDE a DES.

### Empirical MDP via DES rollouts

State (q_1, …, q_M, c_arriving) truncated at q_max. For every (s, a):

```text
Run R DES rollouts starting from state s with first action a;
record (next_state, reward) for each rollout;
estimate P̂(s' | s, a) and R̂(s, a) from the empirical distribution.
```

Then ordinary value iteration on (P̂, R̂) gives V* and π*. With M=2, K=2, q_max=5 the MDP has 72 states and the whole pipeline (sample R=50 rollouts per (s,a) → run VI → derive policy) takes ~50 ms.

This is the user's loop in plain TypeScript:

> Observe DES state → choose action → DES simulates next event(s) →
> compute cost/reward → repeat.

### MCTS (UCT) with DES as the rollout oracle

`general/mcts.ts` is a generic UCT (Kocsis–Szepesvári) implementation that takes any `MCTSEnv<S>` (a clone-able state, an `applyAction`, an optional rollout policy). The dispatch adapter (`policyMCTS`) tracks per-machine **head class** in the search tree so that service-rate predictions stay accurate.

A subtle but practically important phenomenon shows up here: when MCTS uses a near-optimal rollout policy (SECT in our experiment), MCTS's UCT estimates can be **dominated by sampling noise** because there's no remaining "mistakes to correct" in the rollout. So MCTS's expected sojourn ≈ SECT's, with extra variance from the search tree. This is documented in Validation Study 5 (the assertion is "MCTS bounded by 2.5× SECT", not "MCTS beats SECT" — the latter is only true on harder instances).

### Empirical results (well-specialised instance)

M=2, K=2, μ_{c,c}=2.0, μ_{c,c'}=0.8, λ=1.6, ρ̄ ≈ 0.43:

| policy | mean sojourn | ratio vs random |
|---|---|---|
| random | 3.32 | 1.00 |
| round-robin | 2.74 | 0.83 |
| shortest-queue | 2.01 | 0.61 |
| MCTS | 1.61 | 0.49 |
| fluid-LP | 1.02 | 0.31 |
| MDP-VI | 0.95 | 0.29 |
| SECT | 0.90 | 0.27 |

The 4× spread between random and SECT confirms the dispatch decision matters. The closeness of SECT, MDP-VI, and fluid-LP confirms that on a well-specialised instance the class-aware greedy heuristic is already optimal within sampling noise.

On the heavily-loaded weak-specialisation instance (M=3, K=3, ρ̄ ≈ 0.85), the picture inverts — the layered methods pull away from the heuristics (random=10.5 vs. SECT=1.31).

### Why this matters

This is the architectural pattern that the user described:

> DES + DP — when the state/action space is small
> DES + RL — when the state/action space is large or stochastic
> DES + MCTS — when sequential decisions matter
> DES + metaheuristics — when optimising policies / schedules
> DES + Bayesian optimisation — when tuning policy parameters

DES is the unifying simulator under all of them, and `simulateDispatch` in `general/dispatch.ts` is the SAME substrate for every policy. The same problem structure is also the entry point for surrogate modelling, fluid limits, sample-average approximation, and hybrid LP/IP-decomposition strategies.

### Running

```bash
npm run dispatch                          # head-to-head comparison + LP printout
SKIP_MDP=1 N_REPS=10 npm run dispatch     # fast smoke
LP_SOLVER=scipy:highs-ipm npm run dispatch  # interior-point on the fluid LP
LP_SOLVER=internal npm run dispatch         # in-process simplex (pure JS)
npm run validate-dispatch                 # five Welch-t studies
npm run test-dispatch                     # 72 unit tests
```

External-references reference (scipy):

```bash
echo '{"M":2,"K":2,"lambda":1.6,"p":[0.6,0.4],
       "mu":[[2.0,0.8],[0.8,2.0]]}' \
  | python3 external-references/dispatch/dispatch.py fluid-lp --method highs
# {"x": [[0.9048, 0.0952], [0.0, 1.0]], "t_star": 0.4343, "status": "optimal"}
```

## Genetic algorithm for TSP, modelled inside the DES (with branch cutting)

The Travelling Salesman Problem solved by an evolutionary search where the GA itself is expressed as a station chain in the DES — see `general/genetic-tsp.ts`, `main-genetic-tsp.ts`.

```
each generation = one DES tick

   Selection  →  Crossover  →  Mutation  →  Feasibility  →  Fitness  →  Replacement
                  (OX)         (inversion /   (branch-cut /
                                swap)         penalize /
                                              repair)
```

Every chromosome is a movable (a city permutation); `OrderCrossover` (OX) provably preserves permutation validity, so the only "constraint cutting" comes from optional **precedence pairs** `(i, j)` meaning "city i must appear before city j on the tour". When such constraints are present, the Feasibility station drops infeasible offspring; with seed=42 on a 15-city instance carrying 4 precedence pairs, **293 infeasible children get cut** while the GA still converges to a feasible best.

### Three constraint-handling policies, compared in `validate-genetic-tsp.ts`

| `feasibility` | What infeasible offspring become |
|---|---|
| `cut`      (default) | dropped; the breeding station retries up to `retryLimit` times before giving up that parent pair |
| `penalize` | accepted, with objective += `penaltyPerViolation` per violated pair (large constant) |
| `repair`   | a small swap heuristic restores feasibility when possible |

On instances with 4+ precedence pairs `cut` and `repair` reliably produce feasible best-tours; `penalize` is sometimes inferior because the GA wastes search on artificially-priced infeasible regions.

### Validation against ground truth

- **Pentagon n=5** (analytical optimum = `n·2R·sin(π/n)`): GA matches to **1e-9**
- **Random n=10** with Held–Karp exact DP: GA within **0.5%** of optimum across 3 seeds
- **1-tree relaxation** lower bound is respected on all tested instances

### Run it

```bash
npm run genetic-tsp                                  # 25-city random
INSTANCE=pentagon N_CITIES=12 npm run genetic-tsp   # pentagon vs Held-Karp
PRECEDENCE=1 FEASIBILITY=cut      npm run genetic-tsp
PRECEDENCE=1 FEASIBILITY=penalize npm run genetic-tsp
PRECEDENCE=1 FEASIBILITY=repair   npm run genetic-tsp
ANIMATE=1 N_CITIES=20 GENERATIONS=80 npm run genetic-tsp
npm run validate-genetic-tsp                         # 16 / 16
npm run test-genetic-tsp                             # 18 / 18
```

## Shortest-path-as-DES: every node is a station, every relaxation is a movable

The pure "graph IS the DES" architecture. See `general/shortest-path-des.ts`, `main-shortest-path.ts`.

Each graph node IS a stationary entity holding `distance` (current best estimate from source) and `predecessor`. Each relaxation IS a "wave" movable carrying `(source_id → target_id, distance_proposal)` along one edge. The Bellman recurrence

\[
d(v) \;=\; \min_{(u,v) \in E} \bigl[\, d(u) + w(u, v) \,\bigr]
\]

is computed iteratively by the DES tick loop. Two scheduling strategies, both expressed in the same architecture:

| Algorithm | One-tick semantics | Negative weights | Convergence guarantee |
|---|---|---|---|
| `shortestPathBellmanFordDES` | every dirty node broadcasts a wave along each outgoing edge | allowed | ≤ \|V\|-1 iterations on graphs without negative cycles; iteration \|V\| catches negative cycles |
| `shortestPathDijkstraDES`    | a global priority queue dictates which station is "active" this tick; that node settles and emits waves to all outgoing edges | rejected (throws) | each node settled at most once |

### Numerical agreement

On the 5-node chain `0(s) → 1(a) → 2(b) → 3(c) → 4(t)`:

```
Bellman-Ford-DES distances = [0, 1, 3, 5, 6]   (ours)
Dijkstra-DES   distances   = [0, 1, 3, 5, 6]   (ours)
networkx Bellman-Ford      = [0, 1, 3, 5, 6]   (Python ref)
networkx Dijkstra          = [0, 1, 3, 5, 6]
```

On 5 random non-negative graphs of 12 nodes each: BF and Dijkstra agree to **0.00e+0**.

### Wave-count theorem in operational form

Validation Study 6 confirms the textbook claim that Dijkstra processes each edge at most once. Both algorithms count `wavesEmitted` directly on the result object, so the cost difference is observed numerically:

```
seed=1:  BF waves = 154, Dijkstra waves = 104
seed=7:  BF waves = 126, Dijkstra waves =  91
seed=42: BF waves = 118, Dijkstra waves =  97
```

### Animation

`ANIMATE=1 npm run shortest-path` writes an HTML player that shows the graph nodes coloured by their current distance estimate (cool-blue = far, hot-yellow = close), edges flashing yellow when a wave fires this tick, and a sidebar with iteration count, source, waves-this-tick, and improved-this-tick.

### Run it

```bash
npm run shortest-path                                       # 5-node chain demo
N_NODES=12 ALGO=both ANIMATE=1 npm run shortest-path        # both algorithms, HTML out
ALGO=dijkstra N_NODES=20 npm run shortest-path
SOURCE=3 npm run shortest-path                              # different source
npm run validate-shortest-path                              # 23 / 23
npm run test-shortest-path                                  # 43 / 43
```

## Incremental / warm-startable LP solver as DES (live add/remove of constraints, variables, objective)

The previous "simplex-in-DES" module solves an LP **once**: load the problem, pivot to optimality, return the answer. Real workloads aren't like that — labour rules change, products are added, prices shift, last quarter's binding constraint becomes irrelevant this quarter. `general/incremental-lp.ts`, `main-incremental-lp.ts`, `animation/scenes/incremental-lp-scene.ts`, plus the validation and test runners, build a **fully online LP solver** in the DES architecture: every modification is a movable arriving at the LP tableau station, every pivot is a tick.

### The DES mapping

| Station (stationary entity)           | Role |
|---------------------------------------|---|
| `LPTableauStation`                     | Owns the dense tableau and basis. Persists across modifications. |
| `EventQueueStation`                    | Holds pending modification events keyed by tick. Drains into `LPTableauStation`. |
| `PivotStation`                         | One pivot per tick; primal or dual depending on which invariant is broken. |
| `CensusStation`                        | Snapshots state for animation. |

Movables: `ConstraintAddEvent`, `ConstraintRemoveEvent`, `ObjectiveChangeEvent`, `VariableAddEvent`, `VariableRemoveEvent`, `PivotEvent`. The solver supports five live modifications:

| Modification | Resulting state | Recovery |
|---|---|---|
| Add `a · x ≤ b` | Possibly primal-infeasible (rhs goes negative) but stays dual-feasible | **Dual simplex** restores feasibility |
| Remove constraint | Both invariants preserved (just drops a row + slack) | No work needed |
| Change `c` | Primal-feasible but possibly dual-infeasible | **Primal simplex** restores optimality |
| Add variable (with column + objective coef) | Primal-feasible; new column may have favourable reduced cost | **Primal simplex** brings it in if profitable |
| Remove variable | Both invariants preserved if non-basic; force-pivot then drop if basic | One pivot, then no work |

### Two implementation gotchas (caught by the validator)

Both bugs hide easily because the "naïve" implementations are *internally* consistent — they just no longer represent the intended LP after a modification.

1. **`applyAddVariable` must transform the new column.** When the user appends `a_new` (the column in original standard form), the tableau has already been pivoted: rows are `[B^{-1}A | B^{-1}I | B^{-1}b]`, not `[A | I | b]`. The newly-inserted column must be `B^{-1} a_new`. Trick: the slack columns `tab[:, numStruct..numStruct+m-1]` jointly *are* `B^{-1}` (they started as `I` and have been multiplied by every pivot since), so `B^{-1} a_new = Σ_k a_new[k] · tab[:, numStruct+k]`. The same trick gives the row-0 reduced-cost entry. Without this fix, *Study 5* in the validator fails by ~50 % of the objective.
2. **`applyRemoveConstraint` must drop the row where the slack is currently basic, not row `index+1`.** After several pivots the slack of constraint `i` may live in any row of the basis. Fix: locate `r* = row where slack_i is basic` (or pivot it back in if non-basic), drop row `r*` and column `slack_i`. Without this fix, *Study 7.c* drops the wrong row and the LP loses its meaning.

### Validation — agrees with the static `solveLPInternal` to ~1e-13

```
Study 1 — Baseline 2D LP                                  PASS  2/2
Study 2 — Add constraint (dual restart)                   PASS  2/2
Study 3 — Remove a binding constraint                     PASS  2/2
Study 4 — Change objective (primal restart)               PASS  2/2
Study 5 — Add a variable mid-run                          PASS  2/2
Study 6 — Remove a variable mid-run                       PASS  2/2
Study 7 — Sequence of all 5 modifications                 PASS  11/11
Study 8 — Randomised 3-variable modification stream       PASS  17/17   max|Δ|=2.13e-14
Study 9 — min-LP sense flip                               PASS  2/2

42 / 42 checks pass.
40 / 40 unit tests pass.
```

### A scenario worth running

The CLI runs a 44-tick story exercising **all 5 modification types and recovery from unboundedness**:

```
tick  0  z=  0.000  x=[0, 0]              [initial state]
tick  1  z=150.000  x=[0, 30]             primal:  gadget enters, material_slack leaves
tick  2  z=206.000  x=[42, 16]            primal:  widget enters, labor_slack leaves
tick  4  z=190.000  x=[30, 20]            EVENT: add x_widget ≤ 30
                                          dual:    labor_slack enters, cap_widget_slack leaves
tick  8  z=210.000  x=[30, 20]            EVENT: change c → (5, 3)         (no-op, vertex unchanged)
tick 12  z=210.000  x=[30, 20]            EVENT: remove labor constraint    (no work needed)
tick 16  z=270.000  x=[0, 20, 30]         EVENT: add new product thingamajig
                                          primal:  thingamajig enters, widget leaves
tick 22  z=210.000  x=[0,  0, 30]         EVENT: remove material constraint  → status = UNBOUNDED
tick 26  z=270.000  x=[0, 20, 30]         EVENT: add budget w+g+t ≤ 50      → re-bounds the LP
                                          primal:  gadget enters, budget_slack leaves
tick 32  z=260.000  x=[0, 20, 30]         EVENT: change c → (1, 1, 8)
tick 36  z=240.000  x=[0,    30]          EVENT: remove gadget (line discontinued)
                                          dual:    budget_slack enters, cap_widget_slack leaves
tick 44  z=240.000  x=[0,    30]          OPTIMAL
```

Run with the animation flag and the polytope physically reshapes when constraints come and go, the gradient arrow rotates when `c` changes, and the optimum dot slides between vertices as primal/dual pivots fire:

```bash
npm run incremental-lp                    # console trace
ANIMATE=1 npm run incremental-lp          # writes out/incremental-lp.html
npm run validate-incremental-lp           # 42 / 42
npm run test-incremental-lp               # 40 / 40
```

### Why this matters for the framework

This module shows that **the DES engine is a viable substrate for online optimisation** — the kind of warm-start re-solve that drives MILP branch-and-bound, dynamic LP-based MPC, and any setting where the LP changes faster than you can re-solve from scratch. Every modification is a movable; every pivot is a tick; the tableau is a long-lived station that swallows events and emits pivots. The same pattern would extend to incremental LU factorisation, online QP, and the dual-feasible warm-start tricks that make commercial solvers fast.

## Stochastic LP: SAA monolithic vs Benders-as-DES, with statistical convergence

The natural follow-up to the warm-startable LP: **what if the LP coefficients are themselves random variables?** That's stochastic linear programming (SLP), and it's where the marriage of DES (scenario sampling), the incremental LP solver (warm-startable master), and MDP-style cuts (piecewise-linear value-function approximation) produces dramatic speedups.

See `general/stochastic-lp.ts`, `main-stochastic-lp.ts`, the JSON adapter
(`model: "stochastic-lp"`), plus the validation runner and unit tests.

### The two-stage SLP problem

```
maximize  c · x  +  E_ω[ Q(x, ω) ]
   s.t.   A x ≤ b,  x ≥ 0
```

where the recourse function Q is itself an LP:

```
Q(x, ω) = max  q · y
          s.t. T(ω) x + W y ≤ h(ω),   y ≥ 0
```

The expectation is over a known distribution of ω (sampled by DES). This subsumes inventory under uncertainty, capacity planning, network design under random demand, robust regression, and many production-planning problems.

### Three solution methods, compared

```
                          Method                      Sample size    Wall-clock
─────────────────────────────────────────────────────────────────────────────────
1. Closed-form newsvendor critical fractile            (no LP)          0 ms
2. Sample Average Approximation (SAA), monolithic LP    N=500          18,948 ms
3. Benders / L-shaped decomposition AS A DES            N=500             10 ms
                                                                  speedup ≈ 1894×
```

All three converge to the same z* on the same scenario set:

```
mono ≡ Benders  agrees to 4.6e-12 on every objective (60/60 checks)
mono ≡ Benders  bit-equivalent for budget-constrained scenarios where no closed form exists
```

### The DES mapping for Benders / L-shaped

```
SamplerStation (DES)  ─►  ScenarioStation_1  ─┐
                          ScenarioStation_2  ─┼─►  AggregatorStation  ─►  MasterStation
                          ...                  │     averages duals        (IncrementalLP)
                          ScenarioStation_N  ─┘     into a single cut          │
                                                                                │
                                                ◄────  applyAddConstraint  ◄────┘
```

- **Every Benders iteration is a tick.** The master proposes (x*, θ*); subproblems are solved at x*; their duals are aggregated; one cut is added to the master.
- **Each cut is a movable.** It arrives at the master via `applyAddConstraint`, which the warm-started dual simplex repairs in O(few) pivots from the previous optimal basis.
- **Subproblem duals are read from the optimal `IncrementalLP` tableau.** Textbook fact: in `max q·y s.t. A y ≤ b, y ≥ 0`, the dual of constraint i equals the reduced cost of slack i at the optimum — i.e., `tab[0][numStruct + i]` after `solveToOptimum()`.

### Statistical convergence — the textbook 1/√N rate

Replicated SAA over 20 seeds at each sample size, comparing to closed-form `z*_true = 1797.857`:

```
N          mean SAA z*       stderr    bias     out-of-sample gap
   10      1854.55 ± 23.79   23.79     +56.69    36.77 ± 5.22
  100      1799.43 ±  9.41    9.41      +1.57     3.21 ± 1.72
1,000      1797.44 ±  2.43    2.43      −0.42    −0.20 ± 1.33
10,000     1797.86 ±  0.90    0.90      +0.003   −0.54 ± 1.33
```

The standard error from N=100 to N=10000 shrinks by **10.48×**, almost exactly the theoretical √100 = 10×. The "out-of-sample gap" is `z*_true − z(SAA-x*)` measured on a separate 50,000-scenario evaluation — it goes from 36.77 (N=10) to within Monte-Carlo noise (N=10000). The SAA *decision* becomes statistically optimal as N grows.

### Why pivoting is the entire ballgame for SLP

The monolithic SAA LP has `1 + N · n_second` variables and `1 + N · m_second` constraints. For N=500 that's 1001 variables × 2001 constraints — solved by dense simplex with 1587 pivots in 18.9 s. **Benders solves the same problem in 10 ms with 28 master iterations.**

Why? Because Benders never instantiates the per-scenario y-variables. It only ever sees the *dual* contribution `(π_s · h_s, π_s · T_s)` from each subproblem, aggregated into a single cut on the master. The master grows by one constraint per iteration; the warm-started dual simplex repairs the master in 1-2 pivots after each cut.

This is the canonical illustration that **pivoting + warm-starting is the engineering substrate that makes large-scale stochastic optimisation feasible**. It generalises to:

| Method | What gets added to the master | Master operation |
|---|---|---|
| Benders / L-shaped | Optimality cut from a scenario | `applyAddConstraint` |
| Dantzig-Wolfe / column generation | New variable from a pricing subproblem | `applyAddVariable` |
| Branch-and-bound | Branching constraint at a B&B node | `applyAddConstraint` |
| SDDP (multi-stage Benders) | Cut on the value function at the next stage | `applyAddConstraint` |
| Chance-constrained scenario approx | Violated chance constraint | `applyAddConstraint` |

All of these are direct applications of the incremental LP solver from the previous section.

### MDP vs DES, complementarity

For SLP the two paradigms work hand in hand:

- **DES** — generates the scenarios (Monte-Carlo simulation of ω). Identical to the role our DES has played for SEIR, FactMachine, dispatch, soccer rotation, etc.
- **LP (with pivoting)** — solves the inner optimisation, with the incremental LP being the right substrate for the master. Cuts are added one at a time and dual-simplex-warm-started.
- **MDP framing** — organises the multi-stage extension. **Benders ≡ value iteration with a piecewise-linear value function**, computed via LP duality. SDDP is multi-stage Benders applied recursively across stages, each stage's LP repeatedly resolved with a growing pool of cuts.

```bash
npm run stochastic-lp                      # default scenario, 200 samples
N=500 BUDGET=80 SEED=7 npm run stochastic-lp
VERBOSE=1 npm run stochastic-lp            # prints each Benders cut
node dist/des/main-from-json.js examples/stochastic-lp.json
npm run validate-stochastic-lp             # 60/60, ~1 minute (statistical study)
npm run test-stochastic-lp                 # 44/44 unit tests
```

## Statistical stochastic optimisation: SDDP, risk, fitting, DRO, adaptive simopt

The stochastic optimisation layer now has JSON-runnable models for the
natural follow-ons to two-stage Benders:

| Model id | What it adds | DES/base-class mapping |
|---|---|---|
| `stochastic-lp` | Existing two-stage SAA vs Benders, now first-class JSON | `FixedPointIterationStation` Benders master + `IncrementalLP` cuts |
| `sddp-capacity` | Multi-stage stochastic capacity expansion via SDDP-style value-function cuts | `CapacityExpansionSDDPStation extends FixedPointIterationStation` |
| `risk-capacity` | Expected-profit, CVaR, chance-constrained, and DRO-lite scenario optimisation | grid-evaluation station with validators and scenario traces |
| `distribution-fit` | MLE vs method-of-moments fitting for demand/service samples | `DistributionFitStation extends FixedPointIterationStation` |
| `adaptive-simopt` | Sequential simulation optimisation across candidate policies | UCB allocation in `AdaptiveSimulationOptimizerStation` |

Every model attaches intrinsic validators, supports the shared JSON spec
envelope, and writes animation by default when run through `main-from-json`
unless `runtime.animate` is set to `false`. Models that take
`runtime.outputs.log` also use the JSONL observability logger. The
scenario-demand schema accepts uniform ranges, fitted distributions
from `distribution-fit`, or explicit empirical point masses.

```bash
npm run from-json -- examples/stochastic-lp-capacity.json
npm run from-json -- examples/sddp-capacity.json
npm run from-json -- examples/risk-capacity-cvar.json
npm run from-json -- examples/distribution-fit-service-times.json
npm run from-json -- examples/adaptive-simopt-capacity.json
npm run test-statistical-optimization       # 42/42 checks
```

## Network flow and traffic flow

The network-flow layer adds three JSON-runnable models:

| Model id | What it adds | DES/base-class mapping |
|---|---|---|
| `max-flow` | Maximum s-t flow with min-cut validation | `MaxFlowOptimizationStation extends DESStation`; one augmenting path per tick |
| `traffic-flow` | Continuous-time traffic on a stationary five-intersection grid with fewer than 300 active cars | `TrafficGridStation extends DESStation`; lanes/intersections/sources/sinks and sparse one-foot grid cells are stationary entities, and cars are moving `CarToken`s with position, velocity, acceleration, jerk, and reaction-time state |
| `smart-traffic-flow` | Same traffic environment, but active cars own `runTimeStep()` | `SmartTrafficWorldStation extends DESStation`; each `SmartTrafficCar extends SmartMovable` and is passed to `runIterativeDES`, so car actors are Fisher-Yates shuffled with the runner and the world commits their proposals at the tick barrier |

Traffic interaction is local and grid-mediated: each small tick applies
source spawning, signal phases, lane-capacity checks, reaction-delayed
leader perception, same-lane headway, jerk-limited acceleration, and
downstream entry constraints. Cars carry their occupied grid cells; the
grid materializes sparse `TrafficCellStation`s for roughly one-foot
spatial bins so nearby-car lookup stays local without ticking every square
foot as an independent station. The `smart-traffic-flow` variant keeps that
same grid/referee role, but moves the acceleration/position proposal into the
car actor itself; the trace records `scheduledSmartCars`, `smartMovableRuns`,
and a sampled actor run order. Smart cars also carry sampled driver traits:
`distancePreference` changes cruising headway, while `startPreference` changes
how much clearance and launch delay a stopped car wants before accelerating.
Those traits are sampled from a convolved discrete PMF from the random-variable
toolkit rather than a flat uniform draw, so most drivers cluster near ordinary
behavior while still allowing tails.
The default smart demo uses `dtSec = 0.1`, so the generated animation steps one
tenth of a simulated second at a time.
Optional behavior-risk accidents let an at-risk car enter a short bad-driving
mode (`speeding`, `brake-too-slow`, or `accelerate-too-fast`); the world only
emits an accident when the follower's body actually touches the leader's rear
bumper at commit time, then records it on the striking car, struck car, and
impacted grid-cell station so validators and animations can review the incident.
External traffic cross-checking is available through a sanctioned optional
SUMO adapter. The validator exports a no-accident smart-traffic baseline into
SUMO XML, calls host-provided `sumo`/`netconvert` binaries only when available,
and compares robust aggregate metrics such as departures, completion rate,
mean travel time, and collision count. If SUMO is absent, the validator reports
that dependency as unavailable without failing the DES run.

```bash
npm run from-json -- examples/max-flow-six-node.json
npm run from-json -- examples/traffic-flow-five-intersection.json
npm run from-json -- examples/smart-traffic-flow-five-intersection.json
npm run validate-smart-traffic-external
npm run test-network-flow                  # 87/87 checks
```

## Internal solver networks

The `internal-solver-network` model exposes optimization/search solvers as
stationary DES entities that emit moving incumbent-solution tokens. It does not
call external solvers; GA, simulated annealing, dynamic-programming knapsack,
shortest path, and exact small-TSP search all run inside the station network.

| Solver kind | What runs per tick | Main stationary entities |
|---|---|---|
| `shortest-path` | One Dijkstra settle or Bellman-Ford wave relaxation | solver, wall-clock checker, solution sink |
| `knapsack-dp` | One item row of exact 0/1 knapsack DP | DP solver, wall-clock checker, solution sink |
| `knapsack-sa` | One simulated-annealing proposal/acceptance step | SA solver, wall-clock checker, solution sink |
| `tsp-sa` | One TSP simulated-annealing proposal/acceptance step | TSP SA solver, wall-clock checker, solution sink |
| `tsp-ga` | One GA generation | TSP GA solver, wall-clock checker, solution sink |
| `tsp-held-karp` | Exact Held-Karp DP for small TSP instances | exact solver, wall-clock checker, solution sink |

Runs default to a 180000 ms wall-clock budget. The checker station emits a
`StopSignalToken` the first time it is ticked after the budget has passed; the
runner then halts through `stopWhen` on the next loop check. Animation is
enabled by default for JSON runs, and selected logs write JSONL start/trace/end
events for debugging.

```bash
npm run from-json -- examples/internal-solver-knapsack-dp.json
npm run from-json -- examples/internal-solver-shortest-path.json
npm run from-json -- examples/internal-solver-tsp-ga.json
npm run test-internal-solver-network       # 21/21 solver-network checks
```

## Feasibility checker pipeline

The `feasibility-pipeline` model is a JSON-runnable program for checking a
user's proposed optimization solution and, optionally, trying to improve it
inside the engine. It currently supports structured linear objectives,
linear constraints, finite/infinite bounds, and continuous/integer/binary
variables.

Pipeline stations:

| Station | Role | Moving token |
|---|---|---|
| `candidate-source` | emits the user's incumbent solution | `CandidateToken` |
| `domain-checker` | checks missing values, bounds, integrality, binary domains | `DomainCheckedToken` |
| `constraint-checker` | computes activities and violations for each linear constraint | `ConstraintCheckedToken` |
| `objective-evaluator` | computes objective, total violation, feasibility, and merit | `FeasibilityEvaluationToken` |
| `improvement-station` | proposes repaired/neighbor candidates and loops them back through the checkers | `CandidateToken` |
| `wall-clock-checker` | emits a stop signal at the first checker tick after the budget | `StopSignalToken` |
| `feasibility-sink` | records the initial, trace, best incumbent, violations, and stop signals | evaluations + stops |

The JSON output reports whether the supplied candidate is feasible, lists
violations, records the best feasible or lowest-violation incumbent, and writes
CSV/HTML/frames/log outputs through the same registry runtime.

```bash
npm run from-json -- examples/feasibility-pipeline-knapsack.json
npm run from-json -- examples/feasibility-pipeline-production.json
npm run test-feasibility-pipeline          # 16/16 feasibility and improvement checks
```

## Math blocks for ODEs and PDEs

The math-block layer adds stationary block/station primitives for calculus-style
models without changing the DES base classes. Blocks extend `DESStation`; scalar
`MathSignal` tokens move through named channels.

| Block family | Examples |
|---|---|
| Sources/sinks | constant, function, expression source, sink recorder |
| Algebra | sum, subtraction, product, gain, saturation |
| Calculus | integrator, differentiator, first-order filter |
| Logic | comparator, and/or/not/xor |
| PDE coupling | 1D Laplacian block feeding cell integrators |

Three JSON-runnable models are registered:

| Model id | What it adds |
|---|---|
| `math-ode-blocks` | User supplies state variables, initial values, and RHS expressions; the adapter assembles integrator blocks and expression RHS blocks into a feedback diagram. |
| `math-heat1d-blocks` | 1D heat equation as stationary cell/boundary blocks coupled by Laplacian blocks, with CFL validation and maximum-principle checks. |
| `math-equation` | User supplies an equation as constrained LaTeX, XML, or structured JSON; the normalizer creates the stationary nodes, moving `MathSignal` edges, initial-condition sources, and solver inputs. |

Animation is enabled by default through the normal `runtime.animate` path, and
observability logs show block-run start/finish plus per-tick debug rows when a
log output is selected.

```bash
npm run from-json -- examples/math-ode-exponential-decay.json
npm run from-json -- examples/math-heat1d-blocks.json
npm run from-json -- examples/math-equation-latex-ode.json
npm run from-json -- examples/math-equation-json-ode.json
npm run from-json -- examples/math-equation-xml-heat1d.json
npm run test-math-blocks                  # 30/30 stationary block, ODE, PDE, equation-ingestion checks
```

## Universal DES model JSON

For interchange and future model families, the portable input shape is
`des/universal-model/v1`. It is deliberately richer than the runnable
`des/model-spec/v1` envelope: it records the original user input, normalized
math, generated stationary entities, moving entity token types, graph edges,
sources/sinks, solver intent, runtime outputs, and metadata.

Top-level sections:

| Section | Purpose |
|---|---|
| `originalInput` | Raw LaTeX/XML/JSON/text/manual source, content type, URI, and provenance metadata. |
| `math` | Normalized variables, parameters, equations, initial/boundary conditions, constraints/objectives, domains, and numerical grid/time settings. |
| `des` | Stationary entities, moving entity token kinds, directed graph edges, source/sink declarations, and observability intent. |
| `solver` | The registered solver target and method/options. For the current math path this is `math-equation`. |
| `runtime` | Same output controls used by the existing registry runner. |

`main-from-json` now accepts this universal schema directly and converts it to
the appropriate registered solver when supported:

```bash
npm run from-json -- examples/universal-math-equation-latex-ode.json
npm run test-universal-model-spec          # 11/11 universal-schema and conversion checks
```

## Multi-stage stochastic programming: SDDP as recursive DES cuts

The multi-stage extension is now implemented in
`general/multistage-stochastic.ts` and registered as JSON model
`multistage-sddp`. The demo is a four-stage inventory/storage problem:
each stage observes random demand, solves a small LP for ordering,
sales, stockout, and ending inventory, then passes the ending inventory
as the next stage's state.

The DES mapping is deliberately parallel to Benders:

| SDDP object | DES interpretation |
|---|---|
| Forward pass | sampled scenario path through stage stations |
| Backward pass | one cut-generation tick per visited stage state |
| Stage value function | `AffineCutPool` station state |
| Future value approximation | theta variable constrained by next-stage cuts |
| Exact benchmark | deterministic-equivalent scenario-tree LP |

The base utility `des-base/cut-pool.ts` provides validated affine cut
pools with upper/lower envelope evaluation, so future Benders, DRO,
outer-approximation, and chance-constraint models can reuse the same
cut machinery.

Current default run:

```
Exact scenario tree: z = 112.176312, 30 nodes
SDDP: optimal in 21 iterations
Policy value: 112.176313
Cuts/stage: [22, 22, 22, 22, 1]
```

```bash
node dist/des/main-from-json.js examples/multistage-sddp.json
npm run test-multistage-stochastic          # 20/20 unit tests
```

## Network flow and traffic flow as DES

`general/max-flow.ts` adds a compact network optimiser: a
`MaxFlowStation` extends `FixedPointIterationStation`, and each DES tick
performs one Edmonds-Karp augmenting-path update. The station carries its
residual graph as state and validates both flow conservation and the
max-flow/min-cut certificate at termination.

`general/stochastic-flow-mdp.ts` is the MDP interpretation of max-flow.
When edge availability or capacity is stochastic, the problem is no
longer "find one static feasible flow"; it is "choose the next routing
action from the current network state." The exact finite-horizon state is
`(current node, remaining edge capacities)`, actions are outgoing edge
attempts or wait, and the Bellman recursion is:

```
V_t(n, c) = max_a E[ r(n, c, a, W_t) + V_{t+1}(n', c') ]
```

This covers the small version of dynamic flow, stochastic networks, and
queueing-control problems: the network is still a flow graph, but the
answer is now a policy rather than a cut certificate. The default demo's
static max-flow upper bound is 4 units; stochastic availability and
routing penalties reduce the optimal expected reward to about 2.91. If
all success probabilities are set to 1 and penalties to 0, the MDP value
recovers the static max-flow value exactly.

`general/traffic-flow.ts` models a small continuous-position traffic
system without changing the base classes, while the registered
`traffic-flow` JSON model in `general/network-flow.ts` adds the animated
micro-simulation path with reaction time, acceleration, jerk, and sparse
one-foot grid-cell stations. `smart-traffic-flow` adds a third form where
cars are `SmartMovable` participants with their own `runTimeStep()`:

| Entity | DES role |
|---|---|
| `TrafficGridStation` | stationary coordinator holding the road graph, sources, sinks, routing cache, and conservation validators |
| `IntersectionStation` | stationary signal controller; the center intersection alternates EW/NS phases |
| `RoadLinkStation` | stationary road segment; owns car positions, speeds, exit credits, capacity, and car-following gaps |
| `TrafficCar` | moving entity flowing source → link → intersection → link → sink |
| `SmartTrafficCar` | smart movable actor; computes its own per-tick motion proposal under runner shuffle |
| `SmartTrafficWorldStation` | stationary referee; owns grid occupancy, signals, source spawning, validation, and proposal commit |

The default traffic demo has five intersections, eight directed road
links, four source/sink OD streams, and a hard cap of 240 active cars
(below the requested 300). Cars interact through the link/grid state:
links sort cars by position, enforce a minimum gap, respect downstream
capacity, and ask the intersection signal before discharging a car.
The JSON traffic model also records each car's velocity, acceleration,
jerk, target acceleration, leader gap/id, and occupied grid cells in the
trace. In `smart-traffic-flow`, optional accident parameters
(`accidentRiskScale`, `accidentFaultDurationSec`, `accidentAccelBoostMps2`,
`accidentFlashSeconds`) and driver-heterogeneity parameters
(`distancePreferenceSpread`, `startPreferenceSpread`) make behavior-risk
rear-end contact events observable in the trace, log, summary, and HTML
animation flash layer.

`general/computer-network.ts` adds the packet-switched computer-network
focus area. Hosts, routers, switches, and directed links are stationary
entities; `NetworkPacket` is the movable entity. A JSON topology declares
nodes, links, and traffic flows, and the runner reports delivery ratio,
wire throughput, application goodput, time in system, drops, queueing
delay, link utilization, per-flow/link cost, time-series buildup, and a
bottleneck ranking.

| Entity | DES role |
|---|---|
| `NetworkHostStation` | stationary endpoint that emits or consumes packets |
| `NetworkRouterStation` / `NetworkSwitchStation` | stationary forwarding nodes with FIFO queues and routing rules |
| `NetworkLinkStation` | stationary directed link with bandwidth serialization, propagation latency, buffer limit, utilization, and cost |
| `NetworkPacket` | moving entity flowing through the declared topology |

Protocol profiles are deliberately lightweight: `raw`, `udp`, `tcp`, and
`http` add different header overheads and startup delays while preserving
the same stationary/movable structure. This lets a JSON topology compare
HTTP-style request traffic against TCP bulk flow and UDP telemetry without
changing code.

The bottleneck lab (`examples/computer-network-bottleneck.json`) sends
HTTP, UDP, and TCP flows through a 5 Mbps WAN link:

```
generated=3857, delivered=838, dropped=3019
offered_load=21.47 Mbps, wire_throughput=5.60 Mbps, goodput=3.67 Mbps
mean_time_in_system=245.1 ms, p95=275.0 ms
top_bottleneck=link:edge-wan (drops observed)
edge-wan utilization=0.985, avg_in_flight=87.8, mean_queue_delay=210.0 ms
```

For an independent second opinion, `external-references/computer-network/network_reference.py`
implements the same topology semantics in source-only Python with no third-party
packages. It is invoked through the sanctioned external module runner
(`spawnSync`, `shell: false`, source path constrained to `external-references/`):

```bash
npm run external-modules
npm run external-computer-network
npm run external-traffic-sumo -- --problem=out/external/traffic/smart-traffic-sumo-problem.json
npm run validate-computer-network
npm run validate-smart-traffic-external
```

Default traffic-flow run:

```
Traffic-flow DES
generated=240, completed=240, active_at_stop=0
max_active=49, blocked_source_tries=271
mean_travel=98.4 sec, p95=202.0 sec
throughput=1440 cars/hour, max-flow bound=32 cars/min
```

```bash
npm run max-flow
npm run stochastic-flow-mdp
npm run traffic-flow
npm run computer-network
npm run computer-network-baseline
npm run computer-network-bottleneck
npm run validate-computer-network
node dist/des/main-from-json.js examples/max-flow.json
node dist/des/main-from-json.js examples/stochastic-flow-mdp.json
node dist/des/main-from-json.js examples/traffic-flow.json
node dist/des/main-from-json.js examples/computer-network.json
node dist/des/main-from-json.js examples/computer-network-bottleneck.json
npm run test-network-flow                    # 83/83 unit tests
npm run test-computer-network                # 40/40 unit tests
```

## Indoor temperature control: bang-bang vs PID vs Fuzzy-PI vs MDP-MPC

A new model in the **control** half of the engine: maintain indoor temperature
within ±2°F of target while minimising heating energy. The outdoor temperature
follows a noisy diurnal pattern (cold morning, mild afternoon); the controller
sees both the current indoor temperature (with sensor noise) and a noisy
6-hour-ahead forecast of the outdoor temperature.

### Stations and movables

| Station | Role | Movable in → out |
|---|---|---|
| OutdoorSource    | emits the TRUE outdoor temperature this tick                 | tick → T_out_true |
| ForecastStation  | emits a NOISY forecast of T_out for the next H ticks         | tick → T_out_forecast[…] |
| ThermostatSensor | emits indoor temperature with optional sensor noise          | T_in → T_in_meas |
| Comparator       | emits error = T_target − T_in_meas                           | T_target, T_in_meas → error |
| ControllerStation | pluggable: bang-bang \| PID \| Fuzzy-PI \| MDP-MPC          | error, forecast → Q_command |
| HeaterActuator   | applies Q to the house, emits an energy event                | Q_command → energy_event |
| House (physics)  | integrates the 1st-order thermal ODE                          | T_out, Q → T_in_next |
| EnergyMeter      | accumulates Σ Q · Δt                                         | energy_event → total_kWh |
| ComfortMonitor   | tallies time outside ±band                                   | T_in → comfort_pct |

Same DES topology in every run; only the ControllerStation changes.

### Four interchangeable controllers

1. **Bang-bang** — heater FULL ON if T_in < target, OFF otherwise. Baseline. Simple, surprisingly hard to beat on the easy scenario.
2. **PID** — classical proportional-integral-derivative with a first-order-filtered derivative (necessary on the 1-min sampled signal — raw discrete `de/dt` has too much spike) and conditional-integration anti-windup.
3. **Fuzzy-PI** — Mamdani fuzzy controller with NL/NS/Z/PS/PL terms on (error, error-rate); 5×5 rule base produces a normalised Δ-Q which is integrated externally to give offset-free tracking like a PI controller but with smooth interpretable rule-based behaviour.
4. **MDP-MPC** — receding-horizon dynamic programming. Each tick the controller solves a finite-horizon MDP on a (T_in × time) grid using the forecast as the disturbance model, executes the first action, advances one tick, re-plans. The action grid has `nLevels` discrete heater powers; `trackWeight` controls how tightly the controller pulls toward T_target inside the band.

### Easy scenario (24h winter, ±2°F band)

| Controller | Energy (kWh) | Comfort (%) | Cost ($) | Indoor range (°F) |
|---|---:|---:|---:|---|
| bang-bang             | 89.42 | 100.0 | 13.41 | [69.5, 70.2] |
| PID (filtered-D)      | 87.89 | 100.0 | 13.18 | [68.1, 70.1] |
| Fuzzy-PI (Mamdani)    | 88.71 | 100.0 | 13.31 | [68.1, 70.8] |
| MDP-MPC (H=6h, w=0.05) | 89.15 | 100.0 | 13.37 | [69.4, 70.1] |

All four maintain comfort; PID happens to be cheapest because it lets T drift to the band's edge more freely than bang-bang.

### Hard scenario (24h, ±1°F band, T_out averaging 15°F with ±20°F swing)

| Controller | Energy (kWh) | Comfort (%) | Cost ($) |
|---|---:|---:|---:|
| bang-bang | 105.08 | 53.7 | 76.64 |
| PID       | 103.67 | 43.4 | 84.84 |
| Fuzzy-PI  | 103.33 | 42.2 | 105.48 |
| **MDP-MPC** | **106.42** | **55.6** | **73.52** |

Once the heater can no longer keep up, the MDP-MPC's forecast-aware planning wins on the controller's OWN cost metric (energy + comfort penalty). The reactive controllers all spend money on power and STILL have more comfort violations.

### Numerical detail worth flagging

The MDP-MPC has to handle the "fine-grained tick problem" carefully: at Δt = 1 minute the per-tick temperature change at any heater level is < 0.1°F, so a coarse value-table grid (say 0.5°F bins) puts every action in the same cell and the DP cannot distinguish them. The implementation linearly interpolates V[k+1] in the continuous next-state instead of snapping to bins.

```bash
npm run temp-control                                     # head-to-head + sensitivity sweep
npm run temp-control-anim -- --controller pid            # → out/temp-control/animation.html
npm run temp-control-anim -- --controller mdp-mpc        # MDP-MPC visualisation
npm run validate-temp-control                            # 21 studies (energy balance, in-band, monotonicity)
npm run test-temp-control                                # 36 unit tests
```

## JSON specification format + model registry: save and re-run any model

Two pieces define a runnable DES program:

1. **Topology** — which stations exist, how they connect, what movables flow. Code.
2. **Parameters** — numeric configuration (τ, K_p, T_target, durations, seeds). Data.

JSON is a great fit for (2) and a poor fit for (1). The JSON spec format takes that distinction seriously: each spec names a registered MODEL ID and supplies its parameters; the registry validates and runs it.

```jsonc
{
  "$schema": "des/model-spec/v1",
  "model": "temp-control",
  "description": "24-hour winter day, PID controller",
  "parameters": {
    "T_target": 70, "band": 2, "duration_h": 24, "dt_min": 1,
    "controller": {"kind": "pid", "Kp": 3, "Ki": 0.5, "Kd": 0.5}
  },
  "runtime": {
    "seed": 42,
    "outputs": { "csv": "out/foo.csv", "html": "out/foo.html" }
  }
}
```

```bash
node dist/des/main-from-json.js examples/temp-control-pid.json   # run it
node dist/des/main-from-json.js --list                            # registered models
node dist/des/main-from-json.js --schema temp-control             # full schema
node dist/des/main-from-json.js --example temp-control            # paste-ready spec
```

### Code or JSON, both work

The JSON envelope is a valid TypeScript object literal, so users who prefer code can write the same spec inline:

```typescript
import {runFromSpec} from './general/des-registry';

const spec = {
  $schema: 'des/model-spec/v1' as const,
  model: 'temp-control',
  parameters: {
    T_target: 70, duration_h: 24,
    controller: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5},
  },
  runtime: {seed: 42},
};
const result = await runFromSpec(spec);
```

For wholly new BEHAVIOUR users register a new model id pointing at a TypeScript adapter (which writes the topology and validation schema). For new VARIANTS of existing models they just write a new JSON.

| File | Purpose |
|---|---|
| `general/des-spec.ts` | Schema types, declarative parameter validator, envelope types. |
| `general/des-registry.ts` | The registry + `runFromSpec()` driver. |
| `general/adapters/temp-control-adapter.ts` | HVAC adapter. |
| `general/adapters/shortest-path-adapter.ts` | Graph SPP adapter. |
| `general/adapters/milp-bnb-adapter.ts` | MILP / branch-and-bound plus explicit IP/MIP solver graph adapters. |
| `general/adapters/simulated-annealing-adapter.ts` | SA on TSP/knapsack adapter. |
| `general/adapters/network-flow-adapter.ts` | Animated/logged max-flow and traffic-flow adapters. |
| `general/adapters/stochastic-flow-mdp-adapter.ts` | JSON adapter for stochastic-flow MDP. |
| `main-from-json.ts` | CLI driver. |
| `examples/*.json` | Concrete spec examples spanning multiple models. |

## MILP via branch-and-bound (composing the IncrementalLP)

`general/milp-bnb.ts` solves

```
max  c · x
s.t. A x ≤ b,  x_j ≥ 0,  x_j ∈ ℤ for j ∈ I
```

by depth-first branch-and-bound, with the LP relaxation at each node solved by our existing `IncrementalLP`. The single LP instance is mutated along the DFS path:

```
applyAddConstraint(x_j ≤ ⌊x_j*⌋)   ↘
  recurse left subtree              ↘ ← parametric simplex warm-start
applyRemoveConstraint(...)          ↗   makes each child LP cheap.
applyAddConstraint(-x_j ≤ -⌈x_j*⌉)  ↗
  recurse right subtree
applyRemoveConstraint(...)
```

DES form: each B&B **node is a station** (holds the IncrementalLP state and the branching-bound stack), each **movable** carries `(LP_z, fractional integer-var indices)` to a brancher that decides prune / commit / branch, and the **incumbent is a global station** that pruning consults.

Built-in problem builders: `buildKnapsackMILP(values, weights, capacity)` and `buildFacilityLocationMILP(p)`.

| Demo (n) | Algorithm | Wall time | Nodes / pivots |
|---|---|---|---|
| 4-item knapsack | B&B + LP relax | < 1 ms | 5 / 10 |
| 12-item knapsack | B&B + LP relax | 4 ms | 61 / 345 |
| 24-item random knapsack | B&B + LP relax | < 1 ms | ≤ 1000 (≪ 2^24 = 16.7M) |
| Pure 2-D LP (no integers) | B&B reduces to root | < 1 ms | 1 / 2 |

Validation: 22 checks pass against brute-force enumeration on knapsacks and against the LP solver when integrality is dropped. Unit tests: 38 checks.

```bash
npm run milp-bnb
npm run validate-milp-bnb
npm run test-milp-bnb
node dist/des/main-from-json.js examples/milp-knapsack.json
```

## IP/MIP solver graph: relaxation, heuristics, cuts, branching

`general/ip-mip-des.ts` is the stronger station-graph interpretation of
integer programming. It accepts the same canonical form:

```
max/min c^T x
s.t.    A x <= b
        x_j >= 0
        x_i integer for i in I
```

but it exposes the solver as cooperating DES stations rather than one
monolithic routine:

| Station | Role | Movables |
|---|---|---|
| `SearchControllerStation` | owns the branch/cut frontier | `NodeToken` |
| `LPRelaxationStation` | stationary LP solver block; backend is selectable | `RelaxationToken` |
| `RoundingRepairStation` | rounds fractional movables, repairs violations, local-search improves | `CandidateToken` |
| `IncumbentStation` | stores the best feasible integer solution anchor | candidate updates |
| `CutGeneratorStation` | adds valid inequalities, currently binary cover cuts | `CutToken` |
| `NodeDecisionStation` | prunes, strengthens, or branches | child nodes / completion |

The LP relaxation backend is selectable per run:

```
auto                       # inspect problem/node shape and choose a concrete backend
incremental-primal-dual   # warm-startable IncrementalLP, primal/dual simplex
des-simplex-dantzig       # LP simplex itself expressed as DES pivots
des-simplex-bland         # anti-cycling DES simplex
internal-simplex          # in-process two-phase simplex
external-highs            # scipy/HiGHS bridge
external-highs-ds         # scipy/HiGHS dual simplex
external-highs-ipm        # scipy/HiGHS interior-point
```

With `lpAlgorithm: "auto"` the branch-and-cut graph builds a
`techniquePlan` from the uploaded model: variable/row counts, density,
binary/integer mix, finite bounds, and connected components of the
constraint-variable graph. Small and branch/cut nodes stay in the
in-engine incremental LP path; large root relaxations are routed toward
HiGHS (`highs`, `highs-ds`, or `highs-ipm`) when available, with automatic
fallback to the in-engine incremental solver if the external bridge is
missing. Inside `IncrementalLP`, each pivot tick still chooses primal vs
dual simplex from the current feasibility flags. The plan also flags
separable decomposition candidates; generic automatic decomposition is
detected and reported here, while the implemented decomposition solvers
remain the Benders/SDDP modules for stochastic/block-structured models.

External solver cross-checks use the sanctioned module runner in
`src/des/runners/external-program.ts`: source wrappers must live under
`external-references/`, interpreter paths come from env vars or `PATH`,
and every invocation uses `spawnSync(..., {shell: false})`. The
registered `ip-mip-reference` module points at the source-only Python
wrapper `external-references/ip-mip/ip_mip_reference.py`, which can exact
enumerate small bounded integer models and can call `scipy.optimize.milp`
when SciPy is installed. No solver binary or executable is vendored.
The generic entry point is `npm run external-module -- <module-id>
[--key=value ...]`; validators use the same registry through
`runExternalModule(<id>, params)`.

This is the path for uploaded problem sets: parse the file into the JSON
model schema, let `auto` choose the relaxation backend or override it,
then the adapter builds the station graph and returns incumbent, bound,
gap, trace, topology, `techniquePlan`, and `lpAlgorithmUsage`.

Default knapsack demo:

```
status=optimal, z*=90, x=[0, 1, 0, 1]
LP backend=auto, usage={incremental-primal-dual: 4}
nodes=4, LP iterations=17, cuts=3, candidates=5
```

```bash
npm run ip-mip-des
LP_ALGO=des-simplex-dantzig npm run ip-mip-des
node dist/des/main-from-json.js examples/ip-mip-des-knapsack.json
npm run test-ip-mip-des                    # 39/39 unit tests
npm run validate-ip-mip-external           # exact external cross-check
```

## Simulated annealing (single-walker metaheuristic)

`general/simulated-annealing.ts` implements the classical Metropolis-cooling local search for any combinatorial problem. Each tick is **one proposal + accept/reject decision**. Stations:

```
CandidateGenerator → CostEvaluator → MetropolisAccept → BestTracker
                            ↑
                  TemperatureScheduler (global)
```

Cooling schedules supported:

| Schedule | Formula | When to use |
|---|---|---|
| Geometric | `T_k = T_0 · α^k` | Practical default (α ≈ 0.99) |
| Logarithmic | `T_k = T_0 / log(2 + k)` | Theoretical guarantee (Hajek 1988) |
| Linear | `T_k = max(0, T_0 − rate·k)` | Simple, predictable |
| Exp-restart | cycles geometric over a period | Basin-hopping |

The `SAProblem<S>` interface is generic. Built-in adapters: `buildTSPSAProblem` (2-opt + or-opt mixed neighbourhood) and `buildKnapsackSAProblem` (bit-flip + capacity penalty). Anyone can plug in a new problem by writing three functions: `cost`, `neighbour`, `initial`.

| Problem (n) | Algorithm | Wall time | Best length / value | Ratio to exact |
|---|---|---|---|---|
| Pentagon TSP (5) | SA, geometric α=0.998 | 1 ms | 293.8926 | **1.0000** |
| Random TSP (12) | Held-Karp / SA / GA | 3 / 7 / 13 ms | 251.26 (all three) | **1.0000** |
| Random TSP (30) | SA / GA, equal compute | 30 / 101 ms | 421.1653 (tie) | — |
| Knapsack (15) | SA / MILP-B&B | 1 / 3 ms | 267.0 / 267.0 | **1.0000** |

Validation: 31 checks (pentagon optimum at 5 seeds, Held-Karp matches at 6 small TSPs, knapsack matches MILP-B&B on 5 random instances, monotonicity / reproducibility / acceptance-rate / stall-limit). Unit tests: 31 checks.

```bash
npm run simulated-annealing
npm run validate-simulated-annealing
npm run test-simulated-annealing
node dist/des/main-from-json.js examples/sa-tsp-pentagon.json
node dist/des/main-from-json.js examples/sa-tsp-random30.json
node dist/des/main-from-json.js examples/sa-knapsack.json
```

## Optimization-as-DES: a base-class hierarchy for iterative algorithms

Five distinct optimization algorithms — Simulated Annealing, Hill Climber, Genetic Algorithm, Q-learning, and PPO — are reimplemented as concrete leaf classes of FOUR algorithm-family base classes, all running on the same `runIterativeDES` engine. The base classes capture distinct PROBLEM SHAPES and use the template-method pattern to enforce structure: `runTimeStep` is the orchestrator (final), abstract hooks are the differentiators (subclasses must override).

```
DESStation                              named typed channels (inboxes by name, outs by name)
├── SingleStateOptimizer<S>             single walker (SA, hill climb, tabu, Newton)
│       hooks: initialState, cost, propose, accept, clone, shouldStop
├── PopulationOptimizer<I>              population-based (GA, PSO, DE, ACO)
│       hooks: initialPopulation, evaluate, select, recombine, mutate, clone, shouldStop
├── RLAgentStation<S, A>                online TD (Q-learning, SARSA, expected SARSA)
│       hooks: pickAction, update, endOfEpisode
├── PolicyGradientAgent<S, A>           rollout-buffered PG (REINFORCE, A2C, PPO, TRPO)
│       hooks: samplePolicyAndValue   (+ counterpart PolicyUpdateStation: runUpdate)
└── EnvironmentStation<S, A>            generic env wrapper (state/transition/action channels)
```

The runner shuffles execution order each tick (Fisher-Yates, injectable RNG → reproducible seeds), and stops on quiescence, `maxTicks`, or a custom `stopWhen` predicate.

### What gets reused vs. specialized

The single biggest payoff is that THE ITERATION LOOP IS FACTORED OUT once. Each leaf only contains the algorithm-specific moves:

| Leaf class                   | Lines  | Hooks implemented                                                           |
| ---------------------------- | ------ | --------------------------------------------------------------------------- |
| `TSPSAOptimizer`             | ~150   | Metropolis `accept`, 2-opt/or-opt mix `propose`, geometric/log/linear cool  |
| `TSPHillClimber`             | ~10    | overrides only `accept` to be strict-improvement (everything else inherited)|
| `TSPGAOptimizer`             | ~150   | tournament `select`, OX `recombine`, inversion+swap `mutate`, NN init       |
| `QLearningAgent`             | ~80    | ε-greedy `pickAction`, off-policy TD `update`, ε-decay `endOfEpisode`       |
| `TabularPPOAgent`            | ~50    | softmax `samplePolicyAndValue`                                              |
| `PPOClipUpdateStation`       | ~100   | GAE advantages + K-epoch clipped surrogate + value-MSE `runUpdate`          |

The PG/PPO pause-resume protocol (handshake between agent and update station: stash pending state → emit `TrainTriggerToken` → update mutates `θ` and `V` → emit `ResumeToken` → resume buffer) is encoded once in the base class, so any future PG variant inherits it for free.

### Ground-truth validation

| Study                    | Reference          | Result |
| ------------------------ | ------------------ | ------ |
| Pentagon TSP n=5         | exact perimeter    | SA, HC, GA all match exactly across 3-5 seeds |
| Random TSP n=10          | Held-Karp          | SA & GA both 0.00% gap |
| 4×4 GridWorld            | value iteration    | Q-learning V(0) = V*(0) = 3.213, 100% greedy success |
| Corridor(8)              | value iteration    | PPO V(0) ≈ 2.052 vs optimal 2.053, 100% greedy success |

Plus structural invariants — best-history monotone non-increasing, HC `accepted == improvements`, GA `meanHistory ≥ bestHistory` pointwise, PPO `updates ≈ steps / rolloutLen`, full seed reproducibility.

```bash
npm run optimization-as-des              # run all five algorithms with comparison tables
npm run validate-optimization-as-des     # 45 checks across 3-5 seeds per algorithm
npm run test-optimization-as-des         # 39 base-class + leaf unit tests
```

### Whole-codebase base-class audit

Three additional bases — `TreeSearchStation<N>`, `FixedPointIterationStation<S>`,
and `ControllerStation<O, U>` — were added on top of the five
optimization-as-DES bases, and seven legacy iterative algorithms were
migrated onto them. Every iterative model in the codebase now extends at
least one `des-base/` class:

| Module                                | Base(s) used                                                     | Composition |
| ------------------------------------- | ---------------------------------------------------------------- | ----------- |
| `general/sa-des.ts` (SA, Hill Climb)  | `SingleStateOptimizer<Tour>`                                     | 2 layers    |
| `general/ga-des.ts` (TSP GA)          | `PopulationOptimizer<Tour>`                                      | 2 layers    |
| `general/qlearning-des.ts`            | `RLAgentStation<number, number>`                                 | 2 layers    |
| `general/ppo-des.ts`                  | `PolicyGradientAgent` + `PolicyUpdateStation`                    | 2 layers    |
| `general/milp-bnb.ts`                 | `TreeSearchStation<MILPNode>` + composes `IncrementalLP`         | 2 layers    |
| `general/mcts.ts`                     | `TreeSearchStation<Node<S>>` (UCT walk)                          | 2 layers    |
| `general/value-iteration.ts`          | `FixedPointIterationStation<Float64Array>`                       | 2 layers    |
| `general/stochastic-lp.ts` (Benders)  | `FixedPointIterationStation<BendersIterState>` + `IncrementalLP` | 2 layers    |
| `general/temp-control.ts` (4 ctrls)   | `ControllerStation` ⟶ `TempControllerBase` ⟶ {bang-bang/PID/fuzzy/MPC} | **4 layers** |
| `general/genetic-tsp.ts`              | `PopulationOptimizer<Tour>` (with `acceptChild` for cut)         | 2 layers    |
| `general/simulated-annealing.ts`      | `SingleStateOptimizer<S>`                                        | 2 layers    |

`PopulationOptimizer` was extended with `acceptChild` + `childRetryLimit`
hooks so hard-constraint regimes (precedence-cut in genetic-TSP) fit the
breeding template without bypassing it.

### Pre-hooked-up validators on every base class

The base classes ship with a **validator protocol** so every iterative
algorithm carries its own intrinsic invariants and external-reference
comparisons:

```ts
station.addValidator(numericValidator({ name: 'q', extract: s => s.q, expected: 1.5, tol: 1e-6 }));
station.addValidator(externalReferenceValidator({
  name: 'vs scipy',
  referencePath: 'out/external/court-mdp/python.json',
  compare: (s, ref) => /* user-supplied comparator returns ValidationCheck[] */,
}));

const summary = runIterativeDES([station]);
//   summary.validation:    ValidationCheck[]   ← intrinsic + ground-truth + external
//   summary.validationOk:  boolean             ← AND of all checks
```

Lifecycle: `runTimeStep* → onFinalize → runValidation → summary.validation`.

| Factory in `des-base/validation.ts` | Purpose                                                |
| ----------------------------------- | ------------------------------------------------------ |
| `numericValidator`                  | scalar comparison (abs / rel tolerance)                |
| `boundValidator`                    | assert value ∈ `[low, high]`                           |
| `monotonicityValidator`             | assert series is non-increasing / non-decreasing       |
| `groundTruthValidator`              | custom equality against a reference                    |
| `intrinsicCheck`                    | wrap a `(station) → boolean` predicate                 |
| `externalReferenceValidator`        | load JSON + user-supplied `compare(station, ref)`      |

Concrete algorithms self-register relevant validators in their
constructors. Highlights:

- **TSPSAOptimizer / TSPGAOptimizer** auto-attach a Held-Karp
  ground-truth validator when `n ≤ 12`, plus monotonicity + permutation
  checks.
- **ValueIterationStation** auto-attaches a γ-contraction monotonicity
  check, AND optionally an external-reference comparator when
  `referencePath` is set in options.
- **BendersStation** auto-attaches `LB ≤ UB`, `gap ≤ tol when optimal`,
  and an optional scipy/extensive-form reference comparator.
- **MILPBnBStation** auto-attaches "search-finished" + "incumbent ⊆ LP
  relaxation" invariants.
- **TempControllerBase** auto-attaches a saturation check on every
  emitted control.

End-to-end demo + 40 unit checks: `npm run test-validation`.

## 7v7 youth soccer rotation: state augmentation made tangible

A real coaching problem: 12 players, 7 on field, 5 on bench, four 20-min periods. Affinity for each (player, position, period). **Fairness constraint: no player benched two consecutive periods.** Goal: pick a four-period schedule that maximises expected match performance.

This is one of the cleanest possible illustrations of the rule **"a Markov chain has memory only if you put the history into the state"** — see `general/soccer-rotation.ts`, `runners/validate-soccer.ts`, and `main-soccer-rotation.ts`.

### State augmentation in one table

The codebase contains two side-by-side MDPs over the same problem, same Hungarian inner solver, same affinity tensor — only the state representation differs:

| MDP variant | State | Affinity | Fairness violations | Per-player bench counts (out of 4) |
|---|---|---|---|---|
| `policyMDPVIMemoryless` | `(t,)` | 21.13 | **13** | P7=4, P10=4, P12=4 (three players sat ALL game), P1=P2=P8=P9=P11=0 (never sat) |
| `policyMDPVI` | `(t, prev_bench)` | 20.15 | **0** | every player sits 0–2 periods, never two in a row |

The memoryless MDP picks the same weakest five players every period because nothing in its state remembers who sat last period. Once `prev_bench` is in the state, the action set in period t can be restricted to bench-sets disjoint from `prev_bench` — and the fairness constraint becomes expressible. The 0.98 affinity drop is exactly the cost of the constraint.

This is the formal "state-space augmentation" theorem in operational form:

> A process is k-step Markov over states `s_t` *iff* it is 1-step Markov over the augmented states `S_t = (s_t, s_{t-1}, …, s_{t-k+1})`.

DES doesn't share this restriction — stations and movables can carry arbitrary local state. But to **lift** a DES to an MDP for tractable optimisation (value iteration, LP relaxation, etc.) the lifted state has to be Markov-sufficient.

### Memory dial: how the state space grows

| Constraint scope | Required state | Size for our problem |
|---|---|---|
| no constraint (memoryless) | `(t,)` | 4 |
| no two consecutive bench periods | `(t, prev_bench)` | 4 × C(12,5) = 3168 |
| no two-of-three-period bench windows | `(t, prev_bench, prev_prev_bench)` | 4 × C(12,5) × 21 ≈ 67k |
| arbitrary lifetime constraints (e.g. "each player plays ≥ 2 periods") | full bench history | 4 × 2¹² ≈ 16k |

Each layer of memory multiplies state-space size by the number of distinct histories the constraint can distinguish. This is exactly the "curse of state-space explosion" you trade off against richer constraints.

### Layered architecture

```
Layer 3 — OPTIMISATION ENGINE
  ├─ random schedule
  ├─ greedy-Hungarian per period (fairness-aware pre-fill)
  ├─ multi-period LP relaxation (simplex / interior-point) → Hungarian-rounded
  ├─ memoryless MDP-VI                 (state = (t,))     ← pedagogical
  └─ augmented MDP-VI exact            (state = (t, prev_bench)) ← optimal

Layer 2 — DECISION ABSTRACTION
  └─ MDP at period boundaries
     reward at each (s, a) = Hungarian-optimal assignment of the 7 on-field
                              players to the 7 positions (= bipartite matching
                              solvable in O(7³) by `general/hungarian.ts`)
     transition: deterministic, s' = (t+1, a)

Layer 1 — PHYSICAL / DYNAMIC WORLD
  └─ DES match simulator: ticks every game-minute, samples Poisson goal
     events from on-field affinity, fires substitution events at the
     start of every 20-min period
```

### Empirical results (seed=4242 instance)

```
LP upper bound on rotation = 20.149449
MDP-VI exact optimum       = 20.149449   ← LP is tight, integrality gap = 0
LP-relaxation (rounded)    = 20.149449   ← rounding preserves the LP value here
greedy-Hungarian per period= 19.601      ← 97% of optimum
memoryless MDP             = 21.132      ← higher (no fairness penalty), VIOLATES
random                     =  9.704      ← VIOLATES fairness
```

All four LP solvers — internal simplex, **DES-engine simplex** (the simplex driven by our own DES tick loop, see the LP section above), scipy:HiGHS-DS, scipy:HiGHS-IPM — agree on the LP optimum to 7.11e-15. The Python scipy reference matches to 9 digits.

### Animation

`ANIMATE=1 npm run soccer` writes an HTML player showing the pitch with the 7 starters in canonical 7v7 positions (GK / 2 backs / 1 sweeper / 2 mids / 1 striker), the 5-player bench at the side, a live on-field affinity bar, the running scoreboard, and a "SUB WINDOW" flash at every 20-minute period boundary. Goal events flash yellow (us) / red (them).

### Running

```bash
npm run soccer                              # head-to-head + per-player fairness audit
ANIMATE=1 npm run soccer                    # writes pitch animation HTML
LP_SOLVER=scipy:highs-ipm npm run soccer    # interior-point on the rotation LP
LP_SOLVER=internal       npm run soccer    # in-process simplex (pure JS)
N_MATCHES=200 SEED=99    npm run soccer    # alternate instance / more matches
npm run validate-soccer                     # 16 / 16 Welch-t studies
npm run test-soccer                         # 49 / 49 unit tests
```

## Newsvendor and multi-period inventory MDP

The classic newsvendor (newsboy) problem and its multi-period
generalisation are the canonical pedagogical examples for value
iteration: the single-period case has a famous closed-form optimum
(the **critical fractile** rule) that we can pin against, and the
multi-period extension has no closed form but well-known structural
results (base-stock for zero fixed cost, (s, S) for positive fixed
cost) that the MDP solver discovers without being told.

### Single-period newsvendor (main-newsvendor.ts)

```
profit(q, D) = p · min(q, D) + s · (q − D)+ − c · q
```

with unit cost `c`, unit price `p`, salvage `s`, and discrete demand
distribution. The optimum order quantity is

```
underage cost  c_u = p − c
overage  cost  c_o = c − s
critical ratio CR  = c_u / (c_u + c_o)

q*  =  inf { q :  P(D ≤ q) ≥ CR }    (smallest q with CDF ≥ CR)
```

We solve it three different ways and check they all agree:

1. **Analytical** — close-form CDF inversion at the critical ratio.
2. **Brute search** — exhaustive enumeration over q ∈ [0, qMax],
   computing E[profit(q)] exactly from the demand PMF.
3. **MDP value iteration** — model the day as a 2-state, A-action MDP
   (state 0 = morning, state 1 = absorbing terminal), action `a` =
   order quantity, expected reward = E[profit(a)], discount γ = 0.

Sample run at `c = 0.5, p = 1, s = 0.1, D ~ Poisson(50)`:

```
(a) Analytical critical-fractile        q*=51   E[profit]=22.4816
(b) Brute search over q ∈ [0, 125]      q*=51   E[profit]=22.4816
(c) MDP value iteration (1-step, γ=0)   q*=51   V(0)=22.4816   iterations=2
(sim) 1000-day simulation at q=51       mean profit/day=22.37  (analytical 22.48)
```

The MDP "discovers" the same q* the critical-fractile formula gives
analytically — the formula is just the closed-form solution of value
iteration's Bellman equation in the special case of a one-step
problem.

### Multi-period inventory MDP (main-inventory-mdp.ts)

When leftover inventory **carries over** to the next day, the problem
becomes a true infinite-horizon MDP:

```
state    x ∈ {0, …, X_max}            (inventory at start of period)
action   a ∈ {0, …, A_max}             (order quantity)
demand   D ~ DemandDist                (independent across periods)
reward   r(x, a, D) = p · min(x+a, D)
                    − c · a − K · 1{a > 0}      (linear + fixed cost)
                    − h · (x + a − D)+          (holding)
                    − L · (D − x − a)+          (lost-sales penalty)
next     x' = (x + a − D)+
discount γ ∈ (0, 1)
```

Value iteration produces V*(x) and π*(x). Two structural results from
inventory theory tell us what the optimal policy *should* look like:

| fixed cost K | optimal policy structure |
|---|---|
| K = 0 | **base-stock**: π*(x) = max(0, S* − x) for some constant S*.   "Order up to S* whenever x < S*". |
| K > 0 | **(s, S)** (Scarf 1960): π*(x) = S* − x if x ≤ s*, else 0.   "When stock falls to s*, re-order up to S*; otherwise don't." |

The MDP solver makes no structural assumption — it discovers these
shapes from data. Sample sweep over fixed-cost K (validation runner):

```
   K       S*    s*    S − s    structure
   0       26    25       1      base-stock
   1       26    21       5      s-S
   5       45    17      28      s-S
  10       47    14      33      s-S
  25       50    10      40      s-S
  50       50     3      47      s-S
```

The (s, S) gap grows monotonically with K — exactly what economic
intuition predicts (higher setup costs ⇒ less frequent, larger
orders), and the validator pins this as a PASS gate.

### Validation studies (validate-newsvendor.ts → 28/28 PASS)

1. **Three solution methods agree** on q* and E[profit] across four
   parameter regimes (Poisson and uniform demand, low/high margins).
2. **Multi-period reduces to single-period at γ → 0**: the MDP
   solver's π(0) at γ = 0 with `holdCost = −salvage` exactly equals
   the newsvendor q*.
3. **Policy structure** is correctly classified by `detectPolicyStructure`
   across the K-sweep above.
4. **Simulation matches Bellman value**: long-run average reward of a
   50 000-day simulation under the discovered policy is within 5% of
   `V(0) · (1 − γ)`.
5. **Bit-exact match against Python (numpy/scipy reference)**: V(0) =
   240.9033 and policy [47, 46, 45, …, 33, 0, 0, 0, 0, 0] match
   between the TS implementation and the standalone numpy MDP solver.

### Run any of them

```bash
npm run build

# Single-period newsvendor: three methods printed side-by-side
LAMBDA=50 UNIT_COST=0.5 UNIT_PRICE=1.0 UNIT_SALVAGE=0.1 \
  node dist/des/main-newsvendor.js

# Multi-period: discover base-stock policy
LAMBDA=20 FIXED_COST=0  GAMMA=0.95 node dist/des/main-inventory-mdp.js

# Multi-period: discover (s, S) policy
LAMBDA=20 FIXED_COST=10 GAMMA=0.95 node dist/des/main-inventory-mdp.js

# Animation: per-day inventory + profit chart
LAMBDA=20 FIXED_COST=10 ANIMATE=1 ANIM_DAYS=120 \
  node dist/des/main-inventory-mdp.js

# Full validation (TS-internal + Python cross-check)
node dist/des/runners/validate-newsvendor.js   # 28/28 PASS

# Python reference standalone (set NEWSVENDOR_PY for non-default python)
python3 external-references/newsvendor/newsvendor.py --lambda 50
python3 external-references/newsvendor/newsvendor.py --multi --lambda 20 --K 10 --p 2.0 --gamma 0.95
```

### Generic value iteration (general/value-iteration.ts)

The newsvendor and inventory MDPs share `general/value-iteration.ts`,
a parameterised value-iteration solver decoupled from the
USACC court MDP. It takes an `MDPSpec` (state count, action count
per state, `outcomes(s, a)` callable) and returns `V*` and the greedy
policy. Pre-builds the transition table once, validates that outcome
probabilities sum to 1 ± 1e-9, supports γ ∈ [0, 1], and converges to
1e-9 tol on practical-sized problems (a few hundred states) in a few
ms. Reusable for any other discrete MDP.

## How is person-to-person transmission actually modelled?

A natural question when looking at the existing SEIR engines is: **how
do they represent interactions between individuals?** The default
answer in compartmental models is mean-field / mass-action, which
is *implicitly* a pairwise-contact model whose pairs are marginalised
away. `main-contact-seir.ts` makes the three plausible interaction
models explicit and runnable side by side.

### The three interaction kernels

```
                  P(S → E in dt)                 What it models
    mass-action   1 − exp(−β · I/N · dt)         Mean-field. Each S sees an
                  with β = c · p                  average prevalence I/N. The
                                                  actual contact pair is invisible.

    pairwise      Each person initiates           Symmetric pair-contact:
                  Poisson(c_i · 0.5 · dt)         contact rate between i and j
                  contacts. Each contact has      is (c_i + c_j) / (2N). High-c
                  pTransmit chance to fire if     people initiate AND are
                  the pair is (S, I).             contacted more, so their
                                                  offspring count grows linearly
                                                  in c.

    triplet       Each S samples Poisson(c · dt)  Complex contagion: simple
                  "triplet meetings", each with   exposure to one I is not
                  two random others. Transmits    enough; it takes 2 simul-
                  with prob pTransmit only when   taneous I to flip an S. Force
                  BOTH partners are I.            of infection scales as
                                                  (I/N)², not I/N — a sharper
                                                  epidemic threshold.
```

All three drive the same E → I (rate σ) and I → R (rate γ) dynamics, so
only the kernel differs. All three accept **heterogeneous contact
rates**: `c_i` is drawn from `Gamma(shape, scale)` with mean
`contactRate` and coefficient of variation `contactRateCV`. CV = 0
gives a homogeneous population; CV > 0 produces a long-tailed
distribution of contact rates (super-spreaders).

### How they compare

`runners/validate-contact-vs-meanfield.ts` runs three studies and
ships the outcomes as PASS/FAIL gates (15/15 PASS at last run):

**Study 1: convergence as N → ∞.** Mass-action and pairwise should
agree on mean attack rate and mean R₀(seed cases) in the homogeneous
limit:

```
  N=  500  attack: mass=82.0% pair=82.6%  Welch p=0.63    R₀(idx): mass=1.83 pair=1.88  Welch p=0.81
  N= 2000  attack: mass=80.7% pair=80.2%  Welch p=0.63    R₀(idx): mass=1.99 pair=2.27  Welch p=0.36
  N= 5000  attack: mass=80.3% pair=80.3%  Welch p=0.99    R₀(idx): mass=2.16 pair=2.23  Welch p=0.81
```

**Study 2: super-spreader (Gini) effect, only visible with explicit
pairs.** With heterogeneous contact rates, a small fraction of cases
should produce a large fraction of secondaries — the **20/80 rule**.
Mean-field can't see this because infectors are picked uniformly at
random; pairwise reveals it because high-c people both initiate and
are contacted more often, multiplicatively boosting their offspring
count.

```
  CV    pairwise Gini  pairwise top-20% share   mass-action Gini  mass-action top-20% share
  ────  ─────────────  ──────────────────────   ────────────────  ─────────────────────────
  0.0   0.693                  68.1%             0.690                    67.9%
  0.5   0.695                  68.7%             0.653                    64.2%
  1.0   0.695                  70.5%             0.695                    69.2%
  2.0   0.764                  78.7%             0.681                    66.8%
```

At CV = 2, pairwise top-20% causes 78.7% of secondaries (Gini 0.764);
mass-action top-20% causes 66.8% (Gini 0.681). The 12pp gap is the
super-spreader effect that only the explicit pair-contact kernel can
produce.

**Study 3: triplet has a sharp epidemic threshold.** Because triplet
transmission rate scales as `(I/N)² · c · p`, a triplet epidemic does
not ignite at low seed density — pairwise does:

```
  I₀     I₀/N     pairwise-attack    triplet-attack
  ─────  ───────  ──────────────     ──────────────
      5  0.0010           62.1%            0.1%
     50  0.0100           80.6%            1.0%
    200  0.0400           83.7%            4.9%
    500  0.1000           84.7%           17.2%
   1000  0.2000           86.9%           58.6%
```

This is the qualitative signature of complex contagion (e.g.
behaviours that require multiple reinforcing exposures, like adopting
a meme or refusing a vaccine). Standard SIR cannot reproduce this
threshold by tuning β alone.

### When to choose which

| Need                                                | Kernel                |
|-----------------------------------------------------|-----------------------|
| Aggregate dynamics in a large homogeneous population | mass-action (cheapest) |
| Explicit per-person transmission tree, super-spreader effects, finite-N variance | pairwise |
| Threshold dynamics, social/complex contagion        | triplet                |
| Anything where dispersion of secondaries matters    | pairwise (CV > 1)     |

The classical SEIR in `main-epidemic-improved.ts` uses mass-action.
The two-disease model in `main-two-disease.ts` also uses mass-action
(per-compartment). The contact-SEIR model here is what you reach for
when those approximations break — small populations, sharp
heterogeneity, super-spreaders, or threshold contagion.

### Run any of them

```bash
npm run build

# Mass-action (mean-field). c = 6, p = 0.05, R₀ ≈ 2.1.
N=2000 KERNEL=mass-action  REPS=10 node dist/des/main-contact-seir.js

# Pairwise with super-spreaders (CV = 2 → some people have c >> mean).
N=2000 KERNEL=pairwise     REPS=10 CONTACT_CV=2 node dist/des/main-contact-seir.js

# Triplet (complex contagion). Needs higher c·p to ignite.
N=2000 KERNEL=triplet      REPS=10 CONTACT_RATE=30 INITIAL_I=200 node dist/des/main-contact-seir.js

# Side-by-side comparison + statistical gates
node dist/des/runners/validate-contact-vs-meanfield.js   # 15/15 PASS

# Animation (smaller N for tractable file size)
N=400 KERNEL=pairwise CONTACT_CV=2 ANIMATE=1 node dist/des/main-contact-seir.js
```

## Random-variable toolkit and convolutions

Several places in the codebase need to compute or sample from sums of
independent random variables. The relevant fact is that the
**distribution of X + Y** for independent X, Y is the **convolution** of
their distributions:

```
  f_{X+Y}(z) = (f_X * f_Y)(z) = ∫ f_X(t) f_Y(z − t) dt
                                  (continuous)
  P(X+Y = k) = Σ_i P(X = i) · P(Y = k − i)        (discrete)
```

`src/des/general/random-variables.ts` provides the toolkit, used in the
two-disease compartments and in any model where the sum of independent
RVs needs to be reasoned about analytically:

  * `discreteConvolve(p, q)`     — O(|p|·|q|) PMF convolution.
  * `discreteConvolveMany(pmfs)` — iterative left-fold for N PMFs.
  * `discreteConvolveSelf(p, n)` — n-fold self-convolution via repeated
                                    squaring (log₂ n convolutions).
  * `binomialPMF(n, p)`          — closed form, stable for n ≤ ~1500.
  * `poissonBinomialPMF(probs)`  — exact PMF of Σ Bernoulli(p_i) for
                                    independent but heterogeneous p_i.
  * `competingRisks(rates, dt)`  — exact `[p_no, p_1, …, p_K]` first-event
                                    probabilities under the formula above.
  * `sampleCategorical(probs, rng)` — uniform-driven sampler.
  * `meanFromPMF(pmf)`, `varianceFromPMF(pmf)`, `normalizePMF(pmf)`.

Each function is pinned by an analytic identity *and* a Monte Carlo
cross-check in `src/des/test/random-variables-test.ts`:

  * `Bernoulli(p)*ⁿ = Binomial(n, p)` — bit-equal for n ∈ {1, 5, 17, 32, 100}.
  * `PoissonBinomial(uniform p) = Binomial(n, p)` — bit-equal.
  * `mean(p ⊕ q) = mean(p) + mean(q)` and `var(p ⊕ q) = var(p) + var(q)`
    for all tested PMF pairs.
  * Convolution is associative — `(p*q)*r = p*(q*r)` to 1e-14.
  * `competingRisks` matches a per-person first-event Monte Carlo
    simulation of competing exponentials to within `5e-3` at N=10⁵ samples.
  * `PoissonBinomial` PMF matches a Bernoulli-sum Monte Carlo histogram
    within `5e-3` at N=2×10⁵ samples.

All 34 tests pass.

The toolkit also enables cross-checks the simulation can compute on its
own output. For example, the two-disease driver computes a Poisson-
binomial PMF over per-person final-death probabilities and reports

```
  simulation:  E[D] = 469.03,  std = 28.02
  PB model  :  E[D] = 469.03,  std = 15.52
```

The means match by construction; the simulation std is ~80% larger than
the independence-assumed PB std, which **quantifies the epidemic
coupling** — per-person death outcomes are not independent because they
share the same I_A(t) and I_B(t) trajectories. The PB std is a *lower
bound* on the simulation std for any positively-coupled epidemic.

## Animation plugin

The engine now ships with a generic animation plugin that turns any
simulation's per-tick state into a self-contained, scrubbable HTML
file. Two execution modes, both single-threaded (no second process
needed):

  * **In-line / "real-time"** — wire the simulation's tick loop to a
    `FrameRecorder.frame(t, tick, build)` call. The recorder appends
    each frame to a JSONL file and (optionally) prints a one-line
    `[anim] t=… tick=… frames=…` status to stderr, giving a live feel.
  * **Post-hoc** — read any `.frames.jsonl` produced earlier (or from
    a different run) and emit HTML with `node dist/des/animation/render.js
    <input.frames.jsonl> [output.html]`.

Both paths produce **byte-identical HTML** for the same input frames.

### Architecture

```
  ┌────────────┐  per-tick    ┌────────────────┐   JSONL    ┌────────────┐
  │ Simulation │──build()────▶│ FrameRecorder  │───────────▶│ frames file│
  └────────────┘  callback    └────────────────┘            └─────┬──────┘
                                       │                         │
                                       │           readAnimation │
                                       ▼                         ▼
                                   stderr "live"             ┌────────────┐
                                   tick line                 │ HtmlPlayer │
                                                             └─────┬──────┘
                                                                   ▼
                                                              standalone
                                                              .html file
```

* `src/des/animation/types.ts`           — `Animation`, `Frame`, `Shape`, `ChartSpec`
* `src/des/animation/frame-recorder.ts`  — record + flush + reload from JSONL
* `src/des/animation/html-player.ts`     — single-file HTML+SVG renderer (vanilla JS, no CDN)
* `src/des/animation/render.ts`          — post-hoc CLI
* `src/des/animation/scenes/`            — per-simulation scene builders

The HTML output is a single file with embedded JSON and a vanilla-JS
SVG player. It has no external dependencies — it can be opened from
disk on any machine. Controls: play/pause (space), step (←/→),
scrubber, speed selector (0.25× … 16×).

### Currently animated simulations

| Simulation        | Run with                       | Output                    |
|-------------------|--------------------------------|---------------------------|
| Two-disease epidemic | `ANIMATE=1 node dist/des/main-two-disease.js` | `out/two-disease.html` (6 compartment bars + line chart) |
| Elevator          | `ANIMATE=1 node dist/des/main-elevator.js` | `out/elevator.html` (building cross-section + occupancy chart) |
| Calculus (1-D heat / 2-D Poisson) | `ANIMATE=1 PROBLEM=pde node dist/des/main-calculus.js` | `out/heat.html` |
| FactMachine POMDP | `ANIMATE=1 node dist/des/main-factmachine.js` | `out/factmachine.html` (belief histogram + price + entropy) |
| LP simplex as DES (2-D polytope walk) | `ANIMATE=1 PROBLEM=2var-diamond node dist/des/main-lp-des.js` | `out/lp-des.html` (vertex-by-vertex pivot trajectory) |

For elevator, set `ANIMATE_DISPATCH=uncoordinated` (or `coordinated`,
`coordinated-pickup`, default) to choose which dispatch mode to
animate. For two-disease, `ANIMATE_REP=k` selects a specific
replication.

### Recommended pattern: render animations POST-HOC, not concurrently

For any simulation where the animation could slow the hot loop (or
where the user wants to scrub / replay at variable speed), the
recommended pattern is:

  1. Run the simulation to completion. Record per-tick state in the
     result object as plain arrays (e.g. `result.beliefSnapshots[t]`,
     `result.priceHistory[t]`). The simulation does NO frame-building
     work inline — it just appends numbers.
  2. After the simulation finishes, walk the recorded arrays and call
     `recorder.frame(t, tick, () => buildScene(t, snapshot))` for each
     captured tick.
  3. `recorder.finish()` emits the standalone HTML file.

This is exactly how `main-factmachine.ts` works: `runFactMachine`
captures `beliefSnapshots`, `priceHistory`, `actionHistory`, etc. into
the result; `renderAnimation(result, params)` is a separate function
that walks the trace and emits the HTML. Benefits:

  * the simulation hot-loop never pays for SVG / JSON serialisation,
    so we can run thousands of replications fast (e.g. policy
    comparison studies);
  * the same recorded trace can be rendered multiple times with
    different scene builders (binary view vs. scalar view, e.g.);
  * the user controls playback speed in the HTML player itself
    (0.25× → 16×), so the recorder's `fps` setting is just a default,
    not a constraint;
  * `ANIM_FRAMES=N` caps total frames (with strided subsampling)
    when T is very large, keeping the HTML file size bounded.

The "in-line / real-time" mode (frame-recorder called directly inside
the tick loop) is still supported and works fine for small simulations
or when you want a live `[anim] t=… tick=…` stderr trail, but it adds
constant per-tick overhead that scales with scene complexity.

### Adding animation to a new simulation

Two functions: a per-tick "scene builder" that returns shapes for the
current state, and a global chart spec for time series. Then plumb the
recorder into the tick loop:

```ts
import {FrameRecorder} from './animation/frame-recorder';
import {buildMyScene, buildMyChart} from './animation/scenes/my-scene';

const rec = new FrameRecorder({
  framesPath: 'out/my-sim.frames.jsonl',
  htmlPath:   'out/my-sim.html',
  width: 900, height: 600, fps: 30,
  title: 'My simulation', liveTickLine: true,
});

for (let tick = 0; tick < N; tick++) {
  // ... advance simulation by one tick ...
  rec.frame(tick * dt, tick, () => buildMyScene(state));
}
rec.setCharts([buildMyChart(timeSeries)]);
await rec.finish();
```

The frames are decoupled from the simulation engine — they're just
arrays of generic SVG shapes (`circle`, `rect`, `line`, `text`, `path`).
A scene builder is typically ~80–200 lines and lives next to the other
scene builders in `src/des/animation/scenes/`.

### Why this fits the engine architecture

Because every simulation in this engine runs on a uniform
`runTimeStep(stepSize)` tick clock, **the natural animation primitive
is one frame per tick** — the same time grid the simulation already
uses. There is no need for the recorder to extrapolate or interpolate.
Contrast with FEL kernels (SimPy, Ciw), where event firings are
unevenly spaced in time; animating those usually requires sampling at
fixed intervals to get a smooth playback.

The frame schema is also intentionally minimal — generic SVG shapes
with positions and colors, no domain types. A scene builder for a new
simulation needs to know nothing about the animation infrastructure
beyond `Shape[]`. That keeps the plugin engine-agnostic; you could
in principle pipe frames through it from any other DES kernel.

## Verification (recap)

The engine ships next to seven other reference implementations of the
same SEIR-with-hospitalization model:

| Reference          | Type           | Where                                  |
|--------------------|----------------|----------------------------------------|
| Closed-form $N^*$  | Algebraic      | `src/des/runners/MATH.md`             |
| Forward-Euler diff | Deterministic  | `src/des/runners/difference-runner.ts`|
| ODE RK4            | Deterministic  | `src/des/runners/ode-runner.ts`       |
| FEL (M/M/1 or M/M/∞)| Stochastic    | `src/des/runners/fel-runner.ts`       |
| Gillespie SSA      | Stochastic     | `src/des/runners/gillespie-runner.ts` |
| scipy + sympy      | Det + Symbolic | `external-references/scipy-ode/`      |
| SimPy              | Stochastic     | `external-references/simpy/`          |
| Ciw                | Stochastic     | `external-references/ciw/`            |
| Octave             | Deterministic  | `external-references/octave/`         |
| R + deSolve        | Deterministic  | `external-references/r-desolve/`      |

All five deterministic methods agree on the closed-form $N^*$ to
floating-point precision. All five stochastic kernels (FEL-individual,
Gillespie, SimPy, Ciw, and the framework's `PerIndividualProcessor` at
small `stepSize`) agree on every time-averaged population within Welch
noise (`p > 0.18` on every compartment). Full results in
[`CHANGELOG.md`](CHANGELOG.md).

## Tunable time-step: approaching continuous time

A central design choice in this engine is that **`stepSize` is a knob, not a
constant**. Every kernel and every simulation in this repo accepts `stepSize`
(or its analog `dt`) at runtime. As `stepSize → 0`, the simulation
**provably converges to its continuous-time counterpart**. As `stepSize`
gets large, the simulation gets faster, less accurate, and eventually
diverges. You pick where on that curve you want to operate.

This is materially different from a classical FEL kernel (SimPy, Ciw,
SSJ, scipy.solve_ivp), which always runs in continuous time and exposes
no equivalent control. Each paradigm has its place:

| Need                                            | Use this engine with…       | Or use a FEL kernel              |
|-------------------------------------------------|-----------------------------|----------------------------------|
| Quick scoping run, accept ~10% error            | `stepSize` ≈ 1 / fastest rate | overkill                         |
| Production-grade accuracy                       | `stepSize` ≪ 1 / fastest rate | natural fit                      |
| Sweep precision-vs-cost trade-off               | yes, this is the headline   | not directly possible            |
| Step-by-step debugging of every station         | yes — fixed-step is uniform | event-jumping makes this awkward |
| Asymptotic O(events) cost on quiet networks     | no                          | yes                              |

### Empirical convergence: same model, three independent demonstrations

The "framework approaches continuous time as `stepSize → 0`" claim is
verified empirically across three of the simulations in this repo, each
against an independent continuous-time reference written in a different
language with a different float library:

**1. SEIR (`stepsize-sweep.ts`).** The framework's three-queue
`EntityProcessor` produces inflated populations in fast compartments
relative to FEL. Sweep over `stepSize ∈ {1.0, 0.5, 0.1, 0.05}`:

```
                framework population vs FEL-fifo, mean over 30 reps
  stepSize  Δ(I-P)    Δ(I-S)    Δ(I-H)
  1.0       +28%      +30%      +35%
  0.5       +14%      +15%      +18%
  0.1       +2.8%     +3.0%     +3.6%
  0.05      +1.4%     +1.5%     +1.8%
```

Halving `stepSize` halves the gap. The new `PerIndividualProcessor`
shrinks the prefactor further; at `stepSize = 0.05` it agrees with FEL,
SimPy, Ciw, and scipy LSODA at Welch `p > 0.18` on every compartment.

**2. Electric circuit (`validate-electric-circuit.ts`).** Forward Euler
on a series RLC step response, compared to the closed-form analytical
underdamped solution and to `scipy.solve_ivp(method='LSODA', rtol=1e-10)`:

```
  dt        ticks   max|V_C - analytical|    max|V_C - scipy|    empirical order
  0.5       60      5.34e+1  (DIVERGED)      5.34e+1               --
  0.1       300     2.51e-1                  2.51e-1              3.33   ← unstable
  0.05      600     1.06e-1                  1.06e-1              1.25
  0.01      3000    1.90e-2                  1.90e-2              1.07
  0.005     6000    9.36e-3                  9.36e-3              1.02
  0.001     30000   1.85e-3                  1.85e-3              1.01   ← textbook
```

Above the stability bound (`dt > 2/ω₀ = 2`), forward Euler blows up —
this is by design and is the correct behaviour of the underlying
numerical method. Below the stability bound, the empirical order
converges to **1.01**, exactly the theoretical first-order accuracy of
forward Euler.

LSODA-vs-analytical is **8.3e-13** — i.e. the framework can be made
arbitrarily close to a continuous-time gold standard by paying with
more ticks.

**3. Elevator (`validate-elevator.ts`).** The framework discretizes
arrivals and dispatch decisions at `dt = 0.5 s`; SimPy fires events at
exact moments. Aggregate metrics:

```
  metric          framework    SimPy      Δ        Δ / dt
  meanWait         8.84 s       9.06 s    -0.22 s   -0.43
  meanTravel      10.00 s      10.61 s    -0.61 s   -1.22
  meanTotal       18.84 s      19.67 s    -0.83 s   -1.65
```

Δ scales with `dt`. Simulating at `dt = 0.05 s` would tighten the gap
by ~10×, at the cost of 10× more ticks.

### What this means architecturally

The fixed-step kernel is *uniform in time*. Every station gets a
`runTimeStep(stepSize)` call every tick — no matter how busy or quiet
the system is. That is a strength when:

- You want predictable runtime (linear in `simT / stepSize`, regardless
  of event density).
- You want to debug by stepping through one tick across the entire
  station graph.
- You want to dial precision *down* for cheap exploratory runs.
- You want to dial precision *up* until residual error is below the
  question you're asking.

It is a weakness when:

- The network is mostly quiet — a FEL kernel with `O(events)` cost would
  blow past the fixed-step kernel's `O(simT / stepSize)`.
- The fastest dynamic in the system requires extremely small `stepSize`
  (e.g. resolving microsecond signals over a multi-day horizon).

The empirical message from the three convergence demonstrations above
is the same: **as long as you size `stepSize` smaller than the inverse
of the fastest rate you care about, this engine recovers continuous-time
accuracy to whatever precision you're willing to pay for.**

## Numerical-precision caveats

JavaScript's `number` is IEEE-754 binary64. mathjs's `BigNumber` is
arbitrary-precision decimal but coerces to `number` whenever it crosses
into ordinary arithmetic. Both have well-known biases that can creep
into a stochastic simulation:

- `0.1 + 0.2 !== 0.3` and similar binary-rounding artifacts.
- Summation of `n` small floats drifts by up to roughly
  `ULP × n` in the worst case (random-walk model gives
  `~ULP × √n`).
- `Math.random()` is implementation-defined and not seedable across
  runs; we replace it with `mulberry32` (32-bit state, exact uniform
  output on a `2^-32` grid) for reproducibility.
- `Number(math.bignumber(x))` round-trips perfectly for decimal
  fractions that fit in 64 bits, but mathjs's transcendental functions
  (`exp`, `log`, …) are series approximations at the configured
  precision, not bit-exact.
- mathjs configuration matters: `BigNumber.precision` defaults to 64
  decimal digits and rounding defaults to half-up; both can affect
  tie-breaking in `floor` / `ceil`.

Where this engine actually exercises mathjs / float arithmetic:

| Site                                                        | Lib used     |
|-------------------------------------------------------------|--------------|
| `runTimeStep(stepSize: math.BigNumber, ...)`                | BigNumber    |
| Per-tick `Number(stepSize)` coercion in `PerIndividualProcessor` | BigNumber → `number` |
| Residence-time draws (`U(a, b)` via `a + (b-a) * rng()`)    | `number`     |
| Histogram bucketing `Math.floor(t / stepSize)`              | `number`     |
| Inter-arrival timing comparisons against `phase1Days`       | `number`     |
| Probability-decision branching (`r < p`)                    | `number`     |
| External kernels (Gillespie, FEL, ODE, difference eq)       | `number` only|

`src/des/test/float-bias-test.ts` measures the actual bias at each of
those sites and asserts conservative bounds. Sample bounds we observe
on this hardware:

- Plain-`number` accumulator drift after 1,000,000 ticks of
  `stepSize = 0.05`: ~6.7e-7 (relative drift ~1.3e-11).
- BigNumber + Kahan-compensated accumulators: drift = 0 to one ULP.
- `U(0.7, 1.3)` sample mean over 1,000,000 draws: within
  ±6.9e-4 of 1.0 (4 standard errors of the ideal mean).
- `U(0.20, 0.40)` sample variance: within ±1.2e-5 of 1/300.
- `mulberry32` chi-square uniformity (100 buckets, 1e6 draws):
  χ² ≈ 110 < 159.7 (α = 1e-4 critical value).
- First 4096 outputs of `mulberry32` are pairwise distinct (a
  permutation property).

The strongest argument that mathjs and float64 are not corrupting
simulation outputs is the cross-implementation agreement. The same
SEIR model is implemented six ways (`framework`, `PerIndividualProcessor`,
in-repo FEL, Gillespie SSA, RK4 ODE, forward-Euler difference equation)
plus five external kernels (SimPy, Ciw, scipy + sympy, Octave, R +
deSolve). Each uses a different float library (V8 + mathjs, CPython
float64, NumPy/SciPy, Octave, R, mpmath via SymPy). All five
deterministic methods agree on the closed-form $N^*$ to floating-point
precision; all five stochastic methods (FEL-individual, Gillespie,
SimPy, Ciw, our `PerIndividualProcessor`) agree on every time-averaged
population at Welch `p > 0.18`. Any systematic mathjs- or JS-specific
bias would have to be smaller than that noise floor.

If you extend the engine, the rules of thumb for keeping bias low are:

1. **Always seed via `withSeed(seed, fn)` for reproducibility.** Do not
   call `Math.random()` directly inside a kernel.
2. **Construct mathjs values from strings, not floats.** Prefer
   `math.bignumber('0.05')` to `math.bignumber(0.05)` — the latter
   passes through V8's `Number` parsing first and absorbs its rounding.
3. **Compare floats with explicit tolerance, not equality.** Avoid
   `time === phase1Days` in favour of `time + 0.5 * stepSize >= phase1Days`.
4. **Don't sum stepSize over millions of ticks unless you really need to.**
   Use `k * stepSize` (multiplicative) when you can — float64 is exact
   for that. The engine's main loop happens to do it the additive way,
   which we measured at <1e-6 absolute drift over 1e6 ticks.
5. **Avoid mathjs transcendental functions in hot paths.** `Math.exp`
   and `Math.log` from V8 are bit-exact for our needs; mathjs's are
   approximations whose precision is configurable but slower.

## Quick start

```bash
cd courses/hdm-fall-2022/des
npm install
npm run build

# Run the headline epidemic model with the framework kernel
node dist/des/main-epidemic-improved.js

# Verify the engine against five other implementations
node dist/des/test/queue-bias-test.js                      # 21/21 PASS
node dist/des/test/float-bias-test.js                      # 33/33 PASS
N=30 STEPSIZE=0.05 node dist/des/runners/validate-references.js
N=5  HORIZON=10000  node dist/des/runners/steady-state.js

# (optional) Add SimPy / Ciw / scipy-ode / Octave / R as second-opinion columns
python3 -m venv .venv-external && source .venv-external/bin/activate
pip install -r external-references/simpy/requirements.txt \
            -r external-references/ciw/requirements.txt \
            -r external-references/scipy-ode/requirements.txt
N=10 bash external-references/run-all.sh
N=20 STEPSIZE=0.05 node dist/des/runners/validate-with-externals.js

# Other simulations (each is self-contained and validates against a different
# external reference: numpy / pure-Python / scipy / SimPy / Python value iteration)
node dist/des/main-convolution.js                        # write framework output
node dist/des/main-backpropagation.js
node dist/des/main-electric-circuit.js
node dist/des/main-elevator.js                           # all three dispatch modes side by side
node dist/des/main-court-mdp.js                          # USACC MDP, 4 policies, V* + π*
node dist/des/main-two-disease.js                        # 6-compartment co-infection model
bash external-references/run-all.sh                      # write reference outputs
node dist/des/runners/validate-convolution.js            # ~1 ULP vs numpy
node dist/des/runners/validate-backpropagation.js        # ~16 ULPs vs Python after 10k SGD steps
node dist/des/runners/validate-electric-circuit.js       # forward-Euler order = 1.01
node dist/des/runners/validate-elevator.js               # aggregate Δ within 10% of SimPy
node dist/des/runners/validate-court-mdp.js              # bit-exact V*; identical π* on 864 states
node dist/des/runners/validate-two-disease.js            # vs LSODA + Gillespie SSA, Welch p=0.6
node dist/des/test/elevator-invariants-test.js           # 7 invariants × 45 configs
node dist/des/test/random-variables-test.js              # 34/34 RV-toolkit identities
node dist/des/test/animation-test.js                     # 24/24 animation plugin tests

# Animations: write a self-contained scrubbable HTML to out/
ANIMATE=1 node dist/des/main-two-disease.js              # → out/two-disease.html
ANIMATE=1 node dist/des/main-elevator.js                 # → out/elevator.html
ANIMATE=1 node dist/des/main-incremental-lp.js           # → out/incremental-lp.html (live LP)
node dist/des/animation/render.js out/two-disease.frames.jsonl  # post-hoc render

# Incremental LP solver (warm-startable, adaptive to live add/remove of
# constraints, variables, objective). 42/42 validation, 40/40 tests.
npm run incremental-lp                                   # console trace
npm run validate-incremental-lp                          # vs static solveLPInternal, max|Δ|=2e-13
npm run test-incremental-lp                              # 40 unit tests

# Stochastic LP via SAA monolithic + Benders-as-DES + closed-form oracle.
# 60/60 validation including statistical convergence at 1/√N rate.
npm run stochastic-lp                                    # console comparison
node dist/des/main-from-json.js examples/stochastic-lp.json
npm run validate-stochastic-lp                           # ≈ 60s; statistical study
npm run test-stochastic-lp                               # 44 unit tests

# Multi-stage stochastic programming via SDDP + exact scenario-tree validation.
node dist/des/main-from-json.js examples/multistage-sddp.json
npm run test-multistage-stochastic                       # 20 unit tests

# Max-flow/min-cut, stochastic-flow MDP, and five-intersection traffic flow.
npm run max-flow
npm run stochastic-flow-mdp
npm run traffic-flow
node dist/des/main-from-json.js examples/max-flow.json
node dist/des/main-from-json.js examples/stochastic-flow-mdp.json
node dist/des/main-from-json.js examples/traffic-flow.json
npm run test-network-flow                                # 83 unit tests

# Indoor temperature control DES — bang-bang vs PID vs Fuzzy-PI vs MDP-MPC
# on a 24h winter day. 21/21 validation studies, 36/36 unit tests.
npm run temp-control                                     # head-to-head + sensitivity sweep
npm run temp-control-anim -- --controller pid            # → out/temp-control/animation.html
npm run validate-temp-control                            # 21 studies
npm run test-temp-control                                # 36 unit tests

# MILP via branch-and-bound, composing IncrementalLP for relaxations at each
# node. 22/22 validation, 38/38 unit tests.
npm run milp-bnb                                         # 5-study console driver
npm run validate-milp-bnb                                # vs brute force / LP solver
npm run test-milp-bnb                                    # 38 unit tests

# Explicit IP/MIP solver graph: LP relaxation station, rounding/repair,
# incumbent station, cut generator, and branch decision station.
npm run ip-mip-des
LP_ALGO=des-simplex-dantzig npm run ip-mip-des
node dist/des/main-from-json.js examples/ip-mip-des-knapsack.json
npm run test-ip-mip-des                                  # 39 unit tests
npm run validate-ip-mip-external                         # external exact solver cross-check

# Simulated annealing on TSP / knapsack (and any SAProblem<S>).
# 31/31 validation, 31/31 unit tests.
npm run simulated-annealing                              # SA + comparison vs GA + Held-Karp
npm run validate-simulated-annealing                     # 31 checks
npm run test-simulated-annealing                         # 31 unit tests

# "Optimization-as-DES": four algorithms (SA, HC, GA, Q-learning, PPO) all
# rebuilt as concrete LEAVES of four algorithm-family base classes
# (SingleStateOptimizer, PopulationOptimizer, RLAgentStation,
# PolicyGradientAgent), running on the SAME runIterativeDES runner.
# Each leaf is ~150 lines and implements ONLY the algorithm-specific hooks.
# 45/45 end-to-end validation across 3-5 seeds, 39/39 unit tests.
npm run optimization-as-des                              # Pentagon TSP + n=12 TSP + GridWorld + Corridor(8)
npm run validate-optimization-as-des                     # SA/HC/GA/Q-learning/PPO vs ground truth
npm run test-optimization-as-des                         # base-class mechanics + leaf convergence

# Run any registered model from a JSON spec file (no recompile required).
# Registered models include temp-control, shortest-path, milp-bnb,
# simulated-annealing, max-flow, traffic-flow, math-equation,
# internal-solver-network, and feasibility-pipeline.
node dist/des/main-from-json.js --list                   # discover models
node dist/des/main-from-json.js --schema temp-control    # parameter schema
node dist/des/main-from-json.js examples/temp-control-pid.json
node dist/des/main-from-json.js examples/temp-control-mdp-mpc.json
node dist/des/main-from-json.js examples/shortest-path-small-chain.json
node dist/des/main-from-json.js examples/milp-knapsack.json
node dist/des/main-from-json.js examples/sa-tsp-pentagon.json
node dist/des/main-from-json.js examples/internal-solver-knapsack-dp.json
node dist/des/main-from-json.js examples/feasibility-pipeline-knapsack.json

# MDP / control-systems sweeps
SEEDS=1,2,3,4,5 LAMBDAS=0.1,0.2,0.3,0.4 SIM_T=3600 \
  node dist/des/runners/compare-elevator-dispatch.js     # uncoordinated vs coordinated dispatch
```

## Layout

```
src/des/
  abstract/                   base classes (Entity, StationaryEntity, ...)
  general/
    general.ts                fisherYatesShuffle, helpers
    prng.ts                   mulberry32 + withSeed for reproducible runs
    random-variables.ts       discreteConvolve, poissonBinomialPMF,
                              competingRisks, binomialPMF, sampleCategorical, …
    stochastic-lp.ts          two-stage stochastic LP: SAA + Benders/L-shaped
    multistage-stochastic.ts  multi-stage inventory/storage SDDP + exact tree
    ip-mip-des.ts             explicit IP/MIP station graph solver
    max-flow.ts               max-flow/min-cut via Edmonds-Karp DES ticks
    stochastic-flow-mdp.ts    MDP interpretation of stochastic max-flow
    traffic-flow.ts           five-intersection traffic DES with moving cars
    adapters/
      statistical-optimization-adapter.ts  JSON adapters for stochastic-lp + risk/fitting/simopt
      multistage-sddp-adapter.ts           JSON adapter for multistage-sddp
      network-flow-adapter.ts              JSON adapter for animated/logged max-flow + traffic-flow
      stochastic-flow-mdp-adapter.ts       JSON adapter for stochastic-flow-mdp
      milp-bnb-adapter.ts                  JSON adapters for milp-bnb + ip-mip-des
    des-base/
      cut-pool.ts             affine cut pools for Benders / SDDP / OA
  entity-source/              source / emitter stations
  entity-processing/
    processing.ts             EntityProcessor (3-queue, M/M/c)
    per-individual-processor.ts  PerIndividualProcessor (1-queue, M/M/inf)
    value-adder.ts            arithmetic stations
  entity-decision/            ProbabilityDecisionEntity, BinaryDecisionEntity
  entity-routing/             entity splitter
  entity-travel/              time-delay travel between stations
  entity-queue/               standalone queue node
  entity-moving/              AbstractMovingEntity
  signals/                    signal-style nodes (mux, integral, derivative)
  observability/
    logger.ts                 JSONL event logger
    validate-epidemic.ts      log-driven sanity checker
  mdp/
    usacc-mdp.ts              USACC MDP definition (states/actions/transitions/rewards)
    value-iteration.ts        finite-state value iteration solver
  runners/                    KERNELS + DRIVERS used for verification
    types.ts                  shared SimConfig / RunResult schema
    framework-runner.ts       runs the framework kernel
    per-individual-runner.ts  runs PerIndividualProcessor
    fel-runner.ts             classical FEL reference (fifo or individual)
    gillespie-runner.ts       direct-method SSA
    ode-runner.ts             RK4 mean-field ODE
    difference-runner.ts      forward-Euler difference equation + closed form
    replicate.ts              N=30 reps + Welch t-test (framework vs FEL)
    stepsize-sweep.ts         dt sweep with CSV/SVG output
    per-individual-vs-fel.ts  PI vs FEL-individual + convergence sweep
    validate-references.ts    PI vs FEL-individual vs Gillespie vs ODE
    validate-with-externals.ts  same + SimPy / Ciw / scipy-ode / Octave / R
    external-program.ts       sanctioned shell-free external program runner
    external-modules.ts       registered source-only external solver/validator modules
    steady-state.ts           open-system N* verification across all kernels
    validate-convolution.ts   max-abs error vs numpy.convolve
    validate-backpropagation.ts max-abs weight error vs Python
    validate-electric-circuit.ts forward-Euler convergence-order vs scipy LSODA
    validate-elevator.ts      aggregate match vs SimPy
    validate-court-mdp.ts     V* / π* match vs Python value iteration
    validate-two-disease.ts   ∫populations + Welch t-test vs LSODA + Gillespie SSA
    validate-ip-mip-external.ts  IP/MIP DES vs source-only exact Python reference
    validate-contact-vs-meanfield.ts  pairwise ≡ mass-action large N; Gini super-spreader; triplet threshold
    validate-newsvendor.ts    critical-fractile ≡ brute ≡ MDP; (s, S) sweep; bit-exact vs numpy reference
    compare-elevator-dispatch.ts  uncoordinated vs coordinated vs coordinated-pickup
    MATH.md                   derivations: ODE -> diff eq -> closed form
    README.md                 kernel/driver index
  reference/
    main-epidemic-fel.ts      stand-alone FEL implementation of the model
    compare-epidemic.ts       early one-shot diff
  test/
    iterator-test.ts          tiny smoke test for LinkedQueue
    queue-bias-test.ts        21-assertion bias test for @oresoftware/linked-queue
    float-bias-test.ts        33-assertion bias test for mathjs/JS float64 ops
    random-variables-test.ts  34-assertion identity + Monte Carlo bias test
    multistage-stochastic-test.ts  SDDP exact-tree + cut-pool tests
    network-flow-test.ts      max-flow/min-cut + traffic invariants
    elevator-invariants-test.ts  7-invariant-family checker × 45 configs
    animation-test.ts         24-assertion smoke test for the animation plugin
  animation/
    types.ts                  Frame / Shape / ChartSpec / Animation types
    frame-recorder.ts         FrameRecorder + readAnimation
    html-player.ts            single-file HTML+SVG renderer
    render.ts                 post-hoc CLI: <frames.jsonl> → <html>
    scenes/
      two-disease-scene.ts    6-bar compartment view + line chart
      elevator-scene.ts       building cross-section + occupancy chart
      contact-seir-scene.ts   per-person dot grid (radius ∝ √c) + chart
      newsvendor-scene.ts     per-period inventory bars + profit chart
  main-fibonacci-recursion.ts hello-world simulation
  main-epidemic.ts            original epidemic model
  main-epidemic-improved.ts   recalibrated SEIR + observability + Markov chain export
  main-markov.ts              Markov chain visualiser
  main-convolution.ts         1-D FIR convolution, validated vs numpy.convolve
  main-backpropagation.ts     2-3-1 sigmoid net trained on XOR, validated vs Python
  main-electric-circuit.ts    series RLC step response, validated vs analytical + scipy LSODA
  main-elevator.ts            3 elevators / 4 floors / SCAN + coordinated MDP dispatch, validated vs SimPy
  main-court-mdp.ts           USACC MDP simulation: 4 stages, 8 actions, 4 policies, validated vs Python VI
  main-two-disease.ts         6-compartment co-infection model, validated vs LSODA + Gillespie SSA
  main-contact-seir.ts        Pair / triplet contact SEIR; super-spreader Gini, complex contagion
  main-newsvendor.ts          Single-period stochastic inventory; analytical / brute / MDP all agree
  main-inventory-mdp.ts       Multi-period inventory MDP; discovers base-stock and (s, S) policies

external-references/          off-the-shelf libraries running the same model
  README.md                   schema + env-var contract
  run-all.sh                  PYTHON_BIN / OCTAVE_BIN / RSCRIPT_BIN aware
  simpy/            seir.py + requirements.txt
  ciw/              seir.py + requirements.txt
  scipy-ode/        seir.py + requirements.txt
  octave/           seir.m
  r-desolve/        seir.R
  convolution/      conv.py     (numpy reference for main-convolution.ts)
  backpropagation/  bp.py       (Python naive reference for main-backpropagation.ts)
  electric-circuit/ circuit.py  (analytical + scipy LSODA for main-electric-circuit.ts)
  elevator/         elevator.py (SimPy continuous-time reference for main-elevator.ts)
  court-mdp/        court_mdp.py (Python value iteration reference for main-court-mdp.ts)
  two-disease/      two_disease.py (LSODA + Gillespie SSA references for main-two-disease.ts)
  ip-mip/           ip_mip_reference.py (source-only IP/MIP reference solver)

CHANGELOG.md                  full validation results + change log
```

## Further reading

- [`CHANGELOG.md`](CHANGELOG.md) — full inventory of changes and the
  validation results that confirm engine accuracy.
- [`src/des/runners/MATH.md`](src/des/runners/MATH.md) — continuous-time
  ODE, discrete-time difference equation, closed-form steady state, and
  forward-Euler stability bound for the SEIR model.
- [`src/des/runners/README.md`](src/des/runners/README.md) — index of every
  kernel and driver in this repo.
- [`external-references/README.md`](external-references/README.md) — how
  to install and run SimPy / Ciw / scipy-ode / Octave / R + deSolve, plus
  source-only external solver modules such as the IP/MIP reference.
