# SEIR-with-hospitalization: continuous, discrete, and steady-state math

This is the analytical companion to `difference-runner.ts`, `ode-runner.ts`,
and `steady-state.ts`. It derives the model from scratch in three forms
(continuous-time ODE, discrete-time difference equation, closed-form
steady state) and shows how all three match the stochastic kernels in
expectation.

## 1. Setup

Parameters (`SimConfig`):

| Symbol | Meaning | Default |
|--------|---------|---------|
| $\lambda$ | source emission rate (entities/day) = $1/\mu_{arr}$ | $1.0$ |
| $\mu_c$ | mean residence time at compartment $c$ (days) | see `DEFAULT_RESIDENCE` |
| $p_a$ | $\Pr(\text{I-P} \to \text{I-A})$ | $0.40$ |
| $p_h$ | $\Pr(\text{I-S} \to \text{I-H})$ | $0.20$ |
| $p_d$ | $\Pr(\text{I-H} \to \text{D})$ | $0.12$ |
| $N_c(t)$ | expected population at compartment $c$ at time $t$ | – |

Topology (decision nodes flattened):

```
              p_a                                 1 - p_h            1 - p_d
source ──► S ──► E ──► I-P ──┬─────► I-A ─────────► R ◄─────────── I-H
            ▲                │                     ▲                 │
            │                │   1 - p_a           │  p_h            │ p_d
            │                └─► I-S ──────────────┘                 ▼
            │                          │                             D
            │                          │ p_h                         │
            └──────────────────────────┘                             ▼
                       (R loops back to S)                       main-sink
```

For an M/M/inf service discipline (each entity has its own clock), the
exit rate from compartment $c$ is $N_c / \mu_c$ — Little's law in
infinitesimal form.

## 2. Continuous-time mean-field ODE

Each compartment's rate of change is `inflow - outflow`:

$$
\begin{aligned}
\dfrac{dN_S}{dt}  &= \lambda(t) + \dfrac{N_R}{\mu_R} - \dfrac{N_S}{\mu_S} \\
\dfrac{dN_E}{dt}  &= \dfrac{N_S}{\mu_S} - \dfrac{N_E}{\mu_E} \\
\dfrac{dN_{IP}}{dt} &= \dfrac{N_E}{\mu_E} - \dfrac{N_{IP}}{\mu_{IP}} \\
\dfrac{dN_{IA}}{dt} &= p_a \dfrac{N_{IP}}{\mu_{IP}} - \dfrac{N_{IA}}{\mu_{IA}} \\
\dfrac{dN_{IS}}{dt} &= (1 - p_a) \dfrac{N_{IP}}{\mu_{IP}} - \dfrac{N_{IS}}{\mu_{IS}} \\
\dfrac{dN_{IH}}{dt} &= p_h \dfrac{N_{IS}}{\mu_{IS}} - \dfrac{N_{IH}}{\mu_{IH}} \\
\dfrac{dN_R}{dt}  &= \dfrac{N_{IA}}{\mu_{IA}} + (1 - p_h) \dfrac{N_{IS}}{\mu_{IS}} + (1 - p_d) \dfrac{N_{IH}}{\mu_{IH}} - \dfrac{N_R}{\mu_R} \\
\dfrac{dN_D}{dt}  &= p_d \dfrac{N_{IH}}{\mu_{IH}}
\end{aligned}
$$

Stack the alive compartments into a vector $\mathbf{N} = (N_S, N_E, N_{IP},
N_{IA}, N_{IS}, N_{IH}, N_R)^\top$. The system is linear:

$$
\dfrac{d\mathbf{N}}{dt} = A \mathbf{N} + \mathbf{b}(t)
$$

with $\mathbf{b}(t) = (\lambda(t), 0, 0, 0, 0, 0, 0)^\top$ and a fixed
matrix $A$ (negative diagonal $-1/\mu_c$, off-diagonals carrying the
branching coefficients $\frac{p_\ast}{\mu_\ast}$). The implementation
of the right-hand side lives in `ode-runner.ts::deriv()` (RK4) and
`difference-runner.ts` (forward Euler).

## 3. Discrete-time difference equation (forward Euler)

The simplest discretization with step $\Delta t$:

$$
\mathbf{N}(t + \Delta t) = \mathbf{N}(t) + \Delta t \, \big(A\, \mathbf{N}(t) + \mathbf{b}(t)\big)
= (I + \Delta t\, A)\,\mathbf{N}(t) + \Delta t\,\mathbf{b}(t).
$$

This is a linear map $\mathbf{N} \mapsto M \mathbf{N} + \mathbf{c}$ with
$M = I + \Delta t\, A$.

**Stability.** Forward Euler is stable iff every eigenvalue of $M$ has
magnitude $< 1$. For our $A$ the eigenvalues are bounded by the diagonal
$-1/\mu_c$, so we need

$$
\Big|1 - \dfrac{\Delta t}{\mu_c}\Big| < 1 \quad \forall c
\quad\Longleftrightarrow\quad
\Delta t < 2 \min_c \mu_c.
$$

For our defaults $\min_c \mu_c = \mu_D = 0.20$, so $\Delta t < 0.4$. The
`maxStableStep()` helper in `difference-runner.ts` returns this bound.

**Convergence to the ODE.** Forward Euler has local truncation error
$O(\Delta t^2)$ and global error $O(\Delta t)$, so

$$
\mathbf{N}_{\text{Euler}}(t; \Delta t) \to \mathbf{N}_{\text{ODE}}(t)
\quad\text{as }\Delta t \to 0.
$$

This is exactly what `steady-state.ts` shows by sweeping $\Delta t \in
\{0.5, 0.1, 0.05, 0.01\}$.

## 4. Closed-form steady state (open system, $\lambda$ constant)

Set $\frac{d\mathbf{N}}{dt} = 0$ — equivalently, $\mathbf{N}(t+\Delta t) =
\mathbf{N}(t)$ for the difference equation, since both yield $A\mathbf{N}^*
= -\mathbf{b}$. Define the per-compartment _throughput_

$$
f_c \equiv \dfrac{N_c^*}{\mu_c}.
$$

The steady-state equations become flow conservation at each node:

$$
\begin{aligned}
\lambda + f_R    &= f_S       &\text{(into S)}\\
f_S              &= f_E       &\text{(into E)}\\
f_E              &= f_{IP}    &\text{(into I-P)}\\
p_a\,f_{IP}      &= f_{IA}    &\text{(into I-A)}\\
(1-p_a)\,f_{IP}  &= f_{IS}    &\text{(into I-S)}\\
p_h\,f_{IS}      &= f_{IH}    &\text{(into I-H)}\\
f_{IA} + (1-p_h)f_{IS} + (1-p_d)f_{IH} &= f_R   &\text{(into R)}\\
p_d\,f_{IH}      &= f_D       &\text{(into D / out of system)}
\end{aligned}
$$

Substitute $f_E, f_{IP}, f_{IA}, f_{IS}, f_{IH}$ in terms of $f_S$:

$$
f_R = p_a\,f_S + (1-p_h)(1-p_a)\,f_S + (1-p_d)\,p_h(1-p_a)\,f_S
    = f_S\big[1 - (1-p_a)\,p_h\,p_d\big].
$$

Define

$$
\boxed{q \equiv (1 - p_a)\,p_h\,p_d}
$$

— the per-S-pass death fraction. Then $f_R = (1-q)\,f_S$, and the S
balance becomes

$$
\lambda + (1-q)\,f_S = f_S \quad\Longrightarrow\quad \boxed{f_S = \dfrac{\lambda}{q}.}
$$

Plug in defaults: $q = 0.6 \times 0.2 \times 0.12 = 0.0144$,
$f_S = 1.0 / 0.0144 \approx 69.444$/day. Steady-state populations follow
from Little's law $N_c^* = \mu_c \cdot f_c$:

| Compartment | $f_c$ (per day) | $\mu_c$ (days) | $N_c^*$ |
|-------------|------------------|----------------|---------|
| S      | $\lambda / q = 69.444$ | $0.30$ | $20.833$ |
| E      | $69.444$               | $0.30$ | $20.833$ |
| I-P    | $69.444$               | $0.30$ | $20.833$ |
| I-A    | $p_a \cdot 69.444 = 27.778$            | $0.30$ | $8.333$  |
| I-S    | $(1-p_a) \cdot 69.444 = 41.667$        | $0.30$ | $12.500$ |
| I-H    | $p_h(1-p_a) \cdot 69.444 = 8.333$      | $0.30$ | $2.500$  |
| D-flow | $\lambda = 1.000$                      | $0.20$ | $0.200$  |
| R      | $(1-q) \cdot 69.444 = 68.444$          | $2.00$ | $136.889$ |
| **Total alive** | – | – | **$\mathbf{222.911}$** |

Throughput consistency: $f_D = q \cdot f_S = \lambda$. The system kills
entities at the same rate the source emits them, as expected for an open
M/M/inf network at steady state.

## 5. Connection to the simulation runs

Our default `SimConfig` has $\text{sourceCap} = 500$ and
$\text{phase1Days} = 800$, so the source quiesces around day $500$ (mean
inter-arrival $1.0$, cap $500$) and the system drains over the
remaining $700$ days. The simulation never sits in steady state long, so
its time-averaged populations over $[0, T]$ are far below $N_c^*$ —
they're a mix of fill, loaded, and drain phases.

To verify the closed form, `steady-state.ts` runs the simulators in
**open-system mode**: $\text{sourceCap} = \infty$, $\text{phase1Days} =
\infty$, $T = 2000$. After a transient of $\sim \mu_R = 2$ days, the
system sits at $N^*$ and the time-averaged populations match the closed
form within statistical noise.

## 6. What gets verified

`steady-state.ts` outputs five population estimates per compartment:

1. **Closed-form analytical** $N_c^* = \mu_c f_c$ — pure algebra.
2. **Forward-Euler difference equation** at $\Delta t \in \{0.5, 0.1, 0.05, 0.01\}$ — converges to (1) as $\Delta t \to 0$.
3. **ODE RK4** — should match (1) and (2) at $\Delta t \to 0$ within the integrator's truncation error.
4. **Gillespie SSA** — exponential service, M/M/inf at the compartment level. Time-averaged populations match (1) within Welch noise.
5. **FEL-individual** — uniform service, M/M/inf with explicit entities. Same as (4).

If any of these disagree, exactly one of the model derivation, the
discretization, the integrator, or the stochastic kernel is wrong.

## 7. Stability and convergence rate

The matrix $A$ is strictly stable (eigenvalues have negative real part)
because each compartment has a positive exit rate $1/\mu_c$ and the
loop-back through R is fractional ($1-q < 1$). The slowest mode is
governed by $\mu_R = 2.0$ (longest residence), so the open-system
transient decays as $\sim e^{-t/\mu_R}$ and the system reaches steady
state within $\sim 5\mu_R = 10$ days. We use a $50$-day warmup in
`steady-state.ts` to be safe.

For the closed loop, the average number of S-passes before death is
$1/q = 1/0.0144 \approx 69.4$. Each pass takes roughly $\mu_S + \mu_E +
\mu_{IP} + \text{branch} + \mu_R = 0.3 \times 4 + 2.0 = 3.2$ days, so
the mean lifespan of a single entity is $\sim 222$ days. Total
population at steady state is $\sim \lambda \cdot 222 = 222$, matching
the table above.
