// RUST MIGRATION: Target module `src/des/general/control_systems/numerical_solvers.rs`.
// RUST MIGRATION: Convert `OdeSystem` and fixed-step integrators into traits plus structs for plant dynamics and controller-facing simulation.
// RUST MIGRATION: Use `f64` vectors for states/derivatives, inject step size/config explicitly, and return `Result` for dimension or integration errors.
// RUST MIGRATION: Expose any graph-visible pure integration evaluator as a PureTransform-style struct with a `transform` method.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/control-systems/numerical-solvers.rs
//   (module des::general::control_systems::numerical_solvers)
// 1:1 file move. Object-oriented fixed-step ODE integrators (Euler, RK4).
//
// Declarations → Rust:
//   interface OdeSystem                  -> trait OdeSystem (dimension/derivative)
//   abstract class FixedStepIntegrator   -> trait FixedStepIntegrator with a
//                                           required `step` + default `integrate`/`axpy`
//   class ForwardEulerIntegrator / RungeKutta4Integrator extends FixedStepIntegrator
//                                        -> unit struct + impl FixedStepIntegrator
//
// Conversion notes (file-specific):
//   - `step(system: OdeSystem, ...)` takes the system by trait object/generic ->
//     `fn step<S: OdeSystem>(&self, system: &S, ..)` (or `&dyn OdeSystem`).
//   - `derivative` may read MUTABLE conditions on the system object (set between
//     ticks), so callers integrate a `&mut` system one step per DES tick.
//   - `throw new Error` for bad dt/steps are invariant checks -> panic! (or
//     debug_assert!); not recoverable Result paths.
//   - `state: readonly number[]` -> `&[f64]`; returns fresh `Vec<f64>` (no mutate).
// =============================================================================

// =============================================================================
// control-systems/numerical-solvers.ts — class-only fixed-step ODE integrators
// for the control-systems family.
//
// WHY CLASSES, NOT `rk4(f, …)`
// ───────────────────────────
//   The existing `general/ode.ts` exposes the classical solvers as free
//   functions taking an `f(t, y)` closure. The control-systems family is
//   deliberately object-oriented: the thing being integrated is an
//   `OdeSystem` OBJECT whose `derivative(t, state)` METHOD encodes the
//   dynamics (and may read mutable conditions such as the latest control
//   input). The integrators are likewise classes whose `step()` /
//   `integrate()` methods advance such an object.
//
//   This lets a DES plant station hold an `OdeSystem` field, mutate its
//   inputs on each `runTimeStep`, and call `integrator.step(system, …)` to
//   advance exactly one numerical step per discrete tick — the queueing +
//   numerical-methods coupling the project is built around.
// =============================================================================

/** A first-order vector ODE  dx/dt = f(t, x).  Implemented as an OBJECT so
 *  the dynamics live in a method (and can read mutable model inputs) rather
 *  than in a captured closure. */
export interface OdeSystem {
  /** Dimension n of the state vector. */
  dimension(): number;
  /** The right-hand side f(t, x). Must return a length-n vector. */
  derivative(t: number, state: readonly number[]): number[];
}

/** Common vector helpers shared by the integrators. Methods only — kept on a
 *  base class so concrete integrators inherit them instead of calling free
 *  vector functions. */
export abstract class FixedStepIntegrator {
  /** Advance the system by exactly one step of size `dt`, returning the new
   *  state. Pure with respect to `state` (does not mutate the input array). */
  abstract step(system: OdeSystem, t: number, state: readonly number[], dt: number): number[];

  /** Integrate from `t0` for `steps` steps of size `dt`. Returns the time
   *  grid and the state at each grid point (including the initial point). */
  integrate(
    system: OdeSystem,
    t0: number,
    state0: readonly number[],
    dt: number,
    steps: number,
  ): {times: number[]; states: number[][]} {
    if (!(dt > 0)) throw new Error(`${this.constructor.name}.integrate: dt must be > 0`);
    if (!Number.isInteger(steps) || steps < 0) {
      throw new Error(`${this.constructor.name}.integrate: steps must be a non-negative integer`);
    }
    const times: number[] = [t0];
    const states: number[][] = [state0.slice()];
    let t = t0;
    let x = state0.slice();
    for (let k = 0; k < steps; k++) {
      x = this.step(system, t, x, dt);
      t += dt;
      times.push(t);
      states.push(x.slice());
    }
    return {times, states};
  }

  /** out = a + s·b. */
  protected axpy(a: readonly number[], b: readonly number[], s: number): number[] {
    const out = new Array<number>(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + s * b[i];
    return out;
  }
}

/** Forward (explicit) Euler.  x_{n+1} = x_n + dt·f(t_n, x_n). */
export class ForwardEulerIntegrator extends FixedStepIntegrator {
  step(system: OdeSystem, t: number, state: readonly number[], dt: number): number[] {
    const k1 = system.derivative(t, state);
    return this.axpy(state, k1, dt);
  }
}

/** Classical fourth-order Runge–Kutta.  The workhorse for the control-systems
 *  plants (smooth, non-stiff dynamics). */
export class RungeKutta4Integrator extends FixedStepIntegrator {
  step(system: OdeSystem, t: number, state: readonly number[], dt: number): number[] {
    const half = dt / 2;
    const k1 = system.derivative(t, state);
    const k2 = system.derivative(t + half, this.axpy(state, k1, half));
    const k3 = system.derivative(t + half, this.axpy(state, k2, half));
    const k4 = system.derivative(t + dt, this.axpy(state, k3, dt));
    const out = new Array<number>(state.length);
    for (let i = 0; i < state.length; i++) {
      out[i] = state[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
    return out;
  }
}
