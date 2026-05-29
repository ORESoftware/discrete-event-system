
# TODO:
this is like visual programming - for Operations Research

# TODO:
allow user to define custom probability distributions
MLE - maximum likelihood estimators - tell us paramaters to a curve
compare MLE to "method of moments"

# TODO: 
user can provide probability distributions -OR- user can just provide a set of points, and probability of each point.
by default probability of each point is 1/N.

conversely, if the user gives us a function for a distribution, we most likely will find 1000 points to sample from,
in order to save computation time at runtime.

# TODO:

create library/toolkit/sdk for developing your own simulation

# TODO:
explore sparse fixed-step runner optimizations, but do not implement yet.

When time steps are very small, many entities may have `runTimeStep()` called
even though they have no work. The no-work path should stay a very cheap guard,
but future runner work could avoid visiting every idle entity on every tick.

Possible data structures / runner modes:
- active set: `take()`/emits mark target entities active; runner only visits active entities.
- two-phase active set: work created during tick `t` is processed in tick `t + 1`, preserving clean fixed-step semantics.
- dirty flag / work version: entities keep a cheap "has queued work" marker so `hasWork()` does not scan.
- timing wheel / bucket map keyed by tick: entities schedule themselves for a future discrete tick.
- min-heap FEL hybrid for continuous-time future wakeups.

Potential long-term hybrid runner:
- always-tick set for numerical blocks that must run every `dt`.
- active set for queued work delivered through graph connections.
- scheduled set for delayed/future work.

Key semantic question before implementing: should work emitted during a tick be
eligible to run in the same tick, or only on the next tick?
