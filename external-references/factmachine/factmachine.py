#!/usr/bin/env python3
"""External numpy/scipy reference for the FactMachine POMDP.

This is NOT a re-implementation of the full simulation — it just pins the
*Bayesian belief update* and the *win-probability-under-majority* binomial
identity, which are the only two pieces of pure mathematics the TS code
needs to be exactly right about.

Inputs are read from environment variables; output is JSON on stdout.

  PROBLEM=belief    Run the Bayesian filter for a fixed observation
                    sequence and report the posterior over θ.
  PROBLEM=pwin      Compute P(majority of N voters votes YES | θ) for
                    a sweep of θ; used to verify the TS pYesWins().

Required: numpy, scipy. Install with  pip install numpy scipy.
"""

from __future__ import annotations
import json
import os
import sys

try:
    import numpy as np
    from scipy.stats import binom
except Exception as exc:                                       # pragma: no cover
    sys.stderr.write(f"# numpy/scipy not available: {exc}\n")
    sys.exit(1)


def belief_filter() -> dict:
    """Bayesian filter for FactMachine: hidden θ, observed (yes_orders, total)
    each tick, observation model q(θ) = θ·informedness + 0.5·(1-informedness),
    likelihood Binomial(total, q(θ)).

    Inputs (env vars):
      THETA_BINS    bins; default 21
      INFORMEDNESS  default 0.6
      OBS           "yes/total,yes/total,..."  e.g. "12/20,15/22,9/19"
      PRIOR         optional comma-separated prior weights (length=THETA_BINS)
    """
    K = int(os.environ.get('THETA_BINS', '21'))
    informedness = float(os.environ.get('INFORMEDNESS', '0.6'))
    obs_raw = os.environ.get('OBS', '12/20,15/22,9/19,17/20,11/18,14/19,16/20,10/22')
    obs = [tuple(map(int, pair.split('/'))) for pair in obs_raw.split(',')]
    thetas = np.linspace(0, 1, K)
    if 'PRIOR' in os.environ:
        prior = np.array([float(x) for x in os.environ['PRIOR'].split(',')])
        prior = prior / prior.sum()
    else:
        prior = np.ones(K) / K
    b = prior.copy()
    history = [b.tolist()]
    means = [float((b * thetas).sum())]
    entropies = [-float((b[b > 0] * np.log(b[b > 0])).sum())]
    for (y, n) in obs:
        q = thetas * informedness + 0.5 * (1.0 - informedness)
        # Stable log-likelihood (drop the binomial coefficient; constant in θ).
        log_lik = y * np.log(np.clip(q, 1e-300, 1)) + (n - y) * np.log(np.clip(1 - q, 1e-300, 1))
        lik = np.exp(log_lik - log_lik.max())                  # rescale for stability
        post = b * lik
        post = post / post.sum()
        b = post
        history.append(b.tolist())
        means.append(float((b * thetas).sum()))
        entropies.append(-float((b[b > 0] * np.log(b[b > 0])).sum()))
    return {
        'problem': 'belief',
        'K': K,
        'informedness': informedness,
        'obs': obs,
        'final_belief': b.tolist(),
        'final_mean': means[-1],
        'final_entropy': entropies[-1],
        'mean_history': means,
        'entropy_history': entropies,
    }


def pwin_majority() -> dict:
    """Compute P(YES wins majority | θ, N) for θ ∈ {0.1, ..., 0.9} with N=51.
    Uses scipy.stats.binom.sf(half, N, θ). Pinned to the TS pYesWins().
    """
    N = int(os.environ.get('N_VOTERS', '51'))
    half = N // 2
    thetas = np.linspace(0.1, 0.9, 9)
    pwin = [float(binom.sf(half, N, t)) for t in thetas]   # P(X > half)
    return {
        'problem': 'pwin',
        'N': N, 'half': half,
        'thetas': thetas.tolist(),
        'pwin': pwin,
    }


def main() -> int:
    problem = os.environ.get('PROBLEM', 'belief')
    handlers = {
        'belief': belief_filter,
        'pwin':   pwin_majority,
    }
    if problem not in handlers:
        sys.stderr.write(f"unknown PROBLEM='{problem}'.  options: {list(handlers)}\n")
        return 1
    print(json.dumps(handlers[problem]()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
