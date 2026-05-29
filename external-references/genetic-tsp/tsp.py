#!/usr/bin/env python3
"""
external-references/genetic-tsp/tsp.py — reference TSP solvers in
scipy / pure Python for cross-validating our DES-resident GA.

Sub-tools:

    nearest-neighbor   ← reads {coords: [[x,y], ...]} on stdin,
                         emits {tour, length} via the nearest-neighbor
                         heuristic (gives a quick upper bound)

    held-karp          ← reads {coords} on stdin, runs the bitmask DP
                         exact solver (n ≤ 14 is practical), emits
                         {tour, length}. Used as ground truth.

    one-tree-bound     ← reads {coords}, emits {bound} (1-tree relaxation
                         lower bound).

USAGE
-----
    cat <<EOF | python3 tsp.py held-karp
    {"coords": [[0,0], [1,0], [1,1], [0,1]]}
    EOF
"""
import argparse
import json
import math
import sys
import itertools

import numpy as np


def euclid_distance_matrix(coords):
    n = len(coords)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d[i][j] = math.sqrt(dx * dx + dy * dy)
    return d


def tour_length(d, tour):
    n = len(tour)
    return sum(d[tour[i]][tour[(i + 1) % n]] for i in range(n))


def nearest_neighbor(payload):
    coords = payload['coords']
    n = len(coords)
    d = euclid_distance_matrix(coords)
    best = None
    bestLen = float('inf')
    for start in range(n):
        visited = {start}
        tour = [start]
        cur = start
        for _ in range(n - 1):
            nxt = -1; nxtD = float('inf')
            for j in range(n):
                if j in visited: continue
                if d[cur][j] < nxtD: nxtD = d[cur][j]; nxt = j
            tour.append(nxt); visited.add(nxt); cur = nxt
        L = tour_length(d, tour)
        if L < bestLen: bestLen = L; best = tour
    return {'tour': best, 'length': bestLen}


def held_karp(payload):
    coords = payload['coords']
    n = len(coords)
    if n > 16:
        return {'error': f'Held-Karp practical only for n <= 16, got {n}'}
    d = euclid_distance_matrix(coords)
    INF = float('inf')
    # dp[mask][i] = min length starting at 0, visiting `mask`, ending at i.
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0.0
    for mask in range(1, 1 << n):
        if not (mask & 1): continue
        for i in range(n):
            if not (mask & (1 << i)): continue
            if dp[mask][i] == INF: continue
            for j in range(n):
                if mask & (1 << j): continue
                nm = mask | (1 << j)
                cand = dp[mask][i] + d[i][j]
                if cand < dp[nm][j]:
                    dp[nm][j] = cand
                    parent[nm][j] = i
    full = (1 << n) - 1
    bestEnd = 1
    bestLen = INF
    for i in range(1, n):
        cand = dp[full][i] + d[i][0]
        if cand < bestLen: bestLen = cand; bestEnd = i
    tour = []
    mask = full; cur = bestEnd
    while cur != -1:
        tour.append(cur)
        prev = parent[mask][cur]
        mask ^= 1 << cur
        cur = prev
    tour.reverse()
    return {'tour': tour, 'length': bestLen}


def one_tree_bound(payload):
    coords = payload['coords']
    n = len(coords)
    if n < 2: return {'bound': 0.0}
    d = euclid_distance_matrix(coords)
    # Prim's MST on {1, ..., n-1}.
    inTree = [False] * n
    minEdge = [float('inf')] * n
    inTree[1] = True
    mstCost = 0.0
    for j in range(2, n):
        minEdge[j] = d[1][j]
    for _ in range(n - 2):
        best = -1; bestVal = float('inf')
        for j in range(2, n):
            if not inTree[j] and minEdge[j] < bestVal:
                bestVal = minEdge[j]; best = j
        if best == -1: break
        mstCost += bestVal
        inTree[best] = True
        for k in range(2, n):
            if not inTree[k] and d[best][k] < minEdge[k]:
                minEdge[k] = d[best][k]
    edges0 = sorted(d[0][j] for j in range(1, n))
    return {'bound': mstCost + edges0[0] + edges0[1] if len(edges0) >= 2 else mstCost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['nearest-neighbor', 'held-karp', 'one-tree-bound'])
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    fn = {
        'nearest-neighbor': nearest_neighbor,
        'held-karp': held_karp,
        'one-tree-bound': one_tree_bound,
    }[args.cmd]
    result = fn(payload)
    json.dump(result, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
