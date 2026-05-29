#!/usr/bin/env python3
"""
external-references/shortest-path/sp.py — networkx reference for
single-source shortest paths.

Sub-tools:

    bellman-ford   ← reads {numNodes, edges:[[u,v,w], ...], source}
                     on stdin, emits {distance, predecessor} via
                     networkx.single_source_bellman_ford. Negative
                     weights allowed; negative cycles raise.

    dijkstra       ← same input shape, runs networkx.single_source_dijkstra,
                     requires non-negative weights.

USAGE
-----
    cat <<EOF | python3 sp.py bellman-ford
    {"numNodes": 5, "edges": [[0,1,1],[1,2,2],[2,3,2],[3,4,1],[0,4,10]],
     "source": 0}
    EOF
"""
import argparse
import json
import math
import sys

try:
    import networkx as nx
except ImportError:
    nx = None


def build_graph(payload):
    if nx is None: raise RuntimeError('networkx not installed')
    G = nx.DiGraph()
    G.add_nodes_from(range(payload['numNodes']))
    for u, v, w in payload['edges']:
        G.add_edge(u, v, weight=w)
    return G


def to_distance_array(numNodes, lengths_dict):
    arr = []
    for v in range(numNodes):
        arr.append(lengths_dict.get(v, math.inf))
    return arr


def to_predecessor_array(numNodes, paths_dict):
    arr = [-1] * numNodes
    for v, path in paths_dict.items():
        if len(path) >= 2:
            arr[v] = path[-2]
    return arr


def bellman_ford(payload):
    G = build_graph(payload)
    src = payload['source']
    try:
        lengths, paths = nx.single_source_bellman_ford(G, src)
        return {
            'distance': to_distance_array(payload['numNodes'], lengths),
            'predecessor': to_predecessor_array(payload['numNodes'], paths),
            'hasNegativeCycleFromSource': False,
        }
    except nx.NetworkXUnbounded:
        return {'hasNegativeCycleFromSource': True}


def dijkstra(payload):
    G = build_graph(payload)
    src = payload['source']
    lengths, paths = nx.single_source_dijkstra(G, src)
    return {
        'distance': to_distance_array(payload['numNodes'], lengths),
        'predecessor': to_predecessor_array(payload['numNodes'], paths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['bellman-ford', 'dijkstra'])
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    fn = {
        'bellman-ford': bellman_ford,
        'dijkstra': dijkstra,
    }[args.cmd]
    result = fn(payload)
    json.dump(result, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
