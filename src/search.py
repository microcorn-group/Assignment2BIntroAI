"""Uniform Cost Search (UCS) implementation for Assignment 2B.

This provides a baseline pathfinding algorithm (from Part A) that we can
compare against NetworkX utilities. It operates on a NetworkX graph and
returns the minimum-cost path given a weight.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Any, Optional
import heapq
import math
import networkx as nx


WeightFn = Callable[[int, int, Dict[str, Any]], float]


def _get_edge_weight(G: nx.DiGraph, u: int, v: int, weight: Optional[WeightFn | str]) -> float:
    data = G[u][v]
    if callable(weight):
        return float(weight(u, v, data))
    key = weight or 'weight'
    return float(data.get(key, 1.0))


def uniform_cost_search(
    G: nx.DiGraph,
    start: int,
    goal: int,
    weight: Optional[WeightFn | str] = 'weight',
) -> Tuple[List[int], float]:
    """Compute the minimum-cost path from start to goal using UCS.

    Returns (path, cost). If no path exists, returns ([], math.inf).
    """
    if start not in G or goal not in G:
        return [], math.inf

    # Priority queue of (cost_so_far, node, predecessor)
    frontier: List[Tuple[float, int]] = []
    heapq.heappush(frontier, (0.0, start))

    # Track best known cost to each node and predecessor for path reconstruction
    best_cost: Dict[int, float] = {start: 0.0}
    predecessor: Dict[int, Optional[int]] = {start: None}

    while frontier:
        cost, node = heapq.heappop(frontier)
        if node == goal:
            # Reconstruct path
            path: List[int] = []
            cur: Optional[int] = node
            while cur is not None:
                path.append(cur)
                cur = predecessor.get(cur)
            path.reverse()
            return path, cost

        # If we popped a stale entry with higher cost than best known, skip
        if cost > best_cost.get(node, math.inf):
            continue

        for nbr in G.successors(node):
            w = _get_edge_weight(G, node, nbr, weight)
            new_cost = cost + w
            if new_cost < best_cost.get(nbr, math.inf):
                best_cost[nbr] = new_cost
                predecessor[nbr] = node
                heapq.heappush(frontier, (new_cost, nbr))

    return [], math.inf
