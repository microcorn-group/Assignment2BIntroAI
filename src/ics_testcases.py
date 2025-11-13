"""ICS test cases runner.

Covers 10+ scenarios to satisfy Assignment 2B testing requirement:
- Different O-D pairs
- k-shortest vs UCS
- With/without distance-based timing
- Different severity levels (simulated)

Run:
  python -m src.ics_testcases
"""
from __future__ import annotations
import os
from typing import List, Tuple

from .ics_graph import build_graph, k_shortest_paths, path_travel_time
from .search import uniform_cost_search

SEVERITIES = [None, 'minor', 'moderate', 'severe']
OD_PAIRS: List[Tuple[int,int]] = [
    (1, 13), (1, 10), (1, 9), (14, 3), (12, 11), (2, 4), (7, 6), (8, 10), (9, 12), (15, 3)
]


def run_case(severity, use_distance_time: bool, algorithm: str, k: int = 3):
    print(f"\n=== severity={severity or 'none'} distTime={use_distance_time} algo={algorithm} ===")
    G = build_graph(severity, use_distance_time=use_distance_time, speed_kmh=60.0)
    for o, d in OD_PAIRS:
        if algorithm == 'ucs':
            path, cost = uniform_cost_search(G, o, d, weight='weight')
            if path:
                print(f"UCS {o}->{d}: {cost:.2f} min path={path}")
            else:
                print(f"UCS {o}->{d}: no path")
        else:
            routes = k_shortest_paths(G, o, d, k=k)
            if routes:
                costs = [path_travel_time(G, p) for p in routes]
                print(f"KSP {o}->{d}: {[f'{c:.2f}' for c in costs]} min paths={routes}")
            else:
                print(f"KSP {o}->{d}: no path")


def main():
    # 4 severities x 2 timing modes x 2 algorithms = 16 "suites"
    for sev in SEVERITIES:
        for dist in (False, True):
            for algo in ('ksp', 'ucs'):
                run_case(sev, dist, algo)


if __name__ == '__main__':
    main()
