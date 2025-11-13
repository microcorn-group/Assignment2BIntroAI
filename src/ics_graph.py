import networkx as nx
from typing import Dict, List, Tuple, Optional
import math

# Minimal Kuching Heritage nodes (subset from example HTML)
NODES: Dict[int, Dict] = {
    1: {"name": "Masjid Bandaraya Kuching", "lat": 1.558708, "lon": 110.340547},
    2: {"name": "Padang Merdeka", "lat": 1.557014, "lon": 110.343616},
    3: {"name": "Plaza Merdeka", "lat": 1.558558, "lon": 110.344180},
    4: {"name": "St. Thomas' Cathedral", "lat": 1.557108, "lon": 110.345049},
    5: {"name": "Padang Merdeka Car Park (South Gate)", "lat": 1.556950, "lon": 110.344800},
    6: {"name": "Old Courthouse Auditorium", "lat": 1.558856, "lon": 110.344629},
    7: {"name": "Jalan Pearl Junction", "lat": 1.558300, "lon": 110.343200},
    8: {"name": "Haji Openg Junction", "lat": 1.554494, "lon": 110.343009},
    9: {"name": "Sarawak Museum Admin Building", "lat": 1.551219, "lon": 110.343969},
    10: {"name": "Sarawak Museum (Old Building)", "lat": 1.551910, "lon": 110.343500},
    11: {"name": "Sarawak Islamic Heritage Museum", "lat": 1.551450, "lon": 110.342800},
    12: {"name": "Sarawak Art Museum", "lat": 1.553400, "lon": 110.341900},
    13: {"name": "Kuching Waterfront Jetty", "lat": 1.557950, "lon": 110.347900},
    14: {"name": "Wisma Hopoh", "lat": 1.556000, "lon": 110.340200},
    15: {"name": "Heroes Monument", "lat": 1.553000, "lon": 110.340800},
}

# Simple road catalog: (u, v, name, road_type, time_min, camera, way_id)
# Note: times are illustrative and loosely based on the provided HTML. Edges are directed.
EDGES: List[Tuple[int, int, Dict]] = [
    (1, 7, {"name": "Jalan Masjid", "type": "primary", "time": 4.0, "camera": False, "way_id": 2024}),
    (7, 1, {"name": "Jalan Masjid", "type": "primary", "time": 5.0, "camera": False, "way_id": 2025}),
    (7, 3, {"name": "Jalan Pearl", "type": "secondary", "time": 3.0, "camera": False, "way_id": 2008}),
    (3, 7, {"name": "Jalan Pearl", "type": "secondary", "time": 2.0, "camera": False, "way_id": 2009}),
    (3, 4, {"name": "Jalan McDougall", "type": "tertiary", "time": 3.0, "camera": False, "way_id": 2006}),
    (4, 3, {"name": "Jalan McDougall", "type": "tertiary", "time": 4.0, "camera": False, "way_id": 2005}),
    (3, 6, {"name": "Courthouse Link", "type": "secondary", "time": 2.0, "camera": False, "way_id": 2010}),
    (6, 3, {"name": "Courthouse Link", "type": "secondary", "time": 2.0, "camera": False, "way_id": 2010}),
    (6, 13, {"name": "Main Bazaar Road", "type": "primary", "time": 7.0, "camera": True, "way_id": 2011}),
    (7, 2, {"name": "Ajibah Abol", "type": "primary", "time": 6.0, "camera": True, "way_id": 2001}),
    (2, 7, {"name": "Ajibah Abol", "type": "primary", "time": 4.0, "camera": False, "way_id": 2002}),
    (2, 5, {"name": "Cathedral Access", "type": "service", "time": 2.0, "camera": False, "way_id": 2007}),
    (5, 4, {"name": "McDougall Inner", "type": "service", "time": 3.0, "camera": False, "way_id": 2027}),
    (7, 8, {"name": "Tun Abang Haji Openg", "type": "secondary", "time": 5.0, "camera": False, "way_id": 2012}),
    (8, 7, {"name": "Tun Abang Haji Openg", "type": "secondary", "time": 4.0, "camera": False, "way_id": 2013}),
    (8, 9, {"name": "Tun Abang Haji Openg", "type": "secondary", "time": 4.0, "camera": False, "way_id": 2014}),
    (9, 8, {"name": "Tun Abang Haji Openg", "type": "secondary", "time": 3.0, "camera": False, "way_id": 2015}),
    (9, 10, {"name": "Museum Link", "type": "secondary", "time": 3.0, "camera": True, "way_id": 2016}),
    (10, 9, {"name": "Museum Link", "type": "secondary", "time": 3.0, "camera": False, "way_id": 2017}),
    (9, 11, {"name": "P. Ramlee", "type": "secondary", "time": 4.0, "camera": False, "way_id": 2018}),
    (11, 9, {"name": "P. Ramlee", "type": "secondary", "time": 3.0, "camera": False, "way_id": 2019}),
    (11, 12, {"name": "P. Ramlee Extension", "type": "secondary", "time": 6.0, "camera": False, "way_id": 2020}),
    (12, 11, {"name": "P. Ramlee Extension", "type": "secondary", "time": 5.0, "camera": False, "way_id": 2021}),
    (12, 15, {"name": "Taman Budaya Loop", "type": "secondary", "time": 3.0, "camera": False, "way_id": 2022}),
    (15, 12, {"name": "Taman Budaya Loop", "type": "secondary", "time": 3.0, "camera": False, "way_id": 2023}),
    (14, 1, {"name": "Satok Connector", "type": "secondary", "time": 8.0, "camera": True, "way_id": 2026}),
    (1, 14, {"name": "Satok Connector", "type": "secondary", "time": 8.0, "camera": True, "way_id": 2026}),
    (1, 2, {"name": "Ajibah Abol (via 7)", "type": "primary", "time": 9.0, "camera": False, "via": [1,7,2], "way_id": 3000}),
]

ROAD_COLORS = {
    "primary": "deepskyblue",
    "secondary": "purple",
    "tertiary": "darkblue",
    "service": "slategray",
}

SEVERITY_TO_MULTIPLIER = {
    "minor": 1.2,
    "moderate": 1.8,
    "severe": 3.0,
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_graph(
    severity: Optional[str] = None,
    use_distance_time: bool = False,
    speed_kmh: float = 60.0,
) -> nx.DiGraph:
    """Create and return a directed graph with travel time weights.

    - If use_distance_time is True, per-edge time is computed from great-circle
      distance between node coordinates assuming speed_kmh (default 60 km/h).
      Minutes = (distance_km / speed_kmh) * 60.
    - Otherwise, use the provided 'time' attribute from EDGES.

    If severity is provided, camera-monitored edges have their time multiplied
    according to SEVERITY_TO_MULTIPLIER.
    """
    G = nx.DiGraph()
    for nid, data in NODES.items():
        G.add_node(nid, **data)

    multiplier = SEVERITY_TO_MULTIPLIER.get(severity, 1.0) if severity else 1.0

    for u, v, attrs in EDGES:
        if use_distance_time:
            a, b = NODES[u], NODES[v]
            km = _haversine_km(a['lat'], a['lon'], b['lat'], b['lon'])
            base_minutes = (km / max(speed_kmh, 1e-6)) * 60.0
        else:
            base_minutes = float(attrs["time"]) if "time" in attrs else 1.0
        w = base_minutes * (multiplier if attrs.get("camera", False) else 1.0)
        G.add_edge(u, v, weight=w, **attrs)
    return G


def k_shortest_paths(G: nx.DiGraph, source: int, target: int, k: int = 5) -> List[List[int]]:
    """Compute up to k shortest paths by total weight (minutes)."""
    try:
        gen = nx.shortest_simple_paths(G, source, target, weight="weight")
        paths: List[List[int]] = []
        for i, p in enumerate(gen):
            if i >= k:
                break
            paths.append(p)
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def path_travel_time(G: nx.DiGraph, path: List[int]) -> float:
    t = 0.0
    for a, b in zip(path, path[1:]):
        t += float(G[a][b]["weight"])
    return t
