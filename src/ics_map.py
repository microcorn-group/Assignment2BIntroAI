import folium
from typing import List, Dict, Tuple, Optional
import networkx as nx
from .ics_graph import ROAD_COLORS, NODES, path_travel_time


def _edge_color(attrs: Dict) -> str:
    if attrs.get('camera'):
        return 'crimson'
    return ROAD_COLORS.get(attrs.get('type', 'secondary'), 'gray')


def _edge_style(attrs: Dict) -> Dict:
    style = {
        'color': _edge_color(attrs),
        'weight': 5 if attrs.get('type') != 'service' else 4,
        'opacity': 1.0,
    }
    if attrs.get('camera'):
        style['dashArray'] = '8,4'
        style['weight'] = 6
    return style


def render_map(
    G: nx.DiGraph,
    routes: List[List[int]],
    severity: Optional[str],
    output_html: str,
    incident_panel_html: Optional[str] = None,
) -> None:
    # Center map around average node position
    lats = [d['lat'] for _, d in G.nodes(data=True)]
    lons = [d['lon'] for _, d in G.nodes(data=True)]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))

    m = folium.Map(location=center, zoom_start=17)

    # Legend box
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: rgba(255,255,255,0.9); padding: 8px 12px; border: 1px solid #333; border-radius: 4px; font-size: 12px; line-height: 1.4;">
      <b>Kuching Heritage Graph</b><br>
      Severity: <b>{severity or 'n/a'}</b><br>
      <span style='color:green;font-weight:bold;'>●</span> START<br>
      <span style='color:blue;font-weight:bold;'>●</span> GOAL<br>
      <span style='color:crimson;font-weight:bold;'>▬ ▬</span> Camera road (accident → time ×multiplier)<br>
      <span style='color:deepskyblue;font-weight:bold;'>▬</span> Primary road<br>
      <span style='color:purple;font-weight:bold;'>▬</span> Secondary road<br>
      <span style='color:darkblue;font-weight:bold;'>▬</span> Tertiary road<br>
      <span style='color:slategray;font-weight:bold;'>▬</span> Service/local
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Optional incident info panel (e.g., image + prediction)
    if incident_panel_html:
        m.get_root().html.add_child(folium.Element(incident_panel_html))

    # Draw edges as straight lines between node coordinates
    for u, v, attrs in G.edges(data=True):
        a = G.nodes[u]
        b = G.nodes[v]
        coords = [(a['lat'], a['lon']), (b['lat'], b['lon'])]
        popup = folium.Popup(html=f"<b>{attrs.get('name','')}</b><br>way_id: {attrs.get('way_id','')}")
        folium.PolyLine(locations=coords, **_edge_style(attrs), popup=popup, tooltip=f"{attrs.get('name','')} ({attrs.get('weight'):.1f} min)").add_to(m)

    # Draw nodes
    for nid, data in G.nodes(data=True):
        color = 'white'
        if nid == 1:
            color = 'green'
        folium.CircleMarker(location=(data['lat'], data['lon']), radius=8, fill=True, color='black', fill_opacity=0.9, fill_color=color).add_to(m)
        folium.Marker(location=(data['lat'], data['lon']), icon=folium.DivIcon(html=f"<div style='font-size:10px;font-weight:bold;background-color:rgba(255,255,255,0.8);border:1px solid black;border-radius:3px;padding:2px 3px;white-space:nowrap;'>{nid}: {data['name']}</div>")).add_to(m)

    # Overlay routes
    colors = ['#2E7D32', '#FF9800', '#1976D2', '#6A1B9A', '#37474F']
    for i, path in enumerate(routes):
        latlons = [(G.nodes[n]['lat'], G.nodes[n]['lon']) for n in path]
        tmin = path_travel_time(G, path)
        folium.PolyLine(latlons, color=colors[i % len(colors)], weight=6 if i == 0 else 5, opacity=0.85, tooltip=f"Route #{i+1} — {tmin:.1f} min").add_to(m)

    m.save(output_html)
