import pandas as pd
import requests
from pathlib import Path

excel_path = Path("fichier_de_travail_corridor.xlsx")
df = pd.read_excel(excel_path)
raw_nodes = set(df['u'].dropna().astype(int)).union(set(df['v'].dropna().astype(int)))
node_ids = sorted(n for n in raw_nodes if n > 0)
print(f"Total unique positive nodes: {len(node_ids)} / {len(raw_nodes)} total")
url = "https://overpass-api.de/api/interpreter"
node_data = {}
chunk_size = 50
for i in range(0, len(node_ids), chunk_size):
    chunk = node_ids[i:i+chunk_size]
    ids_str = ",".join(str(n) for n in chunk)
    query = f"[out:json];node(id:{ids_str});out body;"
    resp = requests.get(url, params={'data': query}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    for element in data.get('elements', []):
        node_data[element['id']] = {
            'lat': element.get('lat'),
            'lon': element.get('lon'),
            'tags': element.get('tags', {})
        }
print(f"Retrieved metadata for {len(node_data)} nodes")

missing_nodes = set(node_ids) - set(node_data.keys())
if missing_nodes:
    print(f"Warning: {len(missing_nodes)} OSM nodes missing metadata (likely deleted/hidden)")
else:
    print("All positive nodes resolved in OSM")


def extract(node_id):
    if node_id <= 0:
        return (False, None, None, None, None, None, None, None, None, None)
    info = node_data.get(int(node_id))
    if not info:
        return (False, None, None, None, None, None, None, None, None, None)
    tags = info.get('tags', {})
    has_signal = tags.get('highway') == 'traffic_signals' or bool(tags.get('traffic_signals'))
    if not has_signal:
        has_signal = any(k.startswith('traffic_signals:') for k in tags)
    junction = tags.get('junction')
    signal_detail = "; ".join(f"{k}={v}" for k, v in tags.items() if k.startswith('traffic_signals')) or None
    phases = tags.get('traffic_signals:phases') or tags.get('traffic_signals:multiphase')
    controller = tags.get('traffic_signals:direction') or tags.get('traffic_signals:control')
    crossing = tags.get('crossing')
    traffic_calming = tags.get('traffic_calming')
    highway_tag = tags.get('highway')
    return (
        has_signal,
        junction,
        signal_detail,
        phases,
        controller,
        crossing,
        traffic_calming,
        highway_tag,
        info.get('lat'),
        info.get('lon')
    )

u_info = df['u'].apply(lambda x: extract(int(x)))
v_info = df['v'].apply(lambda x: extract(int(x)))

u_cols = list(zip(*u_info))
v_cols = list(zip(*v_info))

(df['u_has_signal'], df['u_junction_type'], df['u_signal_tags'], df['u_signal_phases'], df['u_signal_controller'],
 df['u_crossing_type'], df['u_traffic_calming'], df['u_highway_tag'], df['u_lat'], df['u_lon']) = u_cols
(df['v_has_signal'], df['v_junction_type'], df['v_signal_tags'], df['v_signal_phases'], df['v_signal_controller'],
 df['v_crossing_type'], df['v_traffic_calming'], df['v_highway_tag'], df['v_lat'], df['v_lon']) = v_cols

out_path = Path("fichier_de_travail_corridor_enriched.xlsx")
df.to_excel(out_path, index=False)
print(f"Saved enriched file to {out_path}")
