import json

with open('depviz.json') as f:
    data = json.load(f)

edges = data.get('edges', [])
print(f"Total edges in file: {len(edges)}")

# Filter exactly like list_arz_structure.py does
arz_edges = []
for e in edges:
    from_id = e.get('from', '')
    to_id = e.get('to', '')
    # Include edge if both endpoints are in arz_model (not arz_model_gpu)
    if from_id and to_id:
        if 'arz_model/' in from_id and 'arz_model_gpu/' not in from_id:
            if 'arz_model/' in to_id and 'arz_model_gpu/' not in to_id:
                arz_edges.append(e)

print(f"Filtered arz_model edges: {len(arz_edges)}")

if len(arz_edges) > 0:
    print("\nFirst 5 edges:")
    for e in arz_edges[:5]:
        print(f"  {e['from'][:70]}")
        print(f"  -> {e['to'][:70]}\n")
