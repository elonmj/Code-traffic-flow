import json

data = json.load(open('depviz.json'))

# Check what kinds of nodes we have
all_nodes = data['nodes']
kinds = {}
for n in all_nodes:
    kind = n.get('kind', 'NO_KIND')
    if kind not in kinds:
        kinds[kind] = 0
    kinds[kind] += 1

print("Node kinds found:")
for kind, count in sorted(kinds.items()):
    print(f"  {kind}: {count}")

# Show sample of each kind
print("\nSample IDs for each kind:")
for kind in sorted(kinds.keys()):
    samples = [n for n in all_nodes if n.get('kind') == kind][:2]
    print(f"\n{kind}:")
    for s in samples:
        print(f"  ID: {s.get('id', 'NO_ID')[:80]}")
        print(f"  Label: {s.get('label', 'NO_LABEL')}")


