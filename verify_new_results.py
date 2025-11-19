import pickle
import os

file_path = r"d:\Projets\Alibi\Code project\kaggle\results\generic-test-runner-kernel\network_simulation_results.pkl"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("Keys in results:", data.keys())

if 'history' in data:
    history = data['history']
    print("Keys in history:", history.keys())
    
    if 'traffic_lights' in history:
        tl_data = history['traffic_lights']
        print(f"Traffic lights data found for {len(tl_data)} nodes.")
        for node_id, node_data in list(tl_data.items())[:3]:
            print(f"  Node {node_id}: {list(node_data.keys())}")
            if 'green_segments' in node_data:
                print(f"    First 5 steps: {node_data['green_segments'][:5]}")
    else:
        print("❌ 'traffic_lights' key NOT found in history.")
else:
    print("❌ 'history' key NOT found in results.")
