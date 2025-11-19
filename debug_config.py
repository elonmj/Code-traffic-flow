import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from arz_model.config.config_factory import create_victoria_island_config

def debug_config_generation():
    data_dir = os.path.join(project_root, 'arz_model', 'data')
    enriched_path = os.path.join(data_dir, 'fichier_de_travail_corridor_enriched.xlsx')
    
    print(f"Enriched path: {enriched_path}")
    print(f"Exists: {os.path.exists(enriched_path)}")
    
    try:
        config = create_victoria_island_config(
            enriched_path=enriched_path,
            use_cache=False # Force regeneration
        )
        
        signalized_count = 0
        for node in config.nodes:
            if node.type == 'signalized':
                signalized_count += 1
                print(f"Signalized node found: {node.id}")
                if node.traffic_light_config:
                    print(f"  Config: {node.traffic_light_config}")
        
        print(f"Total signalized nodes: {signalized_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config_generation()
