"""
Script pour inspecter le contenu des fichiers cache et valider leur structure.
"""
import pickle
from pathlib import Path
from datetime import datetime

def inspect_cache_file(cache_path: Path):
    """Inspecte un fichier cache et affiche son contenu."""
    print(f"\n{'='*80}")
    print(f"📦 INSPECTION: {cache_path.name}")
    print(f"{'='*80}")
    
    if not cache_path.exists():
        print("❌ Fichier n'existe pas!")
        return
    
    # Taille du fichier
    size_bytes = cache_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"📏 Taille: {size_bytes:,} bytes ({size_mb:.2f} MB)")
    
    # Chargement et inspection
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"\n🔑 Clés du cache:")
        for key in cache_data.keys():
            print(f"   - {key}")
        
        print(f"\n📊 Détails du cache:")
        for key, value in cache_data.items():
            if key == 'states_history':
                print(f"   • {key}: {len(value)} états")
                if value:
                    first_state = value[0]
                    print(f"     └─ Shape premier état: {first_state.shape}")
                    print(f"     └─ Dtype: {first_state.dtype}")
            elif key == 'timestamp':
                dt = datetime.fromisoformat(value)
                print(f"   • {key}: {value}")
                print(f"     └─ Créé il y a: {(datetime.now() - dt).total_seconds():.0f}s")
            else:
                print(f"   • {key}: {value}")
        
        # Validation de la structure
        print(f"\n✅ VALIDATIONS:")
        required_keys = ['states_history', 'scenario_path', 'cache_version', 'timestamp']
        for key in required_keys:
            status = "✅" if key in cache_data else "❌"
            print(f"   {status} {key} présent")
        
        # Vérification version
        if cache_data.get('cache_version') == '1.0':
            print(f"   ✅ Version cache valide (1.0)")
        else:
            print(f"   ⚠️  Version cache: {cache_data.get('cache_version')}")
        
        return cache_data
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

if __name__ == "__main__":
    # Chemin vers le répertoire cache
    project_root = Path(__file__).parent
    cache_dir = project_root / "validation_ch7" / "cache" / "section_7_6"
    
    print(f"🔍 INSPECTION DES FICHIERS CACHE")
    print(f"📁 Répertoire: {cache_dir}")
    
    # Liste tous les fichiers .pkl
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"\n📦 {len(cache_files)} fichier(s) cache trouvé(s):")
    for cache_file in cache_files:
        print(f"   - {cache_file.name}")
    
    # Inspection de chaque fichier
    for cache_file in cache_files:
        cache_data = inspect_cache_file(cache_file)
    
    print(f"\n{'='*80}")
    print(f"✅ INSPECTION TERMINÉE")
    print(f"{'='*80}")
