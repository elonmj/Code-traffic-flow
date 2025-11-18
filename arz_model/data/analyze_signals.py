"""Analyse rapide des feux tricolores détectés"""
import pandas as pd

df = pd.read_excel('fichier_de_travail_complet_enriched.xlsx')

print("=" * 70)
print("📊 ANALYSE DES FEUX TRICOLORES DÉTECTÉS PAR OSM")
print("=" * 70)

# Statistiques générales
print(f"\n📈 STATISTIQUES GÉNÉRALES:")
print(f"   • Total segments: {len(df)}")
print(f"   • Segments avec feux à u: {df['u_has_signal'].sum()}")
print(f"   • Segments avec feux à v: {df['v_has_signal'].sum()}")

# Nœuds uniques avec feux
signals_u = df[df['u_has_signal'] == True]
signals_v = df[df['v_has_signal'] == True]
unique_u = set(signals_u['u'].values)
unique_v = set(signals_v['v'].values)
all_signal_nodes = unique_u.union(unique_v)

print(f"\n🚦 NŒUDS UNIQUES AVEC FEUX:")
print(f"   • Total: {len(all_signal_nodes)}")
print(f"   • IDs: {sorted(all_signal_nodes)}")

# Rues concernées
print(f"\n🛣️  RUES AVEC FEUX TRICOLORES:")
streets_u = signals_u['name_clean'].dropna().unique()
streets_v = signals_v['name_clean'].dropna().unique()
all_streets = set(streets_u) | set(streets_v)
for street in sorted(all_streets):
    count_u = len(signals_u[signals_u['name_clean'] == street])
    count_v = len(signals_v[signals_v['name_clean'] == street])
    print(f"   • {street}: {count_u + count_v} occurrences")

# Détails techniques
print(f"\n🔧 DÉTAILS TECHNIQUES:")
print(f"\n   Colonnes OSM ajoutées:")
osm_cols = [c for c in df.columns if c.startswith('u_') or c.startswith('v_')]
for col in sorted(osm_cols):
    non_null = df[col].notna().sum()
    if non_null > 0:
        print(f"      - {col}: {non_null} valeurs non-nulles")

# Exemples de métadonnées feux
print(f"\n📝 EXEMPLES DE MÉTADONNÉES (premiers 5):")
sample = df[df['u_has_signal'] == True].head()
for idx, row in sample.iterrows():
    print(f"\n   Segment {idx}: {row['name_clean']}")
    print(f"      Node ID: {row['u']}")
    print(f"      Highway tag: {row['u_highway_tag']}")
    print(f"      Signal tags: {row['u_signal_tags']}")
    print(f"      Junction: {row['u_junction_type']}")
    print(f"      Coords: ({row['u_lat']:.6f}, {row['u_lon']:.6f})")

print("\n" + "=" * 70)
