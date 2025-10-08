"""
Analyse des événements TensorBoard générés par les entraînements RL.
"""
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json

# Répertoire TensorBoard
TB_DIR = 'validation_output/results/elonmj_arz-validation-76rlperformance-pmrk/section_7_6_rl_performance/data/models/tensorboard/'

print("=" * 80)
print("ANALYSE DES ÉVÉNEMENTS TENSORBOARD")
print("=" * 80)

# Lister les runs
if not os.path.exists(TB_DIR):
    print(f"\n❌ ERROR: Directory not found: {TB_DIR}")
    exit(1)

runs = [d for d in os.listdir(TB_DIR) if os.path.isdir(os.path.join(TB_DIR, d))]
print(f"\n📂 RUNS DÉTECTÉS: {len(runs)}")
for run in runs:
    print(f"   - {run}")

# Analyser chaque run
results = {}

for run_name in runs:
    print(f"\n{'=' * 80}")
    print(f"RUN: {run_name}")
    print(f"{'=' * 80}")
    
    run_dir = os.path.join(TB_DIR, run_name)
    
    # Trouver le fichier d'événements
    event_files = [f for f in os.listdir(run_dir) if f.startswith('events')]
    if not event_files:
        print(f"  ⚠️  No event files found")
        continue
    
    event_file = event_files[0]
    event_path = os.path.join(run_dir, event_file)
    
    print(f"\n📄 Event file: {event_file}")
    
    # Charger avec EventAccumulator
    ea = EventAccumulator(event_path)
    ea.Reload()
    
    # Afficher les tags disponibles
    tags = ea.Tags()
    print(f"\n📊 Tags disponibles:")
    print(f"   - Scalars: {len(tags['scalars'])} tags")
    print(f"   - Images: {len(tags['images'])} tags")
    print(f"   - Histograms: {len(tags['histograms'])} tags")
    
    # Analyser les scalars
    if tags['scalars']:
        print(f"\n📈 SCALAR METRICS:")
        results[run_name] = {}
        
        for tag in tags['scalars']:
            scalar_events = ea.Scalars(tag)
            print(f"\n   🔹 {tag}:")
            print(f"      Nombre de points: {len(scalar_events)}")
            
            if scalar_events:
                values = [e.value for e in scalar_events]
                steps = [e.step for e in scalar_events]
                
                print(f"      Premier step: {steps[0]}")
                print(f"      Dernier step: {steps[-1]}")
                print(f"      Première valeur: {values[0]:.4f}")
                print(f"      Dernière valeur: {values[-1]:.4f}")
                
                # Sauvegarder pour analyse
                results[run_name][tag] = {
                    'steps': steps,
                    'values': values,
                    'num_points': len(scalar_events)
                }
                
                # Afficher tous les points si peu de données
                if len(scalar_events) <= 5:
                    print(f"      Tous les points:")
                    for e in scalar_events:
                        print(f"         Step {e.step}: {e.value:.6f}")
    else:
        print("   ⚠️  No scalar metrics found")

# Résumé comparatif
print(f"\n{'=' * 80}")
print("RÉSUMÉ COMPARATIF DES 3 RUNS")
print(f"{'=' * 80}")

if results:
    # Créer un tableau comparatif
    print(f"\n{'Metric':<30} | {'PPO_1':<15} | {'PPO_2':<15} | {'PPO_3':<15}")
    print("-" * 80)
    
    # Pour chaque métrique commune
    all_metrics = set()
    for run_data in results.values():
        all_metrics.update(run_data.keys())
    
    for metric in sorted(all_metrics):
        row = f"{metric:<30}"
        for run_name in ['PPO_1', 'PPO_2', 'PPO_3']:
            if run_name in results and metric in results[run_name]:
                data = results[run_name][metric]
                final_value = data['values'][-1] if data['values'] else 0.0
                row += f" | {final_value:>15.4f}"
            else:
                row += f" | {'N/A':>15}"
        print(row)

# Sauvegarder les résultats
output_file = "tensorboard_analysis.json"
with open(output_file, 'w') as f:
    # Convertir pour JSON (enlever les numpy arrays)
    json_results = {}
    for run_name, run_data in results.items():
        json_results[run_name] = {}
        for metric, metric_data in run_data.items():
            json_results[run_name][metric] = {
                'steps': metric_data['steps'],
                'values': metric_data['values'],
                'num_points': metric_data['num_points']
            }
    json.dump(json_results, f, indent=2)

print(f"\n✅ Analyse complète sauvegardée dans: {output_file}")

# Interprétation
print(f"\n{'=' * 80}")
print("INTERPRÉTATION")
print(f"{'=' * 80}")

print("""
📌 QUICK TEST (2 timesteps) - LIMITES:

Les 3 runs ont été exécutés avec QUICK_TEST=1 (seulement 2 timesteps).
Cela signifie qu'il n'y a pratiquement AUCUNE donnée d'apprentissage.

🔍 Ce que montre chaque métrique:

• rollout/ep_rew_mean: Récompense moyenne par épisode
  → Avec 2 timesteps, c'est juste l'initialisation (pas d'apprentissage)
  
• rollout/ep_len_mean: Longueur moyenne des épisodes
  → Devrait être ~2 (les 2 timesteps du quick test)
  
• time/fps: Vitesse d'entraînement (frames per second)
  → Indicateur de performance du système (GPU vs CPU)

⚠️  POUR VOIR L'APPRENTISSAGE RÉEL:

Vous devez lancer un entraînement COMPLET (20,000+ timesteps) pour observer:
  - Convergence de la récompense
  - Amélioration progressive
  - Stabilisation du policy

💡 UTILISATION DE TENSORBOARD UI:

Pour visualiser graphiquement:
  tensorboard --logdir=validation_output/results/.../tensorboard/
  
Puis ouvrir: http://localhost:6006
""")

print("\n✅ Analyse terminée!")
