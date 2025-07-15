#!/usr/bin/env python3
"""
Script d'analyse pour la validation Phase 4.3 - WENO5 + SSP-RK3 GPU.

Analyse comparative détaillée des résultats CPU vs GPU pour validation de l'implémentation
WENO5 + SSP-RK3 sur GPU.

Usage:
    python analyze_phase43_validation.py [--input_dir output_gpu_phase43] [--plots]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from code.io.data_manager import load_simulation_data
    from code.visualization.plotting import plot_profiles, plot_spacetime
    from code.analysis.metrics import calculate_error_metrics
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    print("Continuing with basic analysis...")

def load_phase43_data(input_dir):
    """
    Charge les données de validation Phase 4.3.
    
    Args:
        input_dir (str): Dossier contenant les résultats exportés
        
    Returns:
        dict: Données chargées avec métadonnées
    """
    print(f"🔍 Chargement des données depuis: {input_dir}")
    
    # Trouver les fichiers
    cpu_files = glob.glob(os.path.join(input_dir, "cpu_*.npz"))
    gpu_files = glob.glob(os.path.join(input_dir, "gpu_*.npz"))
    metadata_files = glob.glob(os.path.join(input_dir, "validation_metadata_*.json"))
    
    if not cpu_files or not gpu_files:
        raise FileNotFoundError("Fichiers CPU ou GPU manquants dans le dossier")
    
    # Charger les données
    cpu_file = cpu_files[0]
    gpu_file = gpu_files[0] 
    
    print(f"   📁 CPU: {os.path.basename(cpu_file)}")
    print(f"   📁 GPU: {os.path.basename(gpu_file)}")
    
    cpu_data = np.load(cpu_file, allow_pickle=True)
    gpu_data = np.load(gpu_file, allow_pickle=True)
    
    # Charger les métadonnées si disponibles
    metadata = {}
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        print(f"   📋 Métadonnées: {os.path.basename(metadata_files[0])}")
    
    return {
        'cpu_data': cpu_data,
        'gpu_data': gpu_data,
        'metadata': metadata,
        'cpu_file': cpu_file,
        'gpu_file': gpu_file
    }

def analyze_convergence_accuracy(data):
    """
    Analyse de précision et convergence CPU vs GPU.
    """
    print("\n🎯 ANALYSE DE PRÉCISION CPU vs GPU")
    print("="*50)
    
    cpu_data = data['cpu_data']
    gpu_data = data['gpu_data']
    
    # Vérifier la structure des données
    print("📊 Structure des données:")
    print(f"   CPU arrays: {cpu_data.files}")
    print(f"   GPU arrays: {gpu_data.files}")
    
    # Analyser les états temporels
    if 'states' in cpu_data and 'states' in gpu_data:
        cpu_states = cpu_data['states']
        gpu_states = gpu_data['states']
        
        print(f"\n📈 États temporels:")
        print(f"   CPU shape: {cpu_states.shape}")
        print(f"   GPU shape: {gpu_states.shape}")
        
        if cpu_states.shape == gpu_states.shape:
            # Calcul des erreurs
            abs_error = np.abs(cpu_states - gpu_states)
            rel_error = abs_error / (np.abs(cpu_states) + 1e-12)
            
            max_abs_error = np.max(abs_error)
            mean_abs_error = np.mean(abs_error)
            max_rel_error = np.max(rel_error)
            mean_rel_error = np.mean(rel_error)
            
            print(f"\n🔍 Erreurs CPU vs GPU:")
            print(f"   Erreur absolue max: {max_abs_error:.2e}")
            print(f"   Erreur absolue moyenne: {mean_abs_error:.2e}")
            print(f"   Erreur relative max: {max_rel_error:.2e}")
            print(f"   Erreur relative moyenne: {mean_rel_error:.2e}")
            
            # Test de tolérance
            rtol, atol = 1e-10, 1e-12
            is_close = np.allclose(cpu_states, gpu_states, rtol=rtol, atol=atol)
            print(f"\n✅ Test np.allclose (rtol={rtol}, atol={atol}): {is_close}")
            
            return {
                'max_abs_error': max_abs_error,
                'mean_abs_error': mean_abs_error,
                'max_rel_error': max_rel_error,
                'mean_rel_error': mean_rel_error,
                'is_close': is_close
            }
        else:
            print("❌ Formes différentes entre CPU et GPU")
            return None
    else:
        print("⚠️ Données 'states' non trouvées")
        return None

def analyze_performance(data):
    """
    Analyse de performance et speedup.
    """
    print("\n🚀 ANALYSE DE PERFORMANCE")
    print("="*30)
    
    metadata = data['metadata']
    
    if 'cpu_duration' in metadata and 'gpu_duration' in metadata:
        cpu_time = metadata['cpu_duration']
        gpu_time = metadata['gpu_duration']
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"⏱️ Temps d'exécution:")
        print(f"   CPU: {cpu_time:.2f} secondes")
        print(f"   GPU: {gpu_time:.2f} secondes")
        print(f"   Speedup: {speedup:.2f}x")
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        }
    else:
        print("⚠️ Données de performance non disponibles")
        return None

def create_validation_plots(data, output_dir):
    """
    Crée les graphiques de validation.
    """
    print("\n📊 CRÉATION DES GRAPHIQUES")
    print("="*30)
    
    cpu_data = data['cpu_data']
    gpu_data = data['gpu_data']
    
    if 'states' not in cpu_data or 'states' not in gpu_data:
        print("⚠️ Impossible de créer les graphiques - données manquantes")
        return
    
    cpu_states = cpu_data['states']
    gpu_states = gpu_data['states']
    
    if cpu_states.shape != gpu_states.shape:
        print("❌ Formes incompatibles pour les graphiques")
        return
    
    # Graphique de comparaison à t final
    plt.figure(figsize=(15, 10))
    
    # Densité motos
    plt.subplot(2, 2, 1)
    plt.plot(cpu_states[-1, 0, :], 'b-', label='CPU', linewidth=2)
    plt.plot(gpu_states[-1, 0, :], 'r--', label='GPU', linewidth=1)
    plt.title('Densité motos (rho_m) - t final')
    plt.ylabel('rho_m')
    plt.legend()
    plt.grid(True)
    
    # Vitesse motos
    plt.subplot(2, 2, 2)
    plt.plot(cpu_states[-1, 1, :], 'b-', label='CPU', linewidth=2)
    plt.plot(gpu_states[-1, 1, :], 'r--', label='GPU', linewidth=1)
    plt.title('Vitesse motos (w_m) - t final')
    plt.ylabel('w_m')
    plt.legend()
    plt.grid(True)
    
    # Densité voitures
    plt.subplot(2, 2, 3)
    plt.plot(cpu_states[-1, 2, :], 'b-', label='CPU', linewidth=2)
    plt.plot(gpu_states[-1, 2, :], 'r--', label='GPU', linewidth=1)
    plt.title('Densité voitures (rho_c) - t final')
    plt.ylabel('rho_c')
    plt.xlabel('Position')
    plt.legend()
    plt.grid(True)
    
    # Vitesse voitures
    plt.subplot(2, 2, 4)
    plt.plot(cpu_states[-1, 3, :], 'b-', label='CPU', linewidth=2)
    plt.plot(gpu_states[-1, 3, :], 'r--', label='GPU', linewidth=1)
    plt.title('Vitesse voitures (w_c) - t final')
    plt.ylabel('w_c')
    plt.xlabel('Position')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'phase43_comparison_final.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 {plot_file}")
    
    # Graphique des erreurs
    plt.figure(figsize=(12, 8))
    
    error = np.abs(cpu_states - gpu_states)
    
    plt.subplot(2, 2, 1)
    plt.semilogy(error[-1, 0, :], 'r-', linewidth=2)
    plt.title('Erreur absolue rho_m')
    plt.ylabel('|CPU - GPU|')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(error[-1, 1, :], 'r-', linewidth=2)
    plt.title('Erreur absolue w_m')
    plt.ylabel('|CPU - GPU|')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(error[-1, 2, :], 'r-', linewidth=2)
    plt.title('Erreur absolue rho_c')
    plt.ylabel('|CPU - GPU|')
    plt.xlabel('Position')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.semilogy(error[-1, 3, :], 'r-', linewidth=2)
    plt.title('Erreur absolue w_c')
    plt.ylabel('|CPU - GPU|')
    plt.xlabel('Position')
    plt.grid(True)
    
    plt.tight_layout()
    error_plot_file = os.path.join(output_dir, 'phase43_errors.png')
    plt.savefig(error_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 {error_plot_file}")

def generate_report(data, accuracy_results, performance_results, output_dir):
    """
    Génère un rapport de validation complet.
    """
    print("\n📋 GÉNÉRATION DU RAPPORT")
    print("="*30)
    
    report = {
        'validation_date': datetime.now().isoformat(),
        'phase': '4.3',
        'objective': 'Validation WENO5 + SSP-RK3 GPU',
        'methodology': 'Comparaison CPU vs GPU avec métriques de précision',
        'files_analyzed': {
            'cpu': os.path.basename(data['cpu_file']),
            'gpu': os.path.basename(data['gpu_file'])
        }
    }
    
    if accuracy_results:
        report['accuracy'] = accuracy_results
        status = "PASS" if accuracy_results['is_close'] else "FAIL"
        report['validation_status'] = status
    
    if performance_results:
        report['performance'] = performance_results
    
    if data['metadata']:
        report['simulation_metadata'] = data['metadata']
    
    # Sauvegarder le rapport
    report_file = os.path.join(output_dir, 'phase43_validation_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   💾 {report_file}")
    
    # Rapport texte
    txt_report_file = os.path.join(output_dir, 'phase43_validation_summary.txt')
    with open(txt_report_file, 'w') as f:
        f.write("RAPPORT DE VALIDATION PHASE 4.3 - WENO5 + SSP-RK3 GPU\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Objectif: Validation de l'implémentation WENO5 + SSP-RK3 sur GPU\n\n")
        
        if accuracy_results:
            f.write("RÉSULTATS DE PRÉCISION:\n")
            f.write(f"  Erreur absolue max: {accuracy_results['max_abs_error']:.2e}\n")
            f.write(f"  Erreur absolue moyenne: {accuracy_results['mean_abs_error']:.2e}\n")
            f.write(f"  Erreur relative max: {accuracy_results['max_rel_error']:.2e}\n")
            f.write(f"  Erreur relative moyenne: {accuracy_results['mean_rel_error']:.2e}\n")
            f.write(f"  Test de proximité: {'PASS' if accuracy_results['is_close'] else 'FAIL'}\n\n")
        
        if performance_results:
            f.write("RÉSULTATS DE PERFORMANCE:\n")
            f.write(f"  Temps CPU: {performance_results['cpu_time']:.2f} s\n")
            f.write(f"  Temps GPU: {performance_results['gpu_time']:.2f} s\n")
            f.write(f"  Speedup: {performance_results['speedup']:.2f}x\n\n")
        
        if accuracy_results:
            status = "VALIDATION RÉUSSIE ✅" if accuracy_results['is_close'] else "VALIDATION ÉCHOUÉE ❌"
            f.write(f"STATUT FINAL: {status}\n")
    
    print(f"   💾 {txt_report_file}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Analyse de validation Phase 4.3 WENO5 + SSP-RK3")
    parser.add_argument('--input_dir', default='output_gpu_phase43', 
                       help='Dossier contenant les résultats exportés')
    parser.add_argument('--plots', action='store_true', 
                       help='Créer les graphiques de validation')
    parser.add_argument('--output_dir', default='analysis_phase43',
                       help='Dossier de sortie pour les analyses')
    
    args = parser.parse_args()
    
    print("🔬 ANALYSE VALIDATION PHASE 4.3 - WENO5 + SSP-RK3 GPU")
    print("="*60)
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Charger les données
        data = load_phase43_data(args.input_dir)
        
        # Analyser la précision
        accuracy_results = analyze_convergence_accuracy(data)
        
        # Analyser la performance
        performance_results = analyze_performance(data)
        
        # Créer les graphiques si demandé
        if args.plots:
            create_validation_plots(data, args.output_dir)
        
        # Générer le rapport
        report = generate_report(data, accuracy_results, performance_results, args.output_dir)
        
        print(f"\n🎉 ANALYSE TERMINÉE")
        print(f"   📁 Résultats dans: {args.output_dir}")
        
        if accuracy_results:
            status = "✅ PASS" if accuracy_results['is_close'] else "❌ FAIL"
            print(f"   🎯 Validation: {status}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
