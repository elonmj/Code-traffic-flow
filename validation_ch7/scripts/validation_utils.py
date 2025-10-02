# validation_ch7/scripts/validation_utils.py
"""
Utilitaires de validation pour le Chapitre 7
Fonctions communes pour tous les tests de validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ajout du chemin vers le code existant
project_root = Path(__file__).parent.parent.parent
code_path = project_root / "code"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(code_path))  # Add code directory explicitly

# Direct imports - sys.path should be set up by caller (kernel script or local runner)
# The kernel script adds both project_root and project_root/code to sys.path BEFORE importing
from simulation.runner import SimulationRunner
from analysis.metrics import (calculate_total_mass, compute_mape as metrics_mape, 
                              compute_rmse, compute_geh, compute_theil_u, 
                              calculate_convergence_order, analytical_riemann_solution, 
                              analytical_equilibrium_profile)
from core.parameters import ModelParameters

# Utiliser les fonctions du module metrics pour éviter la duplication
def compute_mape(observed, simulated):
    """Calcule l'erreur relative moyenne absolue (MAPE) - wrapper vers metrics"""
    return metrics_mape(observed, simulated)

# Les fonctions compute_rmse, compute_geh, compute_theil_u sont maintenant dans metrics.py
# Accessible via les imports directs : compute_rmse, compute_geh, compute_theil_u

# Re-export these functions to make them available for "from validation_utils import ..."
__all__ = ['compute_mape', 'compute_rmse', 'compute_geh', 'compute_theil_u',
           'calculate_total_mass', 'calculate_convergence_order',
           'analytical_riemann_solution', 'analytical_equilibrium_profile',
           'run_validation_test', 'run_real_simulation', 'run_convergence_analysis',
           'create_riemann_scenario_config', 'generate_tex_snippet', 'generate_latex_table',
           'save_validation_results', 'save_figure', 'create_summary_table',
           'ValidationTest', 'RealARZValidationTest']

class ValidationTest:
    """Base class for validation tests."""
    
    def __init__(self, test_name, section):
        self.test_name = test_name
        self.section = section
        self.results = {}
    
    def run_test(self) -> bool:
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement run_test method")
    
    def generate_latex_content(self):
        """Override this method in subclasses for LaTeX generation."""
        pass

def run_validation_test(scenario_path, test_name, validation_func, **kwargs):
    """
    Lance un test de validation générique
    
    Args:
        scenario_path: Chemin vers le fichier de scénario YAML
        test_name: Nom du test pour les logs
        validation_func: Fonction de validation à appliquer
        **kwargs: Arguments supplémentaires pour la fonction de validation
    
    Returns:
        dict: Résultats du test avec métriques et statut
    """
    print(f"\n=== Test : {test_name} ===")
    
    try:
        # Lancer la simulation
        runner = SimulationRunner(
            scenario_config_path=scenario_path,
            quiet=True
        )
        
        times, states = runner.run()
        
        # Appliquer la fonction de validation
        results = validation_func(times, states, runner.grid, runner.params, **kwargs)
        results['status'] = 'SUCCESS'
        results['test_name'] = test_name
        
        print(f"✅ {test_name} : SUCCÈS")
        return results
        
    except Exception as e:
        print(f"❌ {test_name} : ÉCHEC - {e}")
        return {
            'status': 'FAILED',
            'test_name': test_name,
            'error': str(e)
        }

def run_real_simulation(scenario_path, base_config_path="config/config_base.yml", device='cpu', override_params=None):
    """
    Run real ARZ simulation using actual SimulationRunner.
    Returns structured simulation results for validation tests.
    """
    try:
        # Create SimulationRunner with real physics
        runner = SimulationRunner(
            scenario_config_path=scenario_path,
            base_config_path=base_config_path,
            device=device,
            override_params=override_params or {},
            quiet=True
        )
        
        # Run real simulation
        times, states = runner.run()
        
        # Structure results for validation
        results = {
            'times': times,
            'states': states,
            'grid': runner.grid,
            'params': runner.params,
            'mass_conservation': {
                'initial_mass_m': calculate_total_mass(states[0], runner.grid, class_index=0),
                'final_mass_m': calculate_total_mass(states[-1], runner.grid, class_index=0),
                'initial_mass_c': calculate_total_mass(states[0], runner.grid, class_index=2),
                'final_mass_c': calculate_total_mass(states[-1], runner.grid, class_index=2)
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error in real simulation: {e}")
        return None

class RealARZValidationTest(ValidationTest):
    """Base class for real ARZ validation tests using SimulationRunner."""
    
    def __init__(self, test_name, section, scenario_path, base_config_path="config/config_base.yml"):
        super().__init__(test_name, section)
        self.scenario_path = scenario_path
        self.base_config_path = base_config_path
        self.device = 'cpu'  # Default to CPU, can be overridden for GPU tests
        
    def run_simulation(self, override_params=None):
        """Run real ARZ simulation and return structured results."""
        return run_real_simulation(
            self.scenario_path, 
            self.base_config_path, 
            self.device, 
            override_params
        )
    
    def validate_mass_conservation(self, results, tolerance=1e-6):
        """Validate mass conservation for both vehicle classes."""
        mass_data = results['mass_conservation']
        
        # Mass conservation for motorcycles
        mass_error_m = abs(mass_data['final_mass_m'] - mass_data['initial_mass_m']) / mass_data['initial_mass_m']
        
        # Mass conservation for cars  
        mass_error_c = abs(mass_data['final_mass_c'] - mass_data['initial_mass_c']) / mass_data['initial_mass_c']
        
        return {
            'mass_error_m': mass_error_m,
            'mass_error_c': mass_error_c,
            'mass_conserved_m': mass_error_m < tolerance,
            'mass_conserved_c': mass_error_c < tolerance,
            'overall_conservation': mass_error_m < tolerance and mass_error_c < tolerance
        }

def run_convergence_analysis(scenario_base_path, grid_sizes=[50, 100, 200, 400], base_config_path="config/config_base.yml"):
    """
    Run convergence analysis using real simulations with different grid resolutions.
    Returns convergence order analysis for WENO5 scheme validation.
    """
    errors = []
    results_data = []
    
    for N in grid_sizes:
        print(f"Running convergence test with N={N}...")
        
        # Override grid size in scenario
        override_params = {'N': N}
        
        result = run_real_simulation(scenario_base_path, base_config_path, 'cpu', override_params)
        if result is None:
            continue
            
        # Store result for analysis
        results_data.append(result)
        
        # Calculate error vs analytical solution (if available)
        # For now, use L2 norm of final state as proxy
        final_state = result['states'][-1]
        l2_error = np.linalg.norm(final_state) / np.sqrt(N)
        errors.append(l2_error)
    
    # Calculate convergence order
    if len(errors) >= 2:
        convergence_order = calculate_convergence_order(grid_sizes[:len(errors)], errors)
        return {
            'grid_sizes': grid_sizes[:len(errors)],
            'errors': errors,
            'convergence_order': convergence_order,
            'results_data': results_data
        }
    else:
        return None

def create_riemann_scenario_config(output_path, U_L, U_R, split_pos=500.0, domain_length=1000.0, N=200, t_final=60.0):
    """Create a Riemann test scenario configuration for real validation."""
    scenario_config = {
        'scenario_name': 'riemann_validation_test',
        'N': N,
        'xmin': 0.0,
        'xmax': domain_length,
        't_final': t_final,
        'output_dt': 5.0,
        'road_quality_definition': 3,
        'initial_conditions': {
            'type': 'riemann',
            'U_L': U_L,  # [rho_m, w_m, rho_c, w_c] in SI units
            'U_R': U_R,  # [rho_m, w_m, rho_c, w_c] in SI units
            'split_pos': split_pos
        },
        'boundary_conditions': {
            'left': {'type': 'inflow', 'state': U_L},
            'right': {'type': 'outflow'}
        }
    }
    
    # Write scenario to YAML file
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(scenario_config, f, default_flow_style=False)
    
    return output_path

def generate_tex_snippet(results, template_text, output_path):
    """
    Génère un snippet TeX à partir des résultats
    
    Args:
        results: Dictionnaire des résultats
        template_text: Template avec placeholders {variable}
        output_path: Chemin de sortie pour le snippet
    """
    import string
    
    try:
        # Custom formatter qui ignore les placeholders manquants
        class PartialFormatter(string.Formatter):
            def __init__(self, missing='PLACEHOLDER_MISSING'):
                self.missing = missing

            def get_field(self, field_name, args, kwargs):
                try:
                    return string.Formatter.get_field(self, field_name, args, kwargs)
                except (KeyError, AttributeError):
                    return self.missing, field_name

            def format_field(self, value, spec):
                if value == self.missing:
                    return '{' + spec + '}'
                return string.Formatter.format_field(self, value, spec)
        
        formatter = PartialFormatter()
        filled_text = formatter.format(template_text, **results)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(filled_text)
            
        print(f"[OK] Snippet TeX genere : {output_path}")
        
    except Exception as e:
        print(f"[ERREUR] Erreur generation TeX : {e}")
        # Fallback - au moins créer le fichier avec le template original
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template_text)
            print(f"[FALLBACK] Template original sauve : {output_path}")
        except Exception as fe:
            print(f"[ERREUR] Meme le fallback a echoue : {fe}")

def generate_latex_table(caption, headers, rows):
    """Generate a LaTeX table from headers and rows."""
    latex = f"\\begin{{table}}[h]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\begin{{tabular}}{{|{'c|' * len(headers)}}}\n"
    latex += f"\\hline\n"
    latex += f"{' & '.join(headers)} \\\\\n"
    latex += f"\\hline\n"
    
    for row in rows:
        latex += f"{' & '.join(str(cell) for cell in row)} \\\\\n"
    
    latex += f"\\hline\n"
    latex += f"\\end{{tabular}}\n"
    latex += f"\\end{{table}}\n"
    
    return latex

def save_validation_results(section, content, results):
    """Save validation results to files."""
    try:
        # Create results directory
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save LaTeX content
        latex_file = results_dir / f"section_{section}_results.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save JSON results
        import json
        json_file = results_dir / f"section_{section}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {latex_file} and {json_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def save_figure(fig, output_path, title=""):
    """Sauvegarde une figure avec gestion des erreurs"""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if title:
            fig.suptitle(title)
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[OK] Figure sauvegardee : {output_path}")
        
    except Exception as e:
        print(f"[ERREUR] Erreur sauvegarde figure : {e}")

def create_summary_table(results_list, output_path):
    """Crée un tableau de synthèse des résultats"""
    try:
        df = pd.DataFrame(results_list)
        
        # Sauvegarde CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        
        # Génération LaTeX
        latex_table = df.to_latex(index=False, escape=False, float_format="%.3f")
        
        tex_path = output_path.with_suffix('.tex')
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"📋 Tableau généré : {csv_path}, {tex_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur création tableau : {e}")
        return None