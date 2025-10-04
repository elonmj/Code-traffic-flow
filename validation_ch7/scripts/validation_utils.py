# validation_ch7/scripts/validation_utils.py
"""
Utilitaires de validation pour le Chapitre 7
Fonctions communes pour tous les tests de validation
"""

import matplotlib
matplotlib.use('Agg')  # Backend headless pour Kaggle GPU (pas de display)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ajout du chemin vers le code existant
project_root = Path(__file__).parent.parent.parent
arz_model_path = project_root / "arz_model"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(arz_model_path))  # Add arz_model directory explicitly

# Direct imports using arz_model prefix to work with Python -m execution
# This avoids relative import issues when arz_model uses internal relative imports
from arz_model.simulation.runner import SimulationRunner
from arz_model.analysis.metrics import (calculate_total_mass, compute_mape as metrics_mape, 
                              compute_rmse, compute_geh, compute_theil_u, 
                              calculate_convergence_order, analytical_riemann_solution, 
                              analytical_equilibrium_profile)
from arz_model.core.parameters import ModelParameters

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
           'ValidationTest', 'RealARZValidationTest', 'ValidationSection']


class ValidationSection:
    """
    CLASSE DE BASE ARCHITECTURALE - Une section = Un dossier organisé
    
    Définit l'architecture propre pour TOUTES les sections de validation.
    Chaque test (7.3, 7.4, 7.5, etc.) hérite de cette classe et obtient
    automatiquement la structure organisée.
    
    Architecture Standard:
    validation_output/results/{kernel_slug}/section_7_X_name/
    ├── figures/              # PNG visualisations (300 DPI)
    ├── data/
    │   ├── npz/             # Données simulation binaires
    │   ├── scenarios/       # Configurations YAML
    │   └── metrics/         # CSV/JSON métriques
    ├── latex/               # Templates LaTeX remplis
    └── session_summary.json # Metadata de session
    
    NOTE: Lors des tests locaux, output_base="validation_output/results/local_test"
          Lors de l'upload Kaggle, le manager copie depuis validation_output/
    """
    
    def __init__(self, section_name: str, output_base: str = "validation_output/results/local_test"):
        """
        Initialise l'architecture standard pour une section
        
        Args:
            section_name: Nom de la section (ex: "section_7_3_analytical")
            output_base: Dossier racine (défaut: "validation_output/results/local_test")
                        En production Kaggle: "validation_output/results/{kernel_slug}"
        """
        self.section_name = section_name
        self.output_dir = Path(output_base) / section_name
        
        # Structure organisée par TYPE de contenu
        self.figures_dir = self.output_dir / "figures"
        self.npz_dir = self.output_dir / "data" / "npz"
        self.scenarios_dir = self.output_dir / "data" / "scenarios"
        self.metrics_dir = self.output_dir / "data" / "metrics"
        self.latex_dir = self.output_dir / "latex"
        
        # Créer toute la structure automatiquement
        self._create_directories()
        
        # Initialiser le dictionnaire de résultats pour LaTeX
        self.results = {}
    
    def _create_directories(self):
        """Crée tous les dossiers de la structure organisée"""
        directories = [
            self.figures_dir,
            self.npz_dir,
            self.scenarios_dir,
            self.metrics_dir,
            self.latex_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"[ARCHITECTURE] Structure créée: {self.output_dir}")
        # Use absolute paths for display to avoid relative_to() errors on Kaggle
        print(f"  - Figures:   {self.figures_dir}")
        print(f"  - NPZ:       {self.npz_dir}")
        print(f"  - Scenarios: {self.scenarios_dir}")
        print(f"  - Metrics:   {self.metrics_dir}")
        print(f"  - LaTeX:     {self.latex_dir}")
    
    def save_session_summary(self, additional_info: dict = None):
        """
        Génère session_summary.json avec comptage d'artefacts
        
        Args:
            additional_info: Dictionnaire avec infos supplémentaires à inclure
        """
        from datetime import datetime
        import json
        
        summary = {
            'section': self.section_name,
            'timestamp': datetime.now().isoformat(),
            'artifacts': {
                'figures': len(list(self.figures_dir.glob('*.png'))),
                'npz_files': len(list(self.npz_dir.glob('*.npz'))),
                'scenarios': len(list(self.scenarios_dir.glob('*.yml'))),
                'latex_files': len(list(self.latex_dir.glob('*.tex'))),
                'csv_files': len(list(self.metrics_dir.glob('*.csv'))),
                'json_files': len(list(self.metrics_dir.glob('*.json')))
            }
        }
        
        if additional_info:
            summary.update(additional_info)
        
        summary_path = self.output_dir / 'session_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[SUMMARY] Created: {summary_path}")
        return summary


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
        
        print(f"[OK] {test_name} : SUCCES")
        return results
        
    except Exception as e:
        print(f"[ERREUR] {test_name} : ECHEC - {e}")
        return {
            'status': 'FAILED',
            'test_name': test_name,
            'error': str(e)
        }

def run_real_simulation(scenario_path, base_config_path=None, device='cpu', override_params=None):
    """
    Run real ARZ simulation using actual SimulationRunner.
    Returns structured simulation results for validation tests.
    """
    try:
        # Default base_config_path to project root
        if base_config_path is None:
            project_root = Path(__file__).parent.parent.parent
            # Try multiple possible locations for config_base.yml
            possible_paths = [
                project_root / "config" / "config_base.yml",  # Primary location
                project_root / "scenarios" / "config_base.yml",  # Alternative
                project_root / "arz_model" / "config" / "config_base.yml",  # Third option
            ]
            
            base_config_path = None
            for path in possible_paths:
                if path.exists():
                    base_config_path = str(path)
                    break
            
            if base_config_path is None:
                raise FileNotFoundError(
                    f"config_base.yml not found in any of: {[str(p) for p in possible_paths]}"
                )
        
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
    
    def __init__(self, test_name, section, scenario_path, base_config_path="scenarios/config_base.yml"):
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

def run_convergence_analysis(scenario_base_path, grid_sizes=[50, 100, 200, 400], base_config_path=None):
    """
    Run convergence analysis using real simulations with different grid resolutions.
    Returns convergence order analysis for WENO5 scheme validation.
    """
    # Default base_config_path to project root
    if base_config_path is None:
        project_root = Path(__file__).parent.parent.parent
        base_config_path = str(project_root / "scenarios" / "config_base.yml")
    
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
        
        print(f"[TABLE] Tableau genere : {csv_path}, {tex_path}")
        
        return df
        
    except Exception as e:
        print(f"[ERREUR] Erreur creation tableau : {e}")
        return None


# ============================================================================
# PUBLICATION-QUALITY PLOTTING UTILITIES FOR CHAPTER 7
# ============================================================================

def setup_publication_style():
    """Configure matplotlib pour des figures publication-ready (thèse)"""
    plt.rcParams.update({
        # Police et taille
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Résolution et qualité
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        
        # LaTeX rendering (optionnel, nécessite LaTeX installé)
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'text.latex.preamble': r'\usepackage{amsmath}',
        
        # Grille et style
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
        
        # Couleurs
        'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),
        
        # Lignes
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })


def plot_riemann_solution(x, rho_sim, v_sim, rho_exact=None, v_exact=None, 
                          case_name="Riemann Problem", output_path=None, 
                          show_analytical=True):
    """
    Génère une figure publication-ready pour un problème de Riemann.
    
    Args:
        x: array, coordonnées spatiales
        rho_sim: array, densité simulée
        v_sim: array, vitesse simulée
        rho_exact: array (optionnel), densité analytique
        v_exact: array (optionnel), vitesse analytique
        case_name: str, nom du cas pour le titre
        output_path: str/Path, chemin de sauvegarde
        show_analytical: bool, afficher solution analytique si disponible
    
    Returns:
        matplotlib.figure.Figure: Figure générée
    """
    setup_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot densité
    ax1.plot(x, rho_sim, 'b-', linewidth=2, label='Simulation ARZ-RL')
    if show_analytical and rho_exact is not None:
        ax1.plot(x, rho_exact, 'r--', linewidth=1.5, alpha=0.7, label='Solution analytique')
    ax1.set_xlabel('Position $x$ (m)')
    ax1.set_ylabel(r'Densité $\rho$ (véh/m)')
    ax1.set_title(f'{case_name} - Profil de densité')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot vitesse
    ax2.plot(x, v_sim, 'b-', linewidth=2, label='Simulation ARZ-RL')
    if show_analytical and v_exact is not None:
        ax2.plot(x, v_exact, 'r--', linewidth=1.5, alpha=0.7, label='Solution analytique')
    ax2.set_xlabel('Position $x$ (m)')
    ax2.set_ylabel('Vitesse $v$ (m/s)')
    ax2.set_title(f'{case_name} - Profil de vitesse')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[FIGURE] Saved: {output_path}")
    
    return fig


def plot_convergence_order(grid_sizes, errors, theoretical_order=5.0,
                           output_path=None, scheme_name="WENO5"):
    """
    Génère un plot log-log de l'ordre de convergence.
    
    Args:
        grid_sizes: array, tailles de grille testées
        errors: array, erreurs L2 correspondantes
        theoretical_order: float, ordre théorique attendu
        output_path: str/Path, chemin de sauvegarde
        scheme_name: str, nom du schéma numérique
    
    Returns:
        matplotlib.figure.Figure: Figure générée
    """
    setup_publication_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot des erreurs observées
    ax.loglog(grid_sizes, errors, 'bo-', linewidth=2, markersize=8, 
              label=f'{scheme_name} - Erreur observée')
    
    # Ligne théorique (pente = -theoretical_order)
    if len(grid_sizes) >= 2:
        # Calcul de la ligne théorique passant par le premier point
        N0, E0 = grid_sizes[0], errors[0]
        N_theory = np.array([grid_sizes[0], grid_sizes[-1]])
        E_theory = E0 * (N_theory / N0) ** (-theoretical_order)
        
        ax.loglog(N_theory, E_theory, 'r--', linewidth=1.5, alpha=0.7,
                 label=f'Ordre théorique $O(h^{{{theoretical_order:.0f}}})$')
    
    # Calcul de l'ordre observé moyen
    if len(grid_sizes) >= 2 and len(errors) >= 2:
        log_N = np.log(grid_sizes)
        log_E = np.log(errors)
        observed_order = -np.polyfit(log_N, log_E, 1)[0]
        
        ax.text(0.05, 0.95, f'Ordre observé: {observed_order:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Nombre de cellules $N$')
    ax.set_ylabel('Erreur $L^2$')
    ax.set_title(f'Analyse de convergence - Schéma {scheme_name}')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[FIGURE] Saved: {output_path}")
    
    return fig


def plot_fundamental_diagram(rho, v, rho_eq=None, v_eq=None,
                             output_path=None, title="Diagramme Fondamental"):
    """
    Génère un diagramme fondamental (densité vs vitesse ou débit).
    
    Args:
        rho: array, densités
        v: array, vitesses
        rho_eq: array (optionnel), densités d'équilibre théoriques
        v_eq: array (optionnel), vitesses d'équilibre théoriques
        output_path: str/Path, chemin de sauvegarde
        title: str, titre de la figure
    
    Returns:
        matplotlib.figure.Figure: Figure générée
    """
    setup_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Diagramme vitesse-densité
    ax1.scatter(rho, v, alpha=0.5, s=10, label='Simulation')
    if rho_eq is not None and v_eq is not None:
        ax1.plot(rho_eq, v_eq, 'r-', linewidth=2, label='Équilibre théorique')
    ax1.set_xlabel(r'Densité $\rho$ (véh/m)')
    ax1.set_ylabel('Vitesse $v$ (m/s)')
    ax1.set_title('Relation vitesse-densité')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Diagramme débit-densité
    q = rho * v
    ax2.scatter(rho, q, alpha=0.5, s=10, label='Simulation')
    if rho_eq is not None and v_eq is not None:
        q_eq = rho_eq * v_eq
        ax2.plot(rho_eq, q_eq, 'r-', linewidth=2, label='Équilibre théorique')
    ax2.set_xlabel(r'Densité $\rho$ (véh/m)')
    ax2.set_ylabel(r'Débit $q$ (véh/s)')
    ax2.set_title('Relation débit-densité')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[FIGURE] Saved: {output_path}")
    
    return fig


# Update __all__ to include new plotting functions
__all__.extend(['setup_publication_style', 'plot_riemann_solution', 
                'plot_convergence_order', 'plot_fundamental_diagram'])