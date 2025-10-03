# Guide de Validation Kaggle - Chapitre 7

## Architecture Intégrée : Test → NPZ + Figures + LaTeX

### Principe Fondamental
**Chaque test génère ses propres artefacts PENDANT l'exécution**, pas en post-traitement.

```
Test Python → Simulation ARZ → NPZ + PNG + LaTeX
(sur Kaggle GPU)     ↓
                Download complet
                     ↓
         validation_ch7/results/section_X/
         ├── npz/ (données brutes)
         ├── figures/ (PNG 300 DPI)
         ├── section_X_content.tex
         └── session_summary.json
```

---

## 📋 Structure des Fichiers

### Chaque section a 3 fichiers clés :

1. **`validation_ch7/scripts/test_section_7_X.py`**
   - Classe de test héritant de `RealARZValidationTest`
   - Méthodes : `test_*()` qui génèrent figures + NPZ + métriques
   - Méthode : `generate_section_content()` qui remplit le template LaTeX

2. **`validation_ch7/templates/section_7_X.tex`**
   - Template LaTeX avec placeholders `{variable}`
   - Blocs `\begin{figure}` avec chemins relatifs vers `figures/`
   - Tableaux de métriques auto-remplis

3. **`validation_ch7/scripts/validation_utils.py`**
   - Fonctions partagées : `plot_riemann_solution()`, `plot_convergence_order()`, etc.
   - Backend matplotlib configuré en mode headless (`matplotlib.use('Agg')`)

---

## 🚀 Workflow Complet

### Étape 1 : Développement Local (optionnel)
```bash
# Tester la structure (peut échouer sur config, c'est OK)
cd validation_ch7/scripts
python test_section_7_3_analytical.py
```

### Étape 2 : Upload Kaggle
```python
from validation_kaggle_manager import ValidationKaggleManager

manager = ValidationKaggleManager(
    repo_url="https://github.com/elonmj/Code-traffic-flow",
    branch="main"
)

# Upload et exécution GPU (21 minutes)
manager.run_validation_section(
    section_id="section_7_3_analytical",
    monitor_progress=True
)
```

### Étape 3 : Monitoring
Le script affiche en temps réel :
```
[INFO] Kernel created: validation-section-7-3-analytical-v1
[INFO] Kernel status: running
[TRACKING] Repository cloned
[TRACKING] Dependencies ready
[TRACKING] Validation execution finished
[TRACKING] Artifacts copied
```

### Étape 4 : Download Automatique
```
validation_ch7/results/section_7_3_analytical/validation_results/
├── npz/
│   ├── riemann_test_1_choc_simple_motos.npz
│   ├── riemann_test_2_rarefaction_voitures.npz
│   ├── riemann_test_3_vacuum_motos.npz
│   ├── riemann_test_4_contact_discontinu.npz
│   ├── riemann_test_5_multi-classes_interaction.npz
│   └── convergence_weno5.npz
├── figures/
│   ├── riemann_test_1_choc_simple_motos.png (300 DPI)
│   ├── riemann_test_2_rarefaction_voitures.png
│   ├── riemann_test_3_vacuum_motos.png
│   ├── riemann_test_4_contact_discontinu.png
│   ├── riemann_test_5_multi-classes_interaction.png
│   └── convergence_order_weno5.png
├── section_7_3_content.tex (LaTeX avec métriques remplies)
└── session_summary.json
```

---

## 🎨 Pattern de Génération de Figures

### Dans chaque méthode `test_*()` :

```python
def test_riemann_problems(self):
    """Teste 5 problèmes de Riemann avec génération de figures"""
    
    for case in self.riemann_cases:
        # 1. Créer scénario
        scenario = create_riemann_scenario_config(...)
        
        # 2. Simuler
        result = run_real_simulation(scenario, device='cuda')
        
        # 3. Extraire données
        rho_sim = result['states'][-1][0, :]
        v_sim = np.where(rho_sim > 1e-8, w_sim / rho_sim, 0.0)
        
        # 4. GÉNÉRER FIGURE (intégré au test !)
        fig = plot_riemann_solution(
            x_sim, rho_sim, v_sim,
            rho_exact, v_exact,
            case_name=case['name'],
            output_path=self.figures_dir / f"riemann_test_{i}.png"
        )
        plt.close(fig)
        
        # 5. Sauvegarder NPZ
        np.savez(self.npz_dir / f"riemann_test_{i}.npz",
                 x=x_sim, rho=rho_sim, v=v_sim, ...)
        
        # 6. Calculer métriques
        l2_error = np.linalg.norm(rho_sim - rho_exact) / np.sqrt(len(x))
        
        # 7. Stocker pour LaTeX
        self.results[f'riemann_case_{i}_l2_error'] = f"{l2_error:.2e}"
```

### Fonctions de Plotting (validation_utils.py)

```python
def plot_riemann_solution(x, rho_sim, v_sim, rho_exact, v_exact, 
                          case_name, output_path):
    """
    Génère figure 2-panel publication-ready
    - Panel gauche : Densité (simulation vs analytique)
    - Panel droit : Vitesse (simulation vs analytique)
    - Style : serif fonts, 300 DPI, grille, légendes
    """
    setup_publication_style()  # Configure rcParams
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Densité
    ax1.plot(x, rho_sim, 'b-', lw=2, label='Simulation ARZ-RL')
    ax1.plot(x, rho_exact, 'r--', lw=1.5, alpha=0.7, label='Analytique')
    ax1.set_xlabel('Position $x$ (m)')
    ax1.set_ylabel(r'Densité $\rho$ (véh/m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Vitesse (même structure)
    ...
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig
```

---

## 📊 Template LaTeX

### Structure du template `section_7_X.tex` :

```latex
\section{Validation par Solutions Analytiques}

\subsection{Problèmes de Riemann}

% Figure auto-générée
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/riemann_test_1_choc_simple_motos.png}
    \caption{Choc simple motos - Comparaison profils simulation vs analytique. 
             Erreur L2 : {riemann_case_1_l2_error}}
    \label{fig:riemann_test_1}
\end{figure}

% Tableau de métriques
\begin{table}[H]
\centering
\caption{Métriques de validation - Problèmes de Riemann}
\begin{tabular}{|l|c|c|c|}
\hline
Cas & Erreur L2 $\rho$ & Erreur L2 $v$ & Temps CPU (s) \\
\hline
Choc simple & {riemann_case_1_l2_error} & {riemann_case_1_v_error} & {riemann_case_1_runtime} \\
Raréfaction & {riemann_case_2_l2_error} & {riemann_case_2_v_error} & {riemann_case_2_runtime} \\
\hline
\end{tabular}
\end{table}
```

### Remplissage automatique :

```python
def generate_section_content(self):
    """Remplit le template et génère section_7_3_content.tex"""
    
    template_path = Path("validation_ch7/templates/section_7_3_analytical.tex")
    output_path = self.output_dir / "section_7_3_content.tex"
    
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            template_text = f.read()
        
        # Remplir avec self.results (dict avec toutes les métriques)
        generate_tex_snippet(self.results, template_text, output_path)
        
        print(f"[TEX] Generated: {output_path}")
```

---

## 🔧 Configuration Kaggle

### Prérequis :
1. **Backend Matplotlib Headless** : `matplotlib.use('Agg')` en début de fichier
2. **Chemins Absolus** : Utiliser `project_root = Path(__file__).parent.parent.parent`
3. **Directories** : Créer `figures/`, `npz/` dès l'init de la classe

### Script Kernel (généré automatiquement) :
```python
# validation_kaggle_manager.py génère automatiquement le script
# Étapes :
# 1. Clone repo GitHub
# 2. Install dependencies (PyYAML, matplotlib, pandas, scipy, numpy)
# 3. Run: python -m validation_ch7.scripts.test_section_7_3_analytical
# 4. Copy validation_ch7/results/ → /kaggle/working/validation_results/
# 5. Cleanup repo (garde seulement validation_results/)
# 6. Generate session_summary.json
```

---

## 🎯 Extension aux Sections 7.4-7.7

### Chaque section suit le MÊME pattern :

| Section | Figures Clés | Métriques |
|---------|-------------|-----------|
| **7.3 Analytique** | Riemann (5), Convergence (1) | L2 error, convergence order |
| **7.4 Calibration** | Comparaison observé/simulé | MAPE, RMSE, GEH, Theil-U |
| **7.5 Digital Twin** | Heatmaps spatio-temporels | Corrélation, erreur RMS |
| **7.6 Perf RL** | Courbes d'apprentissage | Reward moyen, stabilité |
| **7.7 Robustesse** | Impact qualité infrastructure | Écart-type vitesse, débit |

### Créer une nouvelle section :

1. **Copier template** : `test_section_7_3_analytical.py` → `test_section_7_4_calibration.py`
2. **Adapter méthodes** : Remplacer `test_riemann_problems()` par `test_calibration_comparison()`
3. **Créer fonctions plot** : Ajouter `plot_calibration_comparison()` dans `validation_utils.py`
4. **Template LaTeX** : Créer `templates/section_7_4_calibration.tex` avec nouveaux placeholders
5. **Configurer** : Ajouter section dans `VALIDATION_SECTIONS` de `validation_kaggle_manager.py`

---

## 📈 Monitoring et Debug

### Logs Kaggle :
```
[INFO] ARZ-RL VALIDATION: SECTION 7.3 - ANALYTICAL
[INFO] TRACKING_SUCCESS: Repository cloned
[INFO] TRACKING_SUCCESS: Dependencies ready
[TEST] [RIEMANN] Running test 1: Choc simple motos
[TEST] [FIGURE] Saved: figures/riemann_test_1_choc_simple_motos.png
[TEST] [NPZ] Saved: npz/riemann_test_1_choc_simple_motos.npz
[INFO] TRACKING_SUCCESS: Validation execution finished
[ARTIFACTS] PNG files: 6
[ARTIFACTS] NPZ files: 6
[INFO] TRACKING_SUCCESS: Artifacts copied
```

### Debug Local (si nécessaire) :
```python
# Vérifier structure output
import os
output_dir = Path("validation_ch7/results")
print("NPZ files:", list(output_dir.glob("npz/*.npz")))
print("Figures:", list(output_dir.glob("figures/*.png")))

# Vérifier template rempli
with open(output_dir / "section_7_3_content.tex") as f:
    content = f.read()
    # Vérifier que les placeholders sont remplis
    assert "{riemann_case_1_l2_error}" not in content
```

---

## ⚠️ Points d'Attention

### 1. **Encoding Unicode** (Windows)
- ❌ Ne PAS utiliser emojis (❌✅🚨) dans print statements
- ✅ Utiliser ASCII: `[ERREUR]`, `[OK]`, `[FIGURE]`

### 2. **Paths Relatifs vs Absolus**
- ❌ `base_config_path = "scenarios/config_base.yml"`
- ✅ `base_config_path = project_root / "scenarios" / "config_base.yml"`

### 3. **Variable Scope**
- ❌ Utiliser variable avant extraction : `plot_riemann_solution(rho_sim, ...)` puis `rho_sim = ...`
- ✅ Extraire AVANT utilisation : `rho_sim = ...; plot_riemann_solution(rho_sim, ...)`

### 4. **Matplotlib Backend**
- ❌ Pas de backend configuré → crash sur Kaggle headless
- ✅ `matplotlib.use('Agg')` en TOUT DÉBUT de fichier (avant `import matplotlib.pyplot`)

### 5. **Directory Creation**
```python
def __init__(self):
    self.output_dir = Path("validation_ch7/results")
    self.figures_dir = self.output_dir / "figures"
    self.npz_dir = self.output_dir / "npz"
    
    # CRITIQUE : Créer avant utilisation
    self.figures_dir.mkdir(parents=True, exist_ok=True)
    self.npz_dir.mkdir(parents=True, exist_ok=True)
```

---

## 🎓 Résumé : Checklist Nouveau Test

- [ ] Créer `test_section_7_X.py` avec `matplotlib.use('Agg')`
- [ ] Hériter de `RealARZValidationTest`
- [ ] Méthodes `test_*()` génèrent figures + NPZ + métriques
- [ ] Méthode `generate_section_content()` remplit template
- [ ] Créer fonctions plot dans `validation_utils.py`
- [ ] Créer template LaTeX `templates/section_7_X.tex`
- [ ] Ajouter section dans `VALIDATION_SECTIONS`
- [ ] Tester upload : `manager.run_validation_section("section_7_X")`
- [ ] Vérifier download : NPZ + Figures + LaTeX complet
- [ ] Intégrer dans thèse : `\input{validation_ch7/results/section_7_X_content.tex}`

---

## 📚 Références

- **Kernel Script Template** : `validation_kaggle_manager.py` ligne 166-440
- **Exemple Complet** : `test_section_7_3_analytical.py`
- **Fonctions Plot** : `validation_utils.py` ligne 430+
- **Template LaTeX** : `templates/section_7_3_analytical.tex`

---

**Auteur** : ValidationKaggleManager v2.0  
**Dernière MAJ** : 2025-10-03  
**Pattern** : Test-Integrated Figure Generation
