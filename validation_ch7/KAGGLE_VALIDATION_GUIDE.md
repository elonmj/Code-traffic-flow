# Guide de Validation Kaggle - Chapitre 7

## 🏗️ Architecture Propre et Organisée (v2.0)

### Principe Fondamental
**Une section = Un dossier organisé par type de contenu**

Chaque test génère ses artefacts PENDANT l'exécution dans une structure claire et navigable.

```
Test Python → Simulation ARZ → Structure Organisée
(sur Kaggle GPU)     ↓
                Download complet
                     ↓
         validation_output/results/{kernel_slug}/
         └── section_7_3_analytical/
             ├── figures/              # Visualisations PNG (300 DPI)
             ├── data/
             │   ├── npz/             # Données simulation
             │   ├── scenarios/       # Config YAML
             │   └── metrics/         # CSV/JSON métriques
             ├── latex/               # Template rempli
             │   └── section_7_3_content.tex
             └── session_summary.json # Metadata section
```

### Avantages de Cette Architecture

✅ **Navigabilité** : Chaque type de fichier a son dossier dédié
✅ **Pas de pollution** : Section 7.3 ne contient QUE les fichiers 7.3
✅ **Pas de duplication** : Un seul dossier `validation_output/` (pas de `validation_ch7/results/`)
✅ **Scalable** : Facile d'ajouter sections 7.4-7.7 sans conflit
✅ **Git-friendly** : Structure claire pour versioning

---

## 📋 Structure des Fichiers

### Chaque section a 3 fichiers clés :

1. **`validation_ch7/scripts/test_section_7_X.py`**
   - Classe de test avec structure organisée dans `__init__()`
   - Crée automatiquement : `figures/`, `data/npz/`, `data/scenarios/`, `data/metrics/`, `latex/`
   - Méthodes : `test_*()` qui génèrent ET sauvegardent dans les bons dossiers
   - Méthode : `generate_section_content()` qui crée `session_summary.json`

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
# Génère dans: validation_ch7/results/section_7_3_analytical/
```

### Étape 2 : Upload Kaggle
```python
from validation_kaggle_manager import ValidationKaggleManager

manager = ValidationKaggleManager(
    repo_url="https://github.com/elonmj/Code-traffic-flow",
    branch="main"
)

# Upload et exécution GPU (~21 minutes)
manager.run_validation_section(
    section_id="section_7_3_analytical",
    monitor_progress=True
)
```

**Ce qui se passe automatiquement :**
1. Kernel créé avec GPU P100
2. Repository cloné + dependencies installées
3. **Artefacts locaux copiés** : `validation_ch7/results/section_7_3_analytical/` → `/kaggle/working/section_7_3_analytical/`
   - Préserve structure organisée : `figures/`, `data/`, `latex/`
4. **Tests exécutés → Résultats ajoutés dans structure existante**
5. Kernel push vers Kaggle

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
validation_output/results/{kernel_slug}/
└── section_7_3_analytical/
    ├── figures/
    │   ├── riemann_test_1_choc_simple_motos.png (300 DPI)
    │   ├── riemann_test_2_rarefaction_voitures.png
    │   ├── riemann_test_3_vacuum_motos.png
    │   ├── riemann_test_4_contact_discontinu.png
    │   ├── riemann_test_5_multi-classes_interaction.png
    │   └── convergence_order_weno5.png
    ├── data/
    │   ├── npz/
    │   │   ├── riemann_test_1_choc_simple_motos.npz
    │   │   ├── riemann_test_2_rarefaction_voitures.npz
    │   │   ├── riemann_test_3_vacuum_motos.npz
    │   │   ├── riemann_test_4_contact_discontinu.npz
    │   │   ├── riemann_test_5_multi-classes_interaction.npz
    │   │   └── convergence_weno5.npz
    │   ├── scenarios/
    │   │   ├── riemann_test_1.yml
    │   │   ├── riemann_test_2.yml
    │   │   ├── riemann_test_3.yml
    │   │   ├── riemann_test_4.yml
    │   │   └── riemann_test_5.yml
    │   └── metrics/
    │       ├── riemann_validation_metrics.csv
    │       └── convergence_metrics.json
    ├── latex/
    │   └── section_7_3_content.tex (template rempli)
    └── session_summary.json (metadata)

---

## 🎨 Pattern de Génération de Figures

### Dans chaque méthode `test_*()` :

```python
def test_riemann_problems(self):
    """Teste 5 problèmes de Riemann avec génération de figures"""
    
    for i, case in enumerate(self.riemann_cases, 1):
        # 1. Charger scénario YAML
        scenario_path = config_dir / case['scenario_file']
        
        # 2. Copier scénario dans data/scenarios/ pour archivage
        shutil.copy2(scenario_path, self.scenarios_dir / f"riemann_test_{i}.yml")
        
        # 3. Simuler
        result = run_real_simulation(scenario, device='cuda')
        
        # 4. Extraire données
        rho_sim = result['states'][-1][0, :]
        v_sim = np.where(rho_sim > 1e-8, w_sim / rho_sim, 0.0)
        
        # 5. GÉNÉRER FIGURE dans figures/ (intégré au test !)
        fig = plot_riemann_solution(
            x_sim, rho_sim, v_sim,
            rho_exact, v_exact,
            case_name=case['name'],
            output_path=self.figures_dir / f"riemann_test_{i}.png"  # → figures/
        )
        plt.close(fig)
        
        # 6. Sauvegarder NPZ dans data/npz/
        np.savez(self.npz_dir / f"riemann_test_{i}.npz",
                 x=x_sim, rho=rho_sim, v=v_sim, ...)
        
        # 7. Calculer métriques
        l2_error = np.linalg.norm(rho_sim - rho_exact) / np.sqrt(len(x))
        
        # 8. Stocker pour LaTeX
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
    output_path = self.latex_dir / "section_7_3_content.tex"  # → latex/
    
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            template_text = f.read()
        
        # Remplir avec self.results (dict avec toutes les métriques)
        generate_tex_snippet(self.results, template_text, output_path)
        
        print(f"[TEX] Generated: {output_path}")
    
    # Sauvegarder métriques CSV dans data/metrics/
    metrics_csv = self.metrics_dir / "riemann_validation_metrics.csv"
    pd.DataFrame(self.results, index=[0]).to_csv(metrics_csv, index=False)
    
    # Créer session summary JSON à la racine de la section
    summary = {
        'section': 'section_7_3_analytical',
        'timestamp': datetime.now().isoformat(),
        'artifacts': {
            'figures': len(list(self.figures_dir.glob('*.png'))),
            'npz_files': len(list(self.npz_dir.glob('*.npz'))),
            'scenarios': len(list(self.scenarios_dir.glob('*.yml'))),
            'latex_files': len(list(self.latex_dir.glob('*.tex'))),
            'metrics_files': len(list(self.metrics_dir.glob('*.csv')))
        }
    }
    (self.output_dir / 'session_summary.json').write_text(json.dumps(summary, indent=2))
```

---

## 🔧 Configuration Kaggle

### Prérequis :
1. **Backend Matplotlib Headless** : `matplotlib.use('Agg')` en début de fichier
2. **Chemins Absolus** : Utiliser `project_root = Path(__file__).parent.parent.parent`
3. **Directories Organisées** : Classe de test crée `figures/`, `data/npz/`, `data/scenarios/`, `data/metrics/`, `latex/` dans `__init__()`

### Script Kernel (généré automatiquement par `validation_kaggle_manager.py`) :
```python
# ÉTAPES AUTOMATIQUES :
# 1. Clone repo GitHub
# 2. Install dependencies (PyYAML, matplotlib, pandas, scipy, numpy)
# 3. Run: python -m validation_ch7.scripts.test_section_7_3_analytical
# 4. Copy SEULEMENT validation_ch7/results/section_7_3_analytical/ → /kaggle/working/section_7_3_analytical/
#    - Préserve la structure organisée (figures/, data/, latex/)
#    - PAS de wrapper validation_results/
#    - PAS de pollution cross-section
# 5. Cleanup repo (garde seulement section_7_3_analytical/)
# 6. Generate kernel_metadata.json avec artifact counts par type
```

### Artefacts Comptés par Type :
- **NPZ** : `data/npz/*.npz`
- **PNG** : `figures/*.png`
- **YAML** : `data/scenarios/*.yml`
- **TEX** : `latex/*.tex`
- **JSON** : `*.json` (session_summary)
- **CSV** : `data/metrics/*.csv`

---

## 🎯 Extension aux Sections 7.4-7.7

### Chaque section suit le MÊME pattern organisé :

| Section | Figures Clés | Métriques |
|---------|-------------|-----------|
| **7.3 Analytique** | Riemann (5), Convergence (1) | L2 error, convergence order |
| **7.4 Calibration** | Comparaison observé/simulé | MAPE, RMSE, GEH, Theil-U |
| **7.5 Digital Twin** | Heatmaps spatio-temporels | Corrélation, erreur RMS |
| **7.6 Perf RL** | Courbes d'apprentissage | Reward moyen, stabilité |
| **7.7 Robustesse** | Impact qualité infrastructure | Écart-type vitesse, débit |

### Créer une nouvelle section :

1. **Copier template script** : `test_section_7_3_analytical.py` → `test_section_7_4_calibration.py`
2. **Adapter `__init__()`** : Changer `output_dir="validation_ch7/results/section_7_4_calibration"`
3. **Adapter méthodes test** : Remplacer `test_riemann_problems()` par `test_calibration_comparison()`
4. **Créer fonctions plot** : Ajouter `plot_calibration_comparison()` dans `validation_utils.py`
5. **Template LaTeX** : Créer `templates/section_7_4_calibration.tex` avec nouveaux placeholders
6. **Configurer manager** : Ajouter section dans `VALIDATION_SECTIONS` de `validation_kaggle_manager.py`

**Résultat :** Structure identique mais contenu spécifique à la section
```
validation_ch7/results/section_7_4_calibration/
├── figures/
├── data/
│   ├── npz/
│   ├── scenarios/
│   └── metrics/
├── latex/
└── session_summary.json
```

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
def __init__(self, output_dir="validation_ch7/results/section_7_3_analytical"):
    self.output_dir = Path(output_dir)
    
    # Structure organisée par type
    self.figures_dir = self.output_dir / "figures"
    self.npz_dir = self.output_dir / "data" / "npz"
    self.scenarios_dir = self.output_dir / "data" / "scenarios"
    self.metrics_dir = self.output_dir / "data" / "metrics"
    self.latex_dir = self.output_dir / "latex"
    
    # CRITIQUE : Créer TOUS les dossiers avant utilisation
    for directory in [self.figures_dir, self.npz_dir, self.scenarios_dir, 
                     self.metrics_dir, self.latex_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[INIT] Created: {directory}")
```

### 6. **Migration des Fichiers Existants**
Si vous avez déjà des `validation_output/` avec l'ancienne structure désorganisée :

```bash
# Créer script de migration pour réorganiser par type
python migrate_validation_structure.py
```

Le script va :
- Déplacer `*.png` → `figures/`
- Déplacer `*.npz` → `data/npz/`
- Déplacer `*.yml` → `data/scenarios/`
- Déplacer `*.tex` → `latex/`
- Déplacer `*.csv`, `*.json` (metrics) → `data/metrics/`
- Supprimer dossiers imbriqués `validation_results/`

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
