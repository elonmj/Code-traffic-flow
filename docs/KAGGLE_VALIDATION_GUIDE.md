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

 
 - - - 
 
 # #     O p t i o n   1   S e n s i t i v i t y   V a l i d a t i o n 
 
 # # #   Q u i c k   T e s t   C h e c k l i s t 
 
 A f t e r   i m p l e m e n t i n g   s e n s i t i v i t y   f i x e s   ( c o m m i t   e 8 2 5 3 0 0 ) : 
 
 # # # #   1 .   P r e p a r e   K a g g l e   E n v i r o n m e n t 
 -   [   ]   U p d a t e   k e r n e l   m e t a d a t a   i f   n e e d e d 
 -   [   ]   E n s u r e   G P U   a c c e l e r a t o r   e n a b l e d 
 -   [   ]   C h e c k   d e p e n d e n c i e s   ( s t a b l e - b a s e l i n e s 3 ,   p y y a m l ,   n u m b a ) 
 
 # # # #   2 .   L a u n c h   Q u i c k   T e s t 
 \ \ \  a s h 
 p y t h o n   t o o l s / v a l i d a t i o n _ k a g g l e _ m a n a g e r . p y   - - k e r n e l   g g v i   - - q u i c k - t e s t 
 \ \ \ 
 
 E x p e c t e d   d u r a t i o n :   ~ 2   m i n u t e s   o n   T 4   G P U 
 
 # # # #   3 .   M o n i t o r   E x e c u t i o n 
 C h e c k   \ d e b u g . l o g \   f o r   B C   u p d a t e s : 
 \ \ \ 
 [ B C   U P D A T E ]   l e f t     p h a s e   0   ( o u t f l o w ) 
       O u t f l o w :   z e r o - o r d e r   e x t r a p o l a t i o n 
 [ B C   U P D A T E ]   l e f t     p h a s e   1   ( i n f l o w ) 
       I n f l o w   s t a t e :   r h o _ m = 0 . 1 0 0 0 ,   w _ m = 1 5 . 0 ,   r h o _ c = 0 . 1 2 0 0 ,   w _ c = 1 2 . 0 
 \ \ \ 
 
 # # # #   4 .   A n a l y z e   S t a t e   D i v e r g e n c e 
 \ \ \ p y t h o n 
 #   C h e c k   d e b u g . l o g   f o r   s t a t e   h a s h   e v o l u t i o n 
 B a s e l i n e   s t e p   0 :   h a s h   =   - 4 4 4 9 0 3 0 4 1 7 5 4 5 1 5 6 2 2 9       D i f f e r e n t 
 R L   s t e p   0 :               h a s h   =     4 0 3 3 0 5 3 0 2 6 5 5 0 7 0 2 2 3 3       D i f f e r e n t 
 B a s e l i n e   s t e p   1 :   h a s h   =   - 7 3 5 8 3 4 2 1 5 3 1 0 1 0 7 5 2 8 2       S h o u l d   S T A Y   d i f f e r e n t 
 R L   s t e p   1 :               h a s h   =     8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0       S h o u l d   S T A Y   d i f f e r e n t 
 \ \ \ 
 
 * * E x p e c t e d * * :   H a s h e s   s t a y   d i f f e r e n t   ( n o t   c o n v e r g e n c e   l i k e   b e f o r e ) 
 
 # # # #   5 .   V e r i f y   M e t r i c s   D i v e r g e n c e 
 \ \ \ p y t h o n 
 #   C h e c k   f i n a l   m e t r i c s 
 B a s e l i n e :   f l o w = 2 1 . 8 4 4 ,   e f f i c i e n c y = 3 . 4 9 5 
 R L :               f l o w = 1 8 . 2 3 2 ,   e f f i c i e n c y = 4 . 1 1 2       S h o u l d   b e   d i f f e r e n t ! 
 \ \ \ 
 
 * * E x p e c t e d * * :   F l o w / e f f i c i e n c y   v a l u e s   d i f f e r   s i g n i f i c a n t l y 
 
 # # # #   6 .   D e c i s i o n   P o i n t 
 
 * *   S U C C E S S * *   ( s t a t e s   d i v e r g e ,   m e t r i c s   d i f f e r ) : 
 -   P r o c e e d   t o   f u l l   1 0 , 0 0 0   t i m e s t e p   t r a i n i n g 
 -   G e n e r a t e   L a T e X   t h e s i s   s e c t i o n 
 -   V a l i d a t e   R 5   r e v e n d i c a t i o n 
 
 * *   S T I L L   I D E N T I C A L * *   ( s t a t e s   c o n v e r g e ,   m e t r i c s   s a m e ) : 
 -   I n v e s t i g a t e   O p t i o n   2   ( c o n t i n u o u s   a c t i o n   s p a c e ) 
 -   C h e c k   B C   l o g g i n g   i s   a c t u a l l y   a p p e a r i n g 
 -   V e r i f y   R i e m a n n   I C   i s   c r e a t i n g   s h o c k   w a v e 
 -   C o n s i d e r   e v e n   s m a l l e r   d o m a i n   ( 5 0 0 m   i n s t e a d   o f   1 k m ) 
 
 # # #   F u l l   T r a i n i n g   ( A f t e r   Q u i c k   T e s t   V a l i d a t e s ) 
 
 \ \ \  a s h 
 #   R e m o v e   - - q u i c k - t e s t   f l a g   f o r   f u l l   t r a i n i n g 
 p y t h o n   t o o l s / v a l i d a t i o n _ k a g g l e _ m a n a g e r . p y   - - k e r n e l   g g v i 
 \ \ \ 
 
 E x p e c t e d   d u r a t i o n :   ~ 1 5   m i n u t e s   o n   T 4   G P U 
 T o t a l   t i m e s t e p s :   1 0 , 0 0 0 
 E p i s o d e s :   V a r i a b l e   ( d e p e n d s   o n   c o n v e r g e n c e ) 
 
 # # #   F i l e s   t o   C h e c k 
 
 1 .   * * \ d e b u g . l o g \ * * :   B C   u p d a t e s ,   s t a t e   h a s h e s ,   e r r o r   m e s s a g e s 
 2 .   * * \ 	 r a f f i c _ l i g h t _ c o n t r o l . y m l \ * * :   V e r i f y   N = 1 0 0 ,   x m a x = 1 0 0 0 ,   t y p e = r i e m a n n 
 3 .   * * \ c o m p r e h e n s i v e _ v a l i d a t i o n _ r e p o r t . j s o n \ * * :   F i n a l   m e t r i c s   c o m p a r i s o n 
 
 # # #   S u c c e s s   M e t r i c s 
 
 |   M e t r i c   |   T a r g e t   |   M e a s u r e m e n t   | 
 | - - - - - - - - | - - - - - - - - | - - - - - - - - - - - - - | 
 |   S t a t e   d i v e r g e n c e   |   H a s h e s   s t a y   d i f f e r e n t   |   d e b u g . l o g   s t a t e   h a s h e s   | 
 |   C o n t r o l   o b s e r v a b l e   |   B C   l o g s   s h o w   p h a s e   c h a n g e s   |   d e b u g . l o g   B C   u p d a t e s   | 
 |   M e t r i c s   d i v e r g e n c e   |   f l o w / e f f i c i e n c y   d i f f e r   |   c o m p r e h e n s i v e _ v a l i d a t i o n _ r e p o r t . j s o n   | 
 |   T r a i n i n g   c o n v e r g e n c e   |   R e w a r d   i n c r e a s e s   |   P P O   t r a i n i n g   l o g s   | 
 
  
 