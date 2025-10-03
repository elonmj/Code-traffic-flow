# Architecture de Validation - Règles Absolues

##  Principe Architecturale Fondamental

**UN SEUL endroit pour les résultats: alidation_output/**

`
Code project/
 validation_ch7/           # CODE SEULEMENT (tests, templates, configs)
    scripts/             # Scripts de test Python
    templates/           # Templates LaTeX
    configs/             # Configurations spécifiques

 validation_output/        # RÉSULTATS SEULEMENT (générés par tests)
     results/
         local_test/      # Tests locaux
            section_7_X_name/
                figures/
                data/
                   npz/
                   scenarios/
                   metrics/
                latex/
                session_summary.json
        
         {kernel_slug}/   # Résultats Kaggle téléchargés
             section_7_X_name/
                 (même structure)
`

##  Règles Strictes

###  INTERDIT
- ~~alidation_ch7/results/~~  **SUPPRIMÉ**
- ~~alidation_ch7/scripts/validation_ch7/~~  **SUPPRIMÉ**
- Fichiers éparpillés (YML, TEX, JSON) à la racine
- Triple imbrication alidation_results/section/validation_results/

###  OBLIGATOIRE
- **Tous les tests héritent de ValidationSection**
- **Résultats dans alidation_output/ UNIQUEMENT**
- **Structure organisée par type**: igures/, data/npz/, data/scenarios/, data/metrics/, latex/
- **Un fichier session_summary.json** par section

##  Pattern à Suivre

### Créer un Nouveau Test (Exemple Section 7.5)

\\\python
# validation_ch7/scripts/test_section_7_5_digital_twin.py

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from validation_utils import ValidationSection

class DigitalTwinValidationTests(ValidationSection):
    \"\"\"Tests de validation Digital Twin\"\"\"
    
    def __init__(self):
        # HÉRITE de l'architecture standard
        super().__init__(section_name=\"section_7_5_digital_twin\")
        
        # Configuration spécifique à 7.5
        self.heatmap_config = {...}
        
    def test_spatiotemporal_accuracy(self):
        # Générer données
        result = run_simulation(...)
        
        # Sauvegarder NPZ dans self.npz_dir
        np.savez(self.npz_dir / \"heatmap_data.npz\", ...)
        
        # Générer figure dans self.figures_dir
        fig = plot_heatmap(...)
        fig.savefig(self.figures_dir / \"spatiotemporal_heatmap.png\")
        
        # Métriques dans self.metrics_dir
        pd.DataFrame(metrics).to_csv(self.metrics_dir / \"accuracy.csv\")
        
    def generate_section_content(self):
        # Exécuter tests
        results = self.test_spatiotemporal_accuracy()
        
        # LaTeX dans self.latex_dir
        with open(self.latex_dir / \"section_7_5_content.tex\", 'w') as f:
            f.write(template.format(**results))
        
        # Session summary
        self.save_session_summary(additional_info={'status': 'completed'})
\\\

### Upload sur Kaggle

Le ValidationKaggleManager copie automatiquement:
`
validation_output/results/local_test/section_7_X_name/
 (upload GitHub)
 (exécution Kaggle)
 /kaggle/working/section_7_X_name/
 (download)
 validation_output/results/{kernel_slug}/section_7_X_name/
`

##  Checklist Avant Commit

- [ ] Test hérite de ValidationSection
- [ ] Aucun chemin hardcodé vers alidation_ch7/results/
- [ ] Utilise self.figures_dir, self.npz_dir, self.scenarios_dir, self.metrics_dir, self.latex_dir
- [ ] Appelle self.save_session_summary() à la fin
- [ ] Aucun dossier esults/ dans alidation_ch7/

##  Avantages de Cette Architecture

1. **Séparation propre**: Code (validation_ch7) vs Résultats (validation_output)
2. **Pas de duplication**: Un seul endroit pour les résultats
3. **Gitignore simple**: alidation_output/ dans .gitignore
4. **Scalable**: Facile d'ajouter sections 7.4-7.7 sans conflit
5. **Maintenable**: Classe de base garantit la cohérence
