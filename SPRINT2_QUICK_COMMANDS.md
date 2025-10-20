# 🚀 COMMANDES RAPIDES - SPRINT 2

## Validation Rapide (<5s)
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python quick_test_riemann.py
```

## Regénérer TOUTES les Figures PNG
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python generate_all_png.py
```

## Tests Individuels

### Test 1 - Shock Motos
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python test_riemann_motos_shock.py
```

### Test 2 - Rarefaction Motos
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python test_riemann_motos_rarefaction.py
```

### Test 3 - Shock Voitures
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python test_riemann_voitures_shock.py
```

### Test 4 - Rarefaction Voitures
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python test_riemann_voitures_rarefaction.py
```

### Test 5 - Multiclass (CRITIQUE) ⭐
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python test_riemann_multiclass.py
```

### Étude de Convergence
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau1_mathematical_foundations"
python convergence_study.py
```

## Vérification Livrables

### Lister Figures PNG
```powershell
Get-ChildItem "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES\figures\*.png"
```

### Lister JSON
```powershell
Get-ChildItem "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES\results\*.json"
```

### Voir Structure Complète
```powershell
Get-ChildItem "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES" -Recurse
```

## Copier vers LaTeX

### Copier tout le dossier
```powershell
Copy-Item "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES" -Destination "C:\chemin\vers\these\" -Recurse
```

### Copier seulement LaTeX
```powershell
Copy-Item "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES\latex\*" -Destination "C:\chemin\vers\these\validation\"
Copy-Item "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES\figures\*" -Destination "C:\chemin\vers\these\figures\"
```

## Documentation

### Lire README principal
```bash
cat "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES\README.md"
```

### Lire Synthèse Exécutive
```bash
cat "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES\EXECUTIVE_SUMMARY.md"
```

### Lire Guide LaTeX
```bash
cat "d:\Projets\Alibi\Code project\SPRINT2_DELIVERABLES\latex\GUIDE_INTEGRATION_LATEX.md"
```

## Livrables Sprint 2

| Fichier | Description |
|---------|-------------|
| `SPRINT2_DELIVERABLES/figures/*.png` | 6 figures PNG 300 DPI |
| `SPRINT2_DELIVERABLES/results/*.json` | 6 résultats JSON |
| `SPRINT2_DELIVERABLES/latex/table71_updated.tex` | Tableau 7.1 |
| `SPRINT2_DELIVERABLES/latex/figures_integration.tex` | Figures LaTeX |
| `SPRINT2_DELIVERABLES/README.md` | Documentation complète |
| `SPRINT2_DELIVERABLES/EXECUTIVE_SUMMARY.md` | Synthèse |
| `SPRINT2_COMPLETE.md` | Récapitulatif final |

---

**Status:** ✅ SPRINT 2 COMPLET  
**Prêt pour:** SPRINT 3 (Niveau 2 - Phénomènes Physiques)
