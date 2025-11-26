# 📥 Guide de Téléchargement des Résultats Kaggle

## ⚠️ IMPORTANT - Lisez ceci avant tout téléchargement

L'API Kaggle Python peut renvoyer des erreurs `403 Forbidden` pour certaines opérations.
**La solution qui fonctionne TOUJOURS** est d'utiliser la CLI Kaggle avec la variable d'environnement `KAGGLE_CONFIG_DIR`.

---

## 🚀 Commande de Téléchargement (COPIER-COLLER)

### PowerShell (Windows)
```powershell
$env:KAGGLE_CONFIG_DIR = "d:\Projets\Alibi\Code project"
kaggle kernels output elonmj/generic-test-runner-kernel -p "d:\Projets\Alibi\Code project\kaggle\results\<NOM_DU_TEST>"
```

### Exemple concret :
```powershell
$env:KAGGLE_CONFIG_DIR = "d:\Projets\Alibi\Code project"
New-Item -ItemType Directory -Force -Path "d:\Projets\Alibi\Code project\kaggle\results\cuda_closure_fix_test" | Out-Null
kaggle kernels output elonmj/generic-test-runner-kernel -p "d:\Projets\Alibi\Code project\kaggle\results\cuda_closure_fix_test"
```

---

## 📋 Checklist de Téléchargement

1. ✅ Vérifier que le kernel est terminé (status = "complete")
2. ✅ Définir `KAGGLE_CONFIG_DIR` vers le dossier contenant `kaggle.json`
3. ✅ Créer le dossier de destination si nécessaire
4. ✅ Exécuter `kaggle kernels output <slug> -p <destination>`
5. ✅ Vérifier les fichiers téléchargés

---

## 🔧 Dépannage

### Erreur 403 Forbidden
```
403 Client Error: Forbidden for url: https://www.kaggle.com/api/v1/kernels/output
```

**Solution** : Toujours définir `$env:KAGGLE_CONFIG_DIR` AVANT la commande kaggle.

### Fichier kaggle.json introuvable
Le fichier `kaggle.json` doit être dans `d:\Projets\Alibi\Code project\kaggle.json`
avec le format :
```json
{"username":"elonmj","key":"VOTRE_CLE_API"}
```

### Kernel pas encore terminé
Vérifier le status avec :
```powershell
$env:KAGGLE_CONFIG_DIR = "d:\Projets\Alibi\Code project"
kaggle kernels list --user elonmj
```

---

## 📁 Structure des Résultats Téléchargés

```
kaggle/results/<NOM_DU_TEST>/
├── generic-test-runner-kernel.log    # Log complet du kernel
├── test_log.txt                      # Log de l'expérience
└── simulation_results/
    ├── traffic_signal_fix_test/
    │   └── test_results.json         # Résultats du test
    ├── thesis_stage1/                # Validation Stage 1
    ├── thesis_stage2/                # Training RL Stage 2
    └── thesis_figures/               # Figures générées
```

---

## 🎯 Rappel pour l'IA (GitHub Copilot)

**NE JAMAIS interrompre le monitoring Kaggle local !**

Quand tu lances `python kaggle_runner/executor.py --target ...`, c'est une **opération bloquante**.
- ❌ NE PAS couper la commande
- ❌ NE PAS se "désabonner" du terminal
- ✅ ATTENDRE que le kernel termine naturellement
- ✅ Le monitoring affichera le status jusqu'à completion

Si le monitoring est interrompu, utiliser la commande de téléchargement ci-dessus
une fois le kernel terminé (vérifiable sur https://www.kaggle.com/code/elonmj/generic-test-runner-kernel).

---

*Document créé le 2025-11-26 suite à des problèmes de téléchargement récurrents.*
