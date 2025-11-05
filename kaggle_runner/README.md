# Kaggle Validation Workflow

**Architecture CI/CD pour tests ARZ-RL sur GPU Kaggle**

## Problème résolu

Après 10 itérations manuelles de debugging (Kernels 1-10), ce workflow élimine :
- ✅ Création de multiples kernels (xpwy, cjsh, gfax...) → **UN kernel, updates multiples**
- ✅ Monitoring manuel des artifacts → **Auto-download avec paths explicites**
- ✅ Configuration floue → **Séparation config/tests/execution**
- ✅ Git add manuel → **Auto-commit/push avant Kaggle**

## Architecture

```
/kaggle
├── config/                      # Configurations de tests (YAML)
│   └── gpu_stability_test.yml   # Spécification test GPU
├── tests/                       # Tests copiés depuis validation_ch7
│   └── test_gpu_stability.py    # Code de test (COPIÉ, pas réécrit)
├── executor.py                  # Point d'entrée unique (CLI)
├── kernel_manager.py            # Gestion kernel (`update` pas `push`)
├── artifact_downloader.py       # Téléchargement automatique
└── README.md                    # Ce fichier
```

## Utilisation

```bash
# Test GPU stability (Experiment A)
python kaggle/executor.py --test gpu_stability

# Résultats téléchargés automatiquement vers :
# kaggle/results/<timestamp>/
```

## API Kaggle

**CORRECT** (utilisé ici):
```bash
kaggle kernels update -p /kaggle  # MAJ kernel existant
kaggle kernels output <kernel> -p kaggle/results/<timestamp>
```

**INCORRECT** (ancien système):
```bash
kaggle kernels push -p /path  # ❌ Crée nouveau kernel chaque fois
```

## Références

- Kaggle API officielle : https://github.com/Kaggle/kaggle-api
- Kernel 10 (succès technique) : v_max=172.11 m/s @ t=1.0s
- Experiment A : Test θ coupling causality hypothesis
