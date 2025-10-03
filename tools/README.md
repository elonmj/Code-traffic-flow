# Tools - Utilitaires de Validation

Scripts utilitaires pour le système de validation ARZ-RL.

##  Monitoring & Debugging

### monitor_kernel.py
Monitore un kernel Kaggle existant en temps réel.

\\\ash
python tools/monitor_kernel.py
# Modifiez kernel_slug dans le fichier avant exécution
\\\

### erify_kaggle_npz.py
Vérifie l'intégrité des fichiers NPZ téléchargés depuis Kaggle.

\\\ash
python tools/verify_kaggle_npz.py
\\\

### debug_encoding_validation.py
Debug les problèmes d'encodage dans les kernels générés.

\\\ash
python tools/debug_encoding_validation.py
\\\

##  Pre-Upload Validation

### 	est_kaggle_config.py
Valide la configuration avant upload Kaggle (credentials, repo, structure).

\\\ash
python tools/test_kaggle_config.py
\\\

##  Testing

### 	est_minimal_riemann_npz.py
Test minimal de génération NPZ pour validation Riemann.

\\\ash
python tools/test_minimal_riemann_npz.py
\\\

---

**Note:** Ces scripts sont des **utilitaires**, pas du code principal.  
Pour lancer la validation complète, utilisez python validation_ch7/scripts/validation_cli.py.
