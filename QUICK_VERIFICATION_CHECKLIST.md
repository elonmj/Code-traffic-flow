# 🔍 Quick Verification Checklist - Post-Archivage

**Date**: October 22, 2025  
**Commit**: 240bafb  

---

## ✅ Avant d'utiliser le projet, vérifier:

### 1. Structure Archive OK
```bash
# Vérifier que l'archive existe
ls -la _archive/2024_phase13_calibration/

# Output expected:
# - digital_twin_calibrator.py
# - spatiotemporal_validator.py
# - tomtom_collector.py
# - ... (9 modules total)
# - README.md
# - TRANSITION_SUMMARY.md
```

### 2. Modules Conservés Disponibles
```bash
# real_data_loader.py doit exister
test -f arz_model/calibration/data/real_data_loader.py && echo "✅ OK" || echo "❌ FAIL"

# groups/ doit exister
test -d arz_model/calibration/data/groups && echo "✅ OK" || echo "❌ FAIL"

# victoria_island_corridor.json dans groups/
test -f arz_model/calibration/data/groups/victoria_island_corridor.json && echo "✅ OK" || echo "❌ FAIL"
```

### 3. Imports Legacy Échouent
```bash
# Tester que les vieux imports échouent (attendu)
python -c "from arz_model.calibration.data import GroupManager" 2>&1 | grep -q "ModuleNotFoundError" && echo "✅ Expected failure" || echo "❌ Unexpected success"
python -c "from arz_model.calibration.data.digital_twin_calibrator import DigitalTwinCalibrator" 2>&1 | grep -q "ModuleNotFoundError" && echo "✅ Expected failure" || echo "❌ Unexpected success"
```

### 4. Imports Conservés Fonctionnent
```bash
# Ces imports DOIVENT fonctionner
python -c "from arz_model.calibration.data import RealDataLoader; print('✅ RealDataLoader works')" || echo "❌ FAIL"
```

### 5. Section 7.4 Toujours OK
```bash
# Vérifier que test_section_7_4_calibration charge
cd validation_ch7/scripts
python -c "from test_section_7_4_calibration import RealCalibrationValidationTest; print('✅ Section 7.4 imports OK')" || echo "❌ FAIL"
```

### 6. Ch7 Validation Non-Affectée
```bash
# Tester que validation_kaggle_manager reconnaît Section 7.4
cd "d:\Projets\Alibi\Code project"
python validation_ch7/scripts/validation_kaggle_manager.py --section 7.4 --help | grep -q "section_7_4_calibration" && echo "✅ Section 7.4 recognized" || echo "❌ FAIL"
```

---

## 🔧 Si quelque chose échoue:

### Problem: `ModuleNotFoundError: No module named 'arz_model.calibration.data.XXX'`
**Solution**: C'est ATTENDU! Ce module est archivé. Si vous en avez besoin:
```bash
# Option 1: Accéder depuis archive
cat _archive/2024_phase13_calibration/digital_twin_calibrator.py

# Option 2: Restaurer depuis git
git show HEAD^:arz_model/calibration/data/digital_twin_calibrator.py > temp.py
```

### Problem: `FileNotFoundError: real_data_loader.py not found`
**Solution**: C'est une erreur GRAVE. Vérifier:
```bash
git status
# Si le fichier manque:
git restore arz_model/calibration/data/real_data_loader.py
```

### Problem: `FileNotFoundError: victoria_island_corridor.json not found`
**Solution**: C'est une erreur. Vérifier:
```bash
git status
git restore arz_model/calibration/data/groups/victoria_island_corridor.json
```

### Problem: Section 7.4 échoue
**Solution**: Vérifier que real_data_loader fonctionne:
```bash
python -c "from arz_model.calibration.data import RealDataLoader; print(RealDataLoader.__doc__)"
# Si ça échoue, restaurer le fichier via git
```

---

## 🚀 Si tout est OK:

✅ Archivage successful  
✅ Legacy code sécurisé  
✅ Active components préservés  
✅ Ch7 validation ready to go  

**Prochaines étapes**: Continuer avec validation_ch7 pipeline normalement

---

## 📚 Documentation de Référence:

- **Archivage Analysis**: `.audit/CALIBRATION_DATA_OBSOLESCENCE_AUDIT.md`
- **Archive README**: `_archive/2024_phase13_calibration/README.md`
- **Transition Guide**: `_archive/2024_phase13_calibration/TRANSITION_SUMMARY.md`
- **Complete Report**: `ARCHIVAGE_COMPLETE_REPORT.md`

---

## 🔗 Git Reference:

```bash
# Voir le commit d'archivage
git show 240bafb

# Voir ce qui a changé
git diff 39165be..240bafb

# Restaurer un module archivé
git checkout 39165be -- arz_model/calibration/data/digital_twin_calibrator.py

# Voir l'historique d'un fichier archivé
git log -- _archive/2024_phase13_calibration/digital_twin_calibrator.py
```

---

**Last Updated**: October 22, 2025  
**Status**: ✅ Ready for Use  
**Commit**: 240bafb
