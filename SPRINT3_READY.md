# 🚀 SPRINT 3 - PRÊT À DÉMARRER

## Ce qui va être créé

### ✅ Test 1: Gap-Filling (PRIORITÉ 1)
**Fichier:** `gap_filling_test.py` (~300 lignes)

**Scénario:**
- 20 motos (0-100m) à 40 km/h rattrapent 10 voitures (100-1000m) à 25 km/h
- Durée: 300s, segment 1000m, α=0.5

**Outputs:**
- `gap_filling_evolution.png` (3 snapshots: t=0, 150, 300s)
- `gap_filling_metrics.png` (bar chart comparatif)
- `gap_filling_test.json` (métriques)

**Métriques:**
- Vitesse motos vs voitures
- Taux infiltration
- Δv maintenu

---

### ✅ Test 2: Interweaving (PRIORITÉ 2)
**Fichier:** `interweaving_test.py` (~250 lignes)

**Scénario:**
- 15 motos + 15 voitures mélangés
- Distribution homogène initiale
- Durée: 400s

**Outputs:**
- `interweaving_distribution.png` (évolution distribution spatiale)
- `interweaving_test.json`

---

### ✅ Test 3: Diagrammes Fondamentaux (PRIORITÉ 3)
**Fichier:** `fundamental_diagrams.py` (~200 lignes)

**Courbes théoriques:**
- V-ρ pour motos et voitures
- Q-ρ pour motos et voitures

**Outputs:**
- `fundamental_diagrams.png` (2x2 subplots)
- `fundamental_diagrams.json`

---

### ✅ Quick Test (Validation Rapide)
**Fichier:** `quick_test_niveau2.py` (~100 lignes)

Execute les 3 tests et affiche résumé.

---

## Estimation Temps

| Tâche | Durée |
|-------|-------|
| Gap-filling test | 1.5h |
| Interweaving test | 1h |
| Diagrammes fondamentaux | 45min |
| Quick test + intégration | 30min |
| Documentation + LaTeX | 30min |
| **TOTAL** | **~4-5h** |

---

## Pattern Établi (Sprint 2)

```
1. Créer test files
2. Exécuter et valider
3. Générer PNG (300 DPI)
4. Créer JSON résultats
5. Organiser dans SPRINT3_DELIVERABLES/
6. LaTeX integration files
7. Documentation
```

---

## ❓ Question Avant de Continuer

**Veux-tu que je:**

A) 🚀 **CONTINUE DIRECTEMENT** - Je crée tout le code maintenant  
B) 🎨 **VOIR UN EXEMPLE** - Je te montre d'abord gap_filling_test.py avant de tout faire  
C) 📝 **AJUSTER LE PLAN** - Tu as des changements à proposer

**Dis-moi juste la lettre et j'agis !**

---

**Status actuel:** ✅ SPRINT 2 complet, dossiers Sprint 3 créés, plan établi  
**Prochaine étape:** Implémentation Test 1 (Gap-Filling)
