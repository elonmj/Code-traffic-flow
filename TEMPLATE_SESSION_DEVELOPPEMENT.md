# 🔄 Template de Cycle de Développement

**Utilise ce template pour structurer tes sessions de développement**

---

## 📋 Session Info

- **Date:** ___________
- **Objectif:** ___________________________________________
- **Complexité estimée:** ☐ Simple ☐ Moyenne ☐ Complexe
- **Itérations prévues:** ☐ 1 ☐ 2-3 ☐ 3+

---

## Phase 1: 🔍 Context Gathering (13% du temps)

**Checklist:**
- [ ] Lire les fichiers principaux (au moins 3-5 fichiers)
- [ ] Comprendre l'architecture actuelle
- [ ] Identifier les dépendances
- [ ] Noter les contraintes techniques

**Fichiers lus:**
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________
4. ___________________________________________
5. ___________________________________________

**Outils utilisés:** `read_file`, `grep_search`, `semantic_search`

**Notes de contexte:**
```
___________________________________________
___________________________________________
___________________________________________
```

**✅ Phase terminée:** ☐ Contexte suffisant | ☐ Besoin de plus d'info

---

## Phase 2: 📚 Research (18% du temps) [Optionnel]

**Nécessaire si:**
- [ ] Technologie inconnue
- [ ] Pattern architectural nouveau
- [ ] API/bibliothèque non documentée

**Recherches effectuées:**
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________

**Documentation consultée:**
- ___________________________________________
- ___________________________________________

**Outils utilisés:** `fetch_webpage`, `semantic_search`, `grep_search`

**Insights clés:**
```
___________________________________________
___________________________________________
___________________________________________
```

**✅ Phase terminée:** ☐ Compréhension suffisante | ☐ Besoin d'expérimentation

---

## Phase 3: 🧠 Analysis & Planning (8% du temps)

**Problème identifié:**
```
___________________________________________
___________________________________________
```

**Cause racine:**
```
___________________________________________
___________________________________________
```

**Solution proposée:**
```
Approche: ___________________________________________

Étapes:
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________
4. ___________________________________________
```

**Alternatives considérées:**
- Option A: ___________________________________________ [☐ Retenue ☐ Écartée]
- Option B: ___________________________________________ [☐ Retenue ☐ Écartée]

**Décision finale:**
```
___________________________________________
Raison: ___________________________________________
```

**Risques identifiés:**
- [ ] Performance
- [ ] Compatibilité
- [ ] Sécurité
- [ ] Maintenance
- [ ] Autre: ___________________________________________

**✅ Phase terminée:** ☐ Plan clair | ☐ Besoin de validation

---

## Phase 4: 💻 Implementation (2% du temps)

**Séquence d'outils recommandée:**
`read_file` → `replace_string` → `run_terminal` (test) → `run_terminal` (commit)

**Fichiers modifiés:**
1. ___________________________________________ [Lignes: ___-___]
2. ___________________________________________ [Lignes: ___-___]
3. ___________________________________________ [Lignes: ___-___]

**Changements effectués:**
```python
# Fichier: ___________________________________________
# Ligne: ___________________________________________
# Ancien code:
___________________________________________

# Nouveau code:
___________________________________________

# Raison: ___________________________________________
```

**Outils utilisés:** `replace_string`, `create_file`

**⚠️ Attention:** Toujours tester après implémentation!

**✅ Phase terminée:** ☐ Code écrit | ☐ Prêt pour test

---

## Phase 5: 🧪 Testing (32% du temps) [CRITIQUE]

**Stratégie de test:**
- [ ] Quick test (15 min) - **FAIRE EN PREMIER**
- [ ] Unit tests
- [ ] Integration tests
- [ ] Full validation (2h)

**Quick Test:**
```bash
Commande: ___________________________________________
Résultat attendu: ___________________________________________
Résultat obtenu: ___________________________________________
```

**Tests exécutés:**
1. ___________________________________________ [☐ ✅ Succès ☐ ❌ Échec]
2. ___________________________________________ [☐ ✅ Succès ☐ ❌ Échec]
3. ___________________________________________ [☐ ✅ Succès ☐ ❌ Échec]

**Outils utilisés:** `run_terminal`, `get_errors`

**Logs/Erreurs:**
```
___________________________________________
___________________________________________
___________________________________________
```

**✅ Phase terminée:** 
- ☐ Tous tests passent → Aller à Validation Finale
- ☐ Échecs détectés → Aller à Debugging

---

## Phase 6: 🐛 Debugging (26% du temps) [Si nécessaire]

**Itération #___ (Moyenne: 1.8 itérations)**

**Erreur rencontrée:**
```
Type: ___________________________________________
Message: ___________________________________________
Stack trace: ___________________________________________
```

**Investigation:**
```bash
# Outils utilisés:
grep_search: ___________________________________________
read_file: ___________________________________________
```

**Hypothèse:**
```
___________________________________________
___________________________________________
```

**Fix appliqué:**
```python
# Fichier: ___________________________________________
# Changement:
___________________________________________
```

**Test du fix:**
```bash
Commande: ___________________________________________
Résultat: ☐ ✅ Corrigé ☐ ❌ Persiste
```

**Si persiste:** Recommencer une nouvelle itération (limite recommandée: 3)

**✅ Phase terminée:** ☐ Bug corrigé → Retour à Testing

---

## Phase 7: ✅ Validation Finale

**Checklist finale:**
- [ ] Tous les tests passent
- [ ] Code review (auto-review si solo)
- [ ] Documentation à jour
- [ ] Commit avec message descriptif
- [ ] Push vers remote

**Métriques de succès:**
```
Tests réussis: _____ / _____
Performance: _____________________
Qualité du code: _____________________
```

**Commit:**
```bash
git add ___________________________________________
git commit -m "___________________________________________"
git push origin ___________________________________________
```

**✅ Session terminée avec succès!**

---

## 📊 Rétrospective de Session

**Temps passé par phase:**
- Context Gathering: _____ min (objectif: 13%)
- Research: _____ min (objectif: 18%)
- Analysis: _____ min (objectif: 8%)
- Implementation: _____ min (objectif: 2%)
- Testing: _____ min (objectif: 32%)
- Debugging: _____ min (objectif: 26%)

**Statistiques:**
- Nombre d'itérations: _____ (objectif: 1-2)
- Taux de succès: ☐ Premier coup ☐ 2e tentative ☐ 3e+ tentative
- Séquences d'outils utilisées: ___________________________________________

**Ce qui a bien fonctionné:**
```
___________________________________________
___________________________________________
```

**Ce qui peut être amélioré:**
```
___________________________________________
___________________________________________
```

**Leçons apprises:**
```
___________________________________________
___________________________________________
```

**Pattern réutilisable détecté:**
```
Situation: ___________________________________________
Solution: ___________________________________________
Outils: ___________________________________________
```

---

## 🎯 Prochaines Sessions

**Actions de suivi:**
- [ ] ___________________________________________
- [ ] ___________________________________________
- [ ] ___________________________________________

**Améliorations du workflow:**
- [ ] ___________________________________________
- [ ] ___________________________________________

---

**📝 Notes additionnelles:**
```
___________________________________________
___________________________________________
___________________________________________
___________________________________________
___________________________________________
```

---

## 📈 Métriques Cumulatives (À suivre au fil des sessions)

| Session | Objectif | Itérations | Succès | Temps Total | Pattern Dominant |
|---------|----------|------------|--------|-------------|------------------|
| 1       |          |            |        |             |                  |
| 2       |          |            |        |             |                  |
| 3       |          |            |        |             |                  |
| ...     |          |            |        |             |                  |

**Tendances observées:**
- Amélioration du temps: ☐ Oui ☐ Non ☐ Stable
- Réduction des itérations: ☐ Oui ☐ Non ☐ Stable
- Patterns récurrents identifiés: ___________________________________________

---

**🔄 Template créé à partir de l'analyse de 14,114 lignes de développement réel**  
**📊 Basé sur 1,233 phases, 369 cycles, 75.1% de taux de succès**  
**Version:** 1.0  
**Date:** 2025-10-11
