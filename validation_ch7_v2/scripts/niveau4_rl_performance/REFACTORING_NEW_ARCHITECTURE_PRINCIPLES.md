# Nouvelle Architecture - Principes et Structure

**Document**: Définition des Principes Architecturaux et Structure des Modules
**Objectif**: Guider l'implémentation du refactoring selon Clean Architecture

---

## Principes Architecturaux (8 Principes FONDAMENTAUX)

### 1. Clean Architecture (Robert C. Martin)
- **Couches**: Domain (métier)  Infrastructure (technique)  Entry Points (interfaces)
- **Règle de dépendance**: Les couches internes ne dépendent JAMAIS des couches externes
- **Testabilité**: Domain 100% testable sans infrastructure

### 2. Single Responsibility Principle (SRP)
- Chaque classe = 1 seule responsabilité, 1 seule raison de changer
- God Class (1877 lignes)  7 classes spécialisées (<300 lignes chacune)

### 3. Dependency Inversion Principle (DIP)
- Code métier dépend d'abstractions (interfaces), pas de détails (implémentations)
- Injection de dépendances via constructeur

### 4. Configuration as Data (12-Factor App)
- Configuration externalisée en YAML (pas hardcodée en Python)
- Quick test = changement config, pas modification code

### 5. Domain-Driven Design (DDD)
- Langage ubiquitaire: Cache, Checkpoint, Controller, Orchestrator
- Bounded contexts clairs

### 6. Gestion d'Erreurs Explicite
- Try-except avec logging structuré
- Recovery automatique (cache corrompu  régénération)

### 7. Testability First
- Tests unitaires (<1s) pour chaque module
- Pyramide: 100+ unit tests, 10 integration, 2 E2E

### 8. Structured Logging
- Logs structurés (JSON) pour métriques automatiques
- Contexte riche: scenario, config_hash, timestep, reward, etc.

---

## Structure des Modules (Clean Architecture)

\\\
validation_ch7_v2/scripts/niveau4_rl_performance/
 domain/                          # COUCHE MÉTIER (testable sans infra)
    cache/
       cache_manager.py        # Innovation 1: Cache Additif Baseline
    checkpoint/
       checkpoint_manager.py   # Innovation 2: Config-Hashing Checkpoints
       config_hasher.py        # Innovation 5: Rotation
    controllers/
       baseline_controller.py  # Innovation 3: State Serialization
       rl_controller.py        # Innovation 3: State Serialization
    orchestration/
       training_orchestrator.py # Orchestration entraînement
    interfaces.py               # Abstractions (DIP)

 infrastructure/                  # COUCHE TECHNIQUE (adaptateurs)
    cache/
       pickle_storage.py       # Implémentation storage pickle
    config/
       config_loader.py        # Chargement YAML
    logging/
        structured_logger.py    # Innovation 7: Dual Logging

 entry_points/                    # COUCHE INTERFACE
    cli.py                      # 1 seul CLI (Click)

 tests/                           # TESTS (pyramide)
    unit/                       # 100+ tests (<1s chacun)
    integration/                # 10 tests (~5-10 min)
    e2e/                        # 2 tests (3-4h sur GPU)

 config/
     section_7_6_rl_performance.yaml  # Innovation 6: DRY + Config externe
\\\

---

## Implémentation - Ordre de Priorité

**Phase 1 - Infrastructure (Fondations)**:
1. Créer interfaces abstraites (domain/interfaces.py)
2. Implémenter ConfigManager + YAML
3. Implémenter PickleCacheStorage
4. Implémenter StructuredLogger

**Phase 2 - Domain (Logique Métier)**:
5. CacheManager (Innovation 1)
6. CheckpointManager + ConfigHasher (Innovations 2, 5)
7. BaselineController + RLController (Innovation 3)
8. TrainingOrchestrator

**Phase 3 - Entry Points**:
9. CLI unique (Click)
10. Tests unitaires pour chaque module

**Phase 4 - Validation**:
11. Tests integration
12. Test E2E quick mode local
13. Déploiement Kaggle GPU

---

## Checklist de Conformité

Avant de valider chaque module, vérifier:
- [ ] Respecte SRP (1 responsabilité)
- [ ] Dépendances injectées (pas de création inline)
- [ ] Tests unitaires (<1s) écrits et passants
- [ ] Gestion d'erreurs explicite
- [ ] Logging structuré
- [ ] Innovation correspondante préservée (vérifier table de correspondance)

