# Revue Architecturale du Modèle ARZ

Ce document compile une série de critiques architecturales constructives sur la base de code du modèle ARZ, identifiées lors d'une session d'analyse et de débogage.

---

### Critique 1 : La Gestion de la Configuration comme État Mutable

**Observation :** L'objet `self.params` (une instance de `ModelParameters`) n'est pas traité comme une configuration immuable. Il est activement et temporairement modifié pendant l'exécution de la boucle de simulation, notamment dans `network_grid.py` pour passer des informations aux solveurs de segments.

**Critique :** Utiliser un objet de configuration comme un canal de communication implicite pour un état éphémère est une source de complexité et de bugs.
- **Lisibilité :** Le code devient difficile à suivre car l'état réel dépend d'une séquence de mutations cachées.
- **Robustesse :** Une erreur d'exécution peut laisser l'objet de configuration dans un état incohérent.
- **Parallélisation :** La mutation d'un état partagé rend la parallélisation de la boucle sur les segments presque impossible.

**Alternative Architecturale :** La communication doit être explicite. Les méthodes devraient accepter les paramètres qui changent à chaque appel comme des arguments directs (ex: `segment.evolve(boundary_conditions=...)`). L'objet de configuration principal resterait ainsi stable et prévisible.

---

### Critique 2 : Le Contrat d'Interface pour les Conditions aux Limites

**Observation :** Le composant de haut niveau (`NetworkGrid`) doit connaître et préparer la structure de données exacte (`{'left': ...}`) attendue par le composant de bas niveau (le solveur physique du segment).

**Critique :** C'est une "abstraction qui fuit". Le module de haut niveau est fortement couplé aux détails d'implémentation du module de bas niveau. Tout changement dans le format attendu par le solveur nécessite des modifications dans l'orchestrateur, violant le principe de séparation des responsabilités.

**Alternative Architecturale :** Le `NetworkGrid` devrait passer des données brutes ou un objet de plus haut niveau. Le composant de bas niveau (ou un adaptateur) devrait être responsable de transformer ces données dans le format précis dont il a besoin, cachant ainsi ses détails d'implémentation.

---

### Critique 3 : La Centralisation des Responsabilités dans `NetworkGrid`

**Observation :** La classe `NetworkGrid` gère la topologie, l'orchestration temporelle, les conditions aux limites, les feux de circulation, et la communication avec les solveurs.

**Critique :** La classe s'approche d'un "God Object", avec une cohésion faible et trop de responsabilités. Cela rend le code difficile à comprendre, à maintenir et surtout à tester de manière isolée.

**Alternative Architecturale :** Déléguer les responsabilités à des objets plus spécialisés (`TopologyManager`, `TrafficLightManager`, `BoundaryConditionManager`). `NetworkGrid` deviendrait un simple orchestrateur qui coordonne ces managers, rendant le système plus modulaire et testable.

---

### Critique 4 : Duplication et Incohérence dans la Logique Numérique

**Observation :** La logique mathématique fondamentale (ex: application des conditions aux limites) est répartie à plusieurs endroits, parfois avec des implémentations légèrement différentes.

**Critique :** Toute duplication de la logique physique est un risque. Une correction de bug ou une amélioration doit être reportée manuellement à plusieurs endroits, avec un risque élevé d'oubli et d'incohérence.

**Alternative Architecturale :** La "source de vérité" pour un concept mathématique donné devrait exister en un seul et unique endroit canonique. Les autres parties du code devraient appeler cette fonction ou méthode centralisée.

---

### Critique 5 : La Divergence des Chemins d'Exécution CPU vs. GPU

**Observation :** Le code contient des blocs `if params.device == 'cuda':`, menant à des implémentations distinctes pour les routines numériques sur CPU (Numba) et GPU (CUDA).

**Critique :** Cela crée un fardeau de maintenance significatif ("deux bases de code"), un risque de divergence des résultats entre les plateformes, et double la complexité des tests.

**Alternative Architecturale :** Viser une abstraction du matériel. Utiliser des bibliothèques comme `CuPy` (qui mime l'API NumPy) permet d'écrire la logique numérique une seule fois. Le même code peut alors s'exécuter sur les deux plateformes.

---

### Critique 6 : Contrats Implicites dans les Schémas d'Intégration Temporelle

**Observation :** Les fonctions d'intégration temporelle (ex: `solve_hyperbolic_step_ssprk3`) ne gèrent pas elles-mêmes les conditions aux limites. Elles dépendent de leur appelant (`strang_splitting_step`) pour appliquer ces conditions au bon moment entre chaque sous-étape.

**Critique :** C'est un contrat implicite fragile. La correction du schéma numérique dépend d'une orchestration externe non garantie par la signature de la fonction. Une erreur dans la boucle d'appel mènera à des résultats faux de manière silencieuse.

**Alternative Architecturale :** Un composant numérique devrait être autonome. La fonction d'intégration devrait prendre en argument la *spécification* des BC et se charger elle-même de les appliquer au besoin. Cela la rend plus robuste et réutilisable.

---

### Critique 7 : Manque d'Outils d'Inspection et de Débogage Structurés

**Observation :** Le débogage a nécessité l'injection manuelle et répétée d'instructions `print` dans le noyau numérique.

**Critique :** Le modèle manque d'un mécanisme intégré pour inspecter son état pendant l'exécution. C'est une fonctionnalité essentielle pour les modèles complexes, pas un luxe.

**Alternative Architecturale :** Implémenter un système de "callbacks" (fonctions de rappel). Les solveurs pourraient accepter une liste de fonctions à appeler à des étapes clés, en leur passant l'état actuel. Cela permet de brancher n'importe quelle logique de journalisation, de sauvegarde ou de visualisation de manière non intrusive.

---

### Critique 8 : La Surcharge Temporaire de Paramètres comme Anti-Pattern

**Observation :** La base de code, et `network_grid.py` en particulier, utilise un pattern récurrent qui consiste à sauvegarder une valeur de `self.params`, à la remplacer par une valeur temporaire, à appeler une sous-fonction, puis à restaurer la valeur originale.
```python
# Exemple de pattern observé
saved_value = self.params.attribute
self.params.attribute = temporary_value
sub_function(self.params, ...)
self.params.attribute = saved_value
```

**Critique Sincère :** Ce pattern est un "code smell" (un symptôme de problème de conception) très fort. C'est une manière de "tricher" avec le passage de paramètres. Au lieu de passer l'information nécessaire à une fonction par ses arguments (par la "porte d'entrée"), on la place dans un état global partagé que la fonction doit connaître et lire (par la "porte de derrière").
- **Dépendances Cachées :** La fonction `sub_function` devient impossible à comprendre ou à utiliser seule. Sa correction dépend d'un état qui a été manipulé par son appelant, une dépendance qui n'est visible nulle part dans sa signature.
- **Extrêmement Fragile :** Comme mentionné, ce pattern n'est généralement pas protégé par des blocs `try...finally`. Si une exception survient dans `sub_function`, la restauration n'a jamais lieu, et l'objet `params` est définitivement corrompu pour toutes les opérations futures. Le fait même d'avoir besoin d'un `try...finally` ici est un signe que le design est problématique.
- **Violation d'Encapsulation :** Un objet (`NetworkGrid`) ne devrait pas avoir à connaître et à manipuler les détails internes de la configuration (`params`) de cette manière. La configuration devrait être considérée comme en lecture seule pendant une opération.

**Alternative Architecturale :** La seule alternative propre est le passage explicite de paramètres. Si `sub_function` a besoin d'une valeur spécifique pour `attribute`, elle doit la recevoir en argument : `sub_function(params, attribute=temporary_value, ...)`. Cela rend l'échange d'information clair, direct, prévisible et robuste.

---

### Critique 9 : Corruption des Conditions aux Limites par le Solveur d'EDO

**Observation :** Il a été découvert que le solveur d'EDO (`solve_ode_step_cpu`), responsable de l'application des termes sources (relaxation, etc.), itérait sur l'intégralité du vecteur d'état, y compris les cellules fantômes.

**Critique :** C'est une faille architecturale critique. Le calcul des termes sources de l'EDO doit être strictement confiné au domaine physique. En opérant sur les cellules fantômes, le solveur écrase et corrompt les conditions aux limites qui viennent d'être établies par l'étape du solveur hyperbolique. Cette corruption annule silencieusement tout apport ou sortie de flux aux bords du domaine, brisant la physique de la simulation. L'intégrité des cellules fantômes entre les étapes d'un schéma de splitting est fondamentale.

**Alternative Architecturale :** La boucle du solveur d'EDO doit impérativement itérer uniquement sur les cellules physiques (par exemple, de `grid.num_ghost_cells` à `grid.num_ghost_cells + grid.N_physical`). Les cellules fantômes doivent être traitées comme une zone en lecture seule par les solveurs internes, leur mise à jour étant la responsabilité exclusive du module de conditions aux limites.

---

### Critique 10 : Incohérence dans l'Application de la Logique de Parsing

**Observation :** Une fonction utilitaire (`_parse_bc_state`) a été introduite pour gérer des formats de conditions aux limites hybrides (par exemple, `{ 'rho_m': ..., 'v_m': ... }`). Cependant, la logique d'application des conditions aux limites pour les flux entrants n'utilisait pas cette fonction. Elle continuait de vérifier directement un format différent et plus ancien (`'state': [...]`), rendant le nouveau format inutilisable en pratique.

**Critique :** Cela révèle une implémentation incohérente et un échec à adopter les nouvelles fonctions utilitaires centralisées. Du "code mort" ou inutilisé est créé, et des bugs apparaissent car une fonctionnalité semble implémentée mais n'est jamais appelée. C'est une violation du principe DRY (Don't Repeat Yourself), car la logique de parsing est implicitement éparpillée et incomplète.

**Alternative Architecturale :** Lorsqu'une fonction utilitaire est créée pour centraliser une logique, tous les chemins de code pertinents doivent être immédiatement refactorisés pour l'utiliser. Des tests d'intégration doivent valider que les différents formats de données sont correctement gérés de bout en bout.

---

### Critique 11 : Défaillance en Cascade due à la Communication Implicite par État Mutable

**Observation :** L'échec persistant du test d'intégration (`test_congestion_forms_during_red_signal`) est le résultat direct d'une cascade de défaillances dont la cause première est l'anti-pattern de communication par mutation d'état (`self.params`).

**Critique :** Ce choix de conception n'est pas un simple "code smell" ; il est la cause fondamentale de l'échec physique de la simulation observée.
- **Fragilité Démontrée :** La tentative de passer des conditions aux limites (BC) spécifiques à un segment en surchargeant temporairement `self.params.boundary_conditions` a échoué. La raison est que le contrat d'appel de la fonction sous-jacente (`strang_splitting_step`) était lui-même ambigu, et ne lisait pas cet état temporaire comme attendu. Le système est trop fragile pour qu'on puisse raisonner sur son état.
- **Débogage Impossible :** La communication étant implicite (un "pass-by-side-effect"), il est devenu impossible de tracer le flux de données. Les `print` de débogage semblaient ne pas s'exécuter, non pas à cause d'un problème de logging, mais parce que les chemins de code attendus n'étaient jamais atteints, en raison de ce couplage par état caché.
- **Effet Domino :** Cette unique faille de conception a provoqué une réaction en chaîne fatale :
    1.  `NetworkGrid` échoue à communiquer la condition d'entrée (`inflow`) au solveur du segment.
    2.  Le solveur, ne recevant aucune instruction, applique sa condition par défaut : une sortie (`outflow`).
    3.  La condition de sortie extrapole la densité depuis l'intérieur du domaine (qui est de 0).
    4.  Le calcul de flux à la frontière du domaine résulte en un flux nul.
    5.  Aucune masse (véhicule) n'entre jamais dans le segment.
    6.  Le test, qui s'attend à une augmentation de la densité, échoue logiquement.

**Conclusion Architecturale :** Le **vrai problème** est que l'architecture repose sur une communication implicite et étatiste au lieu d'un passage de paramètres explicite et fonctionnel. Cela rend le système opaque, cassant, et impossible à déboguer. La solution n'est pas d'ajouter plus de rustines (`try...finally`) ou une gestion d'état plus complexe, mais de **refactoriser l'interaction fondamentale**. `NetworkGrid` doit appeler le solveur de segment avec des paramètres explicites, rendant le contrat clair, le flux de données traçable, et le système robuste. Exemple : `segment.evolve(dt, boundary_conditions=segment_bc)`.
