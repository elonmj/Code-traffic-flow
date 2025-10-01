**\section{Validation de l'Environnement d'Entraînement}**
*   **Objectif de la validation :** S'assurer que l'environnement est stable, réactif et exempt de bugs avant de lancer un entraînement coûteux.
*   **Test de Stabilité avec un Agent Aléatoire :**
    *   **Méthodologie :** Lancer un agent qui choisit des actions au hasard pendant un grand nombre de `steps`.
    *   **Critères de succès :** Absence d'erreurs d'exécution, valeurs d'observation et de récompense dans des plages plausibles.
*   **Test de Réactivité avec un Contrôleur de Référence (Baseline) :**
    *   **Méthodologie :** Implémenter un agent simple qui suit une politique de contrôle à cycles fixes (la méthode traditionnelle).
    *   **Critères de succès :**
        1.  Observer la formation et la dissipation logiques des files d'attente en fonction des phases de feux.
        2.  Vérifier que le système atteint un état d'équilibre stationnaire sans que les densités ne divergent.
    *   **(Placeholder pour un graphique montrant l'évolution des files d'attente avec le contrôleur fixe)**
*   **(Placeholder pour un tableau récapitulatif des tests de validation et de leurs résultats)**



## Validation de l'environnement RL

### Objectifs de validation

* Vérifier que l'environnement reproduit fidèlement le comportement du jumeau numérique calibré (conservation de masse, ondes de choc, creeping).
* Vérifier la stabilité numérique et l'absence d'artéfacts dues à l'interaction agent/sim (p.ex. actions impossibles, violation contraintes).;

### Tests recommandés

1. **Test de cohérence physique** : en mode piloté (script), appliquer une séquence de phases connue et comparer sorties (densité / flux) à une exécution standalone du simulateur. Erreur tolérée < tol\_physique (ex. 1%).
2. **Test d'edge cases** : très forte demande, incidents, variation brusque d'arrivée — vérifier que l'environnement ne crash pas et renvoie des `info` utiles.
3. **Test de reproductibilité** : fixé `seed` → mêmes trajectoires.
4. **Bench RL minimal** : faire tourner un agent aléatoire et vérifier que l'histogramme des rewards correspond à l'intuition (moyenne négative stable) ; ensuite lancement d'un DQN/Double DQN (chap. 10) pour vérifier progression. (Ton chapitre 10 détaille l'entraînement Double DQN).;

### KPI à remonter dans `info`

* débit moyen (veh/h) par classe ;
* longueur moyenne des files par lien ;
* temps moyen d'attente par classe ;
* nombre de changements de phase (frequency) ;
* métriques CPU / temps réel par step (pour optimisation).

---