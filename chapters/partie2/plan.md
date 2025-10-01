

### **Plan de Travail : Développement et Calibration du Jumeau Numérique sur le Corridor de Victoria Island, Lagos**

**Objectif Général :** Transformer le corridor pilote d'Akin Adesola / Adeola Odeku en un environnement de simulation numérique (jumeau numérique) fonctionnel, calibré et validé. Ce jumeau numérique servira de base pour l'entraînement d'un agent d'intelligence artificielle visant à optimiser les feux de signalisation.

---

#### **Phase 1 : Construction et Discrétisation de l'Environnement Numérique (Semaines 1-2)**

L'objectif de cette phase est de créer une représentation digitale fidèle de la topologie et des caractéristiques statiques du corridor.

*   **Tâche 1.1 : Modélisation Géographique du Réseau**
    *   **Action :** Utiliser OpenStreetMap (OSM) et les images satellites (Google Maps) pour extraire la topologie exacte du réseau : géométrie des rues, nombre de voies, localisation précise des intersections clés (Akin Adesola/Adeola Odeku, Akin Adesola/Saka Tinubu, etc.).
    *   **Livrable :** Un graphe du réseau définissant les segments (arcs) et les intersections (nœuds).

*   **Tâche 1.2 : Classification de l'Infrastructure et Définition de `R(x)`**
    *   **Action :** Pour chaque segment du graphe, assigner une valeur à l'indicateur de qualité de la route `R(x)`. Utiliser Google Street View pour une évaluation visuelle (chaussée lisse, pavée, dégradée, nids-de-poule) et les attributs `fclass` d'OSM.
    *   **Exemple de classification :** `R=1` (excellent état), `R=2` (bon), `R=3` (moyen), `R=4` (dégradé).
    *   **Livrable :** Une carte du corridor où chaque segment est annoté avec sa valeur `R(x)`.

*   **Tâche 1.3 : Paramétrage des Intersections**
    *   **Action :** Pour chaque nœud du réseau, documenter sa configuration : type (carrefour à feux, non signalisé), phases de feux actuelles (si connues), mouvements autorisés (tourne-à-gauche, tout droit), etc.
    *   **Livrable :** Une fiche de configuration pour chaque intersection, prête à être implémentée dans le modèle.

---

#### **Phase 2 : Acquisition et Prétraitement des Données Dynamiques (Semaines 2-4)**

Cette phase se concentre sur la collecte des données de trafic qui donneront vie au jumeau numérique.

*   **Tâche 2.1 : Déploiement de la Collecte de Données via l'API TomTom**
    *   **Action :** Développer et exécuter des scripts pour interroger périodiquement l'API TomTom et collecter les données de trafic (vitesses moyennes, temps de parcours, niveau de congestion) pour les segments définis dans la Tâche 1.1.
    *   **Livrable :** Une base de données temporelles contenant les indicateurs de trafic pour l'ensemble du corridor.

*   **Tâche 2.2 : Estimation de la Composition du Trafic (Multi-classes)**
    *   **Action :** Comme les données TomTom sont agrégées, estimer la proportion de motos (classe *m*) et d'autres véhicules (classe *c*).
    *   **Méthode :** Combiner une revue de la littérature sur le trafic à Lagos avec des analyses observationnelles (par exemple, analyse d'images/vidéos du corridor à différentes heures pour compter manuellement les proportions).
    *   **Livrable :** Des profils de composition du trafic typiques (par exemple, 75% motos en heure de pointe, 60% en heure creuse).

*   **Tâche 2.3 : Définition des Conditions aux Limites**
    *   **Action :** Utiliser les données de flux inférées de TomTom et les estimations de composition pour définir les conditions d'entrée et de sortie du corridor (demande de trafic aux extrémités du réseau).
    *   **Livrable :** Des fonctions de demande en entrée du corridor pour chaque classe de véhicule, variant au cours d'une journée type.

---

#### **Phase 3 : Calibration Hiérarchique du Modèle ARZ Étendu (Semaines 5-8)**

C'est le cœur du projet. L'objectif est de régler systématiquement les paramètres du modèle pour que ses simulations correspondent aux données réelles.

*   **Tâche 3.1 : Calibration des Paramètres Physiques Fondamentaux**
    *   **Action :** Calibrer les paramètres de base à l'aide des données les plus directes.
        *   **Vitesse maximale (`V_max,i`) :** Estimer à partir des vitesses TomTom en heures très creuses, en fonction de la qualité de la route `R(x)`.
        *   **Densité de blocage (`ρ_jam,i`) :** Utiliser les valeurs de la littérature et les affiner par observation des files d'attente maximales aux feux.
    *   **Livrable :** Valeurs calibrées pour `V_max,i(R)` et `ρ_jam,i`.

*   **Tâche 3.2 : Calibration des Paramètres Comportementaux Spécifiques aux Motos**
    *   **Action :** Ajuster les paramètres qui modélisent les comportements uniques des motos.
        *   **Vitesse de "creeping" (`V_creeping`) :** Régler cette valeur pour que la vitesse simulée des motos en forte congestion corresponde aux faibles vitesses observées dans les données TomTom.
        *   **Gap-filling (`α`) et temps de réaction (`τ_m`) :** Régler ces paramètres de manière itérative. L'objectif est que le débit global et la manière dont les congestions se forment et se dissipent dans la simulation correspondent à la dynamique observée dans les données.
    *   **Livrable :** Valeurs calibrées pour `V_creeping`, `α(ρ)`, `τ_m(ρ)`.

*   **Tâche 3.3 : Calibration des Paramètres d'Intersection (`θ_k`)**
    *   **Action :** Mettre en œuvre la méthodologie de calibration hybride décrite dans votre thèse.
        1.  **Créer le "laboratoire numérique" :** Modéliser l'intersection clé (Akin Adesola/Adeola Odeku) dans un simulateur microscopique comme SUMO, en utilisant la composition de trafic estimée (Tâche 2.2).
        2.  **Générer des données de référence :** Simuler des cycles de feux pour extraire les profils de vitesse et de densité en amont et en aval de l'intersection.
        3.  **Optimiser `θ_k` :** Reconstruire la variable lagrangienne `w` à partir des sorties de SUMO et trouver les `θ_k` (pour motos et voitures) qui minimisent l'erreur entre le `w` prédit par votre modèle de couplage et le `w` reconstruit.
    *   **Livrable :** Valeurs calibrées de `θ_k` pour chaque classe de véhicule aux intersections à feux.

---

#### **Phase 4 : Validation Globale et Analyse de Sensibilité (Semaines 9-10)**

Cette phase vise à s'assurer que le modèle est non seulement calibré, mais aussi robuste et fiable.

*   **Tâche 4.1 : Validation Croisée**
    *   **Action :** Simuler une période (un jour ou une semaine) qui n'a pas été utilisée pour la calibration. Comparer les sorties du jumeau numérique (temps de parcours, vitesses moyennes par segment) avec les données TomTom correspondantes.
    *   **Livrable :** Métriques d'erreur (ex: RMSE, MAPE) et graphiques comparatifs démontrant la performance prédictive du modèle.

*   **Tâche 4.2 : Validation Phénoménologique**
    *   **Action :** Vérifier qualitativement si le jumeau numérique reproduit les phénomènes de trafic connus du corridor : localisation des goulots d'étranglement récurrents, formation de files d'attente aux intersections, effet visible du "creeping" des motos, etc.
    *   **Livrable :** Un rapport d'analyse qualitative confirmant le réalisme du comportement du modèle.

*   **Tâche 4.3 : Analyse de Sensibilité**
    *   **Action :** Faire varier les paramètres les plus incertains (ex: `α`, `β`, `θ_k`) dans une plage de valeurs plausibles pour évaluer leur impact sur les résultats de la simulation.
    *   **Livrable :** Une analyse identifiant les paramètres les plus influents, ce qui aide à comprendre les limites et la robustesse du jumeau numérique.

---

#### **Phase 5 : Intégration à l'Environnement d'Apprentissage par Renforcement (Semaines 11-12)**

La phase finale consiste à préparer le jumeau numérique pour qu'il devienne le terrain d'entraînement de l'agent d'IA.

*   **Tâche 5.1 : "Gym-ification" du Simulateur**
    *   **Action :** Encapsuler le simulateur du jumeau numérique dans une interface standard de type OpenAI Gym/Gymnasium. Cela implique de créer une classe `Env` avec les méthodes `reset()`, `step()`, `render()`.
    *   **Livrable :** Un environnement `RL` prêt à l'emploi.

*   **Tâche 5.2 : Définition de l'Espace (État, Action, Récompense)**
    *   **Action :** Formaliser précisément l'interaction entre l'agent et l'environnement.
        *   **État :** Vecteur contenant les informations pertinentes (ex: densités et vitesses des 50 derniers mètres avant chaque feu, phase de feu actuelle).
        *   **Action :** Espace d'actions possibles (ex: `[passer_a_la_phase_suivante, maintenir_phase_actuelle]`).
        *   **Récompense :** Fonction mathématique à optimiser (ex: récompense négative proportionnelle au temps d'attente total de tous les véhicules, ou positive proportionnelle au flux de sortie).
    *   **Livrable :** Spécification complète de l'environnement RL.

*   **Tâche 5.3 : Implémentation de la Politique de Référence**
    *   **Action :** Modéliser la stratégie de contrôle des feux actuelle (probablement un plan à temps fixe) dans le simulateur.
    *   **Livrable :** Une performance de référence (temps de trajet moyen, etc.) contre laquelle l'agent RL devra prouver son efficacité.