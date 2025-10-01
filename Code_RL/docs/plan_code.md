
Dossier de conception et d’implémentation – Environnement RL de contrôle des feux (avec simulateur ARZ externe)

1) Résumé exécutif
- But: entraîner un agent RL à contrôler des feux de signalisation via un environnement Gymnasium, en s’appuyant sur un simulateur ARZ multi-classes externe.
- Cible V1: un carrefour (ou petit groupe), actions discrètes maintenir/passer, phases prédéfinies, sécurité assurée par une couche “feux”.
- Résultats attendus: réduction des temps d’attente et des files, stabilité des cycles, reproductibilité expérimentale.

2) Objectifs et critères de succès
- Réduction du temps d’attente moyen ≥ 15% vs baseline fixe.
- Diminution de la longueur de file max ≥ 10%.
- Stabilité: ≤ 8 changements de phase/heure (ou seuil métier).
- Reproductibilité: résultats stables (écart-type < 5%) sur 5 seeds.
- Robustesse: dégradation < 10% en scénarios non vus (demande/latences).

3) Périmètre et hors périmètre
- Périmètre: contrôle des feux; intégration à un simulateur ARZ via endpoint; entraînement et évaluation RL.
- Hors périmètre V1: modification du solveur ARZ; coordination multi-intersections avancée; actions continues; déploiement temps réel.

4) Hypothèses et contraintes
- Simulateur ARZ accessible via endpoint (HTTP/IPC/py-binding) avec pas fin Δt_sim ≤ 1 s.
- Décision RL toutes Δt_dec = 10 s; k = Δt_dec / Δt_sim entier.
- Données disponibles: états macro par branche (ρ, v), files, flux; mapping clair entre branches et phases.
- Sécurité opérationnelle obligatoire: min-green, intergreen (jaune+all-red), max-green fail-safe, interdiction conflits.

5) Exigences
- Fonctionnelles:
  - Appliquer une action Discrete(2): 0=maintenir, 1=passer à la phase suivante.
  - Produire observations normalisées, récompenses, KPIs.
  - Enregistrer logs d’épisode et configs.
- Non-fonctionnelles:
  - Robustesse aux timeouts et erreurs endpoint.
  - Débit suffisant pour entraînement (≥ 5–10 env steps/s).
  - Tests unitaires et intégration; conformité Gymnasium.

6) Architecture (logique)
- endpoint client: contrat minimal reset/set_signal/step/get_metrics/health.
- signals: phases, timers, safety layer (règles dures).
- env Gymnasium: orchestration RL (reset/step, observation, reward).
- reward: calcul pondéré, clipping.
- eval: KPIs et reporting.
- utils: config centralisée, seeds, logging.
- rl: script d’entraînement DQN baseline (exemple).

7) Synchronisation temporelle
- Δt_sim (solveur) << Δt_dec (RL). k = Δt_dec / Δt_sim (valide via config).
- Séquences transitoires (jaune/all-red) insérées par la safety layer; peuvent occuper tout/partie d’une fenêtre Δt_dec.
- Agrégation des métriques sur la fenêtre Δt_dec.

8) Spécification de l’endpoint ARZ (contrat d’interface)
- reset(scenario, seed) → state0, t0
- set_signal(signal_plan) → ack
- step(dt, repeat_k) → state, t
- get_metrics() → agrégats (si non inclus dans state)
- health() → statut/latence
- État (par branche et classe): rho_m, v_m, rho_c, v_c, queue_len, flow (+ option phase_id).
- Erreurs: timeouts, invalid command; stratégie: retries bornés, fail-fast avec info d’épisode.

9) Modèle des feux et safety layer
- Phases: liste ordonnée, mouvements non conflictuels.
- Timers: min_green, max_green (fail-safe), yellow, all_red.
- Safety layer:
  - masque le “passer” si min_green non atteint;
  - impose séquences yellow/all_red en transition;
  - force fin de vert à max_green;
  - interdit conflits.
- Traçabilité: log de la timeline des phases et timers.

10) Spécification de l’environnement Gymnasium
- action_space: Discrete(2).
- observation_space: Box(normalisé, float32).
- reset():
  - endpoint.reset, init phases/timers, construire observation.
- step(action):
  - safety layer → action effective + transitions;
  - set_signal, endpoint.step(dt, k);
  - agrégation métriques, observation, reward, terminated/truncated, info.
- Wrappers utiles: Normalization, ActionMasking (option), TimeLimit.

11) Observation et normalisation
- Par branche observée: rho_m/ρmax, v_m/vfree, rho_c/ρmax, v_c/vfree, queue/Lmax.
- Signal: phase one-hot + temps depuis début de vert (normalisé).
- Normalisation systématique; lissage EWMA léger optionnel.
- Consistance: mêmes dimensions entre épisodes; mapping branches↔endpoint en config.

12) Récompense (conception)
- r = −(w1·attente_moy + w2·sum(files) + w3·stops_proxy) − w4·switch_penalty + w5·throughput
- stops_proxy: détection vitesse < seuil.
- switch_penalty: pénalise la commutation; inactive si safety force.
- throughput: flux sortant agrégé (positif).
- Clipping/mise à l’échelle; pondérations configurables.

13) Configuration (spécification)
- endpoint.yaml: protocole, URL/IPC, Δt_sim, timeouts, retries.
- signals.yaml: définition des phases, min/max_green, yellow, all_red.
- network.yaml: branches observées, mapping IDs ↔ endpoint.
- env.yaml: Δt_dec, normalisations (ρmax, vfree, Lmax), pondérations de récompense, seeds.
- Convention: validations à l’initialisation, erreurs explicites si incohérences.

14) KPIs et reporting
- KPIs par épisode: temps d’attente moyen, longueur de file max/moy, throughput, nb de changements de phase, récompense cumulée.
- Par classe: métriques spécifiques motos vs voitures (équité).
- Reporting: fichiers CSV/JSON, courbes synthèse.

15) Stratégie de tests et validation
- Conformité Gymnasium: espaces, reset/step signatures, determinism (seed).
- Unitaires:
  - safety layer: min/intergreen/max/no-conflict;
  - reward: monotonicité vs files/attente;
  - client: timeouts/retries, mocks avec latences et erreurs.
- Intégration: boucle complète (mock endpoint → env) et (endpoint réel → env).
- Propriétés physiques (si métriques disponibles): conservation masse (tolérance), non-négativité.

16) Protocole expérimental (entraînement/évaluation)
- Entraînement: DQN baseline (MLP 2–3 couches), epsilon-greedy, target network, replay buffer, Huber loss, Adam.
- Scénarios: demandes variées (pics/plateaux); seeds multiples (≥5).
- Évaluation: moyenne et écart-type des KPIs; comparatif vs baseline fixe.
- Robustesse: scénarios non vus (domain shift léger).
- Arrêt: early stopping sur validation (récompense moyenne stabilisée).

17) Journalisation, suivi et reproductibilité
- Logs épisodes: KPIs, transitions de phases, récompenses.
- Artefacts: configs actives, seeds, checkpoints, version code (hash).
- Scripts: entraînement et évaluation “one-click” avec configuration paramétrable.

18) Performance et résilience
- Batch des sous-pas: step(repeat_k) côté endpoint; agrégation côté client.
- Monitoring: FPS, latences; ajustement Δt_dec/k si nécessaire.
- Tolérance erreurs: arrêt propre d’épisode, info clair; backoff pour retries.

19) Livrables V1
- Client endpoint robuste + mocks.
- SignalController avec safety layer complet.
- TrafficSignalEnv (Gymnasium) opérationnel.
- Configs YAML d’exemple pour 1 carrefour.
- Tests unitaires clés; script d’entraînement DQN baseline.
- Rapport d’évaluation: KPIs vs baseline fixe, analyse stabilité.

20) Risques et mitigations
- Latences/erreurs endpoint: retries bornés, timeouts, fail-fast par épisode.
- Incohérences de mapping: validations fortes des configs au démarrage.
- Overfitting scénario unique: scénarios variés + seeds multiples.
- Instabilité RL: clipping récompense/gradient, target network, pénalité de switch.

21) Roadmap (post-V1)
- Multi-intersections simple (indépendants); actions enrichies (durées).
- Comparatif algos (Double/Dueling DQN, PPO).
- Meilleure estimation de files; métriques additionnelles.

22) Glossaire
- Δt_sim: pas de simulation ARZ. Δt_dec: pas de décision RL. k = Δt_dec/Δt_sim.
- Phase: ensemble de mouvements non conflictuels. Intergreen: jaune + all-red.
- Observation: vecteur d’état normalisé fourni à l’agent.
- Reward: signal d’optimisation combinant attente, files, stops, throughput.

Tout est cohérent avec votre chapitre de conception: espace d’états basé sur grandeurs ARZ, actions discrètes maintenir/passer, Δt_dec=10 s, interface Gymnasium standard, KPIs définis, sécurité assurée au niveau “feux”, simulateur ARZ traité comme endpoint externe.