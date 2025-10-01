
## Compte-Rendu Hebdomadaire : Choix et Analyse du Corridor d'Étude Principal


### **1. Objectif du Rapport :**

Ce compte-rendu présente le corridor urbain que nous avons choisi pour notre étude. Il détaille pourquoi ce choix s'impose, en le reliant aux caractéristiques du trafic en Afrique de l'Ouest et aux besoins de notre modèle ARZ. L'idée est de montrer que notre sélection est bien justifiée et pertinente pour le projet.

### **2. Choix du Corridor Pilote : L'axe Principal de Victoria Island, Lagos**

Après avoir étudié différentes options dans les villes d'Afrique de l'Ouest, et en tenant compte des données disponibles via TomTom, nous avons décidé de nous concentrer sur l'axe formé par **Akin Adesola Street et Adeola Odeku Street**, au cœur de **Victoria Island (VI)** à Lagos.

Ce corridor est particulièrement intéressant car il réunit les principaux défis que nous voulons modéliser :
*   **Un Mélange Très Varié de Véhicules :** On y trouve de tout : voitures, bus, tricycles, et surtout une majorité de motos ("Okada"). C'est parfait pour tester notre modèle multi-classes.
*   **Des Embouteillages Constants :** En tant que quartier d'affaires, VI est toujours très fréquenté. Le trafic y est dense, ce qui nous permettra d'observer le "creeping" des motos dans les bouchons et comment les motos se faufilent ("gap-filling", "interweaving").
*   **Beaucoup d'Intersections avec Feux :** Le corridor est traversé par plusieurs grandes intersections qui sont toutes régulées par des feux. C'est donc un endroit idéal pour tester notre système d'optimisation par IA.
*   **Des Routes d'État Variable :** On voit bien que toutes les routes ne sont pas en parfait état. Certaines sont bien entretenues, d'autres un peu moins. Ça correspond bien à ce qu'on veut modéliser avec l'impact de l'infrastructure (`R(x)`).

#### **2.1. Description Précise du Corridor**

Le corridor fait environ 2 à 3 kilomètres de long. Il comprend deux rues principales et plusieurs intersections importantes :

*   **L'artère principale (Nord-Sud) :** **Akin Adesola Street**. C'est une rue majeure dans le quartier.
*   **L'artère principale (Est-Ouest) :** **Adeola Odeku Street**. C'est une avenue très fréquentée qui traverse le centre économique.
*   **Intersection Clé 1 (le cœur du sujet) :** Là où **Akin Adesola croise Adeola Odeku**. C'est un gros carrefour avec des feux.
*   **Intersection Clé 2 :** L'intersection d'**Akin Adesola avec Saka Tinubu Street**. Une autre rue importante.
*   **Intersection Clé 3 :** L'accès à **Ahmadu Bello Way**, une grande route le long de la côte.

#### **2.2. Lien avec les Paramètres de Notre Modèle ARZ**

L'analyse de ce corridor nous permet de justifier les paramètres de notre modèle :

*   **Impact de l'Infrastructure (`R(x)`) :**
    *   En regardant sur Google Maps, on voit bien que toutes les routes ne sont pas pareilles. Certaines sont bien lisses (on peut dire `R=1` ou `R=2`), d'autres sont plus abîmées (`R=3` ou `R=4`).
    *   *Pour le modèle :* Ça valide bien l'idée que la vitesse maximale des motos et voitures (\(V_{max,i}\)) doit dépendre de la qualité de la route (\(R(x)\)). On va pouvoir cartographier ça sur notre corridor.

    *   **[EMPLACEMENT POUR UNE CARTE DU CORRIDOR AVEC TYPES DE ROUTES]**
        *   *Ce qu'il faudrait mettre ici :* Une carte de Victoria Island montrant clairement Akin Adesola, Adeola Odeku, Saka Tinubu, Ahmadu Bello Way. On y indiquera avec des couleurs différentes les segments selon qu'ils sont en bon état (`R=1`/`R=2`) ou plus dégradés (`R=3`/`R=4`). Les intersections clés seront aussi marquées.
        *   *Légende suggérée :* "Carte du corridor d'étude à Victoria Island, Lagos. Les couleurs indiquent la qualité de la route (\(R\)) pour chaque segment."

*   **Comportements Spécifiques des Motos (`α`, \(V_{creeping}\), \(\tau_m\)) :**
    *   Ce qu'on voit tous les jours ici, c'est que les motos se faufilent partout ("gap-filling", "interweaving"). Elles utilisent tous les petits espaces pour avancer, même quand ça bloque.
    *   Quand il y a de gros embouteillages, on les voit avancer doucement mais sûrement : c'est le **"creeping"**.
    *   *Pour le modèle :* Ça confirme que :
        *   On doit utiliser un `α < 1` dans le calcul de la densité perçue par les motos (`p_{eff,m} = p_m + \alpha p_c`) pour montrer qu'elles sont moins gênées par les voitures.
        *   Il faut une vitesse de "creeping" \(V_{creeping} > 0\) pour les motos, pour qu'elles puissent bouger même dans les bouchons.
        *   Et peut-être que le temps de réaction des motos \(\tau_m\) est plus court que celui des voitures \(\tau_c\), vu leur agilité.

  

*   **Intersections et Files d'Attente :**
    *   Il y a pas mal de feux de signalisation dans ce corridor, et aux intersections, on voit souvent des files d'attente se former. Les motos se placent souvent devant tout le monde ("front-loading"), ce qui complique la donne.
    *   *Pour le modèle :* C'est là qu'intervient notre modèle d'intersection. Il faut qu'il puisse bien gérer ces arrivées de flux, les feux, et les files d'attente, pour avoir une image réaliste de la circulation sur tout le corridor.


### **3. Données Disponibles et Stratégie de Collecte**

#### **3.1. Source Principale : API TomTom**
*   **Ce qu'on obtient :** Des infos sur les vitesses moyennes, les temps de parcours, et des indicateurs de congestion pour les routes.
*   **Ce qui manque :** On ne sait pas combien il y a de motos et de voitures séparément. C'est un souci pour bien caler notre modèle.
*   **Notre approche :** On utilisera ces données globales pour avoir une idée des vitesses générales. Ensuite, on combinera ça avec ce qu'on voit sur le terrain et ce qu'on lit dans les études pour estimer le nombre de motos et de voitures, et les comportements des conducteurs.

#### **3.2. Autres Sources d'Informations :**
*   **Google Maps (Street View & Trafic) :** Indispensable pour voir le type de route (`R(x)`), son état, comment sont les intersections, et comment les gens conduisent. Les vitesses indiquées par Google nous donneront une première idée pour \(V_{max,i}\).
*   **OpenStreetMap (OSM) :** Très utile pour avoir les cartes précises des rues et savoir quel type de route c'est (`fclass`), ce qui nous aidera pour \(R(x)\).
*   **Littérature Scientifique et Observations Locales :** On va se baser sur ce qui a déjà été étudié sur le trafic au Bénin, au Nigeria, et dans d'autres pays d'Afrique de l'Ouest pour faire des hypothèses raisonnables sur les paramètres comportementaux (\(\alpha, V_{creeping}, \tau_m/\tau_c, P_i\)), vu qu'on n'a pas de données précises pour tout calibrer.

### **4. Conclusion et Prochaines Étapes**

Le corridor choisi à Victoria Island est parfait pour notre projet. Il montre bien tous les aspects du trafic que l'on veut étudier, et grâce à TomTom, on aura des données pour commencer.

Pour la suite :
1.  On va bien définir toutes les rues et intersections du corridor pour le simulateur.
2.  On lance le script pour récupérer les données TomTom pour cet axe précis.
3.  On commence à regarder les données OSM pour classifier les routes (`R(x)`).
4.  On commencera à régler les premiers paramètres du modèle en se basant sur tout ça.

