# Descriptions des Figures pour la Thèse (Chapitre 7)

Voici une sélection des trois diagrammes de Hovmöller les plus pertinents pour illustrer la dynamique du modèle ARZ multi-classes, accompagnés de descriptions concrètes pour le manuscrit.

## 1. Choc Simple (Motos)
**Fichier :** `heatmap_choc_simple_motos.png`

**Description :**
> "La Figure 7.1 illustre la propagation d'une onde de choc pure dans le flux des motos. Initialement, une discontinuité sépare un état de faible densité (aval) d'un état de forte densité (amont). Au cours du temps, cette discontinuité se propage vers l'amont (x décroissant) tout en conservant un profil net, caractéristique des solutions faibles entropiques des lois de conservation hyperboliques.
>
> Sur les cartes de chaleur, on observe clairement la frontière abrupte (le choc) où la vitesse des motos chute instantanément de 8 m/s à environ 6 m/s, tandis que la densité augmente. La linéarité de la trajectoire du choc dans le plan espace-temps confirme que la vitesse de propagation de l'onde est constante, conformément à la condition de Rankine-Hugoniot."

## 2. Détente (Voitures)
**Fichier :** `heatmap_detente_voitures.png`

**Description :**
> "La Figure 7.2 présente le cas d'une onde de détente (ou raréfaction) pour la classe des voitures. Contrairement au choc, la transition initiale entre l'état dense et l'état fluide ne reste pas abrupte mais s'étale progressivement au cours du temps.
>
> Le diagramme de Hovmöller montre un éventail de caractéristiques (zone en forme de V) où la densité diminue continûment et la vitesse augmente. Physiquement, cela correspond à des véhicules qui accélèrent pour s'éloigner d'une zone congestionnée, créant ainsi de l'espace entre eux. L'étalement de l'onde démontre la nature dispersive de la détente dans le modèle ARZ, satisfaisant la condition d'entropie de Lax."

## 3. Interaction Multi-classes
**Fichier :** `heatmap_interaction_multiclasse.png`

**Description :**
> "La Figure 7.3 met en évidence le couplage dynamique entre les deux classes de véhicules, qui constitue l'apport principal de ce modèle. Le scénario initialise une perturbation de densité différente pour les motos et les voitures.
>
> On observe la formation d'une structure d'ondes complexe résultant de l'interaction : les variations de vitesse des voitures induisent une modification du comportement des motos via le terme de relaxation (friction dynamique). Le diagramme montre que les ondes ne se propagent pas indépendamment ; les discontinuités d'une classe se réfractent au travers des ondes de l'autre classe, illustrant l'échange de quantité de mouvement entre les populations. Ce comportement est crucial pour modéliser réalistement le trafic hétérogène en milieu urbain dense."
