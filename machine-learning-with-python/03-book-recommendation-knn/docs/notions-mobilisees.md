# Notions mobilisées par le projet

Ce que cet exercice fait travailler, notion par notion, avec le lien vers le code qui
l'implémente. L'énoncé freeCodeCamp donne les instructions sans jamais les expliquer : ce document
comble cet écart.

Distinct de [enonce-freecodecamp.md](enonce-freecodecamp.md), qui est le cahier des charges
imposé et ne doit pas être commenté.

Les références `recommender.py:NN` pointent sur les lignes exactes au moment de la rédaction. Elles
se décalent dès que le fichier est modifié : en cas de doute, chercher le nom de la fonction ou de
la constante plutôt que de se fier au numéro. Ici, chaque ligne citée est en outre verrouillée par
`test_units.py` (classe `TestReferencesDeLaDocumentation`), qui échoue si l'ancre a bougé.

## Table des matières

- [1. Le filtrage collaboratif, et ce qu'il n'est pas](#1-le-filtrage-collaboratif-et-ce-quil-nest-pas)
- [2. Le jeu Book-Crossings : trois pièges de données réelles](#2-le-jeu-book-crossings--trois-pièges-de-données-réelles)
  - [Le piège central : 62,3 % des notes valent 0](#le-piège-central--623--des-notes-valent-0)
- [3. Le filtrage par seuils et la significativité statistique](#3-le-filtrage-par-seuils-et-la-significativité-statistique)
- [4. La matrice creuse et le pivot](#4-la-matrice-creuse-et-le-pivot)
- [5. La distance cosinus, et l'invariance qu'elle offre vraiment](#5-la-distance-cosinus-et-linvariance-quelle-offre-vraiment)
- [6. Les k plus proches voisins : un algorithme sans apprentissage](#6-les-k-plus-proches-voisins--un-algorithme-sans-apprentissage)
- [7. Le faux piège : l'ordre de la liste de sortie](#7-le-faux-piège--lordre-de-la-liste-de-sortie)
- [8. Le contrat freeCodeCamp et la séparation des préoccupations](#8-le-contrat-freecodecamp-et-la-séparation-des-préoccupations)
- [9. Ce que la mesure a effectivement montré](#9-ce-que-la-mesure-a-effectivement-montré)
- [10. Notions voisines, non implémentées ici](#10-notions-voisines-non-implémentées-ici)

## 1. Le filtrage collaboratif, et ce qu'il n'est pas

Le projet appartient à une famille précise : le **filtrage collaboratif item-item**. Il vaut de la
situer, car le vocabulaire des projets 01 et 02 ne s'y applique pas.

Dans le projet 02, chaque image portait son information en elle : un tenseur de pixels suffisait à
décider chat ou chien. Ici, un livre pris isolément ne contient **rien**. Son titre n'est pas lu,
son auteur non plus, son résumé encore moins. Ce qui le caractérise, c'est uniquement la colonne de
qui l'a noté. L'information est donc **relationnelle** : elle vit dans les chevauchements entre
profils, pas dans les objets.

Deux conséquences que l'on retrouvera plus bas :

- **le démarrage à froid** (*cold start*) est structurel. Un livre que personne n'a noté n'a aucun
  voisin possible ; aucune quantité de calcul n'y remédie. C'est ce qui justifie le filtrage de la
  section 3, et ce qui rend `KeyError` la bonne réponse à un titre absent, plutôt qu'une liste vide ;
- **la popularité contamine la similarité**. Un livre noté par beaucoup de monde a des chances de
  chevaucher n'importe quel autre profil, indépendamment de tout goût partagé. La section 9 le mesure.

On distingue le filtrage **item-item** (comparer les livres entre eux, ce que fait ce projet) du
filtrage **user-user** (comparer les utilisateurs). Le choix n'est pas neutre : les profils de
livres sont plus stables dans le temps que ceux des utilisateurs, et il y a ici 673 livres pour 888
utilisateurs, donc une matrice de distances de 673² au lieu de 888², soit 42,6 % de cases en moins.
Amazon a documenté ce choix dès 2003 pour exactement ces raisons (voir Sources).

## 2. Le jeu Book-Crossings : trois pièges de données réelles

Notes annoncées de 1 à 10 sur 271 379 livres. L'énoncé parle de 90 000 utilisateurs ; le fichier en
contient en réalité **105 283** distincts, l'écart venant de l'arrondi de freeCodeCamp et non du
chargement.

Trois particularités que le projet ne peut pas ignorer :

- **les fichiers ne sont pas du CSV au sens standard** : séparateur `;`, encodage ISO-8859-1, champs
  quotés. `load_data` fixe ces choix à la lecture
  ([recommender.py:48](../recommender.py#L48)). Se tromper d'encodage ne lève aucune erreur, cela
  corrompt silencieusement les titres accentués, qui deviennent alors des livres distincts ;
- **les notes sont très creuses** : chaque utilisateur ne note qu'une infime fraction du catalogue,
  d'où le filtrage (section 3) et la parcimonie (section 4) ;
- **le 0 domine le jeu**, et c'est le point que l'énoncé passe entièrement sous silence.

### Le piège central : 62,3 % des notes valent 0

La valeur `0` n'existe pas sur l'échelle 1-10 annoncée. Elle est pourtant portée par **62,3 % des
notes brutes**, et **74,6 % de celles retenues après filtrage**.

Ce ne sont pas des données corrompues : dans Book-Crossings, `0` encode une **interaction implicite**,
c'est-à-dire un livre que l'utilisateur a signalé avoir lu ou possédé sans lui attribuer de note. La
distinction entre *feedback explicite* (un jugement chiffré) et *feedback implicite* (un clic, un
achat, une lecture) est l'une des lignes de partage majeures de la littérature sur les systèmes de
recommandation, et le jeu mélange les deux dans une même colonne.

Le projet les traite comme des notes ordinaires. C'est ce que fait le corrigé officiel, et le
reproduire est la condition pour passer la cellule de test. Mais il faut nommer ce que cela coûte,
parce que rien dans le code ne le signale : la section 4 en tire les conséquences, et la section 9
les mesure.

## 3. Le filtrage par seuils et la significativité statistique

Sans filtrage, un livre noté deux fois par deux personnes qui n'ont rien noté d'autre a un profil
presque vide : sa distance à tout le reste est quasi constante, il ne peut être « proche » de rien
de façon significative. L'énoncé impose donc de retirer les utilisateurs à moins de 200 notes et
les livres à moins de 100 notes ([recommender.py:93](../recommender.py#L93)).

Mesuré sur le jeu : il reste **673 livres × 888 utilisateurs** et **49 781 notes**, au lieu de
271 379 × 105 283. Le signal est concentré, le reste était trop épars pour être mesurable.

L'ordre de grandeur mérite qu'on s'y arrête : le filtrage élimine **99,8 %** des livres et
**99,2 %** des utilisateurs. Ce n'est pas un dégrossissage, c'est un changement de population. Le
modèle final ne parle que des livres populaires notés par des lecteurs très actifs, et les
recommandations qu'il produit ne sont valides que dans ce sous-monde.

Nuance que la section 2 impose : ce seuil compte des *notes* au sens du fichier, or les trois quarts
d'entre elles valent 0. « Un livre à au moins 100 notes » signifie donc en pratique « un livre avec
lequel au moins 100 utilisateurs ont interagi ». Le filtrage sélectionne la **popularité** bien plus
que l'intensité du jugement.

Un détail d'implémentation qui a son importance : les deux seuils sont appliqués **simultanément**,
sur les comptages calculés avant tout retrait ([recommender.py:105-111](../recommender.py#L105-L111)).
Un filtrage séquentiel (retirer les utilisateurs, recompter, retirer les livres) donnerait un
résultat différent, et itérer jusqu'à point fixe en donnerait un troisième, plus petit. L'énoncé ne
tranche pas ; le corrigé officiel utilise la version simultanée, ce qui verrouille le choix.

## 4. La matrice creuse et le pivot

La forme qu'attend `NearestNeighbors` est une matrice **livres × utilisateurs** : chaque ligne est
le profil d'un livre, chaque colonne celui d'un utilisateur. `build_matrix` pivote les notes et
remplace les valeurs manquantes par 0 ([recommender.py:114](../recommender.py#L114)).

Le choix du 0 est plus profond qu'il n'y paraît. La distance cosinus ne connaît pas la notion
d'absence : elle lit des coordonnées numériques. Mettre 0 est neutre pour le produit scalaire, à la
différence d'une note moyenne, qui ferait exactement le contraire en rapprochant tous les livres par
défaut.

Mais ce `fillna(0)` a un prix qu'il faut nommer : **il rend le 0 ambigu**. Une case à 0 peut
désormais signifier « cet utilisateur n'a pas noté ce livre » aussi bien que « il l'a noté 0 », et
d'après la section 2 le second cas est majoritaire. Le décompte le rend visible :

| Grandeur | Valeur |
|---|---|
| Cases de la matrice | 597 624 |
| Notes retenues par le filtrage | 49 781 |
| Cases non nulles | 12 425 |
| Parcimonie effective | 2,08 % |

Des 49 781 notes retenues, 37 141 valent déjà 0 (74,6 %). Le chemin vers la
matrice en perd 264 dès le rattachement des ISBN (le fichier des livres ignore
certains ISBN) puis 381 doublons utilisateur-titre : il reste 49 136 cellules,
dont 36 711 à 0 ; les autres absences, elles, ne correspondent à aucune note.

**Ce que le modèle compare est donc un profil de co-interaction, pas un profil d'appréciation.** Deux
livres sont proches quand les mêmes personnes les ont touchés, pas quand elles les ont aimés
pareillement. On l'accepte parce que c'est le comportement du corrigé officiel, pas parce que c'est
la modélisation la plus fidèle du problème.

Un détail de pivot : le jeu référence un même titre sous plusieurs ISBN (rééditions, formats poche
et relié) : **50 titres** sont dans ce cas dans la matrice filtrée. `build_matrix` rattache les notes
par ISBN puis les réunit sur le titre, et écarte les doublons utilisateur-titre avant de pivoter
(**381 lignes**), sans quoi `pd.pivot` refuserait de construire une cellule à partir de deux valeurs.

## 5. La distance cosinus, et l'invariance qu'elle offre vraiment

Deux livres sont similaires si leurs profils pointent dans la même direction, quelle que soit leur
norme : c'est l'angle, pas la longueur, qui compte
([recommender.py:141](../recommender.py#L141)). La distance vaut `1 - cos(θ)`, soit 0 pour deux
profils colinéaires et 1 pour deux profils orthogonaux.

C'est ici qu'une formulation répandue mérite d'être corrigée, car l'erreur se transporte facilement
d'un projet à l'autre. On lit souvent que le cosinus « annule la générosité d'échelle d'un
utilisateur qui note tout haut ». C'est inexact, pour deux raisons distinctes.

**D'abord, la nature de l'invariance.** Le cosinus est invariant par **homothétie** : multiplier une
ligne entière par un facteur ne change pas la distance d'un chiffre. Il n'est pas invariant par
**translation** : ajouter une constante à toutes les notes d'un profil la fait bouger. Vérifié sur
la matrice du projet, sur la ligne « 1984 » (la transformation y porte) mesurée contre la ligne
témoin « 1st to Die: A Novel » :

| Transformation appliquée à une ligne | Distance à une ligne témoin |
|---|---|
| Aucune | 0,909955 |
| Multipliée par 2 (homothétie) | 0,909955 |
| +2 sur chaque note existante (translation) | 0,909956 |

Or un notateur généreux relève du second cas, pas du premier.

**Ensuite, la dimension concernée.** Les vecteurs comparés sont des **lignes**, donc des livres. La
générosité d'un utilisateur affecte une **colonne**. L'invariance par homothétie sur les lignes ne
peut donc rien contre un biais qui vit sur les colonnes, même en principe.

Ce que le cosinus neutralise réellement, c'est l'écart de **popularité entre deux livres** : un
titre noté 400 fois et un autre noté 120 fois ne sont pas éloignés par cette seule différence de
norme. C'est déjà utile, et c'est l'invariance qui compte dans un filtrage item-item, mais ce n'est
pas celle que la formulation courante annonce.

Corriger réellement les biais de notation demande une étape explicite, absente ici : centrer chaque
colonne sur la moyenne de son utilisateur, ce qui donne la *centered cosine*, équivalente à la
corrélation de Pearson (section 10).

Enfin, `NearestNeighbors` n'accepte le cosinus qu'avec `algorithm="brute"`
([recommender.py:154](../recommender.py#L154)). Ce n'est pas un défaut d'implémentation : les arbres
KD et Ball exigent une vraie distance métrique, et le cosinus ne respecte pas l'inégalité
triangulaire. Contrainte sans coût ici, avec 673 lignes de 888 colonnes : le balayage exhaustif est
immédiat, et la matrice dense ne pèse que 4,8 Mo.

## 6. Les k plus proches voisins : un algorithme sans apprentissage

k-NN est **paresseux** (*lazy learning*) : il n'estime aucun paramètre, ne minimise aucune fonction
de perte, ne fait aucune passe d'optimisation. `fit` se contente de mémoriser la matrice
([recommender.py:182](../recommender.py#L182)). Tout le calcul est reporté à la requête.

C'est un renversement complet par rapport au projet 02, et il vaut d'être explicité :

| | Projet 02 (CNN) | Projet 03 (k-NN) |
|---|---|---|
| Coût à l'entraînement | Élevé (30 epochs) | Nul |
| Coût à la prédiction | Faible (une passe avant) | Élevé (balayage complet) |
| Paramètres appris | 1 667 169 | 0 |
| Risque de surapprentissage | Central | Sans objet |

Le mot « entraîner » est donc trompeur ici, et la `RuntimeError` de `_require_fitted` ne protège pas
d'un modèle mal appris mais d'un modèle **sans données**.

Conséquence directe : le risque de ce projet n'est pas le surapprentissage, il est dans la
**préparation des données**. Il n'y a rien à surajuster ; il y a tout à mal filtrer, mal pivoter,
mal interpréter. C'est pourquoi les sections 3 et 4 pèsent plus lourd que celle-ci.

Le k du projet est imposé par le contrat : 5 recommandations, plus le livre lui-même. `kneighbors`
demande donc n+1 voisins et écarte le premier, qui est le livre passé en argument, à distance nulle
([recommender.py:187](../recommender.py#L187)).

Ce « +1 » repose sur une hypothèse rarement dite : que le plus proche voisin d'un livre est
toujours lui-même. C'est vrai tant que les lignes sont distinctes. Deux livres aux profils
rigoureusement identiques seraient tous deux à distance 0, et rien ne garantirait alors lequel
`kneighbors` renvoie en premier. Le cas ne se produit pas sur ce jeu après dédoublonnage des titres,
mais l'hypothèse mérite d'être connue plutôt que subie.

## 7. Le faux piège : l'ordre de la liste de sortie

`kneighbors` renvoie les voisins par **distance croissante** : le plus proche en premier. La cellule
de test freeCodeCamp attend l'inverse. Le corrigé commence par « Catch 22 », distance 0,794, la
recommandation *la plus lointaine*, et finit par « The Vampire Lestat », 0,518, la plus proche.

Ne pas inverser fait échouer la cellule dès sa première assertion, avec un modèle par ailleurs
parfaitement correct. `recommend` inverse donc la liste avant de la renvoyer, et `test_units.py`
verrouille ce choix.

C'est un piège de **conformité**, pas de compréhension : il ne dit rien sur k-NN, seulement sur la
lecture attentive du corrigé attendu. Il vaut d'être signalé pour ce qu'il est, car passer une heure
à douter de sa distance alors que seul l'ordre est en cause est une expérience courante sur ce
projet.

## 8. Le contrat freeCodeCamp et la séparation des préoccupations

La fonction soumise, `get_recommends(titre)`, doit renvoyer `[titre, [[titre, distance] × 5]]`.
Cette signature à un seul argument pousse naturellement vers un modèle en variable globale, seule
façon évidente pour la fonction d'accéder aux données.

`make_get_recommends` ([recommender.py:223](../recommender.py#L223)) évite cela en fabriquant la
fonction par **fermeture** sur un modèle déjà entraîné : la contrainte externe est satisfaite sans
qu'aucun état global n'apparaisse, et les tests peuvent construire autant de fonctions
indépendantes qu'ils veulent. C'est cohérent avec la structure de modules plats décidée en projet 01
du projet 01, où le notebook n'héberge aucune logique.

Cette séparation a un coût d'entrée assumé : le modèle doit être entraîné avant que la fonction
existe. Deux erreurs sont donc rendues explicites plutôt que silencieuses :

- un modèle non entraîné lève une `RuntimeError` claire. La garde est centralisée dans
  `_require_fitted` ([recommender.py:177](../recommender.py#L177)) : sans elle, accéder à `titles`
  avant `fit` donnait un `AttributeError` sur `NoneType` là où `recommend` donnait un message
  lisible, soit deux visages pour un seul et même état invalide ;
- un titre absent de la matrice filtrée lève une `KeyError` explicative, qui rappelle combien de
  livres ont survécu au filtrage. C'est le cas de démarrage à froid de la section 1, et il est plus
  fréquent qu'on ne le croit : 99,8 % des livres du jeu déclenchent cette erreur.

## 9. Ce que la mesure a effectivement montré

Les sections précédentes annoncent des risques. Trois ont été mesurés sur le jeu réel, et le
résultat nuance ce que l'intuition suggérait.

**Le corrigé est reproduit à 4,4 × 10⁻⁸ près**, pour une tolérance de 10⁻⁴ demandée par le test,
soit une marge de plus de trois ordres de grandeur. L'écart résiduel vient de la version de
scikit-learn et de l'ordre des sommations flottantes, pas d'une différence de méthode.

**Le biais de popularité existe, mais il est modeste.** La section 1 laissait craindre que les
livres très notés soient proches de tout. La corrélation mesurée entre le nombre de notes d'un livre
et sa distance moyenne à tous les autres vaut **-0,227** : négative comme attendu (plus de notes,
distances plus faibles), mais faible. Le filtrage de la section 3 y est probablement pour quelque
chose, puisqu'il a déjà écarté les profils les plus déséquilibrés. On ne peut pas conclure qu'il
était nécessaire pour cela : la mesure ne départage pas cette hypothèse d'une autre où le biais
serait intrinsèquement faible sur ce jeu.

**« Catch 22 » est le résultat le plus instructif du projet.** Recommandé pour un roman de vampires
d'Anne Rice, il n'a avec lui aucune parenté de genre, d'auteur ou d'époque. C'est exactement ce que
la section 4 laissait attendre : le modèle mesure de la co-interaction, et « Catch 22 » est un
classique que beaucoup de gens déclarent avoir lu. Il apparaît d'ailleurs comme la recommandation
**la plus lointaine** des cinq (0,794 contre 0,518 pour la plus proche), les quatre autres étant
toutes des Anne Rice. Le voisinage est donc cohérent sur ses quatre premiers éléments et devient
générique au cinquième : on voit le signal s'épuiser.

## 10. Notions voisines, non implémentées ici

Ces mécanismes ne sont **pas** dans le code. Ils sont décrits parce qu'ils prolongent directement les
notions ci-dessus et constituent la suite naturelle du projet.

**Le centrage par utilisateur** (*centered cosine*, ou corrélation de Pearson) répond au biais
identifié en section 5. Il consiste à soustraire de chaque colonne la moyenne des notes de son
utilisateur avant de calculer les distances, de sorte qu'une note vaut « au-dessus ou en dessous de
ce que cette personne donne habituellement » plutôt qu'une valeur absolue. C'est la correction que
le cosinus simple n'apporte pas. Elle est inapplicable telle quelle ici : avec 74,6 % de notes à 0
confondues avec l'absence, la moyenne par utilisateur n'aurait pas de sens.

**La matrice creuse** (`scipy.sparse.csr_matrix`) est le format standard de ce genre de problème :
elle ne stocke que les valeurs non nulles, soit 12 425 au lieu de 597 624 cases, et `NearestNeighbors`
l'accepte directement. Elle est superflue à cette échelle (4,8 Mo en dense) mais deviendrait
indispensable sans le filtrage de la section 3, où la matrice complète compterait 271 379 × 105 283
cases, soit près de 230 Go en dense.

**La factorisation matricielle** (SVD, ALS) est l'approche qui a supplanté k-NN sur les gros
catalogues, popularisée par le prix Netflix en 2009. Au lieu de comparer directement les profils, on
décompose la matrice en un produit de deux matrices de rang réduit, révélant des facteurs latents
(des « axes de goût » non nommés). L'avantage décisif est la généralisation : deux livres peuvent
être reconnus proches sans avoir un seul lecteur en commun, ce dont k-NN est structurellement
incapable.

**L'évaluation d'un recommandeur** est le grand absent de ce projet, et l'écart avec les projets 01
et 02 est frappant. Ici, le seul critère est la conformité à cinq distances de référence : aucune
mesure ne dit si les recommandations sont *bonnes*. Un système de recommandation s'évalue par
d'autres moyens : precision@k et recall@k sur un jeu de test, NDCG pour tenir compte du rang, mais
aussi des métriques que la seule justesse ignore : la **couverture** (quelle part du catalogue est
jamais recommandée), la **diversité** et la **nouveauté** (un système qui ne recommande que des
best-sellers a d'excellents scores de justesse et aucun intérêt). Rien de tout cela n'est
mesurable sans jeu de test ni retour utilisateur, ce qui est la limite honnête de l'exercice.

## Sources

- Sarwar et al., *Item-Based Collaborative Filtering Recommendation Algorithms*, WWW 2001 :
  https://dl.acm.org/doi/10.1145/371920.372071
- Linden, Smith & York, *Amazon.com Recommendations: Item-to-Item Collaborative Filtering*, IEEE
  Internet Computing 2003 : https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
- Ziegler et al., *Improving Recommendation Lists Through Topic Diversification*, WWW 2005 :
  https://dl.acm.org/doi/10.1145/1060745.1060754 (article à l'origine du jeu Book-Crossings)
- Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008 :
  https://ieeexplore.ieee.org/document/4781121 (distinction feedback explicite / implicite)
- Koren, Bell & Volinsky, *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer
  2009 : https://ieeexplore.ieee.org/document/5197422
- Documentation scikit-learn, `NearestNeighbors` et contraintes sur les métriques :
  https://scikit-learn.org/stable/modules/neighbors.html
- [notions-mobilisees.md](../../02-cat-dog-classifier/docs/notions-mobilisees.md) du projet 02, sur
  le contraste entre apprentissage paramétrique et paresseux.
