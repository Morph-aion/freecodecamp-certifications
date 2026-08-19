# Notions mobilisées par le projet

Ce que cet exercice fait travailler, notion par notion, avec le lien vers le code qui
l'implémente. L'énoncé freeCodeCamp donne les instructions sans jamais les expliquer : ce document
comble cet écart.

Distinct de [enonce-freecodecamp.md](enonce-freecodecamp.md), qui est le cahier des charges
imposé et ne doit pas être commenté.

Les références `classifier.py:NN` pointent sur les lignes exactes au moment de la rédaction. Elles
se décalent dès que le fichier est modifié : en cas de doute, chercher le nom de la fonction ou de
la constante plutôt que de se fier au numéro.

## Table des matières

- [1. Pourquoi un CNN : la hiérarchie des motifs](#1-pourquoi-un-cnn--la-hiérarchie-des-motifs)
- [2. Les non-linéarités : ReLU et sigmoïde](#2-les-non-linéarités--relu-et-sigmoïde)
- [3. Pourquoi `binary_crossentropy`](#3-pourquoi-binary_crossentropy)
  - [L'optimiseur : ce qui descend le gradient](#loptimiseur--ce-qui-descend-le-gradient)
- [4. Les deux garde-fous contre le surapprentissage](#4-les-deux-garde-fous-contre-le-surapprentissage)
- [5. Le nombre d'epochs, et le paradoxe qu'il cache](#5-le-nombre-depochs-et-le-paradoxe-quil-cache)
- [6. La reproductibilité](#6-la-reproductibilité)
- [7. Notions voisines, non implémentées ici](#7-notions-voisines-non-implémentées-ici)

## 1. Pourquoi un CNN : la hiérarchie des motifs

Quatre blocs `Conv2D` + `MaxPooling2D`, avec un nombre de filtres croissant (16, 32, 64, 64) :
[classifier.py:312-319](../classifier.py#L312-L319).

**Ce que fait la convolution** : au lieu de connecter chaque pixel à chaque neurone, elle fait
glisser un petit filtre (ici 3×3) sur toute l'image. Le même filtre s'applique partout, ce qui
apporte deux choses. D'abord une économie massive de paramètres : un filtre 3×3 en compte 9, quelle
que soit la taille de l'image. Ensuite l'**invariance par translation** : un motif reconnu en haut
à gauche est reconnu de la même façon en bas à droite.

**Ce que fait l'empilement** : les premières couches captent des motifs locaux (bords, coins,
textures), les suivantes des combinaisons de ces motifs, et ainsi de suite. C'est la hiérarchie qui
donne son nom à l'apprentissage profond.

Nuance à ne pas gommer : dire que « la couche 3 détecte les oreilles » est une simplification
pédagogique commode mais qui n'a rien d'automatique. Les représentations apprises ne se
décomposent pas nécessairement en concepts nommables, et une bonne partie de la recherche en
interprétabilité consiste précisément à essayer de savoir ce que ces couches encodent vraiment.

**Ce que fait le pooling** : `MaxPooling2D(2, 2)` ([classifier.py:313](../classifier.py#L313))
ne garde que le maximum de chaque carré 2×2, ce qui divise la résolution par deux. On perd la
position exacte du motif, on gagne en robustesse aux petits décalages, et surtout on élargit le
**champ réceptif** : après quatre blocs, un neurone « voit » une portion bien plus large de
l'image d'origine qu'un filtre 3×3.

D'où la forme en entonnoir : la résolution décroît (150 → 74 → 36 → 17 → 7) pendant que le
nombre de filtres croît (16 → 64). On échange de la précision spatiale contre de la richesse
sémantique.

## 2. Les non-linéarités : ReLU et sigmoïde

**ReLU** (`max(0, x)`), sur chaque `Conv2D` et sur la couche dense :
[classifier.py:312-322](../classifier.py#L312-L322).

Sans fonction non linéaire entre les couches, empiler dix couches linéaires reviendrait
exactement à une seule : la composition de transformations linéaires est linéaire. La
non-linéarité est ce qui rend la profondeur utile.

ReLU plutôt que sigmoïde dans les couches cachées, parce qu'elle **ne sature pas** du côté
positif : sa dérivée vaut 1, donc le gradient traverse les couches sans s'atténuer. Une sigmoïde,
elle, écrase les grandes valeurs vers 0 ou 1, où sa dérivée tend vers zéro : le gradient
disparaît en remontant les couches (problème du *vanishing gradient*), et les premières couches
n'apprennent plus.

Contrepartie honnête : un neurone ReLU dont la sortie est toujours négative a un gradient nul et
ne se réveille jamais (*dying ReLU*). Des variantes existent (Leaky ReLU, ELU) ; à cette échelle
le problème est rarement gênant.

**Sigmoïde en sortie** ([classifier.py:323](../classifier.py#L323)) : là, la saturation n'est
plus un défaut, c'est l'objectif. Elle comprime la sortie dans [0, 1], ce qui s'interprète comme la
probabilité que l'image soit un chien. C'est ce que `round()` transforme en décision binaire dans
`evaluate` ([classifier.py:597](../classifier.py#L597)).

Un seul neurone de sortie suffit pour deux classes : `p(chien)` détermine `p(chat) = 1 − p`. Avec
trois classes ou plus, on utiliserait autant de neurones et une activation `softmax`, qui
normalise l'ensemble pour que la somme fasse 1.

Détail qui compte : la classe 1 correspond bien à « chien » parce que `flow_from_directory` ordonne
les classes alphabétiquement, et que `cats` précède `dogs`. Inverser les noms de dossiers
inverserait silencieusement l'interprétation.

## 3. Pourquoi `binary_crossentropy`

Déclarée à la compilation : [classifier.py:328](../classifier.py#L328).

La perte mesure l'erreur sur les **probabilités**, pas sur les classes. Sa formule pour une
observation :

```
perte = -( y · log(p) + (1-y) · log(1-p) )
```

où `y` est l'étiquette réelle (0 ou 1) et `p` la probabilité prédite.

La propriété qui la rend adaptée : le terme logarithmique **diverge** quand on se trompe avec
certitude. Prédire 0,99 pour un chat coûte `-log(0.01) ≈ 4,6`, alors que prédire 0,6 pour ce même
chat ne coûte que `-log(0.4) ≈ 0,92`. L'erreur confiante est punie bien plus lourdement que
l'erreur hésitante, ce qui pousse le modèle à ne pas afficher de certitude injustifiée.

C'est le même principe que le log-loss utilisé dans le projet 01, et pour la même raison : c'est
une **règle de score strictement propre**, optimisée en espérance uniquement si l'on rapporte sa
croyance honnête. La continuité entre les deux projets n'est pas fortuite.

Pourquoi pas la justesse (*accuracy*) comme fonction de perte : elle est constante par morceaux,
donc de gradient nul presque partout. Il n'y aurait rien à descendre. La justesse sert à
**mesurer**, la crossentropy à **optimiser** ; c'est pourquoi `compile` reçoit les deux, dans des
rôles différents ([classifier.py:327-329](../classifier.py#L327-L329)).

### L'optimiseur : ce qui descend le gradient

La perte dit *où* aller, l'optimiseur dit *comment* y aller. `compile` reçoit `optimizer="adam"`
([classifier.py:327](../classifier.py#L327)), choix par défaut rarement discuté et qui mérite de
l'être.

La descente de gradient simple (SGD) applique un pas de taille fixe dans la direction opposée au
gradient. Deux difficultés : un pas trop grand fait osciller autour du minimum, un pas trop petit
rend la convergence interminable, et le bon réglage varie selon les paramètres.

**Adam** (Kingma & Ba, 2015) adapte le pas paramètre par paramètre, en combinant deux mémoires :
la moyenne des gradients récents (*momentum*, qui lisse les à-coups) et celle de leurs carrés
(qui réduit le pas là où le gradient est instable). En pratique, il converge vite sans réglage
manuel, ce qui explique son statut de défaut dans la plupart des projets.

Contrepartie honnête : ce défaut est commode, pas optimal. La littérature rapporte que SGD avec
momentum et un calendrier de taux d'apprentissage bien réglé généralise parfois mieux qu'Adam sur
des tâches de vision. Ce n'est pas la question ici, où le modèle n'a pas surappris (voir la
mesure en fin de section 4), mais c'est le genre de choix qu'on hérite sans l'examiner.

C'est aussi ce paramètre que `ReduceLROnPlateau` viendrait moduler (section 7).

## 4. Les deux garde-fous contre le surapprentissage

2000 images d'entraînement, c'est peu pour un réseau de cette taille. Le risque n'est pas qu'il
échoue à apprendre, c'est qu'il apprenne **trop bien ces images-là**, en mémorisant des détails
propres à chacune plutôt que ce qui distingue un chat d'un chien.

**Augmentation de données**, six transformations aléatoires appliquées au vol :
[classifier.py:251-257](../classifier.py#L251-L257).

Chaque epoch, la même photo arrive légèrement tournée, décalée, zoomée ou retournée. Le réseau ne
revoit donc jamais exactement le même tenseur d'entrée, ce qui rend la mémorisation pixel à pixel
beaucoup plus difficile.

Formulation à nuancer, cependant : dire que la mémorisation devient « impossible » serait trop
fort. Les variantes restent proches de l'originale, et un réseau assez grand entraîné assez
longtemps finira par surapprendre malgré l'augmentation. Elle repousse le problème, elle ne le
supprime pas.

Le retournement **vertical** est volontairement exclu : aucune photo du jeu ne montre un animal la
tête en bas, et apprendre cette invariance gaspillerait de la capacité pour rien.

**Dropout(0.5)**, placé après `Flatten` et avant la couche dense :
[classifier.py:321](../classifier.py#L321).

À chaque passage d'entraînement, la moitié des activations est mise à zéro au hasard. Le réseau ne
peut donc pas faire reposer sa décision sur un neurone précis, toujours disponible : il doit
construire des représentations **redondantes**. On l'interprète souvent comme l'entraînement
implicite d'un ensemble de sous-réseaux qui partagent leurs poids.

Le placement n'est pas indifférent : il vise la couche dense, qui concentre l'écrasante majorité
des paramètres. Le décompte exact vaut d'être fait, car il est contre-intuitif.

Après les quatre blocs, la résolution est tombée de 150×150 à 7×7 (chaque bloc retire 2 pixels
par la convolution 3×3 sans remplissage, puis divise par 2 au pooling). L'aplatissement produit
donc 7 × 7 × 64 = **3136 entrées**, et la connexion vers `Dense(512)` coûte
3136 × 512 + 512 = **1 606 144 paramètres**.

| Couche | Paramètres | Part |
|---|---|---|
| `Conv2D(16)` | 448 | 0,0 % |
| `Conv2D(32)` | 4 640 | 0,3 % |
| `Conv2D(64)` | 18 496 | 1,1 % |
| `Conv2D(64)` | 36 928 | 2,2 % |
| **`Dense(512)`** | **1 606 144** | **96,3 %** |
| `Dense(1)` | 513 | 0,0 % |
| Total | 1 667 169 | |

Les quatre couches convolutives réunies ne pèsent que **3,6 %** du modèle : le partage de poids
qui fait la force de la convolution la rend aussi très économe. C'est donc dans l'unique couche
dense que la mémorisation se loge, et c'est exactement là que le dropout est placé.

Le dropout n'agit qu'à l'entraînement, et le sens de la compensation mérite d'être précis, car
l'intuition l'inverse volontiers. Keras applique un *inverted dropout* : **à l'entraînement**, les
unités conservées sont multipliées par `1 / (1 - rate)`, de sorte que la somme des entrées reste
inchangée malgré les unités éteintes. **En inférence, la couche ne fait rien du tout** : aucune
unité n'est coupée, aucune mise à l'échelle n'est appliquée.

L'intérêt de cette convention est pratique : le modèle déployé n'a aucun traitement particulier à
prévoir, la couche devient transparente. La documentation Keras est explicite sur les deux points
(« Inputs not set to 0 are scaled up by `1 / (1 - rate)` », « no values are dropped during
inference »).

**Comment vérifier que ça marche** : les courbes produites par `save_history_plots`
([classifier.py:514](../classifier.py#L514)). La signature
du surapprentissage est un **écart croissant** entre la justesse d'entraînement, qui continue de
grimper, et celle de validation, qui stagne ou redescend. C'est le diagnostic central de ce
projet, l'équivalent de ce qu'était le graphe comparatif des stratégies dans le projet 01.

### Ce que la mesure a effectivement montré

Tout ce qui précède décrit un risque **anticipé**. L'entraînement réel raconte autre chose, et
c'est instructif : sur 30 epochs, les deux courbes se suivent du début à la fin, avec un écart
final de 2,1 points seulement (0,802 contre 0,781), et la perte de validation descend encore à la
dernière epoch.

**Le surapprentissage n'a pas eu lieu.** Deux garde-fous ont été posés contre un phénomène qui ne
s'est pas produit dans les conditions retenues. Ce n'est pas l'accident d'un run isolé :
`repetitions.py` relance le pipeline sur trois graines et trouve un écart de **0,022 ± 0,001**,
soit une reproductibilité quasi parfaite. On ne peut pas en conclure qu'ils étaient inutiles
(peut-être l'ont-ils précisément empêché), ni qu'ils étaient nécessaires : la mesure ne départage
pas ces deux hypothèses. Il faudrait entraîner sans eux pour le savoir.

Ce qu'elle montre en revanche sans ambiguïté, c'est que le modèle **apprenait encore** quand on
l'a arrêté. La question pratique n'est donc pas « comment arrêter avant qu'il surapprenne ? »
mais « jusqu'où monterait-il avec plus d'epochs ? ». C'est l'inverse du paradoxe décrit en
section 5, et un rappel utile : un risque documenté n'est pas un risque observé.

## 5. Le nombre d'epochs, et le paradoxe qu'il cache

Une *epoch* est un passage complet sur le jeu d'entraînement. `steps_per_epoch` en fixe le
découpage en lots : `ceil(2000 / 128) = 16` lots ici, calculé plutôt que codé en dur
([classifier.py:352-353](../classifier.py#L352-L353)), car une
valeur trop basse n'entraînerait que sur une fraction du jeu, silencieusement.

`EPOCHS = 30` ([classifier.py:98](../classifier.py#L98)) prend de la marge parce que
l'évaluation ne porte que sur 50 images : une seule image mal classée vaut 2 points de
pourcentage, et un score juste au-dessus de 63 % ne serait pas
un résultat fiable.

**Le paradoxe** : plus d'epochs n'est pas toujours mieux. Passé un certain point, la perte de
validation remonte alors que celle d'entraînement continue de descendre. Le modèle progresse sur ce
qu'il a vu et régresse sur le reste. Il existe donc un optimum, et il n'est pas connu à l'avance :
c'est ce que les courbes permettent de repérer après coup.

## 6. La reproductibilité

`RANDOM_SEED = 42` ([classifier.py:93](../classifier.py#L93)), posée en tête de `main()` avant
toute construction : [classifier.py:616-617](../classifier.py#L616-L617).

Trois sources d'aléa interviennent : l'initialisation des poids, les transformations
d'augmentation, et le mélange des lots. Sans graine fixée, deux exécutions du même code donnent
des résultats différents, et il devient impossible de savoir si un changement a amélioré le modèle
ou si l'écart n'est que du bruit. C'est particulièrement critique ici, où 3 images d'écart valent
6 points.

Limite à connaître : sur GPU, `np.random.seed` et `tf.random.set_seed` ne suffisent pas. Les
noyaux CUDA parallèles introduisent un non-déterminisme dans l'ordre des sommations flottantes.
`tf.config.experimental.enable_op_determinism()` le corrige, au prix de la vitesse.

## 7. Notions voisines, non implémentées ici

Ces mécanismes ne sont **pas** dans le code. Ils sont décrits parce qu'ils prolongent directement
les notions ci-dessus, et qu'ils constituent la suite naturelle si le projet se prolonge.

**EarlyStopping** surveille la perte de validation et interrompt l'entraînement quand elle cesse
de s'améliorer pendant un nombre d'epochs donné (*patience*), en restaurant au besoin les
meilleurs poids. L'idée est de ne pas fixer la durée d'entraînement à l'avance : c'est la
validation qui décide quand s'arrêter. C'est la réponse directe au paradoxe de la section 5.

**Mais la mesure le rend inutile ici** : la perte de validation descendait encore à l'epoch 30,
`EarlyStopping` n'aurait donc jamais déclenché. Il répond à un problème que ce modèle n'a pas, du
moins dans cette configuration. Il redeviendrait pertinent si l'on augmentait franchement le
nombre d'epochs, ou si l'on retirait un des garde-fous, deux situations où le surapprentissage
finirait par apparaître.

C'est un bon rappel sur l'outillage : ajouter un callback parce qu'il figure dans les bonnes
pratiques, sans avoir observé le problème qu'il traite, revient à traiter un symptôme absent.

**ReduceLROnPlateau** divise le taux d'apprentissage quand la validation plafonne. Le taux
d'apprentissage est la taille du pas de descente de gradient : un pas trop grand fait osciller
autour du minimum sans jamais s'y poser, un pas trop petit rend la convergence interminable. Le
réduire en fin de parcours permet d'affiner une solution déjà approchée.

**Transfer learning** : partir d'un réseau pré-entraîné sur ImageNet (MobileNetV2, VGG16), geler
ses couches convolutives et ne réentraîner que la tête de classification. Sur ce jeu de données,
cette approche atteint couramment 95 % là où l'entraînement de zéro plafonne vers 80 à 85 %.

Elle est écartée ici pour une raison assumée : l'énoncé impose `Sequential` et `Conv2D`, et
surtout tout l'intérêt pédagogique de l'exercice est le compromis sur 2000 images. Le transfer
learning donnerait un meilleur score en supprimant la difficulté qu'on cherche à comprendre.

## Sources

- Documentation Keras, couches de convolution et de pooling : https://keras.io/api/layers/
- Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR
  2014 : https://jmlr.org/papers/v15/srivastava14a.html
- Documentation Keras, couche `Dropout` (sens exact de la mise à l'échelle) :
  https://keras.io/api/layers/regularization_layers/dropout/
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015 :
  https://arxiv.org/abs/1412.6980
- Documentation Keras, callbacks (`EarlyStopping`, `ReduceLROnPlateau`) :
  https://keras.io/api/callbacks/
- TensorFlow, reproductibilité et déterminisme des opérations :
  https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
- Le cadrage théorique du projet 01, sur les règles de score strictement propres, dont la
  crossentropy relève.
