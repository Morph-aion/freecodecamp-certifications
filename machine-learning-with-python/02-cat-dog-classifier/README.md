# 02 : Cat and Dog Image Classifier

Deuxième des 5 projets de la certification *Machine Learning with Python*. Classifier des images
de chats et de chiens avec un réseau de neurones convolutif, à **63 % de justesse minimum**
(70 % en bonus).

## Ce qui change par rapport au projet 01

Premier projet à réseau de neurones, et surtout **premier changement de régime**. Pierre-Papier-
Ciseaux était un problème séquentiel adversarial : un adversaire réagissait à mes coups, l'ordre
portait l'information, et le mécanisme générateur évoluait en cours de partie.

Ici, rien de tel. Les images sont **indépendantes et identiquement distribuées** : aucun ordre,
aucun adversaire, une distribution fixe. Le vocabulaire du projet 01 (opponent model, niveaux
méta, non-stationnarité) ne s'applique pas, et parler de *DGP* redevient légitime, ce que le
cadrage du 01 avait justement écarté.

Conséquence pratique : le risque principal n'est plus l'adaptation d'un adversaire, c'est le
**surapprentissage** sur seulement 2000 images d'entraînement. Les courbes justesse/perte
remplacent le graphe comparatif de stratégies comme outil de diagnostic central.

(Ce risque était anticipé ; la mesure ne l'a pas confirmé, voir la section « État ». Le diagnostic
par les courbes reste central, il a simplement dit autre chose que prévu.)

## Lancement

```bash
python -m unittest test_units   # tests unitaires, aucun besoin de TensorFlow
uv run classifier.py            # pipeline complet
```

Le script télécharge le jeu de données au premier lancement (73 Mo), entraîne le modèle, et
compare ses prédictions au corrigé officiel. Compter plusieurs minutes selon la machine.

Les données ne sont pas versionnées : `classifier.py` les retélécharge automatiquement.

## Fichiers

| Fichier | Rôle |
|---|---|
| `classifier.py` | Pipeline complet : téléchargement, générateurs, modèle, entraînement, évaluation |
| `test_units.py` | Tests unitaires : ordre du corrigé, cohérence, références de la doc. Aucun besoin de TensorFlow |
| `repetitions.py` | Relance le pipeline sur plusieurs graines, mesure la variance |
| `docs/enonce-freecodecamp.md` | Traduction de l'énoncé officiel, cahier des charges imposé |
| `docs/notions-mobilisees.md` | Les notions que l'exercice fait travailler, expliquées |
| `data/raw/` | Jeu de données brut, non versionné : créé et rempli au premier lancement |
| `models/` | Modèle entraîné (`.keras`), gitignoré |
| `figures/` | Les six figures du pipeline, versionnées |

**Pourquoi `data/raw/` sans `intermediate/` ni `processed/`** : la convention complète (Kedro,
reprise dans `templates/cookiecutter-data-science-python`) prévoit trois étages. Ce projet n'en
utilise qu'un, parce qu'il n'écrit aucune donnée intermédiaire : `ImageDataGenerator` applique
redimensionnement, normalisation et augmentation **à la volée**, en mémoire, entre le disque et le
réseau. Créer les deux autres dossiers les laisserait vides.

Le nom `raw/` est conservé malgré tout, parce qu'il qualifie la nature des données (brutes,
reproductibles depuis l'URL) et non une étape de pipeline.

## Le faux piège du projet

`flow_from_directory` liste les fichiers par ordre **lexicographique**, où `10.jpg` précède
`2.jpg`. L'intuition pousse alors à réordonner les prédictions par numéro de fichier avant de les
comparer au corrigé.

**Il ne faut pas.** Le corrigé freeCodeCamp est lui aussi indexé sur l'ordre du générateur, et la
comparaison se fait par `zip` direct. Réordonner ferait chuter la concordance à **28 images sur
50, soit 56 %**, sous le seuil, y compris pour un modèle parfait.

Vérification faite en regardant les images plutôt qu'en raisonnant sur les noms :

| Position dans le générateur | Fichier | Contenu réel | `ANSWERS` |
|---|---|---|---|
| 0 | `1.jpg` | chien | 1 |
| 1 | `10.jpg` | **chat** | **0** |
| (absent) | `2.jpg` | chien | (n/a) |

Si le corrigé suivait l'ordre numérique, `ANSWERS[1]` décrirait `2.jpg`, un chien, et vaudrait 1.
Il vaut 0, donc il décrit `10.jpg`, un chat.

Ce point est verrouillé par `test_units.py` (classe `TestOrdreDuCorrige`), qui teste les deux
hypothèses et chiffre le coût de l'erreur, pour qu'une « correction » intuitive ne puisse pas
réintroduire le décalage. `shuffle=False` reste indispensable : sans lui l'ordre varierait à
chaque exécution.

## Les six figures

Écrites dans `figures/`, numérotées dans l'ordre du pipeline. Les trois premières correspondent
aux cellules 4, 6 et 9 du notebook officiel ; la sixième vient de `repetitions.py`.

| Figure | Ce qu'elle montre | Ce qu'on y vérifie |
|---|---|---|
| `01-echantillon-entrainement.png` | Cinq images telles que le réseau les reçoit | Que les générateurs fonctionnent, avant d'attendre 30 epochs |
| `02-augmentation.png` | La même image transformée cinq fois | Que l'augmentation déforme sans rendre méconnaissable |
| `03-courbes.png` | Justesse et perte, entraînement contre validation | L'écart entre les deux courbes, signature du surapprentissage |
| `04-predictions-test.png` | Les 50 images de test annotées, erreurs en rouge | **Sur quoi** le modèle se trompe, pas seulement combien |
| `05-cartes-activation.png` | Ce que chaque couche convolutive met en évidence | La hiérarchie des motifs, du local au global |
| `06-variance-seeds.png` | Justesse de validation sur plusieurs graines | Que la trajectoire est reproductible, pas propre à un run |

Résultat mesuré : les courbes ne montrent **aucun surapprentissage**, contrairement à ce que le
projet anticipait. Voir la section « État » plus bas.

Les deux premières sont produites **avant** l'entraînement : si les images arrivent mal, autant le
savoir tout de suite.

La quatrième est la plus instructive. Un score de 80 % ne dit pas quelles images posent problème ;
la grille annotée, si. La cinquième rend visible la notion n° 1 de
[notions-mobilisees.md](docs/notions-mobilisees.md), à lire avec la réserve qui l'accompagne : une
carte d'activation montre *où* un filtre réagit, pas *ce qu'il représente*.

## Choix retenus

| Choix | Raison |
|---|---|
| CNN entraîné de zéro | Imposé par l'énoncé (`Sequential`, `Conv2D`). Le transfer learning atteindrait 95 % sans rien apprendre du compromis de l'exercice |
| `ImageDataGenerator` | Déprécié en Keras 3 au profit de `image_dataset_from_directory`, mais explicitement exigé par l'énoncé |
| 6 transformations d'augmentation | Fourchette 4 à 6 demandée. Retournement vertical exclu : aucun animal à l'envers dans le jeu de test |
| `Dropout(0.5)` | Non demandé, mais 2000 images suffisent à ce que le réseau les apprenne par cœur |
| 30 epochs | Marge prise pour la variance de l'échantillon. **Mesuré depuis : insuffisant**, le modèle apprenait encore à l'arrêt |
| `Input(shape=...)` explicite | `input_shape=` dans la première `Conv2D` émet un avertissement en Keras 3 |

## État : défi réussi, 80,7 % ± 2,3 %

Run de référence (graine 42) : **41 images correctement classées sur 50**, soit 82 %. Mesuré sur
trois graines, le score s'établit à **80,7 % ± 2,3 %**, toujours au-dessus du seuil de 63 % et du
bonus de 70 %. Détail plus bas.

| Métrique (dernière epoch) | Entraînement | Validation |
|---|---|---|
| Justesse | 0,802 | 0,781 |
| Perte | 0,440 | 0,460 |

### Ce que la mesure a démenti

Tout le projet documentait le **surapprentissage** comme risque principal : 2000 images, un réseau
de 1,67 million de paramètres, deux garde-fous mis en place pour ça. Les courbes
([figures/03-courbes.png](figures/03-courbes.png)) montrent qu'il n'a pas eu lieu.

Les deux courbes se suivent du début à la fin, avec un écart final de **2,1 points** seulement
entre justesse d'entraînement et de validation. La perte de validation descend encore à la
dernière epoch. Ce n'est pas un modèle qui a mémorisé, c'est un modèle **encore en train
d'apprendre quand on l'a arrêté**.

Deux lectures, que la mesure actuelle ne permet pas de départager :

- L'augmentation et le dropout sont si efficaces qu'ils **contraignent trop** le modèle, qui reste
  sous-entraîné à 30 epochs.
- 30 epochs sont simplement insuffisantes pour cette architecture, et le score continuerait de
  monter.

Conséquence pratique : `EarlyStopping`, présenté en section 7 de
[docs/notions-mobilisees.md](docs/notions-mobilisees.md) comme la réponse au paradoxe des epochs,
n'aurait **rien à arrêter ici**. Le chantier utile est l'inverse : augmenter `EPOCHS` et observer
où les courbes se séparent enfin.

### Ce que la grille des prédictions montre

[figures/04-predictions-test.png](figures/04-predictions-test.png) répond à la question que le
score ne traite pas : *sur quoi* le modèle se trompe. Les erreurs ne sont pas réparties au hasard,
elles se concentrent sur les images où l'animal est **tenu en main, partiellement masqué ou dans
une posture inhabituelle**. Les réussites franches (98 %, 100 %) sont des animaux nets, cadrés,
sur fond neutre.

### Réserve sur la fiabilité du chiffre, désormais mesurée

50 images de test, c'est peu : 3 images d'écart valent 6 points de pourcentage. Cette réserve
figurait dans le README sans être chiffrée. `repetitions.py` la mesure, en relançant le pipeline
complet sur d'autres graines :

| seed | justesse train | justesse val | écart | test |
|---|---|---|---|---|
| 42 (run de référence) | 0,802 | 0,781 | 0,021 | 41/50 |
| 7 | 0,797 | 0,774 | 0,023 | 41/50 |
| 2025 | 0,790 | 0,769 | 0,021 | 39/50 |

**Écart train-validation : 0,022 ± 0,001.** C'est le résultat le plus solide des trois runs : la
conclusion « pas de surapprentissage » n'était pas l'accident d'un run isolé, elle se reproduit à
l'identique. La figure [06-variance-seeds.png](figures/06-variance-seeds.png) montre des
trajectoires de validation qui se superposent presque parfaitement.

**Score de test : 80,7 % ± 2,3 %**, étendue de 2 images sur 50, soit 4 points. Le 82 % annoncé
plus haut est donc le haut de la fourchette, pas la valeur typique. Le score reste très au-dessus
du seuil dans les trois cas (le pire fait 39/50, soit 78 %), mais **annoncer « 82 % » sans
intervalle surestimait la précision de la mesure**.

Réserve sur cette mesure elle-même : trois runs, c'est peu pour estimer une dispersion, et
l'écart-type reporté est lui-même très incertain. Ce qu'ils établissent solidement, c'est l'ordre
de grandeur (quelques points, pas quinze) et la reproductibilité de l'écart train-validation.

Le corpus est par ailleurs équilibré (24 chiens, 26 chats), ce qui exclut qu'un classifieur
constant atteigne le seuil.

URL de soumission : *à compléter une fois le projet soumis.*
