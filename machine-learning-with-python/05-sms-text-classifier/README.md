# 05 : Neural Network SMS Text Classifier

Cinquième et dernier projet de la certification *Machine Learning with Python*.
Classifier des SMS en **ham** (normal) ou **spam** (publicité) à l'aide d'un
**réseau de neurones** Keras. Le jeu de données est déjà séparé en train/test
par freeCodeCamp.

## Ce qui change par rapport au projet 04

Le projet 04 (régression) traitait des **données tabulaires** : chaque
observation était un vecteur de features numériques. Ici, le problème est
**textuel** : chaque observation est une chaîne de caractères (un message SMS),
et le modèle doit apprendre à en extraire le sens.

C'est le second projet à réseau de neurones (après Cat/Dog), mais le régime est
différent : les images étaient des tenseurs prêts à l'emploi, ici le texte doit
être **vectorisé** avant d'entrer dans le réseau. La couche
`TextVectorization` de Keras fait ce travail, et c'est la principale nouveauté
technique de ce projet.

Le risque principal change lui aussi : il n'est plus dans la représentation des
features mais dans le **déséquilibre de classes**, qui rend la justesse globale
trompeuse (voir plus bas).

## Lancement

```bash
uv run classifier.py              # pipeline complet
uv run test_units.py              # tests unitaires (entraînent un modèle, ~10 s)
uv run --with marimo marimo edit --sandbox notebook.py # notebook Marimo
```

Les tests s'exécutent via `uv run test_units.py` et non `python -m unittest` :
ils entraînent un vrai modèle sur 3 epochs, donc TensorFlow et scikit-learn
doivent être résolus depuis l'en-tête PEP 723 du fichier.

Depuis la racine de la certification :

```bash
make test PROJET=05-sms-text-classifier
make lint
make fix
```

Le script télécharge les deux TSV depuis le CDN freeCodeCamp au premier
lancement (`data/raw/` est gitignoré), entraîne le modèle, affiche les
métriques de détection des spam, et vérifie que les 7 messages du test officiel
sont correctement classés.

## Fichiers

| Fichier | Rôle |
|---|---|
| `classifier.py` | Pipeline complet : chargement TSV, vectorisation, modèle Keras, entraînement, `predict_message` |
| `test_units.py` | Tests unitaires : format des données, conversion texte, contrat `predict_message` sur un modèle réellement entraîné |
| `notebook.py` | Notebook Marimo : exploration, visualisation, entraînement, test officiel |
| `docs/enonce-freecodecamp.md` | Traduction de l'énoncé officiel, cahier des charges imposé |
| `docs/notions-mobilisees.md` | Les notions que l'exercice fait travailler, expliquées |
| `data/raw/` | Jeu de données brut (TSV), non versionné : créé et rempli au premier lancement |

## Le faux piège du projet : le déséquilibre de classes

Le jeu de données contient **86,6 % de ham et 13,4 % de spam**. Un modèle
trivial qui prédit toujours « ham » obtient 86,6 % de justesse, bien au-dessus
du seuil implicite de nombreux benchmarks. La justesse seule ne dit rien sur la
capacité à détecter les spam.

Il faut regarder la **matrice de confusion** (*confusion matrix*) et le
**rappel sur la classe spam** (*recall*, capacité à trouver tous les spam)
plutôt que la justesse globale (*accuracy*).

Les étiquettes de métriques portent le terme français suivi du terme anglais :
c'est sous le nom anglais qu'on les retrouve dans scikit-learn et dans la
littérature. « Précision » mérite l'attention : en français courant il évoque
l'exactitude, alors que la *precision* en ML est le taux de justesse des
alertes, distinct de la justesse globale.

## Le vrai piège : la fonction `predict_message`

L'énoncé demande une fonction qui renvoie `[probabilité, "ham" ou "spam"]`. La
probabilité doit être entre 0 et 1, où **0 = ham** et **1 = spam**. C'est
l'inverse de l'ordre alphabétique, et c'est cohérent avec la sortie `sigmoid`
du réseau (0 = négatif = ham, 1 = positif = spam).

La fonction doit aussi **segmenter le texte en mots** avant de le passer au
modèle, car le jeu de données d'entraînement est un TSV brut (pas un tensor).
La couche `TextVectorization` de Keras gère ce cas si elle est configurée avec
`output_mode="int"`.

## Keras 3 refuse les Series pandas de texte

Le code qui fonctionnait avec Keras 2 échoue avec **TensorFlow 2.21 / Keras 3** :

```
ValueError: Invalid dtype: str
```

Keras 3 ne convertit plus implicitement une `pandas.Series` de chaînes passée à
`fit()`. Les tableaux numpy de dtype `<U…` sont refusés eux aussi ; seul le
dtype `object` passe. D'où `as_text_array()`, appliqué à chaque entrée texte.

Un `keras.Input(shape=(), dtype=tf.string)` explicite ouvre le `Sequential` :
sans lui, les couches restent « unbuilt » et `model.summary()` annonce
`Total params: 0` tant que `fit()` n'a pas tourné.

Ce point vaut d'être vérifié avant de porter le code dans Colab, dont la
version de Keras peut différer de celle installée ici.

## Une fuite dans le découpage fourni

Le découpage train/valid vient de freeCodeCamp, il n'a pas été refait ici. Il
contient **128 messages présents dans les deux fichiers**, plus 244 doublons
internes au train. Les métriques de validation sont donc légèrement optimistes.

Ce n'est pas corrigé (le sujet impose le découpage fourni), mais il faut le
savoir avant d'interpréter un écart de quelques dixièmes de point.

Les 7 messages du test officiel, eux, sont **absents des deux corpus** (vérifié
par un test) : le test final mesure bien de la généralisation.

## Choix retenus

| Choix | Raison |
|---|---|
| `TextVectorization` + `Embedding` + `GlobalAveragePooling1D` + `Dense` | Architecture simple, rapide, suffisante pour ce corpus |
| `max_tokens=1000`, `output_sequence_length=50` | 1000 tokens couvrent 79,8 % des tokens rencontrés ; `50` ne tronque que 1,2 % des SMS |
| `binary_crossentropy` + `sigmoid` | Classification binaire standard |
| Pas de `Dropout` | Le modèle est petit (quelques milliers de paramètres), le surapprentissage est limité |
| Labels : 0 = ham, 1 = spam | Convention sigmoid, cohérente avec `predict_message` |
| Pas de `mask_zero=True` | Produit une perte `nan` sur ce corpus, voir ci-dessous |

## Pourquoi `mask_zero=True` casse le modèle ici

L'optimisation paraît évidente : les SMS font 13 tokens en médiane pour une
séquence de 50, donc `GlobalAveragePooling1D` moyenne surtout du padding.
Activer `mask_zero=True` sur l'`Embedding` devrait exclure ces positions.

Mesuré sur trois graines, l'effet est l'inverse :

| Configuration | Justesse | Rappel spam | Test officiel |
|---|---|---|---|
| `mask_zero=False` (retenu) | 0,986 | 0,934 | 7/7 |
| `mask_zero=True` | 0,866 | **0,000** | 4/7 |

Un rappel de 0,000 pour une justesse de 0,866 est la signature du modèle
dégénéré : il prédit « ham » pour tout, et la justesse n'est que le taux de ham
du corpus.

La cause est une division par zéro. **Deux messages du corpus d'entraînement
(`:)` et `:-) :-)`) ne contiennent aucun mot du vocabulaire** : après
vectorisation, leur séquence est entièrement du padding. Avec le masquage,
`GlobalAveragePooling1D` moyenne sur zéro position valide, produit `nan`, et le
`nan` se propage à tous les poids dès le premier batch. La perte affichée passe
de 0,237 à `nan` en trois epochs.

Le masquage serait exploitable après filtrage des messages vides, mais le gain
attendu ne justifie pas la fragilité ajoutée : le modèle non masqué apprend à
ignorer le padding par lui-même, puisque l'embedding de l'indice 0 est un
paramètre libre que l'entraînement pousse vers un vecteur neutre.

## Approches publiques comparées

Ce projet a été résolu sans consulter de solution existante. Une recherche
menée après coup situe ce qui a été écrit ici par rapport à ce qui circule, et
attribue ce qui vient d'ailleurs.

**L'architecture vient du tutoriel officiel TensorFlow.** Le pipeline
`TextVectorization` → `Embedding` → `GlobalAveragePooling1D` → `Dense(1, sigmoid)`
est celui de [Basic text classification](https://www.tensorflow.org/tutorials/keras/text_classification),
qui utilise lui aussi `embedding_dim = 16`. Ce n'est pas une convergence
fortuite : c'est le pipeline canonique de Keras pour la classification binaire
de texte, et le notebook fourni par freeCodeCamp impose déjà ces imports.

**Deux familles dominent les solutions publiques**, et celle retenue ici est la
plus légère :

| Famille | Exemple public | Taille |
|---|---|---|
| Bidirectional LSTM | [letientai.io](https://letientai.io/freecodecamp/ai/nn/) : Embedding(64) + 2 LSTM | ~10× ce modèle |
| Embedding + pooling (retenue) | [ceblfe](https://github.com/ceblfe/fcc_sms_text_classification) : Embedding(10000, 128) | ~10× ce modèle |

Les hyperparamètres retenus ici (`1000 / 50 / 16 / 24`) n'ont été trouvés chez
personne : ce modèle atteint une justesse comparable (0,98) avec 16 433
paramètres, environ dix fois moins que les solutions publiées.

**Ce qui n'a pas d'équivalent trouvé** : le rappel et la précision sur la classe
spam (les solutions publiques rapportent la justesse seule), la fuite train/valid
documentée plus haut, et le diagnostic du refus des `pandas.Series` par Keras 3.
Le symptôme est connu ([tensorflow#65237](https://github.com/tensorflow/tensorflow/issues/65237),
issue ouverte sans résolution), mais formulé pour des tableaux `numpy.str_`,
pas pour une Series.

## État : défi réussi, 7/7 au test officiel

Pipeline validé, les 7 messages du test officiel sont correctement classés
(graine fixée à 42).

| Métrique | Valeur |
|---|---|
| Justesse (*accuracy*) | 0,984 |
| **Rappel spam** (*recall*) | **0,936** |
| Précision spam (*precision*) | 0,946 |
| Référence « toujours ham » | 0,866 |

Matrice de confusion sur les 1392 messages de validation :

| | prédit ham | prédit spam |
|---|---|---|
| **réel ham** | 1195 | 10 |
| **réel spam** | 12 | 175 |

175 spam trouvés sur 187, 12 manqués. C'est le rappel qui porte l'information :
la justesse de 0,984 se compare à 0,866 pour un modèle qui dirait toujours
« ham », alors que le rappel d'un tel modèle serait nul.

### Le dimensionnement du vocabulaire

`max_tokens=1000` ne couvre pas la totalité du vocabulaire (8181 mots dans le
train) : 79,8 % des tokens rencontrés sont reconnus, le reste devient `[UNK]`.
Porter la limite à 3000 augmente le rappel
spam (0,957) et baisse légèrement la justesse (0,981) : le gain n'est pas net,
et la valeur d'origine est conservée. `output_sequence_length=50` est mieux
calibré, il ne tronque que 1,2 % des messages.

Le livrable freeCodeCamp est un notebook Google Colab : `notebook.py` (Marimo)
tient le raisonnement et les figures, il reste à en porter le contenu dans le
notebook Colab officiel pour la soumission. Vérifier à cette occasion la version
de Keras côté Colab (voir la section sur `Invalid dtype: str`).

URL de soumission : *à compléter une fois le projet soumis.*
