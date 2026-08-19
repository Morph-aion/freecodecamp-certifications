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

- [1. La vectorisation de texte](#1-la-vectorisation-de-texte)
- [2. L'embedding : représentation dense des mots](#2-lembedding-représentation-dense-des-mots)
- [3. GlobalAveragePooling1D : résumer une séquence](#3-globalaveragepooling1d-résumer-une-séquence)
- [4. Classification binaire et sigmoid](#4-classification-binaire-et-sigmoid)
- [5. Le déséquilibre de classes](#5-le-déséquilibre-de-classes)
- [6. Notions voisines, non implémentées ici](#6-notions-voisines-non-implémentées-ici)

## 1. La vectorisation de texte

`TextVectorization(max_tokens=1000, output_sequence_length=50)` :
[classifier.py:99-103](../classifier.py#L99-L103).

**Le problème** : un réseau de neurones ne peut pas traiter du texte brut. Il a besoin de
nombres. La vectorisation convertit chaque message en une séquence d'entiers, où chaque entier
représente un mot du vocabulaire.

**Ce que fait `adapt`** : parcourt tous les messages d'entraînement, construit le vocabulaire
(les 1000 mots les plus fréquents), et crée la table de correspondance mot → indice.

**Pourquoi `max_tokens=1000`** : les SMS sont courts (rarement plus de 30 mots) et le
vocabulaire est restreint (pas de jargon technique). Le vocabulaire complet du train compte
8181 mots ; en le plafonnant à 1000, **79,8 % des tokens rencontrés** sont reconnus, le reste
tombant dans `[UNK]`.

Attention au décompte : sur les 1000 places, deux sont réservées par Keras au padding (indice 0)
et à `[UNK]` (indice 1) : il ne reste que 998 mots réels. La couverture se mesure sur les tokens
effectivement vectorisés, pas sur la taille du vocabulaire.

**Pourquoi `output_sequence_length=50`** : tous les messages doivent avoir la même longueur
pour former un batch. Les messages plus courts sont padés, les plus longs sont tronqués.
50 est une borne haute confortable.

## 2. L'embedding : représentation dense des mots

`Embedding(MAX_TOKENS, EMBEDDING_DIM)` : [classifier.py:115](../classifier.py#L115).

**Le principe** : au lieu de représenter un mot par un entier (one-hot creux et de haute
dimension), l'embedding apprend un vecteur dense de dimension fixe (ici 16) pour chaque mot.
Les mots similaires ont des vecteurs proches dans cet espace.

**Ce qui est appris** : pendant l'entraînement, les vecteurs d'embedding sont ajustés pour
minimiser la loss. Au début, ils sont aléatoires. À la fin, ils encodent des informations
sémantiques : « free » et « win » seront plus proches que « free » et « milk ».

**Analogie** : chaque mot est un point dans un espace à 16 dimensions. Les messages ham et
spam forment des nuages distincts dans cet espace, et le réseau apprend à les séparer.

## 3. GlobalAveragePooling1D : résumer une séquence

`GlobalAveragePooling1D()` : [classifier.py:116](../classifier.py#L116).

**Le problème** : après l'embedding, chaque message est une matrice de forme `(50, 16)`,
50 positions, chacune un vecteur de 16 dimensions. Mais le réseau dense attend un vecteur
1D fixe.

**Ce que fait le pooling** : il moyenne les 50 vecteurs de position en un seul vecteur de
dimension 16. C'est la moyenne des représentations de tous les mots du message.

**Pourquoi pas un LSTM** : pour des SMS courts (10-20 mots en moyenne), l'information
séquentielle compte moins que la présence de mots-clés. « free », « cash », « call » sont
des signaux spam quel que soit leur ordre. Le pooling est plus rapide et suffisant.

**Ce qu'on perd** : l'ordre des mots. « not happy » et « happy not » donneraient le même
résumé. Pour des SMS, c'est un arbitrage acceptable.

## 4. Classification binaire et sigmoid

`Dense(1, activation="sigmoid")` : [classifier.py:118](../classifier.py#L118).

**Le principe** : la sigmoïde écrase toute valeur réelle dans l'intervalle [0, 1]. C'est
une probabilité : 0 = ham (négatif), 1 = spam (positif).

**Pourquoi `binary_crossentropy`** : c'est la loss standard pour la classification binaire.
Elle mesure l'écart entre la probabilité prédite et le label réel (0 ou 1). Quand le modèle
prédit 0.99 pour un ham (label 0), la loss est très élevée, ce qui pousse le modèle à se
corriger.

**Le seuil 0.5** : par convention, on classe en « ham » si p < 0.5, « spam » si p ≥ 0.5.
Ce seuil peut être ajusté pour favoriser le rappel (*recall* : détecter plus de spam au prix
de faux positifs) ou la précision (*precision* : moins de faux positifs au prix de spam
manqués).

## 5. Le déséquilibre de classes

Mesuré sur le jeu d'entraînement : **86,6 % ham, 13,4 % spam**.

**Le piège** : un modèle trivial qui prédit toujours « ham » obtient 86,6 % de justesse
(*accuracy*). C'est un faux sentiment de performance.

**Ce qu'il faut regarder** :
- Le **rappel sur spam** (*recall*) : combien de spam le modèle détecte-t-il ? (vrais
  positifs / spam totaux)
- La **précision** (*precision*) : parmi les messages prédits spam, combien le sont
  réellement ? Attention au faux ami : en français courant « précision » évoque l'exactitude
  générale, alors qu'il s'agit ici du taux de justesse des seules alertes spam.
- La **matrice de confusion** (*confusion matrix*) : combien de ham prédits spam (faux
  positifs), combien de spam prédits ham (faux négatifs) ?

Mesuré par `evaluate_model()` : justesse 0,984, rappel spam 0,936, précision spam 0,946,
contre 0,866 pour la référence « toujours ham ».

**Pourquoi on ne corrige pas le déséquilibre** : il n'est pas extrême (87/13, pas 99/1), et le
test officiel freeCodeCamp vérifie les 7 messages un par un, pas une métrique agrégée.
Un modèle bien entraîné apprend naturellement les deux classes.

## 6. Notions voisines, non implémentées ici

- **LSTM / GRU** : couches récurrentes qui capturent l'ordre des mots. Plus lents à
  entraîner, plus utiles pour des textes longs. La comparaison RNN/transformer sur ce
  corpus reste une piste ouverte.
- **TF-IDF** : pondération classique qui donne plus de poids aux mots rares. Utilisable
  avec scikit-learn (MultinomialNB, LogisticRegression) mais ce n'est pas un réseau de
  neurones.
- **Transformers (BERT, etc.)** : modèles de pointe en NLP, mais massivement surdimensionnés
  pour des SMS. Un petit transformer (comparé dans le doc de recherche) serait plus adapté.
- **Sous-échantillonnage / sur-échantillonnage** : techniques pour rééquilibrer les classes.
  Inutile ici vu le faible déséquilibre et la taille du corpus.
