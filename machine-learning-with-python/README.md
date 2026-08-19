# Machine Learning with Python

https://www.freecodecamp.org/learn/machine-learning-with-python/

Certification validée par les 5 projets ci-dessous, chacun vérifié contre la suite de tests
officielle de freeCodeCamp.

## Projets

| # | Projet | Nature | Statut | Dossier |
|---|---|---|---|---|
| 1 | Rock-Paper-Scissors | Probabilités/statistique (Markov) | Défi réussi, à soumettre | [01-rock-paper-scissors/](01-rock-paper-scissors/) |
| 2 | Cat and Dog Image Classifier | Deep learning (CNN, TensorFlow/Keras) | Défi réussi, à soumettre | [02-cat-dog-classifier/](02-cat-dog-classifier/) |
| 3 | Book Recommendation Engine (KNN) | ML classique (k-plus-proches-voisins) | Défi réussi, à soumettre | [03-book-recommendation-knn/](03-book-recommendation-knn/) |
| 4 | Linear Regression Health Costs Calculator | ML classique (régression linéaire) | Défi réussi, à soumettre | [04-health-costs-regression/](04-health-costs-regression/) |
| 5 | Neural Network SMS Text Classifier | Deep learning (NLP, réseau de neurones) | Défi réussi, à soumettre | [05-sms-text-classifier/](05-sms-text-classifier/) |

**Deux familles de projets.** Cat/Dog et SMS mobilisent un réseau de neurones (TensorFlow,
Keras) ; RPS, KNN et régression relèvent du machine learning et de la statistique classiques.
Les premiers demandent plus de temps de calcul et de débogage, les seconds plus de raisonnement
sur la représentation des données.

## Format de travail, et format de soumission

Deux choses distinctes, qui ne se confondent pas.

**Le travail** se fait ici, en Python exécutable : la logique dans un module
(`regression.py`, `classifier.py`…), les tests dans `test_units.py`, et un
notebook **Marimo** (`notebook.py`) qui orchestre et commente sans héberger de
logique. Les dépendances sont déclarées en en-tête PEP 723, donc
`uv run notebook.py` suffit, sans environnement à installer.

**La soumission** suit ce qu'impose chaque énoncé, et ce n'est pas le même
support :

| Projet | Support de soumission |
|---|---|
| 01 Rock-Paper-Scissors | code de départ fourni via Ona, soumission de l'URL |
| 02, 03, 04, 05 | notebook **Google Colaboratory**, partage du lien activé |

Pour les quatre projets Colab, le contenu du module est à porter dans les
cellules du notebook officiel, entre les cellules d'import et la cellule de test
fournies par freeCodeCamp. Le notebook Marimo de ce dépôt ne se colle pas tel
quel : c'est un format différent, il sert à comprendre et à mesurer, pas à
soumettre.
