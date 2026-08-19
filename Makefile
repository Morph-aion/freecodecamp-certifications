# Outillage commun aux certifications freeCodeCamp.
#
# Rien n'est installé dans les projets : `uv run --with` résout les outils à la
# volée, comme les tests le font déjà avec pandas et scikit-learn. Un seul
# endroit à mettre à jour quand une commande change.
#
#     make lint     vérifie sans rien modifier
#     make format   applique le formatage
#     make fix      corrige ce qui est corrigeable, puis formate
#     make test     lance la suite d'un projet (PROJET=...)
#     make check    lint + format --check + test : ce qu'on lance avant de pousser

# Version épinglée : ruff 0.16 s'est mis à formater les blocs de code dans les
# fichiers Markdown, ce que 0.15 ignorait. Sans épinglage, la CI installe la
# dernière version et échoue sur des fichiers que le poste local trouve
# conformes. Relever ce numéro est un choix délibéré, pas un effet de bord.
RUFF := uv run --with ruff==0.15.18 ruff
# `uv run test_units.py` et non `python -m unittest` : seule cette forme lit
# l'en-tête PEP 723 du fichier de test et résout ses dépendances (certains
# projets ont besoin de tensorflow ou matplotlib, pas seulement de pandas).
PYTHON_TEST := uv run

# Certification et projet visés par `make test`. Surchargeables :
#     make test PROJET=01-rock-paper-scissors
#     make test CERTIF=data-analysis-with-python PROJET=01-mean-variance-standard-deviation-calculator
CERTIF ?= machine-learning-with-python
PROJET ?= 03-book-recommendation-knn
PROJET_DIR := $(CERTIF)/$(PROJET)

.PHONY: lint format fix test check help

help:
	@grep -E '^#     make' $(MAKEFILE_LIST) | sed 's/^#     //'

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .

fix:
	$(RUFF) check . --fix
	$(RUFF) format .

# `cd` dans le projet : les tests importent leurs modules par nom court
# (`from recommender import ...`) et lisent `data/raw/` en relatif.
test:
	cd $(PROJET_DIR) && $(PYTHON_TEST) test_units.py

check: lint test
