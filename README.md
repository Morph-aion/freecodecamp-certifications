# Certifications freeCodeCamp

Suivi des certifications freeCodeCamp préparées dans le cadre de la matière "certification" du M2.

## État d'avancement

| Certification | Statut | Dossier |
|---|---|---|
| Machine Learning with Python | 5 projets réussis, à soumettre | [machine-learning-with-python/](machine-learning-with-python/) |
| Data Analysis with Python | 5 projets réussis, à soumettre | [data-analysis-with-python/](data-analysis-with-python/) |
| Scientific Computing with Python | En attente | |

## Conventions

- Un dossier par certification, un sous-dossier par projet exigé.
- Notebooks en [Marimo](https://marimo.io) sandboxed (`notebook.py`), dépendances déclarées en
  en-tête PEP 723 : pas de `requirements.txt` ni de `pyproject.toml` par projet. Seul le projet
  Cat/Dog n'en a pas : son entraînement dure trop longtemps pour un notebook réactif, la mesure
  multi-graines passe par `repetitions.py`.
- `data/` : présent seulement si le projet manipule un dataset fourni. Les données brutes vivent
  dans `data/raw/`, jamais versionnées : elles sont reproductibles depuis leur URL, qui doit être
  documentée dans le code. Les étages `intermediate/` et `processed/` de la convention Kedro ne
  sont créés que si le projet écrit réellement des données transformées sur disque, ce qui n'est
  pas le cas quand les transformations sont appliquées à la volée en mémoire.
- `models/` : présent seulement si le projet produit un artefact entraîné qu'il vaut la peine
  de conserver.
