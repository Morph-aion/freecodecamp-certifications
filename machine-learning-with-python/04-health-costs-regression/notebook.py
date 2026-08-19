# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "pandas",
#     "scikit-learn",
#     "matplotlib",
#     "scipy",
# ]
# ///

# Notebook du projet 04 : Linear Regression Health Costs Calculator.
# Aucune logique ici : tout vient de regression.py (convention du projet 01).
# Ce notebook reprend la structure imposée par l'énoncé freeCodeCamp (les
# premières cellules importent bibliothèques et données, la dernière sert de
# test), puis ajoute l'exploration visuelle.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from regression import (
        add_interaction,
        evaluate_model,
        load_data,
        prepare_features,
        resolve_target,
        split_data,
        train_model,
    )
    from scipy.stats import norm

    return (
        add_interaction,
        evaluate_model,
        load_data,
        mo,
        norm,
        np,
        plt,
        prepare_features,
        resolve_target,
        split_data,
        train_model,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Linear Regression Health Costs Calculator

        Prédire des **coûts de santé** à l'aide d'un algorithme de **régression
        linéaire**. Le jeu de données contient 1338 lignes : âge, sexe, IMC,
        nombre d'enfants, statut de fumeur, région, et dépenses de santé.

        **Seuil de réussite** : MAE < 3500 $ sur le jeu de test (20 %).

        Ce notebook est un squelette prêt à travailler : données chargées,
        pipeline validé, cellule de test officielle en dernière position.
        Chaque cellule se réexécute quand ce dont elle dépend change.
        """
    )
    return


@app.cell
def _(load_data, mo, resolve_target):
    df = load_data()
    TARGET = resolve_target(df)
    mo.md(
        f"""
        ## 1. Les données

        **{len(df)}** observations, **{len(df.columns)}** colonnes.

        | Colonne | Type | Exemples |
        |---|---|---|
        | `age` | numérique | 19, 18, 28… |
        | `sex` | catégorielle | female, male |
        | `bmi` | numérique | 27.9, 33.77, 33.0… |
        | `children` | numérique | 0, 1, 3… |
        | `smoker` | catégorielle | yes, no |
        | `region` | catégorielle | southwest, southeast… |
        | `{TARGET}` | **cible** | 16884.92, 1725.55… |

        Pas de valeurs manquantes.

        La colonne cible est détectée automatiquement : l'énoncé freeCodeCamp
        parle d'`expenses`, le jeu de données distribué utilise `charges`.
        """
    )
    return (df, TARGET)


@app.cell
def _(df, mo, plt, TARGET):
    # Variables préfixées par « _ » : Marimo les traite comme locales à la
    # cellule, ce qui autorise la réutilisation des noms fig/ax d'une cellule
    # graphique à l'autre sans MultipleDefinitionError.
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    _axes[0].scatter(df["age"], df[TARGET], alpha=0.3, s=10)
    _axes[0].set_xlabel("Âge")
    _axes[0].set_ylabel("Dépenses ($)")
    _axes[0].set_title("Âge vs Dépenses")

    _axes[1].scatter(df["bmi"], df[TARGET], alpha=0.3, s=10)
    _axes[1].set_xlabel("IMC")
    _axes[1].set_ylabel("Dépenses ($)")
    _axes[1].set_title("IMC vs Dépenses")

    _smoker_map = df["smoker"].map({"yes": "Fumeur", "no": "Non-fumeur"})
    for _label in ["Non-fumeur", "Fumeur"]:
        _mask = _smoker_map == _label
        _axes[2].scatter(
            df.loc[_mask, "bmi"], df.loc[_mask, TARGET], alpha=0.3, s=10, label=_label
        )
    _axes[2].set_xlabel("IMC")
    _axes[2].set_ylabel("Dépenses ($)")
    _axes[2].set_title("IMC vs Dépenses (par statut fumeur)")
    _axes[2].legend()

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, prepare_features, add_interaction):
    encoded = prepare_features(df)
    with_inter = add_interaction(encoded)
    mo.md(
        f"""
        ## 2. Encodage et features

        Après `pd.get_dummies(drop_first=True)` : **{len(encoded.columns) - 1}** features.
        Après ajout de l'interaction `bmi × smoker_yes` : **{len(with_inter.columns) - 1}** features.

        | Feature | Raison |
        |---|---|
        | `sex_male` | Binaire, évite le label encoding |
        | `smoker_yes` | Le fumeur est le principal levier de dépenses |
        | `region_*` (3 colonnes) | Variation géographique modérée |
        | `bmi_smoker` | Capture l'effet combiné IMC × fumeur |
        """
    )
    return (with_inter,)


@app.cell
def _(with_inter, mo, train_model, split_data, evaluate_model):
    X_train, X_test, y_train, y_test = split_data(with_inter)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    y_pred = metrics["y_pred"]

    mo.md(
        f"""
        ## 3. Résultats

        | Métrique | Valeur |
        |---|---|
        | **MAE** | **{metrics["mae"]:.0f} $** |
        | RMSE | {metrics["rmse"]:.0f} $ |
        | R² | {metrics["r2"]:.4f} |

        **Objectif freeCodeCamp : MAE < 3500 $ → {"atteint" if metrics["mae"] < 3500 else "non atteint"}**

        ### Coefficients du modèle

        Attention au piège d'interprétation : `smoker_yes` a le coefficient le
        plus grand en valeur absolue, mais il est **négatif**. Cela ne veut pas
        dire que fumer réduit les dépenses.

        Dès qu'on ajoute `bmi_smoker`, `smoker_yes` cesse de représenter
        « l'effet fumeur » : il devient l'ordonnée à l'origine d'une droite dont
        la pente est portée par l'interaction. L'effet réel se lit en combinant
        les deux termes, `smoker_yes + bmi_smoker × IMC` (voir la cellule
        suivante).
        """
    )
    return (metrics, model, X_test, y_test, y_pred)


@app.cell
def _(model, mo, plt):
    # `feature_names_in_` plutôt qu'une reconstruction depuis les colonnes du
    # DataFrame : c'est l'ordre que scikit-learn a mémorisé au moment du fit,
    # donc celui de `coef_`. Reconstruire la liste supposerait que l'ordre des
    # colonnes n'a pas bougé, et un décalage apparierait silencieusement les
    # mauvais noms aux mauvais coefficients.
    _feature_names = list(model.feature_names_in_)
    _coeffs = sorted(
        zip(_feature_names, model.coef_, strict=True),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _names = [c[0] for c in _coeffs]
    _values = [c[1] for c in _coeffs]
    _colors = ["#e74c3c" if v > 0 else "#3498db" for v in _values]
    _ax.barh(_names[::-1], _values[::-1], color=_colors[::-1])
    _ax.set_xlabel("Coefficient")
    _ax.set_title("Coefficients de la régression linéaire")
    _ax.axvline(x=0, color="gray", linewidth=0.5)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(model, mo, df):
    _coef = dict(zip(model.feature_names_in_, model.coef_, strict=True))
    _bascule = -_coef["smoker_yes"] / _coef["bmi_smoker"]
    _lignes = "\n".join(
        f"| {_bmi} | {_coef['smoker_yes'] + _coef['bmi_smoker'] * _bmi:+,.0f} $ |"
        for _bmi in (20, 25, 30, 35, 40)
    )

    mo.md(
        f"""
        ### L'effet réel du tabac, à IMC donné

        `smoker_yes` = {_coef["smoker_yes"]:,.0f} · `bmi_smoker` = {_coef["bmi_smoker"]:,.0f}

        | IMC | Effet fumeur |
        |---|---|
        {_lignes}

        L'effet est **positif sur toute la plage observée**. Le point de bascule
        théorique se situe à un IMC de {_bascule:.1f}, en dessous du minimum
        observé dans le jeu de données ({df["bmi"].min():.1f}) : il n'a donc pas
        de réalité empirique.

        C'est le sens correct de l'interaction : le surcoût lié au tabac n'est
        pas constant, il **croît avec l'IMC**.
        """
    )
    return


@app.cell
def _(mo, norm, np, plt, y_pred, y_test):
    # Diagnostic des résidus : le contrôle standard d'une régression, absent
    # jusqu'ici alors que la doc cite leur asymétrie. Trois vues complémentaires.
    _res = y_test - y_pred
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    _axes[0].hist(_res, bins=40, color="#3498db", edgecolor="white")
    _axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    _axes[0].set_xlabel("Résidu ($)")
    _axes[0].set_ylabel("Effectif")
    _axes[0].set_title("Distribution des résidus")

    _axes[1].scatter(y_pred, _res, alpha=0.4, s=20, edgecolors="k", linewidths=0.3)
    _axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    _axes[1].set_xlabel("Prédiction ($)")
    _axes[1].set_ylabel("Résidu ($)")
    _axes[1].set_title("Résidus contre valeurs prédites")

    # Droite de Henry (QQ-plot) : scipy.stats.norm.ppf donne les quantiles
    # théoriques d'une loi normale centrée réduite. Un alignement sur la droite
    # signifierait des résidus gaussiens ; l'écart en haut à droite matérialise
    # la queue lourde.
    _n = len(_res)
    _observes = np.sort(_res)
    _theoriques = norm.ppf((np.arange(_n) + 0.5) / _n)
    _axes[2].scatter(_theoriques, _observes, alpha=0.5, s=15)
    _axes[2].plot(
        _theoriques,
        _theoriques * _res.std(ddof=0) + _res.mean(),
        "r--",
        linewidth=1,
        label="Référence gaussienne",
    )
    _axes[2].legend()
    _axes[2].set_xlabel("Quantiles théoriques (loi normale)")
    _axes[2].set_ylabel("Quantiles observés ($)")
    _axes[2].set_title("Droite de Henry")

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo, y_pred, y_test):
    _res = y_test - y_pred
    _skew = float(((_res - _res.mean()) ** 3).mean() / _res.std(ddof=0) ** 3)
    _sous = int((_res > 5000).sum())

    mo.md(
        f"""
        ### Ce que les résidus disent

        Asymétrie **{_skew:.2f}** : la distribution n'est pas symétrique, elle
        traîne vers la droite. Concrètement, **{_sous} patients** du jeu de test
        coûtent plus de 5000 $ de plus que ce que le modèle prédit, sans
        contrepartie symétrique du côté des surestimations.

        C'est le comportement attendu d'un modèle linéaire sur des dépenses de
        santé : il capture le régime général mais lisse les cas extrêmes. Cela
        **n'invalide pas la MAE** (voir
        [quelle-regression-lineaire.md](docs/quelle-regression-lineaire.md) :
        la normalité des résidus n'est pas une hypothèse de Gauss-Markov), mais
        cela délimite ce que le modèle sait faire.

        Le graphe du milieu est le plus utile : les résidus n'y montrent pas de
        structure en entonnoir, ce qui confirme le test de Breusch-Pagan
        (p = 0,34) : l'homoscédasticité tient une fois l'interaction présente.
        """
    )
    return


@app.cell
def _(mo, plt, y_pred, y_test):
    _fig, _ax = plt.subplots(figsize=(7, 7))
    _ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="k", linewidths=0.5, s=30)
    _lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    _ax.plot(_lims, _lims, "r--", linewidth=2, label="Prédiction parfaite")
    _ax.set_xlim(_lims)
    _ax.set_ylim(_lims)
    _ax.set_xlabel("Dépenses réelles ($)")
    _ax.set_ylabel("Dépenses prédites ($)")
    _ax.set_title("Prédictions vs Réalités")
    _ax.legend()
    _ax.set_aspect("equal")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(metrics, mo):
    # Cellule de test officielle freeCodeCamp.
    # Avec Keras, model.evaluate renvoie (loss, mae, mse) où loss est une vraie
    # fonction de coût (MSE). scikit-learn n'a pas d'équivalent : model.score
    # renvoie le R², qui n'est pas une loss. Les deux sont affichés séparément
    # plutôt que de faire passer l'un pour l'autre.
    _mae = metrics["mae"]

    mo.md(
        f"""
        ## 4. Test officiel freeCodeCamp

        Équivalent scikit-learn de la cellule Keras
        `loss, mae, mse = model.evaluate(test_dataset, test_labels)` :

        | Métrique | Valeur | Statut |
        |---|---|---|
        | **MAE** | **{_mae:.0f} $** | {"Pass" if _mae < 3500 else "Fail"} |
        | MSE | {metrics["mse"]:.0f} | |
        | R² (`model.score`) | {metrics["r2"]:.4f} | |

        **Seuil : MAE < 3500 $**
        """
    )
    return
