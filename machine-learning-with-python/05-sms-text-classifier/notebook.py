# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib",
#     "pandas",
#     "scikit-learn",
#     "tensorflow",
# ]
# ///

# Notebook du projet 05 : Neural Network SMS Text Classifier.
# Aucune logique ici : tout vient de classifier.py (convention du projet 01).
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
    from classifier import (
        SEED,
        TEST_ANSWERS,
        TEST_MESSAGES,
        build_model,
        build_vectorizer,
        encode_labels,
        evaluate_model,
        load_data,
        predict_message,
        train_model,
    )
    from tensorflow import keras

    return (
        SEED,
        TEST_ANSWERS,
        TEST_MESSAGES,
        build_model,
        build_vectorizer,
        encode_labels,
        evaluate_model,
        keras,
        load_data,
        mo,
        plt,
        predict_message,
        train_model,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Neural Network SMS Text Classifier

        Classifier des SMS en **ham** (normal) ou **spam** (publicité) avec un
        réseau de neurones Keras. Le jeu de données est déjà séparé en train/test
        par freeCodeCamp.

        **Test officiel** : 7 messages spécifiques doivent être correctement classés.

        Ce notebook est un squelette prêt à travailler : données chargées,
        pipeline validé, cellule de test officielle en dernière position.
        Chaque cellule se réexécute quand ce dont elle dépend change.
        """
    )
    return


@app.cell
def _(load_data, mo):
    train_df, valid_df = load_data()
    total_spam = (train_df["type"] == "spam").sum()
    total_ham = (train_df["type"] == "ham").sum()

    mo.md(
        f"""
        ## 1. Les données

        **Train** : {len(train_df)} messages ({total_ham} ham, {total_spam} spam)
        **Valid** : {len(valid_df)} messages

        | Colonne | Type | Contenu |
        |---|---|---|
        | `type` | catégorielle | `ham` ou `spam` |
        | `text` | texte | Le message SMS brut |

        **Déséquilibre** : {total_ham / len(train_df) * 100:.1f} % ham, {total_spam / len(train_df) * 100:.1f} % spam.
        Un modèle trivial qui prédit toujours « ham » obtient déjà
        {total_ham / len(train_df) * 100:.1f} % de justesse.
        """
    )
    return (train_df, valid_df)


@app.cell
def _(mo, plt, train_df):
    # Variables préfixées par « _ » : Marimo les traite comme locales à la
    # cellule, ce qui autorise la réutilisation des noms fig/axes d'une cellule
    # graphique à l'autre sans MultipleDefinitionError.
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

    _counts = train_df["type"].value_counts()
    _axes[0].bar(_counts.index, _counts.values, color=["#2ecc71", "#e74c3c"])
    _axes[0].set_ylabel("Nombre de messages")
    _axes[0].set_title("Distribution ham / spam")
    for _i, _v in enumerate(_counts.values):
        _axes[0].text(_i, _v + 20, str(_v), ha="center", fontweight="bold")

    _lengths = train_df["text"].str.len()
    _axes[1].hist(
        _lengths[train_df["type"] == "ham"],
        bins=30,
        alpha=0.6,
        label="ham",
        color="#2ecc71",
    )
    _axes[1].hist(
        _lengths[train_df["type"] == "spam"],
        bins=30,
        alpha=0.6,
        label="spam",
        color="#e74c3c",
    )
    _axes[1].set_xlabel("Longueur du message (caractères)")
    _axes[1].set_ylabel("Nombre de messages")
    _axes[1].set_title("Distribution des longueurs")
    _axes[1].legend()

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(
    SEED, build_model, build_vectorizer, encode_labels, keras, mo, train_df, valid_df
):
    # Graine fixée avant la construction du modèle : sans elle l'initialisation
    # des poids varie d'une exécution à l'autre, et un message limite du test
    # officiel peut basculer d'un côté ou de l'autre du seuil de 0,5.
    keras.utils.set_random_seed(SEED)

    train_labels = encode_labels(train_df)
    valid_labels = encode_labels(valid_df)

    vectorize_layer = build_vectorizer(train_df["text"])
    model = build_model(vectorize_layer)

    mo.md(
        """
        ## 2. Architecture du modèle

        ```
        TextVectorization(max_tokens=1000, output_sequence_length=50)
        → Embedding(1000, 16)
        → GlobalAveragePooling1D()
        → Dense(24, relu)
        → Dense(1, sigmoid)
        ```

        **Paramètres** : ~17 000. Modèle léger, entraînement rapide.

        La couche `TextVectorization` convertit le texte brut en séquence
        d'entiers. `Embedding` apprend un vecteur dense par mot.
        `GlobalAveragePooling1D` moyenne les vecteurs de la séquence en un
        seul vecteur fixe : c'est le choix le plus simple, et il suffit ici
        parce que les SMS sont courts et que l'ordre des mots compte moins
        que la présence de mots-clés (« free », « cash », « call »).
        """
    )
    return (model, train_labels, valid_labels)


@app.cell
def _(mo, model, train_df, train_labels, train_model, valid_df, valid_labels):
    history = train_model(
        model, train_df["text"], train_labels, valid_df["text"], valid_labels
    )

    mo.md("## 3. Entraînement : voir les courbes ci-dessous")
    return (history,)


@app.cell
def _(history, mo, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

    _axes[0].plot(history.history["accuracy"], label="train")
    _axes[0].plot(history.history["val_accuracy"], label="validation")
    _axes[0].set_xlabel("Epoch")
    _axes[0].set_ylabel("Justesse (accuracy)")
    _axes[0].set_title("Justesse (accuracy)")
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(history.history["loss"], label="train")
    _axes[1].plot(history.history["val_loss"], label="validation")
    _axes[1].set_xlabel("Epoch")
    _axes[1].set_ylabel("Perte (loss)")
    _axes[1].set_title("Perte (loss, binary_crossentropy)")
    _axes[1].legend()
    _axes[1].grid(True, alpha=0.3)

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(history, mo):
    # Lecture des courbes plutôt que simple affichage : c'est le diagnostic
    # standard d'un réseau, et il détermine si 30 epochs est le bon budget.
    _vl = history.history["val_loss"]
    _tl = history.history["loss"]
    _meilleure = _vl.index(min(_vl)) + 1
    _ecart_final = _vl[-1] - _tl[-1]

    mo.md(
        f"""
        ### Ce que les courbes disent

        La perte de validation atteint son minimum à l'**epoch {_meilleure}**
        sur les {len(_vl)} exécutées, puis remonte : au-delà, le modèle continue
        d'améliorer son ajustement au jeu d'entraînement sans gagner en
        généralisation. C'est du **surapprentissage**, visible à l'écart entre
        les deux courbes de perte, qui atteint {_ecart_final:.3f} à la fin
        contre {_vl[0] - _tl[0]:+.3f} au départ.

        Deux lectures possibles, et elles ne s'opposent pas :

        - **30 epochs est un budget trop large.** Un `EarlyStopping` sur
          `val_loss` avec patience arrêterait vers l'epoch {_meilleure}, pour un
          résultat équivalent en une fraction du temps.
        - **Le surapprentissage reste sans conséquence ici.** La justesse de
          validation ne se dégrade pas ; c'est la perte qui monte, parce que le
          modèle devient trop confiant sur ses erreurs. Le test officiel se
          jugeant sur des décisions binaires et non sur des probabilités
          calibrées, le seuil de 0,5 absorbe cette dérive.

        Le choix de garder 30 epochs est donc assumé plutôt que subi : il rend
        le résultat reproductible sans dépendre d'un critère d'arrêt.
        """
    )
    return


@app.cell
def _(evaluate_model, mo, model, valid_df, valid_labels):
    metrics = evaluate_model(model, valid_df["text"], valid_labels)
    _reference = 1 - valid_labels.mean()
    (_tn, _fp), (_fn, _tp) = metrics["confusion"]

    mo.md(
        f"""
        ## 3 bis. Les métriques qui comptent

        La justesse seule est trompeuse sur un jeu déséquilibré : prédire
        toujours « ham » donnerait déjà **{_reference:.1%}**.

        | Métrique | Valeur |
        |---|---|
        | Justesse (*accuracy*) | {metrics["accuracy"]:.4f} |
        | **Rappel spam** (*recall*) | **{metrics["recall_spam"]:.4f}** |
        | Précision spam (*precision*) | {metrics["precision_spam"]:.4f} |
        | **F1 spam** | **{metrics["f1_spam"]:.4f}** |
        | Référence « toujours ham » | {_reference:.4f} |

        « Précision » est le terme à surveiller : en français courant il évoque
        l'exactitude générale, alors qu'en ML la *precision* est le taux de
        justesse des alertes spam, à ne pas confondre avec la justesse globale.

        Le **F1** est leur moyenne harmonique. Contrairement à une moyenne
        simple, il s'effondre dès que l'un des deux s'effondre : un modèle au
        rappel nul aurait un F1 nul, quelle que soit sa précision. C'est la
        métrique de synthèse qui résiste au déséquilibre de classes.

        **Matrice de confusion**

        | | prédit ham | prédit spam |
        |---|---|---|
        | **réel ham** | {_tn} | {_fp} |
        | **réel spam** | {_fn} | {_tp} |

        Le rappel dit quelle proportion des spam a été trouvée :
        {_tp} sur {_tp + _fn}. Les {_fn} spam manqués sont l'erreur qui compte
        ici, alors que la justesse globale les noie dans la masse des ham.
        """
    )
    return


@app.cell
def _(metrics, mo, valid_df, valid_labels):
    # Les erreurs individuelles, que les métriques agrégées masquent : c'est là
    # que se lit ce que le modèle ne sait pas faire.
    _prob = metrics["y_prob"]
    _pred = (_prob >= 0.5).astype(int)
    _vrai = valid_labels.astype(int)

    _fn = [
        (t, p)
        for t, p, y, q in zip(valid_df["text"], _prob, _vrai, _pred, strict=True)
        if y == 1 and q == 0
    ]
    _fp = [
        (t, p)
        for t, p, y, q in zip(valid_df["text"], _prob, _vrai, _pred, strict=True)
        if y == 0 and q == 1
    ]

    _entete = "| Message | Probabilité spam |\n|---|---|\n"
    _lignes_fn = _entete + "".join(
        f"| `{_t[:58]}` | {_p:.3f} |\n"
        for _t, _p in sorted(_fn, key=lambda e: -abs(e[1] - 0.5))[:5]
    )
    _lignes_fp = _entete + "".join(
        f"| `{_t[:58]}` | {_p:.3f} |\n"
        for _t, _p in sorted(_fp, key=lambda e: -abs(e[1] - 0.5))[:5]
    )

    mo.md(
        f"""
        ## 3 ter. Les erreurs, une par une

        {len(_fn)} spam manqués (*faux négatifs*) et {len(_fp)} ham signalés à
        tort (*faux positifs*) sur {len(_vrai)} messages de validation.

        Les cinq plus flagrants de chaque catégorie, classés par distance au
        seuil : ce sont les erreurs dont le modèle est le plus « sûr », donc les
        plus révélatrices.

        **Spam manqués**

        {_lignes_fn}
        **Ham signalés à tort**

        {_lignes_fp}

        Sur ce corpus, les faux négatifs sont typiquement des spam courts ou
        rédigés sans les mots-clés commerciaux appris (« free », « call »,
        « win »), que le modèle ne peut pas distinguer d'un message ordinaire.
        C'est la limite d'un sac de mots moyenné : sans l'ordre ni le contexte,
        un spam qui n'emploie pas le vocabulaire attendu passe.
        """
    )
    return


@app.cell
def _(TEST_ANSWERS, TEST_MESSAGES, mo, model, predict_message):
    # Les 7 messages viennent de classifier.py : une seule source, pas de copie
    # à maintenir en parallèle du script et des tests.
    _passed = True
    _rows = ""
    _marges = []
    for _msg, _ans in zip(TEST_MESSAGES, TEST_ANSWERS, strict=True):
        _prob, _pred = predict_message(model, _msg)
        _marges.append((abs(_prob - 0.5), _msg))
        if _pred != _ans:
            _passed = False
        _status = "OK" if _pred == _ans else "RATÉ"
        _short = _msg[:40] + "…" if len(_msg) > 40 else _msg
        _rows += f"| {_status} | `{_short}` | {_prob:.4f} | **{_pred}** | {_ans} |\n"

    _marge, _limite = min(_marges)

    mo.md(
        f"""
        ## 4. Test officiel freeCodeCamp

        ```python
        def predict_message(pred_text):
            # returns [probability, "ham" or "spam"]
        ```

        | | Message | Prob. | Prédit | Attendu |
        |---|---|---|---|---|
        {_rows}

        **{"Challenge réussi." if _passed else "Certains messages sont mal classés."}**

        Marge la plus faible : **{_marge:.3f}** du seuil de 0,5, sur le message
        « {_limite[:50]}… ». Une prédiction à 0,52 serait techniquement juste
        mais fragile : c'est exactement ce message qui bascule quand la graine
        change, et la raison pour laquelle `SEED` est fixé avant la construction
        du modèle.
        """
    )
    return
