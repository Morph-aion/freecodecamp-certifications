# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scikit-learn",
#     "tensorflow",
# ]
# ///
"""Pipeline complet : chargement, vectorisation, modèle Keras, entraînement, évaluation.

Classifie des SMS en ham/spam avec un réseau de neurones.
Test officiel freeCodeCamp : 7 messages correctement classés.
"""

import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "raw"
DATA_BASE_URL = "https://cdn.freecodecamp.org/project-data/sms"
DATA_FILES = ("train-data.tsv", "valid-data.tsv")
MAX_TOKENS = 1000
OUTPUT_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 16
EPOCHS = 30
BATCH_SIZE = 32
SEED = 42

# Les 7 messages du test officiel freeCodeCamp, source unique : le notebook et
# les tests les importent d'ici plutôt que d'en garder une copie.
TEST_MESSAGES = (
    "how are you doing today",
    "sale today! to stop texts call 98912460324",
    "i dont want to go. can we try it a different day? available sat",
    "our new mobile video service is live. just install on your phone to start watching.",
    "you have won £1000 cash! call to claim your prize.",
    "i'll bring it tomorrow. don't forget the milk.",
    "wow, is your arm alright. that happened to me one time too",
)
TEST_ANSWERS = ("ham", "spam", "ham", "spam", "spam", "ham", "ham")


def download_file(name: str, dest: pathlib.Path) -> None:
    """Télécharge un fichier de données depuis le CDN freeCodeCamp."""
    url = f"{DATA_BASE_URL}/{name}"
    print(f"Téléchargement de {name} depuis {url}…")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Le CDN freeCodeCamp renvoie 403 sans User-Agent explicite : celui par
    # défaut d'urllib (« Python-urllib/x.y ») est filtré.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            dest.write_bytes(response.read())
    except OSError as err:
        print(f"Erreur : téléchargement impossible ({err})")
        print(f"Télécharger manuellement {url} vers {dest}")
        sys.exit(1)


def load_data():
    """Charge les TSV train et valid, en les téléchargeant si absents."""
    paths = []
    for name in DATA_FILES:
        path = DATA_DIR / name
        if not path.exists():
            download_file(name, path)
        paths.append(path)

    # Tuple explicite plutôt qu'un générateur déballé : avec un générateur, un
    # nombre de fichiers différent de deux lèverait un ValueError sans message
    # exploitable, loin de la cause.
    train_df, valid_df = (
        pd.read_csv(path, sep="\t", header=None, names=["type", "text"])
        for path in paths
    )
    return train_df, valid_df


def encode_labels(df):
    """Convertit ham/spam en 0/1.

    Une valeur inattendue produirait un NaN silencieux via `map`, et Keras
    s'entraînerait sur des étiquettes NaN sans rien signaler : le cas est donc
    vérifié ici plutôt que laissé aux seuls tests.
    """
    labels = df["type"].map({"ham": 0, "spam": 1})
    if labels.isna().any():
        inconnues = sorted(set(df.loc[labels.isna(), "type"].unique()))
        raise ValueError(
            f"Étiquettes inattendues, « ham » ou « spam » attendus : {inconnues}"
        )
    return labels.values.astype("float32")


def as_text_array(texts) -> np.ndarray:
    """Convertit des textes en tableau numpy d'objets, exploitable par Keras 3.

    Keras 3 refuse les pandas.Series de chaînes (« Invalid dtype: str ») et les
    tableaux numpy de dtype `<U…` : seul le dtype `object` est accepté en
    entrée d'une couche TextVectorization.
    """
    return np.asarray(texts, dtype=object)


def build_vectorizer(train_texts):
    """Construit la couche TextVectorization sur les textes d'entraînement."""
    vectorize_layer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    )
    vectorize_layer.adapt(as_text_array(train_texts))
    return vectorize_layer


def build_model(vectorize_layer):
    """Construit le modèle Keras."""
    model = keras.Sequential(
        [
            # Input explicite : sans lui les couches restent « unbuilt » et
            # model.summary() annonce « Total params: 0 » jusqu'au premier fit.
            keras.Input(shape=(), dtype=tf.string),
            vectorize_layer,
            layers.Embedding(MAX_TOKENS, EMBEDDING_DIM),
            layers.GlobalAveragePooling1D(),
            layers.Dense(24, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model,
    train_texts,
    train_labels,
    valid_texts,
    valid_labels,
    epochs: int = EPOCHS,
    verbose: int = 1,
):
    """Entraîne le modèle."""
    history = model.fit(
        as_text_array(train_texts),
        train_labels,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(as_text_array(valid_texts), valid_labels),
        verbose=verbose,
    )
    return history


def predict_message(model, text):
    """Prédit ham/spam pour un message.

    Renvoie [probabilité, "ham" ou "spam"] : la probabilité est celle de la
    classe spam (0 = ham, 1 = spam), conformément à l'énoncé.
    """
    prob = float(model.predict(as_text_array([text]), verbose=0)[0][0])
    label = "spam" if prob >= 0.5 else "ham"
    return [prob, label]


def evaluate_model(model, valid_texts, valid_labels) -> dict:
    """Évalue le modèle : justesse, mais surtout rappel et précision sur spam.

    Le jeu est déséquilibré (~87 % ham) : un modèle qui prédit toujours « ham »
    atteint déjà 87 % de justesse. Le rappel sur la classe spam est la métrique
    qui distingue un vrai classifieur d'un modèle dégénéré.
    """
    y_prob = model.predict(as_text_array(valid_texts), verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.asarray(valid_labels).astype(int)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "recall_spam": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_spam": float(precision_score(y_true, y_pred, zero_division=0)),
        # Moyenne harmonique du rappel et de la précision : elle s'effondre dès
        # que l'un des deux s'effondre, contrairement à leur moyenne simple.
        # C'est la métrique de synthèse usuelle sur un jeu déséquilibré.
        "f1_spam": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred),
        "y_prob": y_prob,
    }


def test_official(model):
    """Teste les 7 messages du test officiel freeCodeCamp."""
    passed = True
    for msg, ans in zip(TEST_MESSAGES, TEST_ANSWERS, strict=True):
        prediction = predict_message(model, msg)
        status = "OK  " if prediction[1] == ans else "RATÉ"
        if prediction[1] != ans:
            passed = False
        print(f"  {status} {prediction[1]:4s} (p={prediction[0]:.4f}), attendu : {ans}")
        print(f"     «{msg}»")

    return passed


def main():
    """Pipeline complet."""
    keras.utils.set_random_seed(SEED)

    print("Chargement des données…")
    train_df, valid_df = load_data()
    print(
        f"  Train : {len(train_df)} messages ({(train_df['type'] == 'spam').sum()} spam)"
    )
    print(
        f"  Valid : {len(valid_df)} messages ({(valid_df['type'] == 'spam').sum()} spam)"
    )

    train_labels = encode_labels(train_df)
    valid_labels = encode_labels(valid_df)

    print("Vectorisation…")
    vectorize_layer = build_vectorizer(train_df["text"])

    print("Construction du modèle…")
    model = build_model(vectorize_layer)
    model.summary()

    print(f"\nEntraînement ({EPOCHS} epochs)…")
    train_model(model, train_df["text"], train_labels, valid_df["text"], valid_labels)

    print("\nÉvaluation sur le jeu de validation…")
    metrics = evaluate_model(model, valid_df["text"], valid_labels)
    taux_ham = 1 - valid_labels.mean()
    # Étiquettes en français suivies du terme anglais : c'est sous ce nom que
    # les métriques apparaissent dans scikit-learn et dans la littérature.
    print(f"  Justesse (accuracy)        : {metrics['accuracy']:.4f}")
    print(f"  Rappel spam (recall)       : {metrics['recall_spam']:.4f}")
    print(f"  Précision spam (precision) : {metrics['precision_spam']:.4f}")
    print(f"  F1 spam                    : {metrics['f1_spam']:.4f}")
    print(f"  Référence « toujours ham » : {taux_ham:.4f} de justesse")
    (tn, fp), (fn, tp) = metrics["confusion"]
    print(
        f"  Confusion : {tp} spam trouvés, {fn} manqués, {fp} ham signalés à tort, {tn} ham corrects"
    )

    print("\nTest officiel freeCodeCamp :")
    passed = test_official(model)
    if passed:
        print("\nLes 7 messages sont correctement classés")
    else:
        print("\nCertains messages sont mal classés")


if __name__ == "__main__":
    main()
