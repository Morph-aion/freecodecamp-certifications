# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tensorflow>=2.16",
#     "numpy>=1.26",
#     "matplotlib>=3.10",
#     # Requis par les transformations affines d'ImageDataGenerator (rotation,
#     # décalage, cisaillement). TensorFlow ne le tire pas automatiquement, et
#     # l'absence ne se voit qu'au premier lot augmenté.
#     "scipy>=1.13",
# ]
# ///

"""Cat and Dog Image Classifier, projet 02 de la certification freeCodeCamp.

Classifie des images de chats et de chiens avec un réseau de neurones convolutif
entraîné de zéro. Le défi demande au moins 63 % de justesse sur les 50 images de
test (70 % en bonus).

Pourquoi un CNN entraîné de zéro plutôt qu'un modèle pré-entraîné : l'énoncé
impose `Sequential`, `Conv2D`/`MaxPooling2D` et `ImageDataGenerator`. Le transfer
learning atteindrait 95 % sans effort, mais on n'apprendrait rien du compromis
central de l'exercice, qui est le surapprentissage sur 2000 images seulement.

Ce que l'énoncé impose, et qui explique certains choix qui paraîtraient datés :
    * `ImageDataGenerator` et `flow_from_directory`, dépréciés en Keras 3 au
      profit de `image_dataset_from_directory`, mais explicitement exigés ici ;
    * 4 à 6 transformations aléatoires d'augmentation ;
    * `shuffle=False` sur le jeu de test, pour que l'ordre des prédictions soit
      prévisible.

Pipeline :
    1. télécharger le jeu de données dans `data/` s'il n'y est pas déjà ;
    2. déplacer les images de test dans un sous-répertoire (voir
       `prepare_test_dir`) ;
    3. créer les générateurs (entraînement augmenté, validation, test) ;
    4. construire et entraîner le CNN ;
    5. sauvegarder le modèle dans `models/` ;
    6. prédire les 50 images de test et comparer au corrigé.

Cinq figures sont écrites dans `figures/`, numérotées dans l'ordre du pipeline :
échantillon d'entraînement, effet de l'augmentation, courbes d'apprentissage,
prédictions sur le jeu de test, et cartes d'activation des couches convolutives.
Les trois premières correspondent aux cellules 4, 6 et 9 du notebook officiel.

Lancement :
    uv run classifier.py
"""

import math
import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Figures numérotées dans l'ordre du pipeline, pour qu'un lecteur les parcoure
# comme un récit : générateurs, augmentation, entraînement, résultats, mécanisme.
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Convention `data/raw/` : données brutes, immuables, jamais modifiées sur place.
# Ce projet n'a ni `intermediate/` ni `processed/` : les transformations
# (redimensionnement, normalisation, augmentation) sont appliquées à la volée par
# `ImageDataGenerator`, rien n'est écrit sur disque entre le brut et le modèle.
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

DATA_URL = "https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip"
ZIP_PATH = os.path.join(RAW_DIR, "cats_and_dogs.zip")
DATASET_DIR = os.path.join(RAW_DIR, "cats_and_dogs")

BATCH_SIZE = 128
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Graine globale pour que l'initiale des poids, l'augmentation et le mélange des
# lots soient identiques d'un run à l'autre : deux exécutions sont comparables.
RANDOM_SEED = 42

# 15 epochs suffisent en théorie à dépasser 63 %, mais l'évaluation ne porte que
# sur 50 images : une image mal classée pèse 2 points de pourcentage. On prend de
# la marge plutôt que de jouer sur le fil.
EPOCHS = 30

# Corrigé officiel freeCodeCamp : 1 = chien, 0 = chat, dans l'ordre de production
# du générateur, c'est-à-dire l'ordre LEXICOGRAPHIQUE des noms de fichiers
# (1.jpg, 10.jpg, 11.jpg, ..., 2.jpg, 20.jpg, ...) et non l'ordre numérique.
# Vérifié sur les images : la 2e valeur du corrigé vaut 0 (chat) et correspond
# bien à 10.jpg, qui montre un chat, tandis que 2.jpg montre un chien.
# Ne pas « corriger » cet ordre : voir `predict`.
ANSWERS = [
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]


def download_dataset():
    """Télécharge et extrait le jeu de données dans `data/raw/` si nécessaire.

    L'archive fait environ 73 Mo décompressée. Elle est supprimée après
    extraction pour ne pas doubler l'occupation disque. Le répertoire n'est pas
    versionné : ces données sont reproductibles à l'identique depuis l'URL, seule
    celle-ci mérite d'être conservée dans le code.

    Returns:
        str: Chemin du répertoire `cats_and_dogs` contenant `train`,
            `validation` et `test`.
    """
    if os.path.isdir(os.path.join(DATASET_DIR, "train")):
        print("Jeu de données déjà présent dans data/raw/cats_and_dogs.")
        return DATASET_DIR

    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Téléchargement de {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print("Extraction de l'archive ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)

    os.remove(ZIP_PATH)
    return DATASET_DIR


def prepare_test_dir(test_dir):
    """Place les images de test dans un sous-répertoire.

    `flow_from_directory` attend une arborescence `répertoire/classe/images`, car
    il déduit les étiquettes du nom des sous-dossiers. Or le jeu de test est
    livré à plat et sans étiquettes, ce que l'énoncé signale comme le point « le
    plus délicat » de l'exercice. On crée donc une classe factice `c_and_d` pour
    satisfaire cette contrainte de forme.

    L'opération est idempotente : relancer le script ne déplace rien deux fois.

    Écart assumé à la règle d'immuabilité de `data/raw/` : cette fonction
    réorganise des données brutes sur place. Le compromis est acceptable ici car
    aucune image n'est modifiée ni supprimée, seulement déplacée d'un niveau, et
    le répertoire entier est reconstructible depuis l'URL en cas de doute. La
    solution pure (copier vers `data/processed/`) dupliquerait 73 Mo pour un
    simple changement d'arborescence.

    Args:
        test_dir (str): Répertoire `test` du jeu de données.
    """
    test_subdir = os.path.join(test_dir, "c_and_d")
    os.makedirs(test_subdir, exist_ok=True)

    for filename in os.listdir(test_dir):
        src = os.path.join(test_dir, filename)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(test_subdir, filename)
        if not os.path.exists(dst):
            os.rename(src, dst)


def build_generators(train_dir, validation_dir, test_dir, seed=None):
    """Crée les trois générateurs d'images.

    Seul le générateur d'entraînement applique des transformations aléatoires.
    Augmenter la validation ou le test fausserait la mesure : on veut évaluer sur
    les images réelles, pas sur des variantes déformées.

    Les six transformations retenues (l'énoncé en demande 4 à 6) sont toutes
    plausibles pour des photos d'animaux. Le retournement vertical est
    volontairement exclu : un chat à l'envers n'existe pas dans le jeu de test, et
    l'apprendre gaspillerait de la capacité.

    Args:
        train_dir (str): Répertoire d'entraînement, un sous-dossier par classe.
        validation_dir (str): Répertoire de validation, même structure.
        test_dir (str): Répertoire de test, contenant la classe factice.
        seed (int, optional): Graine posée sur les générateurs d'entraînement et
            de validation. Sans elle, `flow_from_directory` mélange l'ordre des
            lots et tire l'augmentation de façon imprévisible, et deux runs du
            même `RANDOM_SEED` ne seraient pas identiques malgré ce que promet le
            commentaire en tête de module. Passer `None` conserve l'ancien
            comportement non déterministe.

    Returns:
        tuple: Les générateurs `(train, validation, test)`. Le générateur de test
            a `shuffle=False` et `class_mode=None` : pas d'étiquettes à produire,
            et un ordre stable, celui sur lequel le corrigé est indexé.
    """
    train_image_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10.0,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_image_generator = ImageDataGenerator(rescale=1.0 / 255)

    train_data_gen = train_image_generator.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        seed=seed,
    )
    val_data_gen = validation_image_generator.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        seed=seed,
    )
    test_data_gen = test_image_generator.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
    )

    return train_data_gen, val_data_gen, test_data_gen


def build_model():
    """Construit le CNN.

    Architecture classique en entonnoir : quatre blocs `Conv2D` + `MaxPooling2D`
    dont le nombre de filtres croît (16, 32, 64, 64) pendant que la résolution
    décroît. Les premières couches captent des motifs locaux (bords, textures),
    les dernières des structures plus larges (formes, parties d'animal).

    Le `Dropout(0.5)` avant la couche dense n'est pas demandé par l'énoncé mais
    répond au vrai risque de l'exercice : avec 2000 images seulement, un réseau de
    cette taille apprend vite les images par cœur. L'augmentation et le dropout
    sont les deux garde-fous.

    La sortie est un unique neurone sigmoïde : la probabilité que l'image soit un
    chien. La classe chat correspond donc à une probabilité proche de zéro, ce que
    l'ordre alphabétique de `flow_from_directory` garantit (cats avant dogs).

    Returns:
        keras.Model: Modèle compilé, prêt pour `fit`.
    """
    model = Sequential(
        [
            # `Input` explicite plutôt que `input_shape=` dans la première
            # Conv2D : cette seconde forme émet un avertissement en Keras 3.
            Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            Conv2D(16, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_data_gen, val_data_gen):
    """Entraîne le modèle.

    `steps_per_epoch` et `validation_steps` se déduisent du nombre d'images et de
    la taille de batch, avec un arrondi **supérieur** pour ne pas laisser de côté
    le dernier lot incomplet. Les fixer trop bas revient à n'entraîner que sur une
    fraction du jeu : c'est silencieux, aucune erreur n'est levée, et le modèle est
    simplement moins bon.

    Args:
        model (keras.Model): Modèle compilé.
        train_data_gen: Générateur d'entraînement.
        val_data_gen: Générateur de validation.

    Returns:
        keras.callbacks.History: Historique des métriques par epoch, consommé par
            `save_history_plots`.
    """
    steps_per_epoch = math.ceil(train_data_gen.samples / BATCH_SIZE)
    validation_steps = math.ceil(val_data_gen.samples / BATCH_SIZE)
    print(
        f"Entraînement : {train_data_gen.samples} images, "
        f"{steps_per_epoch} lots par epoch, {EPOCHS} epochs."
    )

    return model.fit(
        train_data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=validation_steps,
        verbose=1,
    )


def plot_images(images, out_path, title, probabilities=None, answers=None):
    """Affiche une grille d'images, avec leur prédiction si elle est fournie.

    Équivalent de la fonction `plotImages` du notebook freeCodeCamp (cellule 4),
    étendue pour indiquer si la prédiction était correcte. C'est la visualisation
    la plus instructive du projet : un score de 80 % ne dit pas *sur quoi* le
    modèle échoue, ces images le montrent.

    Args:
        images (numpy.ndarray): Lot d'images normalisées entre 0 et 1.
        out_path (str): Chemin du PNG à écrire.
        title (str): Titre de la figure.
        probabilities (list[float], optional): Probabilité « chien » par image.
        answers (list[int], optional): Vérité terrain, pour colorer les erreurs.
    """
    count = len(images)
    columns = min(5, count)
    rows = math.ceil(count / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows))
    axes = axes.flatten() if count > 1 else [axes]

    for index, (ax, image) in enumerate(zip(axes, images, strict=True)):
        ax.imshow(image)
        ax.axis("off")

        if probabilities is None:
            continue

        probability = probabilities[index]
        etiquette = "chien" if probability > 0.5 else "chat"
        confiance = probability if probability > 0.5 else 1 - probability
        legende = f"{etiquette} {confiance:.0%}"
        couleur = "#4a5568"

        if answers is not None:
            correct = round(probability) == answers[index]
            # Vert / rouge : l'oeil repère les erreurs sans lire les chiffres.
            couleur = "#2f855a" if correct else "#c53030"
            legende += "" if correct else "  (faux)"

        ax.set_title(legende, fontsize=9, color=couleur)

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    print(f"Figure sauvegardée dans {out_path}")


def save_augmentation_sample(train_dir, out_path):
    """Montre la même image transformée cinq fois par l'augmentation.

    Correspond à la cellule 6 du notebook freeCodeCamp. Cette figure rend concret
    ce que `rotation_range=15` ou `zoom_range=0.2` font réellement : sans elle,
    l'augmentation reste une liste d'arguments dont on ne mesure pas l'effet.

    C'est aussi un garde-fou : des transformations trop agressives produiraient
    des images méconnaissables, et le réseau apprendrait sur du bruit. Un coup
    d'oeil suffit à le voir.

    Args:
        train_dir (str): Répertoire d'entraînement.
        out_path (str): Chemin du PNG à écrire.
    """
    generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10.0,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    # batch_size=1 : le générateur renvoie cinq variantes de la MÊME image.
    flow = generator.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=1,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
    )
    variantes = [flow[0][0][0] for _ in range(5)]
    plot_images(
        variantes,
        out_path,
        "Augmentation : cinq variantes tirées du même répertoire",
    )


def save_activation_maps(model, image, out_path, filters_per_layer=6):
    """Montre ce que « voient » les premières couches convolutives.

    Rend visible la hiérarchie des motifs décrite dans
    `docs/notions-mobilisees.md` : les premiers filtres réagissent aux bords et
    aux textures, les suivants à des structures de plus en plus larges, sur une
    résolution de plus en plus faible.

    À lire avec prudence : une carte d'activation montre *où* un filtre s'active,
    pas *ce qu'il représente*. L'interprétation en concepts nommables (« celui-ci
    détecte les oreilles ») est une commodité, pas un fait établi.

    Args:
        model (keras.Model): Modèle entraîné.
        image (numpy.ndarray): Une image normalisée, de forme (H, W, 3).
        out_path (str): Chemin du PNG à écrire.
        filters_per_layer (int): Nombre de filtres affichés par couche.
    """
    couches = [layer for layer in model.layers if isinstance(layer, Conv2D)]
    extracteur = tf.keras.Model(
        inputs=model.inputs, outputs=[layer.output for layer in couches]
    )
    activations = extracteur.predict(np.expand_dims(image, axis=0), verbose=0)

    fig, axes = plt.subplots(
        len(couches),
        filters_per_layer,
        figsize=(2 * filters_per_layer, 2.2 * len(couches)),
    )
    for ligne, (couche, activation) in enumerate(
        zip(couches, activations, strict=True)
    ):
        for colonne in range(filters_per_layer):
            ax = axes[ligne, colonne]
            ax.imshow(activation[0, :, :, colonne], cmap="viridis")
            ax.axis("off")
            if colonne == 0:
                ax.set_ylabel(couche.name)
        # Le nom de la couche et sa résolution, en marge de la première colonne.
        hauteur, largeur = activation.shape[1:3]
        axes[ligne, 0].set_title(
            f"{couche.name}  ({hauteur}×{largeur})", fontsize=9, loc="left"
        )

    fig.suptitle("Cartes d'activation : ce que chaque couche met en évidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    print(f"Cartes d'activation sauvegardées dans {out_path}")


def save_history_plots(history, out_path):
    """Trace la justesse et la perte, entraînement contre validation.

    Ces deux courbes sont l'outil de diagnostic central de ce projet. L'écart
    entre les deux se lit directement : si la justesse d'entraînement grimpe alors
    que celle de validation stagne ou redescend, le modèle surapprend, et il faut
    plus d'augmentation ou plus de dropout, pas plus d'epochs.

    Args:
        history (keras.callbacks.History): Sortie de `model.fit`.
        out_path (str): Chemin du PNG à écrire.
    """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(accuracy) + 1)

    fig, (ax_accuracy, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
    ax_accuracy.plot(epochs_range, accuracy, label="entraînement")
    ax_accuracy.plot(epochs_range, val_accuracy, label="validation")
    ax_accuracy.legend(loc="lower right")
    ax_accuracy.set_title("Justesse")
    ax_accuracy.set_xlabel("epoch")

    ax_loss.plot(epochs_range, loss, label="entraînement")
    ax_loss.plot(epochs_range, val_loss, label="validation")
    ax_loss.legend(loc="upper right")
    ax_loss.set_title("Perte")
    ax_loss.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Courbes sauvegardées dans {out_path}")


def predict(model, test_data_gen):
    """Prédit les 50 images de test, dans l'ordre du générateur.

    Aucun réordonnancement : `ANSWERS` est indexé sur l'ordre de production du
    générateur, pas sur l'ordre numérique des noms de fichiers. Les deux diffèrent,
    puisque `flow_from_directory` trie lexicographiquement (`10.jpg` avant
    `2.jpg`), mais le corrigé freeCodeCamp suit ce même tri.

    Vérification faite en regardant les images plutôt qu'en raisonnant sur les
    noms : la deuxième image produite par le générateur est `10.jpg`, qui montre
    un chat, et `ANSWERS[1]` vaut 0 (chat). Alors que `2.jpg` montre un chien. Le
    corrigé suit donc bien l'ordre lexicographique.

    Ce point mérite l'attention parce que l'intuition pousse à « corriger » un
    décalage qui n'existe pas : réordonner les probabilités par numéro de fichier
    ferait chuter le score à 56 %, sous le seuil, y compris pour un modèle parfait.

    `shuffle=False` sur le générateur reste indispensable : sans lui, l'ordre
    varierait à chaque exécution et aucune comparaison ne serait possible.

    Args:
        model (keras.Model): Modèle entraîné.
        test_data_gen: Générateur de test, obligatoirement `shuffle=False`.

    Returns:
        list[float]: Probabilité que chaque image soit un chien, dans l'ordre du
            générateur, directement comparable à `ANSWERS`.
    """
    return [float(p[0]) for p in model.predict(test_data_gen)]


def evaluate(probabilities):
    """Compare les prédictions au corrigé freeCodeCamp.

    Le seuil est de 63 %, soit 32 images sur 50. Une seule image vaut 2 points de
    pourcentage : à cette taille d'échantillon, deux entraînements successifs du
    même modèle peuvent donner des scores sensiblement différents. Un résultat
    tout juste au-dessus du seuil n'est donc pas un résultat fiable.

    Args:
        probabilities (list[float]): Probabilités « chien », dans l'ordre du
            générateur, directement comparable à `ANSWERS`.
    """
    correct = sum(
        1
        for probability, answer in zip(probabilities, ANSWERS, strict=True)
        if round(probability) == answer
    )
    percentage = correct / len(ANSWERS)

    print(
        f"Le modèle a correctement identifié {percentage:.2%} des images "
        f"({correct}/{len(ANSWERS)})."
    )

    if percentage >= 0.70:
        print("Défi réussi, avec le bonus des 70 %.")
    elif percentage > 0.63:
        print("Défi réussi.")
    else:
        print("Pas encore : il faut identifier au moins 63 % des images.")


def main():
    """Enchaîne le pipeline complet, du téléchargement à l'évaluation."""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    dataset_dir = download_dataset()

    train_dir = os.path.join(dataset_dir, "train")
    validation_dir = os.path.join(dataset_dir, "validation")
    test_dir = os.path.join(dataset_dir, "test")

    prepare_test_dir(test_dir)

    train_data_gen, val_data_gen, test_data_gen = build_generators(
        train_dir, validation_dir, test_dir, seed=RANDOM_SEED
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Deux figures produites AVANT l'entraînement : elles valident que les
    # générateurs fonctionnent, ce qui évite d'attendre 30 epochs pour découvrir
    # que les images arrivaient mal.
    plot_images(
        train_data_gen[0][0][:5],
        os.path.join(FIGURES_DIR, "01-echantillon-entrainement.png"),
        "Échantillon du jeu d'entraînement (déjà augmenté)",
    )
    save_augmentation_sample(
        train_dir, os.path.join(FIGURES_DIR, "02-augmentation.png")
    )

    model = build_model()
    model.summary()

    history = train_model(model, train_data_gen, val_data_gen)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "cat_dog_classifier.keras")
    model.save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    save_history_plots(history, os.path.join(FIGURES_DIR, "03-courbes.png"))

    probabilities = predict(model, test_data_gen)

    # Les 50 images de test avec leur prédiction : la figure qui montre *sur quoi*
    # le modèle se trompe, là où le score final ne donne qu'un chiffre.
    test_data_gen.reset()
    images_test = np.concatenate([test_data_gen[i] for i in range(len(test_data_gen))])
    plot_images(
        images_test,
        os.path.join(FIGURES_DIR, "04-predictions-test.png"),
        "Prédictions sur les 50 images de test (vert : correct, rouge : erreur)",
        probabilities=probabilities,
        answers=ANSWERS,
    )

    save_activation_maps(
        model,
        images_test[0],
        os.path.join(FIGURES_DIR, "05-cartes-activation.png"),
    )

    evaluate(probabilities)


if __name__ == "__main__":
    main()
