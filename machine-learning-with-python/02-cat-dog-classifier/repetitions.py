# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tensorflow>=2.16",
#     "numpy>=1.26",
#     "matplotlib>=3.10",
#     # Requis par les transformations affines d'ImageDataGenerator : même raison
#     # que dans classifier.py, où le détail est commenté.
#     "scipy>=1.13",
# ]
# ///

"""Répète l'entraînement du classifieur sur plusieurs seeds.

Répond à la question que `classifier.py` documente mais ne règle pas : les
chiffres du projet reposent sur un seul run. Un score de 82 % sur 50 images,
c'est 41 bonnes réponses : 3 images d'écart valent 6 points, et `evaluate`
avertit lui-même que deux entraînements du même modèle peuvent donner des
résultats sensiblement différents.

Ce script relance le même pipeline (même architecture, mêmes générateurs, même
nombre d'epochs) sur plusieurs seeds et rapporte moyenne et écart-type sur :

    * l'écart de justesse train - validation à la dernière epoch : la
      conclusion « pas de surapprentissage » du README tient-elle pour un seul
      run, ou est-elle robuste ?
    * le score sur les 50 images de test : le 82 % est-il représentatif ?

Chaque seed fixe à la fois l'initiale des poids (`tf.random.set_seed`) et
l'ordre des lots plus l'augmentation (seed posé sur les générateurs via
`build_generators`). Un seed donné est donc entièrement reproductible.

Le run historique (seed 42) n'est pas relancé par défaut : ses chiffres figurent
dans le README et servent de référence documentée. Le script lance par défaut
les seeds 7 et 2025, qui encadrent la variabilité autour de cette référence.

Lancement :
    uv run repetitions.py [SEED ...]
    uv run repetitions.py 42 --epochs 1   # sonde de coût, 1 epoch par seed
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from classifier import (
    ANSWERS,
    BATCH_SIZE,
    EPOCHS,
    FIGURES_DIR,
    build_generators,
    build_model,
    download_dataset,
    predict,
    prepare_test_dir,
)

DEFAULT_SEEDS = (7, 2025)


def run_seed(seed, epochs):
    """Entraîne le pipeline complet sur un seed et relève les métriques.

    Chaque seed construit ses générateurs à neuf (donc un ordre des lots et une
    augmentation qui lui sont propres), initialise les poids et entraîne
    exactement comme `classifier.py`, avec le même nombre d'epochs.

    Args:
        seed (int): Graine de l'initiale des poids, du mélange et de
            l'augmentation.
        epochs (int): Nombre d'epochs, à ajuster pour la sonde de coût.

    Returns:
        dict: Métriques du run : `seed`, justesse train et validation à la
            dernière epoch, écart train - validation, nombre de bonnes réponses
            sur les 50 images de test, et courbes de justesse par epoch.
    """
    print(f"\n=== Seed {seed} ===")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset_dir = download_dataset()
    train_dir = os.path.join(dataset_dir, "train")
    validation_dir = os.path.join(dataset_dir, "validation")
    test_dir = os.path.join(dataset_dir, "test")
    prepare_test_dir(test_dir)

    train_data_gen, val_data_gen, test_data_gen = build_generators(
        train_dir, validation_dir, test_dir, seed=seed
    )

    # Même calcul que `train_model` de classifier.py : arrondi supérieur pour ne
    # pas laisser de côté le dernier lot incomplet.
    steps_per_epoch = math.ceil(train_data_gen.samples / BATCH_SIZE)
    validation_steps = math.ceil(val_data_gen.samples / BATCH_SIZE)

    model = build_model()
    history = model.fit(
        train_data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=validation_steps,
        verbose=2,
    )

    probabilities = predict(model, test_data_gen)
    correct = sum(
        1
        for probability, answer in zip(probabilities, ANSWERS, strict=True)
        if round(probability) == answer
    )

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    return {
        "seed": seed,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": train_acc - val_acc,
        "test_correct": correct,
        "train_history": history.history["accuracy"],
        "val_history": history.history["val_accuracy"],
    }


def report(results):
    """Affiche le tableau par seed puis les agrégats moyenne ± écart-type.

    L'écart-type de la moyenne d'échantillon se calcule avec `ddof=1` dès que
    plusieurs runs existent, sinon la division se ferait par zéro.

    Args:
        results (list[dict]): Sorties de `run_seed`.
    """
    print("\n=== Résultats ===")
    print(f"{'seed':>4} {'train':>7} {'val':>7} {'écart':>7} {'test':>7}")
    for result in results:
        print(
            f"{result['seed']:>4} {result['train_acc']:.3f} {result['val_acc']:.3f} "
            f"{result['gap']:.3f} {result['test_correct']:>3}/50"
        )

    n = len(results)
    ddof = 1 if n > 1 else 0
    gaps = np.array([result["gap"] for result in results])
    scores = np.array([result["test_correct"] for result in results])

    print()
    print(
        f"Écart train - validation : {gaps.mean():.3f} ± {gaps.std(ddof=ddof):.3f} "
        f"(moyenne ± écart-type, n={n})"
    )
    print(
        f"Score de test            : {scores.mean() / 50:.1%} ± "
        f"{scores.std(ddof=ddof) / 50:.1%} "
        f"({scores.mean():.1f}/50 ± {scores.std(ddof=ddof):.1f} images)"
    )

    if n > 1:
        spread = scores.max() - scores.min()
        print(
            f"Étendue des scores de test : {spread} image(s) sur 50, soit "
            f"{spread * 2} points de pourcentage."
        )

    if scores.min() > 0.63 * 50:
        print("Même le run le moins bon dépasse le seuil des 63 %.")


def plot_val_curves(results, out_path):
    """Trace la justesse de validation par seed, avec la bande moyenne ± écart.

    Complète les courbes de la figure 03 : là où `save_history_plots` montre un
    run unique, celle-ci montre si la trajectoire est reproductible ou si un run
    isolé était trompeur.

    Args:
        results (list[dict]): Sorties de `run_seed`.
        out_path (str): Chemin du PNG à écrire.
    """
    epochs_range = np.arange(1, len(results[0]["val_history"]) + 1)
    val_curves = np.array([result["val_history"] for result in results])
    mean = val_curves.mean(axis=0)
    ddof = 1 if len(results) > 1 else 0
    std = val_curves.std(axis=0, ddof=ddof)

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(
            epochs_range,
            result["val_history"],
            label=f"seed {result['seed']}",
            alpha=0.7,
        )
    ax.fill_between(
        epochs_range, mean - std, mean + std, alpha=0.2, label="moyenne ± écart"
    )
    ax.plot(epochs_range, mean, "k--", linewidth=1, label="moyenne")
    ax.set_xlabel("epoch")
    ax.set_ylabel("justesse de validation")
    ax.set_title("Justesse de validation par seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure sauvegardée dans {out_path}")


def parse_args(argv):
    """Analyse les arguments : seeds à lancer, et nombre d'epochs (sonde)."""
    parser = argparse.ArgumentParser(
        description="Relance le pipeline du classifieur chats/chiens sur plusieurs seeds."
    )
    parser.add_argument(
        "seeds",
        nargs="*",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds à lancer (défaut : 7 2025).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Nombre d'epochs par run (défaut : {EPOCHS}).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = [run_seed(seed, args.epochs) for seed in args.seeds]
    report(results)
    plot_val_curves(results, os.path.join(FIGURES_DIR, "06-variance-seeds.png"))


if __name__ == "__main__":
    main()
