# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib==3.11.1",
# ]
# ///

# Notebook d'exploration. Ne contient aucune logique de stratégie ni de métrique :
# tout vient de strategies.py, metrics.py et harness.py (convention du projet 01). Ici j'orchestre
# et je visualise, rien d'autre.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from harness import BOT_NAMES, compare, play_match
    from metrics import calibration_bins
    from strategies import STRATEGIES

    return (
        BOT_NAMES,
        STRATEGIES,
        calibration_bins,
        compare,
        mo,
        play_match,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # Pierre-Papier-Ciseaux : battre quatre bots à 60 %

    Un joueur aléatoire gagne 50 % des parties. Le projet demande **60 % contre
    chacun des quatre bots**, sur 1000 coups. Dix points d'écart, quatre fois de
    suite.

    Ce notebook raconte comment j'y suis arrivé, et surtout **pourquoi ma première
    idée était la bonne pour trois bots et la mauvaise pour le quatrième**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Poser un plancher avant d'optimiser

    Ma première stratégie est la plus bête possible : compter les coups de
    l'adversaire, sans tenir compte de leur ordre. Une chaîne de Markov d'ordre 0,
    autrement dit une simple loi catégorielle, estimée par moyenne a posteriori
    d'un Dirichlet(1,1,1).

    C'est exactement le lissage de Laplace (add-one) : les deux formulations sont
    la même, et le prior m'évite la division par zéro au premier coup, quand je
    n'ai encore rien observé. Le détail est dans
    un document de cadrage tenu hors de ce dépôt.

    Je lui adjoins un témoin, `uniform`, qui ne croit rien du tout. Ce n'est pas
    du zèle : sans plancher, je n'ai aucune échelle pour juger la suite.
    """)
    return


@app.cell
def _(compare):
    results_floor = compare(
        strategy_names=("uniform", "markov_order_0"),
        games=1000,
    )
    return (results_floor,)


@app.cell
def _(mo, results_floor):
    mo.md(
        "```\n"
        + "\n".join(
            f"{r.strategy:<16} {r.bot:<9} {r.win_rate:>5.1f}%" for r in results_floor
        )
        + "\n```"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    Un détail dans cette table explique déjà presque tout, à condition de ne pas
    lire trop vite. `uniform` ne croit rien, mais `best_response` doit quand même
    trancher : sur une distribution plate, `max` départage par l'ordre de `MOVES`
    et renvoie toujours le même coup. Le témoin joue donc **P en permanence**.

    Contre ce joueur parfaitement prévisible, les quatre bots ne se comportent pas
    de la même façon, et c'est là qu'est l'information :

    - **abbey, kris, mrugesh** : 0 %, 0 %, 0,2 %. Un adversaire prévisible ne perd
      pas la moitié du temps, il perd *tout le temps*. Ces trois-là ne jouent pas
      dans leur coin, **ils réagissent à ce que je joue** ;
    - **quincy** : 66,7 %, au-dessus du seuil de 60 %. Un coup fixe le bat. C'est
      la signature inverse : quincy **ne réagit pas**, il déroule sa séquence quoi
      qu'il arrive. Contre lui, être prévisible ne coûte rien.

    Le calcul se refait à la main. Sa table est `R, R, P, P, S`, mais son compteur
    est incrémenté avant lecture : la séquence réellement observée est donc
    `R, P, P, S, R`, la même à un décalage de phase près. Sur une période, P fixe
    gagne deux fois (les deux R), fait nul deux fois (les deux P), perd une fois
    (le S), soit 2/3 des parties décisives.

    Deux régimes, donc, séparés par une seule mesure et avant toute lecture de
    code : réactif contre non réactif. C'est le premier acte du diagnostic, et
    l'acte 3 montrera que la vraie ligne de partage est encore ailleurs.

    Deuxième surprise, dans l'autre sens : `markov_order_0`, censé être plus malin
    que le témoin, plafonne à 50 % contre quincy là où le coup fixe fait 66,7 %.

    L'explication ne tient pas à un temps de convergence, mais à quelque chose de
    plus instructif. Sur le cycle de quincy, les fréquences globales valent R 40 %,
    P 40 %, S 20 % : les deux premières sont *à égalité*. Le comptage ne départage
    donc jamais vraiment, et la prédiction bascule de R à P au gré du dernier coup
    observé. Le témoin, lui, ne bascule pas.

    Le résultat se lit dans le détail des issues : sur 1000 coups, **200 victoires,
    200 défaites et 600 égalités**. L'ordre 0 finit par osciller en phase avec sa
    cible, et jouer le même coup qu'elle produit un nul. Il ne perd pas plus qu'il
    ne gagne, il annule presque tout.

    Un modèle plus riche n'est donc pas automatiquement meilleur, et la raison
    n'est pas qu'il apprend lentement : c'est qu'il regarde la mauvaise chose. Une
    périodicité n'est pas une fréquence. Compter les coups d'une séquence cyclique
    détruit exactement l'information qui la caractérise, son ordre.

    ## 2. Ajouter de la mémoire

    Je conditionne maintenant la prédiction sur les *k* derniers coups adverses.
    Un Dirichlet par contexte : 3 contextes à l'ordre 1, 9 à l'ordre 2, 27 à
    l'ordre 3.

    Mon cadrage laissait la question de l'ordre ouverte. Je la tranche par la
    mesure.
    """)
    return


@app.cell
def _(compare):
    results_orders = compare(
        strategy_names=(
            "markov_order_0",
            "markov_order_1",
            "markov_order_2",
            "markov_order_3",
        ),
        games=1000,
    )
    return (results_orders,)


@app.cell
def _(BOT_NAMES, plt, results_orders):
    fig_orders, axes_orders = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for _ax, _bot in zip(axes_orders, BOT_NAMES, strict=True):
        _rows = [r for r in results_orders if r.bot == _bot]
        _orders = [int(r.strategy.split("_")[-1]) for r in _rows]
        _rates = [r.win_rate for r in _rows]

        # Les limites Y sont fixées AVANT de peindre la zone d'échec : avec
        # sharey=True, un axhspan posé après peut ne pas être rendu sur certains
        # panneaux.
        _ax.set_ylim(15, 112)
        _ax.axhspan(15, 60, color="#c53030", alpha=0.08, zorder=0)
        _ax.axhline(60, color="#c53030", linestyle="--", linewidth=1, zorder=1)
        _ax.plot(_orders, _rates, color="#2b6cb0", linewidth=1.5, zorder=2)
        _ax.scatter(
            _orders,
            _rates,
            s=55,
            zorder=3,
            color=["#2f855a" if v >= 60 else "#c53030" for v in _rates],
        )
        for _o, _v in zip(_orders, _rates, strict=True):
            # Une décimale : 59,7 arrondi à « 60 » se lirait comme un succès.
            _ax.annotate(
                f"{_v:.1f}",
                (_o, _v),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                fontweight="bold" if _v < 60 else "normal",
                color="#c53030" if _v < 60 else "#4a5568",
            )

        _echecs = sum(1 for v in _rates if v < 60)
        _ax.set_title(
            f"{_bot}" + ("  (plafonne sous 60 %)" if _echecs == len(_rates) else ""),
            fontweight="bold" if _echecs == len(_rates) else "normal",
        )
        _ax.set_xlabel("ordre de la chaîne")
        _ax.set_xticks(_orders)
    axes_orders[0].set_ylabel("taux de victoire (%)")
    fig_orders.suptitle(
        "Taux de victoire selon l'ordre de mémoire (zone rouge : sous le seuil des 60 %)"
    )
    fig_orders.tight_layout()
    fig_orders
    return


@app.cell
def _(mo):
    mo.md("""
    Trois bots tombent : quincy s'effondre à 99,9 % dès l'ordre 2, kris atteint
    91,4 % à l'ordre 3, mrugesh se stabilise autour de 84 %.

    Précision sur ce dernier : son taux n'est pas exactement reproductible d'un
    processus à l'autre (environ 0,8 point de variation), parce que son code
    départage les égalités via l'ordre d'itération d'un `set`, qui dépend de
    `PYTHONHASHSEED`. Les trois autres bots sont déterministes.

    Le quatrième panneau est celui qui compte. **Abbey progresse énormément** :
    de 25,1 % à 59,7 %, soit +34,6 points, la plus forte progression des quatre.
    La mémoire lui fait beaucoup d'effet.

    Et pourtant elle s'arrête **0,3 point sous le seuil**, et l'ordre 3 ne fait pas
    mieux. Ce n'est pas un mur : c'est un plafond, qui tombe au pire endroit
    possible.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Vérifier que ce n'est pas juste un manque de mémoire

    Avant de conclure, l'hypothèse simple mérite d'être testée : peut-être qu'un
    ordre plus élevé suffirait.

    | Ordre | 1 | 2 | 3 | 4 | 5 |
    |---|---|---|---|---|---|
    | abbey | 49,9 % | 59,7 % | 59,2 % | 60,4 % | 57,4 % |

    Non. Ça oscille autour de 59 %, sans tendance. L'ordre 4 passe de justesse
    (60,4 %) mais l'ordre 5 redescend : c'est du bruit, pas un gain. Le plafond ne
    vient pas du manque de mémoire.

    ### Le corrigé : j'ouvre le code des bots

    Jusqu'ici je m'étais interdit de le lire, pour que le modèle soit construit sur
    l'observation seule. La mesure est faite : je peux confronter.

    ```python
    def kris(prev_opponent_play):            # aucun état
        return ideal_response[prev_opponent_play]

    def mrugesh(prev_opponent_play, opponent_history=[]):
        last_ten = opponent_history[-10:]    # MON historique, pas le sien
        return ideal_response[max(set(last_ten), key=last_ten.count)]

    def abbey(prev_opponent_play, opponent_history=[], play_order=[{...}]):
        last_two = "".join(opponent_history[-2:])
        play_order[0][last_two] += 1         # un ÉTAT qui s'accumule
    ```

    Première surprise : **kris et mrugesh réagissent eux aussi à mes coups**. Ma
    typologie mentale (« bots à séquence fixe » contre « bots réactifs ») était
    fausse. Trois bots sur quatre sont réactifs.

    Alors pourquoi Markov les bat-il si bien ? Parce que leur réponse est une
    **fonction déterministe** de mon jeu : leur séquence encode déjà la mienne.
    Modéliser l'une revient à modéliser l'autre, en une étape au lieu de deux.

    Abbey fait autre chose. Il ne réagit pas au dernier coup, il **accumule un
    modèle de mes paires de coups** et l'affine tout au long de la partie. Cette
    mémoire casse l'équivalence : sa réponse n'est plus une fonction simple de mon
    dernier coup.

    Le vrai axe n'est donc pas « fixe contre réactif », mais **réactif sans mémoire
    contre réactif avec état**. Et contre le second, prédire ses coups directement
    est une impasse : ce qu'il joue n'est pas une propriété de lui, c'est une
    fonction de ce qu'il croit que je vais jouer.

    Je modélisais le mauvais objet.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Changer d'objet : me modéliser moi

    Si abbey prédit ma séquence pour jouer le contre, alors la prédire moi-même me
    dit ce qu'il va jouer. Je modélise donc **mon propre historique**, celui qu'il
    observe, et j'en déduis sa réponse.

    C'est le premier niveau méta d'Iocaine Powder : prédire non pas l'adversaire,
    mais l'adversaire en train de me prédire.
    """)
    return


@app.cell
def _(compare):
    results_self = compare(
        strategy_names=("markov_order_3", "self_model_1"), games=1000
    )
    return (results_self,)


@app.cell
def _(BOT_NAMES, plt, results_self):
    fig_self, ax_self = plt.subplots(figsize=(9, 3.4))
    width_self = 0.38
    for _offset, (_name, _label, _color) in enumerate(
        [
            ("markov_order_3", "markov ordre 3", "#2b6cb0"),
            ("self_model_1", "self_model ordre 1", "#d69e2e"),
        ]
    ):
        _rows = [r for r in results_self if r.strategy == _name]
        _positions = [i + _offset * width_self for i in range(len(BOT_NAMES))]
        _values = [r.win_rate for r in _rows]
        ax_self.bar(_positions, _values, width_self, label=_label, color=_color)
        for _p, _v in zip(_positions, _values, strict=True):
            ax_self.annotate(
                f"{_v:.0f}",
                (_p, _v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
                color="#4a5568",
            )
    ax_self.axhline(60, color="#c53030", linestyle="--", linewidth=1)
    ax_self.set_xticks([i + width_self / 2 for i in range(len(BOT_NAMES))])
    ax_self.set_xticklabels(BOT_NAMES)
    ax_self.set_ylabel("taux de victoire (%)")
    ax_self.set_ylim(0, 112)
    ax_self.set_title("Deux stratégies, deux domaines de compétence disjoints")
    ax_self.legend(loc="lower right", fontsize=8)
    fig_self.tight_layout()
    fig_self
    return


@app.cell
def _(mo):
    mo.md("""
    Le résultat est net et symétrique. `self_model_1` fait **83,3 % contre
    abbey**, le seul bot que Markov ne passait pas. Et il retombe à ~50 % partout
    ailleurs, là où Markov excelle.

    Deux stratégies, deux domaines de compétence **disjoints**. Aucune ne suffit
    seule.

    ## 5. Ne pas choisir : laisser la mesure choisir

    Puisque le bon modèle dépend de l'adversaire, je ne le fixe pas à l'avance. Je
    fais tourner quatre experts en parallèle (Markov ordres 2 et 3, `self_model_1`
    et `self_model_2`), je les note au **score de Brier** sur une fenêtre glissante de
    60 coups, et je joue celui qui prédit le mieux en ce moment.

    La fenêtre fait office de facteur d'oubli : si l'adversaire change de
    comportement, la sélection suit.

    C'est ici que le contrat retenu paie. Une stratégie renvoie une
    **distribution**, jamais un coup unique. Sans cela je ne pourrais pas noter mes
    experts : sur une prédiction ponctuelle, le score de Brier dégénère en simple
    taux d'erreur et toute l'information de confiance disparaît.
    """)
    return


@app.cell
def _(compare):
    results_final = compare(strategy_names=("mixture",), games=1000)
    return (results_final,)


@app.cell
def _(BOT_NAMES, plt, results_final):
    fig_final, ax_final = plt.subplots(figsize=(9, 3.4))
    rates_final = [r.win_rate for r in results_final]
    _intervals = [r.win_rate_interval for r in results_final]
    _errors = [
        [rate - low for rate, (low, _) in zip(rates_final, _intervals, strict=True)],
        [high - rate for rate, (_, high) in zip(rates_final, _intervals, strict=True)],
    ]
    bars_final = ax_final.bar(
        BOT_NAMES,
        rates_final,
        color=["#2f855a" if rate >= 60 else "#c53030" for rate in rates_final],
        yerr=_errors,
        capsize=5,
    )
    ax_final.axhline(60, color="#c53030", linestyle="--", linewidth=1)
    ax_final.set_ylabel("taux de victoire (%)")
    ax_final.set_ylim(0, 112)
    ax_final.set_title("Stratégie retenue, barres d'erreur : IC de Wilson à 95 %")
    ax_final.bar_label(bars_final, fmt="%.1f%%", padding=8)
    fig_final.tight_layout()
    fig_final
    return


@app.cell
def _(mo):
    mo.md("""
    Les quatre barres passent, et aucun intervalle de confiance ne touche la ligne.

    ### Ce que la mixture fait réellement

    En observant quel expert elle sélectionne à chaque coup, on voit qu'elle ne se
    comporte pas de la même façon selon l'adversaire :

    | bot | expert dominant | alternance |
    |---|---|---|
    | quincy | markov ordre 2 (99 %) | aucune |
    | kris | markov ordre 3 (~82 %) | rare |
    | mrugesh | markov ordre 3 (~81 %) | rare |
    | **abbey** | **markov ordre 3 (50 %)** | **`self_model_1` 31 %** |

    Contre trois bots elle reste sur Markov et ne fait qu'ajouter du bruit. Contre
    abbey, elle **alterne réellement** entre les deux familles. C'est là, et là
    seulement, qu'elle sert à quelque chose.

    ### Ce que ce choix me coûte

    Il serait malhonnête de m'arrêter à « quatre barres vertes ». Comparée au
    meilleur expert pris seul :

    | bot | markov ordre 3 | mixture | écart |
    |---|---|---|---|
    | quincy | 99,9 % | 99,9 % | 0 |
    | abbey | 59,2 % | **67,1 %** | **+7,9** |
    | kris | **91,4 %** | 75,7 % | **−15,7** |
    | mrugesh | ~84 % | ~84 % | légèrement négatif |

    Sur kris, je perds près de 16 points. Sur mrugesh, l'écart est trop petit
    pour être chiffré (ce bot n'est pas déterministe), mais son signe est stable
    sur cinq processus : la mixture reste toujours en dessous. **En moyenne,
    Markov ordre 3 est meilleur que ma stratégie finale**, et la mixture ne gagne
    sur aucun des trois bots que Markov battait déjà.

    Mais le critère du projet n'est pas la moyenne, c'est le **minimum** : il faut
    passer 60 % quatre fois. Markov s'arrête à 59,2 % contre abbey. J'échange donc
    de la performance là où j'étais confortable contre le point qui me manquait là
    où je bloquais.

    C'est un arbitrage assumé, pas une supériorité. Une sélection plus prudente,
    qui ne quitterait Markov que lorsqu'il faiblit vraiment, garderait sans doute
    les deux : c'est le prochain chantier.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Regarder ce que la stratégie croit, pas seulement ce qu'elle gagne

    Le taux de victoire ne dit pas *pourquoi* je gagne. Deux diagnostics que seule
    une sortie probabiliste rend possibles.

    D'abord la **calibration** : quand le modèle annonce 70 %, l'événement
    arrive-t-il 70 % du temps ? Un modèle calibré suit la diagonale.
    """)
    return


@app.cell
def _(STRATEGIES, play_match):
    match_abbey = play_match(
        STRATEGIES["mixture"], "abbey", games=1000, strategy_name="mixture"
    )
    match_quincy = play_match(
        STRATEGIES["mixture"], "quincy", games=1000, strategy_name="mixture"
    )
    return match_abbey, match_quincy


@app.cell
def _(calibration_bins, match_abbey, match_quincy, plt):
    fig_calib, ax_calib = plt.subplots(figsize=(5.2, 5))
    ax_calib.plot([0, 1], [0, 1], "--", color="#a0aec0", label="calibration parfaite")
    for _match, _label, _color in [
        (match_quincy, "vs quincy", "#2b6cb0"),
        (match_abbey, "vs abbey", "#d69e2e"),
    ]:
        _bins = calibration_bins(_match.predictions, _match.outcomes)
        ax_calib.scatter(
            [confidence for confidence, _, _ in _bins],
            [observed for _, observed, _ in _bins],
            s=[max(15, count / 8) for _, _, count in _bins],
            alpha=0.75,
            label=_label,
            color=_color,
        )
    ax_calib.set_xlabel("probabilité annoncée")
    ax_calib.set_ylabel("fréquence observée")
    ax_calib.set_title("Calibration (taille = nombre de prédictions)")
    ax_calib.legend()
    fig_calib.tight_layout()
    fig_calib
    return


@app.cell
def _(match_abbey, match_quincy, mo):
    mo.md(
        "| Match | Brier ↓ | Log-loss ↓ | Accuracy ↑ |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {m.strategy} vs {m.bot} | {m.score('brier'):.4f} "
            f"| {m.score('log_loss'):.4f} | {m.score('accuracy'):.3f} |"
            for m in (match_quincy, match_abbey)
        )
    )
    return


@app.cell
def _(mo):
    mo.md("""
    L'écart entre les deux matchs est parlant. Contre quincy, le modèle est quasi
    certain et il a raison. Contre abbey, il reste beaucoup plus prudent, et cette
    prudence est justifiée : il affronte un adversaire qui s'adapte.

    Détail à ne pas gommer : contre abbey, je **gagne 67 % en prédisant mal**
    (Brier élevé). Battre un adversaire ne veut pas dire l'avoir compris.

    ## 7. Explorer

    Le banc est ouvert : n'importe quelle stratégie du registre, n'importe quel
    bot, n'importe quelle règle de décision.
    """)
    return


@app.cell
def _(BOT_NAMES, STRATEGIES, mo):
    picker_strategy = mo.ui.dropdown(
        options=list(STRATEGIES), value="mixture", label="Stratégie"
    )
    picker_bot = mo.ui.dropdown(options=list(BOT_NAMES), value="abbey", label="Bot")
    picker_decision = mo.ui.dropdown(
        options=["best_response", "sample"], value="best_response", label="Décision"
    )
    mo.hstack([picker_strategy, picker_bot, picker_decision])
    return picker_bot, picker_decision, picker_strategy


@app.cell
def _(STRATEGIES, picker_bot, picker_decision, picker_strategy, play_match):
    explored = play_match(
        STRATEGIES[picker_strategy.value],
        picker_bot.value,
        games=1000,
        decision=picker_decision.value,
        strategy_name=picker_strategy.value,
    )
    return (explored,)


@app.cell
def _(explored, mo):
    mo.md(
        f"""
        **{explored.strategy}** vs **{explored.bot}** ({explored.decision})

        | | |
        |---|---|
        | Taux de victoire | **{explored.win_rate:.1f} %** {"seuil atteint" if explored.win_rate >= 60 else "sous le seuil"} |
        | IC de Wilson 95 % | [{explored.win_rate_interval[0]:.1f} %, {explored.win_rate_interval[1]:.1f} %] |
        | Victoires / défaites / nulles | {explored.wins} / {explored.losses} / {explored.ties} |
        | Brier | {explored.score("brier"):.4f} |
        | Log-loss | {explored.score("log_loss"):.4f} |
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ce que ce projet m'a appris

    Ma question de départ était « quel ordre de chaîne de Markov ? ». C'était la
    bonne question pour trois bots sur quatre, et la mauvaise pour le dernier.
    Contre un adversaire qui accumule un modèle de moi, **aucune profondeur de
    mémoire ne suffit** : le problème n'est pas la quantité d'information passée,
    c'est l'objet modélisé.

    Trois décisions se sont révélées payantes :

    - **Le plancher d'abord.** Commencer par l'ordre 0 et un témoin uniforme n'a
      rien fait gagner, mais m'a donné l'échelle et validé le banc de mesure avant
      toute optimisation. C'est aussi ce plancher qui a livré l'indice décisif :
      `uniform` joue un coup fixe, et le contraste de ses résultats sépare déjà
      les deux régimes (0 % contre les trois bots réactifs, 66,7 % contre quincy
      qui ne réagit pas).
    - **La sortie probabiliste** (contrat de sortie probabiliste). Sans distribution, pas de score de
      Brier exploitable, donc pas de sélection d'experts, donc pas de solution.
    - **Lire le code des bots en dernier.** Gardé comme corrigé, il a servi à
      valider le mécanisme après la mesure. Il a d'ailleurs invalidé ma typologie
      sans invalider mes résultats : une hypothèse fausse peut mener à des
      conclusions justes, et c'est précisément ce qu'un corrigé permet de voir.

    ### Le piège où je suis tombé

    Pendant un temps, ce notebook affichait abbey à 78 % à l'ordre 2, quand le test
    officiel donnait 59,7 %. Deux mesures du même objet, incompatibles.

    La cause était dans mon banc d'essai : je croyais réinitialiser l'état interne
    des bots entre deux matchs, mais je **copiais** leurs historiques accumulés au
    lieu de les vider. Abbey arrivait au match suivant avec la mémoire du
    précédent, ce qui le rendait artificiellement facile à battre.

    C'est exactement le piège que `implementation-notes.md` documente et que
    le cadrage initial avait identifié : je l'ai reproduit en croyant m'en prémunir. La
    leçon vaut plus que le bug : **quand deux mesures du même objet divergent, ne
    pas choisir celle qui arrange.** L'écart était le signal.

    ---

    Le détail des raisonnements est tenu hors de ce dépôt, le contrat imposé dans
    `docs/freecodecamp/`.
    """)
    return


if __name__ == "__main__":
    app.run()
