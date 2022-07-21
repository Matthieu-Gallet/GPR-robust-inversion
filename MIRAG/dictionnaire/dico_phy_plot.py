"""
Module d'affichage des résultats liés au dictionnaire créé de manière physique
"""

import numpy as np
import matplotlib.pyplot as plt


def affichage_RT(gammat, ttot, thetai, polarisation, typec):
    r"""affichage des coefficients de réflexion et transmission :
    partie réelle et partie imaginaire

    Parameters
    ----------
    gammat :float
        tableau des transmissions totales
    ttot :float
        tableau des réflexions totales
    thetai :float
        tableau des angles étudiés
    polarisation :string{"TE","TM"}
        polarisation electrique "TE" ou magnétique "TM"
    typec :string
        Texte supplémentaire pour le titre

    Returns
    -------
    None
    """
    # plt.close('all')
    f1, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 6))
    ax1.plot((thetai * 180 / np.pi).real, gammat.real, label="Re($\Gamma_{tot}$)")
    ax1.plot((thetai * 180 / np.pi).real, gammat.imag, "--", label="Im($\Gamma_{tot}$)")
    ax1.legend(loc="upper right")
    ax1.set_title("Réflexion totale", fontsize="small")
    ax1.grid()

    ax2.plot((thetai * 180 / np.pi).real, ttot.real, label="Re($T_{tot}$)")
    ax2.plot((thetai * 180 / np.pi).real, ttot.imag, "--", label="Im($T_{tot}$)")
    ax2.legend(loc="upper right")
    ax2.set_title("Transmission totale", fontsize="small")
    ax2.grid()
    plt.suptitle(
        "Graphes des coefficients totaux de réfléxion et transmission en fonction de l'angle d'incidence\n pour le mode "
        + str(polarisation)
        + " en mode "
        + typec
    )
    plt.show()


def affichage_RT2(gammat, ttot, thetai, polarisation):
    r"""affichage des coefficients de réflexion et transmission:
    module et argument

    Parameters
    ----------
    gammat :float
        tableau des transmissions totales
    ttot :float
        tableau des réflexions totales
    thetai :float
        tableau des angles étudiés
    polarisation :string{"TE","TM"}
        polarisation electrique "TE" ou magnétique "TM"


    Returns
    -------
    None
    """
    # plt.close('all')
    f1, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    color1 = "tab:blue"
    color2 = "tab:orange"
    lns1 = ax1.plot(
        (thetai * 180 / np.pi).real,
        np.abs(gammat),
        label="mod($\Gamma_{tot}$)",
        color=color1,
    )
    ax1.set_title("Réflexion totale")
    ax2 = ax1.twinx()
    lns2 = ax2.plot(
        (thetai * 180 / np.pi).real,
        np.angle(gammat, deg=True),
        "--",
        label="arg($\Gamma_{tot}$)",
        color=color2,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.grid()

    lns3 = ax3.plot(
        (thetai * 180 / np.pi).real, np.abs(ttot), label="mod($T_{tot}$)", color=color1
    )
    ax3.set_title("Transmission totale")
    ax4 = ax3.twinx()
    lns4 = ax4.plot(
        (thetai * 180 / np.pi).real,
        np.angle(ttot, deg=True),
        "--",
        label="arg($T_{tot}$)",
        color=color2,
    )
    ax3.tick_params(axis="y", labelcolor=color1)
    ax4.tick_params(axis="y", labelcolor=color2)
    ax3.grid()
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    lns0 = lns3 + lns4
    labs0 = [l.get_label() for l in lns0]
    ax3.set_xlabel("angle incidence (°)")
    ax1.set_ylabel("mod.", color=color1)
    ax2.set_ylabel("arg. (°)", color=color2)
    ax3.set_ylabel("mod.", color=color1)
    ax4.set_ylabel("arg. (°)", color=color2)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(-180, 180)
    ax3.set_ylim(0, 1)
    ax4.set_ylim(-180, 180)

    ax1.legend(lns, labs, framealpha=1)
    ax3.legend(lns0, labs0, framealpha=1)
    plt.show()


def affichage_sum_mod2(gammat, ttot, thetai):
    r"""affichage de la somme des carrés des modules de Gamma et T

    Parameters
    ----------
    gammat :float
        tableau des transmissions totales
    ttot :float
        tableau des réflexions totales
    thetai :float
        tableau des angles étudiés

    Returns
    -------
    None
    """
    sum_mod2 = np.abs(gammat) ** 2 + np.abs(ttot) ** 2
    fig, ax = plt.subplots(1, figsize=(9, 3))
    ax.plot(
        (thetai * 180 / np.pi).real,
        sum_mod2,
        label="mod($\Gamma_{tot}$)²+mod($T_{tot}$)²",
    )
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid()
    plt.title(
        "Somme du carré des modules de $\Gamma_{tot}$ et de $T_{tot}$ \n (traduit la conservation de l'énergie)"
    )
    plt.show()


def affichage_Eeq_an(eps2, thetai, polarisation):
    r"""fonction d'affichage permittivités équivalentes homogénéisées en
    fonction de l'angle d'incidence pour la version analytique
    de l'homogeneisation

    Parameters
    ----------
    eps2 :float
        tableau des permittivités équivalentes homogénéisées
    thetai :float
        tableau des angles étudiés
    polarisation :string{"TE","TM"}
        polarisation electrique "TE" ou magnétique "TM"

    Returns
    -------
    None
    """
    # plt.close('all')
    if polarisation == "TE":
        f2, (ax3) = plt.subplots(1, figsize=(9, 6))
        ax3.plot(
            (thetai * 180 / np.pi).real, eps2.real, label="Re($\hat{\epsilon_{eq}}$)"
        )
        ax3.plot(
            (thetai * 180 / np.pi).real,
            eps2.imag,
            "--",
            label="Im($\hat{\epsilon_{eq}}$)",
        )
        ax3.legend(loc="upper right")
        ax3.grid()
        # ax3.set_ylim(0,1)
        plt.suptitle(
            "Graphe de la permittivité effective équivalente multicouche estimée \n pour le mode TE en fonction de l'angle d'incidence (résolution analytique)"
        )
        plt.xlim(0, 90)
        plt.show()
    else:
        f2, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(9, 12))
        ax3.plot(
            (thetai * 180 / np.pi).real,
            eps2[0].real,
            label="Re($\hat{\epsilon_{eq1}}$)",
        )
        ax3.plot(
            (thetai * 180 / np.pi).real,
            eps2[0].imag,
            "--",
            label="Im($\hat{\epsilon_{eq1}}$)",
        )
        ax3.legend(loc="upper right")
        ax3.grid()
        # ax3.set_ylim(0,1)

        ax4.plot(
            (thetai * 180 / np.pi).real,
            eps2[1].real,
            label="Re($\hat{\epsilon_{eq2}}$)",
        )
        ax4.plot(
            (thetai * 180 / np.pi).real,
            eps2[1].imag,
            "--",
            label="Im($\hat{\epsilon_{eq2}}$)",
        )
        ax4.legend(loc="upper right")
        ax4.grid()
        # ax4.set_ylim(0,1)

        plt.suptitle(
            "Graphe de la permittivité effective équivalente multicouche estimée \n pour le mode TM en fonction de l'angle d'incidence (résolution analytique) \n deux racines distinctes $\epsilon_{eq1}$ en haut et $\epsilon_{eq2}$ en bas"
        )
        plt.xlim(0, 90)
        plt.show()
