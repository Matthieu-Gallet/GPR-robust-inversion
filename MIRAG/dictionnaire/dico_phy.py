'''
Module de création du dictionnaire d'hyperbole pour l'``ADMM`` convolutionnel par approche physique
'''

import numpy as np
from .dico_func import *

def dico_PHY2(Dimxt, pos, v, dim_n_m, A=None, sigma=5):
    r"""Une fonction pour générer un atome ou une hyperbole afin
    de créer un dictionnaire pour la reconstruction GPR, et ce à partir
    de la modèlisation physique du problème.

    Parameters
    ----------
    Dimxt :list{[256,256]}
        dimension de l'hyperbole (Nx * Ny)
    pos :list{[60,128]}
        position centrée de l'hyperbole
    v :float
        vitesse de propagation  pour l'hyperbole
    dim_n_m :list{[900,40]}
        dimensions réelles de l'image (y = [ns] et x= [m])
    A :int{None}, optional
        matrice d'atténuation  (Nx * Ny)
    sigma :int{5}, optional
        paramètre épaisseur de l'hyperbole.


    Returns
    -------
    h : float
        atome construit de dimension (Nx*Ny)
    """
    # Setting mesh for x and t axis
    Nt = Dimxt[1]
    Nx = Dimxt[0]
    x, t = np.meshgrid(range(1, Nx + 1), range(1, Nt + 1))
    t_0 = pos[1] * dim_n_m[0] / Nt
    x_0 = pos[0] * dim_n_m[1] / Nx
    R = 0
    # R = R*dim_n_m[1]/Nx
    ## NT
    t = t * dim_n_m[0] / Nt
    ## NX
    x = x * dim_n_m[1] / Nx

    v = 1e-9 * v
    P = 2 / v
    # g = P * np.sqrt(((t_0/P)+R)**2 + (x-x_0)**2)-P*R
    g = P * np.sqrt((t_0 / P) ** 2 + (x - x_0) ** 2)

    # Defining ricker wavelet from equation (3)
    r = (
        2
        / (np.sqrt(3 * sigma) * np.pi ** (0.25))
        * (1 - ((t - g) / sigma) ** 2)
        * np.exp(-0.5 * ((t - g) / sigma) ** 2)
    )
    # b, a = butter(2, 0.0025)
    # r = filtfilt(b, a, r)
    r = r * Nx / dim_n_m[0]
    # Finally, we construct the atom with equation (2)
    h = A * r
    return h


def create_dicoPHY2(paraDic):
    r"""Fonction supérieure de crétion dictionnaire d'hyperboles ou d'atomes
    par modélisation physique

    Parameters
    ----------
    paraDic :dic
        dictionnaire des paramètres de création

    Notes
    -----
    - paraDic['dim'] (array) : Dimension of the radargramme
    - paraDic['std'] (array) : Input vector of sigma parameter
    - paraDic['position'] (array) : Position of the top of the hyperbola
    - paraDic['v_prop'] (array) : Input vector of velocity
    - paraDic['coef'] (array) : 2D filter coefficient for the size of the hyperbola
    - paraDic['size_ns_m'] (array) : value of corresponding measure in time and meter for the radar.

    Returns
    -------
    DioH : dic
        structure de la forme {"atoms": Dico, "param": param} où ``Dico`` est le tenseur
        des hyperboles (Nx*Ny*K) et ``param`` le tenseur des caractéristiques correspondants
        pour chaque hyperbole (K*3) (vitesse, coeff atténuation, sigma)
    """
    pos = paraDic["position"]
    vpr = paraDic["v_prop"]
    sig = paraDic["std"]
    dimPHY = paraDic["size_ns_m"]
    Dico = np.zeros(
        (
            paraDic["dim"][1],
            paraDic["dim"][0],
            len(vpr) * len(paraDic["coef"] * len(sig)),
        )
    )
    param = np.zeros((len(vpr) * len(paraDic["coef"] * len(sig)), 3), dtype="complex")
    co = 0
    for v in vpr:
        for s_a in paraDic["coef"]:
            for st in sig:
                A = filtre2D_B(
                    paraDic["dim"][0], paraDic["dim"][1], pos[0], pos[1], coef=s_a
                )
                at = dico_PHY2(paraDic["dim"], pos, v, dimPHY, A=A, sigma=st)
                at = at / np.linalg.norm(at, "fro")
                Dico[:, :, co] = at
                param[co, :] = np.array([v, s_a, st])
                co = co + 1
        DioH = {"atoms": Dico, "param": param}
    return DioH


def param_loadPHY(size, opt):
    r"""Charge les paramètres et
    retourne un dictionnaire d'atomes de dimension "size"

    Parameters
    ----------
    size :list{[256,256]}
        dimension de l'image originale, donc des atomes.
    opt :dic
        dictionnaire d'options de creation d'hyperboles

    Returns
    -------
    atm :dic
        structure de la forme {"atoms": Dico, "param": param} où ``Dico`` est le tenseur
        des hyperboles (Nx*Ny*K) et ``param`` le tenseur des caractéristiques correspondants
        pour chaque hyperbole (K*3)
    opt :dic
        dictionnaire d'entrée augmenté des valeurs utilisées par la fonctions
        (pour le suivit des modifications)


    .. warning::
       **ATTENTION** 
       
       Modification des paramètres de création du dico directement dans le fichier
       python !! 

       **A faire**

       - Ajout de l'option lecture des paramètres par un fichier YAML
    """
    # Ajout yaml reader
    Nx = size[1]
    Nt = size[0]

    paraDic = {}
    paraDic["dim"] = [Nx, Nt]
    paraDic["position"] = [int(Nx / 2), int(Nt / 4)]
    paraDic["size_ns_m"] = [96, 46]

    paraDic["dimF"] = opt["dimRAw"]
    paraDic["size_ns_mB"] = [96, 46]
    paraDic["dim"] = [Nx, Nt]
    paraDic["position"] = [int(paraDic["dim"][0] / 2), int(paraDic["dim"][1] / 4)]

    paraDic["size_ns_m"] = [
        paraDic["dim"][1] * paraDic["size_ns_mB"][0] / paraDic["dimF"][1],
        paraDic["dim"][0] * paraDic["size_ns_mB"][1] / paraDic["dimF"][0],
    ]

    paraDic["freq"] = 350e6
    paraDic["cond"] = 0
    paraDic["std"] = [0.5]
    paraDic["coef"] = [0.75]
    paraDic["thick_air"] = 0.75
    paraDic["perm_eff"] = np.linspace(5, 100, 101)
    paraDic["v_prop"] = eps2vprop(paraDic, margeR=0.5, diff=0.1)

    for i in paraDic.keys():
        if i == "perm_eff":
            opt["dico_param_" + i] = paraDic[i].tolist()
        elif i == "v_prop":
            opt["dico_param_" + i] = paraDic[i].tolist()
        else:
            opt["dico_param_" + i] = paraDic[i]

    atm = create_dicoPHY2(paraDic)

    return atm, opt
