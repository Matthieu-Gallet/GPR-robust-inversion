"""
Module compémentaire de création du dictionnaire d'hyperbole pour l'``ADMM`` convolutionnel
pour les approches mathématiques et physiques
"""

from tqdm import tqdm
import numpy as np


def filtre2D_B(Nx, Nt, x, t, coef=1):
    r"""Réalise un filtre 2D de type hanning centré sur la position
    de l'atome avec un facteur de dilation coef

    Parameters
    ----------
    Nx :int
        dimension horizontale de l'image de l'atome
    Nt :int
        dimension verticale de l'image de l'atome
    x :int
        coordonnée centrale horizontale de l'atome (en pixels)
    t :int
        coordonnée centrale verticale de l'atome (en pixels)
    coef :int{1}, optional
        coeff de dilation du filtre (1 = filtre pour fenêtre 256x256)


    Returns
    -------
    M : float
        Matrice d'atténuation centrée (Nx*Ny)
    """
    M = np.zeros((Nt, Nx))
    rdimX = int(coef * 256)
    if rdimX % 2 != 0:
        rdimX = rdimX + 1
    Dx = np.hanning(rdimX).reshape(rdimX, 1)
    Dx = Dx @ Dx.T
    coor_win = np.array(
        [
            t - int(rdimX / 2),
            t + int(rdimX / 2),
            x - int(rdimX / 2),
            x + int(rdimX / 2),
        ],
        dtype=int,
    )
    if coor_win[0] < 0:
        Dx = Dx[np.abs(coor_win[0]) :, :]
        coor_win[0] = 0
    if coor_win[1] > M.shape[0]:
        Dx = Dx[: M.shape[0] - coor_win[1], :]
        coor_win[1] = M.shape[0]
    if coor_win[2] < 0:
        Dx = Dx[:, np.abs(coor_win[2]) :]
        coor_win[2] = 0
    if coor_win[3] > M.shape[1]:
        Dx = Dx[:, : M.shape[1] - coor_win[3]]
        coor_win[3] = M.shape[1]
    M[coor_win[0] : coor_win[1], coor_win[2] : coor_win[3]] = Dx
    return M


# Fonctions dictionnaire physique
def ref(n, param, polarisation):
    r"""fonctions pour calculer les coefficients de reflexion à la n_ième interface

    Parameters
    ----------
    n :int
        nombre de couche
    param :float
        dictionnaires des paramètres donné par parametre_MAXGs ou parametre_4L
    polarisation :str{"TE","TM"}
        polarisation désirée : electrique "TE" ou magnetique "TM".


    Returns
    -------
    out : float
        coefficients reflexion à la n_ième interface

    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    if polarisation == "TM":
        return (
            -param["eta"][n] * np.cos(param["theta"][n])
            + param["eta"][n + 1] * np.cos(param["theta"][n + 1])
        ) / (
            param["eta"][n] * np.cos(param["theta"][n])
            + param["eta"][n + 1] * np.cos(param["theta"][n + 1])
        )
    elif polarisation == "TE":
        return (
            param["eta"][n + 1] * np.cos(param["theta"][n])
            - param["eta"][n] * np.cos(param["theta"][n + 1])
        ) / (
            param["eta"][n + 1] * np.cos(param["theta"][n])
            + param["eta"][n] * np.cos(param["theta"][n + 1])
        )
    else:
        print("erreur de polarisation")


def tra(n, param, polarisation):
    r"""fonctions pour calculer les coefficients transmission à la n_ième interface

    Parameters
    ----------
    n :int
        nombre de couche
    param :float
        dictionnaires des paramètres donné par parametre_MAXGs ou parametre_4L
    polarisation :str{"TE","TM"}
        polarisation désirée : electrique "TE" ou magnetique "TM".


    Returns
    -------
    out : float
        coefficients transmission à la n_ième interface

    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    if polarisation == "TM":
        return (2 * param["eta"][n + 1] * np.cos(param["theta"][n])) / (
            param["eta"][n] * np.cos(param["theta"][n])
            + param["eta"][n + 1] * np.cos(param["theta"][n + 1])
        )
    elif polarisation == "TE":
        return (2 * param["eta"][n + 1] * np.cos(param["theta"][n])) / (
            param["eta"][n + 1] * np.cos(param["theta"][n])
            + param["eta"][n] * np.cos(param["theta"][n + 1])
        )
    else:
        print("erreur de polarisation")


def Deph1(n, param):
    r"""fonctions pour calculer les termes de déphasage (et, si pertes, d'atténuation)

    Parameters
    ----------
    n :int
        nombre de couche
    param :float
        dictionnaires des paramètres donné par parametre_MAXGs ou parametre_4L


    Returns
    -------
    out : float
        déphasage à la couche n

    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    return np.exp(
        -param["c_propag"][n + 1] * param["haut"][n + 1] * np.cos(param["theta"][n + 1])
    )


def Deph2(n, param):
    r"""fonctions pour calculer les termes de déphasage (et, si pertes, d'atténuation)

    Parameters
    ----------
    n :int
        nombre de couche
    param :float
        dictionnaires des paramètres donné par parametre_MAXGs ou parametre_4L


    Returns
    -------
    out : float
        déphasage à la couche n

    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    return np.exp(
        -2
        * param["c_propag"][n + 1]
        * param["haut"][n + 1]
        * np.cos(param["theta"][n + 1])
    )


def ref_tot(x, param, polarisation):
    r"""fonction récursive pour déterminer le coefficient de reflexion total

    Parameters
    ----------
    x :int{1}
        numero de la couche de démarrage (1)
    param :float
        dictionnaires des paramètres donné par parametre_MAXGs ou parametre_4L
    polarisation :str{"TE","TM"}
        polarisation désirée : electrique "TE" ou magnetique "TM".

    Returns
    -------
    out : float
        array des réflexions totales
    
    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    if x == param["nb_c"]:
        return ref(x, param, polarisation)
    else:
        return (
            ref(x, param, polarisation)
            + ref_tot(x + 1, param, polarisation) * Deph2(x, param)
        ) / (
            1
            + ref(x, param, polarisation)
            * ref_tot(x + 1, param, polarisation)
            * Deph2(x, param)
        )


def tran_tot(x, param, polarisation):
    r"""fonction récursive pour déterminer le coefficient de transmission total

    Parameters
    ----------
    x :int{1}
        numero de la couche de démarrage (1)
    param :float
        dictionnaires des paramètres donné par parametre_MAXGs ou parametre_4L
    polarisation :str{"TE","TM"}
        polarisation désirée : electrique "TE" ou magnetique "TM".

    Returns
    -------
    out : float
        array des transmissions totales
    
    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    if x == param["nb_c"]:
        return tra(x, param, polarisation)
    else:
        return (
            tra(x, param, polarisation)
            * (tran_tot(x + 1, param, polarisation))
            * Deph1(x, param)
        ) / (
            1
            + ref(x, param, polarisation)
            * ref_tot(x + 1, param, polarisation)
            * Deph2(x, param)
        )


def homogeneisation_an(angle, gamma_cp, cste, c_i, polarisation="TE"):
    r"""méthode analytique de l'homogénéisation.

    Fonction de calcul de la permittivité équivalente donnant le même
    cofficient de réflexion total pour plusieurs couches qu'une couche
    à l'interface du vide.

    Parameters
    ----------
    angle :float
        angle d'incidence de l'onde considérée
    gamma_cp :float
        coefficient de réflexion total(2/4 couches)
    cste :float
        dictionnaire des constantes:

        - pulsation (w)
        - perméabilité vide (mu0)
        - permittivité vide (e0)
    c_i :float
        dictionnaire des paramètres physiques de la première couche 
        perméabilité,permittivité et conductivité du matériaux (mui,ei,sigi) 
    polarisation :str{"TE","TM"}
        polarisation désirée : electrique "TE" ou magnetique "TM".

    Returns
    -------
    e_eq : float
        permittivité équivalente relative
    """
    y_propi = np.sqrt(
        1j * cste["w"] * c_i["mui"] * c_i["sigi"]
        - c_i["mui"] * c_i["ei"] * cste["w"] ** 2
    )

    etai = 1j * cste["w"] * c_i["mui"] / y_propi
    M = (y_propi * np.sin(angle) / cste["w"]) ** 2

    if polarisation == "TE":
        K = (np.cos(angle) * (1 - gamma_cp) / (1 + gamma_cp)) ** 2
        e_eq = (K * cste["mu0"] / (etai ** 2) - (M / cste["mu0"])) / cste["e0"]

    elif polarisation == "TM":
        K = (np.cos(angle) * (1 + gamma_cp) / (1 - gamma_cp)) ** 2
        A = K * etai ** 2 / cste["mu0"]
        B = -1
        C = -(M / cste["mu0"])
        DELTA = B ** 2 - 4 * A * C
        e_eq1 = (-B + np.sqrt(DELTA)) / (2 * A)
        e_eq2 = (-B - np.sqrt(DELTA)) / (2 * A)
        e_eq = np.array([e_eq1, e_eq2]) / cste["e0"]

    else:
        print("Erreur polarisation")

    return e_eq


def parametre_MAXGs(frequence=350e6, conductivity=0, permitivity=1, thick_air=1):
    r"""Retourne le dictionnaire complet des paramètres nécessaire à
    l'homogénisation de la permittivité pour un coefficient de réflexion totale
    dans le cas d'un modèle à 2 couches (air-(matrice+inclusion))

    Parameters
    ----------
    frequence :float{350E6}, optional
        fréquence de fonctionnement du radar
    conductivity :float{0.0}, optional
        conductivité du milieu
    permitivity :list{[1.0, 1.0]}, optional
        permittivités des matériaux (relatives)(matrice+cible)
    thick_air :float{1.0}, optional
        épaisseur de la couche d'air (m)


    Returns
    -------
    cste :float
        dictionnaire des constantes (fréquence, longeur d'onde, permittivité du vide et pulsation)
    c_i :float
        REDONDANT dictionnaire des paramètres de la première couche
        (permeabilite,permittivité,conductivité)
    p :float
        dictionnaire complet de tous les paramètres pour chaque couche
        (permeabilite,permittivité,conductivité,angle)
    
    .. warning::
       **A faire**

       Modifier la fonction pour éviter la redondance de la variable de sortie `c_i`
      

    See Also
    --------
    parametre_4L
    """
    f = frequence

    cste = {}
    cste["mu0"] = 4e-7 * np.pi
    cste["e0"] = 8.85e-12
    cste["c"] = 1 / np.sqrt(cste["e0"] * cste["mu0"])
    cste["w"] = 2 * np.pi * f
    cste["lbda"] = cste["c"] / f

    haut_air = thick_air / cste["lbda"]
    p = {}
    p["nb_c"] = 2
    p["haut"] = cste["lbda"] * np.array([0, haut_air, 1e10, 1e10])
    p["permit"] = cste["e0"] * np.array([1, 1, permitivity, permitivity])
    p["permeab"] = cste["mu0"] * np.array([1, 1, 1, 1])
    p["conduct"] = np.array([0, 0, conductivity, 0])
    ### impédances
    p["eta"] = np.sqrt(
        1j * cste["w"] * p["permeab"] / (1j * cste["w"] * p["permit"] + p["conduct"])
    )
    ### indices de réfraction
    p["ind"] = np.sqrt(p["permit"] * p["permeab"] / (cste["e0"] * cste["mu0"]))
    ### constantes de propagation
    p["c_propag"] = np.sqrt(
        1j * cste["w"] * p["permeab"] * p["conduct"]
        - p["permeab"] * p["permit"] * cste["w"] ** 2
    )
    ### angles d'incidence de l'onde sur les différentes couches
    # theta_incident = np.array([0,22.5*np.pi/180,45*np.pi/180])
    theta_incident = np.array([0])
    p["theta"] = np.zeros([p["nb_c"] + 2, len(theta_incident)], dtype="cfloat")
    p["theta"][0] = theta_incident
    for n in range(0, p["nb_c"] + 1):
        p["theta"][n + 1] = np.arcsin(
            np.sin(p["theta"][n]) * p["ind"][n] / p["ind"][n + 1]
        )

    ##### caractéristiques du milieu incident
    c_i = {}
    c_i["mui"] = p["permeab"][0]
    c_i["ei"] = p["permit"][0]
    c_i["sigi"] = p["conduct"][0]

    return cste, c_i, p


def parametre_4L(
    frequence=350e6,
    conductivity=0.0,
    permitivity=[1.0, 1.0],
    thick_air=1.0,
    thick_mat=[1.0, 1.0],
):
    r"""Retourne le dictionnaire complet des paramètres nécessaire à
    l'homogénisation de la permittivité pour un coefficient de réflexion totale
    dans le cas d'un modèle à 4 couches (air-matrice-cible-matrice)

    Parameters
    ----------
    frequence :float{350E6}, optional
        fréquence de fonctionnement du radar
    conductivity :float{0.0}, optional
        conductivité du milieu
    permitivity :list{[1.0, 1.0]}, optional
        permittivités des matériaux (relatives)(matrice+cible)
    thick_air :float{1.0}, optional
        épaisseur de la couche d'air (m)
    thick_mat :list{[1.0, 1.0]}, optional
        épaisseur des matériaux (m)(matrice+cible)


    Returns
    -------
    cste :float
        dictionnaire des constantes (fréquence, longeur d'onde, permittivité du vide et pulsation)
    c_i :float
        REDONDANT dictionnaire des paramètres de la première couche
        (permeabilite,permittivité,conductivité)
    p :float
        dictionnaire complet de tous les paramètres pour chaque couche
        (permeabilite,permittivité,conductivité,angle)
    
    .. warning::
       **A faire**

       Modifier la fonction pour éviter la redondance de la variable de sortie `c_i`
      

    See Also
    --------
    parametre_MAXGs
    """
    f = frequence

    cste = c_i = p = {}
    cste["mu0"] = 4e-7 * np.pi
    cste["e0"] = 8.85e-12
    cste["c"] = 1 / np.sqrt(cste["e0"] * cste["mu0"])
    cste["w"] = 2 * np.pi * f
    cste["lbda"] = cste["c"] / f

    haut_air = thick_air / cste["lbda"]
    haut_matrice = thick_mat[0] * f * np.sqrt(permitivity[0]) / cste["c"]
    haut_cible = thick_mat[1] * f * np.sqrt(permitivity[1]) / cste["c"]
    # print(haut_air,haut_cible,haut_matrice)
    p["nb_c"] = 3
    p["haut"] = cste["lbda"] * np.array([0, haut_air, haut_matrice, haut_cible, 1e10])
    p["permit"] = cste["e0"] * np.array(
        [1, 1, permitivity[0], permitivity[1], permitivity[0]]
    )
    p["permeab"] = cste["mu0"] * np.array([1, 1, 1, 1, 1])
    p["conduct"] = np.array([0, 0, conductivity, 0, 0])
    ### impédances
    p["eta"] = np.sqrt(
        1j * cste["w"] * p["permeab"] / (1j * cste["w"] * p["permit"] + p["conduct"])
    )
    ### indices de réfraction
    p["ind"] = np.sqrt(p["permit"] * p["permeab"] / (cste["e0"] * cste["mu0"]))
    ### constantes de propagation
    p["c_propag"] = np.sqrt(
        1j * cste["w"] * p["permeab"] * p["conduct"]
        - p["permeab"] * p["permit"] * cste["w"] ** 2
    )
    ### angles d'incidence de l'onde sur les différentes couches
    # theta_incident = np.arange(0,45*np.pi/180,(1*np.pi/180))
    # theta_incident = np.array([0,22.5*np.pi/180,45*np.pi/180])
    theta_incident = np.array([0])
    p["theta"] = np.zeros([p["nb_c"] + 2, len(theta_incident)], dtype="cfloat")
    p["theta"][0] = theta_incident
    for n in range(0, p["nb_c"] + 1):
        p["theta"][n + 1] = np.arcsin(
            np.sin(p["theta"][n]) * p["ind"][n] / p["ind"][n + 1]
        )

    ##### caractéristiques du milieu incident
    c_i["mui"] = p["permeab"][0]
    c_i["ei"] = p["permit"][0]
    c_i["sigi"] = p["conduct"][0]

    return cste, c_i, p


def atompos2C(ta, tm, freq, cond, marge=0.1):
    r"""Retourne la permittivité équivalente homogénéisée
    par le modèle 1 ("parametre_MAXGs") à 2 couches: (air-(matrice+inclusions)).
    Peut être en modifié en utilisant le modèle 2 ("parametre_4L")
    qui utilise 4 couches: (air-matrice-cible-matrice)

    Parameters
    ----------
    ta :float
        épaisseur de la couche d'air (m)
    tm :float
        permittivité effective du milieu (relative)
    freq :float
        fréquence de fonctionnement du radar
    cond :float
        conductivité du milieu
    marge :float{0.1}, optional
        marge de variation sur la permittivité du milieu et l'épaisseur de l'air
        pour obtenir une solution.


    Returns
    -------
    out : float
        permittivité équivalente homogénéisée

    Notes
    -----
    Cette approche est intéressante mais reste subjective avec notamment les marges acceptées et 
    la boucle ``while`` qui nous éloigne un peu du monde physique (~bricolage) et alourdisse le traitement. 

    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    t = np.array([])
    for i in range(int(1e2)):
        co = 0
        eq = -1
        while eq < 0:
            ta_rd = np.random.rand() * (2 * marge * ta) + (ta - marge * ta)
            tm_rd = np.random.rand() * (2 * marge * tm) + (tm - marge * tm)
            cste, ci, param = parametre_MAXGs(
                frequence=freq, conductivity=cond, permitivity=tm_rd, thick_air=ta_rd
            )
            param["R_mlt"] = ref_tot(0, param, polarisation="TE")
            eq = homogeneisation_an(
                param["theta"][0], param["R_mlt"], cste, ci, polarisation="TE"
            )
            if co > 25:
                eq = -1
                break
            co = co + 1
        t = np.append(t, eq)
    return np.mean(t)


def maxwell_garnett(m, n, deli):
    r"""Retourne la permittivité effective d'un matériau de type
    inclusion dans une matrice.

    Pour plus d'informations : https://en.wikipedia.org/wiki/Effective_medium_approximations

    Parameters
    ----------
    m :float
        vecteur de permittivité de matrice
    n :float
        vecteur de permittivité d'inclusions (cibles)
    deli :float
        fraction volumique des inclusions

    Returns
    -------
    eff : float
        tableau de permittivité effective
    """
    eff = []
    for e_m in m:
        for e_i in n:
            num = (2 * deli * (e_i - e_m)) + e_i + (2 * e_m)
            denum = (2 * e_m) + e_i - (deli * (e_i - e_m))
            e_eff = e_m * num / denum
            eff.append(e_eff)
    return eff


def v_prop(eps, sig, ome):
    r"""Calcule la vitesse de propagation d'une onde à partir de
    la permittivité, conductivité et fréquence.

    Parameters
    ----------
    eps float
        permittivité électrique (peut être complexe) (:math:`F/m`)
    sig float
        conductivité électrique (peut être complexe) (:math:`\Omega \cdot m`)
    ome float
        fréquence électrique (:math:`Hz`)

    Returns
    -------
    out : float
        vitesse de propagation
    """
    celer = 3e8
    mu = 1
    e_eff = np.real(eps) - (np.imag(sig) / ome)
    s_eff = np.real(sig) + (np.imag(eps) * ome)
    eps = np.real(e_eff)
    sig = np.real(s_eff)
    # print(p["eps"],p["sig"])
    denum1 = np.sqrt(1 + (sig / ome) ** 2) + 1
    denumF = np.sqrt(0.5 * mu * eps * denum1)
    return celer / denumF


def eps2vprop(par_vp, margeR=1e-4, diff=0.075):
    r"""Retourne la vitesse de propagation fictive de matériaux
    à permittivité équivalente à partir des paramètres physiques
    et du modèle d'homogénéisation

    Parameters
    ----------
    par_vp :float
        dictionnaire de paramètres (fréquence,permittivité effective du milieu,conductivité, épaisseur de l'air)
    margeR :float{1e-4}, optional
        marge appliqué aux paramètres du dico pour trouver les permittivtés
    diff :float{0.075}, optional
        critère de différence entre chaque vitesse calculée (%).

    Returns
    -------
    vprM : float
        array de vitesse de propagation

    Notes
    -----
    Cette approche est intéressante mais reste subjective avec notamment les marges acceptées et le
    critère de différence qui nous éloigne un peu du monde physique (~bricolage) 
    """
    freq = par_vp["freq"]
    cond = par_vp["cond"]
    thick_air = par_vp["thick_air"]
    perm_eff = par_vp["perm_eff"]
    eq_tot = np.array([])
    # k = 0
    pbar = tqdm(total=len(perm_eff), leave=True)
    for ef in perm_eff:
        eq = atompos2C(thick_air, ef, freq, cond, marge=margeR)
        eq_tot = np.append(eq_tot, eq)
        # k = k + 1
        pbar.update(1)
    pbar.close()
    eq_tot_eq = eq_tot[np.where(np.real(eq_tot) > 0)]
    vprM = np.sort(v_prop(eq_tot_eq, cond, freq))
    S = vprM[0]
    for u in range(1, len(vprM)):
        if (vprM[u] < (1 + diff) * S) | (vprM[u] > 1e9):
            vprM[u] = -1
        else:
            S = vprM[u]
    vprM = vprM[np.where(vprM > 0)]
    return vprM
