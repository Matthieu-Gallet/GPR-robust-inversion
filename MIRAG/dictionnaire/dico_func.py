"""
Complementary module for creating the hyperbola dictionary for convolutional ADMM
for mathematical and physical approaches
"""

from tqdm import tqdm
import numpy as np


def filtre2D_B(Nx, Nt, x, t, coef=1):
    r"""Performs a 2D Hanning filter centered on the atomic position
    with a dilation factor coef

    Parameters
    ----------
    Nx :int
        horizontal dimension of the atom image
    Nt :int
        vertical dimension of the atom image
    x :int
        horizontal central coordinate of the atom (in pixels)
    t :int
        vertical central coordinate of the atom (in pixels)
    coef :int{1}, optional
        filter dilation coefficient (1 = filter for 256x256 window)


    Returns
    -------
    M : float
        Centered attenuation matrix (Nx*Ny)
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
    r"""functions to calculate the reflection coefficients at the nth interface

    Parameters
    ----------
    n :int
        number of layers
    param :float
        dictionaries of parameters given by parametre_MAXGs or parametre_4L
    polarization :str{"TE", "TM"}
        desired polarization : electric "TE" or magnetic "TM".


    Returns
    -------
    out : float
        reflection coefficients at the n_th interface

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
    r"""functions to calculate the transmission coefficients at the nth interface

    Parameters
    ----------
    n :int
        number of layers
    param :float
        dictionaries of parameters given by parametre_MAXGs or parametre_4L
    polarization :str{"TE", "TM"}
        desired polarization : electric "TE" or magnetic "TM".


    Returns
    -------
    out : float
        transmission coefficients at the n_th interface

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
    r"""functions for calculating phase shift (and, if losses, attenuation) terms

    Parameters
    ----------
    n :int
        number of layers
    param :float
        dictionaries of parameters given by parametre_MAXGs or parametre_4L


    Returns
    -------
    out : float
        phase shift at layer n

    See Also
    --------
    parametre_MAXGs,parametre_4L
    """
    return np.exp(
        -param["c_propag"][n + 1] * param["haut"][n + 1] * np.cos(param["theta"][n + 1])
    )


def Deph2(n, param):
    r"""functions for calculating phase shift (and, if losses, attenuation) terms

    Parameters
    ----------
    n :int
        number of layers
    param :float
        dictionaries of parameters given by parametre_MAXGs or parametre_4L


    Returns
    -------
    out : float
        phase shift at layer n

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
    r"""recursive function to determine the total reflection coefficient

    Parameters
    ----------
    x :int{1}
        number of the starting layer (1)
    param :float
        dictionaries of parameters given by parametre_MAXGs or parametre_4L
    polarization :str{"TE", "TM"}
        desired polarization : electric "TE" or magnetic "TM".

    Returns
    -------
    out : float
        array of total reflections
    
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
    r"""recursive function to determine the total transmission coefficient

    Parameters
    ----------
    x :int{1}
        number of the starting layer (1)
    param :float
        dictionaries of parameters given by parametre_MAXGs or parametre_4L
    polarization :str{"TE", "TM"}
        desired polarization : electric "TE" or magnetic "TM".

    Returns
    -------
    out : float
        array of total transmissions
    
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
    r"""analytical method of homogenization.

    Equivalent permittivity calculation function giving the same
    total reflection coefficient for several layers as a layer
    at the vacuum interface.

    Parameters
    ----------
    angle :float
        angle of incidence of the wave considered
    gamma_cp :float
        total reflection coefficient (2/4 layers)
    cste :float
        dictionary of constants:

        - pulsation (w)
        - empty permeability (mu0)
        - empty permittivity (e0)
    c_i :float
        dictionary of physical parameters of the first layer 
        permeability, permittivity and conductivity of the material (mui,ei,sigi) 
    polarization :str{"TE", "TM"}
        desired polarization: electric "TE" or magnetic "TM".

    Returns
    -------
    e_eq : float
        relative equivalent permittivity
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
    r"""Returns the complete dictionary of parameters needed to
    homogenization of the permittivity for a total reflection coefficient
    in the case of a 2 layer model (air-(matrix+inclusion))

    Parameters
    ----------
    frequence :float{350E6}, optional
        frequency of operation of the radar
    conductivity :float{0.0}, optional
        conductivity of the medium
    permitivity :list{[1.0, 1.0]}, optional
        material permittivity (relative)(matrix+target)
    thick_air :float{1.0}, optional
        thickness of the air layer (m)


    Returns
    -------
    cste :float
        dictionary of constants (frequency, wavelength, vacuum permittivity and pulsation)
    c_i :float
        REDUNDANT dictionary of parameters of the first layer
        (permeability, permittivity, conductivity)
    p :float
        complete dictionary of all parameters for each layer
        (permeability, permittivity, conductivity, angle)
    
    ... warning::
       **To do**

       Modify the function to avoid redundancy of the output variable `c_i`.
      

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
    r"""Returns the complete dictionary of parameters needed to
    homogenization of the permittivity for a total reflection coefficient
    in the case of a 4 layer model (air-matrix-target-matrix)

    Parameters
    ----------
    frequence :float{350E6}, optional
        frequency of operation of the radar
    conductivity :float{0.0}, optional
        conductivity of the medium
    permitivity :list{[1.0, 1.0]}, optional
        material permittivity (relative)(matrix+target)
    thick_air :float{1.0}, optional
        thickness of the air layer (m)
    thick_mat :list{[1.0, 1.0]}, optional
        material thickness (m)(matrix+target)


    Returns
    -------
    cste :float
        dictionary of constants (frequency, wavelength, vacuum permittivity and pulsation)
    c_i :float
        REDUNDANT dictionary of parameters of the first layer
        (permeability, permittivity, conductivity)
    p :float
        complete dictionary of all parameters for each layer
        (permeability, permittivity, conductivity, angle)
    
    ... warning::
       **To do**

       Modify the function to avoid redundancy of the output variable `c_i`.
      

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
    r"""Returns the equivalent permittivity homogenized
    by model 1 ("parametre_MAXGs") with 2 layers: (air-(matrix+inclusions)).
    Can be modified by using model 2 ("parametre_4L")
    which uses 4 layers: (air-matrix-target-matrix)

    Parameters
    ----------
    ta :float
        thickness of the air layer (m)
    tm :float
        effective permittivity of the medium (relative)
    freq :float
        operating frequency of the radar
    cond :float
        conductivity of the medium
    margin :float{0.1}, optional
        variation margin on the permittivity of the medium and the thickness of the air
        to obtain a solution.


    Returns
    -------
    out : float
        homogenized equivalent permittivity

    Notes
    -----
    This approach is interesting but remains subjective with notably the accepted margins and 
    the "while" loop which takes us away from the physical world (~tinkering) and makes the processing more cumbersome. 

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
    r"""Returns the effective permittivity of a material of type
    inclusion in a matrix.

    For more information : https://en.wikipedia.org/wiki/Effective_medium_approximations

    Parameters
    ----------
    m :float
        matrix permittivity vector
    n :float
        permittivity vector of inclusions (targets)
    deli :float
        volume fraction of inclusions

    Returns
    -------
    eff : float
        effective permittivity table
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
    r"""Calculates the speed of propagation of a wave from the permittivity
    permittivity, conductivity and frequency.

    Parameters
    ----------
    eps float
        electrical permittivity (can be complex) (:math:`F/m`)
    sig float
        electrical conductivity (can be complex) (:math:`\Omega \cdot m`)
    ome float
        electrical frequency (:math:`Hz`)

    Returns
    -------
    out : float
        propagation speed
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
