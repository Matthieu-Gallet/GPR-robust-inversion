"""
Module de filtrage des reconstructions par seuillage / abandon d'atomes du dictionnaire
"""

import numpy as np

def SVD_gpr(ref,rank):
    """ Make a SVD of the reference image and put to zero the first rank singular values.

    Parameters
    ----------
    ref :  float
        Reference image (Nx,Ny)
    rank :  int
        Rank of the SVD to set to zero.

    Returns
    -------
    ref_svd : float
        Reference image with the first rank singular values set to zero.
    """
    U, D, VT = np.linalg.svd(ref, full_matrices=False)
    D[:rank]=0
    A_remake = (U @ np.diag(D) @ VT)
    return A_remake

def dropR(C, Dic, p_acond=500, acond=0.1):
    r"""Supprime après ADMM les atomes non désirés pour la reconstruction

    .. warning::
        A utiliser spécifiquement pour l'ADMM 1 contrainte (ADMM 2 ou Source separation
        utilise la matrice creuse L)

    Parameters
    ----------
    C :float
        Cartes des coefficients à alléger (Nx,Ny,K) avec K le nombre d'atomes, 
        Nx,Ny les dimensions de la reconstruction.
    Dic :float
        Dictionnaire à alléger (Nx,Ny,K) avec K le nombre d'atomes,
        Nx,Ny les dimensions de la reconstruction.
    p_acond :int{500}, optional
        condition sur le paramètre p du dictionnaire.
        Garde tous les atomes tq :math:`a/p < p_{cond}`.
    acond :float{0.1}, optional
        condition sur le paramètre a du dictionnaire.
        Garde tous les atomes tq :math:`a > a_{cond}`.


    Returns
    -------
    Cprim :float
        Cartes des coefficients allégés (Nx,Ny,K') avec K'<K
    Hprim :float
        Dictionnaire allégé (Nx,Ny,K') avec K'<K
    """
    cond1 = Dic["param"][:, 0] / Dic["param"][:, 2] < p_acond
    cond2 = Dic["param"][:, 2] > acond
    l_red = np.where(cond1 & cond2)[0]

    Cprim = C[:, :, l_red]
    Him = Dic["atoms"][:, :, l_red]
    par = Dic["param"][l_red, :]

    Hprim = {"atoms": Him, "param": par}
    return Cprim, Hprim


def thresh_Ck(Cprim, seuil=0.45):
    r"""Seuillage des cartes C_k.
    Met à zéros les valeurs qui respecte les condtions suivantes pour un signal découpé 
    en histogramme à 1000 tranches:

    .. math:: 0.5\cdot\mathrm{seuil}*1000< \mathrm{signal}<1000*(1- 0.5\cdot\mathrm{seuil}) 

    Parameters
    ----------
    Cprim :float
        Tenseur C_k à seuiller (dynamique 0-1) (Nx,Ny,K)
    seuil :float{0.45}, optional
        seuil des valeurs (entre 0 et 1).

    Returns
    -------
    Cter :float
        Tenseur des cartes de coefficients seuillé (Nx,Ny,K)
    """
    Cter = np.zeros(Cprim.shape)
    for i in range(Cprim.shape[2]):
        Q = np.real(Cprim[:, :, i])
        _, bin = np.histogram(Q, 1000)
        bin_min = bin[int(seuil/2 * 1000)]
        bin_max = bin[int(1000 - (seuil/2 * 1000))]
        Cter[:, :, i] = np.where((Q < bin_min) | (Q > bin_max), Q, 0)
    return Cter


def submax_Ck(Csec, seuil=0.1):
    r"""Seuillage des cartes C_k.
    Ne garde que les valeurs qui respectent les conditions:

    .. math:: 0.5\cdot\mathrm{seuil}*1000< \mathrm{signal}<1000*(1- 0.5\cdot\mathrm{seuil}) 

    Utile pour faire ressortir les signaux faibles (principalement ADMM1).
    
    Parameters
    ----------
    Csec :float
        Tenseur C_k à seuiller (dynamique 0-1) (Nx,Ny,K)
    seuil :float{0.1}, optional
        seuil des valeurs (entre 0 et 1).

    Returns
    -------
    Cfin :float
        Tenseur des cartes de coefficients seuillé (Nx,Ny,K)
    """
    Cprim = Csec.copy()
    Cfin = np.zeros(Cprim.shape)
    for i in range(Cprim.shape[2]):
        Q = np.real(Cprim[:, :, i])
        _, bin = np.histogram(Q, 1000)
        bin_min = bin[int(seuil/2 * 1000)]
        bin_max = bin[int(1000 - (seuil/2 * 1000))]
        Cfin[:, :, i] = np.where((Q < bin_min) | (Q > bin_max), 0, Q)
    return Cfin