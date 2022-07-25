"""
Module for filtering reconstructions by thresholding / dropping atoms from the dictionary
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
    r"""Removes after ADMM the unwanted atoms for reconstruction

    .. warning::
        To be used specifically for the constrained ADMM 1 (ADMM 2 or Source separation
        uses the hollow matrix L)

    Parameters
    ----------
    C :float
        Maps of the coefficients to be reduced (Nx,Ny,K) with K the number of atoms, 
        Nx,Ny the dimensions of the reconstruction.
    Dic :float
        Dictionary to be reduced (Nx,Ny,K) with K the number of atoms,
        Nx,Ny the dimensions of the reconstruction.
    p_acond :int{500}, optional
        condition on the parameter p of the dictionary.
        Keep all the atoms tq :math:`a/p < p_{cond}`.
    acond :float{0.1}, optional
        condition on the parameter a of the dictionary.
        Keep all atoms tq :math:`a > a_{cond}`.


    Returns
    -------
    Cprim :float
        Maps of the reduced coefficients (Nx,Ny,K') with K'<K
    Hprim :float
        Lightweight dictionary (Nx,Ny,K') with K'<K
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
    r"""Thresholding of C_k cards.
    Sets to zero the values which respect the following conditions for a signal sliced 
    in histogram with 1000 slices:

    .. math:: 0.5\cdot\mathrm{threshold}*1000< \mathrm{signal}<1000*(1- 0.5\cdot\mathrm{threshold}) 

    Parameters
    ----------
    Cprim :float
        C_k tensor to threshold (dynamic 0-1) (Nx,Ny,K)
    threshold :float{0.45}, optional
        threshold of the values (between 0 and 1).

    Returns
    -------
    Cter :float
        Thresholded coefficient map tensor (Nx,Ny,K)
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
    r"""Thresholding of C_k cards.
    Keeps only the values that meet the conditions:

    .. math:: 0.5\cdot\mathrm{threshold}*1000< \mathrm{signal}<1000*(1- 0.5\cdot\mathrm{threshold}) 

    Useful to highlight weak signals (mainly ADMM1).
    
    Parameters
    ----------
    Csec :float
        C_k tensor to threshold (dynamic 0-1) (Nx,Ny,K)
    threshold :float{0.1}, optional
        threshold of the values (between 0 and 1).

    Returns
    -------
    Cfin :float
        Thresholded coefficient map tensor (Nx,Ny,K)
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