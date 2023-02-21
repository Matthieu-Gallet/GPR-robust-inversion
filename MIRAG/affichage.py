"""
Module for displaying the results of the ADMM in its sparse and separation form 
"""

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
import tikzplotlib

from .optim.admm_func import roll_fft


def cmap_perso(Q):
    r"""Creation of a divergent custom colormap centered on the mean
    of the Q matrix

    Parameters
    ----------
    Q :float
        Matrix to display (mainly sum of C_k) (Nx,Ny)


    Returns
    -------
    cmap :obj
        associated cmap object
    """
    normQ = (Q - Q.min()) / (Q.max() - Q.min())
    colors = ["blue", "lightsteelblue", "white", "lightsalmon", "red"]
    node = [0, normQ.mean() * 0.95, normQ.mean(), normQ.mean() * 1.05, 1]
    if normQ.mean() * 1.05 > 1:
        node = [0, normQ.mean() * 0.95, normQ.mean(), 0.9999, 1]
    cmap_pers = LinearSegmentedColormap.from_list("mycmap",
                                                  list(zip(node, colors)))
    return cmap_pers


def plot_ckmap(alpha, duo=False, t=60, x=128,
               title=["_", "_"], nfile="_", save=False):
    r"""Display one or 2 C_k maps from a personal divergent cmap

    Parameters
    ----------
    alpha :float
        Matrix (M x N)(mainly sum of C_k or detail of a C_k)
    duo :bool{False}, optional
        Display of one or 2 cards.
    t :int{60}, optional
        central position (pixel) of the hyperbolas used (ordinate)
    x :int{128}, optional
        central position (pixel) of the used hyperbolas (abscissa)
    title :list{["_","_"]}, optional
        Titles of the graphs
    nfile :str{"_"}, optional
        Name of the file for the record
    save :bool{False}, optional
        Save the file.

    Returns
    -------
    None
    """
    if duo:
        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        for i in range(2):
            Q = roll_fft(alpha[i], t, x)
            cmapPers = cmap_perso(Q)
            x = ax[0].imshow(Q, cmap=cmapPers, aspect="equal")
            fig.colorbar(x, ax=ax[i], shrink=0.5)
            ax[0].set_title(title[i])
    else:
        Q = roll_fft(alpha, t, x)
        cmapPers = cmap_perso(Q)
        fig, ax = plt.subplots(figsize=(12, 12))
        x = ax.imshow(Q, cmap=cmapPers, aspect="equal")
        fig.colorbar(x, ax=ax, shrink=0.5)
        ax.set_title(title[0])
    if save:
        tikzplotlib.save(nfile + ".tex")


def plot_ckmap_img(T, ck=False, title=["_", "_"],
                   nfile="_", save=False, t=60, x=128):
    r"""Displaying the sum of C_k and the dimensional image

    Parameters
    ----------
    T :float
        array [C_k , img] ([(Nx * Ny * K) , (Nx * Ny)]) or ([(Nx * Ny), (Nx * Ny)])
    ck :bool{False}, optional
        If the corrections/sum on the C_k have been done
    title :list{["_","_"]}, optional
        Titles of the graphs
    nfile :str{"_"}, optional
        Name of the file for the record
    save :bool{False}, optional
        Save the file.
    t :int{60}, optional
        central position (pixel) of the hyperbolas used (ordinate)
    x :int{128}, optional
        central position (pixel) of the hyperbolas used (abscissa)

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if ck:
        Q = roll_fft(T[0], t, x)
        cmapPers = cmap_perso(Q)
        x = ax[0].imshow(Q, cmap=cmapPers, aspect="equal")
    else:
        ax[0].imshow(np.real(T[0]), aspect="equal", cmap="gray")
    ax[0].set_title(title[0])
    ax[1].imshow(np.real(T[1]), aspect="equal", cmap="gray")
    ax[1].set_title(title[1])
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.suptitle(title[2], fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(nfile + ".png")


def plot_recon(Dal, Dico=None, name="_", save=False, compute=True):
    r"""Displays the reconstruction of an image using the dictionary and
    maps of the C_k

    Parameters
    ----------
    Dal :float
        Either the already computed reconstruction (Nx * Ny) or the C_k tensor (Nx * Ny * K).
    Dico :float{None}, optional
        dictionary necessary for the calculation of the reconstruction 
        (if not already computed) (Nx * Ny * K).
    name :str{"_"}, optional
        name of the file to save.
    save :bool{False}, optional
        save or not the image.
    compute :bool{True}, optional
        reconstruction already computed or not.

    Returns
    -------
    None
    """
    if compute:
        Dal1 = fft.ifft2(
            np.sum(fft.fft2(Dal, axes=(0, 1)) * fft.fft2(Dico, axes=(0, 1)), 2),
            axes=(0, 1),
            workers=-1,
        )
    else:
        Dal1 = Dal
    _, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(np.real(Dal1), aspect="equal", cmap="gray")
    if save:
        tikzplotlib.save(name + ".tex")
        # np.savez(name,Dal1)


def plot_atomNSM(atm2, paraDic):
    r"""Display of an atom from the physical dictionary
    with the right dimensions (m and ns)

    Parameters
    ----------
    atm2 :float
        matrix of the atom (Nx * Ny)
    paraDic :dic
        dictionary of the parameters of the atom (size ns\m)
        necessary key : "size_ns_m

    Returns
    -------
    None

    Examples
    --------
    >>> paraDic={}
    >>> paraDic["size_ns_m"]=[900,45]
    >>> plot_atomNSM(atoms,paraDic)
    """
    fig, ax = plt.subplots(1, figsize=(12, 3))
    m = ax.imshow(atm2, cmap="gray", aspect="equal")
    op = ax.get_yticks()
    op1 = ax.get_xticks()
    b = np.round(np.linspace(0, paraDic["size_ns_m"][0], len(op)))
    b1 = np.round(np.linspace(0, paraDic["size_ns_m"][1], len(op1)))
    a = np.hstack(("-1", b.astype("str")))
    ax.set_yticklabels(a)
    a = np.hstack(("-1", b1))
    ax.set_xticklabels(a.astype("str"))
    fig.colorbar(m, ax=ax)
    fig.show()

def scale_0_1(img):
    """ Scale an image between 0 and 1

    Parameters
    ----------
    img :  numpy.ndarray
        Image to scale

    Returns
    -------
    img : numpy.ndarray
        Scaled image
    """
    scaled = (img-img.min())/(img.max()-img.min())
    return scaled

def roc_curve_plot(mask,img,name):
    """ Plot ROC curve 

    Parameters
    ----------
    mask :  float
        mask of the image
    img :  array offloat
        array to be roc-curve plotted
    name :  str
        name of the plot
    """
    f = plt.figure(figsize=(9,7))
    colors = ['k', 'r', 'b', 'g', 'pink', 'orange']
    for i,ref_2 in enumerate(img):
        a = np.array(ref_2,dtype=np.float64)**2
        # b = np.where(mask>128,1,0)#mask/255.0
        b = mask
        b = b.ravel()
        a = a.ravel()
        auc_score = roc_auc_score(b, a)
        fpr, tpr, thresholds = roc_curve(b, a)
        plt.plot(fpr, tpr, c=colors[i], label=f'{name[i]} - ROC curve (area = %0.2f)' % auc_score)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    return f