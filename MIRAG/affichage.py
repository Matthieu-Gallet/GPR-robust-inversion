"""
Module d'affichage des résultats de l'``ADMM`` sous sa forme sparse et separation 
"""

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
import tikzplotlib

from .optim.admm_func import roll_fft


def cmap_perso(Q):
    r"""Création d'une colormap perso divergente centrée sur la moyenne
    de la matrice Q

    Parameters
    ----------
    Q :float
        Matrice à afficher (principalement somme des C_k) (Nx,Ny)


    Returns
    -------
    cmap :obj
        objet cmap associé
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
    r"""Affichage d'une ou 2 cartes C_k à partir d'une cmap divergente perso

    Parameters
    ----------
    alpha :float
        Matrice (M x N)(principalement somme des C_k ou détail d'un C_k)
    duo :bool{False}, optional
        Affichage d'une ou 2 cartes.
    t :int{60}, optional
        position (pixel) centrale des hyperboles utilisées (ordonnée)
    x :int{128}, optional
        position (pixel) centrale des hyperboles utilisées (abscisse)
    title :list{["_","_"]}, optional
        Titres des graphes
    nfile :str{"_"}, optional
        Nom du fichier pour l'enregistrement
    save :bool{False}, optional
        Enregistrer le fichier.

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
    r"""Affichage somme des C_k et image cote-cote

    Parameters
    ----------
    T :float
        tableau [C_k , img] ([(Nx * Ny * K) , (Nx * Ny)]) ou ([(Nx * Ny), (Nx * Ny)])
    ck :bool{False}, optional
        Si les corrections/somme sur les C_k ont ete faites
    title :list{["_","_"]}, optional
        Titres des graphes
    nfile :str{"_"}, optional
        Nom du fichier pour l'enregistrement
    save :bool{False}, optional
        Enregistrer le fichier.
    t :int{60}, optional
        position (pixel) centrale des hyperboles utilisées (ordonnée)
    x :int{128}, optional
        position (pixel) centrale des hyperboles utilisées (abscisse)

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
    r"""Affiche la reconstruction d'une image grace au dictionnaire et
    aux cartes des C_k

    Parameters
    ----------
    Dal :float
        Soit la reconstruction deja calculée (Nx * Ny) soit le tenseur des C_k (Nx * Ny * K).
    Dico :float{None}, optional
        dictionnaire nécessaire pour le calcul de la reconstruction 
        (si non deja calculée) (Nx * Ny * K).
    name :str{"_"}, optional
        nom du fichier a enregistrer.
    save :bool{False}, optional
        enregistrer ou non l'image.
    compute :bool{True}, optional
        reconstruction deja calculée ou non.

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
    r"""Affichage d'un atome du dictionnaire physique
    avec les bonnes dimensions (m et ns)

    Parameters
    ----------
    atm2 :float
        matrice de l'atome (Nx * Ny)
    paraDic :dic
        dictionnaire des paramètres de l'atome (taille ns\m)
        key necessaire : "size_ns_m"

    Returns
    -------
    None

    Examples
    --------
    >>> paraDic={}
    >>> paraDic["size_ns_m"]=[900,45]
    >>> plot_atomNSM(atomes,paraDic)
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
    scaled = (img-img.min())/(img.max()-img.min())
    return scaled
def param_load(size):
    param_dic={}
    param_dic['dim'] = [size[1],size[0]]      # Dim x,t
    param_dic['position'] = [int(param_dic['dim'][0]/2),int(param_dic['dim'][1]/4)]     # pos x_0,t_0
    param_dic['coef'] = [0.91]
    param_dic['scale_a'] = np.geomspace(0.75,35,25)
    param_dic['scale_s'] = [5.5]                         #param sigma
    param_dic['f'] = [100] 
    return create_dicoH(param_dic)
from sklearn.metrics import roc_curve, roc_auc_score

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
    f,ax = plt.subplots(1,figsize=(6.5,6.5))
    for i,ref_2 in enumerate(img):
        a = np.array(ref_2,dtype=np.float64)**2
        b = np.where(mask>128,1,0)#mask/255.0
        b = b.ravel()
        a = a.ravel()
        auc_score = roc_auc_score(b, a)
        fpr, tpr, thresholds = roc_curve(b, a)
        ax.plot([0,1],[0,1],'--',color='black')
        ax.plot(fpr, tpr, label=f'{name[i]} - ROC curve (area = %0.2f)' % auc_score)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
    plt.show()
    
def SVD_gpr(ref,rank):
    U, D, VT = np.linalg.svd(ref, full_matrices=False)
    D[:rank]=0
    A_remake = (U @ np.diag(D) @ VT)
    return A_remake