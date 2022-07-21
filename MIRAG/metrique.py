"""
Module de calcul de métrique pour quantifier la qualité de reconstruction des radargrammes
"""

import cv2
import h5py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


def comp_metric(Iorg,Irec,Ck,path_mask,erode=True,sz=20,pos_rec=[0,100,0,600],dim_win=[80,40],out_win_S=False,v_seuil=3,plot_check=False):
    r"""Métrique de détection hyperboles reconstruites/vérité terrain
    Fonction principale du calcul de la détection des hyperboles par rapport à un mask

    Parameters
    ----------
    Iorg :float
        array image originale normalisée (0-1) (MxN)
    Irec :float
        array image reconstruite normalisée (0-1)(MxN)
    Ck :float
        matrice de la somme des cartes de coefficients normalisées (MxN)
    path_mask :str
         chemin du mask
    erode :bool {True}, optional
        active l'erosion du mask
    sz :int {20}, optional: 
        taille du kernel pour l'erosion.
    pos_rec :list {[0,100,0,600]}, optional
        position de l'image originale par rapport au mask.
    dim_win :list{[80,40]}, optional
        taille de la fenêtre de détection en pixels.
    out_win_S :bool{False}, optional
        retourne la fenêtre seuillée ou non
    v_seuil :int{3}, optional
        Valeur du seuil par rapport à l'écart type des C_k. Par défaut mise à 0 
        des C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean
    plot_check:bool{False},optional
        Active l'affichage des positions des centroids sur le mask

    Returns
    -------
    score : dic
        structure des scores et metriques calculés pour une reconstruction à partir de son masque
    
    Notes
    -----
    - score['MSE_local_max'] (float) : maximum MeanSquareError pour chaque masque entre l'image reconstruite et celle originale
    - score['MSE_global'] (float) : MeanSquareError globale entre l'image reconstruite et celle originale.
    - score['SSI_local_min'] (float) : minimum Structural similarity index pour chaque masque entre l'image reconstruite et celle originale
    - score['SSI_global'] (float) : Structural similarity index globale entre l'image reconstruite et celle originale.    
    - score['TruePositive'] (float) : nombre d'hyperbole détectée dans l'image fournit.
    - score['N_terrain'] (float) : nombre theorique d'hyperboles à partir du mask.
    - score["FalseNegative"] (float) : hyperboles non-créés par la reconstruction mais presentes dans les masques
    - score["FalsePositive"] (float) : hyperboles créés par la reconstruction mais non presentes dans les masques
    - score["precision"] (float) : TruePositive/(TruePositive+FalsePositive)
    - score["rappel"] (float) : TruePositive/(TruePositive+FalseNegative)
    - score["F1_score"] (float) : (precision*rappel)/(precision+rappel)

    .. warning::
       **A faire**

       - Implémentation des métriques:
            * précision : :math:`\mathrm{TruePositive/(TruePositive+FalsePositive)}`
            * rappel    : :math:`\mathrm{TruePositive/(TruePositive+FalseNegative)}`
            * F1-score  : :math:`\mathrm{precision\cdot rappel/(precision+rappel)}`
    """
    c_sta={}
    MSE_ce = []
    C0_ce = []
    SSI_ce = []
    c_sta["mean"] = Ck.mean()
    c_sta["std"] = Ck.std()

    if type(path_mask)==str:
        mask = load_mask(path_mask)
    else:
        mask = path_mask

    centroid = center_mask(mask,erode,sz)
    centroid = centroid_adjust_win(centroid,mask.shape[0],pos_rec)
    if plot_check:
        plt.imshow(mask,cmap="gray")
        plt.scatter(centroid[:,0],centroid[:,1])
        plt.show()

    for cen in centroid:
        org_s = fenetre(cen,Iorg,dim_win)
        rec_s = fenetre(cen,Irec,dim_win)
        ck_s = fenetre(cen,Ck,dim_win)
        Mse_temp,nc0_temp,ssim_l = stats_win(c_sta,ck_s,org_s,rec_s,out_win_S,v_seuil)
        MSE_ce.append(Mse_temp)
        C0_ce.append(nc0_temp)
        SSI_ce.append(ssim_l)

    score = stats_reconstruction(Iorg, Irec,MSE_ce,C0_ce,SSI_ce,centroid,c_sta,Ck,v_seuil)

    return score

def stats_reconstruction(Iorg, Irec,MSE_ce,C0_ce,SSI_ce,centroid,c_sta,Ck,v_seuil):
    r"""Calcule les statistiques et métriques (local et global)
    à partir de la reconstruction.

    Parameters
    ----------
    Iorg :float
        array image originale normalisée (0-1) (MxN)
    Irec :float
        array image reconstruite normalisée (0-1)(MxN)
    MSE_ce : float
        array des MSE locaux sur tous les objets du masque
    C0_ce : float
        array des détection sur tous les objets du masque
    SSI_ce : float
        array des SSI locaux sur tous les objets du masque
    centroid : int
        Nombre d'hyperbole détecté sur le masque
    c_sta : dic
        caractéristique de la matrice Ck. c_sta["mean"] et c_sta["std"]
    Ck :float
        matrice de la somme des cartes de coefficients normalisées (MxN)
    v_seuil :int{3}, optional
        Valeur du seuil par rapport à l'écart type des C_k. Par défaut mise à 0 
        des C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean

    Returns
    -------
    stat_strc : dic
        dictionnaire des statistiques et métriques calculées sur la reconstruction.

    See Also
    --------
    stats_win
    comp_metric
    """
    stat_strc = {}
    Nhyper_glob = detect_Nhyper_rec(c_sta,Ck,v_seuil)
    stat_strc["MSE_global"] = mean_squared_error(Iorg, Irec)
    stat_strc["SSI_global"] = ssim(Iorg, Irec)
    stat_strc["MSE_local_max"] = np.max(MSE_ce)
    stat_strc["SSI_local_min"] = np.min(SSI_ce)
    n_cent = len(centroid)
    TruePositive = np.sum(np.array(C0_ce)>0)
    FalseNegative = n_cent- TruePositive
    FalsePositive = Nhyper_glob - TruePositive
    precision = TruePositive/(TruePositive+FalsePositive)
    rappel = TruePositive/(TruePositive+FalseNegative)
    stat_strc["N_terrain"] = n_cent
    stat_strc["FalseNegative"] = FalseNegative
    stat_strc["FalsePositive"] = FalsePositive
    stat_strc["TruePositive"] = TruePositive
    stat_strc["precision"] = precision
    stat_strc["rappel"] = rappel
    stat_strc["F1_score"] = (precision*rappel)/(precision+rappel)
    return stat_strc


def get_available_masks(name):
    r"""Obtient la liste de tous les masks dans un fichier .h5

    Parameters
    ----------
    name :str
        chemin absolu du fichier .h5

    Returns
    -------
    masks_available : str
        dictionnaire des noms des masks disponibles pour un fichier .h5
    """
    filepath =  name
    masks_available = {}
    with h5py.File( filepath, 'r') as f:
        masks_group = f['Masks']
        for mask_dataset in masks_group.values():
            masks_available.update( { mask_dataset.name : 
                                        {"Mask_Creator" : mask_dataset.attrs.get("Mask_Creator"), 
                                         "Mask_Category" : mask_dataset.attrs.get("Mask_Category"),
                                         "Mask_Validated" : str(mask_dataset.attrs.get("Mask_Validated")),   
                                         "Mask_Creation_Timestamp" : mask_dataset.attrs.get("Mask_Creation_Timestamp") } } )
                                         
    return masks_available


def load_mask(filename):
    r"""Charge le masque à partir d'un fichier .h5

    Parameters
    ----------
    filename :str
        chemin absolu du fichier .h5

    Returns
    -------
    mask : float
        array 2D des masks
    """
    det_mask = get_available_masks(filename)
    mask_name = list(det_mask.keys())
    with h5py.File( filename, 'r') as f:
        sha = f[mask_name[0]][()].shape
        nmask = len(mask_name)
        mask=np.zeros((sha[0],sha[1]))
        for i in range(nmask):
            mask[:,:] = np.logical_or(mask,f[mask_name[i]][()])
    return mask


def mask2scatter(mask,erode=True,size=20):
    r"""Transforme un array de masks en scatter x,y
    Permet d'appliquer une erosion ou non sur l'image 
    (permettant ainsi une meilleur détection des masks)

    Parameters
    ----------
    mask :int
        array des masks
    erode :bool{True}, optional
        active l'érosion des masks.
    size :int{20}, optional
        taille du kernel de l'érosion si sélectionné.

    Returns
    -------
    mask_scatter :int
        scatter x,y des masks
    ero :int
        masks erodé ou non
    """
    if erode :
        ero = cv2.erode(mask, np.ones((size,size),np.uint8),iterations = 1)
    else:
        ero = mask
    xy = np.argwhere(ero==1)
    x = xy[:,1]
    y = mask.shape[0]-xy[:,0]
    mask_scatter = np.array([x,y]).T
    return mask_scatter,ero


def center_mask(mask,erode=True,sz=20):
    r"""Extrait d'un mask les positions des clusters
    Permet d'appliquer une erosion ou non sur l'image 
    (permettant ainsi une meilleur détection des masks)

    Parameters
    ----------
    mask :int
        array des masks
    erode :bool{True}, optional
        active l'érosion des masks.
    size :int{20}, optional
        taille du kernel de l'érosion si sélectionné.

    Returns
    -------
    centroid :int
        Coordonnées des centres de clusters
    """
    m_scat,ero = mask2scatter(mask,erode,sz)
    image = np.uint8(ero*255)
    det_clus,_ = cv2.connectedComponents(image)

    kmeans = KMeans(n_clusters=det_clus, random_state=0).fit(m_scat)
    centroid=kmeans.cluster_centers_
    return centroid


def centroid_adjust_win(centroid,y_mask,dim_rec):
    r"""Retourne les centres des masks ajustés
    L'ajustement se fait par rapport à la taille de l'image,
    pour éviter les débordements.

    Parameters
    ----------
    centroid :float
        tableau des centres des masques
    y_mask :int
        dimension Y de l'image du masque
    dim_rec :int
        tableau des dimensions de l'image utilisé (souvent inférieur au mask)

    Returns
    -------
    cent :int
        centre des masks ajustés a l'image reconstruite
    """
    centroid[:,1]=y_mask-centroid[:,1]
    cent = centroid[(centroid[:,0]-dim_rec[2]>0)&(centroid[:,0]<dim_rec[3])]+[-dim_rec[2],0]
    cent = cent[(cent[:,1]-dim_rec[0]>0)&(cent[:,1]<dim_rec[1])]+[0,-dim_rec[0]]
    return cent


def fenetre(cen,img,dim):
    r"""Retourne une fenetre centrée sur un point de dimension totale dim (LxH)

    Parameters
    ----------
    cen :int
        centre de la fenêtre
    img :float
        image à centrer
    dim :int
        dimension totale de la fenêtre LxH

    Returns
    -------
    out :float
        fenêtre extraite de l'image initiale
    """
    dim = 0.5*np.array(dim)
    cm = np.int32(cen-dim)
    cp = np.int32(cen+dim)
    cm = np.where(cm<0,0,cm)
    cp = np.where(cp>[img.shape[1],img.shape[0]],[img.shape[1],img.shape[0]],cp)
    return img[cm[1]:cp[1],cm[0]:cp[0]]


def stats_win(ck_stats,ck_s,org_s,rec_s,win_S=False,v_seuil=3):
    r"""Réalise les statistiques sur un masque dans image
    à savoir le nombre d'hyperbole détecté (norme C0), le MeanSquareError entre 
    l'image originale et celle originale au niveau du masque.

    Parameters
    ----------
    ck_stats :float
        dictionnaire de la moyenne et l'écart-type des cartes c_k
    ck_s :float
        fenêtre des C_k donnée par la fonction fenetre pour un masque donné
    org_s :float
        fenêtre de l'image originale donnée par la fonction fenetre pour un masque donné
    rec_s :float
        fenêtre de l'image reconstruite donnée par la fonction fenetre pour un masque donné
    win_S :bool{True}, optional
        retourne la fenêtre seuillée ou non
    v_seuil :int{3}, optional
        Valeur du seuil par rapport à l'écart type des C_k. Par défaut mise à 0 
        des C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean

    Returns
    -------
    MSE : float
        MeanSquareError entre l'image reconstruite et celle originale.
    norm_C0 : int
        norme 0 de C_k sur la fenêtre(>0 = détection).
    ssim_loc : float
        indice de similarité local 
    ws :float,optionnal
        fenetre C_k seuillée
    

    .. warning::
       **A faire**

       - Implémentation de la métrique :math:`\mathrm{Structural\ Similarity\ Index}` : 
       https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity

    """
    se_ck = v_seuil*ck_stats["std"]
    ws=np.where((ck_s<ck_stats["mean"]-se_ck)|(ck_s>ck_stats["mean"]+se_ck),ck_s,0)
    MSE = mean_squared_error(org_s, rec_s)
    ssim_loc = ssim(org_s, rec_s)
    norm_C0 = (ws>0).sum()
    #norm_C1 = ws.sum()
    if win_S:
        return MSE,norm_C0,ssim_loc,ws
    else:
        return MSE,norm_C0,ssim_loc

def detect_Nhyper_rec(ck_stats,Ck,v_seuil=3):
    r"""Detecte le nombre d'hyperbole total construite
    par l'algorithme sous réserve d'un seuil

    Parameters
    ----------
    ck_stats : [type]
        [description]
    Ck : [type]
        [description]
    v_seuil :int{3}, optional
        Valeur du seuil par rapport à l'écart type des C_k. Par défaut mise à 0 
        des C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean

    Returns
    -------
    [type]
        [description]
    """
    se_ck = v_seuil*ck_stats["std"]
    ws=np.where((Ck<ck_stats["mean"]-se_ck)|(Ck>ck_stats["mean"]+se_ck),Ck,0)
    norm_C0_glob = (ws>0).sum()
    return norm_C0_glob
