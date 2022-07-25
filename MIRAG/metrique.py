"""
Metric calculation module to quantify the reconstruction quality of radargrams
"""

import cv2
import h5py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


def comp_metric(Iorg,Irec,Ck,path_mask,erode=True,sz=20,pos_rec=[0,100,0,600],dim_win=[80,40],out_win_S=False,v_seuil=3,plot_check=False):
    r"""Metric of detection of reconstructed hyperbolas/ground truth
    Main function of the hyperbola detection calculation with respect to a mask

    Parameters
    ----------
    Iorg :float
        array original normalized image (0-1) (MxN)
    Irec :float
        array normalized reconstructed image (0-1)(MxN)
    Ck :float
        matrix of the sum of the normalized coefficient maps (MxN)
    path_mask :str
         path of the mask
    erode :bool {True}, optional
        activate the erosion of the mask
    sz :int {20}, optional: 
        kernel size for erosion.
    pos_rec :list {[0,100,0,600]}, optional
        position of the original image in relation to the mask.
    dim_win :list{[80,40]}, optional
        size of the detection window in pixels.
    out_win_S :bool{False}, optional
        return the thresholded window or not
    v_threshold :int{3}, optional
        Value of the threshold in relation to the standard deviation of C_k. By default set to 0 
        of C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean
    plot_check:bool{False},optional
        Activate the display of the centroids positions on the mask

    Returns
    -------
    score : dic
        structure of the scores and metrics computed for a reconstruction from its mask
    
    Notes
    -----
    - score['MSE_local_max'] (float) : maximum MeanSquareError for each mask between the reconstructed and the original image
    - score['MSE_global'] (float) : global MeanSquareError between the reconstructed image and the original one.
    - score['SSI_local_min'] (float) : minimum Structural similarity index for each mask between the reconstructed image and the original one
    - score['SSI_global'] (float): Global structural similarity index between the reconstructed image and the original one.    
    - score['TruePositive'] (float) : number of hyperbola detected in the provided image.
    - score['N_terrain'] (float): theoretical number of hyperbolas from the mask.
    - score["FalseNegative"] (float) : hyperbolas not created by the reconstruction but present in the masks
    - score["FalsePositive"] (float) : hyperbolas created by the reconstruction but not present in the masks
    - score["precision"] (float) : TruePositive/(TruePositive+FalsePositive)
    - score["recall"] (float) : TruePositive/(TruePositive+FalseNegative)
    - score["F1_score"] (float) : (precision*recall)/(precision+recall)

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
    r"""Calculates statistics and metrics (local and global)
    from the reconstruction.

    Parameters
    ----------
    Iorg :float
        array original normalized image (0-1) (MxN)
    Irec :float
        array normalized reconstructed image (0-1)(MxN)
    MSE_ce : float
        array of local MSE on all mask objects
    C0_ce : float
        array of detections on all objects of the mask
    SSI_ce : float
        array of local SSI on all mask objects
    centroid : int
        Number of hyperbola detected on the mask
    c_sta : dic
        characteristic of the matrix Ck. c_sta["mean"] and c_sta["std"]
    Ck :float
        matrix of the sum of the normalized coefficient maps (MxN)
    v_threshold :int{3}, optional
        Value of the threshold with respect to the standard deviation of C_k. By default set to 0 
        of C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean

    Returns
    -------
    stat_strc : dic
        dictionary of statistics and metrics calculated on the reconstruction.

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
    r"""Gets the list of all masks in a .h5 file

    Parameters
    ----------
    name :str
        absolute path of the .h5 file

    Returns
    -------
    masks_available : str
        dictionary of available masks names for a .h5 file
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
    r"""Loads the mask from an .h5 file

    Parameters
    ----------
    filename :str
        absolute path of the .h5 file

    Returns
    -------
    mask : float
        2D array of masks
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
    r"""Transforms an array of masks into a scatter x,y
    Allows to apply an erosion or not on the image 
    (allowing a better detection of masks)

    Parameters
    ----------
    mask :int
        array of masks
    erode :bool{True}, optional
        activate erosion of masks.
    size :int{20}, optional
        size of the erosion kernel if selected.

    Returns
    -------
    mask_scatter :int
        scatter x,y of masks
    ero :int
        masks eroded or not
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
    r"""Extract from a mask the positions of the clusters
    Allows to apply an erosion or not on the image 
    (allowing a better detection of masks)

    Parameters
    ----------
    mask :int
        array of masks
    erode :bool{True}, optional
        activate erosion of masks.
    size :int{20}, optional
        size of the erosion kernel if selected.

    Returns
    -------
    centroid :int
        Coordinates of cluster centers
    """
    m_scat,ero = mask2scatter(mask,erode,sz)
    image = np.uint8(ero*255)
    det_clus,_ = cv2.connectedComponents(image)

    kmeans = KMeans(n_clusters=det_clus, random_state=0).fit(m_scat)
    centroid=kmeans.cluster_centers_
    return centroid


def centroid_adjust_win(centroid,y_mask,dim_rec):
    r"""Returns the centers of the adjusted masks
    The adjustment is made in relation to the size of the image,
    to avoid overflows.

    Parameters
    ----------
    centroid :float
        array of mask centers
    y_mask :int
        dimension Y of the mask image
    dim_rec :int
        array of dimensions of the used image (often smaller than the mask)

    Returns
    -------
    cent :int
        center of the masks adjusted to the reconstructed image
    """
    centroid[:,1]=y_mask-centroid[:,1]
    cent = centroid[(centroid[:,0]-dim_rec[2]>0)&(centroid[:,0]<dim_rec[3])]+[-dim_rec[2],0]
    cent = cent[(cent[:,1]-dim_rec[0]>0)&(cent[:,1]<dim_rec[1])]+[0,-dim_rec[0]]
    return cent


def fenetre(cen,img,dim):
    r"""Returns a window centered on a point of total dimension dim (LxH)

    Parameters
    ----------
    cen :int
        center of the window
    img :float
        image to center
    dim :int
        total dimension of the window LxH

    Returns
    -------
    out :float
        window extracted from the initial image
    """
    dim = 0.5*np.array(dim)
    cm = np.int32(cen-dim)
    cp = np.int32(cen+dim)
    cm = np.where(cm<0,0,cm)
    cp = np.where(cp>[img.shape[1],img.shape[0]],[img.shape[1],img.shape[0]],cp)
    return img[cm[1]:cp[1],cm[0]:cp[0]]


def stats_win(ck_stats,ck_s,org_s,rec_s,win_S=False,v_seuil=3):
    r"""Performs statistics on a mask in image
    namely the number of hyperbola detected (C0 norm), the MeanSquareError between 
    the original image and the original one at the mask level.

    Parameters
    ----------
    ck_stats :float
        dictionary of the mean and standard deviation of the c_k maps
    ck_s :float
        window of C_k given by the window function for a given mask
    org_s :float
        window of the original image given by the window function for a given mask
    rec_s :float
        window of the reconstructed image given by the window function for a given mask
    win_S :bool{True}, optional
        returns the window thresholded or not
    v_threshold :int{3}, optional
        Value of the threshold in relation to the standard deviation of the C_k. By default set to 0 
        of C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean

    Returns
    -------
    MSE : float
        MeanSquareError between the reconstructed image and the original one.
    norm_C0 : int
        norm 0 of C_k on the window (>0 = detection).
    ssim_loc : float
        local similarity index 
    ws :float,optionnal
        thresholded window C_k

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
    r"""Detects the total number of hyperbolas constructed
    by the algorithm subject to a threshold

    Parameters
    ----------
    ck_stats : [type]
        [description]
    Ck : [type]
        [description]
    v_threshold :int{3}, optional
        Value of the threshold with respect to the standard deviation of C_k. By default set to 0 
        of C_K.mean-3*C_K.std<C_k< 3*C_K.std+C_K.mean

    Returns
    -------
    norm_C0_glob : int
        total number of hyperbolas detected
    """
    se_ck = v_seuil*ck_stats["std"]
    ws=np.where((Ck<ck_stats["mean"]-se_ck)|(Ck>ck_stats["mean"]+se_ck),Ck,0)
    norm_C0_glob = (ws>0).sum()
    return norm_C0_glob
