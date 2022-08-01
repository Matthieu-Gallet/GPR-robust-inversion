import pickle,time,h5py,sys
import numpy as np
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio,mean_squared_error,structural_similarity

sys.path.append('../')
from MIRAG.optim.sparse_coding import *
from MIRAG.optim.source_separation import *
from MIRAG.optim.huber_source_separation import *
from MIRAG.metrique import *
from MIRAG.dictionnaire.dictionary import *
from MIRAG.filtrage_func import SVD_gpr
from MIRAG.affichage import roc_curve_plot

def scale_0_1(img):
    """ Scale an image between 0 and 1

    Parameters
    ----------
    img : ndarray
        Image to scale

    Returns
    -------
    img_scaled : ndarray
        Scaled image
    """
    scaled = (img-img.min())/(img.max()-img.min())
    return scaled

def param_load(size):
    """ Create the dictionary based on a selected parameters

    Parameters
    ----------
    size :  tuple
        Size of the image

    Returns
    -------
    dico : array
        Array of hyperbolas 
    """
    param_dic={}
    param_dic['dim'] = [size[1],size[0]]      # Dim x,t
    param_dic['position'] = [int(param_dic['dim'][0]/2),int(param_dic['dim'][1]/4)]     # pos x_0,t_0
    param_dic['coef'] = [0.91]
    param_dic['scale_a'] = np.geomspace(0.75,35,25)
    param_dic['scale_s'] = [5.5]                         #param sigma
    param_dic['f'] = [100] 
    return create_dicoH(param_dic)

def open_pkl(name):
    """ Open a pkl file and return the data

    Parameters
    ----------
    name : string
        Name of the pkl file

    Returns
    -------
    data : dict
        Dictionary of the pkl file
    """
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data

def open_h5(name):
    """ Open a h5 file and return the data

    Parameters
    ----------
    name : string
        Name of the h5 file

    Returns
    -------
    data : array
        array of the preprocessed radar h5 file
    mask : array
        array of the mask of the preprocessed radar h5 file
    """
    with h5py.File(name, "r") as f:
        data = f["Processed_Data"]["Processed_Image_001"][()]
        mask = np.zeros_like(f['Masks']['Mask_001'][()])
        for i in f['Masks']:
            mask = np.logical_or(mask,f['Masks'][i][()])
    return data,mask

def metric_noise(ref,signal):
    """
    Compute the metric noise between the reference and the signal of three types : PSNR, MSE, SSIM

    Parameters
    ----------
    ref : ndarray
        Reference image
    signal : ndarray
        noisy image

    Returns
    -------
    metric : dict
        dictionnary of the metrics
    """
    d={"PSNR":[],"MSE":[],"SSIM":[]}
    d["PSNR"]= peak_signal_noise_ratio(ref,signal)
    d["MSE"] = mean_squared_error(ref,signal)
    d["SSIM"]= structural_similarity(ref,signal)
    return d
