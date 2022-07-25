"""
Module for creating the hyperbola dictionary for convolutional ADMM
based on the publication [1]_
"""

import numpy as np

from .dico_func import *


def hyperbola_atomTH(Nx, Nt, x_0, t_0, A=None, sigma=5, a=0.8, f=500):
    r"""A function to generate a hyperbola atom in order to create a
    create a dictionary for GPR reconstruction. (Based on [1]_.)

    Parameters
    ----------
    Nx :int
        Size of the GPR radargram in the distance dimension.
    Nt :int
        Size of the GPR radargram in the time dimension
    x_0 :float
        Position of the hyperobla in the distance dimension
    t_0 :float
        position of the hyperobla in the time dimension
    A :ndarray{None}, optional
        Attenuation array                           
    sigma :int{5}, optional
        Thickness of the hyperbola
    a :float{0.8}, optional
        Affects the opening of the hyperbola.            
    f :int{500}, optional
        Affects the flatness at the top of the hyperbola.

    Returns
    -------
    h : float
        constructed atom of dimension (Nx*Ny)

    
    References
    ----------
    .. [1] 'Parsimonious decomposition of the GPR useful signal from the Hyperbola dictionary',
       Guillaume Terasse, Jean-Marie Nicolas, Emmanuel Trouvé and Émeline Drouet
       
       Available at: https://hal.archives-ouvertes.fr/hal-01351242
    """
    # Setting mesh for x and t axis
    x, t = np.meshgrid(range(1, Nx + 1), range(1, Nt + 1))

    # Defining g using equation (4)
    g = a * np.sqrt(f ** 2 + (x - x_0) ** 2) + t_0 - f * a
    # Defining ricker wavelet from equation (3)
    r = (
        2
        / (np.sqrt(3 * sigma) * np.pi ** (0.25))
        * (1 - ((t - g) / sigma) ** 2)
        * np.exp(-0.5 * ((t - g) / sigma) ** 2)
    )

    # Finally, we construct the atom with equation (2)
    h = A * r
    return h


def create_dicoH(paraDic):
    r"""Superior function for creating hyperbolas or atoms dictionary (Based on [1]_.)

    Parameters
    ----------
    paradic :dic
        dictionnary of parameters

    Notes
    -----
    - paraDic['dim'] (array) : Dimension of the radargram
    - paraDic['scale_a'] (array) : Input vector of a parameter
    - paraDic['scale_s'] (array) : Input vector of sigma parameter
    - paraDic['f'] (array) : Input vector of f parameter
    - paraDic['position'] (array) : Position of the top of the hyperbola

    Returns
    -------
    DioH : dic
        structure of the form {"atoms": Dico, "param": param} where ``Dico`` is the tensor
        of the hyperbolas (Nx*Ny*K) and ``param`` the tensor of the corresponding features
        for each hyperbola (K*4)
    """
    Nx = paraDic["dim"][0]
    Nt = paraDic["dim"][1]
    v_a = paraDic["scale_a"]
    v_sig = paraDic["scale_s"]
    v_f = paraDic["f"]
    pos = paraDic["position"]
    Dico = np.zeros((Nt, Nx, len(v_a) * len(v_sig) * len(v_f) * len(paraDic["coef"])))
    param = np.zeros((len(v_a) * len(v_sig) * len(v_f) * len(paraDic["coef"]), 4))
    co = 0
    for H in v_f:
        for st in v_sig:
            for sx in v_a:
                for s_a in paraDic["coef"]:
                    A = filtre2D_B(Nx, Nt, pos[0], pos[1], coef=s_a)
                    at = hyperbola_atomTH(
                        Nx,
                        Nt,
                        x_0=pos[0],
                        t_0=pos[1],
                        A=A,
                        sigma=st,
                        a=sx,
                        f=H,
                    )
                    at = at / np.linalg.norm(at, "fro")  # Frobenius normalization
                    Dico[:, :, co] = at
                    param[co, :] = np.array([H, st, sx, s_a])
                    co = co + 1
                DioH = {"atoms": Dico, "param": param}
    return DioH

def param_load(size,opt):
    r"""Loads the parameters and
    returns a dictionary of atoms of dimension "size

    Parameters
    ----------
    size :list{[256,256]}
        dimension of the original image, thus of the atoms.
    opt :dic
        dictionary of options for creating hyperbolas

    Returns
    -------
    out :dic
        structure of the form {"atoms": Dico, "param": param} where ``Dico`` is the hyperbola tensor
        of the hyperbolas (Nx*Ny*K) and ``param`` the tensor of the corresponding features
        for each hyperbola (K*4)
    opt :dic
        input dictionary augmented with the values used by the functions
        (for tracking changes)


    .. warning::
       **WARNING** 
       
       Modification of the dictionary creation parameters directly in the
       python file! 
       
       **To do**

       - Add the option to read the parameters from a YAML file
    """
    param_dic={}
    param_dic['dim'] = [size[1],size[0]]      # Dim x,t
    param_dic['position'] = [int(param_dic['dim'][0]/2),int(param_dic['dim'][1]/4)]     # pos x_0,t_0
    param_dic['coef'] = [0.91]
    param_dic['scale_a'] = np.geomspace(0.8,40,23)
    param_dic['scale_s'] = [5]                         #param sigma
    param_dic['f'] = [100] 
    for i in param_dic.keys():
        if i=="scale_a":
            opt["dico_param_"+i]=param_dic[i].tolist()
        else:
            opt["dico_param_"+i]=param_dic[i]

    return create_dicoH(param_dic),opt