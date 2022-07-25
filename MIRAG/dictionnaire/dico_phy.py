'''
Hyperbola dictionary creation module for convolutional ADMM by physical approach
'''

import numpy as np
from .dico_func import *

def dico_PHY2(Dimxt, pos, v, dim_n_m, A=None, sigma=5):
    r"""A function to generate an atom or a hyperbola to create a dictionary for
    create a dictionary for the GPR reconstruction from the physical
    from the physical modeling of the problem.

    Parameters
    ----------
    Dimxt :list{[256,256]}
        dimension of the hyperbola (Nx * Ny)
    pos :list{[60,128]}
        centered position of the hyperbola
    v :float
        propagation speed for the hyperbola
    dim_n_m :list{[900,40]}
        real dimensions of the image (y = [ns] and x= [m])
    A :int{None}, optional
        attenuation matrix (Nx * Ny)
    sigma :int{5}, optional
        parameter thickness of the hyperbola.


    Returns
    -------
    h : float
        constructed atom of dimension (Nx*Ny)
    """
    # Setting mesh for x and t axis
    Nt = Dimxt[1]
    Nx = Dimxt[0]
    x, t = np.meshgrid(range(1, Nx + 1), range(1, Nt + 1))
    t_0 = pos[1] * dim_n_m[0] / Nt
    x_0 = pos[0] * dim_n_m[1] / Nx
    R = 0
    # R = R*dim_n_m[1]/Nx
    ## NT
    t = t * dim_n_m[0] / Nt
    ## NX
    x = x * dim_n_m[1] / Nx

    v = 1e-9 * v
    P = 2 / v
    # g = P * np.sqrt(((t_0/P)+R)**2 + (x-x_0)**2)-P*R
    g = P * np.sqrt((t_0 / P) ** 2 + (x - x_0) ** 2)

    # Defining ricker wavelet from equation (3)
    r = (
        2
        / (np.sqrt(3 * sigma) * np.pi ** (0.25))
        * (1 - ((t - g) / sigma) ** 2)
        * np.exp(-0.5 * ((t - g) / sigma) ** 2)
    )
    # b, a = butter(2, 0.0025)
    # r = filtfilt(b, a, r)
    r = r * Nx / dim_n_m[0]
    # Finally, we construct the atom with equation (2)
    h = A * r
    return h


def create_dicoPHY2(paraDic):
    r"""Higher function of creation of hyperbolas or atoms dictionary
    by physical modeling

    Parameters
    ----------
    paraDic :dic
        dictionary of creation parameters

    Notes
    -----
    - paraDic['dim'] (array) : Dimension of the radargram
    - paraDic['std'] (array) : Input vector of sigma parameter
    - paraDic['position'] (array) : Position of the top of the hyperbola
    - paraDic['v_prop'] (array) : Input vector of velocity
    - paraDic['coef'] (array) : 2D filter coefficient for the size of the hyperbola
    - paraDic['size_ns_m'] (array) : value of corresponding measure in time and meter for the radar.

    Returns
    -------
    DioH : dic
        structure of the form {"atoms": Dico, "param": param} where ``Dico`` is the tensor
        of the hyperbolas (Nx*Ny*K) and ``param`` the tensor of the corresponding features
        for each hyperbola (K*3) (velocity, attenuation coefficient, sigma)
    """
    pos = paraDic["position"]
    vpr = paraDic["v_prop"]
    sig = paraDic["std"]
    dimPHY = paraDic["size_ns_m"]
    Dico = np.zeros(
        (
            paraDic["dim"][1],
            paraDic["dim"][0],
            len(vpr) * len(paraDic["coef"] * len(sig)),
        )
    )
    param = np.zeros((len(vpr) * len(paraDic["coef"] * len(sig)), 3), dtype="complex")
    co = 0
    for v in vpr:
        for s_a in paraDic["coef"]:
            for st in sig:
                A = filtre2D_B(
                    paraDic["dim"][0], paraDic["dim"][1], pos[0], pos[1], coef=s_a
                )
                at = dico_PHY2(paraDic["dim"], pos, v, dimPHY, A=A, sigma=st)
                at = at / np.linalg.norm(at, "fro")
                Dico[:, :, co] = at
                param[co, :] = np.array([v, s_a, st])
                co = co + 1
        DioH = {"atoms": Dico, "param": param}
    return DioH


def param_loadPHY(size, opt):
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
    atm :dic
        structure of the form {"atoms": Dico, "param": param} where ``Dico`` is the hyperbola tensor
        of the hyperbolas (Nx*Ny*K) and ``param`` the tensor of the corresponding features
        for each hyperbola (K*3)
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
    # Ajout yaml reader
    Nx = size[1]
    Nt = size[0]

    paraDic = {}
    paraDic["dim"] = [Nx, Nt]
    paraDic["position"] = [int(Nx / 2), int(Nt / 4)]
    paraDic["size_ns_m"] = [96, 46]

    paraDic["dimF"] = opt["dimRAw"]
    paraDic["size_ns_mB"] = [96, 46]
    paraDic["dim"] = [Nx, Nt]
    paraDic["position"] = [int(paraDic["dim"][0] / 2), int(paraDic["dim"][1] / 4)]

    paraDic["size_ns_m"] = [
        paraDic["dim"][1] * paraDic["size_ns_mB"][0] / paraDic["dimF"][1],
        paraDic["dim"][0] * paraDic["size_ns_mB"][1] / paraDic["dimF"][0],
    ]

    paraDic["freq"] = 350e6
    paraDic["cond"] = 0
    paraDic["std"] = [0.5]
    paraDic["coef"] = [0.75]
    paraDic["thick_air"] = 0.75
    paraDic["perm_eff"] = np.linspace(5, 100, 101)
    paraDic["v_prop"] = eps2vprop(paraDic, margeR=0.5, diff=0.1)

    for i in paraDic.keys():
        if i == "perm_eff":
            opt["dico_param_" + i] = paraDic[i].tolist()
        elif i == "v_prop":
            opt["dico_param_" + i] = paraDic[i].tolist()
        else:
            opt["dico_param_" + i] = paraDic[i]

    atm = create_dicoPHY2(paraDic)

    return atm, opt
