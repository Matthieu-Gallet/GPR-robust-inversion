"""
Complementary module for the ``ADMM`` convolutional

References
----------
.. [1] 'Multichannel sparse recovery of complex-valued signals using Huber’s criterion',Esa Ollila
        Avalaible at: https://arxiv.org/pdf/1504.04184.pdf

.. [2] 'Robust Principal Component Analysis?', Candes & al.
        Avalaible at: http://www.columbia.edu/~jw2966/papers/CLMW11-JACM.pdf

.. [3] 'Distributed Optimization and Statistical Learning
        via the Alternating Direction Method of Multipliers p.23', Stephen Boyd
        Avalaible at: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
"""

import numpy as np

from scipy import linalg
from pywt import threshold
from sklearn.base import BaseEstimator

def huber_complex(x, delta):
    r"""Huber function applied to a complex array as proposed in [1]_

    :math:`\rho_{H, \delta}(x)=\left\{\begin{array}{ll}|x|^{2}, & \text { for }|x| \leq \delta 
    \\2 \delta|x|-\delta^{2}, & \text { for }|x|>\delta\end{array}\right.`

    Parameters
    ----------
    x : complex
        array to apply the Huber function
    delta : float
        threshold parameter

    Returns
    -------
    out : complex
        array with the Huber function applied
    """
    return ((np.abs(x)<=delta)*np.abs(x)**2 + (np.abs(x)>delta)*(2*delta*np.abs(x)-delta**2))

    
def sign_complex_array(x):
    r""" Determine the sign of a complex array, based on [1]_

    :math:`\operatorname{sign}(e)=\left\{\begin{array}{ll}
    e /|e|, & \text { for } e \neq 0 \\
    0, & \text { for } e=0
    \end{array}\right.`

    Parameters
    ----------
    x : complex
        array to determine the sign
    Returns
    -------
    out : int
        array with the sign of the input array
    """
    return np.where(x!=0,x/np.abs(x),0)


def loss_derivative_function_complex(x, delta):
    r""" Derivative of the Huber function, based on [1]_

    :math:`\psi_{H, \delta}(x)=\left\{\begin{array}{ll}
    x, & \text { for }|x| \leq \delta \\
    \delta \operatorname{sign}(x), & \text { for }|x|>\delta
    \end{array}\right.`

    Parameters
    ----------
    x : complex
        array to apply the Huber function
    delta : float
        threshold parameter

    Returns
    -------
    out : complex
        array with the derivative of the Huber function applied
    """
    return np.where(np.abs(x)>delta,delta*sign_complex_array(x),x)


def gradient_huber(S_tilde, X_m_tilde, D_m_tilde, D_m_H_tilde, Z_m_tilde, rho, delta):
    r""" Function to compute the gradient of the Huber function of the following 
    minimization problem:

    :math:`\underset{\mathbf{\hat{c}}}{\operatorname{argmin}}
    \sum_N\mathcal{H}_\delta(\mathbf{\hat{D}} \cdot \mathbf{\hat{c}}-\mathbf{\hat{y}})+
    \frac{\rho}{2}\Big|\Big| {\mathbf{\hat{c}}-\mathbf{\hat{z}}}\Big|\Big|_2^2`

    The gradient is computed as follows:

    :math:`\nabla_\mathbf{\hat{c}}f=\{\mathbf{\tilde{DH}_m}\}_k\odot\Psi_\delta
    \left(\sum_{k}\mathbf{\tilde{D}_m}_k\odot\mathbf{\tilde{X}_m}_k
    -\mathbf{\tilde{S}}\right)+\rho({\mathbf{\tilde{X}_m}-\mathbf{\tilde{Z}_m}})`

    Parameters
    ----------
    S_tilde : complex
        fft of the original signal (Nx,Ny)
    X_m_tilde : complex
        fft of the coefficients maps :math:`\mathbf{\hat{C}_k}` (Nx,Ny,K)
    D_m_tilde : complex
        fft of the dictionary :math:`\mathbf{\hat{D}_k}` (Nx,Ny,K)
    D_m_H_tilde : complex
        fft hermitian of the dictionary :math:`\mathbf{\hat{D}_k}^H` (Nx,Ny,K)
    Z_m_tilde : complex
        fft (auxiliary variable- dual variable)  (Nx,Ny,K)
    rho : float
        regularization parameter
    delta : float
        threshold parameter
    Returns
    -------
    out: complex
        gradient of the Huber function (NX,NY,K)
    """
    r_n_tilde = np.sum(D_m_tilde*X_m_tilde, axis=2,keepdims=True) + S_tilde
    xi = loss_derivative_function_complex(r_n_tilde, delta)
    nabla_w = 0.5*(xi*D_m_H_tilde)
    nabla_q = rho * (X_m_tilde+Z_m_tilde)
    return nabla_w+nabla_q


def proxH(x,seuil,rho):
    """ proximal operator of the Huber function

    Parameters
    ----------
    x :  ndarray
        input signal
    seuil :  float
        threshold
    rho :  float
        regularization parameter

    Returns
    -------
    ndarray
        result of the proximal operator
    """
    return np.where(np.abs(x)<=seuil*(rho+1),x/(rho+1),x-seuil*rho*sign_complex_array(x))


def thresholding_nuclear(M,i,lambdaS,rho):
    r"""Fonction de calcul du seuillage par valeurs singulières
    Elle est la résultante de l'opérateur proximal associé à la 
    norme nucléaire.

    :math:`\underset{\mathbf{{L}}}{\operatorname{argmin}}
    \lambda||\mathbf{L}||_* +
    \frac{\rho_L}{2}\Big|\Big|\mathbf{X}-\mathbf{L}\Big|\Big|_2^2 =
    \mathrm{prox}_{||.||_*,\lambda/\rho_L}(\mathbf{X})
    \\
    \ \mathrm{avec}\ \ \mathrm{prox}_{||.||_*,\lambda/\rho_L}(x)=
    \mathcal{T}_{\lambda/\rho_L}\left(x\right)`

    Basée sur [2]_

    Parameters
    ----------
    M : float
        tenseur de dimension Nx,Ny,K
    i : int
        i ème couche à seuiller (0<i<K)
    lambdaS : float
        Paramètre de parcimonie (si existant >0)
    rho : float
        Paramètre de pénalité >0

    Returns
    -------
    L : float
        Résultat de la minimisation pour une couche i
    """
    [u_temp,Sig_temp,v_temp] = linalg.svd((M)[:,:,i],check_finite=False)
    Sig_temp = diag_thresh(u_temp.shape[0],u_temp.shape[0],Sig_temp)
    L = (u_temp@threshold(Sig_temp,(lambdaS/rho),'soft')@v_temp)
    return L


def update_rhoLS_adp(er_prim,er_dual,rhoLS):
    r"""Fonction d'adaptation des paramètres de pénalités dans
    l'`ADMM` à partir des erreurs faites sur le primal et le dual.
    Basée sur [3]_

    Parameters
    ----------
    er_prim : float
        erreur du primal
    er_dual : float
        erreur du dual
    rhoLS : float
        valeur du paramètre à actualiser

    Returns
    -------
    rhoLS : float
        paramètre de pénalité actualisé
    k : float
        facteur de dilation/contraction utilisé
    """
    if (er_prim>10*er_dual)&(rhoLS<1e5):
        k=2
        rhoLS = k*rhoLS
    elif (er_dual>10*er_prim)&(rhoLS<1e5):
        k=0.5
        rhoLS = k*rhoLS
    else:
        k=1
    return rhoLS,k


def Sherman_MorrisonF(DF_H, b, c, rho):
    r"""Solve a diagonal block linear system with a scaled identity 
    term using the Sherman-Morrison equation

    The solution is obtained by independently solving a set of linear 
    systems of the form (see wohlberg-2015-efficient)

    :math:`(a\cdot a^H +\rho I)x = b`

    In this equation inner products and matrix products are taken along
    the 3rd dimension of the corresponding multi-dimensional arrays; the
    solutions are independent over the 1st and 2nd (and 4th, if 
    non-singleton) dimensions.

    Parameters
    ----------
    DF_H :complex
        Conjugate of Multi-dimensional array containing :math:`a^H`
    b :complex
        Multi-dimensional array containing b
    c :complex
        Multi-dimensional array containing pre-computed quantities :math:`a^H/(a^H\cdot a +\\rho)`
    rho :float
        Scalar rho

    Returns
    -------
    x :complex
        Multi-dimensional array containing linear system solution

    Notes
    -----
    Adapted from matlab code : Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-12-18 
    """
    cb = np.sum(c * b, 2, keepdims=True)
    # cb=np.repeat(cb[:,:,np.newaxis],K,axis=2)
    cba = cb * DF_H
    x = (b - cba) / rho
    return x


def diag_thresh(m, n, s):
    r"""Retourne une matrice diagonale de dimension mxn à partir d'un vecteur s
    Utilisation après `np.linalg.svd` pour obtenir une matrice à partir de S


    Parameters
    ----------
    m :int
        dimension verticale de la matrice souhaitée
    n :int
        dimension horizontale de la matrice souhaitée
    s :float
        vecteur à diagonaliser

    Returns
    -------
    out :float
        Matrice diagonale (à partir de S)    
    """
    q = np.abs(n - m)
    np.diag(s, q)[:, q:]
    if m < n:
        return np.diag(s, q)[:, q:].T
    else:
        return np.diag(s, q)[:, q:]

def SVD_gpr(ref,rank):
    """ Perform a SVD on the reference image and dump the first n rank singular values

    Parameters
    ----------
    ref :  ndarray
        reference image
    rank : int
        rank of the SVD

    Returns
    -------
    A_remake : ndarray
        Reconstructed reference image without the rank-n singular values
         
    """
    U, D, VT = np.linalg.svd(ref, full_matrices=False)
    D[:rank]=0
    diag_D = diag_thresh(U.shape[0],VT.shape[0],D)
    A_remake = (U @ diag_D @ VT)
    return A_remake


def roll_fft(alpha, t, x):
    r"""Correction de la position des coefficients des cartes en fonctions
    de la position centrale des hyperboles + Sommation des C_k

    Parameters
    ----------
    alpha :float
        Tenseur complexe (M x N x K) des cartes de coefficients
    t :int
        position (pixel) centrale des hyperboles utilisées (ordonnée).
    x :int
        position (pixel) centrale des hyperboles utilisées (abscisse).

    Returns
    -------
    out :float
        Matrice corrigée et réduite (M x N)       
    """
    Q = np.real(np.sum(alpha, axis=2))
    Q = np.roll(Q, t, axis=0)
    Q = np.roll(Q, -x, axis=1)
    return Q

class ConvolutionalSparseCoding(BaseEstimator):
    """Base classe for convolutional sparse coding for image processing applications
    Attributes
    ----------
    dictionary : array_like of shape (n_pixelsx, n_pixelx_y, n_atoms)
        Dictionary for sparse coding.
    """ 
    def __init__(self, dictionary) -> None:        
        super().__init__()
        self._set_dictionary(dictionary)


    def _set_dictionary(self, dictionary):
        """Setting the internal dictionary.
        Parameters
        ----------
        dictionary : array_like of shape (n_pixelsx, n_pixelx_y, n_atoms)
        Dictionary for sparse coding.
        Raises
        ------
        AttributeError
            When the dimesnion of the dictionary is not 3.
        """        
        #check_array(dictionary)
        if dictionary.ndim!=3:
            raise AttributeError(f"Dimension of array {dictionary.ndim} != 3")
        self.dictionary = dictionary