# Problem Solving Module for RPM
"""
References
----------
.. [4] 'Convolutional dictionary learning: A comparative review and new algorithms',Garcia-Cardona, C., & Wohlberg, B. (2018).
        Avalaible at: https://arxiv.org/pdf/1709.02893.pdf

.. warning::
    **To do**

    - Complete class documentation (for all attributes)
    - Add logging for tracking
    - Check dictionary strucure such as: :math:`D^H= \mathrm{conjugate}(D)`
    - Add over-relaxation parameter (see `source_separation` and [3]_)
"""


import logging

import numpy as np
from tqdm import tqdm
from scipy import fft

from sklearn.utils.validation import _deprecate_positional_args
from pywt import threshold,threshold_firm

from . import admm_func


def cost_function_addm1(Y,rho,old_auxiliary_,old_dal,dal,primal,auxiliary):
    r"""ADMM cost and error calculation function

    Parameters
    ----------
    Y : float
        original image (Nx,Ny,1)
    rho : float
        penalty parameter
    old_auxiliary_ : float
        auxiliary variable of the previous iteration :math:`\mathbf{S}_k^{i-1}`
    old_dal : float
        reconstruction of the previous iteration :math:`\sum_k{\mathbf{C}_k^{i-1}\star\mathbf{H}_k}^{i-1}`
    dal : float
        convolution product dictionary + coeff map :math:`\sum_k{\mathbf{C}_k^i\star\mathbf{H}_k^i}`
    primal : float
        variable primal :math:`\mathbf{C}_k^i`
    auxiliary : float
        auxiliary variable :math:`\mathbf{S}_k^{i-1}`

    Returns
    -------
    out : list of float
        variation of primal, reconstruction error, error of primal and error of dual
    """
    var_prim_ = np.linalg.norm(dal-old_dal,'fro',axis=(0,1))[0]
    error_rec_ = np.linalg.norm(Y-dal,'fro',axis=(0,1))[0]
    error_primal_ = np.linalg.norm(np.sum(primal-auxiliary,2))
    error_dual_ = np.linalg.norm(np.sum(-rho*(auxiliary-old_auxiliary_),2))
    return var_prim_,error_rec_,error_primal_,error_dual_


def addm_iteration_norm2(lambdaS,c,DHY,DF,DF_H,U,rho,S,penalty="l1",m=-1):
    r"""function for computing an iteration of `ADMM` with the 2 norm
    Based on [4]_

    Parameters
    ----------
    lambdaS : float
        parsimony parameter 
    c : float
        pre-computation of the term for Sherman_MorrisonF 
    DHY : complex
        fft from dictionary * original image
    DF : complex
        fft of the dictionary
    DF_H : complex
        conjugate of the dictionary fft
    U : float
        dual variable
    rho : float
        penalty parameter
    S : float
        auxiliary variable :math:`\mathbf{S}_k`
    penalty : str{"l1", "l0", "FirmThresholding", "l*"}, optional
        data attachment penalty, basic :math:`\sum_k{||\mathbf{S}_k||_1`
    m : int{-1}, optional
        Number of workers (cores) used for the fft

    Returns
    -------
    primal_new:float
        variable primal :math:`\mathbf{C}_k^i` (Nx,Ny,K)
    auxiliary_new:float
        dual variable :math:`\mathbf{S}_k^i` (Nx,Ny,K)
    Dal: float
        convolution product dictionary + map coeff :math:`\sum_k{\mathbf{C}_k^i\star\mathbf{H}_k^i}` (Nx,Ny,1)
    """
    b = DHY + rho*fft.fft2(S-U,axes=(0,1),workers=m)
    alphaf = admm_func.Sherman_MorrisonF(DF_H,b,c,rho)
    primal_new = fft.ifft2(alphaf,axes=(0,1),workers=m)

    #update S => add to admm func
    if penalty=="l1":
        auxiliary_new = threshold(primal_new+U, lambdaS/rho, 'soft')
    elif penalty=="l0":
        auxiliary_new = threshold(primal_new+U, lambdaS/rho, 'hard')
    elif penalty=="FirmThresholding":
        auxiliary_new = threshold_firm(primal_new+U, lambdaS/rho,2)
    elif penalty=="l*":
        M = primal_new+U
        auxiliary_new = np.zeros_like(primal_new)
        for i in range(primal_new.shape[2]):
            auxiliary_new[:,:,i]=admm_func.thresholding_nuclear(M,i,lambdaS,rho)

    Dal = fft.ifft2(np.sum(DF*alphaf,2,keepdims=True),axes=(0,1),workers=m)

    return  primal_new, auxiliary_new, Dal


def addm_iteration_normH(lambdaS,DF, DFH, U, rho, S, primal_tilde, YF,
                         grad_iter_max=50, beta=0.001, thresh=250, penalty="l1", m=-1):
    r"""function for computing an iteration of `ADMM` with the HUber norm

    Parameters
    ----------
    lambdaS : float
        parsimony parameter
    DF : complex
        fft of the dictionary (Nx,Ny,K)
    DF_H : complex
        conjugate of the dictionary fft
    U : float
        dual variable (Nx,Ny,K)
    rho : float
        penalty parameter
    S : float
        auxiliary variable :math:`\mathbf{S}_k` (Nx,Ny,K)
    primal_tilde : float
        fft of the primal :math:`\mathbf{C}_k` (Nx,Ny,K)
    YF : float
        fft of the original image (Nx,Ny,1)
    grad_iter_max{50} : int, optional
        maximum iteration of the gradient descent
    beta : float{0.001}, optional
        learning rate parameter of the gradient 
    thresh : int, optional
        [description], by default 250
    penalty : str{"l1", "l0", "FirmThresholding", "l*"}, optional
        penalty of attachment to data, basic :math:`\sum_k{||\mathbf{S}_k||_1`
    m : int{-1}, optional
        Number of workers (cores) used for the fft

    Returns
    -------
    primal_new:float
        variable primal :math:`\mathbf{C}_k^i` (Nx,Ny,K)
    primal_tilde:float
        fft of the primal variable :math:`\mathbf{C}_k^i` (Nx,Ny,K)
    auxiliary_new:float
        dual variable :math:`\mathbf{S}_k^i` (Nx,Ny,K)
    Dal: float
        convolution product dictionary + map coeff :math:`\sum_k{\mathbf{C}_k^i\star\mathbf{H}_k^i}` (Nx,Ny,1)
    """
    Z_m_tilde = fft.fft2(S-U,axes=(0,1),workers=m)
    for i in range(grad_iter_max):
        primal_tilde = primal_tilde - (beta/(i+1))*admm_func.gradient_subproblem_x_huber2(YF, primal_tilde, DF, DFH, Z_m_tilde, rho, thresh)
    primal_new = fft.ifft2(primal_tilde,axes=(0,1),workers=m)

    #update S
    if penalty=="l1":
        auxiliary_new = threshold(primal_new+U, lambdaS/rho, 'soft')
    elif penalty=="l0":
        auxiliary_new = threshold(primal_new+U, lambdaS/rho, 'hard')
    elif penalty=="FirmThresholding":
        auxiliary_new = threshold_firm(primal_new+U, lambdaS/rho,2)
    elif penalty=="l*":
        M = primal_new+U
        auxiliary_new = np.zeros_like(primal_new)
        for i in range(primal_new.shape[2]):
            auxiliary_new[:,:,i]=admm_func.thresholding_nuclear(M,i,lambdaS,rho)
    Dal = fft.ifft2(np.sum(DF*primal_tilde,2,keepdims=True),axes=(0,1),workers=m)
    
    return  primal_new, primal_tilde, auxiliary_new, Dal


class ADMMSparseCoding(admm_func.ConvolutionalSparseCoding):
    """Convolutional sparse coding with ADMM algorithm for image processing applications.
    Based on [3]_ and [4]_
    
    Attributes
    ----------
    dictionary : array_like of shape (n_pixelsx, n_pixelx_y, n_atoms)
        Dictionary for sparse coding.
    eps : float
        sparsity coeficient. must be greater than 0.
    n_iter : [type]
        [description]
    delta : [type]
        [description]
    rho : [type]
        [description]
        update_rho : [type], optional
            [description], by default None
    save_iterations:
    verbosity:
    iterations_:
    error_:
    """
    @_deprecate_positional_args
    def __init__(self, dictionary, eps, n_iter, delta, rho, 
                update_rho="adaptive", penalty="l1", norm_optim="Frobenius", save_iterations=True,
                verbosity=0) -> None:         
        super().__init__(dictionary)
        self.eps = eps
        self.penalty = penalty
        self.norm_optim = norm_optim
        self.max_grad_iter = 50
        self.alpha = 1E-5
        self.threshold_H = 250
        self.n_iter = n_iter
        self.iter_init_ = n_iter
        self.delta = delta
        self.rho = rho
        self.workers = -1
        self.update_rho = update_rho
        self.save_iterations = save_iterations
        # todo: avec logging
        self.verbosity = verbosity
        self.converged_ = False
        self.scaling = True
        self.iterations_ = None
        self.error_primal_ = []
        self.error_dual_ = []
        self.error_rec_ = []
        self.var_prim_ = []
        self.primal_ = 0
        self.dual_ = 0
        self.dal_ = 0
        self.c_ = 0
        self.c_inter_ =0
        self.auxiliary_ = 0
        self.iterations_save_ = []

    def _update_ca1(self):
        self.c_= self.DF_/(self.rho+self.c_inter_)

    def _update_rho(self):
        if self.update_rho == "adaptive":
            self.rho,k = admm_func.update_rhoLS_adp(self.error_primal_[-1],self.error_dual_[-1],self.rho)
            if self.norm_optim=="Frobenius":
                self._update_ca1()
        else:
            if self.update_rho == "increase":
                self.rho = 1.1*self.rho
                if self.norm_optim=="Frobenius":
                    self._update_ca1()
                k=1.1
            else:
                k=1
        self.dual_ = (self.dual_+self.primal_-self.auxiliary_)/k

    def _iteration_addm(self):
        if self.norm_optim=="Frobenius":
            primal_new, auxiliary_new, dal_new = addm_iteration_norm2(
                                                                      self.eps, self.c_, self.DHY_, self.DF_, self.DF_H_,
                                                                      self.dual_, self.rho, self.auxiliary_,
                                                                      penalty = self.penalty, m=self.workers
                                                                     )  
        elif self.norm_optim=="Huber":
            primal_new, self.primal_tilde_, auxiliary_new, dal_new = addm_iteration_normH(
                                                                      self.eps, self.DF_, self.DF_H_,
                                                                      self.dual_, self.rho, self.auxiliary_, self.primal_tilde_, self.YF, 
                                                                      grad_iter_max = self.max_grad_iter, beta=self.alpha, thresh=self.threshold_H,
                                                                      penalty = self.penalty, m=self.workers
                                                                     )
        else:
            logging.warning("Error norme")
        self._save_iteration(primal_new, auxiliary_new, dal_new)
        

    def _cost_function(self, primal, auxiliary, dal):
        cost = cost_function_addm1(self.Y,self.rho,self.auxiliary_,self.dal_,dal,primal,auxiliary)
        self.var_prim_.append(cost[0])
        self.error_rec_.append(cost[1])
        self.error_primal_.append(cost[2])
        self.error_dual_.append(cost[3])
        

    def _save_iteration(self, primal, auxiliary, dal):
        self._cost_function(primal, auxiliary, dal)
        self.primal_ = primal
        self.auxiliary_ = auxiliary
        self.dal_ = dal
        self._update_rho()

        if self.save_iterations:
            values = list(self.__dict__.values())
            keys = list(self.__dict__.keys())
            dico = dict(zip(keys, values))
            self.iterations_save_ = dico

    def _set_solution(self, dictionary_solution):
        for i in dictionary_solution.keys():
            vars(self)[i]=dictionary_solution[i]
        self.n_iter = self.iter_init_+self.iterations_

    def _initialize_for_estimation(self,X,initial_solution):
        if self.converged_:
            logging.warning("Overwriting past estimation.")

        dim1 = (X.shape[0],X.shape[1],1)
        dimK = self.dictionary.shape
        self.dual_ = np.zeros(dim1)
        self.auxiliary_ = np.zeros(dimK)
        self.primal_tilde_ = np.zeros(dimK)          

        self.Y = X.reshape(dim1)
        if self.scaling:
            self.Y = (self.Y-self.Y.min())/(self.Y.max()-self.Y.min())
        if initial_solution is not None:
           self._set_solution(initial_solution)
           self.converged_ = False
        else:
            self._precompute()
            self.converged_ = False
            self.iterations_ = 0

        if self.save_iterations:
            self.iterations_save_ = []

    def _precompute(self):
            self.DF_ = fft.fft2(self.dictionary,axes=(0,1),workers=self.workers)
            #self.DF_H_ = np.conj(self.DF_.transpose([0,1,2])) 
            # semble équivalent sans le transpose
            self.DF_H_ = np.conj(self.DF_)
            self.YF = fft.fft2(self.Y,axes=(0,1),workers=self.workers)
            # Opérateur de convolution
            if self.norm_optim=="Frobenius":
                self.DHY_ = self.DF_H_*self.YF
                # Calcul constante pour résolution du 1ere sous-pb
                self.c_inter_ = np.sum(self.DF_H_*self.DF_,2,keepdims=True)
                self._update_ca1()

    def fit(self, X, y=None, initial_solution=None):
        X = self._validate_data(X)
        self._initialize_for_estimation(X,initial_solution)
        if self.verbosity >0:
            pbar = tqdm(total = self.n_iter,leave=True)
            pbar.n = self.iterations_

        while not self.converged_:
            self._iteration_addm()
            if self.verbosity >0:
                info = f"rec : {float(self.error_rec_[-1]):.5}  ||dua :  {float(self.error_dual_[-1]):.5} ||pri :  {float(self.error_primal_[-1]):.5} ||rho :  {float(self.rho):.5}"
                pbar.set_description(info)
                pbar.update(1)
            self.iterations_ += 1
            cond_erro = np.max([self.error_rec_[-1], self.error_dual_[-1]]) < self.delta
            cond_iter = self.iterations_ >= self.n_iter
            self.converged_ = cond_iter or cond_erro
        
        if self.verbosity >0:
            pbar.close()
    
        return self