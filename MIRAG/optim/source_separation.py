"""
References
----------
.. [5] 'Sparse Decomposition of the GPR Useful Signal from Hyperbola Dictionary',
       Guillaume Terasse, Jean-Marie Nicolas, Emmanuel Trouvé and Émeline Drouet
       Avalaible at: https://hal.archives-ouvertes.fr/hal-01351242

.. warning::
    **A faire**

    - Terminer la documentation de la classe (pour tous les attributs)
    - Ajouter le logging pour le suivi
    - Ajout condition d'arrêt sur la variation des paramètres en plus du nombre d'itération
    - A vérifier, doute sur la reconstruction : :math:`\sum_{k}\mathbf{H}_k\star \mathbf{C}_k` ou :math:`\sum_{k}\mathbf{H}_k\star \mathbf{S}_k`
"""

import logging

import numpy as np
from tqdm import tqdm
from scipy import linalg,fft
from sklearn.utils.validation import _deprecate_positional_args
from pywt import threshold,threshold_firm

from . import admm_func


def cost_function_addm2(Y,rhoS,rhoL,old_auxiliary,old_dal,old_L,L,dal,primal,auxiliary):
    r"""Fonction de calcul des coûts et erreurs de l'`ADMM`
    à 2 contraintes.

    Parameters
    ----------
    Y : float
        image originale (Nx,Ny,1)
    rhoS : float
        paramètre de pénalité sur la variable S
    rhoL : float
        paramètre de pénalité sur la variable L
    old_auxiliary : float
        variable auxiliaire de l'itération précedente :math:`\mathbf{S}_k^{i-1}`
    old_dal : float
        reconstruction de l'itération précedente :math:`\sum_k{\mathbf{C}_k^{i-1}\star\mathbf{H}_k}^{i-1}`
    old_L : float
        variable de la matrice creuse de l'itération précedente :math:`\mathbf{L}^{i-1}`
    L : float
        variable de la matrice creuse  :math:`\mathbf{L}`
    dal :float
        reconstruction
    primal : float
        variable primal :math:`\mathbf{C}_k^i`
    auxiliary : float
        variable auxiliaire :math:`\mathbf{S}_k^{i-1}`

    Returns
    -------
    var_prim_:float
        variation du primal
    error_rec_:float
        erreur de reconstruction 
    error_primal_:float
        erreur du primal
    error_dual_S:float
        erreur du dual de S :math:`\mathbf{U_S}`
    error_dual_L:float
        erreur du dual de L :math:`\mathbf{U_L}`
    """
    var_prim_ = np.linalg.norm(dal-old_dal,'fro',axis=(0,1))[0]
    error_rec_ = np.linalg.norm(Y-dal-L,'fro',axis=(0,1))[0]
    error_primal_ = np.linalg.norm(np.sum(primal-auxiliary,2))
    error_dual_L = np.linalg.norm(np.sum(-rhoL*(L-old_L),2))
    error_dual_S = np.linalg.norm(np.sum(-rhoS*(auxiliary-old_auxiliary),2))
    return var_prim_,error_rec_,error_primal_,error_dual_S,error_dual_L
   
def addm_iteration_norm2(lambdaS,c,Y,Dal,DF,DF_H,Us,Uy,rhoS,rhoL,S,dim1,over_relax,penalty="l1",m=-1):
    r"""fonction de calcul d'une itération d'`ADMM` avec la norme 2 et la matrice creuse
    Basé sur [4]_ et [5]_

    Parameters
    ----------
    lambdaS : float
        paramètre de parcimonie
    c : float
        pre-calcul du terme pour Sherman_MorrisonF 
    Y : float
        image originale
    Dal : float
        produit de convolution dictionnaire + carte coeff (Nx,Ny,1)
    DF : complex
        fft du dictionnaire (Nx,Ny,K)
    DF_H : complex
        conjugué de la fft du dictionnaire (Nx,Ny,K)
    Us : float
        variable duale de S
    Uy : float
        variable duale de L
    rhoS : float
        paramètre de pénalité sur S
    rhoL : float
        paramètre de pénalité sur L
    S : float
        variable auxiliaire :math:`\mathbf{S}_k`
    dim1 : int
        dimensions de l'image (ex: [256,256,1])
    over_relax : float
        paramètre d'over-relaxation (améliore la convergence pour :math:`\alpha\sim 1.6`)
    penalty : str{"l1","l0","FirmThresholding"}, optional
        penalité d'attache aux données, de base :math:`\sum_k{||\mathbf{S}_k||_1`
    m : int{-1}, optional
        Nombre de workers (coeurs) utilisé pour la fft

    Returns
    -------
    primal_new:float
        variable primal :math:`\mathbf{C}_k^i` (Nx,Ny,K)
    auxiliary_new:float
        variable dual :math:`\mathbf{S}_k^i` (Nx,Ny,K)
    L: float
        matrice creuse (Nx,Ny,1)
    Dal: float
        produit de convolution dictionnaire + carte coeff ?:math:`\sum_k{\mathbf{C}_k^i\star\mathbf{H}_k^i}`?  (Nx,Ny,1)
    """
    [u,Sig,v] = linalg.svd((Y-Dal+Uy)[:,:,0],check_finite=False)
    Sig = admm_func.diag_thresh(u.shape[0],v.shape[0],Sig)
    L = (u@threshold(Sig,(1/rhoL),'soft')@v).reshape(dim1)

    x_b = fft.fft2((Y-L+Uy),axes=(0,1),workers=m)
    z_b = fft.fft2(S-Us,axes=(0,1),workers=m)
    b = rhoL*DF_H*x_b + rhoS*z_b   
    alphaf = admm_func.Sherman_MorrisonF(DF_H,b,c,rhoS)

    primal_new = fft.ifft2(alphaf,axes=(0,1),workers=m)

    if over_relax>0:
        primal_new = over_relax*primal_new-((over_relax-1)*S)

    #update S => add to admm func
    if penalty=="l1":
        auxiliary_new = threshold(primal_new+Us,lambdaS/rhoS, 'soft')
    elif penalty=="l0":
        auxiliary_new = threshold(primal_new+Us,lambdaS/rhoS, 'hard')
    elif penalty=="FT":
        auxiliary_new = threshold_firm(primal_new+Us,lambdaS/rhoS,2)
    
    #SF = fft.fft2(auxiliary_new,axes=(0,1),workers=m)
    #Dal = fft.ifft2(np.sum(DF*SF,2,keepdims=True),axes=(0,1),workers=m)
    Dal = fft.ifft2(np.sum(DF*alphaf,2,keepdims=True),axes=(0,1),workers=m)
    return  primal_new, auxiliary_new, L, Dal

    
class ADMMSourceSep(admm_func.ConvolutionalSparseCoding):
    r"""Convolutional sparse coding with ADMM algorithm for image processing applications.

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
    def __init__(self, dictionary, eps, n_iter, delta, rhoS, rhoL,over_relax=0,
                update_rho="adaptive", penalty="l1", norm_optim="Frobenius", save_iterations=False,
                verbosity=0) -> None:         
        super().__init__(dictionary)
        self.eps = eps
        self.normalize = True
        self.over_relax = over_relax
        self.penalty = penalty
        self.norm_optim = norm_optim
        self.max_grad_iter = 50
        self.alpha = 0.001
        self.threshold_H = 250
        self.n_iter = n_iter
        self.iter_init_ = n_iter
        self.delta = delta
        self.rhoS = rhoS
        self.rhoL = rhoL
        self.workers = -1
        self.update_rho = update_rho
        self.save_iterations = save_iterations
        self.verbosity = verbosity
        self.converged_ = False
        self.iterations_ = None
        self.error_primal_ = []
        self.error_dual_S_ = []
        self.error_dual_L_ = []
        self.error_rec_ = []
        self.var_prim_ = []
        self.auxiliary_ = 0
        self.primal_ = 0
        self.dual_S = 0
        self.dual_L = 0
        self.dal_ = 0
        self.L_ = 0
        self.c_ = 0
        self.iterations_save_ = []

    def _update_c(self):
        self.c_ = self.rhoL*self.DF_/(self.rhoS + self.rhoL*self.c_inter)
    
    def _update_rho(self):
        if self.update_rho == "adaptive":
            self.rhoL,k = admm_func.update_rhoLS_adp(self.error_primal_[-1],self.error_dual_L_[-1],self.rhoL)
            self.rhoS,k = admm_func.update_rhoLS_adp(self.error_primal_[-1],self.error_dual_S_[-1],self.rhoS)
            if self.norm_optim=="Frobenius":
                self._update_c()
        else:
            if self.update_rho == "increase":
                self.rhoL = 1.1*self.rhoL
                self.rhoS = 1.1*self.rhoS
                if self.norm_optim=="Frobenius":
                    self._update_c()
                k=1.1
            else:
                k=1
        self.dual_L = (self.dual_L + self.Y-self.dal_-self.L_)/k
        self.dual_S = (self.dual_S + self.primal_-self.auxiliary_)/k

    def _iteration_addm(self):
        if self.norm_optim=="Frobenius":
            primal_new, auxiliary_new, L_new, dal_new = addm_iteration_norm2(
                                                                      self.eps, self.c_, self.Y, self.dal_, self.DF_, self.DF_H_,
                                                                      self.dual_S,self.dual_L, self.rhoS, self.rhoL, self.auxiliary_, self.dim1,
                                                                      self.over_relax, penalty = self.penalty, m=self.workers
                                                                    )  
        else:
            logging.warning("Error norme")
        self._save_iteration(L_new, primal_new, auxiliary_new, dal_new)
        

    def _cost_function(self, L, primal, auxiliary, dal):
        cost = cost_function_addm2(self.Y,self.rhoS,self.rhoL,self.auxiliary_,self.dal_,self.L_,L,dal,primal,auxiliary)
        self.var_prim_.append(cost[0])
        self.error_rec_.append(cost[1])
        self.error_primal_.append(cost[2])
        self.error_dual_S_.append(cost[3])
        self.error_dual_L_.append(cost[4])
        

    def _save_iteration(self, L, primal, auxiliary, dal):
        self._cost_function(L, primal, auxiliary, dal) 
        self.primal_ = primal
        self.auxiliary_ = auxiliary
        self.dal_ = dal
        self.L_ = L
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

        self.dim1 = (X.shape[0],X.shape[1],1)
        self.dimK = self.dictionary.shape
        self.dual_L = np.zeros(self.dim1)
        self.dual_S = np.zeros(self.dimK)
        self.auxiliary_ = np.zeros(self.dimK)
        self.primal_tilde_ = np.zeros(self.dimK)
            
        self.S_tilde = fft.fft2(X,axes=(0,1),workers=-1)

        self.Y = X.reshape(self.dim1)
        if self.normalize:
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
            self.DF_H_ = np.conj(self.DF_)
            self.YF = fft.fft2(self.Y,axes=(0,1),workers=self.workers)
            if self.norm_optim=="Frobenius":
                self.c_inter = np.sum(self.DF_H_*self.DF_,2,keepdims=True)
                self._update_c()

    def fit(self, X, y=None, initial_solution=None):
        X = self._validate_data(X)
        self._initialize_for_estimation(X,initial_solution)
        pbar = tqdm(total = self.n_iter,leave=True)
        pbar.n = self.iterations_
        while not self.converged_:
            self._iteration_addm()
            info = f"rec : {float(self.error_rec_[-1]):.4}  ||duaS :  {float(self.error_dual_S_[-1]):.3} ||duaL :  {float(self.error_dual_L_[-1]):.3} ||pri :  {float(self.error_primal_[-1]):.4} ||rhoS :  {float(self.rhoS):.3} ||rhoL :  {float(self.rhoL):.3}"
            pbar.set_description(info)
            pbar.update(1)
            self.iterations_ += 1
            # ajout max alpha variation + L variation
            #cond_erro = np.max([self.error_rec_[-1], self.error_dual_[-1]]) < self.delta
            cond_iter = self.iterations_ >= self.n_iter
            self.converged_ = cond_iter #or cond_erro
        pbar.close()
    
        return self