import numpy as np
from tqdm import tqdm
from scipy import fft
from sklearn.utils.validation import _deprecate_positional_args
from pywt import threshold

from . import admm_func    
class ADMMSourceSepHUB(admm_func.ConvolutionalSparseCoding):
    r""" ADMM algorithm for the source separation problem with the Huber function.

    Attributes
    ----------
    dictionary : array_like of shape (n_pixelsx, n_pixelx_y, n_atoms)
        Dictionary for sparse coding.
    eps : float
        sparsity coeficient. must be greater than 0.
    n_iter :  int
        number of iterations.
    delta :  float
        tolerance for the stopping criterion.
    rhoS : float
        regularization parameter for the source.
    rhoL : float
        regularization parameter for L (low-rank matrix).
    update_rho : str
        type of update for rho.
    save_iterations : bool
        if True, save the iterations.
    max_grad_iter : int
        maximum number of iterations for the gradient for the primal variable.
    alpha : float
        step size for the gradient.    
    threshold_H : float
        threshold for the Huber function.
    """    
    @_deprecate_positional_args
    def __init__(self, dictionary, eps, n_iter, delta, rhoS, rhoL,
                 update_rho="adaptive", save_iterations=False, max_grad_iter = 5,
                 alpha = 5E-4, threshold_H = 500, verbosity=0) -> None:         
        super().__init__(dictionary)
        self.eps = eps
        self.normalize = True
        self.max_grad_iter = max_grad_iter
        self.alpha = alpha
        self.threshold_H = threshold_H
        self.n_iter = n_iter
        self.iter_init_ = n_iter
        self.delta = delta
        self.rhoS = rhoS
        self.rhoL = rhoL
        self.workers = -1
        self.update_rho = update_rho
        self.save_iterations = save_iterations
        self.converged_ = False
        self.iterations_ = None
        self.error_primal_C_ = []
        self.error_primal_L_ = []
        self.error_dual_M_ = []
        self.error_dual_S_ = []
        self.error_rec_ = []
        self.auxiliary_S = 0
        self.auxiliary_M = 0
        self.primal_ = 0
        self.dual_S = 0
        self.dual_L = 0
        self.dal_ = 0
        self.L_ = 0
        self.iterations_save_ = []
        self.verbosity = verbosity

    def get_estimate(self):
        return self.dal_

    def _cost_function_huber(self):
            self.error_primal_C_.append(np.linalg.norm(np.sum(self.primal_-self.auxiliary_S,2)))
            self.error_primal_L_.append(np.linalg.norm(np.sum(self.L_-self.auxiliary_M,2)))
            self.error_dual_M_.append(np.linalg.norm(np.sum(-self.rhoL*(self.auxiliary_M-self.old_M),2)))
            self.error_dual_S_.append(np.linalg.norm(np.sum(-self.rhoS*(self.auxiliary_S-self.old_S),2)))
            self.error_rec_.append(np.linalg.norm(self.Y-self.dal_-self.L_,'fro',axis=(0,1)))

    def _update_c(self):
        Z_m_tilde = fft.fft2(self.dual_S-self.auxiliary_S,axes=(0,1),workers=self.workers)
        S_tilde = fft.fft2(self.L_-self.Y,axes=(0,1),workers=self.workers)
        for i in range(self.max_grad_iter):
            self.alphaf = self.alphaf - (self.alpha/(i+1))*admm_func.gradient_huber(S_tilde, self.alphaf, 
                                                           self.DF_, self.DF_H_, Z_m_tilde, self.rhoS, self.threshold_H)      
        self.primal_ = fft.ifft2(self.alphaf,axes=(0,1),workers=self.workers)
        self.dal_ = fft.ifft2(np.sum(self.DF_*self.alphaf,2,keepdims=True),axes=(0,1),workers=self.workers)        
    
    def _update_rho(self):
        self.rhoL = self.rhoL
        self.rhoS = self.rhoS

    def _update_m(self):
        #update M
        self.old_M = self.auxiliary_M
        [u,Sig,v] = np.linalg.svd((self.dual_L+self.L_)[:,:,0])
        Sig = admm_func.diag_thresh(u.shape[0],v.shape[0],Sig)
        self.auxiliary_M = (u@threshold(Sig,(1/self.rhoL),'soft')@v).reshape(self.dim1)#soft

    def _update_l(self):
        #update L
        B = (self.dal_-self.Y)
        self.L_ = admm_func.proxH(self.auxiliary_M-self.dual_L+B,self.threshold_H,1/self.rhoL)-B
    
    def _update_s(self):
        #update S
        self.old_S=self.auxiliary_S
        self.auxiliary_S = threshold(self.primal_+self.dual_S,self.eps/self.rhoS, 'soft')

    def _update_u(self):
        self.dual_S = (self.primal_-self.auxiliary_S +self.dual_S)
        self.dual_L = (self.L_-self.auxiliary_M+self.dual_L)

    def _iteration_addm_huber(self):
        self._update_c()
        self._update_l()
        self._update_s()
        self._update_m()
        self._update_u()
        self._cost_function_huber()

    def _initialize_for_estimation(self,X):
        self.dim1 = (X.shape[0],X.shape[1],1)
        self.dimK = self.dictionary.shape
        self.dual_L = np.zeros(self.dim1)
        self.dual_S = np.zeros(self.dimK)
        self.auxiliary_ = np.zeros(self.dimK)
        self.alphaf = np.zeros(self.dimK)
        self.Y = X.reshape(self.dim1)
        if self.normalize:
            self.Y = (self.Y-self.Y.min())/(self.Y.max()-self.Y.min())
        self._precompute()
        self.converged_ = False
        self.iterations_ = 0
        if self.save_iterations:
            self.iterations_save_ = []

    def _precompute(self):
            self.DF_ = fft.fft2(self.dictionary,axes=(0,1),workers=self.workers)
            self.DF_H_ = np.conj(self.DF_)

    def fit(self, X, y=None):
        X = self._validate_data(X)
        self._initialize_for_estimation(X)
        if self.verbosity >0:
            pbar = tqdm(total = self.n_iter,leave=True)
            pbar.n = self.iterations_
        while not self.converged_:
            self._iteration_addm_huber()
            info = f"rec : {float(self.error_rec_[-1]):.4}  ||duaS :  {float(self.error_dual_S_[-1]):.3} ||duaM :  {float(self.error_dual_M_[-1]):.3} ||priC :  {float(self.error_primal_C_[-1]):.4} ||priL :  {float(self.error_primal_L_[-1]):.4}"
            if self.verbosity >0:
                pbar.set_description(info)
                pbar.update(1)
            self.iterations_ += 1
            cond_iter = self.iterations_ >= self.n_iter
            self.converged_ = cond_iter
        if self.verbosity >0:
            pbar.close()
        return self
