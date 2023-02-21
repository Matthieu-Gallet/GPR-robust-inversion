'''
File: svd_l2_inversion.py
Created Date: Thu Jan 01 1970
Author: Ammar Mian
-----
Last Modified: Fri Jan 13 2023
Modified By: Ammar Mian
-----
Copyright (c) 1970-2023 Université Savoie Mont-Blanc
-----
SVD + L2 inversion without clutter matrix
'''

from sklearn.base import BaseEstimator
from MIRAG.filtrage_func import SVD_gpr
from MIRAG.optim.sparse_coding import ADMMSparseCoding

class SVDL2Inversion(BaseEstimator):
    def __init__(self, dictionary, rank, eps, n_iter, delta, rho, 
                update_rho="adaptive", penalty="l1", norm_optim="Frobenius", save_iterations=False,
                verbosity=0) -> None:
        self.L2Inversion = ADMMSparseCoding(dictionary, eps, n_iter, delta, rho, 
                update_rho, penalty, norm_optim, save_iterations,
                verbosity)
        self.rank = rank

    def fit(self, image):
        image_after_SVD = SVD_gpr(image, self.rank)
        self.L2Inversion.fit(image_after_SVD)
        return self

    def get_estimate(self):
        return self.L2Inversion.dal_

estimator = SVDL2Inversion
