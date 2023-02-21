'''
File: svd_inversion.py
Created Date: Thu Jan 01 1970
Author: Ammar Mian
-----
Last Modified: Fri Jan 13 2023
Modified By: Ammar Mian
-----
Copyright (c) 1970-2023 Université Savoie Mont-Blanc
-----
Method where we use SVD to filter rank 1 (=horizontal) patterns
'''
from MIRAG.filtrage_func import SVD_gpr
from sklearn.base import BaseEstimator


class SVDInversion(BaseEstimator):
    def __init__(self, dictionary, rank=1) -> None:
        self.rank = rank
    
    def fit(self, image):
        self.denoised_image = SVD_gpr(image, self.rank)
        return self

    def get_estimate(self):
        return self.denoised_image

estimator = SVDInversion