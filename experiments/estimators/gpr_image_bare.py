'''
File: gpr_image_bare.py
Created Date: Thu Jan 01 1970
Author: Ammar Mian
-----
Last Modified: Fri Jan 13 2023
Modified By: Ammar Mian
-----
Copyright (c) 1970-2023 Université Savoie Mont-Blanc
-----
Method where we don't do anything and just return original image
'''
from sklearn.base import BaseEstimator

class GPRImage(BaseEstimator):
    def __init__(self, dictionary, **kwargs) -> None:
        super().__init__()

    def fit(self, image):
        self.denoised_image = image
        return self

    def get_estimate(self):
        return self.denoised_image

estimator = GPRImage

