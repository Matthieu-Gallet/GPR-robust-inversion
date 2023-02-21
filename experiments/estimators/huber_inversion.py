'''
File: huber_inversion.py
Created Date: Thu Jan 01 1970
Author: Ammar Mian
-----
Last Modified: Fri Jan 13 2023
Modified By: Ammar Mian
-----
Copyright (c) 1970-2023 Université Savoie Mont-Blanc
-----
Inverison with Huber norm and clutter matrix
'''

from MIRAG.optim.huber_source_separation import ADMMSourceSepHUB
estimator = ADMMSourceSepHUB