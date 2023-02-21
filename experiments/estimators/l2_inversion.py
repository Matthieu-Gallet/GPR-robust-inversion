'''
File: l2_inversion.py
Created Date: Thu Jan 01 1970
Author: Ammar Mian
-----
Last Modified: Fri Jan 13 2023
Modified By: Ammar Mian
-----
Copyright (c) 1970-2023 Université Savoie Mont-Blanc
-----
Method where we use L2-inversion with clutter matrix
'''



from MIRAG.optim.source_separation import ADMMSourceSep
estimator = ADMMSourceSep