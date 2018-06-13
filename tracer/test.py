import numpy as np
from scipy.optimize import linear_sum_assignment

np.random.seed(0)
c = np.random.rand(128, 128)

row_ind, col_ind = linear_sum_assignment(c)
