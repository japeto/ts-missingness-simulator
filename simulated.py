# -*- coding: utf-8 -*-

# Load libraries
# %%
import numpy as np
# import matplotlib
from datetime import datetime
from simulator import generator
from missingness import *

# matplotlib.use('Agg')
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure, title

############### simular serie

# %%
# 1 serie with 4 segments, with a functional part, 100 points. We 
# one variance for the error term.
M = 1
n = 100 
K = 4
sigma_ = 0.1 # Plantear el mismo escenario del paper

# %%
muChoix = np.array([0, 1, 2, 3, 4, 5])
standard_deviation = np.array([sigma_])
varianceError = np.power(standard_deviation, 2)
pHaar = np.array([10, 50, 60])
p1 = 3
p2 = 5
nbsimu = 1
series = generator(M, n, K, muChoix, varianceError, pHaar, p1, p2)

# %%
# Obtaining the serie 
data_serie = np.array((series[5]["mu"] + series[5]["biais"] + series[5]["erreur1"]))
data_serie = data_serie.reshape(-1, 1)

# %%
na_serie = mcar_method(data_serie, percentage=90)
# mar_method(data_serie, 70)

# %%
now = datetime.now()
date = now.strftime("%d-%m-%Y")

# %%
np.savetxt("data/simlated-na-{}.csv".format(str(date)), na_serie, delimiter=",")
