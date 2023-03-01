# -*- coding: utf-8 -*-
"""
Create Fri Aug 07 17:23:45
"""

import numpy as np
import pandas as pd
import random
##########################
### Missingness methods
##########################
# mcar_method
# 
def mcar_method(serie=[], percentage=70, seed=348):
    """
    MCAR missing data mechanism
    If the probability of being missing is the same for all values, 
    then could be perform a missing completely at random (MCAR).
    Parameters:
        serie (array): serie values
        percentage (float): percentage of missing values
        seed (int): Initialize the random number generator
    """
    np.random.seed(seed)
    serie = serie.flatten()
    percentage = percentage/100 if percentage > 1 else percentage
    indices = np.random.randint(0, len(serie), size=round(percentage*serie.shape[0]))
    diff = round(percentage*serie.shape[0]) - len(set(indices))
    if diff > 0:
        np.append( indices, np.random.randint(diff) )
    serie = serie.copy()
    for idx in indices:
        serie[idx] = np.nan
    return serie


# mar_method
# 
def mar_method(serie=[], percentage=70, condition='less', threshold=None, seed=348):
    """
    MAR missing data mechanism
    If the probability of being missing can be conditioned,
    then could be perform a missing at random (MAR).
    Parameters:
        serie (array): serie values
        percentage (float): percentage of missing values
        seed (int): Initialize the random number generator
    """
    np.random.seed(seed)
    serie = serie.flatten()
    percentage = percentage/100 if percentage > 1 else percentage
    threshold =  threshold if threshold else np.average(serie)
    th_serie = serie[serie <= threshold] if condition == 'less' else serie[serie >= threshold]
    indices = np.random.randint(0, len(th_serie), size=round(percentage*th_serie.shape[0]))
    diff = round(percentage*th_serie.shape[0]) - len(set(indices))
    if diff > 0:
        np.append( indices, np.random.randint(diff) )
    
    for idx in indices:
        th_serie[idx] = np.nan

    if condition == 'less':
        serie = np.append(th_serie, serie[serie >= threshold])
    else:
        serie = np.append(serie[serie <= threshold], th_serie)
    return serie