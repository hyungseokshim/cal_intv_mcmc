import numpy as np
from scipy.stats import norm

####################################################################
######################## Reliability models ########################
####################################################################

def mod_weibu(th, t):
    ''' Weibull model with two parameters, th[0] and th[1]'''
    ''' th[0] : beta, th[1] : alpha'''
    return np.exp(-((1/th[0])*t)**th[1])
 
def mod_lognorm(th, t):
    ''' Lognormal model with two parameters,th[0] and th[1]'''
    ''' th[0] : mu, th[1] : sigma'''
    return 1.-norm.cdf((np.log(t)-th[0])/th[1])