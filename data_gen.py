import numpy as np
from numpy.random import weibull,lognormal, randint, choice
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

####################################################################
########## Random sampling from probability distributions ##########
####################################################################

def rnd_weibu(th, samples=1):
    ''' Random number generation following Weibull model'''
    ''' th[0] : beta, th[1] : alpha'''
    return th[0]* weibull(th[1], samples)

def rnd_lognorm(th, samples=1):
    ''' Random number generation following Lognormal model '''
    ''' th[0] : mu, th[1] : sigma'''
    return lognormal(th[0],th[1], samples) # https://en.wikipedia.org/wiki/Log-normal_distribution


####################################################################
################ Generation of calibration reccords ################
####################################################################

def gen_series_data(thetas, rnd_model, mte_n=200, 
                    cal_no_l=10, cal_no_h=10, 
                    itv_l=1, itv_h=1, two_interval=False,
                    policy='RF'):
    ''' generation of calibration record which follows a specific reliability model'''
    
    t =[];y=[]
    for j in range(mte_n):  
        n = randint(cal_no_l, cal_no_h+1)            # set cal.no for a MTE
        cr = np.zeros((n+1,2), dtype=int)      # set cal. record format
        if two_interval:
            itv = np.cumsum(choice([itv_l, itv_h], n))  # time series 
        else:
            itv = np.cumsum(randint(itv_l, itv_h+1, n)) 
        
        cr[1:,0] = itv       # generation of elasped time axis in calibration record
        
        if policy == 'RA':   # if renewal always policy is applied
            rv = rnd_model(thetas, len(cr)-1) 
            cr[1:,1] = 1 - 2 * ((cr[1:,0]-cr[0:-1,0])< rv) # -1 : In-tolerance & renewal, 1 : out-of-tolerance & renewal
        if policy == 'RF':     
            st = 0
            rv = rnd_model(thetas)
            for i in range(len(cr)-1):
                if (cr[i+1,0]-st)< rv:
                    cr[i+1,1] = 0 # 0 : in tolerance without renewal
                else:
                    cr[i+1,1] = 1 # 1 : out of tolerance & renewal
                    st= cr[i+1,0]
                    rv= rnd_model(thetas)

        t.append(cr[:,0])
        y.append(cr[:,1])
    return np.array(t) , np.array(y)

def generate_all_calibration_history(cdt):
    ''' generate all calibration history for various conditions given by cdt'''
    
    history={}    
    for j, _ in enumerate(cdt):
        ty=[]
        for i, _ in enumerate (cdt[j]['itv_l']):
            t, y = gen_series_data(cdt[j]['theta'], 
                                    rnd_model= cdt[j]['rnd_ftn'], 
                                    mte_n= cdt[j]['mte_n'],
                                    cal_no_l= cdt[j]['cal_n'], 
                                    cal_no_h= cdt[j]['cal_n'], 
                                    itv_l= cdt[j]['itv_l'][i], 
                                    itv_h= cdt[j]['itv_h'][i], 
                                    two_interval= cdt[j]['2itvs'],
                                    policy= cdt[j]['policy'])
            ty.append([t, y])
        history[j]={'ty':ty}
    return history

####################################################################
########## Graphical representation of calibration records #########
####################################################################

def plt_oot(res, j, k, start=0, end=25, xlim= 210):
    '''plot OOT map of a case of calibration record '''
    
    t=res[j]['ty'][k][0]
    y=res[j]['ty'][k][1]
    color= np.array (['b', 'r', 'g'])
    height=int(4/13 * (end - start))
    plt.figure(figsize=(8, height))
    plt.xlabel('Elapsed time (arbitrary unit)', fontsize=20, labelpad=15)
    plt.xticks(fontsize=15)
    plt.ylabel('DUT ID', fontsize=20, labelpad=15)
    plt.yticks(fontsize=15)
    plt.ylim(-0.5+start, end+.5)
    plt.xlim(0, xlim)
    plt.hlines(range(len(t)),0, 210, color='k')
            
    for m, _ in enumerate(y):
        for n in range(1, len(t[m])):
            plt.scatter (t[m][n], m, color=color[[y[m][n]==0, y[m][n]==1, y[m][n]==-1]]) 
            # green for IT, blue for renewed IT, red for OOT