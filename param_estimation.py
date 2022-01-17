from scipy.optimize import minimize
import numpy as np
import sys
import pymc3 as pm
import pandas as pd
from theano import shared, tensor as tts
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


############################################
########## Define log likelihood  ##########
############################################

def log_weibull_left(x,alpha,beta):
    ''' Log likelihood for left censoring in weibull model '''
    return tts.log(1-tts.exp(-(x / beta)**alpha))

def log_weibull_right(x,alpha,beta):
    ''' Log likelihood for right censoring in weibull model '''
    return -(x / beta)**alpha

def log_weibull_interval(x,alpha,beta):
    ''' Log likelihood for interval censoring in weibull model '''
    return tts.log(tts.exp(-(x[:,0] / beta)**alpha)-tts.exp(-(x[:,1] / beta)**alpha))

def log_lognorm_left(x, mu, sigma):
    ''' Log likelihood for left censoring in lognormal model '''
    normal = pm.Normal.dist(mu=0, sigma = 1)
    return tts.log(tts.exp(normal.logcdf((tts.log(x)- mu)/sigma)))

def log_lognorm_right(x, mu, sigma):
    ''' Log likelihood for right censoring in lognormal model '''
    normal = pm.Normal.dist(mu=0, sigma = 1)
    return tts.log(1-tts.exp(normal.logcdf((tts.log(x)- mu)/sigma)))

def log_lognorm_interval(x, mu, sigma):
    ''' Log likelihood for interval censoring in lognormal model '''
    normal = pm.Normal.dist(mu=0, sigma = 1)
    return tts.log((1-tts.exp(normal.logcdf((tts.log(x[:,0])- mu)/sigma)))-(1-tts.exp(normal.logcdf((tts.log(x[:,1])- mu)/sigma))))


def neg_loglik_S3(thetas, model, t, y):
    ''' log likelihood function which will be minimized in S3 method'''
    ''' Implementation of equation E-8 in NCSLI RP-1 (2010)'''
    log_lik=0.
    for i, _ in enumerate(t): 
        t_st=0
        last_adjusted=0
        for j, _ in enumerate(t[i]):  
            if y[i, j]!=0:          # 0: not adjusted due to in-tolerance
                last_adjusted = j
                tau_ij =t[i,j]-t_st
                t_st =t[i,j]
                x_ij = (y[i,j]== -1) #True if renewal is for in tolerance
                                     #False otherwise 
                
                r_ij = model(thetas,tau_ij- (t[i,j]-t[i, j-1]))
                ro_ij = model(thetas, tau_ij)/r_ij
                tp = x_ij * np.log(ro_ij) + (1.-x_ij) * np.log(1.-ro_ij) + np.log(r_ij)
                log_lik= log_lik + tp
        # consider an in-tolerance after the last renewal in case of RF
        ###########################################################
        if t[i,-1] != t[i, last_adjusted]:
            r_ij = model(thetas,t[i, -1]-t[i, last_adjusted])     # important!!
            log_lik = log_lik + np.log(r_ij)
        ###########################################################
    return -log_lik


############################################
########## Parameter Estimation   ##########
############################################

def MCMC (y, t, draws=1000, rel_model='Weibull', include_censored=True):
    ''' MCMC sampling using a reliability model ''' 
    life_l=[] # left censored
    life_r=[] # right censored
    life_i=[] # interval censored
    for i, _ in enumerate(y):                      # this 'for' block is to convert t, y to a survival dataset (life)
        t0= t[i][0]                                
        for j in range(1,len(y[i]),1):
            if y[i][j]== 1:
                if j==1 or y[i][j-1]==-1 or y[i][j-1]==1:  #left censored
                    life_l.append(t[i][j]-t0)
                    t0=t[i][j]
                    
                if y[i][j-1]==0:
                    if j==1:
                        pass
                    else:
                        life_i.append([t[i][j-1]-t0,t[i][j]-t0]) #interval censored
                        t0=t[i][j]
                                          
            if y[i][j] == -1:                     # right censored  
                life_r.append (t[i][j]-t0)
                t0=t[i][j]

        if y[i][-1]== 0:                           # censored data at the last calibration record for a MTE 
            life_r.append (t[i][-1]-t0)  
            
    life_l=np.array(life_l)
    life_r=np.array(life_r)
    life_i=np.array(life_i)

    if rel_model=='Weibull':
        with pm.Model() as weibull_model:
            alpha = pm.Uniform('alpha', lower=0, upper=5)   ## revised considering various censoring situation in survival analysis
            beta = pm.Uniform('beta', lower=10, upper=60)                        
            if include_censored:
                if len(life_l)!=0:
                    y_cens_l = pm.Potential('y_cens_l', log_weibull_left(life_l,alpha, beta))
                else:    
                    pass
                if len(life_r)!=0:
                    y_cens_r = pm.Potential('y_cens_r', log_weibull_right(life_r,alpha, beta))
                else:
                    pass
                if len(life_i)!=0:
                    y_cens_i = pm.Potential('y_cens_i', log_weibull_interval(life_i,alpha, beta))
                else:
                    pass
                
            SEED = 2021 # from random.org, for reproducibility
            trace = pm.sample(draws=draws, tune=1500,
                              target_accept= 0.95, chains=2, 
                              random_seed=[SEED, SEED+11], 
                              init='adapt_diag')
            
    elif rel_model=='Lognormal':
        with pm.Model() as lognormal_model:
            mu = pm.Uniform('mu', -5, 5)   
            sigma = pm.Uniform('sigma', 0.01, 10)
            if include_censored:
                if len(life_l)!=0:
                    y_cens_l = pm.Potential('y_cens_l',log_lognorm_left(life_l, mu, sigma))
                else:
                    pass
                if len(life_r)!=0:
                    y_cens_r = pm.Potential('y_cens_r',log_lognorm_right(life_r, mu, sigma))
                else:
                    pass
                if len(life_i)!=0:
                    y_cens_i = pm.Potential('y_cens_i',log_lognorm_interval(life_i, mu, sigma))
                else:
                    pass
        
            SEED = 2021 # from random.org, for reproducibility
            trace = pm.sample(draws=draws, tune=1500,
                              target_accept= 0.95, chains=2, 
                              random_seed=[SEED, SEED+11], 
                              init='adapt_diag')
    return trace

def plot4 (th_theory, th_calc, th_mcmc, rel_model, lbl_model, labels, include_mcmc=True):
    ''' plot 4 plots for various conditions'''
    
    t_plot = np.linspace(0, 60, 100)
    tt_plot = np.linspace(0, 60, 30)
    _, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    
    if lbl_model == 'Weibull':
        for i in range(2):
            for j in range(2):
                id = 2*i +j
                ax[i,j].plot(t_plot, rel_model(th_theory, t_plot), c='k', label= lbl_model + ' model\n$\\beta$={:.2f}, $\\alpha$={:.2f}'.format(th_theory[0], th_theory[1]))
                ax[i,j].scatter(tt_plot, rel_model(th_calc[id], tt_plot), marker='^', c='b',label='S3 method\n$\\beta$={:.2f}, $\\alpha$={:.2f}'.format(th_calc[id][0], th_calc[id][1]))
                if include_mcmc:
                    ax[i,j].scatter(tt_plot, rel_model(th_mcmc[id],tt_plot), s=50, facecolors='none', edgecolors='r', linewidth=1.3, label='MCMC method \n$\\beta$={:.2f}, $\\alpha$={:.2f}'.format(th_mcmc[id][0],th_mcmc[id][1]))
                ax[i,j].legend(labelspacing=0.35, title='Interval = {}'.format(labels[id]), title_fontsize=12, fontsize=12)
                ax[i,j].set_yticks(np.arange(0, 1.1, step=0.1))
                ax[i,j].tick_params(axis='both', labelsize=14)
        ax[1,1].set_xlabel("Elapsed time (arbitrary unit)", fontsize=18);
        ax[0,0].set_ylabel("Reliability", fontsize=18);
        
    else:
        for i in range(2):
            for j in range(2):
                id = 2*i +j
                ax[i,j].plot(t_plot, rel_model(th_theory, t_plot), c='k', label= lbl_model + ' model\n$\\mu$={:.2f}, $\\sigma$={:.2f}'.format(th_theory[0], th_theory[1]))
                ax[i,j].scatter(tt_plot, rel_model(th_calc[id], tt_plot), marker='^', c='b',label='S3 method\n$\\mu$={:.2f}, $\\sigma$={:.2f}'.format(th_calc[id][0], th_calc[id][1]))
                if include_mcmc:
                    ax[i,j].scatter(tt_plot, rel_model(th_mcmc[id],tt_plot), s=50, facecolors='none', edgecolors='r', linewidth=1.3, label='MCMC method \n$\\mu$={:.2f}, $\\sigma$={:.2f}'.format(th_mcmc[id][0],th_mcmc[id][1]))
                ax[i,j].legend(labelspacing=0.35, title='Interval = {}'.format(labels[id]), title_fontsize=12, fontsize=12)
                ax[i,j].set_yticks(np.arange(0, 1.1, step=0.1))
                ax[i,j].tick_params(axis='both', labelsize=14)
        ax[1,1].set_xlabel("Elapsed time (arbitrary unit)", fontsize=18);
        ax[0,0].set_ylabel("Reliability", fontsize=18);        
    

def calculate_and_plot(cdt, res, j, include_mcmc=True, plot=True, include_censored=True):
    '''calculate S3 and MCMC and then plots results for the j-th condition given by cdt'''
    
    rel_model=cdt[j]['model']
    th_calc =[]; th_mcmc =[]; calsums = []; traces=[]
    for i in range(4):
        t =res[j]['ty'][i][0]; y =res[j]['ty'][i][1]
        
        # Estimate parameters using S3 method  
        thetas= minimize(neg_loglik_S3, cdt[j]['inits'], args=(cdt[j]['rel_ftn'],t,y), method='SLSQP')

        if thetas.success:          # if sucessfully minimised
            th_calc.append(thetas.x)

        else: 
            print ('Error: minimization failed because ' + thetas.message)
            sys.exit()
    
        # Estimate parameters using MCMC
        if include_mcmc:
            trace = MCMC (y, t, draws=5000, rel_model=rel_model, include_censored=include_censored)
            calsum = pm.summary(trace)
            traces.append(trace)
            if rel_model == 'Weibull':
                th0 = calsum.iloc[1,0]
                th1 = calsum.iloc[0,0]
                
            elif rel_model == 'Lognormal':
                th0 = calsum.iloc[0,0]
                th1 = calsum.iloc[1,0]
         
            th_mcmc.append([th0, th1])
                            
            calsums.append(calsum)
        
        print('sub-condition %d' %(i+1))
        if rel_model == 'Weibull':
            table = pd.DataFrame({'Method':['S3','MCMC'], 'beta':[thetas.x[0],th0], 'alpha':[thetas.x[1],th1]})
            print(table)
        if rel_model == 'Lognormal':
            table = pd.DataFrame({'Method':['S3','MCMC'], 'mu':[thetas.x[0],th0], 'sigma':[thetas.x[1],th1]})
            print(table)        
            
    res[j]['th_calc'] = th_calc
    res[j]['th_mcmc'] = th_mcmc
    res[j]['mcmc_summary'] = calsums
    res[j]['traces']= traces
        
    if plot:
        plot4 (cdt[j]['theta'], th_calc, th_mcmc, cdt[j]['rel_ftn'], cdt[j]['model'], cdt[j]['labels'], include_mcmc)
        

def show_estimation_result(cdt,res,j):
    
    cond_list=[]
    for i in [0,1,2,3]:
        cond_list.append(cdt[j]['labels'][i])
            
    param1_true_list=[];param2_true_list=[];
    param1_s3_list=[];param2_s3_list=[];
    param1_mcmc_list=[]; param1_mcmc_sd_list=[]; param2_mcmc_list=[]; param2_mcmc_sd_list=[];
    
    if cdt[j]['model']=='Weibull':
        for i in [0,1,2,3]:
            param1_true_list.append('%0.2f'%cdt[j]['theta'][0])
            param2_true_list.append('%0.2f'%cdt[j]['theta'][1])
            param1_s3_list.append('%0.2f'%res[j]['th_calc'][i][0])
            param2_s3_list.append('%0.2f'%res[j]['th_calc'][i][1])
            param1_mcmc_list.append('%0.2f'%res[j]['mcmc_summary'][i]['mean']['beta']) 
            param1_mcmc_sd_list.append('%0.2f'%res[j]['mcmc_summary'][i]['sd']['beta'])  
            param2_mcmc_list.append('%0.2f'%res[j]['mcmc_summary'][i]['mean']['alpha'])
            param2_mcmc_sd_list.append('%0.2f'%res[j]['mcmc_summary'][i]['sd']['alpha'])
                
        result = pd.DataFrame({'Conditions':cond_list, '$\\beta$(True)':param1_true_list,'$\\beta$(S3)':param1_s3_list, '$\\beta$(MCMC)':param1_mcmc_list,'SD($\\beta$)':param1_mcmc_sd_list,'$\\alpha$(True)':param2_true_list,'$\\alpha$(S3)':param2_s3_list,'$\\alpha$(MCMC)':param2_mcmc_list, 'SD($\\alpha$)':param2_mcmc_sd_list})    
    
    if cdt[j]['model']=='Lognormal':
        for i in [0,1,2,3]:
            param1_true_list.append('%0.2f'%cdt[j]['theta'][0])
            param2_true_list.append('%0.2f'%cdt[j]['theta'][1])
            param1_s3_list.append('%0.2f'%res[j]['th_calc'][i][0])
            param2_s3_list.append('%0.2f'%res[j]['th_calc'][i][1])
            param1_mcmc_list.append('%0.2f'%res[j]['mcmc_summary'][i]['mean']['mu']) 
            param1_mcmc_sd_list.append('%0.2f'%res[j]['mcmc_summary'][i]['sd']['mu'])  
            param2_mcmc_list.append('%0.2f'%res[j]['mcmc_summary'][i]['mean']['sigma'])
            param2_mcmc_sd_list.append('%0.2f'%res[j]['mcmc_summary'][i]['sd']['sigma'])
                
        result = pd.DataFrame({'Conditions':cond_list,'$\\mu$\(True\)':param1_true_list,'$\\mu$(S3)':param1_s3_list, '$\\mu$(MCMC)':param1_mcmc_list,'SD($\\mu$)':param1_mcmc_sd_list,'$\\sigma$(True)':param2_true_list,'$\\sigma$(S3)':param2_s3_list,'$\\sigma$(MCMC)':param2_mcmc_list, 'SD($\\sigma$)':param2_mcmc_sd_list}) 
   
    return result
    








