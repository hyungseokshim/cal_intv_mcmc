import numpy as np
import pandas as pd
from statistics import NormalDist
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_interval(cdt, res, j, method='mcmc', target_rel=0.9):
    ''' calculate optimal calibration interval on calibration records'''
    a = NormalDist().inv_cdf(1-target_rel)

    if method == 'mcmc':        
        if cdt[j]['model']=='Weibull':            
            interval_all=[]
            for i in [0,1,2,3]:
                beta_spl = np.array(res[j]['traces'][i]['beta'])
                alpha_spl = np.array(res[j]['traces'][i]['alpha'])
                interval_cal = beta_spl*(-np.log(target_rel))**(1./alpha_spl) # inverse of Weibull Reliability Function
                interval_all.append(interval_cal)

        if cdt[j]['model']=='Lognormal':
            interval_all=[]
            for i in [0,1,2,3]:
                mu_spl = np.array(res[j]['traces'][i]['mu'])
                sigma_spl = np.array(res[j]['traces'][i]['sigma'])
                interval_cal = np.exp(sigma_spl*a+mu_spl)
                interval_all.append(interval_cal)
            
    if method == 's3':
        if cdt[j]['model']=='Weibull':
            interval_all=[]
            for i in [0,1,2,3]:
                beta_spl = np.array(res[j]['th_calc'][i][0])
                alpha_spl = np.array(res[j]['th_calc'][i][1])
                interval_cal = beta_spl*(-np.log(target_rel))**(1./alpha_spl) # inverse of Weibull Reliability Function
                interval_all.append(interval_cal)

        if cdt[j]['model']=='Lognormal':
            interval_all=[]
            for i in [0,1,2,3]:
                mu_spl = np.array(res[j]['th_calc'][i][0])
                sigma_spl = np.array(res[j]['th_calc'][i][1])
                interval_cal = np.exp(sigma_spl*a+mu_spl)
                interval_all.append(interval_cal)

    if method == 'true':
        if cdt[j]['model']=='Weibull':
            interval_all=[]
            for i in [0,1,2,3]:
                beta_spl = np.array(cdt[j]['theta'][0])
                alpha_spl = np.array(cdt[j]['theta'][1])
                interval_cal = beta_spl*(-np.log(target_rel))**(1./alpha_spl) # inverse of Weibull Reliability Function
                interval_all.append(interval_cal)

        if cdt[j]['model']=='Lognormal':
            interval_all=[]
            for i in [0,1,2,3]:
                mu_spl =np.array(cdt[j]['theta'][0])
                sigma_spl = np.array(cdt[j]['theta'][1])
                interval_cal = np.exp(sigma_spl*a+mu_spl)
                interval_all.append(interval_cal)
                    
    return interval_all

def show_interval_result(cdt, res, j, target_rel=0.9):
    cal_mcmc=calculate_interval(cdt, res, j, 'mcmc', target_rel)
    cal_s3=calculate_interval(cdt, res, j, 's3', target_rel)
    cal_true=calculate_interval(cdt, res, j, 'true', target_rel)
    
    cond_list=[]; cal_true_list=[]; cal_s3_list=[]; cal_mcmc_list=[]; cal_mcmc_sd_list=[]
    
    for i in [0,1,2,3]:
        cond_list.append(cdt[j]['labels'][i])
        cal_true_list.append('%0.2f'%cal_true[i])
        cal_s3_list.append('%0.2f'%cal_s3[i])
        cal_mcmc_list.append('%0.2f'%cal_mcmc[i].mean())
        cal_mcmc_sd_list.append('%0.2f'%cal_mcmc[i].std())       
    
    result = pd.DataFrame({'Conditions':cond_list, 'True':cal_true_list, 'S3':cal_s3_list, 'MCMC':cal_mcmc_list, 'SD(MCMC)':cal_mcmc_sd_list})
    
    return result



def plot_interval_mcmc(cdt, res, j, k, target_rel=0.9):
    
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    interval = calculate_interval(cdt, res, j, 'mcmc', target_rel)
    sns.distplot(interval[k], norm_hist=True, kde=True, ax=ax)   
    mean_intv=np.array(interval[k]).mean()
    std_intv=np.array(interval[k]).std()
    ax.vlines(mean_intv, 0, 0.1, color='k')
    ax.text (mean_intv+0.5,0.48,'Mean = {:3.2f}'.format(mean_intv),fontsize=12.0, color='blue')
    ax.text (mean_intv+0.5,0.4,'SD = {:3.2f}'.format(std_intv),fontsize=12.0, color='blue')
    ax.set_xlabel ('Calibration interval (arbitrary unit)', fontsize=18)
    ax.set_ylabel ('Probability density', fontsize=18)
    ax.tick_params(axis='both', labelsize=15)

def plot4_interval_mcmc(cdt, res, j, target_rel=0.9):

    fig, ax = plt.subplots(2,2, figsize=(10, 8), constrained_layout=True)
    interval = calculate_interval(cdt, res, j, 'mcmc', target_rel)
    
    for k in [0,1]:
        for l in [0,1]:
            id=2*k+l
            sns.distplot(interval[id], norm_hist=True, kde=True, ax=ax[k,l])   
            mean_intv=np.array(interval[id]).mean()
            std_intv=np.array(interval[id]).std()
            ax[k,l].vlines(mean_intv, 0, 0.1, color='k')
            ax[k,l].text (mean_intv+0.5,0.48,'Mean = {:3.2f}'.format(mean_intv),fontsize=12.0, color='blue')
            ax[k,l].text (mean_intv+0.5,0.4,'SD = {:3.2f}'.format(std_intv),fontsize=12.0, color='blue')
            ax[k,l].set_xlabel ('Calibration interval (arbitrary unit)', fontsize=18)
            ax[k,l].set_ylabel ('Probability density', fontsize=18)
            ax[k,l].tick_params(axis='both', labelsize=15)


