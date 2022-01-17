import matplotlib.pyplot as plt
from arviz import plot_trace
import time
import numpy as np
from rel_mod import mod_weibu, mod_lognorm

def dist_param(cdt,res,j,k=0):    
    trace = res[j]['traces'][k]
    if cdt[j]['model']=='Weibull':
        print('Condition : %s, %s'%(cdt[j]['model'],cdt[j]['labels'][k])) 
        plot_trace(trace, var_names=['alpha','beta'], chain_prop="color")
            
    else :
        print('Condition : %s, %s'%(cdt[j]['model'],cdt[j]['labels'][k]))
        plot_trace(trace, var_names=['mu','sigma'], chain_prop="color")  
            
def dist_rel(cdt, res, j, k, size = 500, color='r'):  #j:condition, k:sub-condition
    rel_model = cdt[j]['rel_ftn']
    t_plot = np.linspace(0, 60, 100)
    _, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 6), constrained_layout=True)
    
    for i in  range(size):
        th=[]
        if rel_model == mod_weibu:
            th.append(res[j]['traces'][k]['beta'][i:i+1])
            th.append(res[j]['traces'][k]['alpha'][i:i+1])
        if rel_model == mod_lognorm:
            th.append(res[j]['traces'][k]['mu'][i:i+1])  
            th.append(res[j]['traces'][k]['sigma'][i:i+1])
        ax.plot(t_plot, rel_model(th, t_plot), c='k', alpha= 0.007)
                
    if rel_model == mod_weibu:
        lbl_model = 'Weibull'
        
    if rel_model == mod_lognorm:
        lbl_model = 'Lognormal'
        
        
    if lbl_model=='Weibull':        
        ax.plot(t_plot, rel_model(cdt[j]['theta'], t_plot), ls='--', c=color, 
            label= lbl_model + ' model\n$\\beta$={:.2f}, $\\alpha$={:.2f}'.\
                     format(cdt[j]['theta'][0], cdt[j]['theta'][1]))
        
    if lbl_model == 'Lognormal':
        ax.plot(t_plot, rel_model(cdt[j]['theta'], t_plot), ls='--', c=color, 
            label= lbl_model + ' model\n$\\mu$={:.2f}, $\\sigma$={:.2f}'.\
                     format(cdt[j]['theta'][0], cdt[j]['theta'][1]))
        
    ax.set_xlabel("Elapsed time (arbitrary unit)", fontsize=18)
    ax.set_ylabel("Reliability", fontsize=18)
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.tick_params(axis='both', labelsize=14)
    
    ax.legend(fontsize=13)
    ax.text(6, 0.05, 'Plots with {} samples'.format(size),fontsize=11)
    ax.set_title('Condition : %s, %s'%(cdt[j]['model'],cdt[j]['labels'][k]), fontsize=18)
            