from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)

def plot_mcmc_correl(cdt,res,j,k,scatter=True,**kwargs):
    """Plot traces and contours on condition j and subcondition i"""
    
    if cdt[j]['model']=='Weibull':
        trace=[res[j]['traces'][k]['beta'],res[j]['traces'][k]['alpha']]
    if cdt[j]['model']=='Lognormal':
        trace=[res[j]['traces'][k]['mu'],res[j]['traces'][k]['sigma']]       
        
    fig,ax=plt.subplots(figsize=(5, 4), constrained_layout=True)
    
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], linewidths=3, **kwargs)
    if scatter:
        ax.scatter(trace[0], trace[1], s=0.1, c='k')
        
    if cdt[j]['model']=='Weibull':
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\alpha$')
    if cdt[j]['model']=='Lognormal':
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\sigma$')    
        
    correl = pearsonr(trace[0], trace[1])
    plt.text(0.75, 0.8, 'r = {:3.2f}'.format(correl[0]), fontweight='bold', fontsize=12, transform=ax.transAxes)
    
def plot4_mcmc_correl(cdt,res,j,scatter=True,**kwargs):
    """Plot 4 traces and contours on condtion j"""
    trace1=[]
    trace2=[]
    if cdt[j]['model']=='Weibull':
        for i in range(4):
            trace1.append(res[j]['traces'][i]['beta'])
            trace2.append(res[j]['traces'][i]['alpha'])
    if cdt[j]['model']=='Lognormal':
        for i in range(4):
            trace1.append(res[j]['traces'][i]['mu'])
            trace2.append(res[j]['traces'][i]['sigma'])
            
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    
    for k in range(2):
        for l in range(2):
            id = 2*k+l       
            xbins, ybins, sigma = compute_sigma_level(trace1[id], trace2[id])
            ax[k,l].contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], linewidths=3, **kwargs)
            if scatter:
                ax[k,l].scatter(trace1[id], trace2[id], s=0.1, c='k')
        
            if cdt[j]['model']=='Weibull':
                ax[k,l].set_xlabel(r'$\beta$')
                ax[k,l].set_ylabel(r'$\alpha$')
            if cdt[j]['model']=='Lognormal':
                ax[k,l].set_xlabel(r'$\mu$')
                ax[k,l].set_ylabel(r'$\sigma$')   
                
            correl = pearsonr(trace1[id], trace2[id])
            ax[k,l].text(0.75, 0.8, 'r = {:3.2f}'.format(correl[0]), fontweight='bold', fontsize=12, transform=ax[k,l].transAxes)

    
            
            