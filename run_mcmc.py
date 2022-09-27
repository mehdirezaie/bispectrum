
import os
import sys
import matplotlib.pyplot as plt 
import numpy as np
import emcee

from tqdm import tqdm
from glob import glob
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.optimize import minimize
from getdist import plots, MCSamples
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool



def sigma_finder(pk_obj, kmin, kmax, delta=20.0, pk_mol=None):
    if kmax < 0.15:
        bounds = [(1.0, 1.15)]
    else:
        bounds = [(1.0, 1.05)]
    # prep
    pk_obj.prep_k(kmin, kmax)
    pk_obj.prep_cov(pk_mol=pk_mol)
    
    res = minimize(pk_obj, 1., bounds=bounds, 
                   tol=1.0e-8, method='Powell')
    return res.x[0]-1.0


class BAO:
    def __init__(self, k, pk_ratio, is_bk=False, use_hart=False, fill_value=np.nan):
        
        self.is_bk = is_bk
        self.use_hart = use_hart        
        self.k = k            
        self.pk_r = pk_ratio
        print(self.pk_r.shape)
        
        self.pk_rm  = self.pk_r.mean(axis=0)
        if self.is_bk:
            self.pk_rint = LinearNDInterpolator(self.k, self.pk_rm, fill_value=fill_value)
            def model(kg, theta):
                return theta[1]*self.pk_rint(theta[0]*kg) + theta[2]              
        else:
            self.pk_rint = interp1d(self.k, self.pk_rm, bounds_error=False, fill_value=fill_value)
            def model(kg, theta):
                return theta[1]*self.pk_rint(theta[0]*kg) + theta[2]       
        self.model = model
        
    def __call__(self, alpha):
        res = self.pk_rm[self.is_good] - self.pk_rint(alpha*self.k[self.is_good])
        return abs(res.dot(self.icov.dot(res))-1.0)
    
    def prep_k(self, kmin, kmax):
        
        # apply cut on k
        self.is_good = np.ones(self.k.shape[0], '?')
        if self.is_bk:
            for i in range(self.k.shape[1]):
                self.is_good &= (self.k[:, i] > kmin) & (self.k[:, i] < kmax)
        else:
            self.is_good &= (self.k > kmin) & (self.k < kmax)
            
    def prep_cov(self, pk_mol=None):

        nmocks, nbins = self.pk_r[:, self.is_good].shape
        hartlapf = (nmocks-1.0)/(nmocks-nbins-2.0)
        self.cov = np.cov(self.pk_r[:, self.is_good], rowvar=False) #/ nmocks
        
        if pk_mol is not None:
            pk_std = np.diagonal(self.cov)**0.5
            
            pk_cov_ = np.cov(pk_mol['pk'][:,self.is_good, 0], rowvar=0)
            pk_std_ = np.std(pk_mol['pk'][:,self.is_good, 0], axis=0)
            red_cov = pk_cov_/np.outer(pk_std_, pk_std_)
            
            self.cov = red_cov*np.outer(pk_std, pk_std)
            
        #assert np.linalg.det(self.cov)!= 0
        if self.use_hart:
            self.cov = self.cov*hartlapf
        self.icov = np.linalg.inv(self.cov)
        
    def logprior(self, theta):
        ''' The natural logarithm of the prior probability. '''
        lp = 0.
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp += 0. if 0.7  < theta[0] < 1.3 else -np.inf
        lp += 0. if 0.0  < theta[1] < 2.0 else -np.inf
        lp += 0. if -2.0 < theta[2] < 2.0 else -np.inf        
        # lp += 0. if -2.0 < theta[3] < 2.0 else -np.inf        
        # lp += 0. if -20.0 < theta[4] < 20.0 else -np.inf        
        # lp += 0. if -50.0 < theta[5] < 50.0 else -np.inf        
        
        ## Gaussian prior on ?
        #mmu = 3.     # mean of the Gaussian prior
        #msigma = 10. # standard deviation of the Gaussian prior
        #lp += -0.5*((m - mmu)/msigma)**2

        return lp        
        
    def loglike(self, theta):
        '''The natural logarithm of the likelihood.'''
        # evaluate the model
        model_ = self.model(self.k[self.is_good], theta)
        is_fine = ~np.isnan(model_)
        res = self.pk_rm[self.is_good][is_fine] - model_[is_fine]
        # return the log likelihood
        return -0.5*res.dot(self.icov[is_fine,:][:,is_fine].dot(res))
    
    def logpost(self, theta):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta)


class BAOJoint:
    def __init__(self, k1, pk_ratio1, k2, pk_ratio2, use_hart=False, fill_value=np.nan):
        
        self.use_hart = use_hart        
        self.k1 = k1            
        self.pk_r1 = pk_ratio1
        self.k2 = k2            
        self.pk_r2 = pk_ratio2
        
        print(self.pk_r1.shape)
        print(self.pk_r2.shape)        
        
        self.pk_rm  = self.pk_r.mean(axis=0)
        if self.is_bk:
            self.pk_rint = LinearNDInterpolator(self.k, self.pk_rm, fill_value=fill_value)
            def model(kg, theta):
                return theta[1]*self.pk_rint(theta[0]*kg) + theta[2]              
        else:
            self.pk_rint = interp1d(self.k, self.pk_rm, bounds_error=False, fill_value=fill_value)
            def model(kg, theta):
                return theta[1]*self.pk_rint(theta[0]*kg) + theta[2]       
        self.model = model
        
    def __call__(self, alpha):
        res = self.pk_rm[self.is_good] - self.pk_rint(alpha*self.k[self.is_good])
        return abs(res.dot(self.icov.dot(res))-1.0)
    
    def prep_k(self, kmin, kmax):
        
        # apply cut on k
        self.is_good = np.ones(self.k.shape[0], '?')
        if self.is_bk:
            for i in range(self.k.shape[1]):
                self.is_good &= (self.k[:, i] > kmin) & (self.k[:, i] < kmax)
        else:
            self.is_good &= (self.k > kmin) & (self.k < kmax)
            
    def prep_cov(self, pk_mol=None):

        nmocks, nbins = self.pk_r[:, self.is_good].shape
        hartlapf = (nmocks-1.0)/(nmocks-nbins-2.0)
        self.cov = np.cov(self.pk_r[:, self.is_good], rowvar=False) #/ nmocks
        
        if pk_mol is not None:
            pk_std = np.diagonal(self.cov)**0.5
            
            pk_cov_ = np.cov(pk_mol['pk'][:,self.is_good, 0], rowvar=0)
            pk_std_ = np.std(pk_mol['pk'][:,self.is_good, 0], axis=0)
            red_cov = pk_cov_/np.outer(pk_std_, pk_std_)
            
            self.cov = red_cov*np.outer(pk_std, pk_std)
            
        #assert np.linalg.det(self.cov)!= 0
        if self.use_hart:
            self.cov = self.cov*hartlapf
        self.icov = np.linalg.inv(self.cov)
        
    def logprior(self, theta):
        ''' The natural logarithm of the prior probability. '''
        lp = 0.
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp += 0. if 0.7  < theta[0] < 1.3 else -np.inf
        lp += 0. if 0.0  < theta[1] < 2.0 else -np.inf
        lp += 0. if -2.0 < theta[2] < 2.0 else -np.inf        
        # lp += 0. if -2.0 < theta[3] < 2.0 else -np.inf        
        # lp += 0. if -20.0 < theta[4] < 20.0 else -np.inf        
        # lp += 0. if -50.0 < theta[5] < 50.0 else -np.inf        
        
        ## Gaussian prior on ?
        #mmu = 3.     # mean of the Gaussian prior
        #msigma = 10. # standard deviation of the Gaussian prior
        #lp += -0.5*((m - mmu)/msigma)**2

        return lp        
        
    def loglike(self, theta):
        '''The natural logarithm of the likelihood.'''
        # evaluate the model
        model_ = self.model(self.k[self.is_good], theta)
        is_fine = ~np.isnan(model_)
        res = self.pk_rm[self.is_good][is_fine] - model_[is_fine]
        # return the log likelihood
        return -0.5*res.dot(self.icov[is_fine,:][:,is_fine].dot(res))
    
    def logpost(self, theta):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta)

    
    
    
def run_mcmc(kmax, k, pk_ratio, pk_mol=None, is_bk=False, use_hart=False):

    mcmc_file = f'mcmcs/mcmc_is_bk_{is_bk}_kmax_{kmax:.2f}.npz'
    print(k.shape, pk_ratio.shape, mcmc_file)
    
    kmin = 0.015
    ndim = 3
    nwalkers = 10
    nsteps = 10000

    # power spectra
    bao_pk = BAO(k, pk_ratio, is_bk=is_bk, use_hart=use_hart)
    bao_pk.prep_k(kmin, kmax)
    bao_pk.prep_cov(pk_mol=pk_mol)

    # helper functions
    def logpost(thetas):
        val = bao_pk.logpost(thetas)
        return val
    
    def nlogpost(thetas):
        return -1*logpost(thetas)
    
    initial_guess = [1.1, 1.1] + (ndim-2)*[0., ]
    res = minimize(nlogpost, initial_guess, method='Powell')
    #print('Best Fit', res.x)
    #assert res.success
    start = res.x + res.x*0.01*np.random.randn(nwalkers, ndim)
    #print('Initial Points', start)
    #sys.exit()
    
    n = 2
    print('Using', n, 'processes for the pool')    
    with Pool(n) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool)
        sampler.run_mcmc(start, nsteps, progress=True)

    chain = sampler.get_chain()
    ndim = chain.shape[-1]
    chainr = chain[nsteps//2:, :, :].reshape(-1, ndim)
    prcnt = np.percentile(chainr[:, 0], [16, 84])
    sigma = 0.5*(prcnt[1]-prcnt[0])
    
    savez(mcmc_file, **{'chain':sampler.get_chain(), 
                    'log_prob':sampler.get_log_prob(), 
                    'best_fit':res.x,
                    'best_fit_logprob':res.fun,
                    'best_fit_success':res.success, 
                    '#params':ndim})
    del sampler
    # names = ["x%s"%i for i in range(ndim)]
    # labels =  [r"\alpha"] + ["A%d"%i for i in range(ndim-1)]
    # samples = MCSamples(samples=chainr, names=names, labels=labels,
    #                    settings={'mult_bias_correction_order':0,
    #                              'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})    
    # g = plots.get_subplot_plotter(width_inch=14)
    # g.triangle_plot([samples, ], filled=True)
    
    return sigma


if __name__ == "__main__":

    # read 
    is_bk = bool(int(sys.argv[1]))
    print('Bispectrum:', is_bk)
    
    if is_bk:
        y_mol = read_molino_bk()
        y_glm = read_glam_bk()
        y_glm_nobao = read_glam_bk_nobao()
    else:
        y_mol = read_molino_pk()
        y_glm = read_glam_pk()
        y_glm_nobao = read_glam_pk_nobao()

    sigmas = []
    kmax = np.arange(0.05, 0.31, 0.01)        
    for kmax_ in tqdm(kmax):
        sig_ = run_mcmc(kmax_, y_glm['k'], y_glm['pk']/y_glm_nobao['pk'].mean(axis=0), pk_mol=y_mol, is_bk=is_bk)

        sigmas.append([kmax_, sig_])
    np.savetxt(f'./mcmcs/sigmas_mcmc_isbk_{is_bk}.txt', sigmas)