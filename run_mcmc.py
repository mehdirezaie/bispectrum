
import os
import sys
import matplotlib.pyplot as plt 
import numpy as np
import emcee

from glob import glob
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.optimize import minimize
from getdist import plots, MCSamples

#
#--- global variables
#
pk_molino = './pk_molino.npz'
pk_glam   = './pk_glam.npz'
pk_glam_nobao = './pk_glam_nobao.npz'

bk_molino = './bk_molino.npz'
bk_glam   = './bk_glam.npz'
bk_glam_nobao = './bk_glam_nobao.npz'


def read_molino_pk():
    
    if not os.path.exists(pk_molino):        
        pk = []
        files = glob('/home/lado/Molino/pk_molino.z0.0.fiducial.nbody*.hod0.hdf5')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 1:])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, 0], pk=pk)
        np.savez(pk_molino, **ret)
    else:
        ret = np.load(pk_molino)
    
    return ret


def read_glam_pk():

    if not os.path.exists(pk_glam):        
        pk = []
        files = glob('/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/BAO/Pk_CatshortV.0114.*.DAT')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 1])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, 0], pk=pk)
        np.savez(pk_glam, **ret)
    else:
        ret = np.load(pk_glam)
    
    return ret


def read_glam_pk_nobao():

    if not os.path.exists(pk_glam_nobao):        
        pk = []
        files = glob('/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/noBAO/Pk_CatshortV.0114.*.DAT')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 1])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, 0], pk=pk)
        np.savez(pk_glam_nobao, **ret)
    else:
        ret = np.load(pk_glam_nobao)
    
    return ret


def read_molino_bk():

    if not os.path.exists(bk_molino):        
        pk = []
        files = glob('/home/lado/Molino/bk_molino.z0.0.fiducial.nbody*.hod0.hdf5')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 3:])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, :3], pk=pk)
        np.savez(bk_molino, **ret)
    else:
        ret = np.load(bk_molino)
        
    return ret


def read_glam_bk():
    
    if not os.path.exists(bk_glam):        
        pk = []
        files = glob('/mnt/data1/BispectrumGLAM/BAO/Bk_CatshortV.0114.*.h5')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 3])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, :3], pk=pk)
        np.savez(bk_glam, **ret)
    else:
        ret = np.load(bk_glam)    
    
    return ret


def read_glam_bk_nobao():
    
    if not os.path.exists(bk_glam_nobao):        
        pk = []
        files = glob('/mnt/data1/BispectrumGLAM/noBAO/Bk_CatshortV.0114.*.h5')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 3])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, :3], pk=pk)
        np.savez(bk_glam_nobao, **ret)
    else:
        ret = np.load(bk_glam_nobao)    
    
    return ret

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
    def __init__(self, k, pk_ratio, is_bk=False, use_hart=False):
        
        self.is_bk = is_bk
        self.use_hart = use_hart        
        self.k = k            
        self.pk_r = pk_ratio
        print(self.pk_r.shape)
        
        self.pk_rm  = self.pk_r.mean(axis=0)
        if self.is_bk:
            self.pk_rint = LinearNDInterpolator(self.k, self.pk_rm)
        else:
            self.pk_rint = interp1d(self.k, self.pk_rm, bounds_error=False)

    def __call__(self, alpha):
        res = self.pk_rm[self.is_good] - self.pk_rint(alpha*self.k[self.is_good])
        return abs(res.dot(self.icov.dot(res))-1.0)
    
    def model(self, kg, theta):
        if self.is_bk:            
            return theta[1]*self.pk_rint(theta[0]*kg) \
                    + theta[2] \
                    + theta[3]*(1./kg).sum(axis=1) \
                    + theta[4]*(kg).sum(axis=1) \
                    + theta[5]*(kg*kg).sum(axis=1)
        else:
            return theta[1]*self.pk_rint(theta[0]*kg) \
                    + theta[2] \
                    + theta[3]/kg \
                    + theta[4]*kg \
                    + theta[5]*kg*kg

    
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
        lp += 0. if 0.8 < theta[0] < 1.2 else -np.inf
        for param in theta[1:]:
            lp += 0. if -2. < param < 2. else -np.inf
        
        ## Gaussian prior on ?
        #mmu = 3.     # mean of the Gaussian prior
        #msigma = 10. # standard deviation of the Gaussian prior
        #lp += -0.5*((m - mmu)/msigma)**2

        return lp        
        
    def loglike(self, theta):
        '''The natural logarithm of the likelihood.'''
        # evaluate the model
        res = self.pk_rm[self.is_good] - self.model(self.k[self.is_good], theta)
        # return the log likelihood
        return -0.5*res.dot(self.icov.dot(res))
    
    def logpost(self, theta):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta)


def run_mcmc(kmax, k, pk_ratio, pk_mol=None, is_bk=False, use_hart=False):
    print(k.shape, pk_ratio.shape)
    
    kmin = 0.005
    ndim = 6
    nwalkers = 50
    nsteps = 10000

    # power spectra
    bao_pk = BAO(k, pk_ratio, is_bk=is_bk, use_hart=use_hart)
    bao_pk.prep_k(kmin, kmax)
    bao_pk.prep_cov(pk_mol=pk_mol)

    # helper functions
    def logpost(thetas):
        val = bao_pk.logpost(thetas)
        return val if not np.isnan(val) else -np.inf
    
    def nlogpost(thetas):
        return -1*logpost(thetas)
    
    initial_guess = [1.1, 1.1] + (ndim-2)*[0., ]
    res = minimize(nlogpost, initial_guess, method='Powell')
    print('Best Fit', res.x)
    #assert res.success
    start = res.x + 0.1*np.random.randn(nwalkers, ndim)
    print('Initial Points', start)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost)
    sampler.run_mcmc(start, nsteps, progress=True)
    chain = sampler.get_chain()
    ndim = chain.shape[-1]
    chainr = chain[nsteps//2:, :, :].reshape(-1, ndim)

    prcnt = np.percentile(chainr[:, 0], [16, 84])
    sigma = 0.5*(prcnt[1]-prcnt[0])
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
    for kmax_ in kmax:
        sig_ = run_mcmc(kmax_, y_glm['k'], y_glm['pk']/y_glm_nobao['pk'].mean(axis=0), pk_mol=y_mol, is_bk=is_bk)

        sigmas.append([kmax_, sig_])
    np.savetxt(f'./sigmas_mcmc_isbk_{is_bk}.txt', sigmas)
