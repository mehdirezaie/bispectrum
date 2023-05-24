import numpy as np
from scipy.stats import binned_statistic
from src.models import BiSpectrum

from multiprocessing.pool import ThreadPool as Pool
import emcee

mock_range = {'LRGz0':(8000, 8025),
              'ELGz1':(1000, 1025),
              'QSOz2':(5000, 5025)}  

def get_equilateral(k, eps=0.031):
    """ Identify Equilateral """
    indices = []
    for i, ki in enumerate(k):
        if (abs(ki[2]-ki[1]) < eps) & (abs(ki[2]-ki[0]) < eps):
            indices.append(i)
    return np.array(indices)

def get_isoceles(k, eps1=0.01, eps2=0.07):
    """ Identify Isoceles """
    indices = []    
    for i, ki in enumerate(k):
        if (abs(ki[2]-ki[1]) < eps1) & (abs(ki[1]-ki[0]) > eps2):    
            indices.append(i)
    return np.array(indices)

def bin_bispectra(k, r):
    """ Bin Spectra """
    kmin, kmax = np.percentile(k[:, 0], [0, 100])
    bins = np.arange(kmin-0.005, kmax+0.015, 0.01)
    k_bin = 0.5*(bins[1:]+bins[:-1])
    #print(f'marginalize over k2 & k3 given: {bins[:3]} ...')
    
    r_bin = []
    for i in range(r.shape[0]):
        r_bin_ = binned_statistic(k[:, 0], r[i, :], bins=bins)[0]
        r_bin.append(r_bin_)
    r_bin = np.array(r_bin)
    
    return k_bin, r_bin

def bin_bispectrum(k, r):
    """ Bin Spectra """
    kmin, kmax = np.percentile(k[:, 0], [0, 100])
    bins = np.arange(kmin-0.005, kmax+0.015, 0.01)
    k_bin = 0.5*(bins[1:]+bins[:-1])
    #print(f'marginalize over k2 & k3 given: {bins[:3]} ...')
    
    r_bin = binned_statistic(k[:, 0], r, bins=bins)[0]    
    return k_bin, r_bin


class PowerSpectrumData:
    def __init__(self, k, p, p_bestfit, p_smooth):
        self.k = k
        self.p = p
        self.p_bestfit = p_bestfit
        self.p_smooth = p_smooth


class BiSpectrumData:
    def __init__(self, k, b, b_bestfit, b_smooth):
        self.k = k
        self.b = b
        self.b_bestfit = b_bestfit
        self.b_smooth = b_smooth
        
        self.__get_iso()
        self.__get_eqi()
        self.__get_all()
        
    def __get_iso(self):
        
        is_good = get_isoceles(self.k)
        k_good = self.k[is_good]
        b_good = self.b[:, is_good]
        bs_good = self.b_smooth[:, is_good]
        
        self.k_iso, self.b_iso = bin_bispectra(k_good, b_good)
        __, self.bs_iso = bin_bispectra(k_good, bs_good)
        
    def __get_eqi(self):
        
        is_good = get_equilateral(self.k)
        k_good = self.k[is_good]
        b_good = self.b[:, is_good]
        bs_good = self.b_smooth[:, is_good]
        
        self.k_eqi, self.b_eqi = bin_bispectra(k_good, b_good)
        __, self.bs_eqi = bin_bispectra(k_good, bs_good)
        
        
    def __get_all(self):
        
        self.k_all, self.b_all = bin_bispectra(self.k, self.b)
        __, self.bs_all = bin_bispectra(self.k, self.b_smooth)        

        

def get_bispectra(tracer):
    """ Get Data """ 
    print(f'tracer: {tracer}')
  
    # read
    b = []
    b_bestfit = []
    b_smooth = []
    for i in range(*mock_range[tracer]):
        bk = np.loadtxt(f'/Users/mehdi/data/Abacus_smooth/all_bk_{tracer}_{i:d}.txt')      
        b.append(bk[:, 3])
        b_bestfit.append(bk[:, 4])
        b_smooth.append(bk[:, 5])
        
    b = np.array(b)
    b_bestfit = np.array(b_bestfit)
    b_smooth = np.array(b_smooth)
    k = bk[:, :3]
    
    return BiSpectrumData(k, b, b_bestfit, b_smooth)




def get_powerspectra(tracer, verbose=False):
    """ Get Data """ 
    print(f'tracer: {tracer}')
  
    # read
    p = []
    p_bestfit = []
    p_smooth = []
    for i in range(*mock_range[tracer]):
        pk = np.loadtxt(f'/Users/mehdi/data/AbacusData/pk_{tracer}.{i:d}')      
        pk_smooth = np.loadtxt(f'/Users/mehdi/data/Abacus_smooth/all_pk_{tracer}_{i:d}.txt')
        p.append(pk[:, 1])
        p_bestfit.append(pk_smooth[:, 1])
        p_smooth.append(pk_smooth[:, 2])
        
    p = np.array(p)
    p_bestfit = np.array(p_bestfit)
    p_smooth = np.array(p_smooth)
    k = pk[:, 0]
    
    return PowerSpectrumData(k, p, p_bestfit, p_smooth)


def get_corr(x, y):
    cxy = ((x-x.mean())*(y-y.mean())).sum()
    cxx = ((x-x.mean())*(x-x.mean())).sum()
    cyy = ((y-y.mean())*(y-y.mean())).sum()
    return cxy/np.sqrt(cxx*cyy)

def correlate(r_b, r_p):
    
    corr = np.ones((r_b.shape[1], r_p.shape[1]))
    corr[:,:] = np.nan
    for i in range(r_b.shape[1]):
        for j in range(r_p.shape[1]):
            corr[i, j] = get_corr(r_b[:, i], r_p[:, j])
            
    return corr

def get_rcov(y):
    cov_ = np.cov(y, rowvar=0)
    std_ = np.diagonal(cov_)**0.5
    rcov = cov_ / np.outer(std_, std_)    
    return rcov

def prep_rcov(input_files, output_file, icol):    
    n = len(input_files)
    y = []
    for i, file in enumerate(input_files):
        d_ = np.loadtxt(file)
        y.append(d_[:, icol])
        if i%(n//4)==0:print(f'{i}/{n}')
    
    y = np.array(y)
    rcov_raw = get_rcov(y)
    
    if icol==3:
        k = d_[:, :3]
        k_all, y_all = bin_bispectra(k, y)
        rcov_all = get_rcov(y_all)

        is_good = get_isoceles(k)
        k_good = k[is_good]
        y_good = y[:, is_good]
        k_iso, y_iso = bin_bispectra(k_good, y_good)
        rcov_iso = get_rcov(y_iso)

        is_good = get_equilateral(k)
        k_good = k[is_good]
        y_good = y[:, is_good]
        k_eqi, y_eqi = bin_bispectra(k_good, y_good)
        rcov_eqi = get_rcov(y_eqi)
    else:
        k = d_[:, 0]
        rcov_all = None
        rcov_iso = None
        rcov_eqi = None
        k_all = None
        k_iso = None
        k_eqi = None
    
    np.savez(output_file, **{'raw':rcov_raw, 'all':rcov_all, 
                             'iso':rcov_iso, 'eqi':rcov_eqi})
    print(f'wrote {output_file}')
    
    
    
class BisPosterior:
    def __init__(self):
        self.kranges = []
        self.samples = []
        
    
    def __call__(self, p):
        return self.logpost(p)
        
    def add_template(self, k_t, r_t):
        self.k_t = k_t
        self.r_t = BiSpectrum(k_t, r_t)
        
    def add_data(self, k_obs, r_obs, r_cov):
        self.k_obs_ = k_obs*1.
        self.r_obs_ = r_obs*1.
        self.r_cov_ = r_cov*1.
        
    def select_krange(self, kmin=0.00, kmax=0.4):
        self.kranges.append([kmin, kmax])
        
        is_g = (self.k_obs_ > kmin) & (self.k_obs_ < kmax)
        self.is_g = is_g.sum(axis=1) == 3
        
        self.k_obs = self.k_obs_[self.is_g]
        self.r_obs = self.r_obs_[self.is_g]
        self.i_cov = np.linalg.inv(self.r_cov_[self.is_g,:][:, self.is_g])        

    def loglike(self, p):  
        r_m = self.r_t(self.k_obs, p)
        is_g = np.isfinite(r_m)
        if is_g.mean() < 0.5:return -np.inf
        res = r_m[is_g] - self.r_obs[is_g]
        return -0.5*res.dot(self.i_cov[is_g, :][:, is_g].dot(res))
    
    
    def logprior(self, p):
        lp = 0.
        lp += 0. if  0.95 < p[0] < 1.05 else -np.inf
        lp += 0. if  0.8 < p[1] < 1.2 else -np.inf    
        for p_i in p[2:]:
            lp += 0. if  -1. < p_i < 1. else -np.inf
        return lp

    def logpost(self, p):
        return self.logprior(p) + self.loglike(p)
    
    
    def test(self, alphas, color='k', **kw):
        for a in alphas:
            like = self([a, 1.0, 0.00, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print(a, like)
            #plt.scatter(a, -2*like, color=color, **kw) 
            
    def run_mcmc(self):
        
        nwalkers = 22
        ndim   = 11
        nsteps = 1000
        cov = 0.001*np.eye(11)
        best = [1.0, 1.0]+9*[0., ]
        start = np.random.multivariate_normal(best, cov, size=nwalkers)

        with Pool(4) as pool:    
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logpost, pool=pool)
            sampler.run_mcmc(start, nsteps, progress=True)
            
        self.samples.append({'chain':sampler.get_chain(), 
                             'log_prob':sampler.get_log_prob()})
        
    def save(self, path2file):
        np.savez(path2file, **{'samples':self.samples, 'kranges':self.kranges})
    
            
            
    
            
#k_bin, rm_bin = src.utils.bin_bispectrum(self.k[is_good], r_m[is_good])
#k_ix = ((k_bin - k.min()+1.0e-8)/0.01).astype('int')
#residual = (rm_bin - self.r[k_ix])

#is_good = (k_bin > kmin) & (k_bin < kmax)
#is_good = np.isfinite(residual)


#chi2 = residual.dot(np.linalg.inv(rcov[is_good,:][:, is_good]).dot(residual))
#is_finite = ~np.isnan(chi2)
#if is_finite.sum()==0:
#    return -np.inf
#return -0.5*chi2[is_finite].sum()


def load_data(tracer, stat, reduced, template): 
    if stat=='bk':
        m = get_bispectra(tracer)
        if reduced=='raw':
            r = m.b / m.b_smooth.mean(axis=0)
            k = m.k
        elif reduced=='all':
            r = m.b_all/m.bs_all.mean(axis=0)
            k = m.k_all
        elif reduced=='iso':
            r = m.b_iso/m.bs_iso.mean(axis=0)
            k = m.k_iso
        elif reduced=='eqi':
            r = m.b_eqi/m.bs_eqi.mean(axis=0)        
            k = m.k_eqi
    else:
        m = get_powerspectra(tracer)
        r = m.p / m.p_smooth.mean(axis=0)    
        k = m.k

    r_cov_ = np.load(f'/Users/mehdi/github/bispectrum/{stat}_molino.z0.0.fiducial.rcov.npz', allow_pickle=True)
    r_cov = r_cov_[reduced] * np.outer(r.std(axis=0), r.std(axis=0)) 
     
    # --- template: TODO
    k_tem = m.k
    r_tem = m.b.mean(axis=0)/m.b_smooth.mean(axis=0)
    
    # --- measurement
    k_obs = k
    r_obs = r#.mean(axis=0)
    r_cov = r_cov
    
    return (k_obs, r_obs, r_cov), (k_tem, r_tem)



def read_chi2(chain, log_prob, burn_in=200, bins=21):
    x = chain[burn_in:, :, 0].flatten()
    y = -2.*log_prob[burn_in:, :].flatten()
    ym, xb,__ = binned_statistic(x, y, statistic=np.min, bins=bins)
    xm = binned_statistic(x, x, statistic=np.mean, bins=xb)[0]
    return xm, ym    


# def get_cov(y, y_):

#     cov_ = np.cov(y, rowvar=0)
#     std_ = np.diagonal(cov_)**0.5
#     rcov = cov_ / np.outer(std_, std_)    
    
#     std  = np.std(y_, axis=0)
#     return rcov * np.outer(std, std)


# def get_p3(k3, pk_a):
    
#     p3 = []
#     e = 1.0e-6
#     for i, ki in enumerate(k3):

#         ix = int((ki[0]-0.005+e)*100)
#         iy = int((ki[1]-0.005+e)*100)
#         iz = int((ki[2]-0.005+e)*100)
#         #print(i, ki, ix, iy, iz)    
#         p3.append(pk_a[ix]*pk_a[iy]*pk_a[iz])
    
#     return np.array(p3)



# import os
# import numpy as np
# from glob import glob
# #from src.stats import get_cov
# from scipy.stats import binned_statistic

# # path2figs = '/home/mr095415/bisp4desi/figures/'

# # path2mocks =  {
# #                'glam_gal_pk':'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/BAO/Pk_CatshortV.0114.*.DAT',
# #                'glam_gal_bk':'/mnt/data1/BispectrumGLAM/BAO/Bk_CatshortV.0114.*.h5',
# #                'glam_gal_pksmooth':'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/noBAO/Pk_CatshortV.0114.*.DAT',
# #                'glam_gal_bksmooth':'/mnt/data1/BispectrumGLAM/noBAO/Bk_CatshortV.0114.*.h5',
# #                'molino_gal_pk':'/home/lado/Molino/pk_molino.z0.0.fiducial.nbody*.hod0.hdf5',
# #                'molino_gal_bk':'/home/lado/Molino/bk_molino.z0.0.fiducial.nbody*.hod0.hdf5',
# #                'abacus_lrg_pk':'/Users/mehdi/data/AbacusData/pk_LRG*',
# #                'abacus_lrg_bk':'/Users/mehdi/data/AbacusData/bk_LRG*',
# #                'abacus_lrg_bksmooth':'/Users/mehdi/data/Abacus_smooth/all_bk_LRGz0_*.txt',
# #                'abacus_lrg_pksmooth':'/Users/mehdi/data/Abacus_smooth/all_pk_LRGz0_*.txt' 
# #               }


# # def get_name(mock, gal, stat):
# #     return '_'.join([mock, gal, stat])


# # def path2cache(mock, gal, stat):
# #     return ''.join([f'/home/mr095415/bispectrum/cache/', get_name(mock, gal, stat),'.npz'])


# # def savez(file, *args, **kwds):
# #     dirname = os.path.dirname(file)
# #     if not os.path.exists(dirname):
# #         os.makedirs(dirname)
# #     np.savez(file, *args, **kwds)

# # def read_pk(files, iy=1):
# #     """ Reads Power Spectrum Data
# #     """
# #     pk_a = []
# #     for file in files:
# #         d = np.loadtxt(file)
# #         k = d[:30, 0]
# #         pk = d[:30, iy]
        
# #         pk_a.append(pk)
# #     return k, np.array(pk_a)


# # def read_bk(files, iy=4):
# #     """ Reads Power Spectrum Data
# #     """
# #     bk_a = []
# #     for file in files:
# #         d = np.loadtxt(file)
# #         k = d[:, :3]
# #         bk = d[:, iy]
        
# #         bk_a.append(bk)
# #     return k, np.array(bk_a)


# # def read(mock_name, iy):
# #     if 'pk' in mock_name:        
# #         return read_pk(glob(path2mocks[mock_name]), iy)
# #     elif 'bk' in mock_name:
# #         return read_bk(glob(path2mocks[mock_name]), iy)
    
    
# # def read_all(mock, gal, stat, iy):
    
# #     path_ = path2cache(mock, gal, stat)
# #     mock_name =  get_name(mock, gal, stat)
    
# #     if not os.path.exists(path_):
# #         print(f"creating cache with iy={iy}")
# #         x, y = read(mock_name, iy)
# #         savez(path_, **{'x':x, 'y':y})
# #     else:
# #         print(f"loading cache...")
# #         fl = np.load(path_)
# #         x = fl['x']
# #         y = fl['y']
# #     print(y.shape)
# #     return (x, y)
    
    
# # def read_sigmas(chains, ialpha=0):
    
# #     kmax = []
# #     sigma = []
# #     for ch in np.sort(chains):
# #         ch_ = np.load(ch)
# #         kmax.append(ch_['klim'][1])
# #         sigma.append(ch_['chain'][5000:, :, ialpha].std())
        
# #     print(ch_['klim'][0])
# #     return kmax, sigma



# # def read_chi2(d):
# #     alpha_edge = np.linspace(0.95, 1.05, num=21, burn_in=200)
# #     x = d['chain'][:, :, 0].flatten()
# #     y = -2.*d['log_prob'].flatten()

# #     ym = binned_statistic(x, y, statistic=np.min, bins=alpha_edge)[0]
# #     xm = binned_statistic(x, x, statistic=np.mean, bins=alpha_edge)[0]
# #     return xm, ym

# # def read_chi2list(ds):
# #     kmax = []
# #     chi2s = []
# #     for d_ in ds:
# #         di = np.load(d_)
# #         kmax.append(di['klim'][1])
# #         chi2s.append(read_chi2(di))
# #     return kmax, chi2s


# # class Preparer(object):
    
# #     def __init__(self):
# #         # read Molino
# #         self.k, self.pk_gal_molino = read_all('molino', 'gal', 'pk', 1)
# #         self.k3, self.bk_gal_molino = read_all('molino', 'gal', 'bk', 3)
# #         self.c_molino = np.column_stack([self.pk_gal_molino, self.bk_gal_molino])    
# #         print('Done reading Molino')
        
# #     def prep(self, mock, gal):

# #         i = 2 if mock == 'abacus' else 1
# #         j = 4 if mock == 'abacus' else 3

# #         __, pk_gal_glam = read_all(mock, gal, 'pk', 1)
# #         __, pksmooth_gal_glam = read_all(mock, gal, 'pksmooth', i)
# #         __, bk_gal_glam = read_all(mock, gal, 'bk', 3)
# #         __, bksmooth_gal_glam = read_all(mock, gal, 'bksmooth', j)

# #         c_glam = np.column_stack([pk_gal_glam/pksmooth_gal_glam.mean(axis=0),
# #                                   bk_gal_glam/bksmooth_gal_glam.mean(axis=0)])

# #         Cp_glam = get_cov(self.pk_gal_molino, pk_gal_glam/pksmooth_gal_glam.mean(axis=0))
# #         Cb_glam = get_cov(self.bk_gal_molino, bk_gal_glam/bksmooth_gal_glam.mean(axis=0))
# #         Cc_glam = get_cov(self.c_molino, c_glam)

# #         savez(path2cache(mock, gal, 'bkcov'), **{'x':self.k3, 'y':Cb_glam})                
# #         savez(path2cache(mock, gal, 'bkmean'), **{'x':self.k3, 'y':bk_gal_glam.mean(axis=0)})
# #         savez(path2cache(mock, gal, 'bksmoothmean'), **{'x':self.k3, 'y':bksmooth_gal_glam.mean(axis=0)})
# #         savez(path2cache(mock, gal, 'pkcov'), **{'x':self.k, 'y':Cp_glam})                
# #         savez(path2cache(mock, gal, 'pkmean'), **{'x':self.k, 'y':pk_gal_glam.mean(axis=0)})
# #         savez(path2cache(mock, gal, 'pksmoothmean'), **{'x':self.k, 'y':pksmooth_gal_glam.mean(axis=0)})
# #         savez(path2cache(mock, gal, 'pbcov'), **{'x':[None, ], 'y':Cc_glam})   
# #         print(f"Done writing mean and covariances for mock={mock} and gal={gal}")
    
    
# # class Spectrum:
# #     def __init__(self, filename, **kw):
# #         file = np.load(filename, **kw)
# #         self.x = file['x']
# #         self.y = file['y']
