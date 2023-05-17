import numpy as np
from scipy.stats import binned_statistic


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


class SpectrumData:
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

    
        
        #     # select triangles
#     if which=='all':
#         is_good = np.ones(k.shape[0], '?')
#     elif which=='iso':
#         is_good = get_isoceles(k)
#         k = k[is_good]
#         r = r[:, is_good]
#     elif which=='equ':
#         is_good = get_equilateral(k)
#         k = k[is_good]
#         r = r[:, is_good]
#     else:
#         raise NotImplementedError(f'which={which} not implemented')
        
#     # marginalize over k2 and k3
#     k_bin, r_bin = bin_bispectra(k, r)
#     rcov = np.cov(r_bin, rowvar=False)
#     rstd = np.diagonal(rcov)**0.5
    
#     print(f'r.shape: {r_bin.shape}')
        

def get_bispectra(tracer, verbose=False, which='all'):
    """ Get Data """ 
    print(f'tracer: {tracer}, which: {which}')
    mock_range = {'LRGz0':(8000, 8025),
                  'ELGz1':(1000, 1025),
                  'QSOz2':(5000, 5025)}    
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
    
    return SpectrumData(k, b, b_bestfit, b_smooth)

