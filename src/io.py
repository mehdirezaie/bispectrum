import os
import numpy as np
from glob import glob
from src.stats import get_cov
from scipy.stats import binned_statistic

path2figs = '/home/mr095415/bisp4desi/figures/'

path2mocks =  {
               'glam_gal_pk':'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/BAO/Pk_CatshortV.0114.*.DAT',
               'glam_gal_bk':'/mnt/data1/BispectrumGLAM/BAO/Bk_CatshortV.0114.*.h5',
               'glam_gal_pksmooth':'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/noBAO/Pk_CatshortV.0114.*.DAT',
               'glam_gal_bksmooth':'/mnt/data1/BispectrumGLAM/noBAO/Bk_CatshortV.0114.*.h5',
               'molino_gal_pk':'/home/lado/Molino/pk_molino.z0.0.fiducial.nbody*.hod0.hdf5',
               'molino_gal_bk':'/home/lado/Molino/bk_molino.z0.0.fiducial.nbody*.hod0.hdf5',
               'abacus_lrg_pk':'/Users/mehdi/data/AbacusData/pk_LRG*',
               'abacus_lrg_bk':'/Users/mehdi/data/AbacusData/bk_LRG*',
               'abacus_lrg_bksmooth':'/Users/mehdi/data/Abacus_smooth/all_bk_LRGz0_*.txt',
               'abacus_lrg_pksmooth':'/Users/mehdi/data/Abacus_smooth/all_pk_LRGz0_*.txt' 
              }


def get_name(mock, gal, stat):
    return '_'.join([mock, gal, stat])


def path2cache(mock, gal, stat):
    return ''.join([f'/home/mr095415/bispectrum/cache/', get_name(mock, gal, stat),'.npz'])


def savez(file, *args, **kwds):
    dirname = os.path.dirname(file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savez(file, *args, **kwds)

def read_pk(files, iy=1):
    """ Reads Power Spectrum Data
    """
    pk_a = []
    for file in files:
        d = np.loadtxt(file)
        k = d[:30, 0]
        pk = d[:30, iy]
        
        pk_a.append(pk)
    return k, np.array(pk_a)


def read_bk(files, iy=4):
    """ Reads Power Spectrum Data
    """
    bk_a = []
    for file in files:
        d = np.loadtxt(file)
        k = d[:, :3]
        bk = d[:, iy]
        
        bk_a.append(bk)
    return k, np.array(bk_a)


def read(mock_name, iy):
    if 'pk' in mock_name:        
        return read_pk(glob(path2mocks[mock_name]), iy)
    elif 'bk' in mock_name:
        return read_bk(glob(path2mocks[mock_name]), iy)
    
    
def read_all(mock, gal, stat, iy):
    
    path_ = path2cache(mock, gal, stat)
    mock_name =  get_name(mock, gal, stat)
    
    if not os.path.exists(path_):
        print(f"creating cache with iy={iy}")
        x, y = read(mock_name, iy)
        savez(path_, **{'x':x, 'y':y})
    else:
        print(f"loading cache...")
        fl = np.load(path_)
        x = fl['x']
        y = fl['y']
    print(y.shape)
    return (x, y)
    
    
def read_sigmas(chains, ialpha=0):
    
    kmax = []
    sigma = []
    for ch in np.sort(chains):
        ch_ = np.load(ch)
        kmax.append(ch_['klim'][1])
        sigma.append(ch_['chain'][5000:, :, ialpha].std())
        
    print(ch_['klim'][0])
    return kmax, sigma


def read_chi2(d):
    alpha_edge = np.linspace(0.95, 1.05, num=21)
    x = d['chain'][:, :, 0].flatten()
    y = -2.*d['log_prob'].flatten()

    ym = binned_statistic(x, y, statistic=np.min, bins=alpha_edge)[0]
    xm = binned_statistic(x, x, statistic=np.mean, bins=alpha_edge)[0]
    return xm, ym

def read_chi2list(ds):
    kmax = []
    chi2s = []
    for d_ in ds:
        di = np.load(d_)
        kmax.append(di['klim'][1])
        chi2s.append(read_chi2(di))
    return kmax, chi2s


class Preparer(object):
    
    def __init__(self):
        # read Molino
        self.k, self.pk_gal_molino = read_all('molino', 'gal', 'pk', 1)
        self.k3, self.bk_gal_molino = read_all('molino', 'gal', 'bk', 3)
        self.c_molino = np.column_stack([self.pk_gal_molino, self.bk_gal_molino])    
        print('Done reading Molino')
        
    def prep(self, mock, gal):

        i = 2 if mock == 'abacus' else 1
        j = 4 if mock == 'abacus' else 3

        __, pk_gal_glam = read_all(mock, gal, 'pk', 1)
        __, pksmooth_gal_glam = read_all(mock, gal, 'pksmooth', i)
        __, bk_gal_glam = read_all(mock, gal, 'bk', 3)
        __, bksmooth_gal_glam = read_all(mock, gal, 'bksmooth', j)

        c_glam = np.column_stack([pk_gal_glam/pksmooth_gal_glam.mean(axis=0),
                                  bk_gal_glam/bksmooth_gal_glam.mean(axis=0)])

        Cp_glam = get_cov(self.pk_gal_molino, pk_gal_glam/pksmooth_gal_glam.mean(axis=0))
        Cb_glam = get_cov(self.bk_gal_molino, bk_gal_glam/bksmooth_gal_glam.mean(axis=0))
        Cc_glam = get_cov(self.c_molino, c_glam)

        savez(path2cache(mock, gal, 'bkcov'), **{'x':self.k3, 'y':Cb_glam})                
        savez(path2cache(mock, gal, 'bkmean'), **{'x':self.k3, 'y':bk_gal_glam.mean(axis=0)})
        savez(path2cache(mock, gal, 'bksmoothmean'), **{'x':self.k3, 'y':bksmooth_gal_glam.mean(axis=0)})
        savez(path2cache(mock, gal, 'pkcov'), **{'x':self.k, 'y':Cp_glam})                
        savez(path2cache(mock, gal, 'pkmean'), **{'x':self.k, 'y':pk_gal_glam.mean(axis=0)})
        savez(path2cache(mock, gal, 'pksmoothmean'), **{'x':self.k, 'y':pksmooth_gal_glam.mean(axis=0)})
        savez(path2cache(mock, gal, 'pbcov'), **{'x':[None, ], 'y':Cc_glam})   
        print(f"Done writing mean and covariances for mock={mock} and gal={gal}")
    
    
class Spectrum:
    def __init__(self, filename, **kw):
        file = np.load(filename, **kw)
        self.x = file['x']
        self.y = file['y']