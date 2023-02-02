import os
import numpy as np
from glob import glob
from scipy.stats import binned_statistic

path2figs = '/home/mr095415/bisp4desi/figures/'


path2cache = {
               'glam_pk_bao':'./cache/glam_pk_bao.npz',
               'glam_pk_nobao':'./cache/glam_pk_nobao.npz',
               'glam_bk_bao':'./cache/glam_bk_bao.npz',
               'glam_bk_nobao':'./cache/glam_bk_nobao.npz',
               'molino_pk':'./cache/molino_pk_bao.npz',
               'molino_bk':'./cache/molino_bk_bao.npz',
               'abacus_pk':'./cache/abacus_pk_bao.npz',
               'abacus_bk':'./cache/abacus_bk_bao.npz', 
               'abacus_pk_nobao':'./cache/abacus_pk_nobao.npz',     
               'abacus_bk_nobao':'./cache/abacus_bk_nobao.npz',         
              }

path2mocks =  {
               'glam_pk_bao':'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/BAO/Pk_CatshortV.0114.*.DAT',
               'glam_bk_bao':'/mnt/data1/BispectrumGLAM/BAO/Bk_CatshortV.0114.*.h5',
               'glam_pk_nobao':'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/noBAO/Pk_CatshortV.0114.*.DAT',
               'glam_bk_nobao':'/mnt/data1/BispectrumGLAM/noBAO/Bk_CatshortV.0114.*.h5',
               'molino_pk':'/home/lado/Molino/pk_molino.z0.0.fiducial.nbody*.hod0.hdf5',
               'molino_bk':'/home/lado/Molino/bk_molino.z0.0.fiducial.nbody*.hod0.hdf5',
               'abacus_pk':'/mnt/data1/AbacusBisp*/pk_LRG*',
               'abacus_bk':'/mnt/data1/AbacusBisp*/bk_LRG*',
               'abacus_bk_nobao':'/mnt/data1/Abacus_All/all_bispectrum/all_bk_LRGz0_*.txt',
               'abacus_pk_nobao':'/mnt/data1/Abacus_All/all_pk_LRGz0_*.txt' 
              }


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


def read(mock, iy):
    if 'pk' in mock:        
        return read_pk(glob(path2mocks[mock]), iy)
    elif 'bk' in mock:
        return read_bk(glob(path2mocks[mock]), iy)

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


class DataLoader:
    
    def __init__(self):
        pass
    
    def load(self, mock, iy):
        assert mock in path2cache, f"{mock} not processed"
        
        path2file = path2cache[mock]
        
        if not os.path.exists(path2file):
            print(f"creating cache with iy={iy}")
            x, y = read(mock, iy)
            np.savez(path2file, **{'x':x, 'y':y})
        else:
            print(f"loading cache...")
            fl = np.load(path2file)
            x = fl['x']
            y = fl['y']
        print(y.shape)
        return (x, y)
    
    def prep_data(self):

        from .stats import get_p3, get_cov

        k, pk_glam = self.load('glam_pk_bao', 1)
        __, pk_glam_nobao = self.load('glam_pk_nobao', 1)
        __, pk_molino = self.load('molino_pk', 1)
        __, pk_abacus = self.load('abacus_pk', 1)
        __, pk_abacus_nobao = self.load('abacus_pk_nobao', 2)

        k3, bk_glam = self.load('glam_bk_bao', 3)
        __, bk_glam_nobao = self.load('glam_bk_nobao', 3)
        __, bk_molino = self.load('molino_bk', 3)
        __, bk_abacus = self.load('abacus_bk', 3)
        __, bk_abacus_nobao = self.load('abacus_bk_nobao', 4)

        p3_abacus = get_p3(k3, pk_abacus.mean(axis=0))
        p3_molino = get_p3(k3, pk_molino.mean(axis=0))
        p3_glam   = get_p3(k3, pk_glam.mean(axis=0))
        
        c_abacus = np.column_stack([pk_abacus/pk_abacus_nobao.mean(axis=0), bk_abacus/bk_abacus_nobao.mean(axis=0)])
        c_glam = np.column_stack([pk_glam/pk_glam_nobao.mean(axis=0), bk_glam/bk_glam_nobao.mean(axis=0)])
        c_molino = np.column_stack([pk_molino, bk_molino])
        
        #cov_p     = np.cov(pk_molino, rowvar=False)/(pk_molino.var(axis=0)/pk_molino.mean(axis=0)**2)
        #Cp_glam   = (pk_glam.var(axis=0)/pk_glam.mean(axis=0)**2) * cov_p
        #Cp_abacus = (pk_abacus.var(axis=0)/pk_abacus.mean(axis=0)**2*8**1.5) * cov_p
        Cp_glam = get_cov(pk_molino, pk_glam/pk_glam_nobao.mean(axis=0))
        Cp_abacus = get_cov(pk_molino, pk_abacus/pk_abacus_nobao.mean(axis=0))

        #cov_b     = np.cov(bk_molino, rowvar=False) / (bk_molino.var(axis=0)/p3_molino)
        #Cb_glam   = (bk_glam.var(axis=0)/p3_glam)*cov_b
        #Cb_abacus = (bk_abacus.var(axis=0)/p3_abacus*8**1.5)*cov_b
        Cb_glam = get_cov(bk_molino, bk_glam/bk_glam_nobao.mean(axis=0))
        Cb_abacus = get_cov(bk_molino, bk_abacus/bk_abacus_nobao.mean(axis=0))
        
        Cc_glam = get_cov(c_molino, c_glam)
        Cc_abacus = get_cov(c_molino, c_abacus)
        
        # ABACUS
        np.savez('cache/bk_cov_abacus_bao.npz', **{'x':k3, 'y':Cb_abacus})        
        np.savez('cache/bk_mean_abacus_bao.npz', **{'x':k3, 'y':bk_abacus.mean(axis=0)})
        np.savez('cache/bk_mean_abacus_nobao.npz', **{'x':k3, 'y':bk_abacus_nobao.mean(axis=0)})
        np.savez('cache/pk_cov_abacus_bao.npz', **{'x':k, 'y':Cp_abacus})                
        np.savez('cache/pk_mean_abacus_bao.npz', **{'x':k, 'y':pk_abacus.mean(axis=0)})
        np.savez('cache/pk_mean_abacus_nobao.npz', **{'x':k, 'y':pk_abacus_nobao.mean(axis=0)})
        np.savez('cache/pk3_mean_abacus_bao.npz', **{'x':k3, 'y':p3_abacus})
        np.savez('cache/pb_cov_abacus_bao.npz', **{'x':[k, k3], 'y':Cc_abacus})
        
        # GLAM
        np.savez('cache/bk_cov_glam_bao.npz', **{'x':k3, 'y':Cb_glam})                
        np.savez('cache/bk_mean_glam_bao.npz', **{'x':k3, 'y':bk_glam.mean(axis=0)})
        np.savez('cache/bk_mean_glam_nobao.npz', **{'x':k3, 'y':bk_glam_nobao.mean(axis=0)})
        np.savez('cache/pk_cov_glam_bao.npz', **{'x':k, 'y':Cp_glam})                
        np.savez('cache/pk_mean_glam_bao.npz', **{'x':k, 'y':pk_glam.mean(axis=0)})
        np.savez('cache/pk_mean_glam_nobao.npz', **{'x':k, 'y':pk_glam_nobao.mean(axis=0)})
        np.savez('cache/pk3_mean_glam_bao.npz', **{'x':k3, 'y':p3_glam})
        np.savez('cache/pb_cov_glam_bao.npz', **{'x':[k, k3], 'y':Cc_glam})        
        
        # MOLINO
        np.savez('cache/bk_mean_molino_bao.npz', **{'x':k3, 'y':bk_molino.mean(axis=0)})
        np.savez('cache/pk_mean_molino_bao.npz', **{'x':k, 'y':pk_molino.mean(axis=0)})
        np.savez('cache/pk3_mean_molino_bao.npz', **{'x':k3, 'y':p3_molino})
        print('data is prepared')
    
    
class Spectrum:
    def __init__(self, filename, **kw):
        file = np.load(filename, **kw)
        self.x = file['x']
        self.y = file['y']    
