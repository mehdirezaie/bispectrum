import os
import numpy as np
from glob import glob


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
    
    