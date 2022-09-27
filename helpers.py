
import os
import numpy as np

from glob import glob
from scipy.interpolate import LinearNDInterpolator, interp1d


#
#--- global variables
#
pk_molino = './files/pk_molino.npz'
pk_glam   = './files/pk_glam.npz'
pk_glam_nobao = './files/pk_glam_nobao.npz'

bk_molino = './files/bk_molino.npz'
bk_glam   = './files/bk_glam.npz'
bk_glam_nobao = './files/bk_glam_nobao.npz'


class PkModel(object):
    
    def __init__(self, x, y):
        self.y_int = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    
    def __call__(self, x, params): # 7 params
        return params[1]*self.y_int(params[0]*x) + params[2] \
                + params[3]*x \
                + params[4]*(1./x) \
                + params[5]*(x*x) \
                + params[6]*(1./(x*x))


class BkModel(object):
    def __init__(self, x, y):
        self.y_int = LinearNDInterpolator(x, y, fill_value=np.nan)
    
    def __call__(self, x, params): # 11 params
        return params[1]*self.y_int(params[0]*x) + params[2] \
                + params[3]*(x[:, 0] + x[:, 1] + x[:, 2]) \
                + params[4]*(1./x[:, 0] + 1./x[:, 1] + 1/x[:, 2]) \
                + params[5]*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) \
                + params[6]*(1./x[:, 0]**2 + 1./x[:, 1]**2 + 1./x[:, 2]**2) \
                + params[7]*(x[:, 0]*x[:, 1] + x[:, 0]*x[:, 2] + x[:, 1]*x[:, 2]) \
                + params[8]*(1./(x[:, 0]*x[:, 1]) + 1./(x[:, 0]*x[:, 2]) + 1./(x[:, 1]*x[:, 2])) \
                + params[9]*(x[:, 0]*x[:, 1]/x[:, 2] + x[:, 0]*x[:, 2]/x[:, 1] + x[:, 1]*x[:, 2]/x[:, 0]) \
                + params[10]*(x[:, 2]/(x[:, 0]*x[:, 1]) + x[:, 1]/(x[:, 0]*x[:, 2]) + x[:, 0]/(x[:, 1]*x[:, 2]))
    

def get_cov(y_molino, y_glam, plot=False):
    
    cov_ = np.cov(y_molino, rowvar=0)
    std_ = np.diagonal(cov_)**0.5
    rcov = cov_ / np.outer(std_, std_)
    
    #plt.imshow(rcov, vmin=0, vmax=1, cmap='jet', origin='lower')
    #plt.colorbar()
    std = np.std(y_glam, axis=0)
    cov = rcov * np.outer(std, std)
    
    #if plot:
    #    vmin, vmax = np.percentile(cov, [1, 99])
    #    plt.imshow(cov, vmin=vmin, vmax=vmax, origin='lower')        
    return cov


def logprior(theta):
    ''' The natural logarithm of the prior probability. '''
    lp = 0.
    lp += 0. if  0.9 < theta[0] < 1.1 else -np.inf
    lp += 0. if  0.8 < theta[1] < 1.2 else -np.inf
    
    lp += 0. if -0.01 < theta[2]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[3]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[4]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[5]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[6]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[7]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[8]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[9]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[10] < 0.01 else -np.inf
    
    lp += 0. if  0.8  < theta[11] < 1.2   else -np.inf
    lp += 0. if -0.01 < theta[12] < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[13] < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[14] < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[15] < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[16] < 0.01 else -np.inf
    
    ## Gaussian prior on ?
    #mmu = 3.     # mean of the Gaussian prior
    #msigma = 10. # standard deviation of the Gaussian prior
    #lp += -0.5*((m - mmu)/msigma)**2

    return lp 


def savez(output_file, **kwargs): 
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):os.makedirs(dirname)
    np.savez(output_file, **kwargs)
    print(f'# saved {output_file}')
    
    
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
