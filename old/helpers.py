
import os
import numpy as np

from glob import glob
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt 

plt.rc('figure', **{'facecolor':'w'})
plt.rc('font', size=13, family='Sans')

#
#--- global variables
#
pk_molino = './files/pk_molino.npz'
pk_glam   = './files/pk_glam.npz'
pk_glam_nobao = './files/pk_glam_nobao.npz'
pk_abacus = './files/pk_abacus.npz'


bk_molino = './files/bk_molino.npz'
bk_glam   = './files/bk_glam.npz'
bk_glam_nobao = './files/bk_glam_nobao.npz'
bk_abacus = './files/bk_abacus.npz'

path4figs = '/home/mr095415/bisp4desi/figures/'


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
    
    return lp 

def logpriorconst(theta):
    ''' The natural logarithm of the prior probability. '''
    lp = 0.
    lp += 0. if  0.9 < theta[0] < 1.1 else -np.inf
    lp += 0. if  0.8 < theta[1] < 1.2 else -np.inf
    
    lp += 0. if -0.01 < theta[2]  < 0.01 else -np.inf
    lp += 0. if  0.8  < theta[3] < 1.2   else -np.inf
    lp += 0. if -0.01 < theta[4] < 0.01 else -np.inf
    
    return lp 


def logpriorbk(theta):
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
    
    return lp 




def logpriorpk(theta):
    ''' The natural logarithm of the prior probability. '''
    lp = 0.
    lp += 0. if  0.9 < theta[0] < 1.1 else -np.inf
    lp += 0. if  0.8 < theta[1] < 1.2 else -np.inf 
    lp += 0. if -0.01 < theta[2]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[3]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[4]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[5]  < 0.01 else -np.inf
    lp += 0. if -0.01 < theta[6]  < 0.01 else -np.inf
    
    return lp 

def logpriorpkconst(theta):
    ''' The natural logarithm of the prior probability. '''
    lp = 0.
    lp += 0. if  0.9 < theta[0] < 1.1 else -np.inf
    lp += 0. if  0.8 < theta[1] < 1.2 else -np.inf 
    lp += 0. if -0.01 < theta[2]  < 0.01 else -np.inf
    return lp 

logpriorbkconst = logpriorpkconst

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
        savez(pk_molino, **ret)
    else:
        ret = np.load(pk_molino)
    
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
        savez(bk_molino, **ret)
    else:
        ret = np.load(bk_molino)
        
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
        savez(pk_glam, **ret)
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
        savez(pk_glam_nobao, **ret)
    else:
        ret = np.load(pk_glam_nobao)
    
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
        savez(bk_glam, **ret)
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
        savez(bk_glam_nobao, **ret)
    else:
        ret = np.load(bk_glam_nobao)    
    
    return ret


def read_abacus_bk():
    
    if not os.path.exists(bk_abacus):        
        pk = []
        files = glob('/mnt/data1/AbacusBisp*/bk_LRG*')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 3:])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, :3], pk=pk)
        savez(bk_abacus, **ret)
    else:
        ret = np.load(bk_abacus)    
    
    return ret


def read_abacus_pk():
    
    if not os.path.exists(pk_abacus):        
        pk = []
        files = glob('/mnt/data1/AbacusBisp*/pk_LRG*')
        for i, fl in enumerate(files):
            pk_ = np.loadtxt(fl)
            pk.append(pk_[:, 1:])
            if i % (len(files)//4) == 0:print(f'{i:5d}/{len(files):5d} done')

        pk = np.array(pk)
        print(pk.shape)
        ret = dict(k=pk_[:, 0], pk=pk)
        savez(pk_abacus, **ret)
    else:
        ret = np.load(pk_abacus)
    
    return ret





def extract_sigma(fl):
    kmax = float(fl.split('kmax')[1].split('_')[1].split('.npz')[0])
    ch = np.load(fl)

    chain = ch['chain']
    nsteps, __, ndim = chain.shape
    rhat = gelman_rubin(chain[nsteps//2:, :, :])
    #if not np.all(rhat < 1.1):
    #    print(rhat, fl)
    #good_cols = ((chain[nsteps//2:, :, 0] > 1.2) | (chain[nsteps//2:, :, 0] < 0.8)).sum(axis=0) == 0
    
    #if good_cols.sum() == 0:
    #    return kmax, np.nan
    #else:        
    chainr = chain[nsteps//2:, :, 0].flatten()
    #print(chainr.shape)
    prcnt = np.percentile(chainr, [16, 84])
    sigma = 0.5*(prcnt[1]-prcnt[0])

    return kmax, sigma


def loop_files(files):
    sigmas_pk = []
    for fl in files:
        sigmas_pk.append(extract_sigma(fl))
    
    return np.array(sigmas_pk).T


class PkModel(object):
    
    def __init__(self, x, y):
        self.y_int = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    
    def __call__(self, x, params): # 7 params
        return params[1]*self.y_int(params[0]*x) + params[2] \
                + params[3]*x \
                + params[4]*(1./x) \
                + params[5]*(x*x) \
                + params[6]*(1./(x*x))
    
    
class PkModelnoBAO(object):
    
    def __init__(self, x, y):
        pass
    def __call__(self, x, params): # 7 params
        return params[2] \
                + params[3]*x \
                + params[4]*(1./x) \
                + params[5]*(x*x) \
                + params[6]*(1./(x*x))    
    

class PkModelConst(object):
    
    def __init__(self, x, y):
        self.y_int = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    
    def __call__(self, x, params): # 7 params
        return params[1]*self.y_int(params[0]*x) + params[2]
    

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
    
class BkModelnoBAO(object):
    def __init__(self, x, y):
        pass 

    def __call__(self, x, params): # 11 params
        return params[2] \
                + params[3]*(x[:, 0] + x[:, 1] + x[:, 2]) \
                + params[4]*(1./x[:, 0] + 1./x[:, 1] + 1/x[:, 2]) \
                + params[5]*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) \
                + params[6]*(1./x[:, 0]**2 + 1./x[:, 1]**2 + 1./x[:, 2]**2) \
                + params[7]*(x[:, 0]*x[:, 1] + x[:, 0]*x[:, 2] + x[:, 1]*x[:, 2]) \
                + params[8]*(1./(x[:, 0]*x[:, 1]) + 1./(x[:, 0]*x[:, 2]) + 1./(x[:, 1]*x[:, 2])) \
                + params[9]*(x[:, 0]*x[:, 1]/x[:, 2] + x[:, 0]*x[:, 2]/x[:, 1] + x[:, 1]*x[:, 2]/x[:, 0]) \
                + params[10]*(x[:, 2]/(x[:, 0]*x[:, 1]) + x[:, 1]/(x[:, 0]*x[:, 2]) + x[:, 0]/(x[:, 1]*x[:, 2]))    
    
    
class BkModelConst(object):
    def __init__(self, x, y):
        self.y_int = LinearNDInterpolator(x, y, fill_value=np.nan)
    
    def __call__(self, x, params): # 11 params
        return params[1]*self.y_int(params[0]*x) + params[2]


def plot_sigma_kmax():
    
    path4figs_ = os.path.join(path4figs, 'sigma_kmax.pdf')
    
    sigmas_pk_cons = loop_files(glob('mcmcs/mcmc_is_pk_*const*'))
    sigmas_pk_quad = loop_files(glob('mcmcs/mcmc_is_pk_*quad*'))
    sigmas_bk_quad = loop_files(glob('mcmcs/mcmc_is_bk_*quad*'))
    sigmas_bk_cons = loop_files(glob('mcmcs/mcmc_is_bk_*const*'))
    sigmas_jt_quad = loop_files(glob('mcmcs/mcmc_is_joint_*quad*'))
    sigmas_jt_cons = loop_files(glob('mcmcs/mcmc_is_joint_*const*'))

    plt.figure(figsize=(8, 6))
    plt.axes(xlabel=r'$k_{\rm max}$', 
             ylabel=r'$\sigma (\alpha)$')

    plt.scatter(*sigmas_pk_cons, marker='o', color='C0', alpha=0.4, label='Power Spectrum [const]')
    plt.scatter(*sigmas_pk_quad, marker='.', color='C0', alpha=0.8, label='Power Spectrum [quad]')
    plt.scatter(*sigmas_bk_cons, marker='v', color='C1', alpha=0.4, label='Bispectrum [const]')
    plt.scatter(*sigmas_bk_quad, marker='1', color='C1', alpha=0.8, label='Bispectrum [quad]')
    plt.scatter(*sigmas_jt_cons, marker='P', color='C2', alpha=0.4, label='Joint [const]')
    plt.scatter(*sigmas_jt_quad, marker='+', color='C2', alpha=0.8, label='Joint [quad]')

    leg = plt.legend(frameon=False)
    for i, text in enumerate(leg.get_texts()):
        tx = text.get_text()
        if 'Power' in tx:
            c = 'C0'
        elif 'Bi' in tx:
            c = 'C1'
        elif 'Joi' in tx:
            c = 'C2'

        text.set_color(c)

    plt.savefig(path4figs_, bbox_inches='tight')    
    
    
def plot_spectra_glam():
    path4figs_1 = os.path.join(path4figs, 'glam_spec.pdf')
    path4figs_2 = os.path.join(path4figs, 'glam_cov.pdf')
    
    bk_mol = read_molino_bk()
    pk_mol = read_molino_pk()
    pk_glm = read_glam_pk()
    bk_glm = read_glam_bk()
    bk_glm_nobao = read_glam_bk_nobao()
    pk_glm_nobao = read_glam_pk_nobao()

    kp = pk_glm['k']
    kb = bk_glm['k']
    p0_mol = pk_mol['pk'][:, :, 0]
    b0_mol = bk_mol['pk'][:, :, 0]
    p0_glm = pk_glm['pk']
    b0_glm = bk_glm['pk']
    p0_glm_nobao = pk_glm_nobao['pk']
    b0_glm_nobao = bk_glm_nobao['pk']

    rp = p0_glm/p0_glm_nobao.mean(axis=0)
    rb = b0_glm/b0_glm_nobao.mean(axis=0)
    cov1 = get_cov(p0_mol, rp, False)
    cov2 = get_cov(b0_mol, rb, False)    
    
    fg, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    # for i in np.random.choice(np.arange(1098), size=10, replace=False):
    #     ax1.plot(kp, rp[i, :], lw=1, color='grey', alpha=0.1)
    ax1.plot(kp, rp.mean(axis=0),ls='None', marker='.', color='C0', alpha=0.8)    
    ax1.set(ylim=(0.75, 1.25), xlabel=r'k [h Mpc$^{-1}$]',
            ylabel=r'Power Spectrum Ratio')
    # ix = np.arange(rb.shape[1])
    # for i in np.random.choice(np.arange(1098), size=10, replace=False):
    #     ax2.scatter(ix, rb[i, :], color='grey', alpha=0.1)
    ax2.plot(rb.mean(axis=0),ls='None', marker='.', color='C1', alpha=0.5)    
    ax2.set(ylim=(0.75, 1.25), xlabel=r'triangle index', ylabel=r'Bispectrum Ratio')
    fg.savefig(path4figs_1, bbox_inches='tight')    
    
    vmin, vmax = np.percentile(cov1, [1, 99])
    fg, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10.95, 5))
    fg.subplots_adjust(wspace=0.25)
    ax1.imshow(cov1, origin='lower', vmin=vmin, vmax=vmax,
               extent=(0., 0.3, 0., 0.3,))
    ax1.set(xlabel='$k$', ylabel='$k$')
    ax1.axis('equal')

    vmin, vmax = np.percentile(cov2, [1, 99])
    ax2.imshow(cov2, origin='lower', vmin=vmin, vmax=vmax,
               extent=(0., 2600, 0., 2600,))
    ax2.set(xlabel='triangle index', ylabel='triangle index')
    ax2.axis('equal')
    fg.savefig(path4figs_2, bbox_inches='tight')
    
    
def plot_spectra():
    bk_g = read_glam_bk()
    pk_g = read_glam_pk()
    bk_m = read_molino_bk()
    pk_m = read_molino_pk()
    bk_a = read_abacus_bk()
    pk_a = read_abacus_pk()
    
    
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    ax1.plot(pk_m['k'], pk_m['pk'][:, :, 0].mean(axis=0), label='Molino')
    ax1.plot(pk_m['k'], 30*pk_m['pk'][:, :, 0].mean(axis=0), label='30xMolino', color='C0', ls='--')
    ax1.plot(pk_g['k'], pk_g['pk'].mean(axis=0), label='Glam', alpha=0.5, lw=4)
    ax1.plot(pk_a['k'], pk_a['pk'][:, :, 0].mean(axis=0), label='Abacus')

    ax1.set(xlabel='k', ylabel='P0(k)', yscale='log')
    ax1.legend()
    
    ax2.plot(bk_m['pk'][:, :, 0].mean(axis=0), label='Molino')
    ax2.plot(30*bk_m['pk'][:, :, 0].mean(axis=0), label='30xMolino', color='C0', ls='--')
    ax2.plot(bk_g['pk'].mean(axis=0), label='Glam', alpha=0.5, lw=4)
    ax2.plot(bk_a['pk'][:, :, 0].mean(axis=0), label='Abacus')

    ax2.set(xlabel='triangle index', ylabel='B0(k)', yscale='log')        
    ax2.legend()
    
    
def plot_ratios():
    vratio = 8.0
    path4figs_ = os.path.join(path4figs, 'glam_abacus_molino_ratio.pdf')
    
    bk_g = read_glam_bk()
    pk_g = read_glam_pk()
    bk_m = read_molino_bk()
    pk_m = read_molino_pk()
    bk_a = read_abacus_bk()
    pk_a = read_abacus_pk()

    cov_bg = np.cov(bk_g['pk'], rowvar=0)
    cov_pg = np.cov(pk_g['pk'], rowvar=0)
    cov_bm = np.cov(bk_m['pk'][:, :, 0], rowvar=0)
    cov_pm = np.cov(pk_m['pk'][:, :, 0], rowvar=0)
    cov_ba = np.cov(bk_a['pk'][:, :, 0], rowvar=0)
    cov_pa = np.cov(pk_a['pk'][:, :, 0], rowvar=0)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex='col', sharey='col')
    fig.subplots_adjust(hspace=0.02)
    ax = ax.flatten()
    
    ax[0].plot(cov_pg.diagonal()/cov_pm.diagonal(), alpha=0.8, lw=3)
    ax[0].plot((pk_g['pk'].mean(axis=0)/pk_m['pk'][:, :, 0].mean(axis=0))**2, ls='-', alpha=0.8, lw=1)
    ax[0].set(yscale='log')
    ax[0].legend(title='Glam/Molino', frameon=False)
    #lgnd = ax1.legend(title='Glam/Molino', frameon=False, fontsize=12)
    #for i, tx in enumerate(lgnd.get_texts()):tx.set_color('C%d'%i)
    ax[0].text(10, 3000., 'ratio of cov', color='C0')
    ax[0].text(10, 700., 'ratio of amplitudes^2', color='C1')

    ax[2].plot(cov_pa.diagonal()/cov_pm.diagonal(), lw=3, alpha=0.8)
    ax[2].plot((pk_a['pk'][:, :, 0].mean(axis=0)/pk_m['pk'][:, :, 0].mean(axis=0))**2/vratio, ls='-', alpha=0.8, lw=1)
    ax[2].set(yscale='log', xlabel='k-bin index')
    ax[2].legend(title='Abacus/Molino', frameon=False) 
    
    
    ax[1].plot(cov_bg.diagonal()/cov_bm.diagonal(), alpha=0.8, lw=3)
    ax[1].plot((bk_g['pk'].mean(axis=0)/bk_m['pk'][:, :, 0].mean(axis=0))**3, ls='-', alpha=0.8, lw=1)
    ax[1].set(yscale='log')
    ax[1].legend(title='Glam/Molino', frameon=False)
    #lgnd = ax1.legend(title='Glam/Molino', frameon=False, fontsize=12)
    #for i, tx in enumerate(lgnd.get_texts()):tx.set_color('C%d'%i)
    #ax[1].text(10, 3000., 'ratio of cov', color='C0')
    #ax[1].text(10, 50., 'ratio of amplitudes', color='C1')
    ax[1].text(10, 1000., 'ratio of amplitudes^3', color='C1')

    ax[3].plot(cov_ba.diagonal()/cov_bm.diagonal(), lw=3, alpha=0.8)
    ax[3].plot((bk_a['pk'][:, :, 0].mean(axis=0)/bk_m['pk'][:, :, 0].mean(axis=0))**3/vratio**4.0, ls='-', alpha=0.8, lw=1)
    ax[3].set(yscale='log', xlabel='k-bin index')
    ax[3].legend(title='Abacus/Molino', frameon=False)     
    
    for i, axi in enumerate(ax):
        if i%2==0:
            axi.set_ylabel('Power Spectrum Ratio')
        else:
            axi.set_ylabel('Bispectrum Ratio')
    fig.savefig(path4figs_) 
    


def plot_glambk():
    from mpl_toolkits.mplot3d import axes3d
    path4figs_ = os.path.join(path4figs, 'glam_bk.pdf')

    bk_g = read_glam_bk()
    bk_gn = read_glam_bk_nobao()
    r = bk_g['pk'].mean(axis=0)/bk_gn['pk'].mean(axis=0)    
    
    fig = plt.figure()#figsize=(10,6))
    ax = axes3d.Axes3D(fig, auto_add_to_figure=False, elev=45, azim=-45)
    fig.add_axes(ax)
    cax = fig.add_axes([1., 0.2, 0.015, 0.5])
    #fig = plt.figure(figsize=(15, 9))
    #ax = fig.add_subplot(projection='3d')

    cb = ax.scatter(*bk_g['k'].T, c=r, s=10, marker='.', cmap='bwr' , vmin=0.95, vmax=1.05)
    ax.set_xlabel(r'k$_{1}$ [h/Mpc]', labelpad=10)
    ax.set_ylabel(r'k$_{2}$ [h/Mpc]', labelpad=10)
    ax.set_zlabel(r'k$_{3}$ [h/Mpc]', labelpad=10)
    cbar = fig.colorbar(cb, shrink=0.5, cax=cax, extend='both')
    cbar.set_label(r'B(k$_{1}$, k$_{2}$, k$_{3}$)')
    cbar.set_ticks([0.95, 1.0, 1.05])
    fig.savefig(path4figs_)    
