import matplotlib.pyplot as plt
from src.io import path2figs
    
def plot_spectra(dl):
    from src.stats import get_p3
    
    k, pk_glam = dl.load('glam_pk_bao', 1)
    __, pk_glam_nobao = dl.load('glam_pk_nobao', 1)
    __, pk_molino = dl.load('molino_pk', 1)
    __, pk_abacus = dl.load('abacus_pk', 1)
    __, pk_abacus_nobao = dl.load('abacus_pk_nobao', 2)

    k3, bk_glam = dl.load('glam_bk_bao', 3)
    __, bk_glam_nobao = dl.load('glam_bk_nobao', 3)
    __, bk_molino = dl.load('molino_bk', 3)
    __, bk_abacus = dl.load('abacus_bk', 3)
    __, bk_abacus_nobao = dl.load('abacus_bk_nobao', 4)

    p3_abacus = get_p3(k3, pk_abacus.mean(axis=0))
    p3_molino = get_p3(k3, pk_molino.mean(axis=0))
    p3_glam   = get_p3(k3, pk_glam.mean(axis=0))


    # --- plot 1
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 6), sharex='col')
    fig.subplots_adjust(hspace=0.0)
    ax = ax.flatten()
    for ai in ax:
        ai.tick_params(direction='in', right=True, top=True, which='both', axis='both')
        ai.grid(True, ls=':', alpha=0.3)


    ax[0].plot(bk_abacus.mean(axis=0)/bk_abacus_nobao.mean(axis=0), label='ABACUS')
    ax[0].plot(bk_glam.mean(axis=0)/bk_glam_nobao.mean(axis=0),     label='GLAM')
    ax[1].plot(k, pk_abacus.mean(axis=0)/pk_abacus_nobao.mean(axis=0), label='ABACUS')
    ax[1].plot(k, pk_glam.mean(axis=0)/pk_glam_nobao.mean(axis=0),     label='GLAM')
    ax[0].set(ylabel='Bispectrum ratio', ylim=(0.8, 1.2)) 
    ax[1].set(ylabel='Power spectrum ratio', ylim=(0.8, 1.2))

    vr = 8
    p = 1.5
    ax[2].plot(bk_abacus.var(axis=0)/p3_abacus*vr**p, label='ABACUS', alpha=0.8)
    ax[2].plot(bk_glam.var(axis=0)/p3_glam,           label='GLAM')
    ax[2].plot(bk_molino.var(axis=0)/p3_molino,       label='MOLINO')
    ax[3].plot(k, pk_abacus.var(axis=0)/pk_abacus.mean(axis=0)**2*vr**p, label='ABACUS')
    ax[3].plot(k, pk_glam.var(axis=0)/pk_glam.mean(axis=0)**2,           label='GLAM')
    ax[3].plot(k, pk_molino.var(axis=0)/pk_molino.mean(axis=0)**2,       label='MOLINO')
    ax[2].set(xlabel='Triangle index', yscale='log', 
              ylabel='Normalized bispectrum dispersion')#, ylabel=r'$\frac{\sigma^{2}_{B}}{P^{3}V^{1.5}}$')
    ax[3].set(xlabel='k [h/Mpc]',      yscale='log', 
              ylabel='Normalized power spectrum dispersion')#, ylabel=r'$\frac{\sigma^{2}_{P}}{P^{2}V^{1.5}}$')
    ax[3].legend(bbox_to_anchor=(-1, 2.1, 1.5, 0.15), 
                 mode='expand', ncol=3, frameon=False, fontsize=13)
    fig.align_labels()
    fig.savefig(f'{path2figs}spectra.pdf', bbox_inches='tight')

    # fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    # ax[0].set(ylabel=r'$\chi^{2}(\alpha)$', xlabel=r'$\alpha$', title='Scoccimaro template')
    # ax[1].set(xlabel=r'$\alpha$', title='GLAM template')
    # # different curves for different k-max values

    # fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    # ax[0].set(xlabel='Triangle index', title='GLAM', ylabel='Bispectrum ratio')
    # ax[1].set(xlabel=r'kmax', title='GLAM', ylabel=r'$\sigma(\alpha)$')      
    
def plot_rcov(rcov_p, rcov_b):
    fig, ax  = plt.subplots(ncols=2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3)
    cax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    kw = dict(origin='lower', cmap='rainbow_r', vmin=-1, vmax=1)

    map_p = ax[0].imshow(rcov_p, extent=[0.0, 0.3, 0.0, 0.3],   **kw)
    map_b = ax[1].imshow(rcov_b, extent=[0.0, 2600, 0.0, 2600], **kw)

    fig.colorbar(map_p, cax=cax, shrink=0.8, label='correlation matrix', anchor=(0., 0.5))
    ax[0].set_xlabel('k [h/Mpc]')
    ax[0].set_ylabel('k [h/Mpc]')
    ax[1].set_xlabel('Triangle index')
    ax[1].set_ylabel('Triangle index')    