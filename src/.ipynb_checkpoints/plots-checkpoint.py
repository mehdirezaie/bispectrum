
import matplotlib.pyplot as plt
from src.io import path2figs
    
def plot_spectra(xy_b, xy_p, names):
    
#     # --- plot 1
    fig, ax = plt.subplots(ncols=2, figsize=(10, 3), sharey=True)
    fig.subplots_adjust(hspace=0.0)
    ax = ax.flatten()
    for ai in ax:
        ai.tick_params(direction='in', right=True, top=True, which='both', axis='both')
        ai.grid(True, ls=':', alpha=0.3)
    colors = ['#000000', '#db2121']
    
    for i in range(len(xy_b)):
        ax[0].plot(*xy_b[i], label=names[i], alpha=0.8, color=colors[i])
        ax[1].plot(*xy_p[i], label=names[i], alpha=0.8, color=colors[i])
    ax[0].set(ylabel='Bispectrum ratio',     ylim=(0.8, 1.2), xlabel='Triangle index') 
    ax[1].set(ylabel='Power spectrum ratio', ylim=(0.8, 1.2), xlabel='Wavenumber [h/Mpc]')
    lgn = ax[1].legend(frameon=False, fontsize=13)
    for i, txt in enumerate(lgn.get_texts()):
        txt.set_color(colors[i])
    fig.align_labels()
    fig.savefig(f'{path2figs}spectra.pdf', bbox_inches='tight')
    
    
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
    
    
def plot_chi2(chi2s1, chi2s2, title='Mock'):
    fg, ax = plt.subplots()
    nk = len(chi2s2[0])
    for i, ch in enumerate(chi2s1[1]):
        ax.plot(*ch, color=plt.cm.Reds(i/nk), label=' ', zorder=-10)    
    for i, ch in enumerate(chi2s2[1]):
        ax.plot(*ch, color=plt.cm.Blues(i/nk), label=r'%.2f'%chi2s2[0][i])

    ax.legend(bbox_to_anchor=(1.01, 0.0, 0.3, 1), mode='expand', frameon=False, ncol=2, title=r'$k_{\rm max}$',
             columnspacing=0.0)
    ax.text(0.1, 0.8, 'Bispectrum', transform=ax.transAxes, color=plt.cm.Reds(1.0))
    ax.text(0.05, 0.05, 'Power spectrum', transform=ax.transAxes, color=plt.cm.Blues(1.0))
    ax.set(xlabel=r'$\alpha$', ylabel=r'$\chi^{2}_{\rm min}$', ylim=(-0.2, 14), xlim=(0.94, 1.06),
          title=title)   
    fg.savefig(f'../bisp4desi/figures/chi2_{title}.pdf', bbox_inches='tight')    