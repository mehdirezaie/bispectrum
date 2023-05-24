import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import src


    
def setup_color():
    from cycler import cycler
    #colors = ['#000000', '#e41a1c', '#377eb8', '#ff7f00', '#4daf4a',
    #          '#f781bf', '#a65628', '#984ea3',
    #          '#999999']
    #styles = ['-', '--', '-.', ':', '--', '-.', ':', '-', '--']
    #plt.rc('axes', prop_cycle=(cycler('color', colors)+cycler('linestyle', styles)))
    
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.major.width'] = 2.0
    mpl.rcParams['xtick.major.pad'] = 5.0
    mpl.rcParams['xtick.major.size'] = 9.0
    mpl.rcParams['xtick.minor.width'] = 2.0
    mpl.rcParams['xtick.minor.pad'] = 5.0
    mpl.rcParams['xtick.minor.size'] = 6.0
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['ytick.major.width'] = 2.0
    mpl.rcParams['ytick.major.pad'] = 5.0
    mpl.rcParams['ytick.major.size'] = 9.0
    mpl.rcParams['ytick.minor.width'] = 2.0
    mpl.rcParams['ytick.minor.pad'] = 5.0
    mpl.rcParams['ytick.minor.size'] = 6.0
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['font.family'] = "Times"
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.linewidth'] = 2       
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.facecolor'] = 'w'  
        

def plot_data():
    
    lrg = src.utils.get_bispectra('LRGz0')
    elg = src.utils.get_bispectra('ELGz1')
    qso = src.utils.get_bispectra('QSOz2')

    lrg_p = src.utils.get_powerspectra('LRGz0')
    elg_p = src.utils.get_powerspectra('ELGz1')
    qso_p = src.utils.get_powerspectra('QSOz2')


    fig, ax = plt.subplots(nrows=3, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    for i, (tracer, name) in enumerate(zip([lrg, elg, qso], ['LRG', 'ELG', 'QSO'])):
        k3 = tracer.k[:, 0]*tracer.k[:, 1]*tracer.k[:, 2]

        ax[i].errorbar(np.arange(len(k3)), k3*tracer.b.mean(axis=0), yerr=k3*np.std(tracer.b, axis=0), zorder=-10)
        ax[i].plot(k3*tracer.b_bestfit.mean(axis=0), lw=1, alpha=0.5)
        ax[i].annotate(name, (0.05, 0.85), xycoords='axes fraction', color='C0')
        ax[i].set(ylabel=r'$k_{1}k_{2}k_{3}\bf{B}$')    

    ax[2].annotate('Best Fit Bispectrum', 
                   (0.65, 0.1), xycoords='axes fraction', color='C1')
    ax[2].set_xlabel('Triangle Index')
    fig.align_ylabels()
    fig.savefig('../bisp4desi/figures/spectra.pdf', bbox_inches='tight')    
    
    
    fig, ax = plt.subplots(nrows=3, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    for i, (tracer, name) in enumerate(zip([lrg, elg, qso], ['LRG', 'ELG', 'QSO'])):
        ax[i].plot(tracer.b.mean(axis=0)/tracer.b_smooth.mean(axis=0), lw=1, label=name, color='k')
        ax[i].annotate(name, (0.85, 0.85), xycoords='axes fraction', color='k')
        ax[i].axhline(1.0, ls='--', color='r')
        ax[i].set(ylabel=r'$\bf{B}/\bf{B}_{\rm smooth}$', ylim=(0.78, 1.22))
    ax[2].set_xlabel('Triangle Index')
    fig.align_ylabels()
    fig.savefig('../bisp4desi/figures/spectra_ratio.pdf', bbox_inches='tight')    
    
    
    
    def add_plot(ax, x, y, ys, **kw):
        ln = ax.plot(x, y.mean(axis=0)/ys.mean(axis=0), **kw)
        ax.fill_between(x, y.mean(axis=0)/ys.mean(axis=0)-(y.std(axis=0)/ys.mean(axis=0)),
                           y.mean(axis=0)/ys.mean(axis=0)+(y.std(axis=0)/ys.mean(axis=0)),
                       color=ln[0].get_color(), alpha=0.2)

    kw = dict(ls='None', capsize=3, marker='.')
    fig, ax = plt.subplots(nrows=3, figsize=(6, 12), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.05)
    for i, (tracer, name) in enumerate(zip([lrg, elg, qso], ['LRG', 'ELG', 'QSO'])):
        # ax[i].plot(tracer.k_all, tracer.b_all.mean(axis=0)/tracer.bs_all.mean(axis=0), label='All')
        # ax[i].plot(tracer.k_iso, tracer.b_iso.mean(axis=0)/tracer.bs_iso.mean(axis=0), label='Isoceles')    
        # ax[i].plot(tracer.k_eqi, tracer.b_eqi.mean(axis=0)/tracer.bs_eqi.mean(axis=0), label='Equilateral')     
        # ax[i].errorbar(tracer.k_all, tracer.b_all.mean(axis=0)/tracer.bs_all.mean(axis=0), yerr=tracer.b_all.std(axis=0)/tracer.bs_all.mean(axis=0), label='All', **kw)
        # ax[i].errorbar(tracer.k_iso, tracer.b_iso.mean(axis=0)/tracer.bs_iso.mean(axis=0), yerr=tracer.b_iso.std(axis=0)/tracer.bs_iso.mean(axis=0), label='Isoceles', **kw)    
        # ax[i].errorbar(tracer.k_eqi, tracer.b_eqi.mean(axis=0)/tracer.bs_eqi.mean(axis=0), yerr=tracer.b_eqi.std(axis=0)/tracer.bs_eqi.mean(axis=0), label='Equilateral', **kw) 
        add_plot(ax[i], tracer.k_all, tracer.b_all, tracer.bs_all, label='All')
        add_plot(ax[i], tracer.k_iso, tracer.b_iso, tracer.bs_iso, label='Isoceles')
        add_plot(ax[i], tracer.k_eqi, tracer.b_eqi, tracer.bs_eqi, label='Equilateral')
        ax[i].annotate(name, (0.85, 0.85), xycoords='axes fraction', color='k')
        ax[i].axhline(1.0, ls='--', color='r')
    ax[1].set(ylabel=r'$<{\bfB}>_{k_{2},k_{3}}~/~<{\bf B}_{\rm smooth}>_{k_{2},k_{3}}$', ylim=(0.78, 1.22))
    ax[2].set_xlabel(r'$k_{1} [h/{\rm Mpc}]$')
    legend = ax[0].legend(fontsize=14, loc='lower right')#, bbox_to_anchor=(0., 1.1, 1., 0.2))
    for i,txt in enumerate(legend.get_texts()):
        txt.set_color('C%d'%i)
    fig.align_ylabels()
    fig.savefig('../bisp4desi/figures/spectra_ratio_reduced.pdf', bbox_inches='tight')    
    

    kw = dict(ls='None', capsize=3, marker='.')
    fig, ax = plt.subplots(nrows=3, figsize=(6, 12), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.05)
    for i, (tracer, name) in enumerate(zip([lrg_p, elg_p, qso_p], ['LRG', 'ELG', 'QSO'])):
        add_plot(ax[i], tracer.k, tracer.p, tracer.p_smooth, label='All')    
        ax[i].annotate(name, (0.85, 0.85), xycoords='axes fraction', color='k')
        ax[i].axhline(1.0, ls='--', color='r')
    ax[1].set(ylabel=r'${\bfP}~/~{\bf P}_{\rm smooth}$', ylim=(0.78, 1.22))
    ax[2].set_xlabel(r'$k [h/{\rm Mpc}]$')
    legend = ax[0].legend(fontsize=14, loc='lower right')#, bbox_to_anchor=(0., 1.1, 1., 0.2))
    for i,txt in enumerate(legend.get_texts()):
        txt.set_color('C%d'%i)
    fig.align_ylabels()
    fig.savefig('../bisp4desi/figures/powerspectra_ratio.pdf', bbox_inches='tight')    
    
    
    tracer = ['lrg', 'elg', 'qso']
    for ii, (a, b) in enumerate([(lrg, lrg_p), (elg, elg_p), (qso, qso_p)]):
        corr_all = src.utils.correlate(a.b_all/a.bs_all.mean(axis=0), b.p/b.p_smooth.mean(axis=0))
        corr_iso = src.utils.correlate(a.b_iso/a.bs_iso.mean(axis=0), b.p/b.p_smooth.mean(axis=0))
        corr_eqi = src.utils.correlate(a.b_eqi/a.bs_eqi.mean(axis=0), b.p/b.p_smooth.mean(axis=0))
        # because some bins are available
        corr_iso_f = np.zeros((30, 30))*np.nan
        corr_iso_f[:22, :] = corr_iso
        
        fig, ax = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)
        fig.subplots_adjust(wspace=0.0)
        names = ['P x B$_{all}$', 'PxB$_{equi}$', 'PxB$_{iso}$']
        for i, ci in enumerate([corr_all, corr_eqi, corr_iso_f]):
            #print(np.percentile(ci, [0, 100]))
            map_ = ax[i].imshow(ci, origin='lower', vmin=-1, vmax=1., cmap='seismic', extent=[0, 0.3, 0, 0.3], aspect='auto')
            ax[i].set_title(names[i])
            ax[i].set_xlabel('$k$')   
            ax[i].set_xticks(np.arange(0.05, 0.29, 0.05))

        ax[0].set_ylabel('$k_{1}$')
        ax[2].annotate("Not Allowed Isoceles", (0.2, 0.85), xycoords ='axes fraction')
        cax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
        fig.colorbar(map_, cax=cax, label=r'$\rho$')
        fig.savefig(f'../bisp4desi/figures/corrmax_{tracer[ii]}.pdf', bbox_inches='tight')    
    
    
    if not os.path.exists('bk_molino.z0.0.fiducial.rcov.npz'):
        bkfiles = glob('/Users/mehdi/data/fiducial_bk_pk/bk_molino.z0.0.fiducial.*')
        src.utils.prep_rcov(bkfiles, 'bk_molino.z0.0.fiducial.rcov.npz', 3)

    if not os.path.exists('pk_molino.z0.0.fiducial.rcov.npz'):        
        pkfiles = glob('/Users/mehdi/data/fiducial_bk_pk/pk_molino.z0.0.fiducial.*')
        src.utils.prep_rcov(pkfiles, 'pk_molino.z0.0.fiducial.rcov.npz', 1)    
        
    rcov_b = np.load('bk_molino.z0.0.fiducial.rcov.npz', allow_pickle=True)
    rcov_p = np.load('pk_molino.z0.0.fiducial.rcov.npz', allow_pickle=True)
    def fixiso(c):
        c_ = np.zeros((30, 30))*np.nan
        c_[:22, :22] = c
        return c_

    fg, ax = plt.subplots(ncols=4, figsize=(20, 5), sharey=True)
    fg.subplots_adjust(wspace=0.0)

    kw = dict(origin='lower', cmap='rainbow', vmin=0, vmax=1, aspect='auto', extent=[0., 0.3, 0., 0.3])
    titles = ['P', 'B$_{all}$', 'B$_{eqi}$', 'B$_{iso}$']
    for i, r_i in enumerate([rcov_p['raw'], rcov_b['all'],
                             rcov_b['eqi'], fixiso(rcov_b['iso'])]):
        print(np.percentile(r_i, [0, 100]))
        map_ = ax[i].imshow(r_i, **kw)
        ax[i].set(title=titles[i], xlabel='$k$ or $k_{1}$')
        ax[i].set_xticks(np.arange(0., 0.29, 0.1))
    ax[0].set_ylabel('$k$ or $k_{1}$')
    ax[3].annotate("Not Allowed Isoceles", (0.2, 0.85), xycoords ='axes fraction')
    cax = fg.add_axes([0.91, 0.2, 0.01, 0.6])
    fg.colorbar(map_, cax=cax, label=r'MOLINO $\rho$')
    fg.savefig(f'../bisp4desi/figures/corrmax_molino.pdf', bbox_inches='tight')    
    
    
    
    
    
    
    
# def plot_spectra(xy_b, xy_p, names):
    
# #     # --- plot 1
#     fig, ax = plt.subplots(ncols=2, figsize=(10, 3), sharey=True)
#     fig.subplots_adjust(hspace=0.0)
#     ax = ax.flatten()
#     for ai in ax:
#         ai.tick_params(direction='in', right=True, top=True, which='both', axis='both')
#         ai.grid(True, ls=':', alpha=0.3)
#     colors = ['#000000', '#db2121']
    
#     for i in range(len(xy_b)):
#         ax[0].scatter(*xy_b[i], label=names[i], alpha=0.6, color=colors[i], marker='.')
#         ax[1].scatter(*xy_p[i], label=names[i], alpha=0.6, color=colors[i], marker='.')
#     ax[0].set(ylabel='Bispectrum ratio',     ylim=(0.8, 1.2), xlabel='Triangle index') 
#     ax[1].set(ylabel='Power spectrum ratio', ylim=(0.8, 1.2), xlabel='Wavenumber [h/Mpc]')
#     lgn = ax[1].legend(frameon=False, fontsize=13)
#     for i, txt in enumerate(lgn.get_texts()):
#         txt.set_color(colors[i])
#     fig.align_labels()
#     fig.savefig(f'{path2figs}spectra.pdf', bbox_inches='tight')
    
    
# def plot_rcov(rcov_p, rcov_b):
#     fig, ax  = plt.subplots(ncols=2, figsize=(10, 5))
#     fig.subplots_adjust(wspace=0.3)
#     cax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
#     kw = dict(origin='lower', cmap='rainbow_r', vmin=-1, vmax=1)

#     map_p = ax[0].imshow(rcov_p, extent=[0.0, 0.3, 0.0, 0.3],   **kw)
#     map_b = ax[1].imshow(rcov_b, extent=[0.0, 2600, 0.0, 2600], **kw)

#     fig.colorbar(map_p, cax=cax, shrink=0.8, label='correlation matrix', anchor=(0., 0.5))
#     ax[0].set_xlabel('k [h/Mpc]')
#     ax[0].set_ylabel('k [h/Mpc]')
#     ax[1].set_xlabel('Triangle index')
#     ax[1].set_ylabel('Triangle index')   
    
    
# def plot_chi2(chi2s1, chi2s2, title='Mock'):
#     fg, ax = plt.subplots()
#     nk = len(chi2s2[0])
#     for i, ch in enumerate(chi2s1[1]):
#         ax.plot(*ch, color=plt.cm.Reds(i/nk), label=' ', zorder=-10)    
#     for i, ch in enumerate(chi2s2[1]):
#         ax.plot(*ch, color=plt.cm.Blues(i/nk), label=r'%.2f'%chi2s2[0][i])

#     ax.legend(bbox_to_anchor=(1.01, 0.0, 0.3, 1), mode='expand', frameon=False, ncol=2, title=r'$k_{\rm max}$',
#              columnspacing=0.0)
#     ax.text(0.1, 0.8, 'Bispectrum', transform=ax.transAxes, color=plt.cm.Reds(1.0))
#     ax.text(0.05, 0.05, 'Power spectrum', transform=ax.transAxes, color=plt.cm.Blues(1.0))
#     ax.set(xlabel=r'$\alpha$', ylabel=r'$\chi^{2}_{\rm min}$', ylim=(-0.2, 14), xlim=(0.94, 1.06),
#           title=title)   
#     fg.savefig(f'../bisp4desi/figures/chi2_{title}.pdf', bbox_inches='tight')    