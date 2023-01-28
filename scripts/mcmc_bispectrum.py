"""
    Run MCMC for power spectrum like data
    
"""
import os
import numpy as np
import emcee
from scipy.optimize import minimize
from src.models import PowerSpectrum, BiSpectrum

def run_emcee(comm):
    rank = comm.Get_rank()

    if rank==0:
        from src.io import Spectrum       
        
        y = Spectrum(f'../cache/{ns.stat}_mean_{ns.mock}_bao.npz')
        ys = Spectrum(f'../cache/{ns.stat}_mean_{ns.temp}_nobao.npz')
        cov = Spectrum(f'../cache/{ns.stat}_cov_{ns.mock}_bao.npz')
        
        kw = ns.__dict__
        for (key, value) in kw.items():print(f'{key:15s} : {value}')              

        output_dir = os.path.dirname(kw["output_path"])
        if not os.path.exists(output_dir):
            print(f"will create {output_dir}")        
            os.makedirs(output_dir)
    else:
        y = None
        ys = None
        cov = None        
        kw = None
    
    y = comm.bcast(y, root=0)
    ys = comm.bcast(ys, root=0)    
    cov = comm.bcast(cov, root=0)        
    kw = comm.bcast(kw, root=0)            

    
    comm.Barrier()
    
    dk = (kw['kmax'] - kw['kmin']) / comm.Get_size()
    kmax_i = kw['kmax'] - rank*dk
    is_g = (y.x > kw['kmin']) & (y.x < kmax_i) 
    if kw['stat']=='bk':
        is_g = is_g.sum(axis=1) == 3
        
    x_g = y.x[is_g]
    r_g = (y.y/ys.y)[is_g]
    c_g = cov.y[is_g,:,][:, is_g]
    ic_g = np.linalg.inv(c_g)
    
    if kw['stat']=='bk':
        r_int = BiSpectrum(y.x, y.y/ys.y)
    else:
        r_int = PowerSpectrum(y.x, y.y/ys.y)
    
    print(f"rank-{rank} {is_g.mean()*100:.1f}% {kw['kmin']:.2f}<k<{kmax_i:.2f}")

    def loglike(p):
        res = r_g - r_int(x_g, p)
        loss =  -0.5*res.dot(ic_g.dot(res))
        if np.isnan(loss):
            return -np.inf
        else:
            return loss

    def logprior(p):
        lp = 0.
        lp += 0. if  0.9 < p[0] < 1.1 else -np.inf
        lp += 0. if  0.9 < p[1] < 1.1 else -np.inf    
        for p_i in p[2:]:
            lp += 0. if  -1 < p_i < 1. else -np.inf
        return lp

    def logpost(p):
        return loglike(p) + logprior(p)

    def nlogpost(p):
        return -1.*logpost(p)

    np.random.seed(85)
    if kw['stat']=='bk':
        guess = np.array([1.001, 1.001, 1.1e-5, 1.2e-5, 1.3e-5, 1.1e-5, 1.2e-5, 1.3e-5, 1.0e-5, 0.9e-5, 1.1e-5])
    else:
        guess = np.array([1.001, 1.001, 1.1e-5, 1.2e-5, 1.3e-5, 1.1e-5, 1.2e-5])
    #res = minimize(nlogpost, guess, method="Nelder-Mead")  ## takes too much time for Bispectrum
    
    start = (guess + guess*0.01*np.random.randn(kw['nwalkers'], kw['ndim']))
    sampler = emcee.EnsembleSampler(kw['nwalkers'], kw['ndim'], logpost)
    sampler.run_mcmc(start, kw['nsteps'], progress=rank==0)
    
    filename = f"{kw['stat']}_{kw['mock']}_{kw['temp']}_{kw['kmin']:.2f}_{kmax_i:.2f}.npz"
    output = os.path.join(kw['output_path'], filename)
    np.savez(output, **{'chain':sampler.get_chain(), 
                        'log_prob':sampler.get_log_prob(),
                        'klim':(kw['kmin'], kmax_i)})
    
    
    
if __name__ == '__main__':
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank()
    
    if rank == 0:
        from argparse import ArgumentParser
        ap = ArgumentParser(description='MCMC for BAO fit')
        ap.add_argument('--stat', default='bk')
        ap.add_argument('--mock', default='glam')
        ap.add_argument('--temp', default='glam')   
        ap.add_argument('--kmin', type=float, default=0.05)
        ap.add_argument('--kmax', type=float, default=0.25)        
        ap.add_argument('--nwalkers', type=int, default=50)
        ap.add_argument('--ndim', type=int, default=11)
        ap.add_argument('--nsteps', type=int, default=10000)
        ap.add_argument('--output_path', required=True)
        ns = ap.parse_args()        
    else:
        ns = None
        
    run_emcee(comm)
