"""
    Run MCMC for power spectrum like data
    
"""
import sys
import os
import numpy as np
import emcee
from multiprocessing.pool import ThreadPool as Pool

from scipy.optimize import minimize
from src.models import JointSpectrum
from src.io import Spectrum       
    

os.environ["OMP_NUM_THREADS"] = "1"


def run_emcee(comm, rank, ns):
    stat = 'pb'
    
    if rank==0:
        for k,v in ns.__dict__.items():print(f'{k:15s} : {v}')

        cov = Spectrum(f'../cache/pb_cov_{ns.mock}_bao.npz', allow_pickle=True)
        cov = cov.y

        y = Spectrum(f'../cache/pk_mean_{ns.mock}_bao.npz')
        ys = Spectrum(f'../cache/pk_mean_{ns.mock}_nobao.npz')
        x1 = y.x
        r1 = y.y/ys.y
        
        y_ = Spectrum(f'../cache/bk_mean_{ns.mock}_bao.npz')
        ys_ = Spectrum(f'../cache/bk_mean_{ns.mock}_nobao.npz')
        x2 = y_.x
        r2 = y_.y/ys_.y
        
        r = (r1, r2)
        x = (x1, x2)
        
        output_dir = os.path.dirname(ns.output_path)
        if not os.path.exists(output_dir):
            print(f"will create {output_dir}")        
            os.makedirs(output_dir)
    else:
        x=None
        r=None
        cov=None

    ns = comm.bcast(ns, root=0)
    x = comm.bcast(x, root=0)
    r = comm.bcast(r, root=0)
    cov = comm.bcast(cov, root=0)


    # cut k
    kmin = 0.03
    kmax_range = np.linspace(ns.kmin, ns.kmax, num=comm.Get_size())[::-1]
    kmax = kmax_range[rank]   
    is_g0 = (x[0] > kmin) & (x[0] < kmax) 
    is_g1 = (x[1] > kmin) & (x[1] < kmax) 
    is_g1 = is_g1.sum(axis=1) == 3
    
    x_g0 = x[0][is_g0]
    x_g1 = x[1][is_g1]        
    r_g0 = r[0][is_g0]
    r_g1 = r[1][is_g1]
    
    is_g = np.concatenate([is_g0, is_g1])
    r_g = np.concatenate([r_g0, r_g1])
    c_g = cov[is_g,:,][:, is_g]
    ic_g = np.linalg.inv(c_g)
    
    print(f"rank: {rank}, kmin: {kmin:.3f}, kmax: {kmax:.3f}")
    print(f"{is_g.mean()*100:.1f}% {kmin:.3f}<k<{kmax:.3f}")
    
    # initialize model
    r_int = JointSpectrum(x, r)
    
    def loglike(p):
        res = r_g - r_int((x_g0,x_g1), p)
        is_ok = ~np.isnan(res)    
        if is_ok.sum()==0:
            return -np.inf
        else:            
            return  -0.5*res[is_ok].dot(ic_g[is_ok, :][:, is_ok].dot(res[is_ok]))

    def logprior(p):
        lp = 0.
        lp += 0. if  0.8 < p[0] < 1.2 else -np.inf
        lp += 0. if  0.5 < p[1] < 2.0 else -np.inf    
        for p_i in p[2:7]:
            lp += 0. if  -1000. < p_i < 1000. else -np.inf
        lp += 0. if  0.5 < p[7] < 2.0 else -np.inf    
        for p_i in p[8:]:
            lp += 0. if  -1000. < p_i < 1000. else -np.inf            
        return lp

    def logpost(p):
        return loglike(p) + logprior(p)

    def nlogpost(p):
        return -1.*logpost(p)


    guess = np.array([1.01, 1.01, 0.001, 0.001, 0.001, 0.001, 0.001,
                      1.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    ndim = len(guess)

    np.random.seed(85)      
    #res = minimize(nlogpost, guess, method="Nelder-Mead")  ## takes too much time for Bispectrum
    start = (guess + guess*0.02*np.random.randn(ns.nwalkers, ndim))
    #for s in start:
    #    print(s)
    #    print(logpost(s), loglike(s))

    n = 28
    with Pool(n) as pool:        
        sampler = emcee.EnsembleSampler(ns.nwalkers, ndim, logpost, pool=pool)
        sampler.run_mcmc(start, ns.nsteps, progress=rank==0)
    
    filename = f"{stat}_{ns.mock}_{ns.temp}_{kmin:.3f}_{kmax:.3f}.npz"
    output = os.path.join(ns.output_path, filename)
    np.savez(output, **{'chain':sampler.get_chain(), 
                        'log_prob':sampler.get_log_prob(),
                        'klim':(kmin, kmax)})
    
    
    
if __name__ == '__main__':
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank()
    size = comm.Get_size()
   
    if rank==0:
        from argparse import ArgumentParser
        ap = ArgumentParser(description='MCMC for BAO fit')
        ap.add_argument('--mock', default='glam')
        ap.add_argument('--temp', default='glam')   
        ap.add_argument('--kmin', type=float, default=0.15)
        ap.add_argument('--kmax', type=float, default=0.25)        
        ap.add_argument('--nwalkers', type=int, default=40)
        ap.add_argument('--nsteps', type=int, default=10000)
        ap.add_argument('--output_path', required=True)
        ap.add_argument('-v', '--verbose', action='store_true', default=False)
        ns = ap.parse_args()        
    else:
        ns = None

    run_emcee(comm, rank, ns)
