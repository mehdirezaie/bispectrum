
import os
import sys
import numpy as np
import emcee
from scipy.optimize import minimize
from multiprocessing.pool import ThreadPool as Pool

from helpers import (read_molino_bk, read_molino_pk,
                     read_glam_pk, read_glam_bk,
                     read_glam_bk_nobao, read_glam_pk_nobao, 
                     savez)
from helpers import PkModel, BkModelConst, logpriorbkconst, get_cov


os.environ["OMP_NUM_THREADS"] = "1"

def run(kmax, mcmc_file, progress=True):
    ndim = 3 #
    nwalkers = ndim*2
    nsteps = 10000
    kmin = 0.015
    print(f'entered kmax: {kmax}')
    print(f'output: {mcmc_file}')

    bk_mol = read_molino_bk()
    bk_glm = read_glam_bk()
    bk_glm_nobao = read_glam_bk_nobao()

    kb = bk_glm['k']
    y_c = bk_mol['pk'][:, :, 0]
    b0_glm = bk_glm['pk']
    b0_glm_nobao = bk_glm_nobao['pk']
    r_c = b0_glm/b0_glm_nobao.mean(axis=0)
    r_cm = r_c.mean(axis=0)

    bk_int = BkModelConst(kb, r_cm)

    good_b_ = (kb > kmin) & (kb < kmax)
    good_b = good_b_.sum(axis=1) == 3
    cov = get_cov(y_c[:, good_b], r_c[:, good_b], False)
    icov = np.linalg.inv(cov)

    def loglike(theta):
        theta1 = theta.tolist()
        rb_ = bk_int(kb[good_b], theta1)
        res_ = r_cm[good_b] - rb_
        is_ok = ~np.isnan(res_)
        res = res_[is_ok]  
        return -0.5*res.dot(icov[is_ok, :][:, is_ok].dot(res))

    def logpost(theta):
        return logpriorbkconst(theta) + loglike(theta)

    def nlogpost(theta):
        return -1*logpost(theta)
    
    guess = np.array([1.001, 1.001, 1.1e-5])
    # res = minimize(nlogpost, initial_guess) #, method='Powell')    
    start = (guess + guess*0.01*np.random.randn(nwalkers, ndim))
    n = 1

    if progress:
        print(f'Initial points: {start[:nwalkers//2, :]}')
        print(f'Using {n} processes for the pool!')    
    with Pool(n) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool)
        sampler.run_mcmc(start, nsteps, progress=progress)

    savez(mcmc_file, **{'chain':sampler.get_chain(), 
                        'log_prob':sampler.get_log_prob(), 
                        '#params':ndim})


if __name__ == '__main__':
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    kmax = np.arange(0.10, 0.31, 0.01)[::-1]  # give the largest kmax to rank=0
    nk = len(kmax)
    if size != nk:
        if rank==0:print('rerun with -np %d'%nk)
        sys.exit()

    kmax_i = kmax[rank]
    label = 'const'
    ver   = 'v1.1'
    mcmc_file = f'mcmcs/mcmc_is_bk_kmax_{kmax_i:.2f}_{label}_{ver}.npz'
    
    run(kmax_i, mcmc_file, progress=rank==0)
    
        

 
