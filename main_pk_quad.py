
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
from helpers import PkModel, BkModel, logpriorpk, get_cov


os.environ["OMP_NUM_THREADS"] = "1"

def run(kmax, mcmc_file, progress=True):
    ndim = 7 # 7
    nwalkers = ndim*2
    nsteps = 10000
    kmin = 0.015
    print(f'entered kmax: {kmax}')
    print(f'output: {mcmc_file}')

    pk_mol = read_molino_pk()
    pk_glm = read_glam_pk()
    pk_glm_nobao = read_glam_pk_nobao()

    kp = pk_glm['k']
    p0_mol = pk_mol['pk'][:, :, 0]
    p0_glm = pk_glm['pk']
    p0_glm_nobao = pk_glm_nobao['pk']

    rp = p0_glm/p0_glm_nobao.mean(axis=0)
    rp_m = rp.mean(axis=0)

    pk_int = PkModel(kp, rp_m)

    good_p = (kp > kmin) & (kp < kmax)

    cov = get_cov(p0_mol[:, good_p], rp[:, good_p], False)
    icov = np.linalg.inv(cov)

    def loglike(theta):
        theta2 = theta.tolist()
        rc_ = pk_int(kp[good_p], theta2)
        res_ = rp_m[good_p] - rc_
        is_ok = ~np.isnan(res_)
        res = res_[is_ok]  
        return -0.5*res.dot(icov[is_ok, :][:, is_ok].dot(res))

    def logpost(theta):
        return logpriorpk(theta) + loglike(theta)

    def nlogpost(theta):
        return -1*logpost(theta)
    
    guess = np.array([1.001, 1.003, 1.0e-5, 0.9e-5, 1.0e-6,
                      0.8e-5, 0.9e-5])
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
    label = 'quad'
    ver   = 'v1.1'
    mcmc_file = f'mcmcs/mcmc_is_pk_kmax_{kmax_i:.2f}_{label}_{ver}.npz'
    
    run(kmax_i, mcmc_file, progress=rank==0)
    
        

 
