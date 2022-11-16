
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
from helpers import PkModel, BkModel, logprior, get_cov


os.environ["OMP_NUM_THREADS"] = "1"

def run(kmax, mcmc_file, progress=True):
    ndim = 17 # 11 + 6
    nwalkers = ndim*2
    nsteps = 10000
    kmin = 0.015
    print(f'entered kmax: {kmax}')
    print(f'output: {mcmc_file}')

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
    r_c = np.concatenate([rb, rp], axis=1)
    rp_m = rp.mean(axis=0)
    rb_m = rb.mean(axis=0)
    r_cm = np.concatenate([rb_m, rp_m])

    y_c = np.concatenate([b0_mol, p0_mol], axis=1)

    pk_int = PkModel(kp, np.ones_like(rp_m))
    bk_int = BkModel(kb, np.ones_like(rb_m))

    good_p = (kp > kmin) & (kp < kmax)
    good_b_ = (kb > kmin) & (kb < kmax)
    good_b = good_b_.sum(axis=1) == 3
    is_good = np.concatenate([good_b, good_p])

    cov = get_cov(y_c[:, is_good], r_c[:, is_good], False)
    icov = np.linalg.inv(cov)

    def loglike(theta):
        theta = theta.tolist()
        theta1 = theta[:11]
        theta2 = theta[11:]
        theta2.insert(0, theta[0])
        rp_ = pk_int(kp[good_p], theta2)
        rb_ = bk_int(kb[good_b], theta1)
        rc_ = np.concatenate([rb_, rp_])    
        res_ = r_cm[is_good] - rc_
        is_ok = ~np.isnan(res_)
        res = res_[is_ok]  
        return -0.5*res.dot(icov[is_ok, :][:, is_ok].dot(res))

    def logpost(theta):
        return logprior(theta) + loglike(theta)

    def nlogpost(theta):
        return -1*logpost(theta)
    
    guess = np.array([1.001, 1.001, 1.1e-5, 1.2e-5, 1.3e-5, 
                      1.1e-5, 1.2e-5, 1.3e-5, 1.0e-5, 0.9e-5,
                      1.1e-5, 1.003, 1.0e-5, 0.9e-5, 1.0e-6,
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
    label = 'quadnobao'
    ver   = 'v1.1'
    mcmc_file = f'mcmcs/mcmc_is_joint_kmax_{kmax_i:.2f}_{label}_{ver}.npz'
    
    run(kmax_i, mcmc_file, progress=rank==0)
    
        

 
