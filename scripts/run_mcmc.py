import src
import numpy as np
import matplotlib.pyplot as plt

tracer = 'LRGz0'
stat = 'bk'
reduced = 'raw'
nmocks = 25
kmax_range = [0.14, 0.16, 0.18, 0.20]
kmin = 0.1

(k_obs, r_obs, r_cov), (k_tem, r_tem) = src.utils.load_data(tracer, stat, reduced, 'none')


for i in range(nmocks):
    
    ps = src.utils.BisPosterior()
    ps.add_data(k_obs, r_obs[i, :], r_cov)
    ps.add_template(k_tem, r_tem) 

    for kmax in kmax_range:
        ps.select_krange(kmin=kmin, kmax=kmax)
        ps.run_mcmc()
        
    ps.save(f'mcmc_{stat}_{reduced}_{tracer}_mock{i}_p1_p2.npz')