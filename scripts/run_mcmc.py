import sys
import src
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

tracer = sys.argv[1] #'LRGz0'
stat = sys.argv[2] #'pk'
reduced = sys.argv[3] #'raw'
temp = sys.argv[4] #'lado'
nmocks = 25
kmax_range = [0.25,]
kmin = 0.05

if stat=='bk':
    PS = src.utils.BisPosterior if reduced=='raw' else src.utils.RedBisPosterior
else:
    PS = src.utils.PowPosterior

(k_obs, r_obs, r_cov), (k_tem, r_tem) = src.utils.load_data(tracer, stat, reduced, temp)
print(k_obs.shape, r_obs.shape, r_cov.shape, k_tem.shape, r_tem.shape)

for i in range(nmocks):
    
    ps = PS()
    ps.add_data(k_obs, r_obs[i, :], r_cov)
    ps.add_template(k_tem, r_tem) 

    for kmax in kmax_range:
        ps.select_krange(kmin=kmin, kmax=kmax)
        ps.run_mcmc(nsteps=1000)
        
    print(f'saving ... mcmc_{stat}_{reduced}_{tracer}_mock{i}_{temp}_p1_p2_1k.npz')
    ps.save(f'mcmc_{stat}_{reduced}_{tracer}_mock{i}_{temp}_p1_p2_1k.npz')
