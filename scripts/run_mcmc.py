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
use_diag = int(sys.argv[5]) > 0.5

nmocks = 25
kmax_range = [0.17, 0.19, 0.21, 0.23, 0.25] # 0.25 is already done
kmin = 0.05

if stat=='bk':
    PS = src.utils.BisPosterior if reduced=='raw' else src.utils.RedBisPosterior
elif stat=='pk':
    PS = src.utils.PowPosterior
elif stat=='bk9':
    PS = src.utils.BisPosterior9

(k_obs, r_obs, r_cov), (k_tem, r_tem) = src.utils.load_data(tracer, stat, reduced, temp, use_diag)
print(k_obs.shape, r_obs.shape, r_cov.shape, k_tem.shape, r_tem.shape)

for i in range(nmocks):
    
    ps = PS()
    ps.add_data(k_obs, r_obs[i, :], r_cov)
    ps.add_template(k_tem, r_tem) 

    for kmax in kmax_range:
        ps.select_krange(kmin=kmin, kmax=kmax)
        ps.run_mcmc(nsteps=1000)
        
    print(f'saving ... mcmc_{stat}_{reduced}_{tracer}_mock{i}_{temp}_p1_p5_1k_{use_diag}.npz')
    ps.save(f'/localdata/desi/mcmc_{stat}_{reduced}_{tracer}_mock{i}_{temp}_p1_p5_1k_{use_diag}.npz')
