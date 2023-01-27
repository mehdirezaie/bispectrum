import numpy as np

def get_cov(y):
    cov_ = np.cov(y, rowvar=0)
    std_ = np.diagonal(cov_)**0.5
    rcov = cov_ / np.outer(std_, std_)
    return rcov

def get_p3(k3, pk_a):
    
    p3 = []
    e = 1.0e-6
    for i, ki in enumerate(k3):

        ix = int((ki[0]-0.005+e)*100)
        iy = int((ki[1]-0.005+e)*100)
        iz = int((ki[2]-0.005+e)*100)
        #print(i, ki, ix, iy, iz)    
        p3.append(pk_a[ix]*pk_a[iy]*pk_a[iz])
    
    return np.array(p3)