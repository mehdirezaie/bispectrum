import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d

class PowerSpectrum(object):
    
    def __init__(self, x, y):
        self.y_intp = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    
    def __call__(self, x, p): # 7 params
        return p[1]*self.y_intp(p[0]*x) + p[2] + p[3]*x + p[4]*(1./x) + p[5]*(x*x) + p[6]*(1./(x*x))


class BiSpectrum(object):
    
    def __init__(self, x, y):
        self.y_intb = LinearNDInterpolator(x, y, fill_value=np.nan)
        print("interpolation done")
    
    def __call__(self, x, p): # 11 params
        return p[1]*self.y_intb(p[0]*x) + p[2] + p[3]*(x[:, 0] + x[:, 1] + x[:, 2]) \
                + p[4]*(1./x[:, 0] + 1./x[:, 1] + 1/x[:, 2]) \
                + p[5]*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) \
                + p[6]*(1./x[:, 0]**2 + 1./x[:, 1]**2 + 1./x[:, 2]**2) \
                + p[7]*(x[:, 0]*x[:, 1] + x[:, 0]*x[:, 2] + x[:, 1]*x[:, 2]) \
                + p[8]*(1./(x[:, 0]*x[:, 1]) + 1./(x[:, 0]*x[:, 2]) + 1./(x[:, 1]*x[:, 2])) \
                + p[9]*(x[:, 0]*x[:, 1]/x[:, 2] + x[:, 0]*x[:, 2]/x[:, 1] + x[:, 1]*x[:, 2]/x[:, 0]) \
                + p[10]*(x[:, 2]/(x[:, 0]*x[:, 1]) + x[:, 1]/(x[:, 0]*x[:, 2]) + x[:, 0]/(x[:, 1]*x[:, 2]))    
    

class BiSpectrum9(BiSpectrum):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __call__(self, x, p):
        return p[1]*self.y_intb(p[0]*x) + p[2] + p[3]*(x[:, 0] + x[:, 1] + x[:, 2]) \
                + p[4]*(1./x[:, 0] + 1./x[:, 1] + 1/x[:, 2]) \
                + p[5]*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) \
                + p[7]*(x[:, 0]*x[:, 1] + x[:, 0]*x[:, 2] + x[:, 1]*x[:, 2]) \
                + p[9]*(x[:, 0]*x[:, 1]/x[:, 2] + x[:, 0]*x[:, 2]/x[:, 1] + x[:, 1]*x[:, 2]/x[:, 0]) \
                + p[10]*(x[:, 2]/(x[:, 0]*x[:, 1]) + x[:, 1]/(x[:, 0]*x[:, 2]) + x[:, 0]/(x[:, 1]*x[:, 2]))    
 

class DecayedBiSpectrum(BiSpectrum):
    def __init__(self, x, y):
        super().__init__(x, y)
        
    def __call__(self, x, p, sigma=-7.0):
        r = super().__call__(x, p)
        return (r-1.0)*(np.exp(sigma*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)))+1.0
    
    
class JointSpectrum(PowerSpectrum, BiSpectrum):
    
    def __init__(self, x, y):
        PowerSpectrum.__init__(self, x[0], y[0])
        BiSpectrum.__init__(self, x[1], y[1])
    
    def __call__(self, x, p):
        p_ = p.tolist()
        p1 = p_[:7]
        p2 = p_[7:]
        p2.insert(0, p1[0]) 

        r1 = PowerSpectrum.__call__(self, x[0], p1)
        r2 = BiSpectrum.__call__(self, x[1], p2)

        return np.concatenate([r1, r2])
    
    
class PowerSpectrumNoBAO(object):
    
    def __init__(self, x, y):
        pass
        #self.y_intp = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    
    def __call__(self, x, p): # 7 params
        return p[2] + p[3]*x + p[4]*(1./x) + p[5]*(x*x) + p[6]*(1./(x*x))


class BiSpectrumNoBAO(object):
    
    def __init__(self, x, y):
        pass
        #self.y_intb = LinearNDInterpolator(x, y, fill_value=np.nan)
    
    def __call__(self, x, p): # 11 params
        return p[2] + p[3]*(x[:, 0] + x[:, 1] + x[:, 2]) \
                + p[4]*(1./x[:, 0] + 1./x[:, 1] + 1/x[:, 2]) \
                + p[5]*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) \
                + p[6]*(1./x[:, 0]**2 + 1./x[:, 1]**2 + 1./x[:, 2]**2) \
                + p[7]*(x[:, 0]*x[:, 1] + x[:, 0]*x[:, 2] + x[:, 1]*x[:, 2]) \
                + p[8]*(1./(x[:, 0]*x[:, 1]) + 1./(x[:, 0]*x[:, 2]) + 1./(x[:, 1]*x[:, 2])) \
                + p[9]*(x[:, 0]*x[:, 1]/x[:, 2] + x[:, 0]*x[:, 2]/x[:, 1] + x[:, 1]*x[:, 2]/x[:, 0]) \
                + p[10]*(x[:, 2]/(x[:, 0]*x[:, 1]) + x[:, 1]/(x[:, 0]*x[:, 2]) + x[:, 0]/(x[:, 1]*x[:, 2]))    
    
    
class JointSpectrumNoBAO(PowerSpectrum, BiSpectrum):
    
    def __init__(self, x, y):
        PowerSpectrumNoBAO.__init__(self, x[0], y[0])
        BiSpectrumNoBAO.__init__(self, x[1], y[1])
    
    def __call__(self, x, p):
        p_ = p.tolist()
        p1 = p_[:7]
        p2 = p_[7:]
        p2.insert(0, p1[0]) 

        r1 = PowerSpectrumNoBAO.__call__(self, x[0], p1)
        r2 = BiSpectrumNoBAO.__call__(self, x[1], p2)

        return np.concatenate([r1, r2])   
