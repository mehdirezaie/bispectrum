import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d


class PowerSpectrum(object):
    
    def __init__(self, x, y):
        self.y_int = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    
    def __call__(self, x, p): # 7 params
        return p[1]*self.y_int(p[0]*x) + p[2] + p[3]*x + p[4]*(1./x) + p[5]*(x*x) + p[6]*(1./(x*x))
    
class BiSpectrum(object):
    def __init__(self, x, y):
        self.y_int = LinearNDInterpolator(x, y, fill_value=np.nan)
    
    def __call__(self, x, p): # 11 params
        return p[1]*self.y_int(p[0]*x) + p[2] + p[3]*(x[:, 0] + x[:, 1] + x[:, 2]) \
                + p[4]*(1./x[:, 0] + 1./x[:, 1] + 1/x[:, 2]) \
                + p[5]*(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) \
                + p[6]*(1./x[:, 0]**2 + 1./x[:, 1]**2 + 1./x[:, 2]**2) \
                + p[7]*(x[:, 0]*x[:, 1] + x[:, 0]*x[:, 2] + x[:, 1]*x[:, 2]) \
                + p[8]*(1./(x[:, 0]*x[:, 1]) + 1./(x[:, 0]*x[:, 2]) + 1./(x[:, 1]*x[:, 2])) \
                + p[9]*(x[:, 0]*x[:, 1]/x[:, 2] + x[:, 0]*x[:, 2]/x[:, 1] + x[:, 1]*x[:, 2]/x[:, 0]) \
                + p[10]*(x[:, 2]/(x[:, 0]*x[:, 1]) + x[:, 1]/(x[:, 0]*x[:, 2]) + x[:, 0]/(x[:, 1]*x[:, 2]))    