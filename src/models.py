import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from utils import solve_triangular_geometry

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





"""
    tree_level_b00(k1, k2, k3, b1, b2, f, pk)

    compute tree level bispectrum monopole analitically

    -Input:
    - k1, k2, k3::Array{Float64,1}: wavenumbers describing the triangle shape
    - b1, b2, f::Float64: bias parameters and the growth rete. 
    - pk::Interpolations.GriddedInterpolation: interpolator for the power
    spectrum
    -Return:
    -Bi::Array{Float64,1}: bispectrum at those triangles. Same size as k1,k2
    k3
"""
def tree_level_b00(k1, k2, k3, b1, b2, f, pk):
    
    pk1 = pk(k1)
    pk2 = pk(k2)
    pk3 = pk(k3)

    m12, m23, m31 = solve_triangular_geometry(k1, k2, k3)
    nu12 = sqrt(1 - m12**2)
    nu23 = sqrt(1 - m23**2)
    nu31 = sqrt(1 - m31**2)
 
    F12 = 5/7 + m12/2*(k1/k2 + k2/k1) + 2/7*m12**2
    F23 = 5/7 + m23/2*(k2/k3 + k3/k2) + 2/7*m23**2
    F31 = 5/7 + m31/2*(k3/k1 + k1/k3) + 2/7*m31**2
 
    G12 = 3/7 + m12/2*(k1/k2 + k2/k1) + 4/7*m12**2
    G23 = 3/7 + m23/2*(k2/k3 + k3/k2) + 4/7*m23**2
    G31 = 3/7 + m31/2*(k3/k1 + k1/k3) + 4/7*m31**2

    # The following long lines were computed in a WolframMathematica noteboook
    # The notebook is attached to this repo
    Bi = (k3*pk1*pk2*(105*b1**3*(6*F12*k1*k2 + f*k3*(k2*m31 - k1*m12*m31 + k1* nu12* nu31)) + 
    3*b1*f*(35*b2*k1*k2*(1 + m12**2 +  nu12**2) + f*(14*F12*k1*k2*(3*m12**2 +  nu12**2) + 14*G12*k1*k2*(m31**2*(3 + 3*m12**2 +  nu12**2) - 4*m12*m31* nu12* nu31 + 
          (1 + m12**2 + 3* nu12**2)* nu31**2) - 3*f*k3*(-(k2*(5*m12**4*m31 + m31* nu12**2*(2 +  nu12**2) + 2*m12**2*m31*(5 + 3* nu12**2) - 4*m12**3* nu12* nu31 - 
             4*m12* nu12*(1 +  nu12**2)* nu31)) + k1*(10*m12**3*m31 + m12*m31*(5 + 6* nu12**2) - 6*m12**2* nu12* nu31 -  nu12*(1 + 2* nu12**2)* nu31)))) + 
    21*b1**2*(15*b2*k1*k2 + f*(10*F12*k1*k2*(1 + m12**2 +  nu12**2) + 10*G12*k1*k2*(m31**2 +  nu31**2) + f*k3*(k2*(m31*(3 + 6*m12**2 + 2* nu12**2) - 4*m12* nu12* nu31) + 
          k1*(-3*m12**3*m31 - 3*m12*m31*(2 +  nu12**2) + 3*m12**2* nu12* nu31 +  nu12*(2 + 3* nu12**2)* nu31)))) + 
    f**2*(21*b2*k1*k2*(3*m12**2 +  nu12**2) + f*(f*k3*(k1*(-35*m12**3*m31 - 15*m12*m31* nu12**2 + 15*m12**2* nu12* nu31 + 3* nu12**3* nu31) + 
          k2*(35*m12**4*m31 + 30*m12**2*m31* nu12**2 + 3*m31* nu12**4 - 20*m12**3* nu12* nu31 - 12*m12* nu12**3* nu31)) + 
        18*G12*k1*k2*(-4*m12*m31* nu12* nu31 +  nu12**2*(m31**2 +  nu31**2) + m12**2*(5*m31**2 +  nu31**2))))) + 
  k2*pk1*pk3*(105*b1^3*(6*F31*k1*k3 + f*k2*(k3*m12 - k1*m12*m31 + k1* nu12* nu31)) + 
    21*b1**2*(15*b2*k1*k3 + f*(10*G31*k1*k3*(m12**2 +  nu12**2) + 10*F31*k1*k3*(1 + m31**2 +  nu31**2) + f*k2*(-4*k3*m31* nu12* nu31 - 3*k1*m12*m31*(2 + m31**2 +  nu31**2) + 
          k3*m12*(3 + 6*m31**2 + 2* nu31**2) + k1* nu12* nu31*(2 + 3*m31**2 + 3* nu31**2)))) + 
    f**2*(21*b2*k1*k3*(3*m31**2 +  nu31**2) + f*(18*G31*k1*k3*(-4*m12*m31* nu12* nu31 +  nu12**2*(m31**2 +  nu31**2) + m12**2*(5*m31**2 +  nu31**2)) + 
        f*k2*(3*k1* nu12* nu31*(5*m31**2 +  nu31**2) - 4*k3*m31* nu12* nu31*(5*m31**2 + 3* nu31**2) - 5*k1*m12*(7*m31**3 + 3*m31* nu31**2) + 
          k3*m12*(35*m31**4 + 30*m31**2* nu31**2 + 3* nu31**4)))) + 3*b1*f*(35*b2*k1*k3*(1 + m31**2 +  nu31**2) + 
      f*(14*F31*k1*k3*(3*m31**2 +  nu31**2) + 14*G31*k1*k3*(-4*m12*m31* nu12* nu31 + m12**2*(3 + 3*m31**2 +  nu31**2) +  nu12**2*(1 + m31**2 + 3* nu31**2)) - 
        3*f*k2*(4*k3*m31* nu12* nu31*(1 + m31**2 +  nu31**2) - k3*m12*(5*m31**4 +  nu31**2*(2 +  nu31**2) + 2*m31**2*(5 + 3* nu31**2)) + 
          k1*(-( nu12* nu31*(1 + 6*m31**2 + 2* nu31**2)) + m12*m31*(5 + 10*m31**2 + 6* nu31**2)))))) + 
  k1*pk2*pk3*(105*b1^3*(6*F23*k2*k3 + f*k1*(k3*m12 + k2*m31)) + 
    21*b1**2*(15*b2*k2*k3 + f*(10*G23*k2*k3 + 10*F23*k2*k3*(m12**2 + m31**2 +  nu12**2 +  nu31**2) + f*k1*(k3*(3*m12**3 + 6*m12*m31**2 + 3*m12* nu12**2 - 4*m31* nu12* nu31 + 2*m12* nu31**2) + 
          k2*(6*m12**2*m31 + 3*m31**3 + 2*m31* nu12**2 - 4*m12* nu12* nu31 + 3*m31* nu31**2)))) + 
    f**2*(21*b2*k2*k3*(-4*m12*m31* nu12* nu31 + m12**2*(3*m31**2 +  nu31**2) +  nu12**2*(m31**2 + 3* nu31**2)) + 
      f*(18*G23*k2*k3*(-4*m12*m31* nu12* nu31 +  nu12**2*(m31**2 +  nu31**2) + m12**2*(5*m31**2 +  nu31**2)) + 
        f*k1*(k2*(-12*m12**3* nu12* nu31*(5*m31**2 +  nu31**2) + 3*m31* nu12**4*(m31**2 + 5* nu31**2) - 4*m12* nu12**3* nu31*(9*m31**2 + 5* nu31**2) + 6*m12**2*m31* nu12**2*(5*m31**2 + 9* nu31**2) + 
            5*m12**4*(7*m31**3 + 3*m31* nu31**2)) + k3*(-12*m12**2*m31* nu12* nu31*(5*m31**2 + 3* nu31**2) - 4*m31* nu12**3* nu31*(3*m31**2 + 5* nu31**2) + 
            m12**3*(35*m31**4 + 30*m31**2* nu31**2 + 3* nu31**4) + 3*m12* nu12**2*(5*m31**4 + 18*m31**2* nu31**2 + 5* nu31**4))))) + 
    3*b1*f*(35*b2*k2*k3*(m12**2 + m31**2 +  nu12**2 +  nu31**2) + f*(14*G23*k2*k3*(3*m12**2 + 3*m31**2 +  nu12**2 +  nu31**2) + 
        14*F23*k2*k3*(-4*m12*m31* nu12* nu31 + m12**2*(3*m31**2 +  nu31**2) +  nu12**2*(m31**2 + 3* nu31**2)) + 
        3*f*k1*(k3*(-12*m12**2*m31* nu12* nu31 + 2*m12**3*(5*m31**2 +  nu31**2) - 4*m31* nu12* nu31*(m31**2 +  nu12**2 +  nu31**2) + m12*(m31**2 +  nu31**2)*(5*m31**2 + 6* nu12**2 +  nu31**2)) + 
          k2*(5*m12**4*m31 - 4*m12**3* nu12* nu31 - 4*m12* nu12* nu31*(3*m31**2 +  nu12**2 +  nu31**2) + m31* nu12**2*(2*m31**2 +  nu12**2 + 6* nu31**2) + 
            2*m12**2*m31*(5*m31**2 + 3*( nu12**2 +  nu31**2))))))))/(630*k1*k2*k3)

    return Bi
