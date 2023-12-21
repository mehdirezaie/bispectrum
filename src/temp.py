import numpy as np
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit

from scipy import integrate
from numba import jit


def tree_level_b00(k1, k2, k3, f, b1, b2, s0, s1, pk):
    
    twoY00=np.sqrt(1./np.pi)
    
    pk1 = pk(k1)
    pk2 = pk(k2)
    pk3 = pk(k3)

    mu12, mu23, mu31 = solve_triangular_geometry(k1, k2, k3)
    nu12 = np.sqrt(1 - mu12**2)
    nu23 = np.sqrt(1 - mu23**2)
    nu31 = np.sqrt(1 - mu31**2)
 
    F12 = 5/7 + mu12/2*(k1/k2 + k2/k1) + 2/7*mu12**2
    F23 = 5/7 + mu23/2*(k2/k3 + k3/k2) + 2/7*mu23**2
    F31 = 5/7 + mu31/2*(k3/k1 + k1/k3) + 2/7*mu31**2
 
    G12 = 3/7 + mu12/2*(k1/k2 + k2/k1) + 4/7*mu12**2
    G23 = 3/7 + mu23/2*(k2/k3 + k3/k2) + 4/7*mu23**2
    G31 = 3/7 + mu31/2*(k3/k1 + k1/k3) + 4/7*mu31**2

    # The following long lines were computed in a WolframMathematica noteboook
    # The notebook is attached to this repo
    Bi = (k3*pk1*pk2*(105*b1**3*(6*F12*k1*k2 + f*k3*(k2*mu31 - k1*mu12*mu31 + k1* nu12* nu31)) + 
    3*b1*f*(35*b2*k1*k2*(1 + mu12**2 +  nu12**2) + f*(14*F12*k1*k2*(3*mu12**2 +  nu12**2) + 14*G12*k1*k2*(mu31**2*(3 + 3*mu12**2 +  nu12**2) - 4*mu12*mu31* nu12* nu31 + 
          (1 + mu12**2 + 3* nu12**2)* nu31**2) - 3*f*k3*(-(k2*(5*mu12**4*mu31 + mu31* nu12**2*(2 +  nu12**2) + 2*mu12**2*mu31*(5 + 3* nu12**2) - 4*mu12**3* nu12* nu31 - 
             4*mu12* nu12*(1 +  nu12**2)* nu31)) + k1*(10*mu12**3*mu31 + mu12*mu31*(5 + 6*nu12**2) - 6*mu12**2* nu12* nu31 -  nu12*(1 + 2* nu12**2)* nu31)))) + 
    21*b1**2*(15*b2*k1*k2 + f*(10*F12*k1*k2*(1 + mu12**2 +  nu12**2) + 10*G12*k1*k2*(mu31**2 +  nu31**2) + f*k3*(k2*(mu31*(3 + 6*mu12**2 + 2* nu12**2) - 4*mu12* nu12* nu31) + 
          k1*(-3*mu12**3*mu31 - 3*mu12*mu31*(2 +  nu12**2) + 3*mu12**2* nu12* nu31 +  nu12*(2 + 3* nu12**2)* nu31)))) + 
    f**2*(21*b2*k1*k2*(3*mu12**2 +  nu12**2) + f*(f*k3*(k1*(-35*mu12**3*mu31 - 15*mu12*mu31* nu12**2 + 15*mu12**2* nu12* nu31 + 3* nu12**3* nu31) + 
          k2*(35*mu12**4*mu31 + 30*mu12**2*mu31* nu12**2 + 3*mu31* nu12**4 - 20*mu12**3* nu12* nu31 - 12*mu12* nu12**3* nu31)) + 
        18*G12*k1*k2*(-4*mu12*mu31* nu12* nu31 +  nu12**2*(mu31**2 +  nu31**2) + mu12**2*(5*mu31**2 +  nu31**2))))) + 
  k2*pk1*pk3*(105*b1**3*(6*F31*k1*k3 + f*k2*(k3*mu12 - k1*mu12*mu31 + k1* nu12* nu31)) + 
    21*b1**2*(15*b2*k1*k3 + f*(10*G31*k1*k3*(mu12**2 +  nu12**2) + 10*F31*k1*k3*(1 + mu31**2 +  nu31**2) + f*k2*(-4*k3*mu31* nu12* nu31 - 3*k1*mu12*mu31*(2 + mu31**2 +  nu31**2) + 
          k3*mu12*(3 + 6*mu31**2 + 2* nu31**2) + k1* nu12* nu31*(2 + 3*mu31**2 + 3* nu31**2)))) + 
    f**2*(21*b2*k1*k3*(3*mu31**2 +  nu31**2) + f*(18*G31*k1*k3*(-4*mu12*mu31* nu12* nu31 +  nu12**2*(mu31**2 +  nu31**2) + mu12**2*(5*mu31**2 +  nu31**2)) + 
        f*k2*(3*k1* nu12* nu31*(5*mu31**2 +  nu31**2) - 4*k3*mu31* nu12* nu31*(5*mu31**2 + 3* nu31**2) - 5*k1*mu12*(7*mu31**3 + 3*mu31* nu31**2) + 
          k3*mu12*(35*mu31**4 + 30*mu31**2* nu31**2 + 3* nu31**4)))) + 3*b1*f*(35*b2*k1*k3*(1 + mu31**2 +  nu31**2) + 
      f*(14*F31*k1*k3*(3*mu31**2 +  nu31**2) + 14*G31*k1*k3*(-4*mu12*mu31* nu12* nu31 + mu12**2*(3 + 3*mu31**2 +  nu31**2) +  nu12**2*(1 + mu31**2 + 3* nu31**2)) - 
        3*f*k2*(4*k3*mu31* nu12* nu31*(1 + mu31**2 +  nu31**2) - k3*mu12*(5*mu31**4 +  nu31**2*(2 +  nu31**2) + 2*mu31**2*(5 + 3* nu31**2)) + 
          k1*(-( nu12* nu31*(1 + 6*mu31**2 + 2* nu31**2)) + mu12*mu31*(5 + 10*mu31**2 + 6* nu31**2)))))) + 
  k1*pk2*pk3*(105*b1**3*(6*F23*k2*k3 + f*k1*(k3*mu12 + k2*mu31)) + 
    21*b1**2*(15*b2*k2*k3 + f*(10*G23*k2*k3 + 10*F23*k2*k3*(mu12**2 + mu31**2 +  nu12**2 +  nu31**2) + f*k1*(k3*(3*mu12**3 + 6*mu12*mu31**2 + 3*mu12* nu12**2 - 4*mu31* nu12* nu31 + 2*mu12* nu31**2) + 
          k2*(6*mu12**2*mu31 + 3*mu31**3 + 2*mu31* nu12**2 - 4*mu12* nu12* nu31 + 3*mu31* nu31**2)))) + 
    f**2*(21*b2*k2*k3*(-4*mu12*mu31* nu12* nu31 + mu12**2*(3*mu31**2 +  nu31**2) +  nu12**2*(mu31**2 + 3* nu31**2)) + 
      f*(18*G23*k2*k3*(-4*mu12*mu31* nu12* nu31 +  nu12**2*(mu31**2 +  nu31**2) + mu12**2*(5*mu31**2 +  nu31**2)) + 
        f*k1*(k2*(-12*mu12**3* nu12* nu31*(5*mu31**2 +  nu31**2) + 3*mu31* nu12**4*(mu31**2 + 5* nu31**2) - 4*mu12* nu12**3* nu31*(9*mu31**2 + 5* nu31**2) + 6*mu12**2*mu31* nu12**2*(5*mu31**2 + 9* nu31**2) + 
            5*mu12**4*(7*mu31**3 + 3*mu31* nu31**2)) + k3*(-12*mu12**2*mu31* nu12* nu31*(5*mu31**2 + 3* nu31**2) - 4*mu31* nu12**3* nu31*(3*mu31**2 + 5* nu31**2) + 
            mu12**3*(35*mu31**4 + 30*mu31**2* nu31**2 + 3* nu31**4) + 3*mu12* nu12**2*(5*mu31**4 + 18*mu31**2* nu31**2 + 5* nu31**4))))) + 
    3*b1*f*(35*b2*k2*k3*(mu12**2 + mu31**2 +  nu12**2 +  nu31**2) + f*(14*G23*k2*k3*(3*mu12**2 + 3*mu31**2 +  nu12**2 +  nu31**2) + 
        14*F23*k2*k3*(-4*mu12*mu31* nu12* nu31 + mu12**2*(3*mu31**2 +  nu31**2) +  nu12**2*(mu31**2 + 3* nu31**2)) + 
        3*f*k1*(k3*(-12*mu12**2*mu31* nu12* nu31 + 2*mu12**3*(5*mu31**2 +  nu31**2) - 4*mu31* nu12* nu31*(mu31**2 +  nu12**2 +  nu31**2) + mu12*(mu31**2 +  nu31**2)*(5*mu31**2 + 6* nu12**2 +  nu31**2)) + 
          k2*(5*mu12**4*mu31 - 4*mu12**3* nu12* nu31 - 4*mu12* nu12* nu31*(3*mu31**2 +  nu12**2 +  nu31**2) + mu31* nu12**2*(2*mu31**2 +  nu12**2 + 6* nu31**2) + 
            2*mu12**2*mu31*(5*mu31**2 + 3*( nu12**2 +  nu31**2))))))))/(630*k1*k2*k3)

    Bi = (Bi + (pk1 + pk2 + pk3)*s1 + s0)*twoY00
    return Bi



def solve_triangular_geometry(k1, k2, k3):
    #mu12 = (k1**2 + k2**2 - k3**2)/(2*k1*k2)
    #mu31 = (k3**2 + k1**2 - k2**2)/(2*k3*k1)
    #mu23 = (k2**2 + k3**2 - k1**2)/(2*k2*k3)
    
    mu12 = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    mu31 = -(k1 + k2*mu12)/k3
    mu23 = -(k1*mu12 + k2)/k3
    return mu12, mu23, mu31

@jit
def Bisp(var, parc, pk1,pk2,pk3,linear = True):
    '''
    Scoccimaro Bispectrum as a function of 5 variables - k1,k2,k3,mu1,phi12 and cosmological
    parameters parc.
    '''
    k1, k2, k3, mu1, phi12 = var
    #apar, aper, f, b1, b2 = parc
    f, b1, b2 = parc

    mu12, mu23, mu31 = solve_triangular_geometry(k1, k2, k3)    
    mu2 = mu1*mu12 - np.sqrt(1 - mu1**2)*np.sqrt(1 - mu12**2)*np.cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3

    Z1k1 = b1 + f*mu1**2
    Z1k2 = b1 + f*mu2**2
    Z1k3 = b1 + f*mu3**2

    F12 = 5./7. + mu12/2*(k1/k2 + k2/k1) + 2./7.*mu12**2
    F23 = 5./7. + mu23/2*(k2/k3 + k3/k2) + 2./7.*mu23**2
    F31 = 5./7. + mu31/2*(k3/k1 + k1/k3) + 2./7.*mu31**2

    G12 = 3./7. + mu12/2*(k1/k2 + k2/k1) + 4./7.*mu12**2
    G23 = 3./7. + mu23/2*(k2/k3 + k3/k2) + 4./7.*mu23**2
    G31 = 3./7. + mu31/2*(k3/k1 + k1/k3) + 4./7.*mu31**2

    Z2k12 = b2/2. + b1*F12 + f*mu3**2*G12
    Z2k12 -= f*mu3*k3/2.*(mu1/k1*Z1k2 + mu2/k2*Z1k1)
    Z2k23 = b2/2. + b1*F23 + f*mu1**2*G23
    Z2k23 -= f*mu1*k1/2.*(mu2/k2*Z1k3 + mu3/k3*Z1k2)
    Z2k31 = b2/2. + b1*F31 + f*mu2**2*G31
    Z2k31 -= f*mu2*k2/2.*(mu3/k3*Z1k1 + mu1/k1*Z1k3)

    if not linear:
        I = 2*(b1**2+f**2/5+2*b1*f/3)
        pk1 = pk1/I
        pk2 = pk2/I
        pk3 = pk3/I
    
    Bi = Z2k12*Z1k1*Z1k2*pk1*pk2
    Bi += Z2k23*Z1k2*Z1k3*pk2*pk3
    Bi += Z2k31*Z1k3*Z1k1*pk3*pk1

    return 2.*Bi


    

def Bi0(k1, k2, k3, f, b1, b2, s0, s1, pk):
    '''
    Integrate Bisp to get the monopole
    '''
    pk1 = pk(k1)
    pk2 = pk(k2)
    pk3 = pk(k3)
    
    Y00=np.sqrt(1/np.pi)/2
    parc=(f,b1,b2)
    out=[]
    for i in range(len(k1)):
        func = lambda phi12,mu1: Bisp((k1[i],k2[i],k3[i],mu1,phi12), parc,pk1[i],pk2[i],pk3[i])
        ans,_ = integrate.dblquad(func, -1, 1, lambda x: 0, lambda x: np.pi)
        ans = ans + s0 + s1 * (pk1[i] + pk2[i] + pk3[i]) # Add extra nuissance params. This term is not a part of scoccimaro equation. Skip this step if you want to use only scoccimaro eq.
        out.append(2*Y00*ans)
    return np.array(out)


def Bi_wiggle(k1, k2, k3, f, b1, b2, s0, s1, pk, pk_smooth): #kk,pk1,pk2,pk3,pkn1,pkn2,pkn3,f,b1,b2,S0,S1):
    '''
    pkm - full mean power spectrum
    pknm - nobao/smooth mean power spectrum
    k - respective k-values at which power spectrum is measured
    kk - k-triplets where bispectrum is measured
    f,b1,b2 - free parameters
    '''
    B_full = Bi0(k1, k2, k3, b1, b2, f, s0, s1, pk) # Bi0(kk,pk1,pk2,pk3,f,b1,b2,S0,S1)
    B_nbao = Bi0(k1, k2, k3, b1, b2, f, s0, s1, pk_smooth) #Bi0(kk,pkn1,pkn2,pkn3,f,b1,b2,S0,S1)
    B_wig = B_full/B_nbao
    
    return B_wig