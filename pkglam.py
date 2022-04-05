"""
    Code to process power spectrum data

"""
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from scipy.interpolate import interp1d

mpl.use('Agg') 

#--- input parameters ---
ngrid = 80
ixmax  = 1000                        # maximum mock index (FIX: some mocks are missing)
KMIN, KMAX = 0.004, 0.296            # for applying a cut on k, to reduce the cov matrix dimension
kmin_range = [0.005, ]
kmax_range = [0.296]
alphas = np.linspace(0.9, 1.1, 201) # range of alphas
bkr_file = '/mnt/data1/BispectrumGLAM/output/pkr_0114.npz'      # a numpy binary file that has k and ratios of power spectra
output2dalpha = '/mnt/data1/BispectrumGLAM/output/alpha2d_pk.txt'  # a txt file that will contain kmin, kmax, alpha_1simga


f_bao  = lambda ix:f'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/BAO/Pk_CatshortV.0114.{ix:04d}.DAT' 
f_nbao = lambda ix:f'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/noBAO/Pk_CatshortV.0114.{ix:04d}.DAT'	

def read_bkfile(input_file):
	with open(input_file, 'r') as fl:
		lines = fl.readlines()
		bk = []
		for line in lines:
			k1, b = map(float, line.split('\t'))
			bk.append([k1, b])
	return np.array(bk)

def check_if_exists(input_file): # if a file exists
    return os.path.exists(input_file)

def savez(output_file, **kwargs): 
	dirname = os.path.dirname(output_file)
	if not check_if_exists(dirname):os.makedirs(dirname)
	np.savez(output_file, **kwargs)
	#print(f'saved {output_file}')

def read_ratios(ixmax): # read bispectra files and return ratios
	bkr = []
	bk1t = 0
	bk2t = 0
	for ix in range(ixmax+1):
		f_bao_i = f_bao(ix)
		f_nbao_i = f_nbao(ix)
		if (check_if_exists(f_bao_i) and check_if_exists(f_nbao_i)):
			bk1 = read_bkfile(f_bao_i)
			bk2 = read_bkfile(f_nbao_i)
			for i in range(1):assert np.array_equiv(bk1[:, i], bk2[:, i]), f"k%d does not match"%i
			bk1t += bk1[:, -1]
			bk2t += bk2[:, -1]
			bkr.append(bk1[:, -1]/bk2[:, -1])
	#print(bk1)
	k = bk1[:, 0]
	bkr = np.array(bkr).T
	bkrm = bk1t / bk2t
	return (k, bkr, bkrm)

def read(bkr_file): # read bispectrum ratio file and return mean and cov matrix
	if not check_if_exists(bkr_file):
		k, bkr, bkrm = read_ratios(ixmax)
		savez(bkr_file, **{'k':k, 'bkr':bkr, 'bkrm':bkrm})
	else:
		#print(f'{bkr_file} exists. reading ...')
		bkr_ = np.load(bkr_file)
		k = bkr_['k']
		bkr = bkr_['bkr']
		bkrm = bkr_['bkrm']
	return (k, bkr, bkrm)

class Interpolate3D(object):
	""" Interpolater for Pk(k). Uses symmetry to fill the 1D array of P(k)
	"""
	def __init__(self, k, br):
		kmax = max(k)
		kmin = min(k)
		dk = k[1]-k[0]
		nk = k.size
		#print(f'kmin={kmin:.3f}, kmax={kmax:.3f}, dk={dk:.3f}, nk={nk:d}')
		self.br3d_int = interp1d(k, br, bounds_error=False) # turn this to True to see the error

	def __call__(self, array):
		return self.br3d_int(array)

def get_cov(k, bkrm, br, kmax=KMAX, kmin=KMIN):
    # apply cut on k
    #print(f'applying cut on k: {kmin:.3f} < k < {kmax:.3f}')
    is_good = np.ones(k.shape[0], '?')
    if len(k.shape) > 1:
        for i in range(3):
            is_good &= (k[:, i] > kmin) & (k[:, i] < kmax)
    else:
        is_good &= (k > kmin) & (k < kmax)
    kg = k[is_good]
    bg = bkrm[is_good]
    nbins, nmocks = br[is_good, :].shape
    hartlapf = (nmocks-1.0)/(nmocks-nbins-2.0)
    print(f'kmax={kmax}, kmin={kmin}, nbins={nbins}, nmocks={nmocks}')
    print(f'hartlap factor: {hartlapf:.3f}')
    #print(f'kg: {kg}')
    #print(f'bk ratio: {bg}')
    return np.cov(br[is_good, :], rowvar=True)*hartlapf / nmocks    
    
    
def get_alpha1sig(k, bkrm, br, br3d, kmax=KMAX, kmin=KMIN):
	

	# apply cut on k
	#print(f'applying cut on k: {kmin:.3f} < k < {kmax:.3f}')
	is_good = np.ones(k.shape[0], '?')
	is_good &= (k > kmin) & (k < kmax)
	kg = k[is_good]
	bg = bkrm[is_good]
	nbins, nmocks = br[is_good, :].shape
	hartlapf = (nmocks-1.0)/(nmocks-nbins-2.0)
	#print(f'kmax={kmax}, kmin={kmin}, nbins={nbins}, nmocks={nmocks}')
	#print(f'kg: {kg}')
	#print(f'bk ratio: {bg}')
	cov = np.cov(br[is_good, :], rowvar=True)*hartlapf / nmocks
	if np.linalg.det(cov) == 0.0:
		return np.nan

	#print(f'cov. {cov}')
	icov = np.linalg.inv(cov)
	#print(f'k shape: {kg.shape}')
	#print(f'bkrm shape: {bg.shape}')

	# check interpolation
	#print("checking the input k points and interpolated values")
	#print("k P(k) interpolation")
	#print(np.column_stack([kg[:5], bg[:5], br3d(kg[:5])]))

	# 
	#print("run 1D regression, varying alpha, k1'=ak1 ")
	#print("alpha chi2")
	alpha_1sig = np.nan
	for alpha in alphas:
		res  = bg - br3d(alpha*kg)
		chi2 = res.dot(icov.dot(res))
		print(f'{alpha:.3f} {chi2:.5f}')
		if (abs(chi2-1) < 0.1):
			alpha_1sig = alpha
			#break

	return abs(alpha_1sig-1.)

def run():

	# read the ratio of bispectra and ratio of means
	k, br, bkrm = read(bkr_file)
	#print(f'k shape: {k.shape}')
	#print(f'br shape: {br.shape}')
	#print(f'bkrm shape: {bkrm.shape}')

	# fill in the 3D matrix
	br3d = Interpolate3D(k, bkrm)
	#kmin_, kmax_ = 0.1, 0.175
	#dalpha_ = get_alpha1sig(k, bkrm, br, br3d, kmax=kmax_, kmin=kmin_)
	#print('dalpha', dalpha_)
	alpha_1sig = []
	for kmax_ in kmax_range: #np.arange(KMIN, KMAX, 0.01):
		for kmin_ in kmin_range: #np.arange(KMIN, kmax_-0.02, 0.01):
			dalpha_ = get_alpha1sig(k, bkrm, br, br3d, kmax=kmax_, kmin=kmin_)
			alpha_1sig.append([kmin_, kmax_, dalpha_])

	np.savetxt(output2dalpha, np.array(alpha_1sig), header='kmin, kmax, alpha [1sigma]')
	#print(f'wrote {output2dalpha}')

if __name__ == '__main__':
	run()
