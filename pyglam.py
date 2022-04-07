"""
    Code to process Bispectrum data

"""
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d

mpl.use('Agg') 

#--- input parameters ---
is_bk = False
ixmax  = 1000                        # maximum mock index
KMIN, KMAX = 0.004, 0.296            # for applying a cut on k, to reduce the cov matrix dimension
alphas = np.linspace(1.0, 1.8, 8000) # range of alphas


name_tag = 'glam_bk' if is_bk else 'glam_pk'
bkr_file = f'/mnt/data1/BispectrumGLAM/output/{name_tag}_0114.npz' 
alphas_file = f'/mnt/data1/BispectrumGLAM/output/{name_tag}_alphas.txt'

if is_bk:
	f_bao  = lambda ix:f'/mnt/data1/BispectrumGLAM/BAO/Bk_CatshortV.0114.{ix:04d}.h5' 
	f_nbao = lambda ix:f'/mnt/data1/BispectrumGLAM/noBAO/Bk_CatshortV.0114.{ix:04d}.h5'	
else:
	f_bao  = lambda ix:f'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/BAO/Pk_CatshortV.0114.{ix:04d}.DAT' 
	f_nbao = lambda ix:f'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/noBAO/Pk_CatshortV.0114.{ix:04d}.DAT'	


#--- helper functions
def read_bkfile(input_file):
	return np.loadtxt(input_file)

def check_if_exists(input_file): # if a file exists
    return os.path.exists(input_file)


def savez(output_file, **kwargs): 
	dirname = os.path.dirname(output_file)
	if not check_if_exists(dirname):os.makedirs(dirname)
	np.savez(output_file, **kwargs)
	print(f'saved {output_file}')


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
			#for i in range(3):assert np.array_equiv(bk1[:, i], bk2[:, i]), f"k%d does not match"%i
			bk1t += bk1[:, -1]
			bk2t += bk2[:, -1]
			bkr.append(bk1[:, -1])
	#print(bk1)
	nmocks = len(bkr)
	if bk1.shape[1] > 2:
		k = bk1[:, :3]
	else:
		k = bk1[:, 0]

	bkr = np.array(bkr).T / (bk2t[:, np.newaxis]/nmocks)
	bkrm = bk1t / bk2t
	return (k, bkr, bkrm)


def read(bkr_file): # read bispectrum ratio file and return mean and cov matrix
	if not check_if_exists(bkr_file):
		k, bkr, bkrm = read_ratios(ixmax)
		savez(bkr_file, **{'k':k, 'bkr':bkr, 'bkrm':bkrm})
	else:
		print(f'{bkr_file} exists. reading ...')
		bkr_ = np.load(bkr_file)
		k = bkr_['k']
		bkr = bkr_['bkr']
		bkrm = bkr_['bkrm']
	return (k, bkr, bkrm)


class Interpolate3D(object):
	""" Interpolater for B(k1, k2, k3). Uses symmetry to fill the 3D matrix of B
	"""
	def __init__(self, k, br):
		dk = k[1, 2]-k[0, 2]
		kmax = max(k[:, 2])
		kmin = min(k[:, 2])
		nk = int((kmax-kmin)/dk) + 1
		kg = np.arange(kmin, kmax+dk/2, dk)
		#print(f'kmin={kmin:.3f}, kmax={kmax:.3f}, dk={dk:.3f}, nk={nk:d}')
		br3d = np.zeros((nk, nk, nk))*np.nan
		#print(f'initialized 3D B(k): {br3d.shape} with k-grid = {kg}:')
		for ki in range(k.shape[0]):
			i = int((k[ki, 0]-kmin)/dk)
			j = int((k[ki, 1]-kmin)/dk)
			l = int((k[ki, 2]-kmin)/dk)
			br3d[i, j, l] = br[ki]
			br3d[i, l, j] = br[ki]
			br3d[j, i, l] = br[ki]
			br3d[j, l, i] = br[ki]
			br3d[l, j, i] = br[ki]
			br3d[l, i, j] = br[ki]

		self.br3d_int = rgi((kg, kg, kg), br3d, method='linear', bounds_error=False) # turn this to True to see the error
		self.kg = kg

	def __call__(self, *arrays):
		return self.br3d_int(*arrays)


class Interpolate1D(object):
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

def get_alpha1sig(k, bkrm, br, br3d, kmax=KMAX, kmin=KMIN):
	

	# apply cut on k
	#print(f'applying cut on k: {kmin:.3f} < k < {kmax:.3f}')
	is_good = np.ones(k.shape[0], '?')
	if len(k.shape) > 1:
		for i in range(3):is_good &= (k[:, i] > kmin) & (k[:, i] < kmax)
	else:
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
	#print("k1 k2 k3 B(k1, k2, k3) interpolation")
	#print(np.column_stack([kg[:5, :], bg[:5], br3d(kg[:5, :])]))

	# 
	#print("run 1D regression, varying alpha, k1'=ak1, k2'=ak2, k3'=ak3")
	#print("alpha chi2")
	alpha_1sig = np.nan
	for alpha in alphas:
		res  = bg - br3d(alpha*kg)
		chi2 = res.dot(icov.dot(res))
		#print(f'{alpha:.2f} {chi2:.5f}')
		if (abs(chi2-1) < 0.1):
			alpha_1sig = abs(alpha-1.0)
			break

	return alpha_1sig

def run():

	# read the ratio of bispectra and ratio of means
	k, br, bkrm = read(bkr_file)
	print(f'k shape: {k.shape}')
	print(f'br shape: {br.shape}')
	print(f'bkrm shape: {bkrm.shape}')

	if len(k.shape) > 1:
		# fill in the 3D matrix
		br_int = Interpolate3D(k, bkrm)
	else:
		br_int = Interpolate1D(k, bkrm)

	#kmin_, kmax_ = 0.1, 0.175
	#dalpha_ = get_alpha1sig(k, bkrm, br, br3d, kmax=kmax_, kmin=kmin_)
	#print('dalpha', dalpha_)
	alpha_1sig = []
	for kmax_ in np.arange(KMIN, KMAX, 0.01):
		for kmin_ in np.arange(KMIN, kmax_-0.02, 0.01):
			dalpha_ = get_alpha1sig(k, bkrm, br, br_int, kmax=kmax_, kmin=kmin_)
			alpha_1sig.append([kmin_, kmax_, dalpha_])
			print('.', end='')

	np.savetxt(alphas_file, np.array(alpha_1sig), header='kmin, kmax, alpha [1sigma]')
	print(f'wrote {alphas_file}')

if __name__ == '__main__':
	run()
