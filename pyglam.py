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

mpl.use('Agg') 

#--- input parameters ---
ixmax  = 1000                                  # maximum mock index (FIX: some mocks are missing)
KMIN, KMAX = 0.1, 0.175                         # for applying a cut on k, to reduce the cov matrix dimension
bkr_file = '/mnt/data1/BispectrumGLAM/output/bkr_0114.npz'  # a numpy file that has k1,k2,k3 and ratios of bispectra
f_bao  = lambda ix:f'/mnt/data1/BispectrumGLAM/BAO/Bk_CatshortV.0114.{ix:04d}.h5' 
f_nbao = lambda ix:f'/mnt/data1/BispectrumGLAM/noBAO/Bk_CatshortV.0114.{ix:04d}.h5'	

def read_bkfile(input_file):
	with open(input_file, 'r') as fl:
		lines = fl.readlines()
		bk = []
		for line in lines:
			k1, k2, k3, b = map(float, line.split('\t'))
			bk.append([k1, k2, k3, b])
	return np.array(bk)

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
			for i in range(3):assert np.array_equiv(bk1[:, i], bk2[:, i]), f"k%d does not match"%i
			bk1t += bk1[:, -1]
			bk2t += bk2[:, -1]
			bkr.append(bk1[:, -1]/bk2[:, -1])
	print(bk1)
	k = bk1[:, :3]
	bkr = np.array(bkr).T
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
		print(f'kmin={kmin:.3f}, kmax={kmax:.3f}, dk={dk:.3f}, nk={nk:d}')
		br3d = np.zeros((nk, nk, nk))*np.nan
		print(f'initialized 3D B(k): {br3d.shape} with k-grid = {kg}:')
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

def run(nalpha):
	alphas = np.linspace(0.9, 1.1, nalpha) # range of alphas

	# read the ratio of bispectra and ratio of means
	k, br, bkrm = read(bkr_file)
	print(f'k shape: {k.shape}')
	print(f'br shape: {br.shape}')
	print(f'bkrm shape: {bkrm.shape}')

	# fill in the 3D matrix
	br3d = Interpolate3D(k, bkrm)

	# apply cut on k
	print(f'applying cut on k: {KMIN:.3f} < k < {KMAX:.3f}')
	is_good = np.ones(k.shape[0], '?')
	for i in range(3):is_good &= (k[:, i] > KMIN) & (k[:, i] < KMAX)
	kg = k[is_good, :]
	bg = bkrm[is_good]
	hartlapf = 1.0 # FIXME
	cov = np.cov(br[is_good, :], rowvar=True)*hartlapf / br.shape[1]
	icov = np.linalg.inv(cov)
	del br
	print(f'k shape: {kg.shape}')
	print(f'bkrm shape: {bg.shape}')

	# check interpolation
	print("checking the input k points and interpolated values")
	print("k1 k2 k3 B(k1, k2, k3) interpolation")
	print(np.column_stack([kg[:5, :], bg[:5], br3d(kg[:5, :])]))

	# 
	print("run 1D regression, varying alpha, k1'=ak1, k2'=ak2, k3'=ak3")
	print("alpha chi2")
	for alpha in alphas:
		res  = bg - br3d(alpha*kg)
		chi2 = res.dot(icov.dot(res))
		print(f'{alpha:.2f} {chi2:.5f}')


if __name__ == '__main__':
	run(int(sys.argv[1]))
