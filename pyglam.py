"""
    Code to process Bispectrum data

"""
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import emcee

from glob import glob
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool



mpl.use('Agg') 

#--- input parameters ---
is_bk = False
debug = False

npts = 100
ixmax = 1000                        # maximum mock index
KMIN, KMAX = 0.004, 0.296            # for applying a cut on k, to reduce the cov matrix dimension
alphas = np.linspace(1.0, 1.1, npts) # range of alphas
SEED = 85


name_tag = 'glam_bk' if is_bk else 'glam_pk'
bkr_file = f'/mnt/data1/BispectrumGLAM/output/{name_tag}_0114.npz' 
alphas_file = f'/mnt/data1/BispectrumGLAM/output/{name_tag}_alphas.txt'
mcmc_file = f'/mnt/data1/BispectrumGLAM/output/{name_tag}_mcmc.npz'
 
np.random.seed(SEED)



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
	if is_bk:
		f_bao  = lambda ix:f'/mnt/data1/BispectrumGLAM/BAO/Bk_CatshortV.0114.{ix:04d}.h5' 
		f_nbao = lambda ix:f'/mnt/data1/BispectrumGLAM/noBAO/Bk_CatshortV.0114.{ix:04d}.h5'	
	else:
		f_bao  = lambda ix:f'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/BAO/Pk_CatshortV.0114.{ix:04d}.DAT' 
		f_nbao = lambda ix:f'/mnt/data1/BispectrumGLAM/PowerspectrumGLAM/z0.50/noBAO/Pk_CatshortV.0114.{ix:04d}.DAT'	

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
	assert (len(k.shape)>1) == is_bk
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
		self.br3d_int = interp1d(k, br, bounds_error=False, fill_value=0.0) # turn this to True to see the error

	def __call__(self, array):
		return self.br3d_int(array)


def select_k(k, kmin, kmax, bkrm, br):
	# apply cut on k
	print(f'applying cut on k: {kmin:.3f} < k < {kmax:.3f}')
	is_good = np.ones(k.shape[0], '?')
	if is_bk:
		for i in range(3):is_good &= (k[:, i] > kmin) & (k[:, i] < kmax)
	else:
		is_good &= (k > kmin) & (k < kmax)
	
	kg = k[is_good]
	bg = bkrm[is_good]
	nbins, nmocks = br[is_good, :].shape
	hartlapf = (nmocks-1.0)/(nmocks-nbins-2.0)
	print(f'kmax={kmax}, kmin={kmin}, nbins={nbins}, nmocks={nmocks}')
	print(f'hartlap: {hartlapf:.3f}')
	print(f'kg: {kg}')
	print(f'bk ratio: {bg}')
	cov = np.cov(br[is_good, :], rowvar=True)*hartlapf / nmocks
	if np.linalg.det(cov) == 0.0:
		raise RuntimeError("singular covariance, change kmin and kmax")

	icov = np.linalg.inv(cov)
	print(f'k shape: {kg.shape}')
	print(f'bkrm shape: {bg.shape}')
	return kg, bg, icov



def get_alpha1sig(k, bkrm, br, br3d, kmax=KMAX, kmin=KMIN):
	# apply cut on k
	kg, bg, icov = select_k(k, kmin, kmax, bkrm, br)
	
	#print("run 1D regression, varying alpha, k1'=ak1, k2'=ak2, k3'=ak3")
	#print("alpha chi2")
	alpha_1sig = np.nan
	for alpha in alphas:
		res  = bg - br3d(alpha*kg)
		chi2 = res.dot(icov.dot(res))
		if debug:print(f'{alpha:.3f} {chi2:.5f}')
		if (chi2 > 1):
			alpha_1sig = abs(alpha-1.0)
			break
	return alpha_1sig


def run_alpha2d():

	# read the ratio of bispectra and ratio of means
	k, br, bkrm = read(bkr_file)
	print(f'k shape: {k.shape}')
	print(f'br shape: {br.shape}')
	print(f'bkrm shape: {bkrm.shape}')

	if is_bk:
		# fill in the 3D matrix
		br_int = Interpolate3D(k, bkrm)
	else:
		br_int = Interpolate1D(k, bkrm)


	if debug:
		kmin_, kmax_ = 0.1, 0.15
		dalpha_ = get_alpha1sig(k, bkrm, br, br_int, kmax=kmax_, kmin=kmin_)
		print('dalpha', dalpha_)
		sys.exit()

	alpha_1sig = []
	for kmax_ in np.arange(KMIN, KMAX, 0.01):
		for kmin_ in np.arange(KMIN, kmax_-0.02, 0.01):
			dalpha_ = get_alpha1sig(k, bkrm, br, br_int, kmax=kmax_, kmin=kmin_)
			alpha_1sig.append([kmin_, kmax_, dalpha_])
			print('.', end='')

	np.savetxt(alphas_file, np.array(alpha_1sig), header='kmin, kmax, alpha [1sigma]')
	print(f'wrote {alphas_file}')



class Posterior:
    """ Log Posterior for Glam
    """
    def __init__(self, model, y, invcov, x):
        self.model = model
        self.y = y
        self.invcov = invcov
        self.x = x

    def logprior(self, theta):
        ''' The natural logarithm of the prior probability. '''
        lp = 0.
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        a, b, c, d = theta
        lp += 0. if 0.8 < a < 1.2 else -np.inf
        for param in [b, c, d]:
	        lp += 0. if -2. < param < 2. else -np.inf
        
        ## Gaussian prior on ?
        #mmu = 3.     # mean of the Gaussian prior
        #msigma = 10. # standard deviation of the Gaussian prior
        #lp += -0.5*((m - mmu)/msigma)**2

        return lp

    def loglike(self, theta):
        '''The natural logarithm of the likelihood.'''
        # evaluate the model
        md = self.model(self.x, theta)
        # return the log likelihood
        return -0.5*(self.y-md).dot(self.invcov.dot(self.y-md))

    def logpost(self, theta):
        '''The natural logarithm of the posterior.'''
        return self.logprior(theta) + self.loglike(theta)



def run_mcmc():
	kmin = 0.005
	kmax = 0.25
	ndim = 4
	nwalkers = 30
	nsteps = 10000
	
	# read the ratio of bispectra and ratio of means
	k, br, bkrm = read(bkr_file)
	print(f'k shape: {k.shape}')
	print(f'br shape: {br.shape}')
	print(f'bkrm shape: {bkrm.shape}')

	if is_bk:
		# fill in the 3D matrix
		br_int = Interpolate3D(k, bkrm)
	else:
		br_int = Interpolate1D(k, bkrm)

	kg, bg, icov = select_k(k, kmin, kmax, bkrm, br)

	def model(kg, theta):
		return br_int(theta[0]*kg) + theta[1]/kg + theta[2] + theta[3]*kg

	ps = Posterior(model, bg, icov, kg)
	def logpost(param):
		return ps.logpost(param)
	def nlogpost(param):
		return -1.*ps.logpost(param)

	res = minimize(nlogpost, [0.0, 0.0, 0.0, 0.0], method='Powell')
	print(f"Scipy optimizer: {res}")
	# Initial positions of the walkers.
	start = res.x + 1.0e-2*np.random.randn(nwalkers, ndim)
	print(f'scipy opt: {res}')
	print(f'initial guess: {start[:2]} ... {start[-1]}')

	ncpu = cpu_count()
	print("{0} CPUs".format(ncpu))

	#if len(os.sched_getaffinity(0)) < ncpu:
	#	try:
	#		os.sched_setaffinity(0, range(ncpu))
	#	except OSError:
	#		print('Could not set affinity')
	#
	#n = max(len(os.sched_getaffinity(0)), 96)
	n = 2
	print('Using', n, 'processes for the pool')

	with Pool(n) as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool)
		sampler.run_mcmc(start, nsteps, progress=True)


	savez(mcmc_file, **{'chain':sampler.get_chain(), 
                    'log_prob':sampler.get_log_prob(), 
                    'best_fit':res.x,
                    'best_fit_logprob':res.fun,
                    'best_fit_success':res.success, 
                    '#data':kg.shape,
                    '#params':ndim})


if __name__ == '__main__':
	run_mcmc()
