from scipy.stats import norm, expon, binom, bernoulli, ncx2, chi2
import numpy as np
import random
from util import ceil_next, exp_mean_cdf, exp_mean_ppf, sample_entropy, permute_2nd_column
from util import load_float_array, save_1dim_arrays, load_1dim_lists, exp_mean_logcdf

class Board:
    dist_parametertypes = {
        'EXP' : ['lambda0', 'lambda1','min_lambda1'],
        }
    dist_min_samples = {
        'EXP' : 1,
        }
    dists = list(dist_parametertypes.keys()) 
    dist_parameters = [p for l in dist_parametertypes.values() for p in l]

    def __init__(self, dist: str, k:int, m: int, dist_parameters:dict) -> None:
        
        self.size = m
        self.k = k
        self.nodes = np.arange(m)
        self.min_samples = Board.dist_min_samples[dist]
        self.dist = dist
        self.__dict__.update(dist_parameters)
        if dist == 'EXP':
            self.lambda_thresh_leaf = [(l1-self.lambda0)/np.log(l1/self.lambda0)\
                for l1 in [self.min_lambda1,self.lambda1]]
        else:
            raise ValueError
                
        if self.size <= 0:
            raise ValueError('Board size must be greater than 0.')

        if k >= self.size:
            raise ValueError('Number of anomalies must be lower than the board size.')
        self.initialize()


    def estimate(self, samples:np.ndarray, n:np.ndarray) -> tuple:
            if self.dist == 'EXP':
                # just negative mean, there is a constant +1 in the paper which the algorithms
                # are invariant to
                est = -np.sum(samples, axis=-1)/n
            else:
                raise NotImplementedError
            return est

    def estimate_and_bounds(self, t:int, alpha:float, samples:np.ndarray, n:np.ndarray) -> tuple:
        # Different definition as in the paper, but the size of the confidence bound is still monotone in alpha.
        # Since we do not mention explicit values for alpha, this definition works just as well for the
        # error rate vs. sample complexity plot.
        alpha = 1-(1-alpha)**(1/self.size)
        if self.dist == 'EXP':
            est = self.estimate(samples,n)
            upper = est*2*n/chi2.ppf(1-alpha, 2*n)
            lower = est*2*n/chi2.ppf(alpha, 2*n)
            if np.any(lower>est) or np.any(upper<est):
                raise Exception
        else:
            raise NotImplementedError
        
        return est, lower, upper

    def initialize(self) -> None:
        self.removed = []
        self.hiders = random.sample(range(self.size),self.k)
        self.hider_mask = np.zeros(self.size, int)
        self.hider_mask[self.hiders] = 1      

    def check_one_anomaly(self) -> None:
        if self.k != 1:
            raise NotImplementedError('Currently only support one anomaly.')


    def est_leaf(self, knows_anom:bool, samples:np.ndarray) -> tuple:

        if isinstance(samples, (float,int)):
            n = 1
        elif samples.ndim == 1:
            n = samples.shape[0]
        else:
            raise ValueError

        mean = np.sum(samples, axis=-1)/n
        if self.dist == 'EXP':
            lam = 1/mean
            return lam, lam>self.lambda_thresh_leaf[knows_anom]
        else:
            raise NotImplementedError

    np.seterr(all='raise')
    def est1_and_stat(self, samples:np.ndarray, n:np.ndarray=None) -> tuple:
        if n is None:
            n = samples.shape[0]

        mean = np.sum(samples, axis=-1)/n
        if self.dist == 'EXP':
            return np.maximum(1/mean, self.min_lambda1), mean
        else:
            raise NotImplementedError

    def normal_parameter(self) -> float:
        if self.dist == 'EXP':
            return self.lambda0
        else:
            raise NotImplementedError
    
    def abnormal_parameter(self, n_anomalies:int) -> float:

        if self.dist == 'EXP':
            return (1-n_anomalies)*self.lambda0+n_anomalies*self.lambda1
        else:
            raise NotImplementedError

    def logp(self, parameter, samples:np.ndarray, n:np.ndarray=None) -> float:

        if n is None:
            if isinstance(samples, (float,int)):
                n = 1
            elif samples.ndim == 1:
                n = samples.shape[0]
            else:
                raise ValueError

        stat = None
        if isinstance(parameter, str):
            if parameter == '0':
                parameter = self.normal_parameter()
            elif parameter == '1':
                parameter = self.abnormal_parameter(1)
            elif parameter == 'est1':
                parameter, stat = self.est1_and_stat(samples, n)
            else:
                raise NotImplementedError
            
        if self.dist == 'EXP':
            mean = np.sum(samples, axis=-1)/n if stat is None else stat
            return n*(np.log(parameter)-parameter*mean)
        else:
            raise NotImplementedError

    def sample(self, node:int, n_samples:int=1) -> np.ndarray:
        if n_samples == 0: # can save a lot of time!
            return np.array([])
        n_anomalies = np.sum(self.hider_mask[node])
        if self.dist == 'EXP':
            if n_anomalies:
                lam = self.abnormal_parameter(n_anomalies)
            else:
                lam = self.normal_parameter()
            return expon.rvs(scale=1/lam,size=n_samples)
        else:
            raise ValueError
