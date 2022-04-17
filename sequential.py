import numpy as np
from board import Board
from util import fwer
from multiprocessing import Queue
from scipy.stats import bernoulli
import queue, time

def add_registry(cls):
    cls.algorithms = {
            f.__name__:f for f in cls.__dict__.values()
            if hasattr(f,'is_algorithm')
            }
    return cls

@add_registry
class Sequential:
    def __init__(self, board:Board, algorithm:str, c:float, e:float, b:float,
                lgllr:bool=None) -> None:
        self.board = board
        if algorithm in ['NED', 'BLUCB', 'EXFC']:
            if algorithm == 'BLUCB':
                self.b = b
            self.e = e
        if algorithm not in Sequential.algorithms:
            raise NotImplementedError
        self.algorithm = algorithm
        if algorithm in ['NED', 'EXPL']:
            self.c = round(c)
        else:
            self.c = c
        if algorithm in ['DS']:
            if algorithm == 'DS':
                self.llr_thresh = -np.log(c)
            self.lgllr = lgllr
        elif lgllr is not None:
            raise ValueError

        self.run_algorithm = getattr(self,algorithm)

    def register_algorithm(f):
        f.is_algorithm = True
        return f

    def reset(self) -> None:
        self.board.initialize()
        if not hasattr(self,'samples'):
            self.samples = np.zeros((self.board.size, 256))
            self.n = np.zeros(self.board.size, dtype=int)
        else:
            self.samples[:] = 0
            self.n[:] = 0
        self.restart = False
        self.t = 0

    def simulate(self, q:Queue, stop) -> dict:
        run = True
        while not stop.value:
            if run:
                self.reset()
                res = self.run_algorithm()
            try: # solves issues with workers not terminating due to queue
                q.put(res, False)
                run = True
            except queue.Full:
                run = False
                time.sleep(.01)
        stop.value = 0

    def wipe_node_samples(self, nodes:list) -> None:        
        self.samples[nodes] = 0
        self.n[nodes] = 0
    
    @register_algorithm
    def EXPL(self) -> dict:
        if self.c < self.board.size:
            raise ValueError('Budget must be larger or equal to the number of cells.')
        n = self.c//self.board.size
        self.uniform_sampling(self.board.nodes, n)
        sample_from = np.random.choice(self.board.nodes, size=self.c-self.t)
        self.uniform_sampling(sample_from, 1)
        estimate = self.board.estimate(self.samples, self.n)
        sorted = np.argpartition(estimate,-self.board.k)
        return self.return_dict(sorted[-self.board.k:])
    
    @register_algorithm
    def EXFC(self) -> dict:
        self.uniform_sampling(self.board.nodes, 1)
        means, lower, upper = self.board.estimate_and_bounds(self.t, self.c, self.samples, self.n)
        m = 0
        while True:
            self.sample_update(m, 1)
            means[m], lower[m], upper[m] = self.board.estimate_and_bounds(self.t, self.c, self.samples[m], self.n[m])
            sorted = np.argpartition(means,-self.board.k)
            contenders = sorted[-self.board.k:]
            h = contenders[np.argmin(lower[contenders])]
            noncontenders = sorted[:-self.board.k]
            l = noncontenders[np.argmax(upper[noncontenders])]
            if upper[l]-lower[h]<=self.e:
                break
            m = (m+1) % self.board.size
        return self.return_dict(contenders)

    @register_algorithm
    def BLUCB(self) -> dict:
        self.uniform_sampling(self.board.nodes, 1)
        est, lower, upper = self.board.estimate_and_bounds(self.t, self.c, self.samples, 1)
        while True:
            sorted = np.argpartition(est,-self.board.k)
            contenders = sorted[-self.board.k:]
            h = contenders[np.argmin(lower[contenders])]
            noncontenders = sorted[:-self.board.k]
            l = noncontenders[np.argmax(upper[noncontenders])]
            if upper[l]-lower[h]<self.e:
                break
            if self.b < 0:
                uncertain = [h,l]
            else:
                uncertain = [h] if bernoulli.rvs(self.b) else [l]
            self.uniform_sampling(uncertain, 1)
            est[uncertain], lower[uncertain], upper[uncertain] =\
                self.board.estimate_and_bounds(self.t, self.c, self.samples[uncertain], self.n[uncertain])
        return self.return_dict(contenders)

    @register_algorithm
    def NED(self) -> dict:
        if self.c < self.board.size:
            raise ValueError('Budget must be larger or equal to the number of cells.')
        active = self.board.nodes.tolist()
        self.uniform_sampling(active, 1)
        means = self.board.estimate(self.samples, self.n)
        budget = self.c-self.board.size
        old_n = 0
        c = self.board.k/(self.board.k+1)**self.e+\
            sum([x**-self.e for x in range(self.board.k+1,self.board.size+1)])
        for r in range(1,self.board.size-self.board.k+1):
            n = budget/(c*(self.board.size-r+1)**self.e)
            n_sample = self.c-self.t
            if r < self.board.size-self.board.k:
                n_sample = min(round(
                    (self.board.size-r+1)*(n-old_n)
                    )
                    ,n_sample)
            if n_sample > 0:
                n_act = len(active)
                n_uni = n_sample//n_act
                if n_uni>0: # saves a lot of time!
                    self.uniform_sampling(active,n_uni)
                n_random = n_sample-n_act*n_uni
                if n_random > 0: # saves a lot of time!
                    sample_from = np.random.choice(active, size=n_random)
                    self.uniform_sampling(sample_from,1)
                means[active] = self.board.estimate(self.samples[active],self.n[active])
            i = np.argmin(means[active])
            del active[i]
            old_n = n
        return self.return_dict(active)

    @register_algorithm
    def DS(self) -> dict:
        # initial estimate = normal parameter
        est = [self.board.normal_parameter() for _ in range(self.board.size)]
        llr = np.zeros(self.board.size)
        estimated_states = [False for _ in self.board.nodes]
        est_anoms = []
        prioritize_anomaly = True
        if not prioritize_anomaly:
            raise NotImplementedError

        m = 0
        while True:
            if len(est_anoms) == self.board.k:
                first = True
                t=0
                while first or prioritize_anomaly == estimated_states[borderline]:
                    t += 1
                    first = False
                    borderline = est_anoms[np.argmin(llr[est_anoms])]
                    if llr[borderline]>=self.llr_thresh:
                        return self.return_dict(est_anoms)   
                    # phase 2
                    s = self.sample_update(borderline, 1)
                    # phase 3
                    if not self.lgllr: # takes old estimate
                        logp_denominator = self.board.logp('0',s)
                        logp = self.board.logp(est[borderline],s)
                        if logp != logp_denominator: # problems at infinity...
                            llr[borderline] += logp-logp_denominator
                    est[borderline], estimated_states[borderline] = self.board.est_leaf(False, self.current_samples(borderline))
                    if self.lgllr:
                        llr[borderline] = self.board.logp(est[borderline],self.current_samples(borderline))\
                            -self.board.logp('0',self.current_samples(borderline))
                est_anoms.remove(borderline)

            # phase 1
            # for proper estimates
            n_samples = self.board.min_samples-self.n[m]-1
            if n_samples > 0:
                self.sample_update(m, n_samples)

            s = self.sample_update(m, 1)

            if not self.lgllr: # takes old estimate
                logp_denominator = self.board.logp('0',s)
                logp = self.board.logp(est[m],s)
                if logp != logp_denominator: # problems at infinity...
                    llr[m] += logp-logp_denominator
            est[m], new_state = self.board.est_leaf(False, self.current_samples(m))
            if self.lgllr:
                llr[m] = self.board.logp(est[m],self.current_samples(m))\
                    -self.board.logp('0',self.current_samples(m))

            if not estimated_states[m]:
                if new_state:
                    est_anoms.append(m)
            else:
                if not new_state:
                    est_anoms.remove(m)
            estimated_states[m] = new_state
            m = (m+1)%self.board.size
            
    def current_samples(self,node:int) -> np.ndarray:
        return self.samples[node,:self.n[node]]

    def uniform_sampling(self, nodes:list, n_samples:int) -> np.ndarray:
        s = np.zeros((len(nodes), n_samples))
        for i,m in enumerate(nodes):
            s[i] = self.sample_update(m, n_samples)
        return s

    def sample_update(self, node:int, n_samples:int) -> np.ndarray:        
        # extend array if necessary
        while self.n[node]+n_samples>self.samples.shape[1]:
            self.samples = np.hstack((self.samples, np.zeros(self.samples.shape)))
        
        new_samples = self.board.sample(node, n_samples)
        self.samples[node,self.n[node]:self.n[node]+n_samples] = new_samples
        self.n[node] += n_samples
        self.t += n_samples
        return new_samples

    def return_dict(self, anomalies:list) -> dict:
        return {
            'tau': self.t,
            'fwer': fwer(self.board.hiders, anomalies),
            }