from multiprocessing import Process, Queue, Value
import os, time
from experiment import Experiment
from util import save_1dim_arrays
import copy, numpy as np
from arghierarchy import ArgHierarchy
from util import iprint

class ExperimentGroup:

    def __init__(self, base_parameters, settings) -> None:
        self.base_parameters = base_parameters
        self.base_parameters.update(settings)
        self.datafolder = settings['datafolder']
        self.settingkeys = list(settings.keys())
        self.n_experiments = 0
        self.experiments = []
    
    def add_experiment(self, arg_hierarchy:ArgHierarchy) -> None:
        params = copy.deepcopy(arg_hierarchy.dictionary)
        new_parameters = self.base_parameters.copy()
        new_parameters.update(params)
        if self.n_experiments == 0:
            self.base_parameters = new_parameters.copy()
        for p in params:
            if p in self.base_parameters and self.base_parameters[p]!=params[p]:
                del self.base_parameters[p]
        new_parameters['index'] = self.n_experiments
        self.experiments += [Experiment(**new_parameters)]
        self.n_experiments += 1

    def save(self) -> None:
        for e in self.experiments:
            e.save()

    def run(self) -> None:
        pr, qin, qout, n_sim = [], [], [], []
        for e in self.experiments:
            qin += [Queue()]
            qout += [Queue()]
            n_sim += [Value('i', 0)]
            pr += [Process(target=e.simulate, args=(qout[-1], qin[-1], n_sim[-1]))]
            pr[-1].start()

        # give terminated processes to slower experiments
        while self.n_experiments > 0:
            time.sleep(1)
            for from_i, q in enumerate(qin):
                if not q.empty():
                    del qin[from_i]
                    # clear queue
                    leftover = 0
                    while not qout[from_i].empty():
                        leftover += qout[from_i].get()
                    del qout[from_i]
                    del n_sim[from_i]
                    sim = [s.value for s in n_sim]
                    if sim: # if not last experiment running
                        to_i = np.argmin(sim)
                        qout[to_i].put(q.get()+leftover)
                    self.n_experiments -= 1
                    iprint('ex',self.n_experiments)
                    break
        for p in pr:
            p.join()
        print('Done')

    def sim(self) -> int:
        sims = []
        for e in self.experiments:
            e.initialize_sim_and_metrics(True)
            sims += [e.sim]
        index = np.argmin(sims)
        return sims[index], index