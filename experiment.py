import time, os
from sequential import Sequential
from board import Board
from util import get_value_from_filename, save_scalars
from util import load_1dim_arrays, iprint
from contextlib import contextmanager
from multiprocessing import Process, Queue, Value

class Experiment:

    @contextmanager
    def add_attribute_names_to_list(self, l:list) -> None:
        self.add_names_to = l
        yield
        self.add_names_to = None
    
    def __setattr__(self, __name: str, __value) -> None:
        if __name not in ['add_names_to']:
            if hasattr(self, 'add_names_to') and self.add_names_to is not None\
                and __value is not None:
                self.add_names_to.append(__name)
        self.__dict__[__name] = __value

    def __init__(self, max_sim:int, load_previous:bool,
            m:int, k:int, algorithm:str,
            dist:str, nosave:bool, dist_parameters:dict,
            c:float, bias:float, lgllr:bool,
            processes:int, verbose:bool, e:float,
            datafolder:str, index:int,
            notification_interval, save_interval,
            ) -> None:

        # parameters that appear in filename WITHOUT parameters name
        self.words = []
        with self.add_attribute_names_to_list(self.words):
            self.algorithm = algorithm
            if algorithm in ['DS']:
                self.local_test_statistic = 'LGLLR' if lgllr else 'LALLR'
            self.dist = dist

        # parameters that appear in filename WITH parameters name
        self.numbers = []
        with self.add_attribute_names_to_list(self.numbers):
            for pt in Board.dist_parametertypes[dist]:
                self.__setattr__(pt, dist_parameters[pt])
            self.m = m
            if algorithm in ['NED', 'BLUCB', 'EXFC']:
                self.e = e
                if algorithm == 'BLUCB':
                    self.bias = bias
            if algorithm == 'NED':
                self.c = round(c) # round budget
            else:
                self.c = c
            self.k = k
        
        # settings that do not appear in filename
        self.index = index
        self.verbose = verbose
        self.processes = processes
        self.max_sim = max_sim
        # board with equal parameters on all channels
        self.board = Board(
            dist=dist,
            k=self.k,
            m=m,
            dist_parameters=
            {k:dist_parameters[k] for k in Board.dist_parametertypes[dist]}
            )
        seq_args = {
                'board' : self.board,
                'algorithm' : algorithm,
                'c' : self.c,
                'e' : e,
                'b' : bias,
            }
        if self.algorithm in ['DS']:
            seq_args['lgllr'] = lgllr
        self.sequential = Sequential(**seq_args)
        self.datafolder = datafolder
        self.notification_interval = notification_interval
        self.nosave = nosave
        self.state = None
        self.save_interval = save_interval
        self.process_list = []
        self.stop_list = []
        self.queue = Queue(1000)
        self.initialize_sim_and_metrics(load_previous)

    def start_processes(self, n_proc:int) -> None:
        print(f'starting {n_proc} processes in experiment {self.index}')
        for _ in range(n_proc):
            stop = Value('i', 0)
            p = Process(target=self.sequential.simulate, args=(self.queue,stop))
            p.start()
            self.process_list += [p]
            self.stop_list += [stop]
        self.processes = len(self.process_list)
    
    def stop_all_processes(self) -> None:
        for stop in self.stop_list:
            stop.value = 1 # tell to stop
        while any([stop.value for stop in self.stop_list]):
            time.sleep(1) # wait for all processes to stop
            iprint('before',self.queue.qsize())
        iprint('after')
        while not self.queue.empty():
            self.queue.get() # empty queue so that processes can terminate
        for p in self.process_list:
            p.join()
        

    def simulate(self, qin:Queue, qout:Queue, n_sim:Value) -> int:
        total = time.time()
        start_n = total
        start_s = total
        self.start_processes(self.processes)
        try:
            while self.max_sim==-1 or self.sim < self.max_sim:
                if not qin.empty():
                    self.start_processes(qin.get())
                metrics = self.queue.get()
                for k,new_metric in metrics.items():
                    if self.sim == 0:
                        self.metrics[k] = new_metric
                    else:
                        self.metrics[k] = (self.sim*self.metrics[k]+new_metric)/(self.sim + 1)
                self.sim += 1
                n_sim.value = self.sim
                
                if self.verbose:
                    iprint(self.sim, self.metrics)
                t = time.time()
                if not self.nosave and (t-start_s>=self.save_interval or self.sim==self.max_sim):
                    self.save()
                    start_s = t
                if  t-start_n >= self.notification_interval:
                    iprint(f'Simulation step {self.sim} for experiment {self.index} done.')
                    start_n = t
        except KeyboardInterrupt:
            if not self.nosave:
                self.save()
        iprint(f'Experiment {self.index} arrived at {self.sim}'+
            f' simulations after {(time.time()-total)/60} minutes')
        self.stop_all_processes()
        iprint(self.index, 'stopped_processes')
        qout.put(self.processes) 


    def initialize_sim_and_metrics(self, load_previous:bool):
        self.partial_filename = self.get_partial_filename()
        self.metrics = {}
        self.sim = 0
        if load_previous:
            try:
                self.load_metrics_and_sim()
            except FileNotFoundError as e:
                print(f'Nothing to load..\n {e}')
        else:
            print('Not loading...')


    def path(self, filename:str) -> str:
        return os.path.join(self.datafolder, filename)

    def get_partial_filename(self) -> str:
        partial_filename = ''
        for w in self.words:
            string = f'{getattr(self,w)}_'
            partial_filename += string
        for n in self.numbers:
            string = f'{n}_{getattr(self,n)}_'
            partial_filename += string
        return partial_filename

    def load_metrics_and_sim(self) -> tuple:
        files = self.matchfiles(self.partial_filename)
        if not files:
            raise FileNotFoundError(f'Found no filename containing \
                                        {self.partial_filename}')
        self.sim = -1
        for f in files:
            s = round(get_value_from_filename(f, 'sims'))
            if s > self.sim:
                self.sim = s
                file = f
        metrics = load_1dim_arrays(file)
        for k,v in metrics.items():
            if k == 'fdr':
                k = 'fwer'
            if v.shape[0]!=1:
                raise ValueError('Only supports scalar metrics.')
            self.metrics[k]=v[0]


    def matchfiles(self, partial_filename) -> str:
        files = []
        for f in os.listdir(self.datafolder):
            if partial_filename in f:
                files += [self.path(f)]
        return files

    def save(self) -> None:
        if self.sim == 0:
            return
        oldfiles = self.matchfiles(self.partial_filename)
        newfile = self.path(self.partial_filename)\
                    +f'sims_{self.sim}.txt'
        save_scalars(self.metrics, newfile)
        for f in oldfiles:
            if f not in newfile:
                os.remove(f)