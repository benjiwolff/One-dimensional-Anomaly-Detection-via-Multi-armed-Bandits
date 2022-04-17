from experiment_group import ExperimentGroup
import numpy as np, argparse, os
from board import Board
from arghierarchy import ArgHierarchy
from sequential import Sequential

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # required/positional arguments
    presets = ['NED0.8','NED1.2','EXPLit','EXFCit','BLUCB0.9','BLUCB0.5','LUCB','DSit']
    parser.add_argument('algorithm', type=str,
                            help='algorithm',
                            choices=list(Sequential.algorithms.keys())+presets
                            )

    # problem parameters
    parser.add_argument('-d', '--dist', type=str, default='EXP',
                            help='dist',
                            choices=Board.dists)
    parser.add_argument('-m','--n_processes', type=int, default=None)
    parser.add_argument('-i','--iterate', action='store_true',
                    help='iterate over different values for c')
    parser.add_argument('-ap','--anomaly_parameter', type=float, default=None,
                    help='anomaly parameter index, higher index -> harder problem')
    parser.add_argument('-min_ap','--min_anomaly_parameter', type=float, default=None,
                    help='minium anomaly parameter index, higher index -> harder problem')
    parser.add_argument('-k', '--n_anom', type=int, default=1,
                    help='number of anomalies')
    parser.add_argument('-lgllr','--lgllr', action='store_true',
                            help='use LGRLL statistics instead of LALLR (DS only)')

    # algorithm arguments
    parser.add_argument('-c','--sampling_cost', type=float, default=1e-2,
                    help='relative cost of sampling for DS, relative confidence of LUCB and budget of NED')   
    parser.add_argument('-b','--bias', type=float, default=.5,
                    help='bias of BLUCB algorithm, -1 samples from both uncertain processes')   
    parser.add_argument('-e','--e', type=float,
                    help='stopping epsilon for BLUCB and exponent for NED')
                    
    # simulation settings
    parser.add_argument('-p', '--multi_processing', type=int, default=1,
                    help='processes per experiment for parallel simulations')
    parser.add_argument('-v', '--verbose', action='store_true',
                    help='print to terminal')
    parser.add_argument('-noload', '--noload', action='store_true',
                    help='does not load previous simulations')
    parser.add_argument('-folder', '--folder', type=str, default=None,
                    help='relative data folder location')
    parser.add_argument('-nosave','--nosave', action='store_true')
    parser.add_argument('-max_sim','--max_sim', type=float, default=-1,
                            help='maximum number of simulations')
    parser.add_argument('-nint','--notification_interval', type=int, default=60,
                            help='notification interval in seconds')
    parser.add_argument('-sint','--save_interval', type=int, default=60,
                            help='save interval in seconds')

    args = parser.parse_args()

    if args.algorithm in presets:
        args.n_processes = 100
        args.n_anom = 5
        args.anomaly_parameter = 10
        args.iterate = True
        if args.algorithm == 'NED0.8':
            args.algorithm = 'NED'
            args.e = 0.8
        elif args.algorithm == 'NED1.2':
            args.algorithm = 'NED'
            args.e = 1.2
        elif args.algorithm == 'EXPLit':
            args.algorithm = 'EXPL'
        elif args.algorithm == 'EXFCit':
            args.algorithm = 'EXFC'
        elif args.algorithm == 'BLUCB0.9':
            args.bias = 0.9
        elif args.algorithm == 'BLUCB0.5':
            args.bias = 0.5
        elif args.algorithm == 'LUCB':
            args.bias = -1
        elif args.algorithm == 'DSit':
            args.algorithm = 'DS'
        if 'LUCB' in args.algorithm:
            args.algorithm = 'BLUCB'
        if args.algorithm in ['EXFC','BLUCB']:
            args.e = 0.818


    base_parameters = {
        'e' : args.e,
        'k' : args.n_anom,
        'algorithm' : args.algorithm,
        'lgllr' : args.lgllr,
        'bias' : args.bias,
    }

    if args.save_interval <= 0: # only saves when simulation is stopped
        args.save_interval = np.inf

    # folder structure
    folder = 'data'
    if args.folder is not None:
        folder = os.path.join(folder,args.folder)
    else:
        folder = os.path.join(folder,args.algorithm)
        if args.algorithm in ['DS']:
            if args.lgllr:
                folder = os.path.join(folder,'lgllr')
            else:
                folder = os.path.join(folder,'lallr')
    if not os.path.isdir(folder) and not args.nosave:
        try:
            os.makedirs(folder)
        except:
            pass
        
    settings = {
        'datafolder' : folder,
        'max_sim' : int(args.max_sim),
        'load_previous' : not args.noload,
        'notification_interval' : args.notification_interval,
        'save_interval' : args.save_interval,
        'nosave' : args.nosave,
        'processes' : args.multi_processing,
        'verbose' : args.verbose,
    }

    experiments = ExperimentGroup(base_parameters, settings)
    
    params = {}
            

    iterate_over = {}
    params = ArgHierarchy(args.algorithm)
            
    params.dist = args.dist
    if args.dist == 'EXP':
        pname = 'lambda1'
        params.lambda0 = 1
    else:
        raise NotImplementedError
    if args.anomaly_parameter is not None:
        params[pname] = args.anomaly_parameter
    else:
        raise NotImplementedError
    min_pname = 'min_'+pname
    if args.min_anomaly_parameter is not None:
        params[min_pname] = args.min_anomaly_parameter
    else:
        if args.dist == 'EXP':
            params[min_pname] = (params[pname]+params.lambda0)/2
        else:
            raise NotImplementedError
    iterate_over['k'] = 2**np.arange(1,5)
    if args.algorithm == 'BLUCB':
        iterate_over['c'] = 10**-np.arange(1,6,.5)
    if args.algorithm in ['NED','DSFB']:
        iterate_over['c'] = 200+50*np.arange(8)
    if args.algorithm == 'DS':
        iterate_over['c'] = 10**-np.arange(2,6,.5)
    if args.algorithm == 'EXPL':
        iterate_over['c'] = 200+100*np.arange(8)
    if args.algorithm == 'EXFC':
        iterate_over['c'] = 10**-np.arange(.5,4.5,.5)
    params.m = args.n_processes
    params.c = args.sampling_cost

    if args.iterate:
        for i,p in enumerate(iterate_over['c']):
            params['c'] = p
            print(params.dictionary)
            experiments.add_experiment(params)
    else:
        experiments.add_experiment(params)

    # commands
    experiments.run()