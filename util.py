import numpy as np
import re, os
from itertools import zip_longest
from sys import stdout
from scipy.stats import chi2

def save_1dim_arrays(d, filename, delimiter=',',
        header=True) -> None:
    if isinstance(d, tuple):
        d = {i:v for i,v in enumerate(d)}
        header = False
    with open(filename, 'w') as out:
        if header:
            out.write(f'{delimiter.join(d.keys())}\n')
        for row in zip_longest(*d.values(), fillvalue=''):
            out.write(f'{join_cast(delimiter, row)}\n')


def save_scalars(d, filename, delimiter=',',
        header=True) -> None:
    if isinstance(d, tuple):
        d = {i:v for i,v in enumerate(d)}
        header = False
    with open(filename, 'w') as out:
        if header:
            out.write(f'{delimiter.join(d.keys())}\n')
        out.write(f'{join_cast(delimiter, d.values())}\n')

def iprint(*args,**kwargs) -> None:
    print(*args,**kwargs)
    stdout.flush()

def save_float_array(array:np.ndarray, filename:str, delimiter=',') -> np.ndarray:
    with open(filename, 'w') as f:
        for row in array:
            f.write(delimiter.join([str(r) for r in row])+'\n')

def load_float_array(filename:str, delimiter=',') -> np.ndarray:
    d = load_1dim_arrays(filename, delimiter)
    return np.array([col for col in d.values()]).transpose()

def load_1dim_arrays(filename:str, delimiter=',') -> dict:
    d = {}
    with open(filename, 'r') as load:
        header = [a.rstrip() for a in next(load).split(delimiter)]
        hlen = len(header)
        d = {h:[] for h in header}
        for line in load:
            for i, e in enumerate(line.split(delimiter)):
                if i >= hlen:
                    raise IOError('Line had more entries than header.')
                d[header[i]] += [float(e)]
    for k,v in d.items():
        d[k] = np.array(v)
    return d

def load_1dim_lists(filename:str, delimiter=',') -> dict:
    return { k:v.tolist() for k,v in load_1dim_arrays(filename, delimiter).items()}

def join_cast(delimiter:str, iter) -> str:
    iter = [str(a) for a in iter]
    return delimiter.join(iter)

def proper_int_cast(s) -> int:
    return round(float(s))

def ceil(val:float) -> int:
    return round(np.ceil(val))

def ceil_next(val:float) -> int:
    return ceil(np.nextafter(val,1))

def get_value_from_filename(filename:str, name:str):
    pattern = re.compile(f'{name}_(?P<num>\d*(\.\d+)?)')
    try:
        return float(pattern.search(os.path.basename(filename)).group('num'))
    except AttributeError:
        return None

def sample_entropy(samples:np.ndarray) -> float:
    if samples.ndim != 1:
        raise NotImplementedError
    distinct = np.unique(samples)
    return -np.sum([q*np.log(q) for q in [np.count_nonzero(samples==d)/samples.shape[0]
                 for d in distinct]])

def permute_2nd_column(timeseries:np.ndarray) -> np.ndarray:
    if len(timeseries.shape)!=2 or timeseries.shape[1]!=2:
        raise NotImplementedError
    return np.column_stack((timeseries[:,0],np.random.permutation(timeseries[:,1])))

def fdr(true:list, predicted:list):
    declared = len(predicted)
    if declared == 0:
        return 0
    hits = 0
    for p in predicted:
        if p in true:
            hits += 1
    return float(1-hits/declared)

def fwer(true:list, predicted:list):
    declared = len(predicted)
    if declared == 0:
        raise NotImplementedError
    true = true.copy()
    for p in predicted:
        try:
            i = true.index(p)
        except ValueError:
            return 1
        del true[i]
    return float(len(true) > 0)

def matches(prefix:str, f:str, metric:str) -> bool:
    # folder not inclided -> only match string end
    pattern = f'{prefix}sims_\d+'
    pattern += f'_{metric}'
    pattern += '.txt$'
    pattern = re.compile(pattern)
    return True if pattern.search(f) is not None else False

def equal(l, dict_keys:list=None, f:callable=lambda x: x) -> bool:
    if dict_keys is not None:
        l = [l[key] for key in dict_keys]
    val = f(l[0])
    for v in l[1:]:
        if f(v) != val:
            return False
    return True

def k_highest(array: np.ndarray, k: int, f=lambda x: x) -> tuple:
    if k <=0 or k > array.shape[0]:
        raise Exception(f'k is {k} but should be between 0 and {array.shape[0]-1}.')
    # only used to compare, not returned!
    compare = f(array)
    indices = np.argpartition(compare,-k)[-k:]
    return array[indices], indices

def exp_mean_cdf(x:float, lam:float, n:int) -> float:
    return np.exp(exp_mean_logcdf(x,lam,n))

def exp_mean_logcdf(x:float, lam:float, n:int) -> float:
    # log cdf of the mean of n independent exponentially distributed
    # random variables with rate lam
    return chi2.logcdf(2*n*lam*x,2*n)

def exp_mean_ppf(x:float, lam:float, n:int) -> float:
    # ppf of the mean of n independent exponentially distributed
    # random variables with rate lam
    return chi2.ppf(x,2*n)/(2*n*lam)

def k_lowest(array: np.ndarray, k: int, f=lambda x: x) -> tuple:
    return k_highest(array, k, lambda x: -f(x))

def kth_highest(array: np.ndarray, k: int, f=lambda x: x) -> tuple:
    values, indices = k_highest(array, k, f)
    return values[0], indices[0]

def kth_lowest(array: np.ndarray, k: int, f=lambda x: x) -> tuple:
    return kth_highest(array, k, lambda x: -f(x))

def highest_(array: np.ndarray, f=lambda x: x) -> tuple:
    return kth_highest(array, 1, f)

def highest(array: np.ndarray, f=lambda x: x) -> tuple:
    if not isinstance(f, list):
        return highest_(array, f)
    values, indices = [], []
    for func in f:
        value, index = highest_(array, func)
        values += [value]
        indices += [index]
    return values, indices

def lowest(array:np.ndarray, f=lambda x: x) -> tuple:
    return highest(array, lambda x: -f(x))

def remove_indices_from_list(l:list, indices:list,
                                inplace=False) -> list:
    if not inplace:
        l = l.copy()
    for i in sorted(indices, reverse=True):
        del l[i]
    return l
