# statistic.py

import numpy as np

def stat(arr, name=''):
    '''Print statistic of given array

    Parameters
    ----------
    arr : list
        list of number for statistic

    '''
    arr = np.array(arr)
    print((name, arr.shape, 'min:', np.min(arr), 'max:', np.max(arr),
           'std:', np.std(arr), 'mean:', np.mean(arr), 'median:', np.median(arr)))
