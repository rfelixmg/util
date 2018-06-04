def merge_array(x, y, axis=0):
    import numpy as np
    if np.shape(x)[0]:
        if np.shape(y)[0]:
            return np.concatenate([x,y], axis)
        else:
            return x
    elif np.shape(y)[0]:
        return y
    else:
        return np.array([])


def normalize(x, ord=1,axis=-1):
    '''
    Normalize is a function that performs unit normalization
    Please, see http://mathworld.wolfram.com/UnitVector.html
    :param x: Vector
    :return: normalized x
    '''
    from numpy import atleast_2d, linalg, float
    return (atleast_2d(x) / atleast_2d(linalg.norm(atleast_2d(x), ord=ord, axis=axis)).T).astype(float)