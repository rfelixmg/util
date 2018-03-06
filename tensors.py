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
