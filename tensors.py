"""
MIT License

Copyright (c) 2018 Rafael Felix Alves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
def merge_array(x, y, axis=0):
    from numpy import size,atleast_1d, concatenate
    if not size(x):
        return atleast_1d(y)
    elif size(x) and size(y):
        return concatenate([x, atleast_1d(y)], axis)
    elif size(y):
        return atleast_1d(y)
    else:
        return atleast_1d([])

def merge_dict(x, y):
    ans = {}
    for _key in (x.keys() & y.keys()):
        if isinstance(x[_key], dict):
            ans[_key] = merge_dict(x[_key], y[_key])
        else:
            ans[_key] = merge_array(x[_key], y[_key])

    return ans


def normalize(x, ord=1,axis=-1):
    '''
    Normalize is a function that performs unit normalization
    Please, see http://mathworld.wolfram.com/UnitVector.html
    :param x: Vector
    :return: normalized x
    '''
    from numpy import atleast_2d, linalg, float
    return (atleast_2d(x) / atleast_2d(linalg.norm(atleast_2d(x), ord=ord, axis=axis)).T).astype(float)


import numpy as np
import json
class NpJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)