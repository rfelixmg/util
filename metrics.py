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
def softmax(x):
    import numpy as np
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy(y_prob,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    from numpy import log, sum
    m = y.shape[0]
    p = y_prob
    log_likelihood = -log(p[range(m),y])
    loss = sum(log_likelihood) / m
    return loss

def normalize(x, ord=1,axis=-1):
    '''
    Normalize is a function that performs unit normalization
    Please, see http://mathworld.wolfram.com/UnitVector.html
    :param x: Vector
    :return: normalized x
    '''
    from numpy import atleast_2d, linalg, float
    return (atleast_2d(x) / atleast_2d(linalg.norm(atleast_2d(x), ord=ord, axis=axis)).T).astype(float)


def scaler(x, rg=(0.,1.), seed=False):
    from numpy import min, max, array
    if seed:
        return array((((x - seed[0]) / (seed[1] - seed[0])) * (rg[1] - rg[0])) + rg[0])
    else:
        return array((((x - min(x)) / (max(x) - min(x))) * (rg[1] - rg[0])) + rg[0])



def accuracy_per_class(predict_label, true_label, classes):
    '''
    
    :param predict_label: output of model 
    :param true_label: labels from dataset
    :param classes: class labels list() 
    :return: 
    '''
    from numpy import sum, float, array
    nclass = len(classes)
    acc_per_class = []
    for i in range(nclass):
        idx = true_label == classes[i]
        if idx.sum() != 0:
            acc_per_class.append(sum(true_label[idx] == predict_label[idx]) / float(idx.sum()))
    if len(acc_per_class) == 0:
        return 0.
    return array(acc_per_class).mean()


def h_mean(a, b):
    from numpy import float
    return 2*a*b/float(a+b)


def stats(x):
    import numpy as np
    return {'max': np.max(x),
            'min': np.min(x),
            'mean': np.mean(x),
            'var': np.var(x),
            }


def inner_product(x, y):
    from numpy import newaxis
    return (x[:, newaxis] * y).sum(2)


def inner_prediction(y_pred, y_true, _score=False):
    """
    
    :param y_pred: prediction
    :param y_true: class dictionary 
    :param _score: return inner product values?
    :return:  y_pred: argmax of scores range([0, y_true.shape[0]])
    """
    from numpy import array
    y_score = inner_product(y_pred, y_true)
    y_pred = y_score.argmax(axis=1)
    score = array([y_score[key, id] for key, id in enumerate(y_pred)])
    if _score:
        return y_pred, score, y_score
    else:
        return y_pred


def euclidean_distance(x, y):
    from numpy import newaxis, sum, sqrt
    return sqrt(((x[:, newaxis] - y) ** 2).sum(2))

def euclidean_prediction(y_pred, y_true, _score=False):
    """

    :param y_pred: prediction
    :param y_true: class dictionary 
    :param _score: return inner product values?
    :return:  y_pred: argmin of scores range([0, y_true.shape[0]])
    """
    from numpy import array
    y_score = euclidean_distance(y_pred, y_true)
    y_pred = y_score.argmin(axis=1)
    score = array([y_score[key, id] for key, id in enumerate(y_pred)])
    if _score:
        return y_pred, score, y_score
    else:
        return y_pred


def tf_pairwise_euclidean_distance(x, y):
    import tensorflow as tf
    size_x = tf.shape(x)[0]
    size_y = tf.shape(y)[0]
    xx = tf.expand_dims(x, -1)
    xx = tf.tile(xx, tf.stack([1, 1, size_y]))

    yy = tf.expand_dims(y, -1)
    yy = tf.tile(yy, tf.stack([1, 1, size_x]))
    yy = tf.transpose(yy, perm=[2, 1, 0])

    diff = xx - yy
    square_diff = tf.square(diff)

    square_dist = tf.reduce_sum(square_diff, 1)

    return square_dist


def entropy(x, axis=1):
    from numpy import sum, log2
    return -sum(x* log2(x), axis)

def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-x))


def get_tau(x, percentile):
    return (x.max()- x.min()) * percentile

def intersection(a,b):
    from numpy import array
    ans = array([x for x in set(a).intersection(b)])
    return ans



def histogram(x, bins=100, npoints=500, smooth=False):
    from numpy import histogram, r_, linspace, sort, ones
    from scipy.interpolate import interp1d

    nsamples = x.shape[0]

    if(nsamples > bins):
        _qt, _var = histogram(x, bins)
        _qt = r_[0., _qt, 0.]
        _qt = (_qt / nsamples) * 100
        _var = r_[_var[0], _var[1:], _var[-1]]

        if smooth:
            x, y = smoother(_var, _qt, smooth, npoints)
            # xnew = linspace(_var.min(), _var.max(), _var.shape[0])
            # smoother = interp1d(xnew, _qt, kind=smooth)
            # x = linspace(_var.min(), _var.max(), npoints)
            # y = smoother(x)
        else:
            x = _var
            y = _qt
    else:
        sort(x)
        y = ones(x.shape[0])


    return x, y


def smoother(_x, _y, smooth='cubic', npoints=500):
    from scipy.interpolate import interp1d
    from numpy import linspace

    xnew = linspace(_x.min(), _x.max(), _x.shape[0])
    smoother_func = interp1d(xnew, _y, kind=smooth)
    x = linspace(_x.min(), _x.max(), npoints)
    y = smoother_func(x)
    return x, y