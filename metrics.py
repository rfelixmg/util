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