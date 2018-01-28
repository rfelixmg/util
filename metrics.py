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
