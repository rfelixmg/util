def load(root, dtype='h5py'):
    """
    load dataset: load dataset and respective knn features
    :param root: directory for datasets
    :param dtype: default:h5py
    :return: dataset, knn 
    """

    if dtype == 'h5py':
        from util.storage import DataH5py, Container
        dataset = DataH5py().load_dict_from_hdf5('{}/data.h5'.format(root))
        knn = DataH5py().load_dict_from_hdf5('{}/knn.h5'.format(root))

    dataset, knn = Container(dataset), Container(knn)
    dataset.n_classes = knn.openset.ids.shape[0]
    dataset.root = '{}/data.h5'.format(root)
    knn.root = '{}/knn.h5'.format(root)

    return dataset, knn


def normalize(x, ord=1,axis=-1):
    '''
    Normalize is a function that performs unit normalization
    Please, see http://mathworld.wolfram.com/UnitVector.html
    :param x: Vector
    :return: normalized x
    '''
    from numpy import atleast_2d, linalg, float
    return (atleast_2d(x) / atleast_2d(linalg.norm(atleast_2d(x), ord=ord, axis=axis)).T).astype(float)