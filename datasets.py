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
class DataH5Loader:
    def __init__(self, root, shuffle=True):
        self.root = root
        self.shuffle = shuffle
        self.setup()

    def add_root(self, root):
        import numpy as np
        self.filepath = np.r_[self.filepath, [root]]
        self.filename = np.r_[self.filename, [root.split('/')[-1]]]
        self.nfiles = len(self.filepath)
        if self.shuffle:
            self.__randomize__()

    def add_path(self, root):
        import numpy as np
        self.filepath = np.r_[self.filepath, [root]]
        self.filename = np.r_[self.filename, [root.split('/')[-1]]]
        self.nfiles = len(self.filepath)
        if self.shuffle:
            self.__randomize__()

    def __randomize__(self):
        from numpy import arange, random
        rids = arange(0, self.nfiles)
        random.shuffle(rids)
        self.filepath = self.filepath[rids]
        self.filename = self.filename[rids]

    def setup(self):
        from util.files import list_files
        self.filepath = list_files(self.root, True, 'h5')
        self.filename = list_files(self.root, False, 'h5')
        self.nfiles = len(self.filepath)
        _ = self.sample()
        if self.shuffle:
            self.__randomize__()


    def load(self, id):
        obj = load_h5(self.filepath[id])
        obj.filename = self.filename[id]
        return obj

    def sample(self):
        from numpy import random
        _data = self.load(random.randint(0, self.nfiles))
        self.x_shape = _data.train.X.shape
        return _data

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError("CustomRange index out of range")
        return self.load(item)

    def size(self):
        return self.__len__()

    def __len__(self):
        return self.nfiles

# cr = CustomRange(0, 10)
# for i in cr:
#     print(i)


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
    if 'n_classes' not in dataset.__dict__:
        dataset.n_classes = knn.openset.ids.shape[0]

    dataset.root = '{}/data.h5'.format(root)
    knn.root = '{}/knn.h5'.format(root)

    return dataset, knn


def load_imagenet(root, benchmark=False):
    """
    load dataset: load dataset and respective knn features for ImageNet
    :param root: directory for datasets
    :param dtype: default:h5py
    :return: dataset, knn
    """

    from util.storage import DataH5py, Container
    dataset = DataH5Loader('{}/datapoints/'.format(root))
    knn = load_h5('{}/knn.h5'.format(root))

    if benchmark is 'zsl':
        dataset.n_classes = knn.zsl.ids.shape[0]
        # dataset.add_root(root_extra)
    elif benchmark:
        dataset.n_classes = knn.openset.ids.shape[0]
    else:
        dataset.n_classes = knn.train.ids.shape[0]

    dataset.a_shape = knn.train.data.shape
    dataset.root = '{}/datapoints/'.format(root)
    knn.root = '{}/knn.h5'.format(root)

    dataset.val = load_h5('{}/val.h5'.format(root)).val
    dataset.test = Container({'seen': load_h5('{}/test.seen.h5'.format(root)).test,
                              'unseen': load_h5('{}/test.unseen.h5'.format(root)).test})

    return dataset, knn


def load_h5(root):
    from util.storage import DataH5py, Container
    data = DataH5py().load_dict_from_hdf5(root)

    return Container(data)

def normalize(x, ord=1,axis=-1):
    '''
    Normalize is a function that performs unit normalization
    Please, see http://mathworld.wolfram.com/UnitVector.html
    :param x: Vector
    :return: normalized x
    '''
    from numpy import atleast_2d, linalg, float
    return (atleast_2d(x) / atleast_2d(linalg.norm(atleast_2d(x), ord=ord, axis=axis)).T).astype(float)

def join_datasets(dataset, datafake, val_split=0.):
    import numpy as np
    from util.storage import Container
    from sklearn.cross_validation import train_test_split
    import copy

    XS_train, XS_val, ys_train, ys_val = train_test_split(dataset.train.X,
                                                          dataset.train.Y,
                                                          test_size=val_split,
                                                          random_state=42)
    XU_train, XU_val, yu_train, yu_val = train_test_split(datafake.X,
                                                          datafake.Y,
                                                          test_size=val_split,
                                                          random_state=42)
    os_train = np.ones(XS_train.shape[0])
    ou_train = np.ones(XU_train.shape[0])
    
    os_val = np.ones(XS_val.shape[0])
    ou_val = np.ones(XU_val.shape[0])

    dataset.train.X = np.r_[XS_train, XU_train]
    dataset.train.Y = np.r_[ys_train, yu_train]
    dataset.train.O = np.r_[os_train, ou_train]

    dataset.train.seen = Container({'X': XS_train,
                                    'Y': ys_train,
                                    'O': os_train})
    dataset.train.unseen = Container({'X': XU_train,
                                      'Y': yu_train,
                                      'O': ou_train})

    dataset.val.X = np.r_[XS_val, XU_val]
    dataset.val.Y = np.r_[ys_val, yu_val]
    dataset.val.O = np.r_[os_val, ou_val]

    dataset.val.seen = Container({'X': XS_val,
                                  'Y': ys_val,
                                'O': os_val})
    dataset.val.unseen = Container({'X': XU_val,
                                    'Y': yu_val,
                                    'O': ou_val})

    #test
    dataset.test.seen.O = np.ones(dataset.test.seen.Y.shape[0])
    dataset.test.unseen.O = np.zeros(dataset.test.unseen.Y.shape[0])

    dataset.n_classes = np.max([dataset.test.seen.Y.max(), dataset.test.unseen.Y.max()])

    return dataset
