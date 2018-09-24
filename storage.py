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
import h5py
import numpy as np
from util.experiments import AverageMeter


class Json(object):
    def __init__(self):
        pass

    @classmethod
    def save(self, obj, basefile, indent=4):
        '''
        Json().save dict structure as json file
        
        :param obj: dict file
        :param basefile: file name
        :param indent: default 4
        '''

        from json import dump
        import numpy as np
        obj_ = obj

        for field in obj.keys():
            if type(obj[field]) == np.ndarray:
                obj_[field] = obj[field].tolist()

        with open(basefile, 'w') as out:
            dump(obj_, out, sort_keys=True, indent=indent)

    @classmethod
    def load(self, basefile):
        '''
        Load json file as a dict
        :param basefile: filename
        :return: dict
        '''
        from json import load
        with open(basefile, 'r') as out:
            jload = load(out)
        return jload


class DataH5py:
    def __init__(self):
        self.supported_types = (np.ndarray, int, str, bytes,
                                np.int, np.int8, np.int16, np.int32, np.int64,
                                np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
                                np.float, np.float16, np.float32, np.float64, np.float128)
        pass

    def save_dict_to_hdf5(self, dic, filename):
        """
        ....
        """
        with h5py.File(filename, 'w') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, '/', dic)

    def recursively_save_dict_contents_to_group(self, h5file, path, dic):
        """
        ....
        """
        for key, item in dic.items():
            if isinstance(key, (int, np.unicode)):
                key = str(key)
            if isinstance(item, (int, np.unicode)):
                item = str(item)
            if isinstance(item, self.supported_types):
                if isinstance(item, np.ndarray) and item.size > 1:
                    if isinstance(item[0], np.unicode):
                        h5file.create_dataset('{}{}'.format(path, key),
                                          data=np.array(item, dtype='S'))
                    else:
                        h5file['{}{}'.format(path, key)] = item
                else:
                    h5file['{}{}'.format(path, key)] = item
            elif isinstance(item, AverageMeter):
                h5file['{}{}'.format(path, key)] = item.get_list()

            # TODO: better treatment for lists. Preferably, more general.
            elif isinstance(item, list):
                value = np.array([str(subitem) for subitem in item])
                h5file[path + key] = value

            elif isinstance(item, dict):
                self.recursively_save_dict_contents_to_group(h5file,
                                                             path + key + '/', item)

            elif isinstance(item, (Container, Bunch)):
                self.recursively_save_dict_contents_to_group(h5file,
                                                             path + key + '/', item.as_dict())

            elif item is None:
                pass
            else:
                raise ValueError('Cannot save {}:{} type'.format(key, type(item)))

    def load_dict_from_hdf5(self, filename):
        """
        ....
        """
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, '/')

    def recursively_load_dict_contents_from_group(self, h5file, path):
        """
        ....
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(h5file,
                                                                          path + key + '/')
        return ans


class Bunch(dict):
    """Container object for datasets

    copied from scikit-learn
    Dictionary-like object that exposes its keys as attributes.

    """

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def as_dict(self):
        return self.__dict__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass


class Container(object):
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
               setattr(self, key, [Container(sub) if isinstance(sub, dict) else sub for sub in value])
            else:
               setattr(self, key, Container(value) if isinstance(value, dict) else value)

    # def as_dict(self):
    #     return self.__dict__

    def as_dict(self, dtype=None):
        if dtype:
            return {dtype(key): dtype(value) for key, value in self.__dict__.items()}
        else:
            return self.__dict__

    def keys(self):
        return list(self.__dict__.keys())

    def items(self):
        return list(self.__dict__.items())



def hdf2mat(src_, dst_):
    from scipy.io import savemat
    data = DataH5py().load_dict_from_hdf5(src_)

    for key in data.keys():
        savemat('{}/{}'.format(dst_, key), {key: data[key]})








if __name__ == '__main__':
    print('-'*100)
    print(':: Testing file: {}'.format(__file__))
    print('-'*100)

    # _dir = '/media/hdd_2tb/scientific/experiments/eccv18/cub/fclswgan/gan/batch_001/0000_TEST_070318_202809_batch_size-64/data'
    # _dir = '/media/hdd_2tb/scientific/experiments/eccv18/flo/fclswgan/gan/batch_001/0000_TEST_150318_004004_batch_size-64/data'
    _dir = '/media/hdd_2tb/scientific/experiments/eccv18/awa1/fclswgan/gan/batch_001/0000_TEST_220218_163913_batch_size-64/data'

    hdf2mat('{}/data.h5'.format(_dir), _dir)

    # data = DataH5py().load_dict_from_hdf5('/var/scientific/data/eccv18/awa1/data.h5')
    # d = Container(data)
    # print(d.keys())

    # print(d.data.train)
