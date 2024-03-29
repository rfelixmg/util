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
from .experiments import AverageMeter
from .tensors import NpJSONEncoder

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

    def save(self, dic, filename):
        self.save_dict_to_hdf5(dic, filename)

    def load(self, filename):
        return self.load_dict_from_hdf5(filename)

    def save_dict_to_hdf5(self, dic, filename):
        with h5py.File(filename, 'w') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, '/', dic)

    def recursively_save_dict_contents_to_group(self, h5file, path, dic):
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
            ans = {}
            for key, value in self.__dict__.items():
                if isinstance(value, Container):
                    ans[key] = value.as_dict()
                else:
                    ans[key] = value
            return ans

    def get(self, key):
        return self.__dict__[key]

    def keys(self):
        return list(self.__dict__.keys())

    def items(self):
        return list(self.__dict__.items())



def hdf2mat(src_, dst_):
    from scipy.io import savemat
    data = DataH5py().load_dict_from_hdf5(src_)

    for key in data.keys():
        savemat('{}/{}'.format(dst_, key), {key: data[key]})


class Dict_Average_Meter(object):
    def __init__(self):
        pass

    def save(self, fname, save_type='json', flatten=False):
        
        if save_type == 'json':
            import json
            with open(fname, 'w') as fp:
                fp.write(json.dumps(self.as_dict(flatten=flatten), 
                                    cls=NpJSONEncoder,
                                    indent=4))
                
        elif save_type == 'h5py':
            DataH5py().save_dict_to_hdf5(dic=self.as_dict(wobj=True), 
                                         filename=fname)
            
        
    
    def __repr__(self):
        return str(self.as_dict())

    
    def __str__(self):
        return str(self.as_dict())

    def get_iter(self, itr=-1, flatten=True, single_1d=True):
        _out = {}
        def build_flatten(_data, _key=False):
            if not _key:
                _key = '{}'
            else:
                _key += '_{}'
                
            for key, item in _data.items():
                if isinstance(item, dict):
                    build_flatten(item, _key.format(key))
                elif isinstance(item, AverageMeter):
                    if single_1d:
                        if np.array(item.get_iter(itr)).size == 1:
                            _out[_key.format(key)]  = item.get_iter(itr) 
                    else:
                        _out[_key.format(key)]  = np.array(item.get_iter(itr))
                        
        build_flatten(self.__dict__)
        return _out
                    
                    
    def as_dict(self, wobj=False, flatten=False):
        __flatten_dict__ = {}
        def build_print(_data):
            _dict = {}
            for key, item in _data.items():
                if isinstance(item, dict):
                    _dict[key] = build_print(item)
                elif isinstance(item, AverageMeter):
                    _dict[key] = item.get_list()
            return _dict
        
        def build_flatten(_data, _key):
            if not _key:
                _key = '{}'
            else:
                _key += '_{}'
                
            for key, item in _data.items():
                if isinstance(item, dict):
                    build_flatten(item, _key.format(key))
                elif isinstance(item, AverageMeter):
                    __flatten_dict__[_key.format(key)]  = item.get_list()
        if wobj:
            return self.__dict__
        elif flatten:
            build_flatten(self.__dict__, False)
            return __flatten_dict__
        else:
            return build_print(self.__dict__)
    
    def __set_dict__(self, data):
        for key, value in data.items():
            self.__dict__[key] = value

    def get_meter(self, data):
        if self.get_subparam(self.__dict__, data) is False:
            self.set_meter(data)
            return self.get_subparam(self.__dict__, data)
        else:
            return self.get_subparam(self.__dict__, data)
    
    def get_param(self, data):
        return self.get_subparam(self.__dict__, data)

    def get_subparam(self, tree, data):
        levels = data.split('/')
        if(len(levels) > 1):
            if levels[0] in tree:
                return self.get_subparam(tree[levels[0]], '/'.join(levels[1:]))
            else:
                return False
        else:
            if data in tree:
                return tree[data]
            else:
                return False

    def contains(self, namespace):
        return namespace in self.__dict__

    def set_meter(self, namespace):
        levels = namespace.split('/')
        last = len(levels)-1
        tree = self.__dict__
        for key, _level in enumerate(levels):
            if _level in tree:
                
                if key != last:
                    tree = tree[_level]
                else:
                    tree[_level] = AverageMeter()
            else:
                if key != last:
                    tree[_level] = {}
                    tree = tree[_level]
                else:
                    tree[_level] = AverageMeter()
    
    def set_param(self, namespace, data):
        levels = namespace.split('/')
        last = len(levels)-1
        tree = self.__dict__
        for key, _level in enumerate(levels):
            if _level in tree:
                
                if key != last:
                    tree = tree[_level]
                else:
                    tree[_level] = data

            else:
                if key != last:
                    tree[_level] = {}
                    tree = tree[_level]
                else:
                    tree[_level] =data

    def update_meters(self, _base, data):
        for key, item in data.items():
            self.get_meter('{}/{}'.format(_base, key)).update(item)


class DictContainer(object):
    def __init__(self):
        pass

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def as_dict(self):
        return self.__dict__

    def __set_dict__(self, data):
        for key, value in data.items():
            self.__dict__[key] = value

    def get_param(self, data):
        return self.get_subparam(self.__dict__, data)

    def get_subparam(self, tree, data):
        levels = data.split('/')
        if(len(levels) > 1):
            if levels[0] in tree:
                return self.get_subparam(tree[levels[0]], '/'.join(levels[1:]))
            else:
                return False
        else:
            if data in tree:
                return tree[data]
            else:
                return False

    def contains(self, namespace):
        return namespace in self.__dict__

    def set_param(self, namespace, data):
        levels = namespace.split('/')
        last = len(levels)-1
        tree = self.__dict__
        for key, _level in enumerate(levels):
            if _level in tree:
                
                if key != last:
                    tree = tree[_level]
                else:
                    tree[_level] = data

            else:
                if key != last:
                    tree[_level] = {}
                    tree = tree[_level]
                else:
                    tree[_level] = data

if __name__ == '__main__':
    print('-'*100)
    print(':: Testing file: {}'.format(__file__))
    print('-'*100)
