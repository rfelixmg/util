class ImageLoader(object):
    '''
    :: ImageLoader
        Image Loader from jpeg to numpy.ndarray
        
        For every image, this loader a numpy array (height, width, channels). Directory contraints:
        
        /data/
            /class_01/img_001.jpg
            ...
            /class_01/img_00c1.jpg
            
            /class_0C/img_001.jpg
            ...
            /class_0C/img_00cc.jpg
    '''
    def __init__(self, root, batch_size=64, shuffle=False, is_training=False, resize_min=256, crop=(224,224)):
        self.root = root
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.resize_min=resize_min
        self.crop=crop

        self.__setup__()

    def info(self):
        return {'root': self.root,
                'number_files:': self.nfiles,
                'shuffle': int(self.shuffle)}

    def __repr__(self):
        return 'Image loader: {} \n' \
        'number files: {}\n' \
        'batch size: {}\n '\
        'shuffle: {}\n'.format(self.root, self.nfiles, self.batch_size, self.shuffle) 
        

    def _list_files_(self, root, join=True, obj_type=['jpeg', 'jpg', 'png']):
        import os
        from numpy import sort

        files = os.listdir('{}/'.format(root))
        out_files = []
        for obj in files:
            if os.path.isdir(os.path.join(root, obj)) is False:

                if not obj_type:
                    if join:
                        out_files.append(os.path.join(root, obj))
                    else:
                        out_files.append(obj)
                else:
                    
                    if True in [obj.lower().endswith('.{}'.format(_ot)) for _ot in obj_type]:
                        if join:
                            out_files.append(os.path.join(root, obj))
                        else:
                            out_files.append(obj)
                    
        return sort(out_files)
    
    def _list_directories_(self, root, join=True):
        import os
        from numpy import sort
        folders = []
        for fold in os.listdir(root):
            if os.path.isdir(os.path.join(root, fold)):
                if join:
                    folders.append('{}/{}'.format(root, fold))
                else:
                    folders.append(fold)
        return list(sort(folders))
    
    def dataset(self, idx=False):
        '''
        dataset
        :: [key, file, classname]_{i}^{N}
        '''
        if idx:
            return self._dataset[idx]
        else:
            return self._dataset
        
    def __setup__(self):
        '''
        build dataset
        :: [class_id \in (1,C), file, classname]_{i}^{N}
        '''
        self.classespath = self._list_directories_(self.root, join=True)
        self.classesname = self._list_directories_(self.root, join=False)
        _id = 0
        self._dataset = []
        for key, (_classpath, _classname) in enumerate(zip(self.classespath, self.classesname)):
            for _files in self._list_files_(_classpath, True):
                _id += 1
                self._dataset.append([key + 1, _files, _classname, _id])

        self.nfiles = len(self._dataset)
        self.nclasses = len(self.classesname)
        
        if self.shuffle:
            from numpy import arange
            from numpy.random import shuffle
            _idx = arange(0, self.nfiles)
            shuffle(_idx)
            self._dataset = self._dataset[_idx]

    def _resize_scale(self, img, _channels=3):
        from numpy import min, array
        from skimage.transform import resize
        _h, _w = img.shape[0:2]
        smaller_dim = min([_h,_w])
        scale_ratio = self.resize_min / smaller_dim
        _newh, _neww = (int(scale_ratio * _h), int(scale_ratio * _w))
        if len(img.shape) == 2:
            img = img[:,:,None].repeat(_channels, axis=-1)
            
        return resize(img, (_newh,_neww), mode='symmetric', preserve_range=True)

    def _central_crop(self, img):
        _h, _w = img.shape[0:2]    
        amount_to_be_cropped_h = (_h - self.crop[0])
        crop_top = amount_to_be_cropped_h // 2

        amount_to_be_cropped_w = (_w - self.crop[1])
        crop_left = amount_to_be_cropped_w // 2

        return img[crop_top:crop_top+self.crop[0], crop_left:crop_left+self.crop[1], :]

    def _mean_image_subtraction(self, img, _means=[123.68, 116.78, 103.94]):
        return img - _means            

    def _load_instance_(self, impath):
        from skimage import io
        try:
            _img = self._resize_scale(io.imread(impath))
            _img = self._central_crop(_img)
            _img = self._mean_image_subtraction(_img)
            return _img
        except Exception as e:
            raise e
            
    def sample(self, idx=False):
        from numpy.random import randint
        if idx:
            _data = self.dataset(idx)
        else:
            _data = self.dataset(randint(0,self.nfiles))
        return [self._load_instance_(_data[1]), *_data]
    
    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError("CustomRange index out of range")
        return self.sample(item)

    def size(self):
        return self.__len__()

    def __len__(self):
        return self.nfiles

    def len(self):
        return int(self.nfiles / self.batch_size)
            
    def get_batch(self, _start, _end=False):
        '''
            @return:
                return (images [numpy], ground_truth [y], label [str], filename [str])
        '''
        from numpy import arange, array

        if _end:
            assert _start < _end
            assert _end < self.len()
        else:
            _end = self.len()

        for _b in arange(_start, _end):
            _st = _b * self.batch_size
            _ed = _st + self.batch_size

            _data = self._dataset[_st:_ed]
            _imgs = array([self._load_instance_(_instance[1]) for _instance in _data])
            _data = array(_data)

            yield (_imgs, _data[:,0], _data[:,2], _data[:,1])

    def __iter__(self):
        '''
            @return:
                return (images [numpy], ground_truth [y], label [str], filename [str])
        '''
        from numpy import arange, array
        for _id in arange(0, self.nfiles, self.batch_size):
            _data = self._dataset[_id:(_id+self.batch_size)]
            _imgs = array([self._load_instance_(_instance[1]) for _instance in _data])
            _data = array(_data)

            yield (_imgs, _data[:,0], _data[:,2], _data[:,1])