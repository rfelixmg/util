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
    def __init__(self, root, batch_size=512, shuffle=False, is_training=False, resize_min=256):
        self.root = root
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.__setup__()
        

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

    def _resize_scale(self, img, resize_min=256, _channels=3):
        from numpy import min, array
        from skimage.transform import resize
        _h, _w = img.shape[0:2]
        smaller_dim = min([_h,_w])
        scale_ratio = resize_min / smaller_dim
        _newh, _neww = (int(scale_ratio * _h), int(scale_ratio * _w))
        if len(img.shape) == 2:
            img = img[:,:,None].repeat(_channels, axis=-1)
            
        return resize(img, (_newh,_neww), mode='symmetric', preserve_range=True)

    def _central_crop(self, img, crop=(224,224)):
        _h, _w = img.shape[0:2]    
        amount_to_be_cropped_h = (_h - crop[0])
        crop_top = amount_to_be_cropped_h // 2

        amount_to_be_cropped_w = (_w - crop[1])
        crop_left = amount_to_be_cropped_w // 2

        return img[crop_top:crop_top+crop[0], crop_left:crop_left+crop[1], :]

    def _mean_image_subtraction(self, img, _means=[123.68, 116.78, 103.94]):
        return img - _means            

    def _load_instance_(self, data):
        from skimage import io
        try:
            _img = self._resize_scale(io.imread(data[1]))
            _img = self._central_crop(_img)
            _img = self._mean_image_subtraction(_img)
            return (_img, *data)
        except Exception as e:
            print(data)
            raise e
            
    def sample(self, idx=False):
        from numpy.random import randint
        if idx:
            _data = self.dataset(idx)
        else:
            _data = self.dataset(randint(0,self.nfiles))
        return self._load_instance_(_data)
    
    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError("CustomRange index out of range")
        return self.sample(item)

    def size(self):
        return self.__len__()

    def __len__(self):
        return self.nfiles
    
    def __next__(self):
        '''
        [x, y,folder,fname]
        '''
        from numpy import arange, array
        for _id in arange(0, self.nfiles, self.batch_size):
            _data = self._dataset[_id:(_id+self.batch_size)]
            _data = array([self._load_instance_(_instance) for _instance in _data])
            return (_data[:,0], _data[:,1], _data[:,3], _data[:,2])