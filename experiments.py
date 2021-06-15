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
import numpy as np


class CronoMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        import datetime, time

        self.start = datetime.datetime.now()
        self.step = 0.
        self.avg = 0.
        self.sum = 0.
        self.now = time.time()
        self.previous = self.now
        self.count = 0
        self.expected = 1.
        self.accumulative = []

    def get_total(self):
        return self.format_time(measure=self.sum)

    def get_step(self):
        return self.format_time(measure=self.step)

    def from_last(self):
        import time
        return time.time() - self.previous

    def get_avg(self):
        return self.format_time(measure=self.avg)

    def format_time(self, measure):
        return '%.2d:%.2d:%.2f' % (np.int(measure / 360.),
                                   np.int(measure / 60.),
                                   (measure % 60))

    def update(self, n=1):
        import time
        self.now = time.time()
        self.step = self.now - self.previous
        self.previous = self.now
        self.sum += self.step
        self.count += n
        self.avg = self.sum / self.count
        self.accumulative.append(self.step)

    def get_end(self, current, last, txt=False):
        import datetime
        r = last - current + 1
        time_left = self.avg * r
        self.expected = datetime.datetime.now() + \
                        datetime.timedelta(seconds=time_left)
        c_ = ('%.' + str(len(str(last))) + 'd') % (current)
        if txt:
            print
            txt + '(%s/%d) Avg [%s] Prv [%s]  | Estimative %s' % \
                  (c_,
                   last,
                   self.get_step(),
                   self.get_avg(),
                   '{:%d/%b %H:%M:%S}'.format(self.expected))
        else:
            print
            '(%s/%d) Avg [%s] Prv [%s]  | Estimative %s' % \
            (c_,
             last,
             self.get_step(),
             self.get_avg(),
             '{:%d/%b %H:%M:%S}'.format(self.expected))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def max(self):
        assert len(self.list) > 0
        return np.array(self.list).max()

    def min(self):
        assert len(self.list) > 0
        return np.array(self.list).min()

    def mean(self):
        assert len(self.list) > 0
        return np.array(self.list).mean()

    def var(self):
        assert len(self.list) > 0
        return np.array(self.list).var()

    def summary(self):
        return '(min: {:.4g} | mean: {:.4g} | max: {:.4g} | val: {:.4g})'.format(self.min(),
                                                                                 self.mean(),
                                                                                 self.max(),
                                                                                 self.val)

    def stats(self):
        return {'min': self.min(),
                'mean': self.mean(),
                'max': self.max(),
                'val': self.val}

    def get_last(self):
        return self.list[-1]

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.list = [0.]
        self.flag = True

    def get_iter(self, itr=-1):
        return self.list[itr]

    def value(self):
        return self.val

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.flag:
            self.list = [val] * n
            self.flag = False
        else:
            [self.list.append(val) for i in range(n)]

    def repeat(self, n=1):
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.flag:
            self.list = [self.val] * n
            self.flag = False
        else:
            [self.list.append(self.val) for i in range(n)]

    def get_list(self):
        return np.array(self.list)

    def savetxt(self, fname):
        try:
            np.savetxt(fname, np.array(self.list))
        except:
            raise 'Error saving {}'.format(fname)


def generate_result_list(split_val=False):
    return {'train':{},
            'val':{'seen':{},
                   'unseen':{}},
            'test': {'seen': {},
                    'unseen': {}}}

def generate_metric_list(metric_list, split_val=False):
    from util.storage import Container
    if split_val:
        return Container({'train': {cname: AverageMeter() for cname in metric_list},
                          'val': {'seen': {cname: AverageMeter() for cname in metric_list},
                                   'unseen': {cname: AverageMeter() for cname in metric_list}},
                          'test': {'seen': {cname: AverageMeter() for cname in metric_list},
                                   'unseen': {cname: AverageMeter() for cname in metric_list}},
                          'hmean': {'val': AverageMeter(),
                                    'test': AverageMeter(),}})
    else:
        return Container({'train': {cname: AverageMeter() for cname in metric_list},
                          'val': {cname: AverageMeter() for cname in metric_list},
                          'test': {'seen': {cname: AverageMeter() for cname in metric_list},
                                   'unseen': {cname: AverageMeter() for cname in metric_list}},
                          'hmean': AverageMeter()})


class Print(object):
    def __init__(self):
        pass

    def print_inline(self, x):
        from sys import stdout
        stdout.write('\r{}               '.format(x))
        stdout.flush()


def label2hot(y, dim=False):
    if not dim:
        dim = np.max(y) + 1
    return np.eye(dim)[y].astype(np.int)


def hot2label(y):
    return y.argmax(axis=1)


def id2label(x, data):
    """
    id2label given a vector, transform based on a dict
    :param x: vector
    :param data: dict{int id: int label}
    :return: 
    """
    return np.array([data[kk] for kk in x]).astype(np.int)


def list_ids(x, shuffle=True):
    """
    list ids: get a matrix and return a shuffle list of positions
    :param x: 
    :param shuffle: 
    :return: 
    """
    dim_x = x.shape[0]
    # ids_ = np.array(range(dim_x))

    if shuffle:
        ids_ = np.random.permutation(dim_x)
    else:
        ids_ = np.array(range(dim_x))

    return ids_, dim_x

def garbage_checklist(checklist, cname, nmax=10, ctype='min', verbose=False):
    """
    garbage_checklist: this method aims to clean the folder where the models are being saved.
    This method keeps the harddrive clean of undeserible models.

    :param checklist: dictionary {"current":[{'file': '', cname: float(), 'epoch': int{}}]
                                  "deleted":[{'file': '', cname: float(), 'epoch': int{}}]}
    :param cname: name of the criteria parameter
    :param nmax: number of models allowed to be saved on disk
    :param ctype: criteria type, min: minimization or max: maximization

    :return: True: if process correctly, False if some strange behavior happned;


    Example:
    --------

    checklist = {"current":[{'file': '1', 'cname': 1, 'epoch': 1, 'deletable':True},
                            {'file': '2', 'cname': 2, 'epoch': 2, 'deletable':False},
                            {'file': '3', 'cname': 3, 'epoch': 3, 'deletable':True},
                            {'file': '4', 'cname': 4, 'epoch': 4, 'deletable':False},
                            {'file': '5', 'cname': 5, 'epoch': 5, 'deletable':True},
                            {'file': '6', 'cname': 6, 'epoch': 6, 'deletable':False}]}
    garbage_model(checklist, 'cname', nmax=3)
                        
    """
    import shutil
    from os import symlink
    import os
    try:
        if verbose:
            print(':: GargabageModel - Initializing garbage collector... ')
            print(':: GargabageModel - length list: {}'.format(len(checklist['current'])))
        from os import remove
        current_checkpoints = []
        for value in checklist['current']:
            if value['deletable']:
                current_checkpoints.append(value)

        if len(current_checkpoints) > nmax:
            if ctype == 'min':
                current_checkpoints.sort(key=lambda x: x[cname])
            elif ctype == 'max':
                current_checkpoints.sort(key=lambda x: x[cname], reverse=True)
            else:
                raise ":: GargabageModel - Not implemented: criteria == {}".format(ctype)

            if verbose:
                print(':: GargabageModel - deleting model: {}'.format(checklist['current'][-1]['file']))
            delete_item = current_checkpoints[-1]
            sym_item = current_checkpoints[0]
            try:
                remove('{}/best_model'.format(os.path.dirname(sym_item['file'][:-1])))
            except:
                print(':: Removing Symbolic link fail!')
            try:
                symlink(sym_item['file'], '{}/best_model'.format(os.path.dirname(sym_item['file'][:-1])))
            except:
                print(':: Creating Symbolic link fail!')


            # Remove model from folder
            try:
                remove(delete_item['file'])
            except:

                shutil.rmtree(delete_item['file'], ignore_errors=True)

            checklist['deleted'].append(delete_item)
            for key, value in enumerate(checklist['current']):
                if delete_item['epoch'] == value['epoch']:
                    key_delete = key
            del(delete_item, checklist['current'][key_delete])

            return checklist


        else:
            if verbose:
                print(':: GargabageModel - Minimum number of models not reached yet')

            return checklist

    except Exception as e:
        import sys, traceback
        print(':: Exception::GargabageModel - {}'.format(e))
        traceback.print_exc(file=sys.stdout)


def checkpoint_assessment(count_inertia, history=tuple(), max_inertia=3, min_criteria=1., n_element=4):
    """
    checkpoint_assessment: this function assess if the code should save the checkpoint
    :param count_inertia: number of previous inertia
    :param history: lists that should be assess
    :param max_inertia: number maximum of inertia possible
    :param min_criteria: number min of criteria to be assess
    :param n_element: number of elements minimum to assess
    
    :return: dictkey{'save': Bool, 'inertia': Int, 'interrupt': Bool}
    """

    assert len(history) >= 1

    if history[0][1].shape[0] >= n_element:
        criterias = 0
        for ctype, metric, name in history:
            l_element = metric[-n_element:].shape[0] - 1
            if ctype is 'max':
                if metric[-n_element:].argmax() == l_element:
                    criterias += 1
            elif ctype is 'min':
                if metric[-n_element:].argmax() == l_element:
                    criterias += 1

        if criterias >= min_criteria:
            return {'save': True, 'inertia': 0, 'interrupt': False}

        elif count_inertia > max_inertia:
            return {'save': True, 'inertia': count_inertia + 1, 'interrupt': True}
        else:
            return {'save': True, 'inertia': count_inertia, 'interrupt': True}

    else:
        return {'save': True, 'inertia': 0, 'interrupt': False}


def find_config(root):
    from os import listdir
    for tfile in listdir(root):
        if tfile.startswith('configuration_'):
            return '{}/{}'.format(root, tfile)
    return False


def save_checkpoint(model, checkpoint_dir, history, value, epoch, epochs, options, checklist, check_metric, ctype='min'):
    # TODO: is really necessary to save 200mb per checkpoint?
    if (epoch % options.checkpoint == 0) or (epoch + 2 > epochs):
        if options.savepoints:
            if epoch in options.savepoints:
                print(':: Model save point in savepoints {}'.format(options.savepoints))
                model.save(checkpoint_dir, epoch)
                checklist['current'].append({'file': checkpoint_dir,
                                             check_metric: value,
                                             'epoch': epoch,
                                             'deletable': False})
            else:
                checkpoint_info = checkpoint_assessment(0, history=history, max_inertia=options.checkpoints_inertia, min_criteria=1., n_element=5)

                if checkpoint_info['save']:
                    model.save(checkpoint_dir, epoch)
                    checklist['current'].append({'file': checkpoint_dir,
                                                 check_metric: value,
                                                 'epoch': epoch,
                                                 'deletable': True})
                    checklist = garbage_checklist(checklist=checklist, cname=check_metric, ctype=ctype,
                                                  nmax=options.checkpoints_max)
        else:
            checkpoint_info = checkpoint_assessment(0, history=history, max_inertia=options.checkpoints_inertia,
                                                    min_criteria=1., n_element=5)

            if checkpoint_info['save']:
                model.save(checkpoint_dir, epoch)
                checklist['current'].append({'file': checkpoint_dir,
                                             check_metric: value,
                                             'epoch': epoch,
                                             'deletable': True})
                checklist = garbage_checklist(checklist=checklist, cname=check_metric, ctype=ctype,
                                              nmax=options.checkpoints_max)

            # TODO: add inertia interruption to this models
            #if checkpoint_info['interrupt']:
    return checklist