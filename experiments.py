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


class Print(object):
    def __init__(self):
        pass

    def print_inline(self, x):
        from sys import stdout
        from time import sleep
        stdout.write('\r{}               '.format(x))
        stdout.flush()


def label2hot(y, dim=False):
    if not dim:
        dim = np.max(y) + 1
    return np.eye(dim)[y].astype(np.int)


def hot2label(y):
    return y.argmax(axis=1)

