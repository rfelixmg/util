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
def count_directories(root):
    '''
    count_directories return the number of directories in a root
    :param root: base root directory
    :return: number of folders
    '''
    from glob import glob
    return len(glob('{}/*/'.format(root)))


def mkdir(ndirectory, baseroot):
    '''
    make directory
    :param ndirectory: new directory name 
    :param baseroot: base directory
    :return: None
    '''
    import os, warnings

    try:
        os.stat('{}/{}'.format(baseroot,ndirectory))
        warnings.warn('Existent directory: {}/{}'.format(baseroot, ndirectory),
                      RuntimeWarning)
        return '{}/{}'.format(baseroot, ndirectory)
    except:
        os.mkdir('{}/{}'.format(baseroot, ndirectory))
        return '{}/{}'.format(baseroot, ndirectory)


def mkdict(directories, baseroot):
    '''
    
    :param baseroot: 
    :param directories: 
    :return: 
    '''
    if isinstance(directories, list) or isinstance(directories, set):
        for value in directories:
            if isinstance(value, dict):
                mkdict(value, baseroot)
            else:
                mkdir(value, baseroot)

    elif isinstance(directories, dict):
        for ndir, value in directories.items():
            curroot = mkdir(ndir, baseroot)
            mkdict(value, curroot)

def mkexp(baseroot, options, bname, sideinfo=None,
          subdirectories=['checkpoint','results', 'source']):
    '''
    mkexp create a folder for experiments_eccv18
    
    :param baseroot: 
    :param options: 
    :param bname: 
    :param sideinfo: 
    :return: 
    '''

    fname = '{:04}_{}'.format(count_directories(baseroot),
                           bname)
    if isinstance(sideinfo, list):
        for sinfo in sideinfo:
            fname = '{}_{}-{}'.format(fname, sinfo, options.as_dict()[sinfo])

    curroot = mkdir(ndirectory=fname, baseroot=baseroot)
    mkdict(directories=subdirectories, baseroot=curroot)

    return {'root': curroot, 'namespace': fname}





if __name__ == '__main__':
    print('-'*100)
    print(':: Testing file: {}'.format(__file__))
    print('-'*100)

    directories = ['checkpoint', {'validation': ['val1', 'val2']}]
    baseroot= '/tmp/test/'