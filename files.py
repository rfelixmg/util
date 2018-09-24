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
def unique(seq):
    """
        unique:
               Retorna valores unicos de uma lista

        Parametros
        ----------
        seq: array de objetos
            lista de objetos


        Examples
        --------
        # >> import file_utils
        # >>> file_utils.unique([1,1,2,3,3,4])    # lista de numeros
        # array([1,2,3,4])
    """
    set1 = set(seq)
    return list(set1)

def list_directories(root, join=True):
    """
        get_files_path_in_folder:
               Retorna lista de string contendo os diretorios inclusos no diretorio de pesquisa

        Parametros
        ----------
        folder_path: string
            String que indica o diretório


        Examples
        --------
        # >>> import file_utils
        # >>> file_utils.get_folders('C:/')    # palavra a ser corrigida
        array(['C:/Arquivos de Programas/', 'C:/System32/'])
    """
    import os
    from numpy import sort

    folders = []
    for fold in os.listdir(root):
        if os.path.isdir(os.path.join(root, fold)):
            if join:
                folders.append('{}/{}'.format(root, fold))
            else:
                folders.append(fold)

    return sort(folders)

def list_files(root, join=True, obj_type=None):
    import os
    from numpy import sort

    files = os.listdir('{}/'.format(root))
    out_files = []
    for obj in files:
        if os.path.isdir(os.path.join(root, obj)) is False:

            if (obj_type is None):
                if join:
                    out_files.append(os.path.join(root, obj))
                else:
                    out_files.append(obj)

            elif join:
                out_files.append(os.path.join(root, obj)) if obj.endswith('.{}'.format(obj_type)) else False
            else:
                out_files.append(obj) if obj.endswith('.{}'.format(obj_type)) else False

    return sort(out_files)


if __name__ == '__main__':
    print('-' * 100)
    print(':: Testing file: {}'.format(__file__))
    print('-' * 100)

    directories = ['checkpoint', {'validation': ['val1', 'val2']}]
    baseroot = '/tmp/test/'