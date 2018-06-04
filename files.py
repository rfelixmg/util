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