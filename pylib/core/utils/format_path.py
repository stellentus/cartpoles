import os


def fill_run_number(dic, run_num, param_num, data_root):
    if 'load_params' in dic.keys() and dic['load_params']:
        if len(dic['path'].split("{}")) == 2:
            print("Filling in run number only")
        dic['path'] = dic['path'].format(run_num, param_num)

        path = os.path.join(data_root, dic['path'])
        if not os.path.isfile(path):
            print("Run {} doesn't exist. {}".format(run_num, path))
            exit(1)
    else:
        print("Not Loading Parameter")
    return dic
    