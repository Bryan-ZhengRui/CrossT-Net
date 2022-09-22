from Utils.utils1 import *




def datasets_to_idx(dataset):
    """
    :param dataset: train_set or test_set
    :return:
    """
    grd = list(dataset)
    idx1 = list(map(int,[i for i,_,_,_ in grd]))
    idx2 = list(map(int,[i for _,i,_,_ in grd]))
    overlap = [i for _,_,i,_ in grd]
    yaw = [i for _,_,_,i in grd]
    return idx1, idx2, overlap, yaw


def idx2loadmap(idx1, scan_paths):
    """
    :param idx1: the index of the lidar frame（a list）
    :param scan_paths: list of scan paths
    :return: the data list extracted from the scan paths
    """
    data = []
    scan_path = load_files(scan_paths)
    for idx in idx1:
        loadDatapath = scan_path[idx]
        data.append(loadDatapath)
    return data









