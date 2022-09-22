from utils.Lidar_input import *
import numpy as np

seq = '00'

scan_floder = "./ford_campus/depth"
dst_folder = "./data/ford00/depth"
scan_paths = load_files(scan_floder)

for idx in range(len(scan_paths)):
    loadData = np.load(scan_paths[idx])

    loadData = np.expand_dims(loadData, axis=0)


    dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    np.save(dst_path, loadData)
    print("depth trains complished No",idx,"/",len(scan_paths)-1)
print(np.shape(loadData))


scan_floder = "./ford_campus/intensity"
dst_folder = "./data/ford00/intensity"
scan_paths = load_files(scan_floder)

for idx in range(len(scan_paths)):
    loadData = np.load(scan_paths[idx])

    loadData = np.expand_dims(loadData, axis=0)


    dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    np.save(dst_path, loadData)
    print("intensity trains complished No",idx,"/",len(scan_paths)-1)
print(np.shape(loadData))


scan_floder = "./ford_campus/normal"
dst_folder = "./data/ford00/normal"
scan_paths = load_files(scan_floder)

for idx in range(len(scan_paths)):
    loadData = np.load(scan_paths[idx])
    normal_T = loadData
    # normal_T = np.expand_dims(normal_all, )
    normal_T = normal_T.transpose(2,0,1)

    dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    np.save(dst_path, normal_T)
    print("normal trains complished No",idx,"/",len(scan_paths)-1)
print(np.shape(normal_T))
