import torchvision
import os
import numpy as np
from tqdm import tqdm
import sys
import torch
import torch.nn as nn



def split_train_val(ground_truth_mapping):
    """Split the ground truth data into training and validation two parts.
       Args:
         ground_truth_mapping: the raw ground truth mapping array
       Returns:
         train_set: data used for training
         test_set: data used for validation
    """
    # set the ratio of validation data
    train_size = int(len(ground_truth_mapping) / 10)
    test_size = len(ground_truth_mapping) - train_size
    train_set, test_set = torch.utils.data.random_split(ground_truth_mapping, [train_size, test_size])
    print('finished generating training data and validation data')

    return train_set, test_set

def load_files(folder):
  """ Load all files in a folder and sort.
  """
  file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
  file_paths.sort()
  return file_paths



def load_groudtruth(file_path):
    """ load the groundtruth files(.npz) to numpy
        Returns:
               file: file构成：['overlaps'，'seq']
    """
    file = np.load(file_path,allow_pickle=True)
    return file


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num):
    model.eval()
    running_loss = 0.0

    data_loader = tqdm(data_loader, file=sys.stdout)
    # myloss = torch.nn.L1Loss().to(device)
    # myloss = nn.Tanh().to(device)
    myloss = nn.Sigmoid().to(device)
    overlap_error_all = 0.0
    diff_max = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs1, inputs2, labels = data
        outputs = model(inputs1.to(device),inputs2.to(device))
        loss = torch.mean(myloss((torch.abs(outputs.to(device) - labels.to(device))+0.3)*24-13))
        running_loss += loss.item()
        overlap_error_all += (abs(outputs.to(device) - labels.to(device))).sum()
        data_loader.desc = "[valid epoch {}] loss: {:.3f} max overlap_diff:{:.3f}".format(epoch+1, running_loss/(i+1), diff_max )
        diff = (abs(outputs.to(device) - labels.to(device))).max()
        if diff_max < diff:
            diff_max = diff

    overlap_error_mean = overlap_error_all / num
    print('[valid epoch %d] loss: %.5f' %(epoch+1,running_loss /(i+1)))
    print('[valid epoch %d] overlap_error_mean: %.5f,  max overlap error:%.4f' % (epoch + 1, overlap_error_mean,diff_max))
    return running_loss /(i+1)



def l2_norm(input1, input2):
    min1 = torch.min(input1, dim = 2)
    min1 = torch.min(min1.values, dim=2)
    max1 = torch.max(input1, dim=2)
    max1 = torch.max(max1.values, dim=2)
    output1 = torch.sub(input1, min1.values.unsqueeze(2).unsqueeze(2)).true_divide(max1.values.unsqueeze(2).unsqueeze(2) - min1.values.unsqueeze(2).unsqueeze(2))
    min2 = torch.min(input2, dim = 2)
    min2 = torch.min(min2.values, dim=2)
    max2 = torch.max(input2, dim=2)
    max2 = torch.max(max2.values, dim=2)
    output2 = torch.sub(input2, min2.values.unsqueeze(2).unsqueeze(2)).true_divide(max2.values.unsqueeze(2).unsqueeze(2) - min2.values.unsqueeze(2).unsqueeze(2))

    return output1, output2


def l2_norm_one(input1, axit=0):
    norm1 = torch.norm(input1,2,axit,True)

    output1 = torch.div(input1, norm1)

    return output1


def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
          pose_path: (Complete) filename for the pose file
        Returns:
          A numpy array of size nx4x4 with n poses as 4x4 transformation
          matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)



if __name__ == '__main__':

    file = "D:/code/OverlapNet Loop Closing for LiDAR-based SLAM/OverlapNet-master/data/07/depth"
    path = load_files(file)
    print(path[0:9])