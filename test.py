from main_multi_gpu import *
import numpy as np
import matplotlib.mlab as mlab
import matplotlib
import matplotlib.pyplot as plt
import yaml
matplotlib.use('Agg')

def history(diff_all):
    """
    for generating histogram‘ datas
    """
    num = []
    allnum = len(diff_all)
    end = 50
    value_span = 1.0
    numhis = []

    for j in range(0,end):
        num.append(len(diff_all[(diff_all[:] < (j+1)*(value_span/end)) & (diff_all[:] >= j*(value_span/end))]))
        numhis.append(100*num[j]/allnum)

    return numhis


def generate_test_maps(seq):

    depth_all1 = []
    intensity_all1 = []
    normal_all1 = []
    depth_all2 =[]
    intensity_all2 = []
    normal_all2 = []
    overlaps_all = []
    val_set = load_groudtruth("./data/" + seq + "/ground_truth/ground_truth_overlap_yaw.npz")
    idx1, idx2, overlaps, yaw = datasets_to_idx(val_set['overlaps'])
    depth1 = idx2loadmap(idx1, "./data/" + seq + "/depth")
    intensity1 = idx2loadmap(idx1, "./data/" + seq + "/intensity")
    normal1 = idx2loadmap(idx1, "./data/" + seq + "/normal")
    depth2 = idx2loadmap(idx2, "./data/" + seq + "/depth")
    intensity2 = idx2loadmap(idx2, "./data/" + seq + "/intensity")
    normal2 = idx2loadmap(idx2, "./data/" + seq + "/normal")
    depth_all1.extend(depth1)
    intensity_all1.extend(intensity1)
    normal_all1.extend(normal1)
    depth_all2.extend(depth2)
    intensity_all2.extend(intensity2)
    normal_all2.extend(normal2)
    overlaps_all.extend(overlaps)
    print("seq" + seq + " have prepared for test!")
    return depth_all1, intensity_all1, normal_all1, depth_all2, intensity_all2, normal_all2, overlaps_all


def generate_pred_txt(overlap_all, seq='00'):
    """
    :param diff_all:
    :param seq:
    :return:
    """
    a = np.load('./data/'+seq+'/ground_truth/ground_truth_overlap_yaw.npz')
    a = a['overlaps']
    newlist = [i for j in range(len(overlap_all)) for i in overlap_all[j]]
    for i in range(len(newlist)):
        a[i][3] = newlist[i]
    np.savetxt('tmp/'+seq+'predict.txt', a)


@torch.no_grad()
def test(model, data_loader, device, num, batchsize,seq):
    model.eval()
    running_loss = 0.0
    data_loader = tqdm(data_loader, file=sys.stdout,ncols = 160)
    # myloss = torch.nn.L1Loss().to(device)
    # myloss = nn.Tanh().to(device)
    myloss = nn.Sigmoid().to(device)
    overlap_error_all = 0.0
    diff_max = 0.0
    diff_all = []
    overlap_all = []
    idx_max = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs1, inputs2, labels = data
        outputs = model(inputs1.to(device),inputs2.to(device))
        overlap_all.extend(outputs.tolist())
        # loss = abs(outputs - labels.to(device)).mean()
        loss = torch.mean(myloss((torch.abs(outputs.to(device) - labels.to(device))+0.25)*24-13))
        running_loss += loss.item()
        overlap_error_all += (abs(outputs.to(device) - labels.to(device))).sum()
        diff = (abs(outputs.to(device) - labels.to(device))).max()
        diff_all.extend(abs(outputs.to(device) - labels.to(device)).tolist())
        idx = i*batchsize + int((abs(outputs.to(device) - labels.to(device))).argmax(dim=0))
        if diff_max < diff:
            diff_max = diff
            idx_max = idx

        data_loader.desc = "test loss: {:.3f}, diff max:{:.3f}".format( running_loss/(i+1),diff_max)
    overlap_error_mean = overlap_error_all / num
    print('[test ] loss: %.5f' %(running_loss /(i+1)))
    print('[test  overlap_error_mean: %.5f' % ( overlap_error_mean))
    # print('diff_all:',diff_all)
    print(len(diff_all))
    generate_pred_txt(overlap_all = overlap_all, seq = seq)
    return running_loss /(i+1), len(diff_all)






if __name__ == '__main__':

    # load config file
    config_filename = 'config/configfile.yaml'
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)

    # set the related parameters
    gpu_index = config['test']["gpu_index"]
    seq = config['test']["seq_for_test"]
    test_floder = config['test']["floder"]
    batchsize = config['test']["batchsize"]
    load_path = config['test']["load_path"]
    floder = config['test']["floder"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    device = torch.device("cuda:0")



    depth_all_t1, intensity_all_t1, normal_all_t1, depth_all_t2, intensity_all_t2, normal_all_t2, overlaps_all_t = generate_test_maps(seq)
    test_dataset = MyDataset(depth_all_t1, intensity_all_t1, normal_all_t1, depth_all_t2, intensity_all_t2, normal_all_t2,overlaps_all_t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers = 4)
    num = len(depth_all_t1)

    net = CTNet()
    net = nn.DataParallel(net)
    net.to(device)

    try:
        net.load_state_dict(torch.load(load_path))
    except:
        net.module.load_state_dict(torch.load(load_path))

    test_loss, lenth_numhis= test(model=net, data_loader=test_loader, device=device, num=num, batchsize=batchsize, seq = seq)


    a = np.loadtxt( floder + seq + 'predict.txt')
    diff_all = abs(a[:,3]-a[:,2])
    print('finished all Test!')
    # x = np.linspace(0, 1, 1000)
    numhis = np.array(history(diff_all))
    error = np.arange(0, 1.0, 0.02)
    error = error + 0.01
    plt.figure(figsize=(7,4),dpi = 300)
    plt.xlabel('Absolute difference of overlap')
    plt.ylabel('Percentage(%)')
    plt.title('Error distribution of KITTI'+str(seq),fontsize=15)
    plt.xlim([0.0,0.6])
    plt.ylim([0.0,100])
    plt.bar(error,numhis,width=0.02 ,color="green")
    plt.savefig(floder+seq+"diffbar", dpi=500)
    plt.show()




