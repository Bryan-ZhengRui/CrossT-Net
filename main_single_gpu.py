import yaml
from Utils.Lidar_input import *
import sys
from tqdm import tqdm
from data.MyDataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
from models.mymode import *
import os


def generate_training_maps(seqs):
    depth_all1 = []
    intensity_all1 = []
    normal_all1 = []
    depth_all2 =[]
    intensity_all2 = []
    normal_all2 = []
    overlaps_all = []
    for seq in seqs:
        train_set = load_groudtruth("./data/" + seq + "/ground_truth/train_set.npz")
        idx1, idx2, overlaps, yaw = datasets_to_idx(train_set['overlaps'])
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
        print("seq" + seq + " load finished")
    return depth_all1, intensity_all1, normal_all1, depth_all2, intensity_all2, normal_all2, overlaps_all

def generate_val_maps(seqs):
    depth_all1 = []
    intensity_all1 = []
    normal_all1 = []
    depth_all2 =[]
    intensity_all2 = []
    normal_all2 = []
    overlaps_all = []
    for seq in seqs:
        val_set = load_groudtruth("./data/" + seq + "/ground_truth/validation_set.npz")
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

    return depth_all1, intensity_all1, normal_all1, depth_all2, intensity_all2, normal_all2, overlaps_all


def generate_all_maps(seqs):
    depth_all1 = []
    intensity_all1 = []
    normal_all1 = []
    depth_all2 =[]
    intensity_all2 = []
    normal_all2 = []
    overlaps_all = []
    for seq in seqs:
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


if __name__ == '__main__':

    # load config file
    config_filename = 'config/configfile.yaml'
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)

    # set the related parameters
    gpu_index = config['train_singlegpu']["gpu_index"]
    seq_train = config['train_singlegpu']["seq_for_training"]
    seq_val = config['train_singlegpu']["seq_for_validateion"]
    weight_floder = config['train_singlegpu']["weight_saved_floder"]
    learning_rate = config['train_singlegpu']["lr"]
    decay_rate = config['train_singlegpu']["decay_rate"]
    batchsize = config['train_singlegpu']["batchsize"]
    epochs = config['train_singlegpu']["epoch"]

    #select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    device = torch.device("cuda:0")

    seqs1 = seq_train
    seqs2 = seq_val

    depth_all1, intensity_all1, normal_all1, depth_all2, intensity_all2, normal_all2, overlaps_all =  generate_training_maps(seqs1)
    depth_all_v_1, intensity_all_v_1, normal_all_v_1, depth_all_v_2, intensity_all_v_2, normal_all_v_2, overlaps_all_v = generate_val_maps(seqs2)

    print('number of training datas:', len(overlaps_all))
    print('number of validating datas:', len(overlaps_all_v))

    train_dataset = MyDataset(depth_all1, intensity_all1, normal_all1, depth_all2, intensity_all2, normal_all2, overlaps_all)
    val_dataset = MyDataset(depth_all_v_1, intensity_all_v_1, normal_all_v_1, depth_all_v_2, intensity_all_v_2, normal_all_v_2, overlaps_all_v)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batchsize,shuffle=True,pin_memory=True,drop_last = False, num_workers = 4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, pin_memory=True,num_workers = 4)
    print(train_loader)
    net = CTNet()

    net = net.to(device)

    init_lr = learning_rate

    myloss = nn.Sigmoid().to(device)
    # myloss = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=init_lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = decay_rate, last_epoch=-1)
    print ('initial learning rate:',init_lr)
    loss_last = 1.0
    os.makedirs(weight_floder, exist_ok=True)
    for epoch in range(epochs):
        # train

        print('lr:'+str(optimizer.state_dict()['param_groups'][0]['lr']))
        net.train()
        running_loss = 0.0
        batch_loss = 0.0
        train_loader = tqdm(train_loader, file=sys.stdout, ncols = 160 )
        for i,data  in enumerate(train_loader, 0):
            i = i*batchsize
            inputs1, inputs2,labels  =data
            # if i == 0:
            #     print("==========input00:",inputs1[0][0][32])
            #     print("==========input01:", inputs1[1][0][32])
            #     print(inputs1.size())
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs1.to(device),inputs2.to(device))
            # print("============labels:", labels)
            # print("shape", labels.size())
            # print("============outputs:", outputs.to(device))
            # print("shape", outputs.size())
            loss = torch.mean(myloss((torch.abs(outputs.to(device) - labels)+0.3)*24-13))
            # loss = torch.mean(myloss((torch.abs(outputs.to(device) - labels))))
            loss.backward()
            # print(inputs1.grad)
            optimizer.step()
            running_loss += loss.item()
            batch_loss += loss.item()
            if i % batchsize == 0:
                loss_now = running_loss / ((i / batchsize) + 1)
                train_loader.desc = "[epoch {}, {}] epoch loss: {:.4f} batch loss: {:.3f}".format(epoch+1,i+1,loss_now, batch_loss)
                batch_loss = 0.0
            if i % (batchsize * 2000) == 0:
                niter = epoch * len(train_loader) + i / batchsize
        if (epoch+1)%10 == 0:
            torch.save(net.state_dict(), weight_floder+"/weight"+str(epoch+1)+".pth")
        val_loss = evaluate(model=net,data_loader=val_loader,device=device, epoch=epoch, num = len(overlaps_all_v))
        scheduler.step()

    torch.save(net.state_dict(), weight_floder+"/weight_all.pth")
    print('all Training finished!')




