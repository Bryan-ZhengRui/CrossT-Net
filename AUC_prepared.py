import numpy as np
from sklearn.metrics import auc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pykitti

def distance(x1,y1,x2,y2):
    x_2 = (x1-x2)**2
    y_2 = (y1-y2)**2
    return pow((x_2+y_2),0.5)


seq = '00'
a = np.loadtxt('tmp/'+seq+'predict.txt')
# basedir = '../OverlapNet/kitti'
# Specify the dataset to load
# sequence = '00'
# dataset = pykitti.odometry(basedir, sequence)
# print("dataset initialized")

idx = []
idx_gt = []
recall=[]
precision = []
f1 = []
for i in range(a.shape[0]):
    if a[i][0] > 100  and a[i][2]>0.3 and a[i][0]-a[i][1] > 100:
        a[i][2] = 1
        idx_gt.append(int(a[i][1]))
    else :
        a[i][2] = 0

idx_gt = list(set(idx_gt))
print('finished gt')

for i in np.arange(10,99,0.1):
    tp = 0
    fp = 0
    fn = 0
    thred = i/100
    idx_pred = []
    for idx1 in range(101, 4541):
        preds = a[int(((idx1+1)*idx1)/2):int(((idx1+1)*idx1)/2+idx1+1-101),3]
        if np.max(preds) > thred:
            max_id = int(np.argmax(preds)+int(((idx1+1)*idx1)/2))
            # a[max_id][3] = 1
            idx_pred.append(idx1)
            if a[max_id][2] == 1:
                tp += 1
            else:
                fp += 1
        elif np.max(preds) <= thred:
            max_id = int(np.argmax(preds) + int(((idx1 + 1) * idx1) / 2))
            if a[max_id][2] == 1:
                fn += 1
    pcs = tp/(tp+fp+1e-8)
    rec = tp/(tp+fn+1e-8)
    f1.append(2 * (rec * pcs) / (rec + pcs+1e-8))
    precision.append(pcs)
    recall.append(rec)
    print('%.f/%.d',thred,1)


# precision.append(1.0)
# recall.append(1e-8)
print("precision:",precision)
print("recall:",recall)
print('f1 max:',max(f1))




plt.figure()
plt.title('PR CURVE ')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR CURVE (AUC={:.4f},f1 max={:.4f})'.format(auc(recall,precision),max(f1)))
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.2])
plt.plot(recall,precision,'-', linewidth=1.0,color='b')
plt.savefig("tmp/"+seq, dpi=500)
plt.show()




