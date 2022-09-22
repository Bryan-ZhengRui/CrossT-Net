from models.attention_module import *
import models.attention_module



class CTNet(nn.Module):

    def __init__(self):
        super(CTNet, self).__init__()
        self.conv1 = nn.Conv2d(5, 8, kernel_size=(3,9), stride=(2,2),bias=False)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,9), stride=(2,1),bias=False)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3,9), stride=(2, 1),bias=False)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3,9), stride=(2,1),bias=False)
        self.conv5 = nn.Conv2d(64,64, kernel_size=(2,9), stride=(1,1),bias=False)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=(2,9), stride=(1, 1),bias=False)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1),bias=False)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1),bias=False)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 1),bias=False)
        self.conv10= nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 1),bias=False)

        self.laynorm0 = nn.LayerNorm([64,900])
        self.laynorm1 = nn.LayerNorm([31,446])
        self.laynorm2 = nn.LayerNorm([15,438])
        self.laynorm3 = nn.LayerNorm([7,430])
        self.laynorm4 = nn.LayerNorm([3,422])
        self.laynorm5 = nn.LayerNorm([2,414])
        self.laynorm6 = nn.LayerNorm([1,406])
        self.laynorm7 = nn.LayerNorm([1,398])
        self.laynorm8 = nn.LayerNorm([1,390])
        self.laynorm9 = nn.LayerNorm([1,388])
        self.laynorm10 = nn.LayerNorm([1,386])

        self.laynormL1 = nn.LayerNorm(256)
        self.laynormL2 = nn.LayerNorm(64)

        #delta layers
        # self.dense = nn.Linear(484*256, 1)
        self.dense = nn.Linear(16588800, 1)
        # self.dense = nn.Linear(552960, 1)
        # self.dense = nn.Linear(46080, 1)
        self.conv12 = nn.Conv2d(128, 64, kernel_size=(1, 15), stride=(1, 15))
        self.conv13 = nn.Conv2d(64, 128, kernel_size=(15, 1), stride=(15, 1))
        self.conv14 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1,1))

        self.eca_fca = models.attention_module.FeatureFusionNetwork(d_model=128, dim_feedforward=512, num_featurefusion_layers = 2, dropout = 0.01)

        self.linear1 = nn.Linear(128*386, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 1)
        #drop out
        self.drop1 = nn.Dropout(p = 0.05)
        self.drop2 = nn.Dropout(p = 0.01)

        self.apply(self._init_weights)



    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d)):
                nn.init.trunc_normal_(m.weight,mean=0.0, std=1.0)


    def forward(self, x1, x2):

        x1 =F.leaky_relu(self.conv1(x1))
        x2 = F.leaky_relu(self.conv1(x2))
        x1 = self.laynorm1(x1)
        x2 = self.laynorm1(x2)
        x1 =F.leaky_relu(self.conv2(x1))
        x2 = F.leaky_relu(self.conv2(x2))
        x1 = self.laynorm2(x1)
        x2 = self.laynorm2(x2)
        x1 =F.leaky_relu(self.conv3(x1))
        x2 = F.leaky_relu(self.conv3(x2))
        x1 = self.laynorm3(x1)
        x2 = self.laynorm3(x2)
        x1 =F.leaky_relu(self.conv4(x1))
        x2 = F.leaky_relu(self.conv4(x2))
        x1 = self.laynorm4(x1)
        x2 = self.laynorm4(x2)
        x1 =F.leaky_relu(self.conv5(x1))
        x2 = F.leaky_relu(self.conv5(x2))
        x1 = self.laynorm5(x1)
        x2 = self.laynorm5(x2)
        x1 =F.leaky_relu(self.conv6(x1))
        x2 = F.leaky_relu(self.conv6(x2))
        x1 = self.laynorm6(x1)
        x2 = self.laynorm6(x2)
        x1 =F.leaky_relu(self.conv7(x1))
        x2 = F.leaky_relu(self.conv7(x2))
        x1 = self.laynorm7(x1)
        x2 = self.laynorm7(x2)
        x1 =F.leaky_relu(self.conv8(x1))
        x2 = F.leaky_relu(self.conv8(x2))
        x1 = self.laynorm8(x1)
        x2 = self.laynorm8(x2)
        x1 =F.leaky_relu(self.conv9(x1))
        x2 =F.leaky_relu(self.conv9(x2))
        x1 = self.laynorm9(x1)
        x2 = self.laynorm9(x2)
        x1 = F.relu(self.conv10(x1))
        x2 = F.relu(self.conv10(x2))

        x = self.eca_fca(x1,x2)
        x = torch.flatten(x, 1, -1)

        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x = self.laynormL1(x)
        x = F.relu(self.linear2(x))
        x = self.laynormL2(x)
        x = torch.sigmoid(self.linear3(x))

        return x


if __name__ == "__main__":
        net = CTNet()
        summary(net, [[5, 64, 900], [5, 64, 900]], batch_size=16, device="cpu")