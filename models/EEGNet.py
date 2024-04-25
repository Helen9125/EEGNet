import torch
import torch.nn as nn


class Depthwisw_separable_EEGNet(nn.Module):
    def __init__(self, activation_func='elu'):
        super(Depthwisw_separable_EEGNet, self).__init__()

        if activation_func == 'elu':
            activation = nn.ELU()
        elif activation_func == 'relu':
            activation = nn.ReLU()
        elif activation_func == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_func == 'tanh':
            activation = nn.Tanh()

        self.channels = 8

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.channels, (1, 32), padding=0, groups=1),
            nn.Conv2d(self.channels, self.channels, 1, padding=0, groups=self.channels),
            nn.BatchNorm2d(self.channels),
            activation,
            nn.Dropout(0.0)
        )


        self.layer2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, (2, 128), padding=(0, 0), groups=self.channels),
            nn.Conv2d(self.channels, 4, 1, padding=0),
            nn.BatchNorm2d(4),
            activation,
            nn.Dropout(0.0),
            nn.MaxPool2d((1, 4), stride=(1, 2))
        )


        self.layer3 = nn.Sequential(
            nn.ZeroPad2d((2, 1, 4, 3)),
            nn.Conv2d(4, 4, (8, 4)),
            nn.BatchNorm2d(4),
            activation,
            nn.Dropout(0.0)
        )


        self.fc = nn.Linear(1180, 1)

    def forward(self, x):
        x = self.layer1(x)
        #print("layer 1:", x.size())
        x = self.layer2(x)
        #print("layer 2:", x.size())
        x = self.layer3(x)
        #print("layer 3:", x.size())
        x = x.view(x.size(0), -1)
        #print("after view:", x.size())
        x = torch.sigmoid(self.fc(x))
        #print("after sigmoid:", x.size())
        x = torch.squeeze(x, dim=1)
        #print("after squeeze:", x.size())
        return x


#standard
class EEGNet(nn.Module):
    def __init__(self, activation_func='elu'):
        super(EEGNet, self).__init__()

        if activation_func == 'elu':
            activation = nn.ELU()
        elif activation_func == 'relu':
            activation = nn.ReLU()
        elif activation_func == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_func == 'tanh':
            activation = nn.Tanh()


        self.channels = 8

        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.channels, (1, 32), padding=0),
            nn.BatchNorm2d(self.channels),
            activation,
            nn.Dropout(0.0)
        )


        self.layer2 = nn.Sequential(
            nn.ZeroPad2d((self.channels, 17, 0, 1)),
            nn.Conv2d(self.channels, 4, (2, 128)),
            nn.BatchNorm2d(4),
            activation,
            nn.Dropout(0.0),
            nn.MaxPool2d((1, 4), stride=(1, 2))
        )


        self.layer3 = nn.Sequential(
            nn.ZeroPad2d((2, 1, 4, 3)),
            nn.Conv2d(4, 4, (8, 4)),
            nn.BatchNorm2d(4),
            activation,
            nn.Dropout(0.0),
            nn.MaxPool2d((2, 4))
        )

        self.fc = nn.Linear(304, 1)

    def forward(self, x):
        x = self.layer1(x)
        #print("layer 1:", x.size())
        x = self.layer2(x)
        #print("layer 2:", x.size())
        x = self.layer3(x)
        #print("layer 3:", x.size())
        x = x.view(x.size(0), -1)
        #print("after view:", x.size())
        x = torch.sigmoid(self.fc(x))
        #print("after sigmoid:", x.size())
        x = torch.squeeze(x, dim=1)
        #print("after squeeze:", x.size())
        return x





# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        pass

    def forward(self, x):
        pass