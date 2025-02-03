import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_widths=[100,100,50], output_dim=10, dropout=0.5, linear=False):
        super().__init__()
        act = nn.Identity() if linear else nn.ReLU()
        self.layer_input = nn.Sequential(nn.Linear(input_dim, hidden_widths[0]), act)
        self.layer_hidden = nn.ModuleList()
        for i in range(len(hidden_widths)-1):
            self.layer_hidden.append(nn.Sequential(nn.Dropout(dropout),
                                                   nn.Linear(hidden_widths[i], hidden_widths[i+1]),
                                                   act))
        self.layer_output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_widths[-1], output_dim))
    def forward(self, x):
        x = self.layer_input(x)
        for layer in self.layer_hidden:
            x = layer(x)
        return self.layer_output(x)

class CNN2P2(nn.Module):
    def __init__(self, in_channels=1, output_dim=10, channels=[32,64],
                 kernel_size=[5,5], stride=[1,1], padding=[2,2],
                 num_hidden=[3136,128], dropout=0.5):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0],
                      kernel_size=kernel_size[0],
                      stride=stride[0],
                      padding=padding[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1],
                      kernel_size=kernel_size[1],
                      stride=stride[1],
                      padding=padding[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten()
        )
        self.layer3 = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(num_hidden[0], num_hidden[1]),
                                     nn.ReLU())
        self.layer4 = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(num_hidden[1], output_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)