import torch
from torch import nn
import torch.functional as F
from layer_templates_pytorch import *

"""
edit this class to modify your neural network, try mixing Conv2d and Linear layers. See how the training time, loss and 
test error changes. Remember to add an activation layer after each CONV or FC layer. There are two ways to write a set 
of layers: the sequential statement or an explicit assignment. In the sequential assignment you just write a sequence 
of layers which are then wrapped up and called in order exactly as you have written them. This is useful for small 
sequences or simple structures, but prevents you from branching, merging or selectively extract features from the 
layers. The explicit assignment on the other hand allows  to write more complex structures, but requires you to 
instantiate the layers in the __init__ function and connect them inside the forward function."""

#################################### EX1 MNIST ####################################
class default_model(nn.Module):
    def __init__(self):
        super(default_model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 10)
        self.act = nn.ReLU()

        # NOTE : add explicit weight initialization to overwrite default pytorch method, here there is a commented
        #       placeholder to show the syntax. Research the initialization methods online, choose one and justify why
        #       you decided to use that in particular.
        # torch.nn.init.normal(self.linear1.weight)

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        return out

################################## EX2 MNIST with Conv layers ####################################
class model_ex2(nn.Module):
    def __init__(self):
        super(model_ex2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.flatten = nn.Flatten()
        # Fully connected layer, output 10
        # Expected to have in input 32*7*7 = 1568, but instead it says that expects 2048. WHY ?
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        out = self.out(x)
        return out

############################### EX2b Fashion-MNIST ###############################
class model_ex2b(nn.Module):
    def __init__(self):
        super(model_ex2b, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.flatten = nn.Flatten()
        # Fully connected layer, output 10
        # Expected to have in input 32*7*7 = 1568
        self.out = nn.Linear(7*7*32, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        out = self.out(x)
        return out

############################### EX3 CIFAR10 ####################################
class model_ex3(nn.Module):
    def __init__(self):
        super(model_ex3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*8*8, 120)  #32*8*8
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

############################### EX3 CIFAR10 Claudia version ####################################
class CNN_cifar10(nn.Module):
    def __init__(self):
        super(CNN_cifar10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dense = nn.Sequential(
            nn.Linear(32 * 8 * 8, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )
        self.linear2 = nn.Linear(128, 10)
        self.drop25 = nn.Dropout(0.25)
        self.drop50 = nn.Dropout(0.5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop25(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.drop50(x)
        out = self.linear2(x)
        return out

############################### EX3 with RESNET ####################################
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=10):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        # self.layer3 = nn.Sequential(
        #     resblock(128, 256, downsample=True),
        #     resblock(256, 256, downsample=False)
        # )


        # self.layer4 = nn.Sequential(
        #     resblock(256, 512, downsample=True),
        #     resblock(512, 512, downsample=False)
        # )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(128, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        # input = self.layer3(input)
        # input = self.layer4(input)
        input = self.gap(input)
        # input = torch.flatten(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)

        return input
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels))
#         self.downsample = downsample
#         self.relu = nn.ReLU()
#         self.out_channels = out_channels
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
#         #self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
#         #self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         # self.fc = nn.Linear(512, num_classes)
#         self.fc = nn.Linear(1024, num_classes)
#
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channels),
#             )
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.layer0(x)
#         #x = self.layer1(x)
#         #x = self.layer2(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)           # flattening the tensor,
#         x = self.fc(x)
#
#         return x





