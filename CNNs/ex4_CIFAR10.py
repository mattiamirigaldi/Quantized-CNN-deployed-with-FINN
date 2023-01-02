from torch import nn
from torch.nn import Module
import quantization as cs
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant
import torch.nn.functional as F
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType

from brevitas.quant import Uint8ActPerTensorFloat as ActQuant
from brevitas.quant import Int8WeightPerTensorFloat as WeighQuant

#################### CIFAR10 quantized with Pytorch custom classes #############################
class CIFAR10_quant(nn.Module):
    def __init__(self):
        super(CIFAR10_quant, self).__init__()
        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 8
        self.quant_method = 'scale'
        self.alpha_coeff = 10.0

        self.quantization = True

        self.conv1 = cs.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2),
                               act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                               quantization=self.quantization, quant_method=self.quant_method)
        self.conv2 = cs.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2),
                               act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                               quantization=self.quantization, quant_method=self.quant_method)

        self.act = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)

        self.flatten = nn.Flatten()
        self.dense1 = cs.Linear(in_channels=32 * 8 * 8, out_channels=128, act_bit=self.act_bit,
                                weight_bit=self.weight_bit,
                                bias_bit=self.bias_bit, quantization=self.quantization,
                                quant_method=self.quant_method)
        self.dense2 = cs.Linear(in_channels=128, out_channels=10, act_bit=self.act_bit,
                                weight_bit=self.weight_bit,
                                bias_bit=self.bias_bit, quantization=self.quantization,
                                quant_method=self.quant_method)
        self.pool = nn.MaxPool2d(2)
        self.batch16 = nn.BatchNorm2d(16)
        self.batch32 = nn.BatchNorm2d(32)
        self.batch128 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.2)

    # [conv>batchnomr>relu>maxpool]x2>dropout25->flatten>linear>batchnorm>relu>dropout50>linear
    def forward(self, x):
        if self.quantization:
            x = cs.quantization_method[self.quant_method](x, -2 ** (self.act_bit - 1) + 1, 2 ** (self.act_bit - 1) - 1)

        out = self.conv1(x)
        out = self.batch16(out)
        out = self.act(out)
        out = F.max_pool2d(out, 2)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.batch32(out)
        out = self.act(out)
        out = F.max_pool2d(out, 2)
        out = self.drop(out)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.batch128(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.dense2(out)
        return out

################################## CIFAR10 quantized with Brevitas ##################################
class model_CIFAR10_quant_brevitas(nn.Module):
    def __init__(self):
        super(model_CIFAR10_quant_brevitas, self).__init__()
        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 8

        self.alpha_coeff = 10.0

        self.quant_inp = qnn.QuantIdentity(bit_width=self.act_bit, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     bias=False, weight_bit_width=self.weight_bit, weight_quant=WeighQuant,
                                     bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     bias=False, weight_bit_width=self.weight_bit, bias_quant=BiasQuant,
                                     weight_quant=WeighQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU( bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)

        self.dense1 = qnn.QuantLinear(in_features=32*8*8, out_features=64, bias=True, weight_bit_width=self.weight_bit,
                                   weight_quant=WeighQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.dense2 = qnn.QuantLinear(in_features=64, out_features=10, bias=True,
                                     weight_bit_width=self.weight_bit,
                                     weight_quant=WeighQuant, bias_quant=BiasQuant, return_quant_tensor=False)

        #self.pool = qnn.QuantMaxPool2d(2)
        self.flatten = nn.Flatten()
        self.drop1 = qnn.QuantDropout(p=0.2)
        self.drop2 = qnn.QuantDropout(p=0.2)
        self.drop3 = qnn.QuantDropout(p=0.2)
        self.batch16 = nn.BatchNorm2d(16)
        self.batch32 = nn.BatchNorm2d(32)
        self.batch64 = nn.BatchNorm1d(64)
    def forward(self, x):
        out = self.quant_inp(x)
        out = self.conv1(out)
        out = self.batch16(out)
        out = self.relu1(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.batch32(out)
        out = self.relu2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.drop2(out)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.batch64(out)
        out = self.relu3(out)
        out = self.drop2(out)
        out = self.dense2(out)
        return out

################################## CIPHAR10 ResNet quantized  ##################################

class ResidualBlock_quant(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock_quant, self).__init__()
        self.conv1 = nn.Sequential(
            cs.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1,
                      act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                      quantization=self.quantization, quant_method=self.quant_method),
            cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization),
        )
        self.conv2 = nn.Sequential(
            cs.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1,
                      act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                      quantization=self.quantization, quant_method=self.quant_method),
            # cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization),
        )
        self.downsample = downsample
        self.relu = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization),
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_quant(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_quant, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)           # flattening the tensor,
        x = self.fc(x)

        return x
