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

################################## MNIST NN quantized  ##########à####################
class model_MNIST_quant(nn.Module):
    def __init__(self):
        super(model_MNIST_quant, self).__init__()
        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 8
        self.quant_method = 'scale'
        self.alpha_coeff = 10.0

        self.quantization = False  # set to True to enable quantization, set to False to train with FP32

        self.conv1 = nn.Sequential(
            cs.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                      act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                      quantization=self.quantization, quant_method=self.quant_method),
            cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            cs.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                      act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                      quantization=self.quantization, quant_method=self.quant_method),
            cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        # Fully connected layer, output 10
        self.out = cs.Linear(in_channels=32*7*7, out_channels=10, act_bit=self.act_bit, weight_bit=self.weight_bit,
                               bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

    def forward(self, x):
        if self.quantization:
            x = cs.quantization_method[self.quant_method](x, -2 ** (self.act_bit-1) + 1, 2 ** (self.act_bit-1) - 1)

        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        out = self.out(x)
        return out

################################## MNIST NN quantized with BREVITAS ##########à####################
class model_MNIST_quant_brevitas(nn.Module):
    def __init__(self):
        super(model_MNIST_quant_brevitas, self).__init__()
        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 8

        self.alpha_coeff = 123.0

        self.quant_inp = qnn.QuantIdentity(bit_width=self.act_bit, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                    weight_bit_width=self.weight_bit, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=self.act_bit, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     weight_bit_width=self.weight_bit, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=self.act_bit, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(in_features=32*7*7, out_features=10, bias=True, weight_bit_width=self.weight_bit,
                                     bias_quant=BiasQuant)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out

#####################################  MNIST NN quantized with BREVITAS ported to FINN  ####################################
class model_MNIST_quant_brevitasToFinn_UintAct(nn.Module):
    def __init__(self):
        super(model_MNIST_quant_brevitasToFinn_UintAct, self).__init__()
        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 8

        self.alpha_coeff = 10.0

        self.quant_inp = qnn.QuantIdentity(bit_width=self.act_bit, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(in_channels=1, out_channels=3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     bias=False, weight_bit_width=self.weight_bit, weight_quant=WeighQuant,
                                     bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(in_channels=3, out_channels=8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     bias=False, weight_bit_width=self.weight_bit, bias_quant=BiasQuant,
                                     weight_quant=WeighQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.conv3 = qnn.QuantConv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     bias=False, weight_bit_width=self.weight_bit, bias_quant=BiasQuant,
                                     weight_quant=WeighQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)

        self.fc1 = qnn.QuantLinear(in_features=16*7*7, out_features=10, bias=True, weight_bit_width=self.weight_bit,
                                   weight_quant=WeighQuant, bias_quant=BiasQuant, return_quant_tensor=False)
        #self.pool = qnn.QuantMaxPool2d(2)
        self.flatten = nn.Flatten()
    def forward(self, x):
        out = self.quant_inp(x)
        out = self.conv1(out)
        out = self.relu1(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        #out = self.quant_inp(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv3(out)
        out = self.relu3(out)
        #out = F.max_pool2d(out, 2)
        out = self.flatten(out)
        out = self.fc1(out)
        return out
############################################## LeNet5 network #########################################

total_bits = 8  # width for weights and activations
n = 7  # fractional part


class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_channels=1,
                                     out_channels=20,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=total_bits,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT,
                                   bit_width=8,
                                   max_val=1 - 1 / 128.0,
                                   restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   scaling_impl_type=ScalingImplType.CONST)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = qnn.QuantConv2d(in_channels=20,
                                     out_channels=50,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)

        self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT,
                                   bit_width=8,
                                   max_val=1 - 1 / 128.0,
                                   restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   scaling_impl_type=ScalingImplType.CONST)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        """
        # for 32-bit precision FC layers
        self.fc1   = nn.Linear(7*7*50, 500)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(500,10)
        """

        # for fixed-point precision FC layers
        self.fc1 = qnn.QuantLinear(7 * 7 * 50, 500,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=32,
                                   weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   weight_scaling_impl_type=ScalingImplType.CONST,
                                   weight_scaling_const=1.0)

        self.relu3 = qnn.QuantReLU(quant_type=QuantType.INT,
                                   bit_width=8,
                                   max_val=1 - 1 / 128.0,
                                   restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   scaling_impl_type=ScalingImplType.CONST)

        self.fc2 = qnn.QuantLinear(500, 10,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=8,
                                   weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                   weight_scaling_impl_type=ScalingImplType.CONST,
                                   weight_scaling_const=1.0)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

#############################################################################################################
class model_Flavia_brevitas(nn.Module):
    def __init__(self):
        super(model_Flavia_brevitas, self).__init__()
        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 8

        self.alpha_coeff = 10.0

        self.quant_inp = qnn.QuantIdentity(bit_width=self.act_bit, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(in_channels=1, out_channels=3, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4),
                                     bias=False, weight_bit_width=self.weight_bit, weight_quant=WeighQuant,
                                     bias_quant=BiasQuant, return_quant_tensor=True)
        self.act1 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(in_channels=3, out_channels=5, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
                                     bias=False, weight_bit_width=self.weight_bit, bias_quant=BiasQuant,
                                     weight_quant=WeighQuant, return_quant_tensor=True)
        self.act2 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.conv3 = qnn.QuantConv2d(in_channels=5, out_channels=10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     bias=False, weight_bit_width=self.weight_bit, bias_quant=BiasQuant,
                                     weight_quant=WeighQuant, return_quant_tensor=True)
        self.act3 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.conv4 = qnn.QuantConv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                     bias=False, weight_bit_width=self.weight_bit, bias_quant=BiasQuant,
                                     weight_quant=WeighQuant, return_quant_tensor=True)
        self.act4 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)
        self.conv5 = qnn.QuantConv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                     bias=False, weight_bit_width=self.weight_bit, bias_quant=BiasQuant,
                                     weight_quant=WeighQuant, return_quant_tensor=True)
        self.act5 = qnn.QuantReLU(
            bit_width=self.act_bit, act_quant=ActQuant, return_quant_tensor=True)

        self.flatten = nn.Flatten()
        self.dense = qnn.QuantLinear(in_features=32*7*7, out_features=10, bias=True, weight_bit_width=self.weight_bit,
                                   weight_quant=WeighQuant, bias_quant=BiasQuant, return_quant_tensor=False)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv5(out)
        out = self.act5(out)
        out = self.flatten(out)
        out = self.dense(out)
        return out

