"""

Before you execute this exercise, you should read the quantization.py script and check the papers mentioned in there
In this exercise, you should re-write the models for Fashion MNIST and CIFAR10 using only quantized operators.
If you used normalization layers, have a look at the following paper https://arxiv.org/pdf/1712.05877.pdf
In deployment scenarios, where your CNN model has to be executed on a resource constrained device, floating points units
are not feasible and usually not available. To avoid using FP32 arithmetic, you should fold the normalization layer
as described in the aforementioned paper, in order to fuse the FP32 operation into the quantized convolutional layer
during the training phase. Check this link for more information on the formulas used to fold the batch norm into the
convolution weight and bias. Notice that the convolution bias is included in this formula, but it is not useful as the
conv layer is followed by the normalization. Thus, set the bias term to zero.
https://nathanhubens.github.io/fasterai/bn_folding.html

You should also check the brevitas repository and understand the syntax, how to load and save models from pytorch and
export them to a FINN format. In order to export the model, load the weights and assign them to the same neural network
written using brevitas classes, then test again the model to check that you get the same accuracy as the one trained
using the custom classes inside quantization.py.

Save and load torch models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
Brevitas page: https://github.com/Xilinx/brevitas
Brevitas gitter: https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
FINN page: https://finn.readthedocs.io/en/latest/index.html

Assignment:
- Rewrite the two CNNs you used previously for fashion MNIST and CIFAR10 using the custom layers provided in
  quantization.py and retrain them. Try to minimize the model size as much as possible and maximize task accuracy.
- If you decided to include batch normalization layers, fold them in the convolution operation
- Rewrite the two CNNs using Brevitas classes, import the weights and validate the task accuracy.
- Export the fashion MNIST model to FINN to be used in exercise 5, use the following code to export the model and
  follow the instructions below if you encounter any errors while executing it.

    # if FINN_save:
    #     if dataset_type == 'FashionMNIST':
    #         in_tensor = (1, 1, 28, 28)
    #     elif dataset_type == 'CIFAR10':
    #         in_tensor = (1, 3, 32, 32)
    #     else:
    #         exit("invalid dataset")
    #
    #     '''
    #     Due to some bugs inside the onnx manager of brevitas, a small edit is necessary to export models
    #     trained on a GPU-accelerated environment.
    #     Copy and paste the following lines inside brevitas\export\onnx\manager.py at line 89:
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         input_t = torch.empty(input_shape, dtype=torch.float).to(device)
    #     instead of the following line:
    #         input_t = torch.empty(input_shape, dtype=torch.float)
    #     The two functions below are supposed to save the model in a FINN-friendly format, however, are both not well
    #     documented and might fail for various reasons. If this happens, you should use the gitter page linked on the
    #     Brevitas repo to find the solution to your specific problem.
    #     Therefore, save the same model twice and we will try them both with the FINN notebooks.
    #     '''
    #     FINNManager.export(model.to("cpu"), input_shape=in_tensor, export_path=model_directory + model_name + tag + 'finn.onnx')
    #     BrevitasONNXManager.export(model.cpu(), input_shape=in_tensor, export_path=model_directory + model_name + tag + 'brevitas.onnx')
    #


Rules:
- You are free to do everything that you want, as long as the final CNN model is deployed fully quantized.
- The goal is still to write a model that has the best tradeoff between accuracy, model parameters and model size.
- You cannot use any FP32 operation within your model, all inputs and outputs must be quantized
- You can use batch normalization, but must be quantized as well and folded into the convolution
"""

"""here is an example CNN written using the custom functions. The training loop is the same as the one you used
   previously, so you only need to work on the CNN class and the custom functions."""
from torch import nn
import quantization as cs
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant
import torch.nn.functional as F

# TODO: use this example class to implement the quantized version of the CNNs you wrote for fashion MNIST and CIFAR10.
#  Remember: only the fashion MNIST CNN will be ported to FINN
class Exercise4_placeholder(nn.Module):
    def __init__(self):
        super(Exercise4_placeholder, self).__init__()
        # TODO: you can either use the same quantization for all layers and datatypes or use layer-wise mixed-precision
        #  i.e., each datatype of each layer has its own precision
        self.weight_bit = 32  # TODO: select an appropriate quantization for the weights
        self.act_bit = 32  # TODO: select an appropriate quantization for the activations
        self.bias_bit = 32  # TODO: select an appropriate quantization for the bias
        self.quant_method = None  # TODO: use either  'scale' or  'affine' and justify it

        # TODO: what is the alpha_coeff used for? What is a good initial value? How does the value change during training?
        self.alpha_coeff = 123.0

        self.quantization = True  # set to True to enable quantization, set to False to train with FP32

        # TODO: change these placeholder layers with the ones you used in the previous exercise
        self.conv = cs.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                              act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                              quantization=self.quantization, quant_method=self.quant_method)

        self.act = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)

        self.flatten = nn.Flatten()
        self.dense = cs.Linear(in_channels=32, out_channels=10, act_bit=self.act_bit, weight_bit=self.weight_bit,
                               bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

    def forward(self, inputs):
        if self.quantization:
            inputs = cs.quantization_method[self.quant_method](inputs, -2 ** (self.act_bit-1) + 1, 2 ** (self.act_bit-1) - 1)

        out = self.conv(inputs)
        out = self.act(out)

        out = F.adaptive_avg_pool2d(out, 1)  # PLACEHOLDER
        out = self.flatten(out)
        out = self.dense(out)

        return out

# TODO: check carefully the Brevitas git and gitter pages to check for information, requirements and common issues of
#  implementing a quantized CNN for the FINN engine. During exercise 5 you will try to port the fashion MNIST CNN to
#  FINN and run it on a FPGA, so you will have to follow the Brevitas and FINN guidelines to avoid as much as possible
#  unforeseen problems that are a consequence of your CNN design choices.

class Exercise4_placeholder_brevitas(nn.Module):
    def __init__(self):
        super(Exercise4_placeholder_brevitas, self).__init__()

        self.weight_bit = 32  # TODO: check the precision supported by Brevitas and the FINN engine
        self.act_bit = 32   # TODO: check the precision supported by Brevitas and the FINN engine
        self.bias_bit = 32  # TODO: notice how the Bias quantization is fixed in Brevitas. Which values can use use?
        # TODO: adjust the bias_bit in the custom CNN to the precision supported by Brevitas

        self.alpha_coeff = 123.0

        self.quant_inp = qnn.QuantIdentity(bit_width=self.act_bit, return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                    weight_bit_width=self.weight_bit, bias_quant=BiasQuant, return_quant_tensor=True)

        # TODO: in order to use efficient and high accuracy quantization, PACT and DoReFa methods are used during the
        #  training process. PACT is basically a ReLu with a learnable clipping parameter. You should implement a custom
        #  activation function for Brevitas, made of the QuantReLU class and an additional clipping function, to cut-off
        #  any value above the clipping parameter.
        self.relu = qnn.QuantReLU(bit_width=self.act_bit, return_quant_tensor=True)

        # TODO: check the BiasQuant options, see what is the supported precision for the bias.
        self.dense = qnn.QuantLinear(in_features=32, out_features=10, bias=True, weight_bit_width=self.weight_bit,
                                     bias_quant=BiasQuant, return_quant_tensor=True)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.conv(out)
        out = self.act(out)
        out = F.adaptive_avg_pool2d(out, 1)  # PLACEHOLDER
        out = self.dense(out)
        return out
