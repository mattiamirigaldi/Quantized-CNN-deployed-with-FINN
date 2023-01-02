import torch
import os
import brevitas.onnx as bo
from ex4_MNIST import model_MNIST_quant_brevitas
from brevitas.export import FINNManager, BrevitasONNXManager

#The MNIST model is loaded with the parameters that
#have reached the best accuracy results.
model_name = "exercise2"
tag ="bis"

# checkpoint_path= "./saved_models/exercise2bis.pth"
model = model_MNIST_quant_brevitas()
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(torch.load("./saved_models/" + model_name + tag + '.pth',map_location=torch.device('cpu'))['model_state_dict'], strict=True)

# MNIST quantized with brevitas model is exported to ONNX
model.eval()
build_dir = os.getcwd()+"/finn_model/"
print(build_dir)
in_tensor = (1, 1, 28, 28)
FINNManager.export(model.to("cpu"), input_shape=in_tensor, export_path='MNIST_quant_finn.onnx')
BrevitasONNXManager.export(model.cpu(), input_shape=in_tensor, export_path='MNIST_quant_brevitas.onnx')