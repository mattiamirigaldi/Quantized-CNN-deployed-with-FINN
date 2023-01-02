# Quantized-CNN-deployed-with-FINN
In recent years, methods based on deep neural networks(DNNs) have become the standard approach
for complex problems as computer vision. However a good accuracy of CNNs comes at the cost of
high computational complexity, typically in the order of billions of multiply and accumulate (MAC)
operations for recent state of the art networks.
CNNs are usually trained on GPUs using floating points precision for the data (weights and activa-
tions), GPUs are very efficient when processing high amount of data in a massively parallelized way,
such as during the training, but are inherently energy hungry and infeasible for deployment on battery
powered devices. By contrast, FPGAs and ASIC are extremely efficient and can be designed around
the workload that will be processed at the edge.
A recent approach is to co-design both the CNN and its accelerator, carefully tailoring the com-
putation requirements and resources availability with the target task accuracy. Xilinx’s researchers
developed a framework called Brevitas, built upon Pytorch, to train low bit-width quantized CNNs
with the target accelerator in mind. The trained CNN model is then passed to another framework
called FINN (Xilinx again) which generates an accelerator based on a systolic array that can execute
the network efficiently, respecting the performance targets determined by the designer. The result is
a bitstream that can be mounted on a FPGA accelerator, providing an easy way to design and deploy
efficient CNNs.
In this project is shown the workflow of implementing a quantized convolutional neural network ac-
celerator on a Xilinx FPGA using a python-HLS toolchain. The hardware accelerator for the CNN
models is generated with FINN, a systolic array generator, and deployed on a Xilinx FPGA. The
toolchain is written in Python, C++ and HLS
