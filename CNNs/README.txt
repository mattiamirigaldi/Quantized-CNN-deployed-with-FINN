Exercise 0: training and test loop tutorial
Exercise 1: training loop optimization for a 2-layer network.
Exercise 2: design of a convolutional neural network (CNN) for digit classification
Exercise 2bis: design of a CNN for fashion MNIST
Exercise 3: design of a CNN for image classification on CIFAR10, learn to reduce over-fitting and vanishing gradients
Exercise 4: quantization of previous CNN models with fake-quantize methods, port to brevitas and export to FINN
Exercise 5: FINN tutorials, deployment of a small CNN to a custom accelerator generated with FINN

Report instructions:
After completing the exercises, you should write the report of the special project using the same latex template of the
ISA laboratories.
You should organize the report in chapters, one for each exercise (from 1 to 5) and use sections to divide the problem
statement from the approach and results.
In the report you should include:
 - For exercise 1 through 4 the code of CNN models used during the experiments, plus training hyper-parameters, pre- and
   post-processing of data, special constructs/classes/functions that you used outside or within the CNN model during
   the training process
 - For exercise 5 only, you should include the code of the jupyter notebooks that you had to modify in order to port
   your model from FINN onto the ZCU104 development board. You must include a table with all hardware metrics extracted
   from FINN.
 - A critical analysis of the approach used to solve the exercise. You must reference the articles/websites that you
   used in order to make a particular design choice. You can describe the issues and solutions that you encountered
   during the design process.

Report deadline:
None, you can submit it when you are ready. The discussion will be held with me and Professor Maurizio Martina. The
score will be registered after the oral exam.