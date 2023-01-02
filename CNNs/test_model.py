"""
Test script, load a model and verify class accuracy
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np

# TODO: import the correct neural network model
from torch_neural_networks_library import *

# TODO: change the model name, must be equal to the .pth file you generated, but without the extension .pth. Chance also
#       the model instance in the next line.

model = default_model()

# TODO: use the same transform function used for the test dataloader in your training script
transform = transforms.Compose([transforms.ToTensor()])

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,   # set to false if the dataset has been downloaded already
    transform=transform,
)


test_dataloader = DataLoader(test_data, batch_size=16)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
model.load_state_dict(torch.load("./saved_models/" + model_name + '.pth')['model_state_dict'])
model.to(device)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print(model)
params = sum([np.prod(p.size()) for p in model_parameters])
memory = params * 32 / 8 / 1024 / 1024
print("this model has ", params, " parameters")
print("total weight memory is %.4f MB" %(memory))
loss_fn = nn.CrossEntropyLoss()
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


_ = test(test_dataloader, model, loss_fn)
classes = test_data.classes
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for X, y in test_dataloader:
        images, labels = X.to(device), y.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

min_correct = [0,110]
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    if min_correct[1] >= int(accuracy):
        min_correct = [classname, accuracy]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

lowest_class_accuracy = min_correct[1]
print("Worst class accuracy is %.4f for class %s" %(min_correct[1], min_correct[0]))


