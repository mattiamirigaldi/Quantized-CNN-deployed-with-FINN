"""
In this exercise you will port the edits that you did on the "exercise_2.py" script and train a copy of your previous
model on the FashionMNIST dataset, which is a MNIST-like dataset of 28*28 gray images of clothes, taken from Zalando
ads. This is to test the CNN on a slightly more challenging dataset. You should reuse the same training parameters, but
change the input normalization (MNIST and FashionMNIST have a differend std and mean).
Make a copy of the previous NN model and modify it in order to have a worst class accuracy of minimum 70%

Rules:
- You can adjust the batch size according to the memory capacity of your processing unit
- You can NOT change the optimizer, but you can change its parameters
- You can change the epoch size
- You can change the pre-processing functions
- You can fully customize the class NeuralNetwork, thus your CNN model
- You must use at most 6 layers with learnable parameters (do not use norm layers, they are not necessary and count as
  layers with learnable parameters, you will use them in the next exercises)
- The goal is to write a model that has the best tradeoff between accuracy, model parameters and model size.

- The score is evaluated as: (default model size/your model size) * A +
                             (your model min class accuracy/default model min accuracy) * B +
                             (default model parameters/your model parameters) * C +
                             (default epochs/your epochs) * D

- The coefficients are: A = 0.2, B = 0.3, C = 0.3, D= 0.2
- default model: size = 2.555MB, min class accuracy = 1.6, parameters = 669706, epochs = 5
- default optimized model: size = 2.555MB, min class accuracy = 68.40, parameters = 669706, epochs = 5
- optimized CNN model: size = 0.1233 MB, min class accuracy = 70.5, parameters = 32314, epochs = 5
The two default models, one trained without changing any parameter in this script and one trained by tuning the training
loop only (learning rate, data pre-processing) are provided in "saved_models", named exercise1_default.pth and
exercise1_default_optimized.pth respectively. The optimized CNN model is provided for reference and can be found in
"saved_models" as "exercise2_cnn.pth"
"""
if __name__ == '__main__':
    import torch, torchvision, copy
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import transforms
    import numpy as np
    from tqdm import tqdm
    from torch_neural_networks_library import default_model, model_ex2b
    from ex4_MNIST import model_MNIST_quant
    from pathlib import Path
    from find_num_workers import find_num_workers
    from torch.utils.tensorboard import SummaryWriter
    from brevitas.export import FINNManager, BrevitasONNXManager
    import os

    Path("./runs/exercise_2bis").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one

    writer = SummaryWriter('runs/exercise_2bis')
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    training_data = datasets.FashionMNIST(root="data", train=True, download=False, transform=transform_train)
    test_data = datasets.FashionMNIST(root="data", train=False, download=False, transform=transform_test)

    batch_size = 32

    # best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
    best_workers = 2

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())


    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break
    dataiter = iter(copy.deepcopy(test_dataloader))
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image(str(batch_size)+'_FashionMNIST_images', img_grid)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Note: Model in torch_neural_networks_library.py
    # model = model_ex2b()  # create model instance, initialize parameters, send to device
    model = model_MNIST_quant()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # print(model)
    writer.add_graph(model, images)
    model.to(device)
    # Used to debugging summary(), delete if you want.
    params = sum([np.prod(p.size()) for p in model_parameters])
    memory = params * 32 / 8 / 1024 / 1024
    print("this model has ", params, " parameters")
    print("total weight memory is %.4f MB" %(memory))

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)

    def get_argMax_vector(values):
        #print("values:", values)
        #print("values length:",len(values))
        #print("first value vector length:", len(values[0]))
        argMax_vector=[]
        for i in range(len(values[0])):
            max_score = 0
            index = 0
            for j in range(len(values[0][i])):
                to_check = values[0][i][j]
                if to_check > max_score:
                    max_score = values[0][i][j]
                    index = j
                argMax_vector.append(index)
        # print(argMax_vector)
        return argMax_vector




    def train(dataloader, model, loss_fn, optimizer, epoch):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                writer.add_scalar('training loss', loss / 1000, epoch * len(dataloader) + batch)
        return loss

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                #pred = get_argMax_vector(pred)
                # print(pred)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                #correct += ( pred[0].argmax(1) == y).type(torch.float).sum().item()
                #correct += (np.argmax(pred.logits, axis=0) == y).type(torch.float).sum().item()
                #print(correct)
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct

    epochs = 5
    best_correct = 0
    best_model = []
    Path("./saved_models").mkdir(parents=True, exist_ok=True)
    print("Use $ tensorboard --logdir=runs/exercise_2bis to access training statistics")

    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_dataloader, model, loss_fn, optimizer, t)
        current_correct = test(test_dataloader, model, loss_fn)
        writer.add_scalar('test accuracy', current_correct, t)
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'test_acc': current_correct,
        }, "./saved_models/exercise2bis.pth")
        print("Saved PyTorch Model State to model.pth")

    writer.close()
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

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

    default_score = (2.555/memory) * 0.2 + (lowest_class_accuracy/1.6) * 0.3 + (669706.0/params) * 0.3 + (5/epochs) * 0.2
    optimized_score = (2.555/memory) * 0.2 + (lowest_class_accuracy/68.40) * 0.3 + (669706.0/params) * 0.3 + (5/epochs) * 0.2


    print("Score for this exercise against default model from exercise 1 = %.4f" %(default_score))
    print("Score for this exercise against optimized training script from exercise 1 = %.4f" %(optimized_score))

    in_tensor = (1, 1, 28, 28)
    print(os.getcwd())
    build_dir = os.getcwd() + "/finn_model/"
    model_name = 'model_MNIST_quant_brevitas_'
    # FINNManager.export(model.to("cpu"), input_shape=in_tensor, export_path=build_dir + model_name + 'UintAct_finn_v4.onnx')
    # BrevitasONNXManager.export(model.cpu(), input_shape=in_tensor, export_path=build_dir + model_name + 'UintAct_brevitas_v4.onnx')

    """
    Hints:
    1- a small learning rate is too slow at the beginning of the training process, a big one will not grant convergence as 
       the training progress
    2- avoid using too many linear layers, they are over-parametrized for this task, try using other layers
    3- if necessary, use large filters in the first layers only
    4- use less channels in the first layers, more channels in the last ones
    5- template for CONV layer is nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(...), stride=..., padding=..., bias =...)
       you need to define these parameters for each Conv2d instance, do not use default values even if are the same as yours
    6- pay attention to the dimensioning of input-output spatial dimensions, for a single dimension (or 2 dimension in case
       of square images) the formula is out = floor( (in - kernel + 2 * padding) / stride ) + 1
    """

