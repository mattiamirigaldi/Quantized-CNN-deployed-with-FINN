"""
The goal of this exercise is to maximise the accuracy of a given neural network model optimizing the training setup.

Rules:
- You can NOT change the neural network model, but you can change the learnable parameters initialization
- You can adjust the batch size according to the memory capacity of your processing unit
- You can NOT change the optimizer, but you can change its parameters
- You can change the epoch size
- You can change the pre-processing functions
- You can not change the neural network model

- The goal is to write a model that has the best tradeoff between accuracy, model parameters and model size. You will
  compare the model performance against one trained with the default script and one trained with an optimized script

- The score is evaluated as: (your model min class accuracy/default model min accuracy) * A +
                             (default epochs/your epochs) * B
- The coefficients are: A = 0.6, B = 0.4
- default min class accuracy = 5.9417, default epochs = 5
- default optimized min class accuracy = 96.0357, default optimized epochs = 3

The two default models, one trained without changing any parameter in this script and one trained by tuning the training
loop only (learning rate, data pre-processing) are provided in "saved_models", named exercise1_default.pth and
exercise1_default_optimized.pth respectively.
"""

if __name__ == '__main__':
    import torch, torchvision, copy
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import transforms
    import numpy as np
    from tqdm import tqdm
    from torch_neural_networks_library import default_model
    from find_num_workers import find_num_workers
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    Path("./runs/exercise_1").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one
    writer = SummaryWriter('runs/exercise_1')

    # TO NOTICE :
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    # transform_train = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize((0.5, ), (0.5,)),
    #                                      ])
    # transform_test = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize((0.5,), (0.5,)),
    #                                      ])
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])


    training_data = datasets.MNIST(root="data", train=True, download=False, transform=transform_train)
    test_data = datasets.MNIST(root="data", train=False, download=False, transform=transform_test)

    # NOTE : find the optimal batch size for your training setup. The batch size influences how much GPU or system memory is
    #       required, but also influences how fast the optimizer can converge to the optima. Values too big or too
    #       small will slow down your training or even cause a crash sometimes, try to find a good compromise. Use the
    #       average loss and iteration time displayed in the console during the training to tune the batch size.
    batch_size = 32

    #best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
    best_workers = 1   # change this number with the one from the previous function and keep using that for future runs

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())


    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break
    print(test_data.classes)

    dataiter = iter(copy.deepcopy(test_dataloader))
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image(str(batch_size)+'_mnist_images', img_grid)


    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = default_model()  # create model instance, initialize parameters, send to device

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(model)
    writer.add_graph(model, images)
    model.to(device)
    # Used to debugging summary(), delete if you want.
    params = sum([np.prod(p.size()) for p in model_parameters])
    memory = params * 32 / 8 / 1024 / 1024
    print("this model has ", params, " parameters")
    print("total weight memory is %.4f MB" %(memory))

    loss_fn = nn.CrossEntropyLoss()

    # NOTE : in Stochastic Gradient Descent optimizer different parameters can be changed like the learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)

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

            if batch % 1000 == 0:
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
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct

    # NOTE : change the epochs parameter to change how many times the model is trained over the entire dataset. How many
    #       epochs does your model require to reach the optima or oscillate around it? How many epochs does your model
    #       require to get past 80% accuracy? How many for 90%? How can you speed-up the training without increasing the
    #       epochs from the default value of 5?

    epochs = 25
    best_correct = 0
    best_model = []
    Path("./saved_models").mkdir(parents=True, exist_ok=True)
    print("Use $ tensorboard --logdir=runs/exercise_1 to access training statistics")

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
        }, "./saved_models/exercise1.pth")

        print("Saved PyTorch Model State to model.pth")

    writer.close()
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
    default_score = (lowest_class_accuracy/5.9417) * 0.6 + (5/epochs) * 0.4
    score = (lowest_class_accuracy/96.0357) * 0.6 + (3/epochs) * 0.4

    print("Score for this exercise against default training script = %.4f" %(default_score))
    print("Score for this exercise against optimized training script = %.4f" %(score))

