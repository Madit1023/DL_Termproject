import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
import time
import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


# hyper parameters
model_num = 3
total_epoch = 60
lr = 0.1
model_name = 'gaussian5_labelsmooth'
step_size = 0.2
gamma = 0.1
seed_num = 37


def train():
    model.train()
    running_loss = 0.0
            
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('loss (iter %d / %d): %.3f' % (i + 1, len(trainloader), running_loss / 100))
            running_loss = 0.0   

      
                    
def val():
    model.eval()
    # Test the model
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Validation Accuracy : %2f %%' % (100 * correct / total))
    print('Validation     Loss : %3f' % (test_loss / len(testloader)))


for s in range(model_num):
    print('************ model %d ************' % (s))
    seed_number = 10 * s + seed_num
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the data transforms
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.GaussianBlur(5,1),
        transforms.RandomHorizontalFlip(p = 0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.GaussianBlur(5,1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)  # Move the model to the GPU

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
     # optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epoch*step_size, eta_min=1e-6)
     # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_epoch*step_size, gamma=gamma)


    # Train the model
    for epoch in range(total_epoch):
        print("============= Epoch %d =============" % (epoch+1))
        start = time.time()
        train()
        print("training time : ", time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))
        print("------------ Validation ------------")
        start = time.time()
        val()
        print("validation time : ", time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))
        print("")
        scheduler.step()

    print('Finished Training\n')

    # Save the checkpoint of the last model
    PATH = './weights/%s_%d_%d.pth' % (model_name, total_epoch, seed_number)
    torch.save(model.state_dict(), PATH)




