import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
import time

model_name = '6sel_imbalance_step0.2'

# hyper parameters
model_num = 6
total_epoch = 25
lr = 0.01
step_size = 0.2
gamma = 0.1

# Sampling parameters
sampling_num = 1
weighted_cls = [['비행기','새','고양이'],
               ['비행기','자동차','배','트럭'],
               ['고양이','개','말'],
               ['사슴','개구리','말'],
               ['새','고양이','사슴','개구리'],
               ['자동차','배','트럭']]

# Functions

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


# compute sampling rate
sampling_rate = np.zeros((model_num, 10))
name2num = {'비행기':0, '자동차':1, '새':2, '고양이':3,'사슴':4,'개':5,'개구리':6,'말':7, '배':8, '트럭':9}
weighted_cls = [[name2num[n] for n in weighted_cls[i]] for i in range(model_num)]
for i in range(model_num) :
    for j in weighted_cls[i]:
        sampling_rate[i][j] = sampling_num
    sampling_rate[i] /= sum(sampling_rate[i])

# Train n models
for s in range(model_num):
    seed_number = s
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data transforms
    transform_train = transforms.Compose([
        transforms.Resize(256,interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256,interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Sampling weight
    train_sampling = [sampling_rate[s][i] for i in [trainset[j][1] for j in range(len(trainset))]]
    train_sampler = torch.utils.data.WeightedRandomSampler(train_sampling,50000)
    test_sampling = [sampling_rate[s][i] for i in [testset[j][1] for j in range(len(testset))]]
    test_sampler = torch.utils.data.WeightedRandomSampler(test_sampling,10000)

    # Data Loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=0, sampler=train_sampler, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=0, sampler=test_sampler, pin_memory=True)

    # Define model
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)  # Move the model to the GPU

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
     # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
     # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epoch*0.4, eta_min=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_epoch*step_size, gamma=gamma)

    # Train
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

    print('Finished Training')

    # Save last model
    PATH = './weights/%s_%d_%d.pth' % (model_name, total_epoch, seed_number)
    torch.save(model.state_dict(), PATH)




