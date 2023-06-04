import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
import time
import numpy as np
from sklearn.metrics import classification_report

model_num = 6
num_epoch = 25
model_name = '6imbalance2_step0.2'
ten_crop = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sampling parameters
sampling_num = 2
sampling_rate = np.ones((model_num, 10))
name2num = {'비행기':0, '자동차':1, '새':2, '고양이':3,'사슴':4,'개':5,'개구리':6,'말':7, '배':8, '트럭':9}
weighted_cls = [['비행기','새','고양이'],
               ['비행기','자동차','배','트럭'],
               ['고양이','개','말'],
               ['사슴','개구리','말'],
               ['새','고양이','사슴','개구리'],
               ['자동차','배','트럭']]
weighted_cls = [[name2num[n] for n in weighted_cls[i]] for i in range(model_num)]
for i in range(model_num) :
    for j in weighted_cls[i]:
        sampling_rate[i][j] = sampling_num
sampling_rate = torch.tensor(sampling_rate,device=device)
weights = np.array([j for i in weighted_cls for j in i])
weight_sum = np.ones(10)
for i in weights :
    weight_sum[i] +=1
weight_sum = torch.tensor(weight_sum,device=device)


# Define the data transforms
transform_ten = transforms.Compose([
    transforms.Resize(256,interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(crop)) for crop in crops]))
    ])
transform_test = transforms.Compose([
    transforms.Resize(256,interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 test dataset
if ten_crop :
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_ten)
else:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Define the list of models for ensemble
models = []
correct = 0

total = 0
start = time.time()

for i in range(model_num):
    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', num_classes=10)
    model.load_state_dict(torch.load(f"./weights/%s_%d_%d.pth" % (model_name, num_epoch, i)))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the GPU
    models.append(model)

if ten_crop :
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            bs, ncrops, c, h, w = inputs.size()       
            outputs = torch.zeros(bs, 10).to(device)
            for s, model in enumerate(models):
                model_output = model(inputs.view(-1, c, h, w))  # Reshape the input to (bs*10, c, h, w)
                model_output = model_output.view(bs, ncrops, -1).mean(1)  # Average the predictions of the 10 crops
                model_output = model_output * sampling_rate[s]
                outputs += model_output
            outputs /= weight_sum
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

else :
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            bs, c, h, w = inputs.size()       
            outputs = torch.zeros(bs, 10).to(device)
            for s, model in enumerate(models):
                model_output = model(inputs)
                model_output = model_output * sampling_rate[s]
                outputs += model_output
            outputs /= weight_sum
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('====='+str(model_name)+'=====')
print('Accuracy : %2f %%' % (100 * correct / total))
print('Inference time : ', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))
