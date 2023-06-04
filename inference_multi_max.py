import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
import time
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

model_num = 9
num_epoch = 60
model_name = ['gaussian_labelsmooth','gaussian','labelsmooth']
ten_crop = True
seed_num = [0,1,2,0,1,2,0,1,2,0,1,2,27,28,29]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data transforms
transform_ten = transforms.Compose([
    transforms.Resize(256),
    transforms.GaussianBlur(5,1),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(transforms.ToTensor()(crop)) for crop in crops]))
    ])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.GaussianBlur(5,1),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the CIFAR-10 test dataset
if ten_crop :
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_ten)
else:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)


# Define the list of models for ensemble
models = []
correct = 0
total = 0
start = time.time()

for i in range(model_num):
    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', num_classes=10)
    model.load_state_dict(torch.load(f"./weights/%s_%d_%d.pth" % (model_name[i//3], num_epoch, seed_num[i])))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the GPU
    models.append(model)


if ten_crop :
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            bs, ncrops, c, h, w = inputs.size()
            outputs = torch.zeros(bs, 10).to(device)  # Initialize the output tensor with zeros
            for model in models:
                model_output = model(inputs.view(-1, c, h, w))  # Reshape the input to (bs*10, c, h, w)
                model_output = model_output.view(bs, ncrops, -1)
                crop_mean = model_output.mean(1)  # Take the mean prediction across crops
                outputs += crop_mean
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

else :
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            bs, c, h, w = inputs.size()       
            outputs = torch.zeros(bs, 10).to(device)  # Initialize the output tensor with zeros
            for model in models:
                model_output = model(inputs)  # Reshape the input to (bs*10, c, h, w)
                outputs += model_output
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(model_name)
print('Accuracy : %2f %%' % (100 * correct / total))
print('Inference time : ', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))
