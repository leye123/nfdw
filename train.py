from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize
import torch
from model import Model
import torch.nn as nn
from tqdm import trange



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3
num_epochs = 10

'''读取数据'''
train_dataset = ImageFolder(root='new_data/train1', transform=data_transforms['train'])
test_dataset = ImageFolder(root='new_data/test1', transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print(train_loader)

''' 定义模型，优化器，损失函数'''
mymodel = Model()
mymodel.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)


for epoch in trange(num_epochs):
    for inputs, labels in train_loader:
        # 确保 inputs 和 labels 都是 Tensor 类型
        inputs, labels = inputs.to(device), labels.to(device)  # 如果这里报错，则说明 inputs 或 labels 不是 Tensor
        # labels = labels.unsqueeze(0).unsqueeze(1).expand(1, 244, 244)
        # labels = torch.argmax(labels,dim=1)
        print(inputs.shape)
        print(labels.shape)
        # 前向传播
        outputs = mymodel(inputs)
        print(outputs.shape)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = mymodel(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')