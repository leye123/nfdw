from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torch
import albumentations as A
from PIL import Image
from model import Model
import torch.nn as nn
from tqdm import trange
from glob import glob
import numpy as np
import random
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import os
from pre_data import get_list_img

class DeblurDataset(Dataset):
    def __init__(self, path, mode='train'):
        self.path = glob(path)
        self.mode = mode
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Transpose(p=0.3),
                A.Flip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomResizedCrop(height=224, width=224),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=224, width=224),
            ])
        self.blur = A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1)
        ], p=1)
        self.sizex = len(self.path)

    def mixup(self, inp_img, tar_img, mode='mixup'):
        mixup_index_ = random.randint(0, self.sizex - 1)

        mixup_inp_path = self.path[mixup_index_]

        mixup_inp_img = Image.open(mixup_inp_path).convert('RGB')

        mixup_inp_img = np.array(mixup_inp_img)

        mixup_inp_img = self.transform(image=mixup_inp_img)['image']
        mixup_tar_img = self.blur(image=mixup_inp_img)['image']

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        mixup_inp_img = TF.to_tensor(mixup_inp_img)
        mixup_tar_img = TF.to_tensor(mixup_tar_img)

        if mode == 'mixup':
            inp_img = lam * inp_img + (1 - lam) * mixup_inp_img
            tar_img = lam * tar_img + (1 - lam) * mixup_tar_img
        elif mode == 'cutmix':
            img_h, img_w = 224, 224

            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)

            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)

            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))

            inp_img[:, y0:y1, x0:x1] = mixup_inp_img[:, y0:y1, x0:x1]
            tar_img[:, y0:y1, x0:x1] = mixup_tar_img[:, y0:y1, x0:x1]

        return inp_img, tar_img

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        img = Image.open(self.path[item])
        img = np.array(img)

        img = self.transform(image=img)['image']
        blur = self.blur(image=img)['image']

        img = TF.to_tensor(img)
        blur = TF.to_tensor(blur)

        if self.mode == 'train':
            if item > 0 and item % 3 == 0:
                if random.random() > 0.5:
                    img, blur = self.mixup(img, blur, mode='cutmix')
                else:
                    img, blur = self.mixup(img, blur, mode='mixup')

        filename = os.path.basename(self.path[item])
        return blur, img, filename

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(size=224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]),
# }

batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 2e-4
num_epochs = 100

'''读取数据'''
# train_path = get_list_img('new_data/train1/train')
# test_path = get_list_img('new_data/test1/test')
# train_dataset = DeblurDataset(train_path)
# test_dataset = DeblurDataset(test_path)
train_dataset = DeblurDataset('new_data/train1/train/**.jpg')
test_dataset = DeblurDataset('new_data/test1/test/**.jpg')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print(train_loader)

''' 定义模型，优化器，损失函数'''
mymodel = Model()
mymodel.to(device)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(mymodel.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, num_epochs, eta_min=1e-6)

best_psnr = 0
best_epoch = 0
for epoch in trange(num_epochs):
    for inputs, labels, *extra_data in train_loader:
        # 确保 inputs 和 labels 都是 Tensor 类型
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)  # 如果这里报错，则说明 inputs 或 labels 不是 Tensor
            # labels = labels.unsqueeze(0).unsqueeze(1).expand(1, 244, 244)
            # labels = torch.argmax(labels,dim=1)
            # print(inputs.shape)
            # print(labels.shape)
            # 前向传播
            outputs = mymodel(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels) + (1 - structural_similarity_index_measure(outputs, labels)) * 0.1

            # 反向传播和优化
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

        scheduler.step()

    with torch.no_grad():
        psnr = 0
        ssim = 0
        for images, labels, *extra_data in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = mymodel(images)

            psnr += peak_signal_noise_ratio(outputs, labels)
            ssim += structural_similarity_index_measure(outputs, labels)

        psnr = psnr.item() / len(test_loader)
        ssim = ssim.item() / len(test_loader)

        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            torch.save(mymodel.state_dict(), f'{epoch}.pth')

        print(f'PSNR: {psnr}')
        print(f'SSIM: {ssim}')
        print(f'Epoch: {epoch}')
        print(f'Best PSNR: {best_psnr} at epoch {best_epoch}')
