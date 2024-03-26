import os
import cv2
import torch
from PIL import Image
import numpy as np
from test import write_image
from model import Model

img_path = 'new_data/44010000V44010078441322020171_1703560295172_0022.jpg'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
''' 加载模型 '''
model = Model()
model.load_state_dict(torch.load('1.pth'))
model = model.to(device)
model.eval()
or_img = cv2.imread(img_path)
img = torch.tensor(or_img, dtype=torch.float).cuda()
img = img.unsqueeze(0)
img = img.permute([0, 3, 1, 2])
print(or_img.dtype)
outputs = model(img).cpu()
out_img = outputs.detach().numpy()
out_img = np.uint8(out_img)
out_img = out_img.reshape((out_img.shape[2],out_img.shape[3],out_img.shape[1]))
# cv2.imshow('imshow', outputs)
write_image(cv2.add(out_img,or_img))




