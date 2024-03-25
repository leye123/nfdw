import cv2
import os

import numpy as np

import model
import torch
import torch.optim as opt
import gc
from tqdm import tqdm

file_path = 'result/result.npy'
data = np.load(file_path)
print(data.shape)