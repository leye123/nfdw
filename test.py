import cv2
import os

import numpy as np

import model
import torch
import torch.optim as opt
import gc
from tqdm import tqdm


def read_img(file_path):
    for file_name in os.listdir(file_path):
        print(file_name)
        img = cv2.imread(file_path + '\\' + file_name)
        print(img)

def img_resize(file_path,out_path,new_w,new_h):
    for file_name in os.listdir(file_path):
        print(file_name)
        img = cv2.imread(file_path + '/' + file_name)
        out_img = cv2.resize(img,(new_w,new_h))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_img_p = os.path.join(out_path + '/' + file_name)
        cv2.imwrite(out_img_p,out_img)
        print("{}  save sucessful".format(file_name))

def count_files(directory):
    num_files = 0

    for root, dirs, files in os.walk(directory):
        num_files += len(files)

    return num_files

def write_image(processed_image):
    if processed_image.shape[-1] == 3:  # 对于彩色图像

        if processed_image.dtype == np.uint8:  # 如果已经是uint8格式

            # 如果处理过程中色彩空间变为BGR，而你需要保存为RGB格式

            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            cv2.imwrite('output_rgb.jpg', rgb_image)

        else:

            # 如果需要转换为uint8范围以适应JPEG等格式

            normalized_image = (processed_image * 255).astype(np.uint8)

            cv2.imwrite('output_normalized.jpg', normalized_image)

''' resize_img'''
# save_path = 'new_data'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# file_path = 'data/four/four/save_belt'
# height = 640
# weight = 360
# img_resize(file_path,save_path,height,weight)

#
if __name__=='__main__':

    ''' run code'''
    # file_path = 'new_data'
    file_path = './new_data'
    data_l = count_files(file_path)
    print("file_num:{}".format(data_l))
    model_my = model.Model().cuda()
    optim = opt.Adam(model_my.parameters(), lr=0.01)
    data = []
    result = []
    height = 640
    weight = 360
    with tqdm(data_l) as pbar:
        for file_name in os.listdir(file_path):
            optim.zero_grad()
            print(file_name)
            pbar.set_description("file_num")
            img = cv2.imread(file_path + '/' + file_name)
            img = torch.tensor(img, dtype=torch.float).cuda()
            # img = img.unsqueeze(0)
            # img = img.permute([0, 3, 1, 2])
            with torch.no_grad():
                out = model_my(img).cpu()
                out_img = np.array(out)
                '''normalize'''
                max_o = out_img.max()
                out_img = out_img*255/max_o
                out_img = np.uint8(out_img)
                out_img = out_img.reshape((out_img.shape[2],out_img.shape[3],out_img.shape[1]))
                out_img2 = 0
                for i in range(3):
                    for j in range(3):
                        H = (i+1)*210
                        W = (j+1)*120
                        resized_img = cv2.resize(out_img[i:H,j:W], dsize=(height,weight), interpolation=cv2.INTER_LINEAR)
                        out_img2 = cv2.add(resized_img , out_img2)
                out_img2 = cv2.add(out_img2,img)
                cv2.imshow('imshow', out_img2)
                write_image(out_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            data.append(img)
            pbar.update(1)
