import os
import cv2
from PIL import Image

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

def get_list_img(directory):    #获取图片地址
    # 初始化一个空列表用于存储图片路径
    image_paths = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 只处理.jpg文件
            if file.endswith('.jpg'):
                # 构建完整文件路径
                full_path = os.path.join(root, file)
                # 将路径添加到列表中
                image_paths.append(full_path)
    return image_paths

if __name__ =='__main__':
    save_path = 'new_data/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_path = 'data/four/four/save_belt'
    height = 224
    weight = 224
    img_resize(file_path,save_path,height,weight)