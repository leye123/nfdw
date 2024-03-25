import os
import cv2

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

if __name__ =='__main__':
    save_path = 'new_data/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_path = 'data/four/four/save_belt'
    height = 224
    weight = 224
    img_resize(file_path,save_path,height,weight)