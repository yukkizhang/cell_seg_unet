#data_loader
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import torch

# cell segmentation dataset
# get the image and label
# train dataset
def get_train_data(dir):
    imgs = []
    n = int(len(os.listdir(dir)) / 2)
    train_imagelist = []
    train_labellist = []
    for i in os.listdir(dir):  #遍历整个文件夹
        path = os.path.join(dir, i)
        if os.path.isfile(path):  #判断是否为一个文件，排除文件夹
            if os.path.splitext(path)[1]=='.tif':
                train_imagelist.append(i)
            elif os.path.splitext(path)[1]=='.npy':
                train_labellist.append(i)
        # 如果将子文件也遍历的话，需要增加一个递归
        # elif os.path.isdir(path):
        #     newdir=path
        #     CrossOver(newdir,fl)
    for i in range(n):
        imgs.append((train_imagelist[i],train_labellist[i]))

    return imgs

directory = r"D:\2PM_2023\segmentation\data\00000008自动核质比训练集new-all\TrainDataset\StratumSpinosum\train"  #文件夹名称

# image, label=get_train_data(directory)
# print('train_dataset ok')


# npy文件读取与处理
# images = get_train_data(directory)
# n = int(len(images)/2)
# img_list = []
# label_list = []
# for i in range(n):
#     img_path,label_path = images[i]
#     img_path = os.path.join(directory, img_path)
#     label_path = os.path.join(directory, label_path)
#     train_image = Image.open(img_path)
#     train_label = np.load(label_path,allow_pickle=True).item()
#     mask = train_label['masks']
#     img, mask = np.squeeze(train_image), np.squeeze(mask)
#     img_list.append(img)
#     label_list.append(mask)




# test dataset
# fiber segmentation dataset
class TrainDataset(Dataset):
    def __init__(self, directory, transform=None, target_transform=None):
        # image, label=get_train_data(directory)
        # self.image = image
        # self.label = label
        imgs = get_train_data(directory)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,index):
        # img_path = os.path.join(directory, image)
        # label_path = os.path.join(directory, label)
        # x_path, y_path = self.imgs[index]
        img_path,label_path = self.imgs[index]
        img_path = os.path.join(directory, img_path)
        label_path = os.path.join(directory, label_path)
        train_image = Image.open(img_path)
        train_image = np.array(train_image)
        # train_label = Image.open(label_path)
        train_label = np.load(label_path,allow_pickle=True).item()

        mask = train_label['masks']

        train_image, mask = np.squeeze(train_image), np.squeeze(mask)

        train_image = train_image.astype(np.float32)
        # train_image = torch.from_numpy(train_image)
        mask = mask.astype(np.float32)
        # mask = torch.from_numpy(mask)

        if self.transform is not None:
            train_image = self.transform(train_image)
        if self.target_transform is not None:
            train_label = self.target_transform(mask)
        return train_image, train_label

    def __len__(self):
        # return len(self.image),len(self.label)
        return len(self.imgs)

# 加载数据集，并进行transform操作