# to see the Mathematics feature of the initial image and human-made label mask
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.io as io

img_dir = r"D:\2PM_2023\segmentation\data\00000008自动核质比训练集new-all\TrainDataset\GranularLayer\train\cytoplasm"

images = []
labels = []
n = int(len(os.listdir(img_dir)) / 2)

for i in os.listdir(img_dir):
    path = os.path.join(img_dir, i)
    if os.path.isfile(path):
        if os.path.splitext(path)[1] == '.tif':
            images.append(i)
        elif os.path.splitext(path)[1] == '.npy':
            labels.append(i)

image_num = len(images)
label_num = len(labels)


for im in range(label_num):
    label_path = os.path.join(img_dir,labels[im])
    img = np.load(label_path,allow_pickle=True).item()

    outlines = img['outlines']
    colors = img['colors']
    mask = img['masks']
    img_img = img['img']


    # show the information of image in a plot
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(outlines,cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(colors,cmap='gray')
    # plt.subplot(2,2,3)
    # plt.imshow(mask,cmap='gray')
    # plt.subplot(2,2,4)
    # plt.imshow(img_img,cmap='gray')
    # print('ok')

    # plt.show()
    # plt.imshow(mask)
    # plt.show()

    io.imsave(r'D:\2PM_2023\segmentation\self_seg\label\outlines/' + str(im) + '_outlines.tif',outlines)
    io.imsave(r'D:\2PM_2023\segmentation\self_seg\label\original_mask/' + str(im) + '_mask.tif',mask)

    # convert mask into black/white binary
    row, col = mask.shape[0], mask.shape[1]
    for i in range(row):
        for j in range(col):
            if mask[i][j] == 0:
                mask[i][j] = 0
            else:
                mask[i][j] = 255
    # plt.imshow(mask,cmap='gray')
    # plt.show()
    io.imsave(r'D:\2PM_2023\segmentation\self_seg\label\new_mask/' + str(im) + '_newmask.tif',mask)

    # add the outlines
    # mask_outline = mask + outlines
    # plt.imshow(mask_outline,cmap='gray')
    # plt.show()
    # print('outlines')

    


