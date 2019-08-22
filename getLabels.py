#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation,graph, data, io, color
import torch.nn.init
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
import os
from os.path import join
from imageio import imread, imwrite

# use_cuda = torch.cuda.is_available()
# parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
# parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
#                         help='number of superpixels')
# parser.add_argument('--compactness', metavar='C', default=100, type=float,
#                         help='compactness of superpixels')
# parser.add_argument('--input', metavar='FILENAME',
#                         help='input image file name', required=True)
# parser.add_argument('--scale', metavar='S', default=10, type=int,
#                         help='number of superpixels')
# parser.add_argument('--min_size', metavar='MS', default=500, type=int,
#                         help='min number of pixels')
#     # S更高意味着更大的集群
# parser.add_argument('--re_scale', metavar='S', default=20, type=int,
#                         help='re area')
#     # 最小的组件的大小。强制使用后处理
# parser.add_argument('--re_min_size', metavar='MS', default=100, type=int,
#                         help='re min number of pixels')
#
# args = parser.parse_args()
sp_boundary = join(os.getcwd(), 'sp_result', 'reseg',"boundary")
sp_color = join(os.getcwd(), 'sp_result', 'reseg',"color")
if not os.path.exists(sp_boundary):
    os.makedirs(sp_boundary)
if not os.path.exists(sp_color):
    os.makedirs(sp_color)

def get_labels(input,scale = 10,min_size = 500,re_scale = 20 ,re_min_size = 100):
    # load image
    im = cv2.imread(input)  # (321,481,3)
    filename = os.path.splitext(os.path.split(input)[1])[0]

    im_val = []
    im_val = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    im_val_temp = im_val
    im_val = np.array(im_val).flatten()

    # 每个点的像素值（一维）
    im_val = []
    im_val = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    im_val = np.array(im_val).flatten()

    # flez
    labels = segmentation.felzenszwalb(im, scale=scale, sigma=0.8, min_size=min_size)
    boundary = segmentation.mark_boundaries(im, labels, (1, 0, 0))
    # imsave('boundary_%d.png'%args.num_superpixels, boundary)
    labels_temp = labels
    # ave_position,red_average,green_average,blue_average,position = function.init_sp(im,labels)
    labels = labels.reshape(im.shape[0] * im.shape[1])  # 分割后每个超像素的Sk值
    u_labels = np.unique(labels)  # 将Sk作为标签
    l_inds = []  # 每i行表示Si超像素中每个像素的编号
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    mean_list = []
    var_list = []
    std_list = []
    std_m_v = []
    for i in range(len(l_inds)):
        labels_per_sp = im_val[l_inds[i]]
        # 求均值
        arr_mean = np.mean(labels_per_sp)
        # 求方差
        arr_var = np.var(labels_per_sp)
        # 求标准差
        arr_std = np.std(labels_per_sp, ddof=1)

        #
        arr_s = arr_mean / (arr_mean ** 2)

        mean_list.append(arr_mean)
        var_list.append(arr_var)
        std_list.append(arr_std)
        std_m_v.append(arr_s)


    max_val = max(var_list)
    min_val = min(var_list)
    flag = (max_val + min_val) / 2.0

    # print("方差均值：%s" % (flag))

    liqun = []
    liqun_position = []
    for lq in range(len(std_m_v)):
        if (var_list[lq] > flag):
            liqun.append(std_m_v[lq])
            liqun_position.append((lq))  # 离群的位置

    badimage = np.array(np.ones([im.shape[0], im.shape[1]])) * 225
    badimage_rgb = np.array(np.zeros(im.shape))
    img = imread(input)
    for k in liqun_position:
        pos_list = l_inds[k]
        for pos in pos_list:
            m = int(pos / im.shape[1])
            n = int(pos % im.shape[1])
            badimage[m, n] = im_val_temp[m, n]
            badimage_rgb[m, n] = img[m, n]

    labels_bad = segmentation.felzenszwalb(badimage_rgb, scale=re_scale, sigma=0.8, min_size=re_min_size)

    labels_bad = labels_bad + labels_temp+1
    # 分割以颜色显示的颜色盘
    # label_colours = np.random.randint(255, size=(labels_bad.shape[0] * labels_bad.shape[1]))
    # img_one = np.array(np.zeros([labels_bad.shape[0], labels_bad.shape[1]]))
    # for i in range(labels_bad.shape[0]):
    #     for j in range(labels_bad.shape[1]):
    #         img_one[i][j] = label_colours[labels_bad[i][j]]
    # cv2.imwrite(join(sp_color, 'reseg_%s_%d_%d_finetune_%d_%d.png' % (filename, scale, min_size, re_scale, re_min_size)), img_one)
    labels_bad_rgb = color.label2rgb(labels_bad, img, kind='avg')
    cv2.imwrite(
        join(sp_color, 'reseg_%s_%d_%d_finetune_%d_%d.png' % (filename, scale, min_size, re_scale, re_min_size)),
        labels_bad_rgb)

    # boundarys_bad = segmentation.mark_boundaries(img, labels_bad, (1, 0, 0))
    # cv2.imwrite(
    #     join(sp_boundary, 'reseg_%s_%d_%d_finetune_%d_%d.png' % (filename, scale, min_size, re_scale, re_min_size)),
    #     boundarys_bad)
    # plt.figure(num="reseg")
    # plt.title("reseg")
    # plt.imshow(boundarys_bad)
    # plt.savefig(join(sp_boundary,'reseg_%s_%d_%d_finetune_%d_%d.png' % (filename, scale, min_size, re_scale, re_min_size)))
    boundarys_bad = segmentation.mark_boundaries(labels_bad_rgb, labels_bad, (0, 0, 0))
    plt.figure(num="reseg")
    plt.title("reseg")
    plt.imshow(boundarys_bad)
    plt.savefig(join(sp_boundary,'reseg_%s_%d_%d_finetune_%d_%d.png' % (filename, scale, min_size, re_scale, re_min_size)))
    return labels_bad
