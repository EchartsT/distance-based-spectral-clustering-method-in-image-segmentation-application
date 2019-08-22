#EM方法，用纹理和颜色作为测度

from EM_segmentation_method import before_method, image_seg, EM_method
from evaluate import evaluate
import os
from os.path import join, isfile
import cv2
from imageio import imread, imwrite
from skimage import data, io, segmentation, color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave

show_segmentation = False


# DATASET = 'testdata'
DATASET = 'BSD'

# dir setting
# input dir
img_dir = join(os.getcwd(), 'data', DATASET)
img_list = [join(img_dir, f) for f in os.listdir(img_dir) if isfile(join(img_dir, f))]

# groundtruth dir
gt_dir = join(os.getcwd(), 'data', 'groundtruth')

# # save dir/纹理，em,重新分割 color_texture/BSD/EM
# label_dir = join(os.getcwd(), 'color_texture', DATASET, 'EM', 'label')
# boundary_dir = join(os.getcwd(), 'color_texture', DATASET, 'EM', 'boundary')
# color_dir = join(os.getcwd(), 'color_texture', DATASET, 'EM', 'color')
# boundary_dir_one = join(os.getcwd(), 'color_texture', DATASET, 'EM', 'boundary_firtstep')
# color_dir_one = join(os.getcwd(), 'color_texture', DATASET, 'EM', 'boundary_firtstep_color')

# #平均颜色,em,重新分割， color_mean/BSD/EM
label_dir = join(os.getcwd(), 'color_mean', DATASET, 'EM', 'label')
boundary_dir = join(os.getcwd(), 'color_mean', DATASET, 'EM', 'boundary')
color_dir = join(os.getcwd(), 'color_mean', DATASET, 'EM', 'color')
boundary_dir_one = join(os.getcwd(), 'color_mean', DATASET, 'EM', 'boundary_firtstep')
color_dir_one = join(os.getcwd(), 'color_mean', DATASET, 'EM', 'boundary_firtstep_color')

if not os.path.exists(label_dir):
    os.makedirs(label_dir)
if not os.path.exists(boundary_dir):
    os.makedirs(boundary_dir)
if not os.path.exists(boundary_dir_one):
        os.makedirs(boundary_dir_one)
if not os.path.exists(color_dir_one):
        os.makedirs(color_dir_one)
if not os.path.exists(color_dir):
        os.makedirs(color_dir)

# EM method & save results to dir
num = 1

for img_path in img_list:
    if img_path[-3:] == 'jpg':
        num = num+1
        print(img_path)
        img = imread(img_path)
        path, img_name = os.path.split(img_path)

        # #纹理加颜色
        # segments_EM,result_one = EM_method(img_path=img_path, sp_met='felzenszwalb', num_cuts=3, EM_iter=4, K=100,dist_hist = True)
        #颜色均值
        segments_EM, result_one = EM_method(img_path=img_path, sp_met='felzenszwalb', num_cuts=3, EM_iter=4, K=100)

        # # 分割以颜色显示的颜色盘
        # label_colours = np.random.randint(255, size=(result_one.shape[0] * result_one.shape[1]))
        # # save segments label & boundry
        # cv2.imwrite(join(label_dir, img_name), np.uint64(segments_EM))
        #
        # boundary = segmentation.mark_boundaries(img, segments_EM, (1, 0, 0))
        # imsave(join(boundary_dir, img_name), boundary)
        #
        # img_all = np.array(np.zeros([segments_EM.shape[0], segments_EM.shape[1]]))
        # for i in range(segments_EM.shape[0]):
        #     for j in range(segments_EM.shape[1]):
        #         img_all[i][j] =  label_colours[segments_EM[i][j]]
        # imsave(join(color_dir, img_name), img_all)

        # save segments label & boundry
        cv2.imwrite(join(label_dir, img_name), np.uint64(segments_EM))
        segments_EM_color = color.label2rgb(segments_EM,img,kind="avg")
        boundary = segmentation.mark_boundaries(segments_EM_color, segments_EM, (1, 0, 0))
        imsave(join(boundary_dir, img_name), boundary)

        # img_all = np.array(np.zeros([segments_EM.shape[0], segments_EM.shape[1]]))
        # for i in range(segments_EM.shape[0]):
        #     for j in range(segments_EM.shape[1]):
        #         img_all[i][j] = label_colours[segments_EM[i][j]]
        imsave(join(color_dir, img_name), segments_EM_color)

#第一次的结果
        # boundary_one = segmentation.mark_boundaries(img, result_one, (1, 0, 0))
        # imsave(join(boundary_dir_one, img_name), boundary_one)
        # img_one =  np.array(np.zeros([result_one.shape[0], result_one.shape[1]]))
        # for i in range(result_one.shape[0]):
        #     for j in range(result_one.shape[1]):
        #         img_one[i][j] =  label_colours[result_one[i][j]]
        # imsave(join(color_dir_one, img_name), img_one)
        result_one_rgb = color.label2rgb(result_one,img,kind="avg")
        boundary_one = segmentation.mark_boundaries(result_one_rgb, result_one, (1, 0, 0))
        imsave(join(boundary_dir_one, img_name), boundary_one)
        imsave(join(color_dir_one, img_name), boundary_one)


        # save segments label & boundry
        #         cv2.imwrite(join(label_dir, img_name), np.uint64(segments))

        #         boundary = segmentation.mark_boundaries(img, segments, (1, 0, 0))
        #         imsave(join(boundary_dir, img_name), boundary)

        if show_segmentation:
            plt.figure()
            plt.imshow(boundary)
            plt.show()
    # if(num>500):
    #     break


# with open("eval_reseg.txt","w")as f:
#
# # evaluate in test data
#
#     pre, rec, F = evaluate(label_dir, gt_dir, soft_thres=1)
#     p, r, f = sum(pre) / len(pre), sum(rec) / len(rec), sum(F) / len(F)
#     print(p, r, f)
#     str = "soft_thres=1"+str(p)+";" + str(r)+";" +str(f)+";\n"
#     f.write(str)
#     pre, rec, F = evaluate(label_dir, gt_dir, soft_thres=2)
#     p, r, f = sum(pre) / len(pre), sum(rec) / len(rec), sum(F) / len(F)
#     print(p, r, f)
#     str = "soft_thres=2"+str(p)+";" + str(r)+";" +str(f)+";\n"
#     f.write(str)
#     pre, rec, F = evaluate(label_dir, gt_dir, soft_thres=3)
#     p, r, f = sum(pre) / len(pre), sum(rec) / len(rec), sum(F) / len(F)
#     str = "soft_thres=3"+str(p)+";" + str(r)+";" +str(f)+";\n"
#     f.write(str)