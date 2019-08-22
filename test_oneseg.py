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

# save dir
label_dir = join(os.getcwd(), 'result_image_seg', DATASET, 'reseg', 'label')
boundary_dir = join(os.getcwd(), 'result_image_seg', DATASET, 'reseg', 'boundary')
boundary_dir_one = join(os.getcwd(), 'result_image_seg', DATASET, 'reseg', 'color')

if not os.path.exists(label_dir):
    os.makedirs(label_dir)
if not os.path.exists(boundary_dir):
    os.makedirs(boundary_dir)
if not os.path.exists(boundary_dir_one):
        os.makedirs(boundary_dir_one)
# EM method & save results to dir
num = 1
for img_path in img_list:
    if img_path[-3:] == 'jpg':
        num = num+1
        print(img_path)
        img = imread(img_path)
        path, img_name = os.path.split(img_path)

        segments_EM = image_seg(img_path=img_path, sp_met='felzenszwalb', num_cuts=3)

        # save segments label & boundry
        cv2.imwrite(join(label_dir, img_name), np.uint64(segments_EM))

        boundary = segmentation.mark_boundaries(img, segments_EM, (1, 0, 0))
        imsave(join(boundary_dir, img_name), boundary)
        img_one = np.array(np.zeros([segments_EM.shape[0], segments_EM.shape[1]]))
        label_colours = np.random.randint(255, size=(segments_EM.shape[0] * segments_EM.shape[1]))
        for i in range(segments_EM.shape[0]):
            for j in range(segments_EM.shape[1]):
                img_one[i][j] = label_colours[segments_EM[i][j]]
        imsave(join(boundary_dir_one, img_name), img_one)
        # save segments label & boundry
        #         cv2.imwrite(join(label_dir, img_name), np.uint64(segments))

        #         boundary = segmentation.mark_boundaries(img, segments, (1, 0, 0))
        #         imsave(join(boundary_dir, img_name), boundary)

        if show_segmentation:
            plt.figure()
            plt.imshow(boundary)
            plt.show()
    if(num>20):
        break

# with open("eval_orig.txt","w")as f:
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
