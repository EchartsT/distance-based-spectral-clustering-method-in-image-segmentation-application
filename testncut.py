# ncut method
from NCut import ncut
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
label_dir = join(os.getcwd(), 'result_', DATASET, 'ncut', 'label')
boundary_dir = join(os.getcwd(), 'result', DATASET, 'ncut', 'boundary')

if not os.path.exists(label_dir):
    os.makedirs(label_dir)
if not os.path.exists(boundary_dir):
    os.makedirs(boundary_dir)

# ncut method & save results to dir
# for img_path in img_list:
#     if img_path[-3:] == 'jpg':
#         img = imread(img_path)
#         path, img_name = os.path.split(img_path)
# #         if isfile(join(label_dir, img_name)):
# #             continue

#         try:
#             segments_ncut = ncut(img=img, sp_met='fl')
#             # save segments label & boundry
#             cv2.imwrite(join(label_dir, img_name), np.uint64(segments_ncut))

#             boundary = segmentation.mark_boundaries(img, segments_ncut, (1, 0, 0))
#             imsave(join(boundary_dir, img_name), boundary)

#             if show_segmentation:
#                 plt.figure()
#                 plt.imshow(boundary)
#                 plt.show()

#         except:
#             print('error: ', img_name)


# evaluate in test data
pre, rec, F = evaluate(label_dir, gt_dir, soft_thres=1)
p, r, f = sum(pre) / len(pre), sum(rec) / len(rec), sum(F) / len(F)
print(p, r, f)

pre, rec, F = evaluate(label_dir, gt_dir, soft_thres=2)
p, r, f = sum(pre) / len(pre), sum(rec) / len(rec), sum(F) / len(F)
print(p, r, f)