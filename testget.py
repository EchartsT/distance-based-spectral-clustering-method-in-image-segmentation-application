import numpy as np
import scipy
from skimage.segmentation import slic, felzenszwalb
from skimage.future import graph
import networkx as nx
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
from admm_algorithms import admm, relation_density_admm, var_admm

from numpy import linalg as LA
from sklearn.preprocessing import normalize
from imageio import imread

from scipy.spatial import distance
from skimage.feature.texture import local_binary_pattern
import time
import warnings
import getLabels

# segments = getLabels.get_labels("/opt/project/data/BSD/87065.jpg")
p = 123
r = 22
with open("eval_orig.txt","w")as f:
    f.write("%d,%d\n"%(p,r))
    f.write("fff")
