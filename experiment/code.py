import cv2
import glob
import numpy as np
from PIL import Image
import numpy as np

files = glob.glob("data/*.jpg")
seg_masks = glob.glob("../annotated_data/segmentation_mask/*.png")

from matplotlib import pyplot as plt

def test_shape():

    tmp = Image.open(seg_masks[0])
    plt.imshow(np.asarray(tmp))
    # for index, path in enumerate(seg_masks):
    #     img = Image.open(path)
    #     print(img.mode)


def cvt_to_gray():
    for index, path in enumerate(files):
        file_name = path.split("/")[-1]
        img = cv2.imread(files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("tmp/{}".format(file_name), img)


"""
Synthetic poisson blending test
"""
def blend_test():
    img = cv2.imread("input_dst.jpg")
    patch = cv2.imread("input_src.jpg")

    mask = cv2.imread("mask.png")
    indices = np.any(mask != [0, 0, 0], axis=-1)
    mask[indices] = [255, 255, 255]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    h, w, _ = img.shape
    print(img.shape)
    print(patch.shape)
    print(mask.shape)

    center = (500, 500)

    im_clone = cv2.seamlessClone(patch, img, mask, center, cv2.NORMAL_CLONE)

    black_img = np.zeros((h,w,3), dtype=np.uint8)
    cv2.imwrite("output.png", im_clone)
    cv2.imwrite("black.png", black_img)


blend_test()