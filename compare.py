import os
import operator
from PIL import Image
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim


data_folder = os.path.join(os.path.dirname(__file__), "data")

# ref: 
# https://github.com/Damon0626/My-Projects/blob/master/11-Image%20Difference/02-%E5%9B%BE%E7%89%87%E6%89%BE%E4%B8%8D%E5%90%8C.py
# https://blog.csdn.net/enter89/article/details/90293971
# https://blog.csdn.net/wsp_1138886114/article/details/90484345

def get_file_name(path):
    return os.path.basename(path)

def is_same(ta, tb):
    ia = Image.open(ta)
    ib = Image.open(tb)
    rlt = operator.eq(ia, ib)

    just_str = None
    if rlt is True:
        just_str = "is the same as"
    else:
        just_str = "is different from"
    print("{} {} {}".format(get_file_name(ta), just_str, get_file_name(tb)))

    return rlt

def diff(ta, tb):
    ia = cv2.imread(ta)
    shape = ia.shape
    ib = cv2.imread(tb)
    if ib.shape is not shape:
        ib = cv2.resize(ib, (shape[1], shape[0]))
    ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
    print(ga.shape, gb.shape)
    score, diff = ssim(ga, gb, full=True)

    # 如果full为True, 返回两幅图的实际图像差异,值在[-1, 1]，维度同原图像．
    # print(score,'\n', diff.shape)  # 0.87, (600, 400)
    diff = (diff*255).astype('uint8')
    print("SSIM:{}".format(score))

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # ret, thresh = ...
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # 找出面积最大的10个轮廓
    area_index = np.argsort([cv2.contourArea(c) for c in cnts])
    cnts = np.array(cnts)[area_index[-10::]]
    # print(cnts)

    for c in cnts:
	    x, y, w, h = cv2.boundingRect(c)
	    cv2.rectangle(ia, (x, y), (x+w, y+h), (0, 0, 255), 2)
	    cv2.rectangle(ib, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('diff', diff)
    cv2.imshow('thresh', thresh)
    #cv2.imshow('img1', ia)
    #cv2.imshow('img2', ib)
    cv2.waitKey(0)



if __name__ == '__main__':
    org_path = os.path.join(data_folder, "org.jpg")
    org_res_path = os.path.join(data_folder, "org_res.jpg")
    org_size_path = os.path.join(data_folder, "org_size.jpg")
    org_color_path = os.path.join(data_folder, "org_color.jpg")
    org_copy_path = os.path.join(data_folder, "org_copy.jpg")

    is_same(org_path, org_size_path)
    is_same(org_path, org_copy_path)

    #diff(org_path, org_path)
    #diff(org_path, org_color_path)
    #diff(org_path, org_res_path)
    diff(org_path, org_size_path)
    #diff(org_path, org_color_path)
    #diff(org_path, org_copy_path)





