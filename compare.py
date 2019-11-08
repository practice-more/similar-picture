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

# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

# 通过得到RGB每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

# 感知哈希算法(pHash)
def pHash(img):
    size = 256
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def phash_cmp(ta, tb):
    img_ta = cv2.imread(ta)
    img_tb = cv2.imread(tb)
    phash_img_ta = pHash(img_ta)
    phash_img_tb = pHash(img_tb)
    #n = cmpHash(phash_img_ta, phash_img_tb)
    n = classify_hist_with_split(img_ta, img_tb)
    print("{} and {} 的相似度是{}".format(get_file_name(ta), get_file_name(tb), str(n)))


# 自定义计算两个图片相似度函数
def phash_cmp(img1_path,img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """
    try:
        # 读取图片
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # 初始化ORB检测器
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # knn筛选结果
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

        # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        print(len(good))
        print(len(matches))
        similary = len(good) / len(matches)
        print("两张图片相似度为:%s" % similary)
        return similary

    except:
        print('无法计算两张图片相似度')
        return '0'

def sift_func(img_path1,img_path2):
    img_1 = cv2.imread(img_path1)

    img_2 = cv2.imread(img_path2)
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create()
    psd_kp1, psd_des1 = sift.detectAndCompute(gray_1, None)
    psd_kp2, psd_des2 = sift.detectAndCompute(gray_2, None)

    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(psd_des1, psd_des2, k=2)
    goodMatch = []
    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，
        # 基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)

    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    print(goodMatch[:20])
    img_out = cv2.drawMatchesKnn(img_1, psd_kp1,
                                 img_2, psd_kp2,
                                 goodMatch[:20], None, flags=2)
    return img_out

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





