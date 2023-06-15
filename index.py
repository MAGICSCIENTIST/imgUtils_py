import os;
import cv2;
import skimage;
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np;


def pHash(path, hash_size=16, imge_size=32, channel=0):
    """
    get image pHash
    :param image: image file
    :return: image pHash
    """
    img = cv2.imread(path)
    img1 = cv2.resize(img, (imge_size, imge_size),cv2.COLOR_RGB2GRAY)

    # dct 只能处理float32类型，所以需要转换一下
    h, w, c = img1.shape[:3]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img1[:h, :w, channel]

    # DCT二维变换
    # 离散余弦变换，得到dct系数矩阵
    img_dct = cv2.dct(cv2.dct(vis0))
    img_dct = cv2.resize(img_dct, (hash_size,hash_size))
    # 把list变成一维list
    img_list = np.array(img_dct.tolist()).flatten()
    # 计算均值
    img_mean = cv2.mean(img_list)[0]
    avg_list = ['0' if i<img_mean else '1' for i in img_list]
    res = ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,hash_size*hash_size,4)])
    return res

def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)

def checkImageSimilarity(image1, image2, method='ssim'):
    if method == "pHash":
        h1 = pHash(image1)
        h2 = pHash(image2)
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(h1) != len(h2):
            return -1
        # 遍历判断
        for i in range(len(h1)):
            # 不相等则n计数+1，n最终为相似度
            if h1[i] != h2[i]:
                n = n + 1
        return 1 - n / len(h1)
    # 直方图比较
    elif method == "hist":
        img1 = cv2.imread(image1)
        img2 = cv2.imread(image2)
        hist1 = cv2.calcHist([img1], [1], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([img2], [1], None, [256], [0.0, 255.0])
        similarity = hist_similar(hist1, hist2)
        return similarity
    
    # 结构化相似度比较
    elif method == "ssim":
       im1 = cv2.imread(image1)
       im1 = cv2.resize(im1, (im1.shape[0], im1.shape[1]),cv2.COLOR_RGB2GRAY)
       im2 = cv2.imread(image2)
       im2 = cv2.resize(im2, (im2.shape[0], im2.shape[1]),cv2.COLOR_RGB2GRAY)
       
       mse_const = mean_squared_error(im1, im2)
       ssim_const = ssim(im1, im2, data_range=im1.max() - im1.min(), win_size=3)
       return ssim_const

if __name__ == '__main__':    
    root = "E:\\data\\animals\\asd"

    image1 =  os.path.join(root, "GZNR008X-HYH020-20220914-08006.JPG")

    # # total same one
    # image2 =  os.path.join(root, "GZNR008X-HYH020-20220914-08006.JPG")

    # # same one
    # image2 =  os.path.join(root, "GZNR008X-HYH020-20220914-08007.JPG")

    # different one
    image2 =  os.path.join(root, "GZNR008X-HYH020-20220914-12511.JPG")

    # # sometimes same but light different
    # image2 =  os.path.join(root, "GZNR008X-HYH020-20220914-08020.JPG")
    s = checkImageSimilarity(image1=image1, image2=image2, method="pHash")
    print("两张图片的相似度是: {0}%".format(s*100))