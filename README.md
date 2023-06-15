[toc]

#图像工具


## 环境
python3.11
``` cmd
pip install -r requirements.txt
conda install scikit-image
```

## 图像比对
### pHash 感知哈希
感知hash算法，计算两个图片的哈希值，然后计算两个哈希值的汉明距离，汉明距离越小，图片越相似。
pHash()的channel默认指定了图片第0个通道，如果需要综合考虑RGB等的，可手动改一下这里，把三个通道的哈希值合并一下。
``` python
    s = checkImageSimilarity(image1=image1, image2=image2, method="pHash")
    print("两张图片的相似度是: {0}%".format(s*100))
```
### hist 直方图
图像转灰直方图，然后计算两个直方图的相似度（直方图相等记为0，不等的计算差值占最大值的百分比）
``` python
    s = checkImageSimilarity(image1=image1, image2=image2, method="hist")
    print("两张图片的相似度是: {0}%".format(s*100))
```
### ssim 结构化相似度
结构化相似性指数（SSIM）是一种用于测量两幅图像之间的相似度的方法。该指标测量的是有损图像质量的感知损失，即它与人眼所感知到的图像质量的相关性很高。
计算的会慢一点，调用skimage库
``` python
    s = checkImageSimilarity(image1=image1, image2=image2, method="ssim")
    print("两张图片的相似度是: {0}%".format(s*100))
```