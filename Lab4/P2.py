#!/usr/bin/env python
# coding: utf-8

# # Lab4 数字图像编码实验
# **Author: Gao Pengbing (nbgao)**  
# **Email: nbgao@126.com**

# # Work2 有损编码/压缩算法实验
# 查阅JPEG编码的有关资料，对图像进行JPEG压缩，算法步骤必须包括如下几个部分：图像分块，离散余弦变换，量化，ac和dc系数的Z字形编排。  
# **Work2-P1**  
# 质量因子分别选为20，60，80，对比显示原图与不同质量因子下解码后的图像。  
# **Work2-P2**  
# 记录图像大小、压缩比、均方根误差；对结果进行分析。

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, encimg = cv2.imencode('.jpg', img, encode_param)


# In[3]:


def JPEG_compress(img, quality):
    result, img_encode = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
#     print(img_enc)
    img_jpeg = cv2.imdecode(img_encode, 1)
    
    encode_size = len(img_encode) * 8
    origin_size = img.size * 8

    print('\nJPEG 质量因子 = %d' % quality)
    print('压缩编码位数: %d bit = %.2f Byte = %.2f KB' % (encode_size, encode_size/8, encode_size/8/1024))
    print('原始数据位数: %d bit = %.2f Byte = %.2f KB' % (origin_size, origin_size/8, origin_size/8/1024))
    compress_ratio = origin_size / encode_size
    print('压缩比 = %.3f' % compress_ratio)
    return img_jpeg


# In[10]:


for i in range(1,6):
    file_path = '../Image/image'+str(i)+'.jpg'
    img = plt.imread(file_path)
    img_gray = np.uint8(skimage.color.rgb2gray(img)*255)
    h, w = img_gray.shape
    print('原始图像尺寸:', img_gray.shape)
    
    plt.figure(figsize=(12, 12*img.shape[0]/img.shape[1]))
    plt.subplot(221)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    
    quality_list = [20, 60, 80]
#     quality_list = [85, 90, 95]
    img_jpeg = [_ for i in range(3)]
    for i,quality in enumerate(quality_list):
        img_jpeg[i] = JPEG_compress(img, quality)    
    
        # 均方根误差 RMSE
        RMSE = np.sqrt(np.mean((img_jpeg[i] - img)**2))
        # 峰值信噪比 PSNR
        PSNR = 20*np.log10(255/RMSE)
        
        print('均方根差(RMSE) = %.3f' % RMSE)
        print('峰值信噪比(PSNR) = %.3f dB' % PSNR)
    
        plt.subplot(2,2,i+2)
        plt.imshow(img_jpeg[i])
        plt.title('JPEG Quality=%d' % quality_list[i])
        plt.axis('off')
    
    plt.show()


# In[ ]:




