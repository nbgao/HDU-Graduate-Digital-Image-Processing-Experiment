#!/usr/bin/env python
# coding: utf-8

# # Lab4 数字图像编码实验
# **Author: Gao Pengbing (nbgao)**  
# **Email: nbgao@126.com**

# # Work1 无损编码/压缩算法实验
# ## Work1-P1
# **实现行程编码压缩, 肉眼观察压缩效果，并计算原图和压缩以后的尺寸，计算压缩率并比较分析**     

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# #### 行程编码(RLE)-编码器
# 
# 数据量 = 元组数*(最大行程位数+灰度位数)  
# 压缩率 = 原数据量/压缩后数据量

# In[2]:


def RLE_encode(img):
    img_arr = img.ravel()
    code_list = []
    cnt, pix = 1, img_arr[0]
    for i in range(1,len(img_arr)):
        if img_arr[i]==pix:
            cnt += 1
        else:
            code_list.append((cnt, pix))
            pix = img_arr[i]
            cnt = 1
    code_list.append((cnt, pix))
    
    code_list = np.array(code_list)
    cnt_max = np.max(code_list[:,0])
    cnt_bits = int(np.ceil(np.log2(cnt_max)))
    
    tuple_num = len(code_list)
    pixel_bits = int(np.log2(256))
    encode_size = tuple_num * (cnt_bits + pixel_bits)
    origin_size = img_arr.size * pixel_bits
    compress_ratio = encode_size / origin_size

    print('编码元组对数: %d' % tuple_num)
    print('计数位数: %d bit' % cnt_bits)
    print('像素位数: %d bit' % pixel_bits)
    print('压缩编码位数: %d bit = %.2f Byte = %.2f KB' % (encode_size, encode_size/8, encode_size/8/1024))
    print('原始数据位数: %d bit = %.2f Byte = %.2f KB' % (origin_size, origin_size/8, origin_size/8/1024))
    print('压缩率: %.2f%%' % (compress_ratio*100))
    
    # 压缩信息参数
    compress_info = {}
    compress_info['cnt_bits'] = cnt_bits
    compress_info['pixel_bits'] = pixel_bits
    compress_info['img_shape'] = img.shape
    
    return code_list, compress_info


# #### 编码序列化

# In[3]:


def code_serialize(code_list, compress_info):
    cnt_bits, pixel_bits = compress_info['cnt_bits'], compress_info['pixel_bits']
    code = ''
    code_bin = ''
    for t in code_list:
        code += '%d %d '%(t[0], t[1])
        # cnt二进制编码先减1
        bin_0, bin_1 = bin(t[0]-1)[2:].zfill(cnt_bits), bin(t[1])[2:].zfill(pixel_bits)
        code_bin += bin_0 + bin_1
#         code += str(t[0]) + ' ' + str(t[1]) + ' '
    
    return code, code_bin


# #### 行程编码(RLE) 解码 + 图像解压

# In[4]:


def RLE_decode(code_bin, compress_info):
    cnt_bits, pixel_bits, (height, width) = compress_info['cnt_bits'], compress_info['pixel_bits'], compress_info['img_shape']
    
    code_length = len(code_bin)
    tuple_bits = cnt_bits + pixel_bits
    decode_list = []
    img_arr = np.zeros(height*width).astype('uint8')
    i, j = 0, 0
    while i < code_length:
        cnt = int(code_bin[i : i+cnt_bits], base=2) + 1
        pix = int(code_bin[i+cnt_bits: i+tuple_bits], base=2)
        decode_list.append((cnt, pix))
        i += tuple_bits
        while cnt>0:
            img_arr[j] = pix
            cnt -= 1
            j += 1
    
    decode_list = np.array(decode_list)
    img_dec = img_arr.reshape(height, width)

    return img_dec, decode_list


# #### 图像行程编码压缩/解压实验

# In[5]:


for i in range(1,6):
    file_path = '../Image/image'+str(i)+'.jpg'
    img = plt.imread(file_path)
    img_gray = np.uint8(skimage.color.rgb2gray(img)*256)
    h, w = img_gray.shape
    print('原始图像尺寸:', img_gray.shape)
    
    code_list, compress_info = RLE_encode(img_gray)
    print('\n编码元组对列表:\n', code_list, '\n')
    
    code, code_bin = code_serialize(code_list, compress_info)
    print('编码序列:\n', code[:100])
    print('二进制编码序列:\n', code_bin[:100], '\n')
    
    img_dec, decode_list = RLE_decode(code_bin, compress_info)
    
    # 均方根误差 RMSE
    RMSE = np.sqrt(np.mean(np.sum((img_dec - img_gray)**2)))
    # 峰值信噪比 PSNR
    PSNR = 20*np.log10(255/RMSE)
    
    print('均方根误差(RMSE) = %.3f' % RMSE)
    print('峰值信噪比(PSNR) = %.3f' % PSNR)
    
    plt.figure(figsize=(12, 6*img.shape[0]/img.shape[1]))
    plt.subplot(121)
    plt.imshow(img_gray, 'gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img_dec, 'gray')
    plt.title('RLE Uncompress')
    plt.axis('off')
    
    plt.show()


# ## Work1-P2
# **实现哈夫曼压缩, 肉眼观察压缩效果，并计算原图和压缩以后的尺寸，计算压缩率并比较分析** 

# #### Huffman编码-编码

# In[6]:


def Huffman_encode(img):
    img_arr = img.ravel()
    origin_size = img.size*8
    
    # 灰度频率统计
    f = np.zeros(256)
    for i in range(256):
        f[i] = np.sum(img_arr==i) # 各灰度计数
    f = f/img.size # 计算灰度频率
    
    simbols = np.array([i for i in range(256) if f[i]!=0]) # 频率非0符号
#     print('simbols:', simbols)
    simbols_num = len(simbols)
#     print('频率非空数:', simbols_num)
    f = f[simbols] # 去除频率为0
    # 频率升序排序
    sort_index = np.argsort(f)
    f_sort = f[sort_index]
    simbols_sort = simbols[sort_index]

    simbols_index = [[i] for i in range(simbols_num)]
    codeword_tmp = ['' for i in range(simbols_num)]
    
    while len(f_sort)>1:
        p, q = simbols_index[0], simbols_index[1]
        for i in p:
            codeword_tmp[i] = '0' + codeword_tmp[i]
        for i in q:
            codeword_tmp[i] = '1' + codeword_tmp[i]
        f_sort[1] += f_sort[0]
        f_sort = f_sort[1:] # f_sort前2个频率合并
        simbols_index[1] = p+q # 结点列表合并
        simbols_index = simbols_index[1:] # 前2个符号合并为新节点
        
        # 重新排序
        sort_index = np.argsort(f_sort)
        f_sort = f_sort[sort_index]
        simbols_index = np.array(simbols_index)[sort_index]
        
    codeword = np.array([None for i in range(256)])
    codeword[simbols_sort] = codeword_tmp
    print('非空码字长度：', len(codeword_tmp))
    
    
    encode_dict, decode_dict = {}, {}
    for i in range(256):
        encode_dict[i] = codeword[i]
        if codeword[i] is not None:
            decode_dict[codeword[i]] = i
#     print('编码词典:\n', encode_dict)
#     print('解码词典:\n', decode_dict)
   
    huffman_encode = ''
    for i in range(len(img_arr)):
        pix = img_arr[i]             
        huffman_encode += codeword[pix]

    encode_size = len(huffman_encode)
    average_word_size = encode_size/img.size
    print('\nHuffman编码长度: %d bit' % encode_size)
    print('平均码长: %.3f bit' %  average_word_size)
    
    print('压缩数据位数: %d bit = %.2f Byte = %.2f KB' % (encode_size, encode_size/8, encode_size/8/1024))
    print('原始数据位数: %d bit = %.2f Byte = %.2f KB' % (origin_size, origin_size/8, origin_size/8/1024))
    compress_ratio = encode_size/origin_size
    print('压缩率: %.2f%%' % (compress_ratio*100))
    
    # 压缩信息参数
    compress_info = {}
    compress_info['encode_dict'] = encode_dict
    compress_info['decode_dict'] = decode_dict
    compress_info['img_shape'] = img.shape
    
    return huffman_encode, compress_info


# #### Huffman编码-解码

# In[7]:


def Huffman_decode(huffman_encode, compress_info):
    decode_dict, (height, width) = compress_info['decode_dict'], compress_info['img_shape']
    img_dec = np.zeros(height*width)
    encode_size = len(huffman_encode)
    word_list = decode_dict.keys()
    i, k = 0, 0
    while i < encode_size:
        word = huffman_encode[i]
        i += 1
        while word not in word_list:
            word += huffman_encode[i]
            i += 1
        img_dec[k] = decode_dict[word]
        k += 1
    img_dec = img_dec.reshape(height, width)
    return img_dec        


# In[8]:


for i in range(1,6):
    file_path = '../Image/image'+str(i)+'.jpg'
    img = plt.imread(file_path)
    img_gray = np.uint8(skimage.color.rgb2gray(img)*256)
    h, w = img_gray.shape
    print('原始图像尺寸:', img_gray.shape)
    
    huffman_encode, compress_info = Huffman_encode(img_gray)
    print('\nHuffman编码序列:\n', huffman_encode[:100], '\n')
    img_dec = Huffman_decode(huffman_encode, compress_info)
    
    # 均方根误差 RMSE
    RMSE = np.sqrt(np.mean(np.sum((img_dec - img_gray)**2)))
    # 峰值信噪比 PSNR
    PSNR = 20*np.log10(255/RMSE)
    
    print('均方根误差(RMSE) = %.3f' % RMSE)
    print('峰值信噪比(PSNR) = %.3f' % PSNR)
    
    plt.figure(figsize=(12, 6*img.shape[0]/img.shape[1]))
    plt.subplot(121)
    plt.imshow(img_gray, 'gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img_dec, 'gray')
    plt.title('Huffman Uncompress')
    plt.axis('off')
    
    plt.show()


# ## Work1-P3
# **实现一维无损预测压缩, 肉眼观察压缩效果，并计算原图和压缩以后的尺寸，计算压缩率并比较分析**

# #### 一维无损预测编码-编码

# In[9]:


def Predict_encode(img, k):
    code_arr = np.copy(img).astype('int')
    h, w = img.shape
    for i in range(k,w):
        code_arr[:,i] = img[:,i] - np.round(np.mean(img[:,i-k:i], axis=1))
    
    xn, en = code_arr[:,:k], code_arr[:,k:]
    err_max = np.max(np.abs(en))
    err_bits = int(np.ceil(np.log2(err_max)))
    pixel_bits = int(np.log2(256))
    encode_size = h * (pixel_bits*k + (w-k)*err_bits)
    origin_size = img.size * pixel_bits
    compress_ratio = encode_size / origin_size

    print('误差最大值:', err_max)
    print('预测误差位数: %d bit' % err_bits)
    print('压缩编码位数: %d bit = %.2f Byte = %.2f KB' % (encode_size, encode_size/8, encode_size/8/1024))
    print('原始数据位数: %d bit = %.2f Byte = %.2f KB' % (origin_size, origin_size/8, origin_size/8/1024))
    print('压缩率: %.2f%%' % (compress_ratio*100))
    
    return code_arr


# #### 一维无损预测编码-解码

# In[10]:


def Predict_decode(code_arr, k):
    h, w = code_arr.shape
    img_dec = code_arr.copy()
    for i in range(k,w):
        img_dec[:,i] = np.round(np.mean(img_dec[:,i-k:i], axis=1)) + img_dec[:,i]
        
    return img_dec


# In[11]:


for i in range(1,6):
    file_path = '../Image/image'+str(i)+'.jpg'
    img = plt.imread(file_path)
    img_gray = np.uint8(skimage.color.rgb2gray(img)*256)
    h, w = img_gray.shape
    print('原始图像尺寸:', img_gray.shape)
    
    k = 3 # 前k个采样预测
    code_arr = Predict_encode(img_gray, k)
    xn, en = code_arr[:,k], code_arr[:,k:]
    
    print('\n预测编码:')
    print('Xn = \n', xn[:100])
    print('En = \n', en[:100], '\n')
    
    img_dec = Predict_decode(code_arr, k)
    
    # 均方根误差 RMSE
    RMSE = np.sqrt(np.mean((img_dec - img_gray)**2))
    # 峰值信噪比 PSNR
    PSNR = 20*np.log10(255/RMSE)
    
    print('均方根误差(RMSE) = %.3f' % RMSE)
    print('峰值信噪比(PSNR) = %.3f' % PSNR)
    
    plt.figure(figsize=(12, 6*img.shape[0]/img.shape[1]))
    plt.subplot(121)
    plt.imshow(img_gray, 'gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img_dec, 'gray')
    plt.title('Predict Encode Uncompress')
    plt.axis('off')
    
    plt.show()


# In[ ]:




