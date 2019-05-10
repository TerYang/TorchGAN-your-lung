# -*- coding: utf-8 -*-
# @Time   : 19-5-9 上午10:43
# @Author : TerYang
# @contact : adau22@163.com ============================
# My github:https://github.com/TerYang/              ===
# Copyright: MIT License                             ===
# Good good study,day day up!!                       ===
# ======================================================

import pandas as pd
import os
import numpy as np
import re
import multiprocessing as mp

D_PATH = r'/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/tests_data'
thedataset = 'data'
result_url = r'/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/test_analysis'
columns = []


def findminax(url):
    """sort the data file every parameter max and min values"""
    dataset = os.path.basename(url)

    dataset = dataset[:dataset.index('_')]

    print(dataset)

    data1 = pd.read_csv(url, sep=None, header=None, dtype=np.str, engine='python',
                        encoding='utf-8')  # ,nrows=64*64*100 ,nrows=64*64*1
    data = data1.iloc[1:, 1:].values
    data = pd.DataFrame(data, index=data1.iloc[1:, 0].values, columns=data1.iloc[0, 1:].values,
                        dtype=np.float64)  # ,dtype=np.str,copy=True)
    data_n = data.values
    # print(data)
    min_ = np.min(data_n,axis=0)
    max_ = np.max(data_n,axis=0)
    mix = np.concatenate((np.array(min_),np.array(max_))).reshape((2,-1))
    global columns
    columns=data.columns
    return mix.astype(np.float32),columns,dataset


if __name__ == '__main__':
    pattern = re.compile(r'[A-Z]+\_?[A-Z]+')
    urls = []
    dires = os.listdir(D_PATH)  # modules names
    dires = sorted(dires)
    ban = []


    # 找到collection 目录,保存为collection 路径列表
    for root, dirs, files in os.walk(D_PATH, topdown=None):
        for name in dirs:
            if name == 'collection':
                urls.append('{}'.format(os.path.join(root, name)))
    #             '/home/gjj/ada/tests_data/WGAN/collection'

    # 保存目录,工程根目录+测试数据 为结果 目录
    path_ = os.path.join(result_url,thedataset)
    if not os.path.exists(path_):
        os.makedirs(path_)
    path_ = os.path.join(path_,'analysis_.csv')
    f = 1
    for url in urls:
        # 匹配出GAN 名称
        print(url)
        jf = pattern.findall(url)
        # GAN module name
        # print(jf)

        if len(jf):
            jf = jf[-1]
        else:
            print('arise re findall errors! ',url)
            continue

        files = [os.path.join(url,file) for file in os.listdir(url)]

        # module test csv file
        pool  = mp.Pool(processes=len(files))
        return_data = pool.map_async(findminax,files)
        pool.close()
        pool.join()
        # names = [os.path.splitext(file)[0][:os.path.splitext(file)[0].index('_')] for file in os.listdir(url)]
        # # data set name

        names = []
        # data set name
        aa = []
        # extracted data
        columns = return_data.get()[0][1]
        for dat in return_data.get():
            aa.append(dat[0])
            names.append(dat[2])

        names_cop = names.copy()
        for name in names_cop:
            names.insert(names.index(name),name)
        data_ = np.array(aa).reshape((-1,7)).astype(np.float32)

        np.set_printoptions(precision=7)
        source_ = pd.DataFrame(data_,index=names,columns=columns,dtype=np.float32)#,index=names,columns=return_data.get()[0][1]
        # print(source_)
        # exit()
        source_['module'] = [jf for _ in range(data_.shape[0])]
        if f:
            source_.to_csv(path_,sep=',',encoding='utf-8',mode='a',index_label=True)#,float_format='.8f'
            f = 0
        else:
            source_.to_csv(path_,sep=',',encoding='utf-8',mode='a',header=None,index_label=True)#,float_format='.8f'
