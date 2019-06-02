# -*- coding: utf-8 -*-
# @Time   : 19-5-15 下午3:50
# @Author : TerYang
# @contact : adau22@163.com ============================
# My github:https://github.com/TerYang/              ===
# Copyright: MIT License                             ===
# Good good study,day day up!!                       ===
# ======================================================
import pandas as pd
import numpy as np
import os
import threading as td
from queue import Queue
import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.utils.data as Data
import xlsxwriter


test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/data/"
res_url = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/test_analysis/'
datnam = 'data'
# source_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/"


# COLSIZ = 10
# tformat = lambda s: str(s).title().ljust(COLSIZ)
# print('\n%s' % ''.join(map(tformat, vars())))

def count_data_flag(path,mark=None):
    """
    func : read data(attack status records) to train or to validate
    :param path:
    :param mark:
    :return: row(number), flag(label type of list), data(array)
    """

    data1 = pd.read_csv(path, sep=None, header=None, dtype=np.float64, engine='python', encoding='utf-8')#,nrows=64*64*100 ,nrows=64*64*1
    data = data1.values.astype(np.float64)
    # data = np.reshape(data, (-1, 64, 22))
    file = os.path.basename(path)
    # print('{} has data :ndim:{} dtype:{} shape:{}'.format(file,data.ndim, data.dtype, data.shape))
    print('{} has data shaped:{}'.format(file, data.shape))
    rows = data.shape[0]
    start = 0
    row = int(rows // 64)
    end = row*64
    if mark:
        if mark == 'test':
            start = int(((rows*0.8)//64)*64)
            row = int((rows-start)//64)
            end = int(start+((rows-start)//64)*64)
        elif mark == 'train':
            row = int((rows*0.8)//64)
            end = int(row * 64)
    else:
        pass

    source_flags = data[start:end,-1].tolist()

    flags = []

    """ solve normal marked as 1"""
    if 'new_data' in path:#new_data:  attack marked as 1,存在任意個1,即標記爲1
        for r in range(row):
            num = 0.
            for item in source_flags[r*64:r*64+64]:
                if item == 1.:
                    num = 1.
                    break
            flags.append(num)

    else:# data :  normal marked as 1,沒有attack即一個0都沒有的情況,就標記爲1,否則,標記爲0
        for r in range(row):
            num = 1.
            for item in source_flags[r*64:r*64+64]:
                if item == 0.:
                    num = 0.
                    break
            flags.append(num)
    one_num = flags.count(1.)
    zero_num = flags.count(0.)
    num = len(flags)
    print('%s,one numbers:%d,zero numbers:%d,total numbers:%d'%(file,one_num,zero_num,num))
    return file,one_num,zero_num,num,flags
    # try:
    #     data = data[start:end,:-1].reshape((-1,64,21))
    # except:
    #     print('Error!!! Error!!! file name: {},data shape: {},flags size:{}'.format(file,data.shape,len(flags)))
    # print('{} start at:{} acquires labels shape:{} data shape{} done read files!!!\n'.format(file, start,len(flags),data.shape))
    # return row, flags,data


def getTrainDiscriminor(path=test_addr,mark=None):
    """
    func: write test parameter to one excel file for a type of GAN
    :param path:
    :param mark:
    :return: flag(label type of list), data(array)
    """
    files = os.listdir(path)
    lens = len(files)
    # print('operate: {}, data folder have  files'.format(mark,lens))
    pool = mp.Pool(processes=lens)

    res_path = os.path.join(res_url,datnam+'_analysis_data_construction.xlsx')
    file_urls = []
    results = []
    for i in os.listdir(path):
        if '.txt' in i:
            file_urls.append(os.path.join(path,i))
            # results.append(pool.apply(testdata,(os.path.join(path,i),mark,)))#_async
            results.append(pool.apply_async(count_data_flag,(os.path.join(path,i),mark,)))#_async
    pool.close()
    pool.join()
    # print(len(results),type(results))
    flags = []
    # data = []
    # sheet 1
    workbook = xlsxwriter.Workbook(res_path)  # 建立文件
    worksheet = workbook.add_worksheet('analysis')  # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    # rownum = 1
    headers = []
    worksheet.write_row('A1', ['filename','one_num','zero_num','total_num'])

    for i,result in enumerate(results,start=1):
        # file,one,zero,num,all_flags
        # result = result#.get()
        result = result.get()[:-1]
        headers.append(result[0])
        worksheet.write_row(row=i,col=0,data=result)
        # rownum += 1

    # sheet 2
    worksheet = workbook.add_worksheet('data')
    worksheet.write_row('A1', headers)

    for i,result in enumerate(results):
        result = result.get()[-1]
        worksheet.write_column(row=1,col=i,data=result)
    workbook.close()

if __name__ == '__main__':
    getTrainDiscriminor(mark='train')


    # results = np.zeros((4,4))
    # res_path = os.path.join(res_url,datnam+'analysis_data_construction.xlsx')
    # workbook = xlsxwriter.Workbook(res_path)  # 建立文件
    # worksheet = workbook.add_worksheet('analysis')  # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
    # # rownum = 1
    # headers = []
    # worksheet.write_row('A1', ['filename','one_num','zero_num','total_num'])
    #
    # for i,result in enumerate(results,start=1):
    #     # file,one,zero,num,all_flags
    #     # result = result#.get()
    #     headers.append(result[0])
    #     worksheet.write_row(row=i,col=0,data=result)
    #     # rownum += 1
    #
    # # sheet 2
    # worksheet = workbook.add_worksheet('data')
    # results = np.ones((4,10))
    # worksheet.write_row(row=0,col=0,data=headers)
    # for i,result in enumerate(results):
    #     # result = result.get()[-1]
    #     worksheet.write_column(row=1,col=i,data=result)
    # workbook.close()