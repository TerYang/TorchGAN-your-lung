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
BATCH_SIZE = 64


# source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/second_merge/"
test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"

source_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/"
# source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/instrusion-dataset/test_data/data/"

### get GAN train data  old ###########################################
def minbatch_test():

    file = 'Attack_free_dataset2_ID_Normalize.txt'
    url = os.path.join(source_addr, file)
    data1 = pd.read_csv(url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=64*64*100)
    data1 = data1.values.astype(np.float32)#
    # print(data1.shape)
    data1 = np.reshape(data1, (-1, 64, 22))
    print('normal :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    num1 = data1.shape[0]
    return num1,data1

def get_data():#train_normal,train_anormal,,num=64*10000

    files = os.listdir(source_addr)
    normals = []
    for file in files:
        normals.append( os.path.join(source_addr,file))
    # print('normals lenght:',normals)

    # normal0_name = os.path.basename(normals[0])
    # normal1_name = os.path.basename(normals[1])
    print('dataset:\n',files)
    # exit()
    data1 = pd.read_csv(normals[0], sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    data1 = data1.values.astype(np.float32)#

    # data2 = pd.read_csv(train_anormal, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=num)
    data2 = pd.read_csv(normals[1], sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    data2 = data2.values.astype(np.float32)#,copy=True
    print('normal1 :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    # print('finished:{}'.format(normal0_name))

    print('normal2 :ndim:{} dtype:{} shape:{}'.format(data2.ndim, data2.dtype, data2.shape))
    # print('finished:{}'.format(normal1_name))
    num1 = data1.shape[0]//64 #int(
    num2 =  data2.shape[0]//64

    data = np.concatenate((data1[:64*num1,:],data2[:64*num2,:]),axis=0)

    # data = np.reshape(data[:num*64,],(-1,64,22)).astype(np.float32)
    data = np.reshape(data, (-1, 64, 22))
    print('data :ndim:{} dtype:{} shape:{}'.format(data.ndim, data.dtype, data.shape))
    print("normal total has {}+{}={} blocks".format(num1,num2,num1+num2))
    print('done read files!!!\n')
    return num1+num2,data

def new_get_norlmal():

    files = os.listdir(source_addr)
    normals = []
    for file in files:
        normals.append(os.path.join(source_addr, file))

    print('dataset:\n', files)
    # exit()
    data1 = pd.read_csv(normals[0], sep=None, header=None, dtype=np.str, engine='python', encoding='utf-8')#,nrows=64*64*100
    data1 = data1.values.astype(np.float32)  #
    # print('normal1 :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    num1 = data1.shape[0] // 64  # int(
    data = np.reshape(data1[:num1*64,],(-1,64,21)).astype(np.float32)
    print('data :ndim:{} dtype:{} shape:{}'.format(data.ndim, data.dtype, data.shape))
    print("normal total has {} blocks".format(num1))
    print('done read files!!!\n')
    return num1 , data

### get GAN test data  new ###########################################
def testdata(path,mark=None):
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
    for r in range(row):
        num = 0.
        for item in source_flags[r*64:r*64+64]:
            if item == 1.:
                num = 1.
                break
        flags.append(num)
    try:
        data = data[start:end,:-1].reshape((-1,64,21))
    except:
        print('Error!!! Error!!! file name: {},data shape: {},flags size:{}'.format(file,data.shape,len(flags)))
    print('{} start at:{} acquires lebels shape:{} data shape{} done read files!!!\n'.format(file, start,len(flags),data.shape))
    return row, flags,data

### get first discriminor data ###########################################
def getTrainDiscriminor(path=test_addr,mark=None):
    """
    func: read attack data to train module such as Discriminator(independently) through multiprocessing
    :param path:
    :param mark:
    :return: flag(label type of list), data(array)
    """
    files = os.listdir(path)
    lens = len(files)
    # print('operate: {}, data folder have  files'.format(mark,lens))
    pool = mp.Pool(processes=lens)

    file_urls = []
    results = []
    for i in os.listdir(path):
        if '.txt' in i:
            file_urls.append(os.path.join(path,i))
            # results.append(pool.apply(testdata,(os.path.join(path,i),mark,)))#_async
            results.append(pool.apply_async(testdata,(os.path.join(path,i),mark,)))#_async
    pool.close()
    pool.join()
    # print(len(results),type(results))
    flags = []
    data = []
    for i,result in enumerate(results):
        # result = result#.get()
        result = result.get()
        # print(type(result),len(result))
        # print(result[1],result[2])
        if i:
            flags.extend(result[1])
            data =np.concatenate((data,result[2]))
        else:
            flags = result[1]
            data = result[2]
    print('return data shape:{},labels shape'.format(data.shape,len(flags)))
    return flags,data


"""get data to new GAN"""
def testToGAN(path,mark=None):
    """
    func: read normal data to train GAN defined as the Class(new gan code)
    :param path:
    :param mark:
    :return: dataloader
    """
    files = os.listdir(path)
    if len(files)>1:
        print('dataset address error at testToGAN')
        return -1
    else:
        path = os.path.join(path,files[0])
    data1 = pd.read_csv(path, sep=None, header=None, dtype=np.float64, engine='python', encoding='utf-8')#,nrows=64*64*100
    data = data1.values.astype(np.float64)
    file = os.path.basename(path)
    print('{} has data shaped:{}'.format(file, data.shape))
    rows = data.shape[0]
    start = 0
    end = rows
    row = int(rows // 64)
    if mark:
        if mark == 'test':
            start = int(((rows*0.8)//64)*64)
            row = int((rows-start)//64)
            end = int(start+((rows-start)//64)*64)
        elif mark == 'train':
            row = int((rows*0.8)//64)
            end = int(row * 64)
    else:
        print('arise error at testToGAN parameter mark')

    data = data[start:end,].reshape((-1,64,21))


    TraindataM = torch.from_numpy(data).float()    # transform to float torchTensor

    TraindataM = torch.unsqueeze(TraindataM,1)
    print(TraindataM.shape)
    TorchDataset = Data.TensorDataset(TraindataM)

    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
    print('{} start at:{} acquires data shape{} done read files!!!\n'.format(file, start,data.shape))
    return train_loader#, Variable(TestdataM),Variable(TestLabelM)

"""get data to testing new GAN"""
def testNormal(path, mark=None):
    """
    func:read Normal data to test.py for validating the trained modules
    :param path:
    :param mark:
    :return: row(number), flag(label type of list), data(array)
    """
    files = os.listdir(path)
    if len(files) > 1:
        print('dataset address error at testToGAN')
        return -1
    else:
        path = os.path.join(path, files[0])
    data1 = pd.read_csv(path, sep=None, header=None, dtype=np.float64, engine='python',
                        encoding='utf-8')  # ,nrows=64*64*100
    data = data1.values.astype(np.float64)
    # data = np.reshape(data, (-1, 64, 22))
    file = os.path.basename(path)
    # print('{} has data :ndim:{} dtype:{} shape:{}'.format(file,data.ndim, data.dtype, data.shape))
    print('{} has data shaped:{}'.format(file, data.shape))
    rows = data.shape[0]
    start = 0
    end = rows
    row = int(rows // 64)
    if mark:
        if mark == 'test':
            start = int(((rows * 0.8) // 64) * 64)
            row = int((rows - start) // 64)
            end = int(start + ((rows - start) // 64) * 64)
        elif mark == 'train':
            row = int((rows * 0.8) // 64)
            end = int(row * 64)
    else:
        print('arise error at testToGAN parameter mark')

    data = data[start:end, ].reshape((-1, 64, 21))
    # num = data.shape[0]
    # print('num:',)
    flag = np.zeros((row,1)).tolist()
    print('{} start at:{} acquires data shape{},flag numbers:{} done read files!!!\n'.format(file, start,data.shape,len(flag)))
    return row, flag, data
