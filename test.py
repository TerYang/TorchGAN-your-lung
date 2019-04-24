import sys
import os
# sys.path.append('../')
import numpy as np
import torch
import time
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.autograd import Variable
from readDataToGAN import *
from GAN import *
########################################################################################################
# module_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-mind/Nets/Dmodule/2019-04-18/module/'
module_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/models/attack_free/GAN/'
# test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"
test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/data/"
# test_addr = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata'
result_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/test'
# test for normal and attack data you should change the readfile func named testdata of testNormal ##############
################### change above parameters while test on different net and for difference data(attack,normal)####


def writelog(content,url=None):
    # a = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-mind/Nets/full/2019-04-17/test_logs/'
    if url == None:
        print(content)
    else:
        collect_url = './collection/{}'.format(url)
        if not os.path.exists(collect_url):
            os.makedirs(collect_url)
        url = os.path.join(collect_url,'_log.txt')
        with open(url, 'a', encoding='utf-8') as f:
            f.writelines(content + '\n')
            print(content)


# def test(path, filename,logmark,num, flags, test):
def test(path, filename,logmark,file, flags, test):
    """
    func: test data runs at every pkl(module)
    :param path: pkl(module) path
    :param logmark: pkl mark,determinate which module
    :param num: test data rows
    :param flags: flag of every test data
    :param test: test data
    :return:no
    """
    # file = os.path.splitext(filename)[0]

    t1 = time.time()
    writelog('',file)
    #Label 0 means normal,size 1*BATCH
    # Label 1 means anormal,size 1*BATCH
    """处理"""
    test = np.expand_dims(test, axis=1)
    Test_data = Variable(torch.from_numpy(test).float())

    # gan 保存模型(有两种保存方法)不一样,和cnn 只保存了模型
    if 'GAN_D.pkl' in path:
        Dnet = discriminator()
        Dnet.load_state_dict(torch.load(path))
    else:
        Dnet = torch.load(path)

    # print(type(Dnet))

    Results = Dnet(Test_data)
    Results = Results.data.numpy()
    # print('Results.shape:{},type size:{}'.format(Results.shape,type(Results)),Results[0])
    # print(np.shape(Results))
    # print(Results[988:1100,0])
    TP = 0  # 真当假
    TN = 0  # 假当正
    NP = 0
    NN = 0
    # print(type(Results.tolist()))
    """
    Precision：P=TP/(TP+FP)
    Recall：R=TP/(TP+FN)
    F1-score：2/(1/P+1/R)
    ROC/AUC：TPR=TP/(TP+FN), FPR=FP/(FP+TN)
    """
    for flag, pre in list(zip(flags,Results.tolist())):
        # print(flag,pre)
        if flag:
            if pre[0] > 0.5:
                TP += 1
            else:
                TN += 1
        else:
            if pre[0] > 0.5:
                NP += 1
            else:
                NN += 1
    # print(TP,TN,NP,NN)
    # print(type(flags),flags[-100:])
    # results = {}
    res = []
    try:
        PT = TP/(TN+TP)
        res.append('{}'.format(PT))
    except ZeroDivisionError:
        res.append('NA')

    try:
        NT = NN/(NN+NP)
        res.append('{}'.format(NT))
    except ZeroDivisionError:
        # writelog('have no P(normaly event)',file)
        res.append('NA')

    try:
        accurate = (TP+NN)/len(flags)
        # results['accurate'] = accurate
        res.append('{}'.format(accurate))
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res.append('NA')

    """csv header 'PT','NT','accurate'"""
    # for key in ['PT','NT','accurate']:
    #     try:
    #         res.append(results[key])
    #     except KeyError:
    #         res.append(0)

    # print(PT,NT,accurate)
    t2 = time.time()
    # print("crect number: {},total Accuracy rate:{}".format(crect, crect / (num + flags)))

    # text = ' '
    # for key, item in results.items():
    #     text += key + ':' + str(item) + ' '
    text = 'PT: ' + res[0] +'\tNT: ' + res[1] +'\taccurate: ' + res[2]
    writelog(text,file)
    writelog('test case: {} had finshed module:{}'.format(filename,logmark),file)
    writelog('time test spent :{}'.format(t2 - t1), file)
    writelog('*'*40,file)

    detail_url = './detail/{}'.format(os.path.splitext(filename)[0])
    if not os.path.exists(detail_url):
        os.makedirs(detail_url)
    url_numpy = os.path.join(detail_url,'module_{}'.format(logmark))
    x = np.concatenate((Results,np.array(flags).reshape((-1,1))),axis=1)
    np.savetxt(url_numpy,x,delimiter=',',fmt='%s')

    return res


def getModulesList(modules_path,mark='old-gan'):
    modules = os.listdir(modules_path)

    num_seq = []
    final_module = ''
    flagOfD = 0
    if mark:
        if mark == 'new-gan':
            final_module = 'GAN_D.pkl'
            new_modul = []
            for module in modules:
                if '_D.pkl' in module:
                    if module == final_module:
                        flagOfD = 1
                    new_modul.append(module)
            # print('new:', new_modul)

        elif mark == 'old-gan':
            final_module = 'Net_D.pkl'
            for module in modules:
                if '.pkl' in module:
                    if module == final_module:
                        flagOfD = 1
                        modules.pop(modules.index(final_module))
                        continue

                else:
                    modules.pop(modules.index(module))
    else:
        print('error arised at getModulesList parameter mark!')
    modules = new_modul
    # print(modules)
    for module in modules:
        if mark == 'new-gan':
            if final_module == module:
                pass
            else:
                num_seq.append(module[module.index('_') + 1:module.index('_D')])

        elif mark == 'old-gan':
            num_seq.append(module[module.index('_') + 1:module.index('.')])

    # print(num_seq)
    num_seq = list(map(int,num_seq))
    # sort_seq = map(int,num_seq.copy())
    sort_seq = sorted(num_seq)
    # print(modules)
    # print(num_seq)
    modules_url = []
    for s in sort_seq:
        modules_url.append(os.path.join(modules_path,modules[num_seq.index(s)]))
    sort_seq = list(map(lambda x: str(x),sort_seq))
    if flagOfD:
        modules_url.append(os.path.join(modules_path,final_module))
        sort_seq.append('D')
    # print(sort_seq)
    # print(len(modules_url),len(sort_seq))
    return modules_url, sort_seq


def multitest(path):

    test_path = path[0]#added 4.20
    module_url = path[1]#added 4.20

    # 选择合适的mark
    module_urls, seqs = getModulesList(module_url,mark='new-gan')
    # print(module_urls)
    # return
    test_file_name = os.path.basename(test_path)
    file = os.path.splitext(test_file_name)[0]

    writelog('start test file: {},run at:{}'.format(test_file_name,time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))),file)
    """检验攻击数据"""
    t_num, t_flag, tests = testdata(test_path,mark='test')  # ,num=64*rows test_normal,test_anormal
    # """检验正常数据"""
    # t_num, t_flag, tests = testNormal(os.path.split(test_path)[0],mark='test')  # ,num=64*rows test_normal,test_anormal

    writelog("test data: {},rows:{}".format(test_file_name, t_num),file)
    # lable 中是否出现全1,或者全0的情况
    try:
        t_flag.index(1.)
    except ValueError:
        writelog("test data : {},has no Positive Points(PP)".format(test_file_name), file)

    try:
        t_flag.index(0.)
    except ValueError:
        writelog("test data : {},has no Negative Points(NP)".format(test_file_name), file)
    # exit()
    # ress = np.empty((1,3))
    ress = []
    for i, url in list(zip(seqs,module_urls)):
        # if 'GAN_D.pkl' not in url:
        #     print('dict:',url)
            tes = test(url, test_file_name, i, file, t_flag, tests)
            ress.append(tes)

    ress = np.array(ress).reshape((-1,3))

    try:
        ress = np.concatenate((np.array(seqs).reshape((-1,1)).astype(np.str),ress),axis=1)
    except ValueError:
        print('--------------',ress.shape, len(seqs),'--------------------------')
    collect_url = './collection/{}'.format(file)
    if not os.path.exists(collect_url):
        os.makedirs(collect_url)
    csv_url = os.path.join(collect_url,file+'_collection.csv')
    # print(csv_url)
    np.savetxt(csv_url,ress,fmt='%s',delimiter=',',encoding='utf-8')


if __name__ == '__main__':
    """
    需要改变os.chdir参数、模型地址 module_path，测试地址test_addr
    """
    """test GAN"""
    # module_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-mind/Nets/full/2019-04-17/GANmodule/D/'
    # test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"
    #
    # # module_urls, seqs = getModulesList(module_path)
    # # t_nors, t_anors, tests = get_data(keyword='test')  # ,num=64*rows test_normal,test_anormal
    # test_urls = [os.path.join(test_addr,file) for file in os.listdir(test_addr)]
    #
    # # print(test_urls)
    # # exit()
    # # t_num, t_flag, tests = testdata(test_urls[0])  # ,num=64*rows test_normal,test_anormal
    #
    # # for i,url in list(zip(seqs,module_urls)):
    # #     file = os.path.basename(url)
    # #     test(url,file,i,t_num, t_flag, tests)
    # pool = mp.Pool(processes=len(test_urls))
    # pool.map(multitest,test_urls)
    # pool.close()
    # pool.join()

    """test Disciminor"""


    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        result_path = result_path +'_{}'.format('new')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    test_urls = [os.path.join(test_addr,file) for file in os.listdir(test_addr)]

    os.chdir(result_path)

    # print('module path:',module_path)
    # print('test data path:',test_addr)
    # print('result location:',result_path)
    # exit()

    # print(test_urls[0])
    # multitest((test_urls[0],module_path))
    # exit()
    lens = len(test_urls)
    pool = mp.Pool(processes=lens)
    pool.map(multitest,zip(test_urls,[module_path for _ in range(lens)]))
    pool.close()
    pool.join()

