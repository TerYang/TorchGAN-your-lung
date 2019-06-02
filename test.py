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
import re
from ACGAN import discriminator as ad

# module_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/models/attack_free_begin_at105'
# module_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/models/attack_free'
module_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/models/retrain'
test_addr = '/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/encoding/data'
#scaler to (-1,1),normal marked R,transfered to 0,attack marked T transfered to 1

# test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"#normal marked R,transfered to 0,attack marked T transfered to 1
# test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/data/"#normal marked R,transfered to 1,attack marked T transfered to 0
gan_addr = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata'
result_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/tests_retrain'
# test for normal and attack data you should change the readfile func named testdata of testNormal ##############
################### change above parameters while test on different net and for difference data(attack,normal)####
columns_ = ['P','N','F1','accurate','recall','PF','NF']

def writelog(content,url=None):
    # a = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-mind/Nets/full/2019-04-17/test_logs/'
    if url == None:
        print(content)
    else:
        collect_url = './logs'
        if not os.path.exists(collect_url):
            os.makedirs(collect_url)
        url = os.path.join(collect_url,'{}_log.txt'.format(url))
        with open(url, 'a', encoding='utf-8') as f:
            f.writelines(content + '\n')
            print(content)


# def test(path, filename,logmark,num, flags, test):
def test(path, logmark, file, flags, test):
    """
    func: test data runs at every pkl(module)
    :param path: pkl(module) path
    :param logmark: pkl mark,determinate which module
    :param num: test data rows
    :param flags: flag of every test data
    :param test: test data
    :return:no
    #Label 0 means normal,size 1*BATCH
    # Label 1 means anormal,size 1*BATCH

    """
    # file = os.path.splitext(file)[0]
    t1 = time.time()
    writelog('',file)
    """处理"""
    test = np.expand_dims(test, axis=1)
    Test_data = Variable(torch.from_numpy(test).float())

    pattern = re.compile(r'\d+\.?\d*')
    jf = pattern.findall(path)
    # gan 保存模型(有两种保存方法)不一样,和cnn 只保存了模型
    # if 'GAN_D.pkl' in path:
    #     Dnet = discriminator()
    #     Dnet.load_state_dict(torch.load(path))
    # else:
    #     Dnet = torch.load(path)
    modulename = ''
    result = np.empty((2, 1))
    if len(jf):
        Dnet = torch.load(path)
        # 直接加载模型
    else:
        # 加载模型后,加载变量
        a = os.path.basename(path)
        c = os.path.splitext(a)[0]
        pattern = re.compile(r'.*?(\S+)\_\d*')
        jj = pattern.findall(c)
        modulename = jj[0]
        if jj[0] == 'ACGAN' or 'ACGAN' in jj[0]:
            Dnet = ad()
        elif jj[0] == 'GAN':
            Dnet = discriminator()
        else:
            Dnet = discriminator()
        try:
            Dnet.load_state_dict(torch.load(path))
        except:
            print(path,jf,jj,modulename,'*-'*40)
    # print(type(Dnet))
    try:
        Results = Dnet(Test_data)
        result = Results.data.numpy()
    except:
        try:
            Results, _ =  Dnet(Test_data)
            result = Results.detach().numpy()
        except:
            print('path:',path,'file:',file)
    Results = result.copy()
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
    # print(Results.tolist(),Results.tolist()[0],'*-'*40)
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
    res = {}
    # 1 正确检出 正例的概率
    try:
        res['P']='{}'.format(TP/(TN+TP))
    except ZeroDivisionError:
        res['P'] = 'NA'
    # 2 正确检出 负例的概率
    try:
        res['N']='{}'.format(NN/(NN+NP))
    except ZeroDivisionError:
        # writelog('have no P(normaly event)',file)
        res['N'] = 'NA'
    # 3 正例 检测出错的概率
    try:
        res['PF']='{}'.format(TN/(TN+TP))
    except ZeroDivisionError:
        res['PF'] ='NA'
    # 4 负例 检测出错的概率
    try:
        res['NF'] ='{}'.format(NP/(NN+NP))
    except ZeroDivisionError:
        # writelog('have no P(normaly event)',file)
        res['NF'] ='NA'
    # 5 accurate
    try:
        res['accurate'] = (TP+NN)/len(flags)
        # results['accurate'] = accurate
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['accurate'] ='NA'
    # 6 recall
    try:
        res['recall'] = TP/(TP+NP)
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['recall'] = 'NA'
    # 7 F1
    if res['P'] == 'NA' or res['recall'] == 'NA':
        res['F1'] = 'NA'
    else:
        try:
            res['F1'] ='{}'.format(2/(1/np.float64(res['P']) +1/np.float64(res['recall'])))
        except RuntimeWarning or RuntimeError or ZeroDivisionError or ValueError or TypeError:
            res['F1'] = 'NA'
            print('P:{},R:{}'.format(res['P'],res['recall']))
    t2 = time.time()
    text = ''
    for key, item in res.items():
        text += key + ':' + str(item) + ','
    writelog(text,file)
    writelog('test case: {} had finshed module:{}'.format(file,logmark),file)
    writelog('time test spent :{}'.format(t2 - t1), file)
    writelog('*'*40,file)

    detail_url = './detail/{}'.format(file)
    if not os.path.exists(detail_url):
        os.makedirs(detail_url)
    url_numpy = os.path.join(detail_url,'module_{}.csv'.format(logmark))
    try:
        x = np.concatenate((Results,np.array(flags).reshape((-1,1))),axis=1).astype(np.str)
    except ValueError:
        try:
            x = np.concatenate((Results, np.array(flags[:Results.shape[0]]).reshape((-1, 1))), axis=1).astype(np.str)
        except:
            pass
        # x = np.concatenate((Results.detach().numpy(), np.array(flags).reshape((-1, 1))), axis=1).astype(np.str)
        writelog('Error at module {} No.{},file name:{},flag shape:{},Results shape:{}'.
                 format(modulename,logmark,file,len(flags),Results.shape),file)
    try:
        x = np.concatenate((np.array(['Results', 'label']).reshape((-1, x.shape[1])), x), axis=0).astype(np.str)
    except:
        print('cannt add header(columns) at func test')
        pass
    np.savetxt(url_numpy,x,delimiter=',',fmt='%s')

    return res


def getModulesList(modules_path,mark='old-gan'):
    """
    func: sort different modules saved at different epoch,sorted name list
    :param modules_path:
    :param mark:
    :return: different module file url(address) saved at different epoch,name list
    """
    modules = os.listdir(modules_path)
    pattern = re.compile(r'\d+\.?\d*')
    num_seq = []
    # final_module = ''
    # flagOfD = 0
    new_modul = []
    for module in modules:
        if '_D.pkl' in module:
            new_modul.append(module)
    # print(new_modul)
    # print(len(new_modul))
    # return
    # if mark:
    #     if mark == 'new-gan':
    #         final_module = 'GAN_D.pkl'
    #         new_modul = []
    #         for module in modules:
    #             if '_D.pkl' in module:
    #                 if module == final_module:
    #                     flagOfD = 1
    #                 new_modul.append(module)
    #
    #     elif mark == 'old-gan':
    #         final_module = 'Net_D.pkl'
    #         for module in modules:
    #             if '.pkl' in module:
    #                 if module == final_module:
    #                     flagOfD = 1
    #                     modules.pop(modules.index(final_module))
    #                     continue
    #
    #             else:
    #                 modules.pop(modules.index(module))
    # else:
    #     print('error arised at getModulesList parameter mark!')
    modules = new_modul
    # print(modules)
    mark_D = 0
    for i,module in enumerate(modules):
        # if mark == 'new-gan':
        #     if final_module in module:
        #         pass
        #     else:
        #         num_seq.append(module[module.index('_') + 1:module.index('_D')])
        #
        # elif mark == 'old-gan':
        #     num_seq.append(module[module.index('_') + 1:module.index('.')])

    # print(num_seq)
        jf = pattern.findall(module)
        if len(jf):
            num_seq.append(jf[0])
        else:
            # 这里必须要取值大于epoch
            num_seq.append('100000')
            mark_D = i

    num_seq = list(map(int,num_seq))
    # sort_seq = map(int,num_seq.copy())
    sort_seq = sorted(num_seq)
    modules_url = []
    for s in sort_seq[:-1]:
        modules_url.append(os.path.join(modules_path,modules[num_seq.index(s)]))

    modules_url.append(os.path.join(modules_path,modules[mark_D]))
    sort_seq = list(map(lambda x: str(x),sort_seq))
    # sort_seq.append('D')
    sort_seq[-1] = 'D'
    return modules_url, sort_seq


def multitest(path):
    """
    func: the base of cycling all test file through multilprocessing pool.generates test collection of all modules' test
    result type of csv and history file type of txt
    :param path: test data path and module located addresss
    :return: None
    """
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
        writelog("test data : {}, label 1:".format(file,t_flag.count(1.)), file)
    except ValueError:
        writelog("test data : {}, has no label 1".format(file), file)

    try:
        t_flag.index(0.)
        writelog("test data : {}, label 0:".format(file,t_flag.count(0.)), file)
    except ValueError:
        writelog("test data : {}, has no label 0".format(file), file)
    # exit()
    # ress = np.empty((1,3))
    ress = []
    f = 0
    lens = 0
    names = []
    for i, url in list(zip(seqs,module_urls)):
        # if 'GAN_D.pkl' not in url:
        #     print('dict:',url)
            tes = test(url, test_file_name, i, file, t_flag, tests)#return dict
            if f:
                pass
            else:
                lens = len(tes.keys())
                names = list(tes.keys())
                f = 1
            for key, item in tes.items():
                ress.append(item)
    ress = np.array(ress).reshape((-1,lens))
    f = 0
    try:
        ress = np.concatenate((np.array(seqs).reshape((-1,1)).astype(np.str),ress),axis=1)
    except ValueError:
        print('--------------',ress.shape, len(seqs),'--------------------------')
        f = 1
    if not f:
        try:
            names.insert(0,'\\')
            ress = np.concatenate((np.array(names).reshape((1,-1)).astype(np.str),ress),axis=0)
        except:
            print('names',names,'len',len(names))
            print('ress',ress.shape)

    collect_url = './collection'
    if not os.path.exists(collect_url):
        os.makedirs(collect_url)
    csv_url = os.path.join(collect_url,file+'_collection.csv')
    # print(csv_url)
    np.savetxt(csv_url,ress,fmt='%s',delimiter=',',encoding='utf-8')


def improve_multil_test(path):
# def improve_multil_test(t1,t2,t3):
    """
    func: the base of cycling all test file through multilprocessing pool.generates test collection of all modules' test
    result type of csv and history file type of txt
    :param path: test data , module located addresss, data file name
    :return: None
    """

    t_num, t_flag, tests = path[0]#added 4.20
    # t_num, t_flag, tests = t1#added 4.20
    module_url = path[1]#added 4.20
    # module_url = t2#added 4.20
    file = path[2]
    # file = t3

    # 选择合适的mark
    module_urls, seqs = getModulesList(module_url,mark='new-gan')

    collect_url = './collection'
    # result_url = './{}'.format(file)
    # if not os.path.exists(result_url):
    #     os.makedirs(result_url)
    # else:
    #     result_url = result_url + '_t'
    #     os.makedirs(result_url)
    # os.chdir(result_url)

    writelog('start test file: {},run at:{}'.format(file,time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))),file)
    """检验攻击数据"""
    # t_num, t_flag, tests = testdata(test_path,mark='test')  # ,num=64*rows test_normal,test_anormal
    # """检验正常数据"""
    # t_num, t_flag, tests = testNormal(os.path.split(test_path)[0],mark='test')  # ,num=64*rows test_normal,test_anormal

    writelog("test data: {}, size:{}".format(file, t_num),file)
    # lable 中是否出现全1,或者全0的情况
    try:
        t_flag.index(1.)
        writelog("test data : {}, label 1:{}".format(file,t_flag.count(1.)), file)
    except ValueError:
        writelog("test data : {}, has no label 1".format(file), file)

    try:
        t_flag.index(0.)
        writelog("test data : {}, label 0:{}".format(file,t_flag.count(0.)), file)
    except ValueError:
        writelog("test data : {}, has no label 0".format(file), file)
    ress = []
    # f = 0
    # lens = 0
    # names = []
    for i, url in list(zip(seqs,module_urls)):
        # if 'GAN_D.pkl' not in url:
        #     print('dict:',url)
        tes = test(url, i, file, t_flag, tests)#return dict
        for elem in columns_:
            # 按照给定的顺序排列
            ress.append(tes[elem])
        # if f:
        #     pass
        # else:
        #     lens = len(tes.keys())
        #     names = list(tes.keys())
        #     f = 1
        # for key, item in tes.items():
        #     ress.append(item)

    ress = np.array(ress).reshape((-1,len(columns_)))
    f = 0
    try:
        ress = np.concatenate((np.array(seqs).reshape((-1,1)).astype(np.str),ress),axis=1)
    #     add module No.
    except ValueError:
        print('--------------',ress.shape, len(seqs),'--------------------------')
        f = 1
    if not f:
        try:
            cols = columns_.copy()
            cols.insert(0,'\\')
            ress = np.concatenate((np.array(cols).reshape((1,-1)).astype(np.str),ress),axis=0)
        #     add the parameter name
        except:
            print('names',cols,'len',len(cols))
            print('ress',ress.shape)
    else:
        try:
            ress = np.concatenate((np.array(columns_).reshape((1, -1)).astype(np.str), ress), axis=0)
            #     add the parameter name
        except:
            print('varify', ress.shape, len(seqs), 'is not equal')

    if not os.path.exists(collect_url):
        os.makedirs(collect_url)
    csv_url = os.path.join(collect_url,file+'_collection.csv')
    # print(csv_url)
    np.savetxt(csv_url,ress,fmt='%s',delimiter=',',encoding='utf-8')
    # 保存该GAN在当前数据集下,训练阶段保存的模型

if __name__ == '__main__':
    """
    需要改变os.chdir参数、模型地址 module_path，测试地址test_addr,默认为选择测试攻击数据,
    如果测试正常数据,需要更换读取数据的函数为testNormal
    """
    """test GAN"""
    # module_path = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-mind/Nets/full/2019-04-17/GANmodule/D/'
    # test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"
    #
    # # module_urls, seqs = getModulesList(module_path)
    # # t_nors, t_anors, tests = get_data(keyword='test')  # ,num=64*rows test_normal,test_anormal
    # test_urls = [os.path.join(test_addr,file) for file in os.listdir(test_addr)]
    #
    print('数据集位置:',test_addr)
    print('结果存放位置:',result_path)
    print('模型路径:',module_path)
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
    # ban = 'ACGAN'
    # ban = ['WGAN','ACGAN','WGAN_GP','LSGAN','GAN','ACGAN_continue']

    """test Disciminor"""
    test_urls = [os.path.join(test_addr, file) for file in os.listdir(test_addr)]#test data
    file_names = [os.path.splitext(file)[0] for file in os.listdir(test_addr)]# test data name
    test_data = [testdata(test_url,mark='test') for test_url in test_urls]#test data,mark='test'

    for root, dirs, files in os.walk(module_path, topdown=None):
        for name in dirs:
            # if name in ban:
            #     continue
            module_url = os.path.join(root, name)

            # print(module_url)
            # continue
            # test path and module path renew
            result_url = os.path.join(result_path,name)
            # result stored at dataset + module name
            if not os.path.exists(result_url):
                os.makedirs(result_url)
            else:
                result_url = result_url +'_{}'.format('new')
                if not os.path.exists(result_url):
                    os.makedirs(result_url)
            os.chdir(result_url)
            # print('module path:',module_path)
            # improve_multil_test(test_data[0],module_url,file_names[0])
            # exit()
            lens = len(test_urls)
            pool = mp.Pool(processes=lens)
            # pool.map(multitest,zip(test_urls,[module_url for _ in range(lens)]))
            pool.map(improve_multil_test,zip(test_data,[module_url for _ in range(lens)],file_names))
            pool.close()
            pool.join()

