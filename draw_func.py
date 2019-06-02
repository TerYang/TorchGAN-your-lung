# -*- coding: utf-8 -*-
# @Time   : 19-4-24 下午9:45
# @Author : TerYang
# @contact : adau22@163.com ============================
# my github:https://github.com/TerYang/              ===
# all rights reserved                                ===
# good good study,day day up!!                       ===
# ======================================================
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from numpy import ndarray
import seaborn as sns


D_PATH = '/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/reclear/analysis'
# D_PATH = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/tests_data'
# D_PATH = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/tests_data'
# thedataset = 'data'
# result_url = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/draw_pictures'


def draw_func(url,color=None):
    """
    func: draw
    :param url:
    :return:
    "-" (solid) – default
    "--" (dashed)
    "-." (dash dot)
    ":" (dotted)
    "None" or " " or "" (nothing)
        'b'	blue
    'g'	green
    'r'	red
    'c'	cyan
    'm'	magenta
    'y'	yellow
    'k'	black
    'w'	white
    """
    df = pd.read_csv(url, engine='python',names=['p','n','a'])
    df = df.dropna(axis=1,how='all')

    # line = 'solid'
    for n in df.columns:
        if n == 'n':
            line = 'dash dot'
            # color = 'm'
        elif n == 'a':
            line = 'dotted'
            # color = 'b'
        else:
            line = 'solid'
            # color = 'r'
        p = pd.Series(df[n])
        p = p.dropna(axis=0, how=any)
        try:
            plt.plot(p, label=n, color=color, linestyle=line, marker='x')  # 'x'
            # plt.annotate("(%s,%s)" % p, xy=p, xytext=(-20, 10), textcoords='offset points')
            plt.xlabel('Modules:No.')
            plt.ylabel('rate')

        except:
            print(p)
            # print('________________________')
            pass

def draw_dataconstrcuct(url):
    df = pd.read_excel(url, header=None, dtype=np.str,)#skiprows=0,
    df.dropna(axis=0,how='all')
    names = df.iloc[0,:].astype(np.str)
    # names = names[(True-names.isin(['nan']))]
    print(type(names),)
    print(names)
    names.drop(axis=0,)
    print(df.iloc[0,:].values)

    # print(df)

if __name__ == '__main__':

    # urls = [os.path.join(D_PATH,file) for file in os.listdir(D_PATH)]
    # print('文件数目:',len(urls))
    #
    # # print(urls)
    # # exit()
    # draw_dataconstrcuct(urls[0])
    #
    # # import seaborn as sns
    # #
    # # sns.set()
    # # import matplotlib.pyplot as plt
    # # fmri = sns.load_dataset("fmri")
    # # print(type(fmri))
    # # print(fmri)
    # # ax = sns.lineplot(x=fmri.index, y=fmri['signal'], data=fmri)#"timepoint","signal"
    # # plt.show()
    # # exit()
    # """画图,画出测试结果曲线图"""
    # urls = []
    # dires = os.listdir(D_PATH)#modules names
    # dires = sorted(dires)
    # ban = ['ACGAN', 'ACGAN_continue' , 'GAN', 'LSGAN', 'WGAN', 'WGAN_GP']
    #
    # for root, dirs, files in os.walk(D_PATH, topdown=None):
    #     for name in dirs:
    #         if name == 'collection':
    #             urls.append('{}'.format(os.path.join(root,name)))
    # file = [os.path.join(url,file_n) for url in urls for file_n in os.listdir(url) if '_collection.csv' in file_n]
    # file = sorted(file)
    # # print(file)
    # # print(dires)
    # # exit()
    #
    # # colors = ['g','r','b','k']
    #
    # # lights = [.1,.3,.5,.7,.9]
    # # current_palette = sns.color_palette()
    # # print(type(current_palette),len(current_palette))
    # # columns = ['P','N','PF','NF','accurate','recall','F1']
    # # sns.palplot(sns.hls_palette(8, l=.3, s=.8))
    # sns.set()
    # path_ = os.path.join(result_url,thedataset)
    # if not os.path.exists(path_):
    #     os.makedirs(path_)
    # for filefolder in dires:#tests　目錄下，每一個模型
    #     if filefolder in ban:
    #         continue
    #
    #     for f in file:
    #         if filefolder in f:
    #             t = plt.figure()
    #             path = os.path.join(path_, filefolder + '/'+os.path.splitext(os.path.basename(f))[0]+ '_loss.png')
    #             if not os.path.exists(os.path.join(path_, filefolder)):
    #                 os.makedirs(os.path.join(path_, filefolder))
    #             df = pd.read_csv(f, engine='python',header=None,dtype=np.str)
    #             # print(df)
    #             columns = df.iloc[0,:].values.tolist()[1:]
    #             # print(columns)
    #             # df = pd.read_csv(f, engine='python',names=columns,header=None)
    #             df = df.iloc[1:,:]
    #             indexs_ = df[0].values.tolist()
    #             # print(indexs_)
    #             df = df.iloc[:,1:].values.astype(np.float64)
    #             # print(df)
    #             df = pd.DataFrame(df,index=indexs_,columns=columns,dtype=np.float64)#
    #             # print(df)
    #             df = df.dropna(axis=1,how='all')
    #             df = df.dropna(axis=0,how='any')
    #             # sns.set(style="whitegrid")
    #             # print(df)
    #             # print(df.columns)
    #             n = len(df.columns)
    #             df.plot()#ok ok ok ok ok ok
    #             plt.legend(loc=4)
    #             plt.savefig(path)
    #             plt.clf()
    #             # plt.cla()
    #             plt.close('all')
    #             # print(df.columns)
    #             # print(df)
    #             # palette = sns.cubehelix_palette(light=light, n_colors=n)
    #             # ax = sns.lineplot(x=df.index[:-1], y=df['N'], data=df[:-1,:])  # "timepoint","signal"
    #
    #             # sns.lineplot(data=df, palette="tab10", linewidth=2.5)
    #
    #             # ax = sns.lineplot(x=df.index, y=df[df.columns], data=df,palette="tab10", linewidth=2.5)#"timepoint","signal"
    #             # print(df)
    #             # exit()
    #             # sns.lineplot(data=df, palette="tab10", linewidth=2.5)
    #             # sns.lineplot(data=df, palette=palette, linewidth=2.5,hue="coherence", style="choice",)
    #             # plt.tight_layout()
    #             # plt.show()
    #
