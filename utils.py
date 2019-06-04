# -*- coding: utf-8 -*-
# @Time   : 19-4-22 下午4:06
# @Author : gjj
# @contact : adau22@163.com ============================
# my github:https://github.com/TerYang/              ===
# copy from network                                  ===
# good good study,day day up!!                       ===
# ======================================================
import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def adjust_learning_rate(optimizer, epoch, val, lr):
    '''
    fun:Sets the learning rate to the initial LR decayed by 10 every val epochs
    :param optimizer: 优化器
    :param epoch: 当前epoch
    :param val: epoch 设置间隔
    :param lr: 初始学习率
    :return: None
    '''
    lr *= 0.1 ** (epoch // val)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def load_interval(self,epoch):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))

def validate(model,data_loader=None,data=None,label=None):#,mark='validate'
    '''
    validate at the same time as training
    func:select data and label or data_loader
    :return:
    '''
    import math
    f1 = lambda l,r:1 if math.fabs(l-r)<0.5 else 0
    model.eval()
    # 为测试
    if data_loader is not None:
        ones = 0
        zeros = 0
        iter = 0
        # 默认不带标签的验证集数据都不需要传入data_loader,传入data 就可以
        flag = 0
        for iter, x_, in enumerate(data_loader):
            # 带标签
            if x_.__len__ == 2:
                l_ = x_[1]
                x_ = x_[0]
                try:
                    D_real,_ = model(x_)
                except:
                    D_real = model(x_)

                # 单通道(1,1,64,21)
                try:
                    if l_.size(dim=0) == 1:
                        l_ = l_.item()
                        if l_ == 0 or l_ == 0.:
                            if D_real.item() < 0.5:
                                ones += 1
                            else:
                                zeros += 1
                        else:
                            if D_real.item() >= 0.5:
                                ones += 1
                            else:
                                zeros += 1
                    # 多通道(64,1,64,21)
                    else:
                        l_ = l_.data.numpy().tolist()
                        D_real = D_real.data.numpy().tolist()
                        ll = list(map(f1, l_, D_real))
                        zeros += ll.count(0)  # 错误判定
                        ones += ll.count(1)  # 正确判定
                except:
                    print('l_:',l_.__class__,l_.shape)

            # 不带标签,默认数据为normal 数据集,即认为,识别为0,即为识别出正常情况
            elif x_.__len__ == 1:

                # 没有标签默认为你normal 数据集,不需要标签
                x_ = x_[0]
                try:
                    D_real,_ = model(x_)
                except:
                    D_real = model(x_)
                try:
                    # 但通道数据
                    if x_.shape[0]==1:
                        flag = 1
                        if D_real.item() <0.5:
                            zeros+=1
                        else:
                            ones += 1

                    # 多通道数据
                    else:
                        l_ = l_.data.numpy().tolist()
                        D_real = D_real.data.numpy().tolist()
                        ll = list(map(f1, l_, D_real))
                        zeros += ll.count(0)  # 错误判定
                        ones += ll.count(1)  # 正确判定
                except:
                    print('x_:',x_.shape,x_.__class__)
        if flag:
            print('validate: D,size%d,zeros:%d,ones:%d'%(iter+1,zeros,ones),end=',')
            print('acc:%.6f,judged as 0.' %(zeros/(iter+1)),end=',')
            return zeros/(iter+1)
        else:
            print('validate: D,size%d,errors:%d,correct:%d'%(iter+1,zeros,ones),end=',')
            print('acc:%.6f,judged as 0.' %(ones/(iter+1)),end=',')
            return ones/(iter+1)
    # 正常
    if data is not None:
        # 带标签
        # a = np.empty((3,1))
        # a.ndim
        if data.ndim == 4:
            pass
        else:
            TraindataM = torch.from_numpy(data).float()  # transform to float torchTensor
            data = torch.unsqueeze(TraindataM, 1)

        try:
            D_real, _ = model(data)
        except:
            D_real = model(data)

        # print(D_real.__class__,D_real.shape)#, D_real.item(), D_real[0]
        # D_real ,= D_real.to_numpy().tolist()
        # D_real = D_real.data.numpy().tolist()

        # print(label.__class__,label.shape)#, D_real.item(), D_real[0]

        if label is not None:
            # print(label.__class__,len(label))#, D_real.item(), D_real[0]

            D_real = D_real.data.numpy()
            D_real = np.squeeze(D_real).tolist()#[[],[],[]]

            ll = list(map(f1, label, D_real))
            zeros = ll.count(0)  # 错误判定
            ones = ll.count(1)  # 正确判定
            print('validate: D,size%d,errors:%d,correct:%d'%(len(ll),zeros,ones),end=',')
            print('acc:%.6f,judged as 0.'%(ones/len(ll)),end=',')
            return ones/(len(ll))
        else:
            # 验证集没有标,认为是normal 数据集,判定为0~0.5之间即可认为正确
            D_real = D_real.data.numpy()
            D_real = np.squeeze(D_real).tolist()#[[],[],[]]

            f = lambda x:1 if x[0] < 0.5 else 0
            ll = list(map(f,D_real))
            zeros = ll.count(0)
            ones = ll.count(1)
            print('validate D,size:%d,zeros:%d,ones:%d'%(len(D_real),zeros,ones),end=',')
            print('acc:%.6f,judged as 0' %(ll.count(0)/len(ll)),end=',')
            return ones/len(ll)


