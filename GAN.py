# -*- coding: utf-8 -*-
# @Time   : 19-4-22 下午4:03
# @Author : gjj
# @contact : adau22@163.com ============================
# my github:https://github.com/TerYang/              ===
# all rights reserved                                ===
# good good study,day day up!!                       ===
# ======================================================
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
import torchvision.transforms
from torchvision import datasets, transforms
from readDataToGAN import *
from tensorboardX import SummaryWriter
import json
from utils import *

class generator(nn.Module):
    """
       convtranspose2d
        #   1.  s =1
        #     o' = i' - 2p + k - 1
        #   2.  s >1
        # o = (i-1)*s +k-2p+n
        # n =  output_padding,p=padding,i=input dims,s=stride,k=kernel
    """
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * (8 * 4)),
            # nn.BatchNorm1d(64 * (8 * 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4,2), 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4,3), 2, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.output_dim, (4,3), 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        # x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = x.view(-1, 64, 8,4)
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 16, (4,2), 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, (4, 2), 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (4,2), 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            # nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.Linear(64 * ( 8* 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        # x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = x.view(-1, 64 * (8 * 4))
        x = self.fc(x)

        return x

class GAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        # self.dataset = args.dataset
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62

        # load dataset
        # self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        # data = self.data_loader.__iter__().__next__()[0]

        """dataset"""
        # self.data_loader = testToGAN(self.dataset,'train')
        self.data_loader = DataloadtoGAN(self.dataset,'train')
        self.valdata = DataloadtoGAN(self.dataset,'validate')
        # 重置dataset
        self.dataset = 'trainAgainLR'
        data = next(iter(self.data_loader ))[0]
        # print('data.shape:',data.shape)#data.shape: torch.Size([64, 1, 64, 21])

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # lr_scheduler
        self.G_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.G_optimizer, mode='max', factor=0.1, patience=4, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)
        self.D_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.D_optimizer, mode='max', factor=0.1, patience=4, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')


        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

        self.writer = SummaryWriter()#log_dir=log_dir,
        self.X = 0

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()#Sets the module in training mode
        print('GAN training start!!,epoch:{},module stored at:{}'.format(self.epoch,self.dataset))

        # print('training start!!')
        start_time = time.time()
        # stored_url = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/models/attack_free/GAN'
        # url = os.path.join(self.save_dir, self.dataset, self.model_name)

        for epoch in range(self.epoch):
        # for epoch in range(110,self.epoch):

        #     if epoch == 0:
        #         self.G = torch.load(os.path.join(url,'GAN_110_G.pkl'))
        #         self.D = torch.load(os.path.join(url,'GAN_110_D.pkl'))
        #         print('reload success!','*'*40)

            self.G.train()
            self.D.train()

            epoch_start_time = time.time()
            # for iter, (x_, _) in enumerate(self.data_loader):
            for iter, x_, in enumerate(self.data_loader):
                x_ = x_[0]
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()),end=',')
                    print('lr_G:%.10f,lr_D:%.10f'%(self.G_optimizer.param_groups[0]['lr'],self.D_optimizer.param_groups[0]['lr']))
                    self.writer.add_scalar('G_loss', G_loss.item(), self.X)
                    # writer.add_scalar('G_loss', -G_loss_D, X)
                    self.writer.add_scalar('D_loss', D_loss.item(), self.X)
                    self.writer.add_scalars('cross loss', {'G_loss': D_loss.item(),
                                                      'D_loss': D_loss.item()}, self.X)
                    self.X += 1

            # schedule D lr
            acc_D = validate(self.D,None,self.valdata)#model,dataload,torch.tensor
            self.D_scheduler.step(acc_D)
            self.D.train()
            # schedule G lr
            acc_G = self.validate_G(self.valdata.shape[0])
            self.G_scheduler.step(acc_D)
            self.G.train()


            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            if epoch %5==0:
                self.load_interval(epoch)
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        with open(os.path.join(save_dir, self.model_name + '_train_hist.json'), "a") as f:
            json.dump(self.train_hist, f)

        self.writer.export_scalars_to_json(os.path.join(save_dir, self.model_name + '.json'))
        self.writer.close()

        self.load_interval(epoch)

        # self.save()
        #utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 #self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        # model.state_dict().keys()
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # .state_dict()只是把模型所有的参数都以OrderedDict的形式存下来
        # for key, v in enumerate(pretrained_net):
        #     print key, v
        # 通过打印，遍历
        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
        # 保存训练过程所有时间 loss参数
        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        """
        func:用来加载模型参数
        :return:
        """
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def load_interval(self,epoch):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))

    def validate_G(self,size):
        # schedule G lr
        self.G.eval()
        acc_G = 0
        sum_all = 0
        for i in range(size // 64):
            z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                x_, z_ = x_.cuda(), z_.cuda()
            G_ = self.G(z_)
            # print('G_.shape:', G_.shape)
            D_fake = self.D(G_)
            # print(D_fake.__class__)
            D_fake = np.squeeze(D_fake.data.numpy(), axis=1)
            # D_fake = D_fake.tolist()
            f = lambda x: 1 if x > 0.5 else 0
            ll = list(map(f, D_fake.tolist()))
            acc_G += ll.count(1)
            sum_all += len(ll)
        zeros = sum_all - acc_G
        ones = acc_G
        print('validate size:%d,zeros:%d,ones:%d' % (sum_all, zeros, ones), end=',')
        print('acc:%.6f,judged as 1.' % (ones / sum_all))
        return ones / sum_all