import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from dataloader import dataloader
from readDataToGAN import *
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from utils import *
import json

"""使用有標籤的攻擊數據進行訓練,
    new_data
    reclear->encodingdata
    addr:'/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/encoding'
"""
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * (8* 4) ),
            # nn.BatchNorm1d(64 * ( 8* 4) ),
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

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 64, 8, 4)
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=2):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 16, (4,2), 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, (4,2), 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (4, 2), 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * ( 8* 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 64 * ( 8* 4))
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c

class ACGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 60
        self.class_num = 2
        self.sample_num = self.class_num ** 2
        self.model_name = 'ACGANwithStep'
        # load dataset
        # self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        # data = self.data_loader.__iter__().__next__()[0]
        """train data"""
        # self.data_loader = testToGAN(self.dataset,'train')
        print('---------- read train dataset -------------')
        flags,data =  getTrainDiscriminor(self.dataset,'train')

        TraindataM = torch.from_numpy(data).float()  # transform to float torchTensor
        TraindataL = torch.from_numpy(np.array(flags).reshape((-1,1))).float()
        TraindataM = torch.unsqueeze(TraindataM, 1)
        # print(TraindataM.shape,TraindataL.shape)
        TorchDataset = Data.TensorDataset(TraindataM,TraindataL)
        self.data_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
        print('run at {},with {} epochs,data size {}'.format(self.model_name,self.epoch,TraindataL.shape[0]))
        print('-----------------------------------------------')


        print('---------- read validate data -------------')
        self.valiflags ,self.validata =  getTrainDiscriminor(self.dataset,'validate')

        TraindataM = torch.from_numpy(data).float()  # transform to float torchTensor
        TraindataL = torch.from_numpy(np.array(flags).reshape((-1,1))).float()
        TraindataM = torch.unsqueeze(TraindataM, 1)
        # print(TraindataM.shape,TraindataL.shape)
        TorchDataset = Data.TensorDataset(TraindataM,TraindataL)

        self.validata_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
        print('run at {},with {} epochs,data size {}'.format(self.model_name,self.epoch,TraindataL.shape[0]))
        print('-----------------------------------------------')

        # 重置dataset
        self.dataset = 'trainAgainLR'
        data = next(iter(self.data_loader ))[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size,class_num=self.class_num)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size,class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # 等间隔调整学习率 StepLR
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.G_optimizer, 30, gamma=0.1, last_epoch=-1)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.D_optimizer, 30, gamma=0.1, last_epoch=-1)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))

        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()
        self.writer = SummaryWriter()#log_dir=log_dir,
        self.X = 0

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['D_lr'] = []
        self.train_hist['G_lr'] = []

        # real: 0, fake:1
        # self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_, self.y_fake_ = torch.zeros(self.batch_size, 1), torch.ones(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        # print('training start!!')
        start_time = time.time()
        print('{} training start!!,data set:{},epoch:{}'.format(self.model_name,self.dataset,self.epoch))

        # stored_url = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-lung/models/attack_free/ACGAN'
        for epoch in range(self.epoch):
            # if epoch==0:
            #     self.G = torch.load(os.path.join(stored_url,'ACGAN_105_G.pkl'))
            #     self.D = torch.load(os.path.join(stored_url,'ACGAN_105_D.pkl'))
            self.G.train()
            self.D.train()

            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.randn((self.batch_size, self.z_dim))
                # y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor), 1)

                #标签散布横着放
                # print('y_vec_.shape:',y_vec_.shape)#torch.Size([64, 2])
                if self.gpu_mode:
                    x_, z_, y_vec_ = x_.cuda(), z_.cuda(), y_vec_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real, C_real = self.D(x_)
                # print(D_real.shape)#torch.Size([64, 1])
                # print(C_real.shape)#torch.Size([64, 2])
                # print(torch.max(y_vec_, 0)[1])#tensor([60, 63])
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])
                # C_real_loss = self.CE_loss(C_real, y_.squeeze(1).type(torch.LongTensor))

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                # C_fake_loss = self.CE_loss(C_fake, y_.squeeze(1).type(torch.LongTensor))
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])
                # C_fake_loss = self.CE_loss(C_fake, y_.squeeze(1).type(torch.LongTensor))

                G_loss += C_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                self.train_hist['D_lr'].append(self.D_optimizer.param_groups[0]['lr'])
                self.train_hist['G_lr'].append(self.G_optimizer.param_groups[0]['lr'])

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 1000) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()),end=',')
                    print('lr_G:%.10f,lr_D:%.10f'%(self.G_optimizer.param_groups[0]['lr'],self.D_optimizer.param_groups[0]['lr']))

                    self.writer.add_scalar('G_loss', G_loss.item(), self.X)
                    # writer.add_scalar('G_loss', -G_loss_D, X)
                    self.writer.add_scalar('D_loss', D_loss.item(), self.X)
                    self.writer.add_scalars('cross loss', {'G_loss': D_loss.item(),
                                                      'D_loss': D_loss.item()}, self.X)

                    self.writer.add_scalars('G_ross with lr_G', {'G_loss': D_loss.item(),
                                                      'lr_G':self.G_optimizer.param_groups[0]['lr']}, self.X)
                    self.writer.add_scalars('D_ross with lr_D', {'D_loss': D_loss.item(),
                                                      'lr_D':self.D_optimizer.param_groups[0]['lr']}, self.X)
                    self.X += 1

            # self.scheduler_D.step()
            # self.scheduler_G.step()

            # 'validate'
            # schedule D lr
            acc_D = validate(self.D,None,self.validata,self.valiflags)#(model,data_loader=None,data=None,label=None) model,dataload,torch.tensor
            # self.D_scheduler.step(acc_D)
            self.scheduler_D.step()
            # self.D.train()

            # schedule G lr
            acc_G = self.validate_G()
            # self.G_scheduler.step(acc_G)
            self.scheduler_G.step()
            print()
            # print('lr_G:%.10f,lr_D:%.10f' % (
            # self.G_optimizer.param_groups[0]['lr'], self.D_optimizer.param_groups[0]['lr']),end='\n\n')

            # self.G.train()

            if epoch % 5 == 0:
                self.load_interval(epoch)
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            # with torch.no_grad():
            #     self.visualize_results((epoch+1))

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

        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def load_interval(self,epoch):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))

    def validate_G(self):
        self.G.eval()
        import math
        acc_G = 0
        sum_all = 0
        for i,(x_,y_) in enumerate(self.validata_loader):
            z_ = torch.randn((self.batch_size, self.z_dim))
            # y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
            # y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor), 1)
            if x_.shape[0]!=64:
                break
            y_vec_ = torch.zeros((x_.shape[0], self.class_num)).scatter_(1, y_.type(torch.LongTensor), 1)
            z_ = torch.rand((self.batch_size, self.z_dim))

            if self.gpu_mode:
                x_, z_, y_vec_ = x_.cuda(), z_.cuda(), y_vec_.cuda()

            G_ = self.G(z_, y_vec_)
            D_fake, C_fake = self.D(G_)
            # if i%1000==0:
            #     print('i:{},D_fake:{},{}'.format(i,D_fake.shape,D_fake.__class__))
            #     i:0,D_fake:torch.Size([64, 1]),<class 'torch.Tensor'>
            try:
                D_fake = D_fake.item()
                y_ = y_.item()
                sum_all += 1
                if D_fake > 0.5:
                    acc_G += 1

            except:
                # print('D_fake:{},{},y_:{},{}'.format(D_fake.__class__,D_fake.shape,y_.__class__,y_.shape))
                # D_fake:<class 'torch.Tensor'>,torch.Size([64, 1]),y_:<class 'torch.Tensor'>,torch.Size([64, 1])
                try:
                    D_fake = np.squeeze(D_fake.data.numpy(), axis=1)
                    # label = np.squeeze(y_.data.numpy(), axis=1)
                    # D_fake = D_fake.tolist()
                    # f1 = lambda l, r: 1 if math.fabs(l - r) < 0.5 else 0
                    # ll = list(map(f1, label.tolist(),D_fake.tolist()))

                    # 生成的数据默认为攻击,判断>0.5 即可
                    f = lambda x: 1 if x > 0.5 else 0
                    ll = list(map(f, D_fake.tolist()))
                    acc_G += ll.count(1)
                    sum_all += len(ll)
                except:
                    print('error!!!!! D_fake:{},{},y_:{},{},error at self.validate_G!!!!'.
                          format(D_fake.__class__, D_fake.shape, y_.__class__, y_.shape))

        print('G,size:%d,errors:%d,corrects:%d' % (sum_all, sum_all - acc_G, acc_G), end=',')
        print('acc:%.6f,judged as 1' % (acc_G / sum_all))
        return acc_G / sum_all

    def validate_D(self):
        import math
        f1 = lambda l, r: 1 if math.fabs(l - r) < 0.5 else 0

        self.D.eval()
        if self.validata.ndim == 4:
            pass
        else:
            TraindataM = torch.from_numpy(self.validata).float()  # transform to float torchTensor
            data = torch.unsqueeze(TraindataM, 1)

        try:
            D_real, _ = self.D(data)
        except:
            D_real = self.D(data)
        if self.valiflags is not None:
            D_real = D_real.data.numpy()
            D_real = np.squeeze(D_real).tolist()#[[],[],[]]

            ll = list(map(f1, self.valiflags, D_real))
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