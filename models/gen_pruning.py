# -*- coding: UTF-8 -*-
from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import torchvision.models as models
import models
# from models import vgg2
# from models import prune_resnet
from models import data_free_resnet
from models import googlenet
from pretrained.CIFAR_ZOO_master.models import preresnet
from models import generator
from dataset import cifar10

from torchstat import stat

import numpy as np
import pickle
from scipy.spatial import distance

from torchstat import stat
# from thop import profile
# import torchvision.models as models


import pdb



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str,default='./data/cifar.python/cifar10', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet110', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train.')
parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--eval_batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[40,70,90,110],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.5, 0.3, 0.3,0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./checkpoints/googlenet', help='Folder to save checkpoints and log.')

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='', type=str, metakmvar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--conv_num_per_layer',type=int,default=9)
# Acceleration
parser.add_argument('--ngpu', type=int, default=4, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate_norm', type=float, default=0.8, help='the remaining ratio of pruning based on Norm')
parser.add_argument('--rate_dist', type=float, default=0.45, help='the reducing ratio of pruning based on Distance')

parser.add_argument('--layer_begin', type=int, default=0, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=324
                    , help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=3, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict',default='False', help='use state dcit or not')
# parser.add_argument('--use_state_dict', dest='use_state_dict',default='True',action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', default='True', help='use pre-trained model or not')
parser.add_argument('--use_cuda', dest='use_cuda', default='True', help='use cuda or not')
# parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
# parser.add_argument('--pretrain_path', default='./pretrained/googlenet.pt', type=str, help='..path of pre-trained model')
# parser.add_argument('--pretrain_path', default='./pretrained/resnet_56.pt', type=str, help='..path of pre-trained model')
# parser.add_argument('--pretrain_path', default='./pretrained/vgg_16_bn.pt', type=str, help='..path of pre-trained model')
parser.add_argument('--pretrain_path', default='./pretrained/resnet34.pth.tar', type=str, help='..path of pre-trained model')
# parser.add_argument('--pretrain_path', default='./Baseline/CIFAR_ZOO_master/experiments/cifar10/preresnet110/preresnet110.pth.tar', type=str, help='..path of pre-trained model')
parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')
parser.add_argument('--lr_decay_step', default='30', type=int)
parser.add_argument( '--miu', type=float,default=0.8,help='The miu of data loss.')

parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=10, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

loader = cifar10.Data(args)
train_loader = loader.loader_train
test_loader = loader.loader_test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        #os.mkdir(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.log'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Norm Pruning Rate: {}".format(args.rate_norm), log)
    print_log("Distance Pruning Rate: {}".format(args.rate_dist), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("use cuda: {}".format(args.use_cuda), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Dist type: {}".format(args.dist_type), log)
    print_log("epochs: {}".format(args.epochs), log)
    print_log("decay: {}".format(args.decay), log)
    print_log("learning_rate: {}".format(args.learning_rate), log)
    print_log("miu: {}".format(args.miu), log)
    print_log("gammas: {}".format(args.gammas), log)
    print_log("schedule: {}".format(args.schedule), log)
    num = torch.cuda.current_device()

    print_log(num, log)



    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer

    # net = models.resnet50()
    # net = models.resnet50(pretrained=True)
    # net = googlenet.googlenet()
    # net = prune_resnet.resnet_56()
    # net = vgg2.vgg_16_bn()
    # net = preresnet.preresnet110(10)
    net = data_free_resnet.ResNet34()
    # stat(net, (3, 32, 32))
    # model_t = prune_resnet.resnet_56()
    # model_t = vgg2.vgg_16_bn()
    # model_t = googlenet.googlenet()
    # model_t = preresnet.preresnet110(10)
    model_t = data_free_resnet.ResNet34()
    model_g = generator.Generator()
    print_log("=> network :\n {}".format(net), log)

    # stat(net, (3, 32, 32))
    # print_log('Flops:  '.format(flops))
    # print_log('Params: '.format(params))



   

    optimizer_S = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    optimizer_G = torch.optim.SGD(model_g.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    net = torch.nn.DataParallel(net.cuda(), device_ids=[0,1,2,3])
    model_g = torch.nn.DataParallel(model_g.cuda(), device_ids=[0,1,2,3])
    model_t = torch.nn.DataParallel(model_t.cuda(), device_ids=[0,1,2,3])
    pretrain = torch.load(args.pretrain_path)
    model_t.load_state_dict(pretrain['state_dict'])


    m = Mask(net,log)
    m.init_length(log)
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("remaining ratio of pruning : Norm is %f" % args.rate_norm)
    print_log("=> remaining ratio of pruning : Norm is  '{}'".format(args.rate_norm), log)
    print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
    print_log("=> reducing ratio of pruning : Distance is  '{}'".format(args.rate_dist), log)
    
    # Main loop


    for epoch in range(opt.n_epochs):

        total_correct = 0
        avg_loss = 0.0
        if opt.dataset != 'MNIST':
            adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

        for i in range(120):
            net.train()
            z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
            optimizer_G.zero_grad()
            optimizer_S.zero_grad()
            gen_imgs = generator(z)
            outputs_T, features_T = teacher(gen_imgs, out_feature=True)
            pred = outputs_T.data.max(1)[1]  # A.max(1)：返回A每一行最大值和下标组成的二维数组
            # print('outputshape')
            # print(outputs_T.size())【512，10】
            # # print('outputs_T.data')
            # # print(outputs_T.data)
            # print('outputs_T.data.max(1)')
            # print(outputs_T.data.max(1))
            # print('outputs_T.data.max(1)[1]')
            # print(outputs_T.data.max(1)[1])
            loss_activation = -features_T.abs().mean()
            loss_one_hot = criterion(outputs_T, pred)
            softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
            loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()
            loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
            loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach())
            loss += loss_kd
            loss.backward()
            optimizer_G.step()
            optimizer_S.step()
            if i == 1:
                print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (
                epoch, opt.n_epochs, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(),
                loss_kd.item()))

        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            m.model = net
            # m.if_zero()
            m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
            # m.do_mask()
            m.do_similar_mask()
            m.if_zero()
            net = m.model
            net = net.cuda()

        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.cuda()
                labels = labels.cuda()
                net.eval()
                output = net(images)
                avg_loss += criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
        accr = round(float(total_correct) / len(data_test), 4)
        if accr > accr_best:
            torch.save(net, opt.output_dir + 'student')
            accr_best = accr


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


# def adjust_learning_rate(optimizer, epoch, gammas, schedule):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.learning_rate
#     assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
#     for (gamma, step) in zip(gammas, schedule):
#         if (epoch >= step):
#             lr = lr * gamma
#         else:
#             break
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Mask:
    def __init__(self, model,log):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.log = log
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            print("filter codebook done")
        else:
            pass
        # print_log('mask{}'.format(codebook),self.log)
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation



    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1-compress_rate))
            # filter_pruned_num = 4
            #print(filter_pruned_num)
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)  # [16,3,3,3]->[16, 27]
            # print_log('weight_vec{}'.format(weight_torch.size()),self.log)
            # print_log('weight_vec.size{}'.format(weight_vec.size()),self.log)
            weight_vec = weight_vec.cpu().numpy()
            kmeans = KMeans(n_clusters=filter_pruned_num, random_state=0)

            clusters = kmeans.fit_predict(weight_vec)
            # print(clusters.shape)
            # 可视化10类中的中心点——最具有代表性的10个数字
            # fig, ax = plt.subplots(2, 5, figsize=(8, 3))
            #centers = kmeans.cluster_centers_.reshape(filter_pruned_num, weight_torch.size()[1], 3, 3)
            centers = kmeans.cluster_centers_
            # for axi, center in zip(ax.flat, centers):
            # axi.set(xticks=[], yticks=[])
            # axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
            weight_vec = torch.from_numpy(weight_vec)
            # if dist_type == "l2" or "cos":
            # norm = torch.norm(weight_vec, 2, 1)#按输入通道来求2范数？输出？
            # print_log('norm{}'.format(norm), self.log)                                                                                                                                                       norm_np = norm.cpu().numpy()
            # elif dist_type == "l1":
            # norm = torch.norm(weight_vec, 1, 1)
            # norm_np = norm.cpu().numpy()
            # filter_small_index = []
            # filter_large_index = []
            # filter_large_index = norm_np.argsort()[filter_pruned_num:]#argsort函数返回的是数组值从小到大的索引值
            # filter_small_index = norm_np.argsort()[:filter_pruned_num]#截取列表

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            # indices = torch.LongTensor(filter_large_index).cuda()
            # weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()#选出2范数大的filter
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec, centers, 'cityblock')
                # similar_matrix = kmeans.fit_transform(weight_vec)

                # print(similar_matrix.shape)
            # elif dist_type == "cos":  # for cos similarity
            # similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=1)  # 每个filter到其他filter的距离之和
            similar_sum = torch.from_numpy(similar_sum)
            # print(similar_sum.size())
            values, indices = torch.kthvalue(similar_sum, similar_pruned_num, dim=0,
                                             out=None)  # 其中indices是原始输入张量input中沿dim维的第 k 个最小值下标
            # print(values)
            # for distance similar: get the filter index with largest similarity == small distance
            # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            # similar_small_index = similar_sum.argsort()[:similar_pruned_num]
            # similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]#权重大的距离其他filter近的filter

            # print('filter_large_index', filter_large_index)
            # print('filter_small_index', filter_small_index)
            # print('similar_sum', similar_sum)
            # print('similar_large_index', similar_large_index)
            # print('similar_small_index', similar_small_index)
            # print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for i in range(0, similar_sum.size()[0]):

                if similar_sum[i] <= values:
                    codebook[
                    i * kernel_length:(i + 1) * kernel_length] = 0
                pass
        return codebook



    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self,log):
        for index, item in enumerate(self.model.parameters()):

            # print_log('item.size{}'.format(item.size()),self.log)
            self.model_size[index] = item.size()

        for index1 in self.model_size: #计算每层参数数量？
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
            # print_log('model_length{}'.format(self.model_length[index1]),log)

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        for index, item in enumerate(self.model.named_parameters()):
            print ('itemname-{}'.format(item[0]))
            self.compress_rate[index] = 1
            self.distance_rate[index] = 1
            # if len(item[1].size()) == 4 :
            if len(item[1].size()) == 4 and 'weight' in item[0]:
                # if not previous_cfg:
                self.compress_rate[index] = rate_norm_per_layer
                self.distance_rate[index] = rate_dist_per_layer
                self.mask_index.append(index)
    #     for index, item in enumerate(self.model.parameters()):
    #         self.compress_rate[index] = 1
    #         self.distance_rate[index] = 1
    #     for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
    #         self.compress_rate[key] = rate_norm_per_layer
    #         self.distance_rate[key] = rate_dist_per_layer
    #     # different setting for  different architecture
    #     last_index = 0
    #     if args.arch == 'resnet18':
    #     # if args.arch == 'resnet20':
    #         last_index = 57
    #     elif args.arch == 'resnet32':
    #         last_index = 93
    #     elif args.arch == 'resnet50':
    #         last_index = 165
    #     elif args.arch == 'resnet110':
    #         last_index = 327
    #     # to jump the last fc layer
    #     # print_log('last_index{}'.format(last_index),self.log)last_index165
    #     self.mask_index = [x for x in range(0, last_index, 3)]
    #
    # #        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:


                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    # def do_mask(self):
    #     for index, item in enumerate(self.model.parameters()):
    #         if index in self.mask_index:
    #             a = item.data.view(self.model_length[index])
    #             b = a * self.mat[index]
    #             item.data = b.view(self.model_size[index])
    #     print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def do_grad_mask(self,log):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # print_log('a{}'.format(a),log)
                # print_log('a.size(){}'.format(a.size()),log)
                # atensor([-0.0054, -0.0072, -0.0087, ..., 0.0036, 0.0045, 0.0029],
                #         device='cuda:0')
                # a.size()
                # torch.Size([2304])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                # b = a * self.mat[index]
                b = a * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        # count = 0

        for index, item in enumerate(self.model.parameters()):

            if index in self.mask_index:
                # print ('_________________________________')
                # print(index)
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                # c = len(b) - np.count_nonzero(b)
                # count += c


                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
        # print ('count')
        # print (count)


if __name__ == '__main__':
    main()
