

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader_1d
import Model as models
import torch.nn as nn
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings

momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4


def train(model):
    src_iter = iter(src_loader)



    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet1.parameters()},
            {'params': model.sharedNet2.parameters()},

            {'params': model.proj_1.parameters()},
            {'params': model.proj_2.parameters()},


            {'params': model.cls_fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc3.parameters(), 'lr': LEARNING_RATE},


        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

        cls_label=src_label[:,0]

        optimizer.zero_grad()


        vib_data = src_data[:, :4096]
        aud_data = src_data[:, 4096:]


        src_pred1,src_pred2,src_pred3,contra,trans= model(vib_data,aud_data,cls_label)


        cls_loss =  F.nll_loss(F.log_softmax(src_pred1, dim=1), cls_label)\
                   +F.nll_loss(F.log_softmax(src_pred2, dim=1), cls_label)\
                   +F.nll_loss(F.log_softmax(src_pred3, dim=1), cls_label)

        loss = cls_loss+contra*hypercontra

        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tcon_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(),contra.item()))

        if i % (log_interval * 10) == 0:



            train_correct, train_loss = test_source(model, src_loader)
            test_correct1,test_correct2,test_correct3 = test_target(model, tgt_test_loader)





def test_target(model,test_loader):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    m = nn.Softmax(dim=1)

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)


            vib_data = tgt_test_data[:, :4096]
            aud_data = tgt_test_data[:, 4096:]


            tgt_pred1, tgt_pred2, tgt_pred3,_,_= model(vib_data,aud_data,tgt_test_label)


            pred_1 = m(tgt_pred3)
            pred_1 = pred_1.data.max(1)[1]
            correct1 += pred_1.eq(tgt_test_label.data.view_as(pred_1)).cpu().sum()


            pred_2 =  m(tgt_pred1)+m(tgt_pred2)
            pred_2 = pred_2.data.max(1)[1]
            correct2 += pred_2.eq(tgt_test_label.data.view_as(pred_2)).cpu().sum()

            pred_3 = m(tgt_pred1) + m(tgt_pred2)+m(tgt_pred3)
            pred_3 = pred_3.data.max(1)[1]
            correct3 += pred_3.eq(tgt_test_label.data.view_as(pred_3)).cpu().sum()


    print('\nC3Accuracy: {}/{} ({:.2f}%)\nC1C2Accuracy: {}/{} ({:.2f}%)\nC1C2C3Accuracy: {}/{} ({:.2f}%)\n'.format(correct1, len(test_loader.dataset),10000. * correct1 / len(test_loader.dataset),
                                                                                                                   correct2, len(test_loader.dataset),10000. * correct2 / len(test_loader.dataset),
                                                                                                                   correct3, len(test_loader.dataset),10000. * correct3 / len(test_loader.dataset)))
    return correct1, correct2, correct3


def test_source(model,test_loader):
    model.eval()

    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    m = nn.Softmax(dim=1)



    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_test_label = tgt_test_label[:, 0]
            # print(tgt_test_data)

            vib_data = tgt_test_data[:, :4096]
            aud_data = tgt_test_data[:, 4096:]

            tgt_pred1, tgt_pred2, tgt_pred3,_ ,_= model(vib_data, aud_data,tgt_test_label)

            pred_1 = m(tgt_pred1)
            pred_1 = pred_1.data.max(1)[1]
            correct1 += pred_1.eq(tgt_test_label.data.view_as(pred_1)).cpu().sum()

            pred_2 = m(tgt_pred2)
            pred_2 = pred_2.data.max(1)[1]
            correct2 += pred_2.eq(tgt_test_label.data.view_as(pred_2)).cpu().sum()

            pred_3 = m(tgt_pred3)
            pred_3 = pred_3.data.max(1)[1]
            correct3 += pred_3.eq(tgt_test_label.data.view_as(pred_3)).cpu().sum()

        print('\nC1Accuracy: {}/{} ({:.2f}%)\nC2Accuracy: {}/{} ({:.2f}%)\nC3Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct1, len(test_loader.dataset), 10000. * correct1 / len(test_loader.dataset),
            correct2, len(test_loader.dataset), 10000. * correct2 / len(test_loader.dataset),
            correct3, len(test_loader.dataset), 10000. * correct3 / len(test_loader.dataset)))


        return correct3,test_loss



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':

    # setup_seed(seed)
    iteration = 8000
    batch_size = 128
    lr = 0.001
    FFT = False

    class_num = 5

    hypercontra=1


    Source = np.array(
        [
            [1, 3, 5, 7, 10, 12, 14, 16],
            [2, 4, 6, 8, 9, 11, 13, 15],
            [1, 3, 5, 7, 9, 11, 13, 15],
            [2, 4, 6, 8, 10, 12, 14, 16],
            [1, 2, 3, 4, 9, 10, 11, 12],
            [5, 6, 7, 8, 13, 14, 15, 16],
            [1, 2, 3, 4, 13, 14, 15, 16],
            [5, 6, 7, 8, 9, 10, 11, 12]
        ])

    Target = np.array(
        [
            [2, 4, 6, 8, 9, 11, 13, 15],
            [1, 3, 5, 7, 10, 12, 14, 16],
            [2, 4, 6, 8, 10, 12, 14, 16],
            [1, 3, 5, 7, 9, 11, 13, 15],
            [5, 6, 7, 8, 13, 14, 15, 16],
            [1, 2, 3, 4, 9, 10, 11, 12],
            [5, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 13, 14, 15, 16]
        ]
    )

    for taskindex in range(8):

        sourcelist = Source[taskindex]
        targetlist = Target[taskindex]

        for repeat in range(10):

            cuda = not no_cuda and torch.cuda.is_available()
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training(sourcelist, FFT, batch_size, kwargs)

            tgt_test_loader = data_loader_1d.load_testing(targetlist, FFT, batch_size, kwargs)

            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)

            model = models.MUGTN(num_classes=class_num)
            # get_parameter_number(model) 计算模型训练参数个数
            print(model)

            if cuda:
                model.cuda()

            train(model)




















