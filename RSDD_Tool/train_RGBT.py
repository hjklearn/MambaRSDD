import torch
from torch import nn
import sys
sys.path.append('/root/autodl-tmp/MambaRSDD/MambaRSDD/')
from rail_defect import RDDataset
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pytorch_iou
from mamba_rail_net import *
import time
import os
from log  import get_logger



def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params +=p.numel()
    print(name)
    print("The number of parameters:{}M".format(num_params/1000000))


IOU = pytorch_iou.IOU(size_average = True).cuda()

class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

################################################################################################################
batchsize = 8
################################################################################################################
rootpath = '/root/autodl-tmp/RGB_D_rail/RGB_D_rail/'
TrainDatasets = RDDataset(rootpath, 'train')
ValDatasets = RDDataset(rootpath, 'value')

train_dataloader = DataLoader(TrainDatasets, batch_size=batchsize, shuffle=True, num_workers=4)
test_dataloader = DataLoader(ValDatasets,batch_size=batchsize, shuffle=True, num_workers=4)

net = Depth_Mamba()
net = net.cuda()
# net.load_state_dict(torch.load('/media/yuride/date/Mamba_SRDD/Pth/DepthMamba_Fulltiny_CLIP_path2024_12_08_19_27_best.pth'))   ########gaiyixia

################################################################################################################
model = 'MambaRSDD_√L1_√L2_√L3_√L4_√L5_path' + time.strftime("%Y_%m_%d_%H_%M")
print_network(net, model)
################################################################################################################
bestpath = '/root/autodl-tmp/MambaRSDD/MambaRSDD/RSDD_Tool/Pth/' + model + '_best.pth'
lastpath = '/root/autodl-tmp/MambaRSDD/MambaRSDD/RSDD_Tool/Pth/' + model + '_last.pth'
################################################################################################################

criterion1 = BCELOSS().cuda()

criterion_val = BCELOSS().cuda()

lr_rate = 1e-4
# optimizer = optim.Adam(net.parameters(), lr_rate, weight_decay=1e-3)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr_rate, weight_decay=1e-3)


best = [10]

step = 0
mae_sum = 0
best_mae = 1
best_epoch = 0

logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)

logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')

epochs = 150

logger.info(f'Epochs:{epochs}  Batchsize:{batchsize}')
for epoch in range(epochs):
    mae_sum = 0
    trainmae = 0
    if (epoch+1) % 20 == 0 and epoch != 0:
        for group in optimizer.param_groups:
            group['lr'] = 0.5 * group['lr']
            print(group['lr'])
            lr_rate = group['lr']

    # lr_rate setting
    epoch_new = epoch + 1
    train_loss = 0
    net = net.train()
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())

        optimizer.zero_grad()
        sal_finally, rgb_list, dep_list = net(image, depth)
        # sal_finally = net(image, depth)

        out1 = torch.sigmoid(sal_finally)
        loss1 = criterion1(out1, label) + IOU(out1, label)

        loss_cilip = 0
        for r, d in zip(rgb_list, dep_list):
            b5, _, _, _, = r.shape
            r5_sha = r.reshape(b5, -1)
            d5_sha = d.reshape(b5, -1)
            logits_5 = torch.matmul(r5_sha, d5_sha.T).cuda() # (B,B)
            labels_5 = torch.arange(logits_5.shape[0]).cuda()  # (0,1,2,3)
            loss_i5 = torch.nn.CrossEntropyLoss().cuda()(logits_5, labels_5)
            loss_t5 = torch.nn.CrossEntropyLoss().cuda()(logits_5.T, labels_5)
            clip_loss = (loss_i5 + loss_t5) / 2
            loss_cilip = loss_cilip + clip_loss

        loss_total = loss1 + loss_cilip

        time = datetime.now()

        if i % 10 == 0 :
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch_new, epochs, i, len(train_dataloader), loss_total.item(), loss1))
        loss_total.backward()
        optimizer.step()
        train_loss = loss_total.item() + train_loss


    net = net.eval()
    eval_loss = 0
    mae = 0

    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):

            imageVal = Variable(sampleTest['RGB'].cuda())
            depthVal = Variable(sampleTest['depth'].cuda())
            labelVal = Variable(sampleTest['label'].float().cuda())

            s1, rgb_list, dep_list = net(imageVal, depthVal)
            # s1 = net(imageVal, depthVal)
            mask = torch.sigmoid(s1)
            loss = criterion_val(mask, labelVal)
            maeval = torch.sum(torch.abs(labelVal - mask)) / (320.0*320.0)

            print('===============', j, '===============', loss.item())
            eval_loss = loss.item() + eval_loss
            mae = mae + maeval.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = '{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    logger.info(
        f'Epoch:{epoch+1:3d}/{epochs:3d} || trainloss:{train_loss / 1500:.8f} valloss:{eval_loss / 362:.8f} || '
        f'valmae:{mae / 362:.8f} || lr_rate:{lr_rate} || spend_time:{time_str}')

    if (mae / 362) <= min(best):
        best.append(mae / 362)
        nummae = epoch+1
        torch.save(net.state_dict(), bestpath)

    torch.save(net.state_dict(), lastpath)
    print('=======best mae epoch:{},best mae:{}'.format(nummae, min(best)))
    logger.info(f'best mae epoch:{nummae:3d}  || best mae:{min(best)}')














