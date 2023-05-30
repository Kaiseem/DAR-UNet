import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import DataLoader
from adabelief_pytorch import AdaBelief
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from utils import SEGDataset
from models import DARUnet
import time
from utils.tta import PatchInferencer

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None,weight=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.weight=weight

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        i = input
        t = target

        # Change the shape of input and target to B x N x num_voxels.
        i = i.view(i.size(0), i.size(1), -1)
        t = t.view(t.size(0), t.size(1), -1)

        # Compute the log proba.
        logpt = F.log_softmax(i, dim=1)
        # Get the proba
        pt = torch.exp(logpt)  # B,H*W or B,N,H*W

        if self.weight is not None:
            class_weight = torch.as_tensor(self.weight)
            class_weight = class_weight.to(i)

            at = class_weight[None, :, None]
            at = at.expand((t.size(0), -1, t.size(2)))
            logpt = logpt * at

        # Compute the loss mini-batch.
        weight = torch.pow(-pt + 1.0, self.gamma)
        loss = torch.mean(-weight * t * logpt, dim=-1)
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def dice_loss_chill(output, gt):
    num = (output*gt).sum(dim=[2, 3, 4])
    denom = output.sum(dim=[2, 3, 4]) + gt.sum(dim=[2, 3, 4]) + 0.001
    return num, denom

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--train_dataroot', type=str, default='')
parser.add_argument('--val_dataroot', type=str, default='')
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--epoch_max', type=int, default=100)



opts = parser.parse_args()


if __name__ == '__main__':
    epoch_max=opts.epoch_max

    train_A_loader = DataLoader(dataset=SEGDataset(opts.train_dataroot,opts.num_classes), batch_size=2, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_A_loader = DataLoader(dataset=SEGDataset(opts.val_dataroot,opts.num_classes), batch_size=1, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

    netS=DARUnet()
    netS.cuda()

    ipp = PatchInferencer(n_class=opts.num_classes,TTA=False)

    optS=AdaBelief(netS.parameters(), lr=5e-4, weight_decay=1e-4, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=True, print_change_log=False)

    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optS, epoch_max, eta_min=5e-7, last_epoch=-1)

    DC=DiceLoss(opts.num_classes)
    FC=FocalLoss()

    iter_num=len(train_A_loader)
    st=time.time()

    print(f'start at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st))}')
    iteration=0
    best_dice=0
    for epoch in range(epoch_max):
        netS.train()
        et=time.time()
        for iteration, train_A_data in enumerate(train_A_loader):
            A_img=train_A_data['img'].cuda()
            A_label=train_A_data['label'].cuda()

            optS.zero_grad()
            A_preds = netS(A_img)
            ce_loss=0
            dc_loss=0
            for pred in A_preds:
                ce_loss+=FC(pred, A_label)*10
                dc_loss+=DC(pred, A_label, softmax=True)
            loss_sup=ce_loss+dc_loss
            loss_sup.backward()
            optS.step()

            sys.stdout.write('\r[{}-{}/{}] Loss DC {:.5f} CE {:.5f} G {:.5f} D {:.5f}'.format(epoch + 1, iteration + 1, iter_num, dc_loss.item(), ce_loss.item(),  0,0))
        scheduler.step()
        if (epoch+1)%10==0:
            torch.save({'seg': netS.state_dict()}, f'{epoch+1}.pt')
        if epoch>-1:
            netS.eval()
            dices=[]
            for val_data in val_A_loader:
                for k in val_data.keys():
                    val_data[k] = val_data[k].cuda().detach()
                with torch.no_grad():
                    output = ipp(netS, val_data['A_img'])
                    pred=F.one_hot(torch.argmax(output,1), 3).permute(0,4,1,2,3)
                    gt=val_data['A_label']
                    num, denom = dice_loss_chill(pred,gt)
                    d = (2 * num / denom)[:, 1:].mean().cpu().numpy()
                    dices.append(d)
            torch.cuda.empty_cache()
            dices=np.mean(dices)
            if best_dice<dices:
                best_dice=dices
                torch.save({'seg': netS.state_dict()}, 'best.pt')
            time.sleep(5)
            print(f'\n val dice{dices} best dice {best_dice} epoch time {time.strftime("%M:%S", time.localtime(time.time()-et))}\n')

