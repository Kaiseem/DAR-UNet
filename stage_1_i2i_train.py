from utils import I2IDataset,create_dirs
from i2i_solver import i2iSolver
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

import random
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='version11')
parser.add_argument('--seed', type=int, default=10)
opts = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.
    Args:
        seed: seed to check
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ia.random.seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))
    return

def dice_loss_chill(output, gt):
    num = (output*gt).sum(dim=[2, 3])
    denom = output.sum(dim=[2, 3]) + gt.sum(dim=[2, 3]) + 0.001
    return num, denom

if __name__ == '__main__':
    check_manual_seed(opts.seed)
    create_dirs(opts.name)
    train_loader = DataLoader(dataset=I2IDataset(train=True), batch_size=4, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(dataset=I2IDataset(train=False), batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    trainer = i2iSolver(opts)
    trainer.cuda()
    iteration=0

    for epoch in range(200):
        for train_data in train_loader:
            for k in train_data.keys():
                train_data[k] = train_data[k].cuda().detach()
            trainer.gan_forward(train_data['A_img'], train_data['B_img'])
            trainer.dis_update()
            trainer.gen_update()
            text=trainer.verbose()
            if iteration%100==0:
                trainer.gan_visual(epoch)
            sys.stdout.write(f'\r Epoch {epoch}, Iter {iteration}, {text}')
            iteration+=1
        trainer.save(epoch)

