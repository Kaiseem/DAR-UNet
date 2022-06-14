import os
import torch
from sklearn.cluster import KMeans
import numpy as np
from i2i_solver import i2iSolver
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='test/i2i_checkpoints/enc_0040.pt')
parser.add_argument('--source_npy_dirpath', type=str, default='datasets/source_training_npy')
parser.add_argument('--target_npy_dirpath', type=str, default='datasets/target_training_npy')
parser.add_argument('--save_npy_dirpath', type=str, default='datasets/source2target_training_npy')
parser.add_argument('--k_means_clusters', type=int, default=6)

opts = parser.parse_args()
trainer=i2iSolver(None)
state_dict = torch.load(opts.ckpt_path)
trainer.enc_c.load_state_dict(state_dict['enc_c'])
trainer.enc_s_a.load_state_dict(state_dict['enc_s_a'])
trainer.enc_s_b.load_state_dict(state_dict['enc_s_b'])
trainer.dec.load_state_dict(state_dict['dec'])
trainer.cuda()

styles=[]
for f2 in os.listdir(opts.target_npy_dirpath):
    if 'label' not in f2:
        imgs = np.load(os.path.join(opts.target_npy_dirpath, f2))
        for i in range(int(imgs.shape[-1]/6),int(imgs.shape[-1]/6*5)):
            img = imgs[:, :, i]
            with torch.no_grad():
                single_img = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
                s=trainer.enc_s_b(single_img).cpu().numpy()[0]
                styles.append(s)
n_clusters=opts.k_means_clusters
k_mean_results = KMeans(n_clusters=opts.k_means_clusters, random_state=9).fit_predict(styles)

for f in os.listdir(opts.source_npy_dirpath):
    imgs = np.load(os.path.join(opts.target_npy_dirpath, f))
    for k in range(n_clusters):
        nimgs = np.zeros_like(imgs, dtype=np.float32)
        idx=random.choice(np.argwhere(k_mean_results==k).flatten().tolist())
        s = torch.from_numpy(styles[idx]).unsqueeze(0).cuda().float()
        for i in range(imgs.shape[-1]):
            img = imgs[:, :, i]
            single_img = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
            transfered_img = trainer.inference(single_img, s)
            transfered_img = (((transfered_img + 1) / 2).cpu().numpy()).astype(np.float32)[0, 0]
            nimgs[:, :, i] = transfered_img
        nlabels = np.load(os.path.join('datasets\source_training_npy', f.replace('img', 'label')))
        np.save(os.path.join('datasets\source2target_training_npy', f.replace('img', f'{k}_img')), nimgs)
        np.save(os.path.join('datasets\source2target_training_npy', f.replace('img', f'{k}_label')), nlabels)





