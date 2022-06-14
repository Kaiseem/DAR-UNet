import volumentations as V
import torch.utils.data as data
import os
import torch
import torch.nn.functional as F

import numpy as np
import random

class SEGDataset(data.Dataset):
    def __init__(self, root,n_class, train=True):
        self.is_train=train
        self.root=root
        self.n_class=n_class
        self.imgs, self.labels = self.load_train_data()
        self.aug=V.Compose([
            V.Rotate((0,0), (0, 0), (-15, 15), interpolation=1,p=0.5),
            V.RandomScale2(scale_limit=[0.9,1.1],interpolation=1, p=0.5),
            V.ElasticTransform((0, 0.25), interpolation=1, p=0.2),
            V.Flip(0, p=0.5),])

    def load_train_data(self):
        imgs = []
        labels = []
        for f in os.listdir(self.root):
            if 'img' in f:
                imgs.append(os.path.join(self.root, f))
                labels.append(os.path.join(self.root, f.replace('img', 'label')))
        print(f'processed data with size {len(imgs)} & {len(labels)}')
        return imgs,labels

    def random_crop(self, img, mask=None, size=[256,256,32]):
        rand_x = random.randint(0, img.shape[0] - size[0])
        rand_y = random.randint(0, img.shape[1] - size[1])
        rand_z = random.randint(0, img.shape[2] - size[2])
        if mask is not None:
            return img[rand_x:rand_x + size[0], rand_y:rand_y + size[1],rand_z:rand_z+size[2]], mask[rand_x:rand_x + size[0], rand_y:rand_y + size[1],rand_z:rand_z+size[2]]
        else:
            return img[rand_x:rand_x + size[0], rand_y:rand_y + size[1],rand_z:rand_z+size[2]]

    def RandomSaturation(self, img, saturation_limit=[0.9,1.1]):
        saturation=random.uniform(saturation_limit[0], saturation_limit[1])
        return np.clip(img*saturation,0,1)

    def RandomBrightness(self, img, intensity_limit=[0, 0.1]):
        brightness=random.uniform(intensity_limit[0], intensity_limit[1])
        return np.clip(img+brightness,0,1)

    def RandomContrast(self,img,contrast_limit=[0.9,1.1]):
        mean=np.mean(img,axis=(0,1,2),keepdims=True)
        contrast = random.uniform(contrast_limit[0], contrast_limit[1])
        return np.clip(img * contrast + mean * (1 - contrast),0,1)

    def ColorJetter(self,img,p=0.5):
        if random.random()>p:
            augs=[self.RandomSaturation,self.RandomBrightness,self.RandomContrast]
            random.shuffle(augs)
            for aug in augs:
                if random.random()>0.5:
                    img=aug(img)
            return img
        else:
            return img

    def __getitem__(self, index):
        if self.is_train:
            A_index = random.randint(0, len(self.imgs) - 1)
        else:
            A_index = index

        A_img = np.load(self.imgs[A_index])

        A_label = np.load(self.labels[A_index])

        if self.is_train:
            (A_img, A_label) = self.random_crop(A_img, A_label)
            A_img = self.ColorJetter(A_img)
            seg_augmented = self.aug(image=A_img, mask=A_label)
            A_img = seg_augmented['image']
            A_label = seg_augmented['mask']
        A_img = A_img * 2 - 1
        A_img = np.expand_dims(A_img.transpose((2, 0, 1)), 0)
        A_img = torch.from_numpy(A_img).float()

        A_label = A_label.transpose((2, 0, 1))
        A_label = torch.from_numpy(A_label).long()
        A_label = F.one_hot(A_label, self.n_class).permute(3, 0, 1, 2).float()
        output = {'img': A_img, 'label': A_label}
        return output

    def __len__(self):
        return len(self.imgs)