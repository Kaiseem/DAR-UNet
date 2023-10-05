import os
import cv2
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

def interpolate_4D(img,size):
    assert len(img.shape)==4
    assert len(size)==3
    img=np.expand_dims(img,axis=0)
    im=torch.from_numpy(img).float()
    im= F.interpolate(im, size = size, mode = "trilinear").clamp(min=0,max=255)
    im=im.numpy()
    im=im[0].astype(np.uint8)
    return im

def adaptive_normalize(arr):
    max_p=1-0.0001*arr.shape[-1]
    arr=arr.astype(np.float32)
    PixelArr = arr[arr > 0]
    PixelArr.sort()
    max_v = PixelArr[int((len(PixelArr) - 1) * max_p + 0.5)]
    arr=np.clip(arr,0,max_v)/max_v
    return arr

if __name__ == '__main__':
    if 1:
        roots=['datasets/source_training','datasets/target_training','datasets/target_validation']
        for root in roots:
            #os.makedirs(root+'_npy')
            for f in os.listdir(root):
                print(root,f)
                if 'Label' not in f:
                    print(nib.load(os.path.join(root, f)).header)
                    data = nib.load(os.path.join(root, f)).get_fdata().astype(np.uint16)
                    print(data.min(),data.max())
                    nim = np.zeros((512, 512, data.shape[-1]))
                    for i in range(data.shape[-1]):
                        nim[:, :, i] = cv2.resize(data[:, :, i], (512, 512), interpolation=cv2.INTER_CUBIC)
                    nim=adaptive_normalize(nim)
                    np.save(root+'_npy/{}'.format(f.replace('.nii.gz', '.npy')),nim)
                else:
                    data = nib.load(os.path.join(root, f)).get_fdata().astype(np.uint16)
                    nim = np.zeros((512, 512, data.shape[-1]))
                    for i in range(data.shape[-1]):
                        nim[:, :, i] = cv2.resize(data[:, :, i], (512, 512), interpolation=cv2.INTER_NEAREST)
                    nim=nim.astype(np.int32)
                    np.save(root+'_npy/{}'.format(f.replace('.nii.gz', '.npy')),nim)

    A_imgs = []
    A_labels = []
    B_imgs = []
    A_val_imgs=[]
    A_val_labels=[]
    for f in os.listdir('./datasets/source_training_npy'):
        if 'Label' not in f:
            if f not in ['crossmoda_96_ceT1.npy','crossmoda_97_ceT1.npy','crossmoda_98_ceT1.npy','crossmoda_99_ceT1.npy','crossmoda_9_ceT1.npy']:
                arr = np.load(os.path.join('datasets/source_training_npy', f))
                lbl=np.load(os.path.join('datasets/source_training_npy', f.replace('ceT1','Label')))
                if arr.shape[-1]==160:
                    A_imgs.append(arr.transpose(2, 0, 1)[:-60])
                    A_labels.append(lbl.transpose(2, 0, 1)[:-60])
                else:
                    A_imgs.append(arr.transpose(2, 0, 1)[:-40])
                    A_labels.append(lbl.transpose(2, 0, 1)[:-40])
            else:
                arr = np.load(os.path.join('datasets/source_training_npy', f))
                lbl=np.load(os.path.join('datasets/source_training_npy', f.replace('ceT1','Label')))
                if arr.shape[-1]==160:
                    A_val_imgs.append(arr.transpose(2, 0, 1)[:-60])
                    A_val_labels.append(lbl.transpose(2, 0, 1)[:-60])
                else:
                    A_val_imgs.append(arr.transpose(2, 0, 1)[:-40])
                    A_val_labels.append(lbl.transpose(2, 0, 1)[:-40])

    for f in os.listdir('./datasets/target_training_npy'):
        arr = np.load(os.path.join('datasets/target_training_npy', f))
        B_imgs.append(arr.transpose(2, 0, 1))

    np.save('./A_imgs.npy', np.concatenate(A_imgs, 0).astype(np.float32))
    np.save('./A_labels.npy', np.concatenate(A_labels, 0).astype(np.uint8))
    np.save('./A_val_imgs.npy', np.concatenate(A_val_imgs, 0).astype(np.float32))
    np.save('./A_val_labels.npy', np.concatenate(A_val_labels, 0).astype(np.uint8))
    np.save('./B_imgs.npy', np.concatenate(B_imgs, 0).astype(np.float32))
