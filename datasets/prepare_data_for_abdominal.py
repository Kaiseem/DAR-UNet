
import nibabel as nib

import matplotlib.pyplot as plt
import pydicom
import os

import numpy as np

import SimpleITK as sitk
import torch
import torch.nn.functional as F
def pad(im):
    shape=im.shape
    #cx,cy,cz=int((bbox[0]+bbox[3])/2),int((bbox[1]+bbox[4])/2),int((bbox[2]+bbox[5])/2)
    cx,cy= int((shape[0])/2),int(shape[1]/2)
    shape=im.shape
    ss=[[256-cx,256+shape[0]-cx],[256-cy,256+shape[1]-cy]]
    empty1=np.zeros([512,512,im.shape[-1]])
    empty1[ss[0][0]:ss[0][1],ss[1][0]:ss[1][1]]=im
    return empty1

for dir in ['source_training_npy','target_training_npy','source_test_npy','target_test_npy']:
    if not os.path.isdir(dir):
        os.makedirs(dir)

root=r'D:\PycharmProjects\visualization\heartseg\abdomen_dataset_nii\abdomen_CT_nii\img'
A_imgs=[]
for f in os.listdir(root):
    img = nib.load(os.path.join(root,f))
    image_data=img.get_fdata().transpose((1,0,2))[::-1]
    UID=int(f[3:7])
    #print(UID)
    mask_data=nib.load(os.path.join(root.replace('img','label'),'label'+f[3:])).get_fdata().transpose((1,0,2))[::-1]
    seg = sitk.ReadImage(os.path.join(root.replace('img','label'),'label'+f[3:]))
    seg_array = sitk.GetArrayFromImage(seg)
    for i in range(14):
        if i in [0,1,2,3,6]:
            pass
        else:
            seg_array[seg_array==i]=0
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    start_slice=max(start_slice-2,0)
    end_slice=min(end_slice+2,image_data.shape[-1])
    image_data=image_data[:,:,start_slice:end_slice+1]
    mask_data=mask_data[:,:,start_slice:end_slice+1]

    image_data[image_data>350]=350
    image_data[image_data<-350]=-350
    image_data= (np.clip(image_data,-350,350)+350)/700
    origin_shape=image_data.shape
    target_shape= [int(origin_shape[0]*img.header['pixdim'][1]),int(origin_shape[1]*img.header['pixdim'][2]), int(origin_shape[2]*img.header['pixdim'][3]/4)]

    image_data=torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
    image_data=F.interpolate(image_data,target_shape, mode = "trilinear").numpy()[0,0]
    image_data=pad(image_data)
    print(target_shape)
    #print(image_data.shape,mask_data.shape,mask_data.dtype,np.unique(mask_data))

    mask_data=torch.from_numpy(mask_data.copy()).unsqueeze(0).unsqueeze(0)
    mask_data=F.interpolate(mask_data,target_shape,mode='nearest').numpy()[0,0]
    mask_data=pad(mask_data)
    old_mask_data=mask_data
    mask_data=np.zeros_like(mask_data)
    mask_data[old_mask_data==6]=1
    mask_data[old_mask_data==2]=2
    mask_data[old_mask_data == 3] = 3
    mask_data[old_mask_data == 1] = 4

    print(image_data.shape,image_data.dtype,image_data.max(),image_data.min(),img.header['pixdim'])
    print(np.unique(mask_data))
    # plt.imshow(image_data[:,:,24],'gray',vmin=0,vmax=1)
    # plt.show()
    if UID not in [1,9,30,32,33,39]:
        A_imgs.append(image_data)
        np.save('source_training_npy/{}'.format(f.replace('.nii.gz','.npy')),image_data)
        np.save('source_training_npy/{}'.format(f.replace('img','label').replace('.nii.gz','.npy')),mask_data)
    else:
        np.save('source_test_npy/{}'.format(f.replace('.nii.gz','.npy')),image_data)
        np.save('source_test_npy/{}'.format(f.replace('img','label').replace('.nii.gz','.npy')),mask_data)
from PIL import Image


'''
The data sets are acquired by a 1.5T Philips MRI, 
which produces 12 bit DICOM images having a resolution of 256 x 256. 
The ISDs vary between 5.5-9 mm (average 7.84 mm), 
x-y spacing is between 1.36 - 1.89 mm (average 1.61 mm) 
and the number of slices is between 26 and 50 (average 36). 
In total, 1594 slices (532 slice per sequence) will be provided for training and 1537 slices will be used for the tests.
'''

B_imgs=[]
root=r'D:\PycharmProjects\visualization\heartseg\abdomen_dataset_nii\abdomen_MR_nii'
for d in os.listdir(root):#
    im_list=[]
    msk_list=[]
    for f in os.listdir(os.path.join(root,d,'T2SPIR/DICOM_anon')):
        dcm = pydicom.read_file(os.path.join(root,d,'T2SPIR/DICOM_anon',f))
        #print(dcm,dcm.PixelSpacing)
        im_list.append(dcm.pixel_array)
    for f in os.listdir(os.path.join(root,d,'T2SPIR/Ground')):
        msk_list.append(np.array(Image.open(os.path.join(root,d,'T2SPIR/Ground',f))))
    pixdim=dcm.PixelSpacing
    assert pixdim[0]==pixdim[1]
    image_data=np.stack(im_list,-1).astype(np.float32)
    image_data/=image_data.max()
    origin_shape=image_data.shape
    mask_data=np.stack(msk_list,-1)

    target_shape= [int(origin_shape[0]*pixdim[0]),int(origin_shape[1]*pixdim[1]),origin_shape[2]*2]
    image_data=torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
    image_data=F.interpolate(image_data,target_shape,mode='trilinear').numpy()[0,0]
    image_data=pad(image_data)

    mask_data = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0)
    mask_data = F.interpolate(mask_data, target_shape, mode='nearest').numpy()[0, 0]
    mask_data = pad(mask_data)

    print(target_shape)
    # print(image_data.shape,image_data.dtype,image_data.max(),image_data.min(),dcm.PixelSpacing)
    # plt.imshow(image_data[:, :, 12], 'gray')
    # plt.show()
    mask_data[mask_data==63]=1
    mask_data[mask_data==126]=2
    mask_data[mask_data==189]=3
    mask_data[mask_data==252]=4
    print(np.unique(mask_data))
    if int(d) not in [1,13,32,38]:
        np.save(f'target_training_npy/{d}_img.npy',image_data)
        np.save(f'target_training_npy/{d}_label.npy',mask_data)
        B_imgs.append(image_data)
    else:
        print(d)
        np.save(f'target_test_npy/{d}_img.npy',image_data)
        np.save(f'target_test_npy/{d}_label.npy',mask_data)

# A_imgs=np.concatenate(A_imgs,-1).transpose((2,0,1))
# np.save('A_imgs.npy',A_imgs)
# B_imgs=np.concatenate(B_imgs,-1).transpose((2,0,1))
# np.save('B_imgs.npy',B_imgs)


