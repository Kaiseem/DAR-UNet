import torch
import torch.nn.functional as F

def rot90(tensor, k, reverse=False):  # B,C,D,W,H
    k = k if not reverse else -k
    return torch.rot90(tensor.detach(), k=k, dims=(3, 4))

def trans1(tensor,intensity):
    tensor=tensor/2+0.5
    tensor=torch.clip(tensor*intensity,0,1)
    return tensor*2-1

def trans2(tensor,intensity):
    tensor=tensor/2+0.5
    tensor=torch.clip(tensor+intensity,0,1)
    return tensor*2-1

class Merger:
    def __init__(self,type: str = 'mean',n: int = 1,):
        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))
        self.output = None
        self.type = type
        self.n = n

    def append(self, x):

        if self.type == 'tsharpen':
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = F.max(self.output, x)
        elif self.type == 'min':
            self.output = F.min(self.output, x)

    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result

class TestTimeAug(object):
    def __init__(self):
        pass

    def __call__(self, model, input):
        merger = Merger(type='mean', n=5)
        with torch.no_grad():
            merger.append(model(input)[0])
            merger.append(model(input.detach().flip(3))[0].flip(3))
            merger.append(model(trans1(input.detach(),1.1))[0])
            merger.append(model(trans1(input.detach(),0.9))[0])
            merger.append(model(trans2(input.detach(),0.1))[0])
        return merger.result

import random

class PatchInferencer(object):
    def __init__(self, n_class=5, batch_size=2, size=[32, 256, 256], stride=[16, 128, 128], TTA=False):
        self.n_class = n_class
        self.size = size
        self.stride = stride
        self.batch_size=batch_size
        assert len(self.size) == len(self.stride)
        if TTA:
            self.tta = TestTimeAug()

    def __call__(self, model,input, softmax=True):
        model.eval()
        ned_pad=False
        b, c, d, w, h = input.size()
        if d<self.size[0] or w<self.size[1] or h <self.size[2]:
            ned_pad=True
            pad_d= max(0,self.size[0]-d)
            pad_w= max(0,self.size[1]-w)
            pad_h= max(0,self.size[2]-h)
            input=torch.nn.functional.pad(input,(pad_h//2,pad_h//2+pad_h%2,pad_w//2,pad_w//2+pad_w%2,pad_d//2,pad_d//2+pad_d%2),mode='constant',value=torch.mean(input).item())
            #print(input.size())
            b, c, d, w, h = input.size()

        output = torch.zeros((b, self.n_class, d, w, h)).cuda()
        mask = torch.zeros((b, self.n_class, d, w, h)).cuda()


        drange = [n for n in range(0, d - self.size[0], self.stride[0])] + [d - self.size[0]]
        wrange = [n for n in range(0, w - self.size[1], self.stride[1])] + [w - self.size[1]]
        hrange = [n for n in range(0, h - self.size[2], self.stride[2])] + [h - self.size[2]]
        patch_pos=[]
        for i in wrange:
            for j in hrange:
                for k in drange:
                    patch_pos.append([k, k + self.size[0], i,i + self.size[1], j,j + self.size[2]])
        while len(patch_pos)%self.batch_size!=0 :
            patch_pos.append(patch_pos[-1])
        random.shuffle(patch_pos)
        assert len(patch_pos)%self.batch_size==0
        for i in range(0,len(patch_pos),self.batch_size):
            batches=[]
            for j in range(self.batch_size):
                tmp_pos=patch_pos[i+j]
                batches.append(input[:, :,  tmp_pos[0]: tmp_pos[1], tmp_pos[2]: tmp_pos[3], tmp_pos[4]: tmp_pos[5]])
            batches=torch.cat(batches,0)
            with torch.no_grad():
                if hasattr(self, 'tta'):
                    tmp_out = self.tta(model, batches)
                else:
                    tmp_out = model(batches)[0]
            for j in range(self.batch_size):
                tmp_pos = patch_pos[i+j]
                output[:, :,  tmp_pos[0]: tmp_pos[1], tmp_pos[2]: tmp_pos[3], tmp_pos[4]: tmp_pos[5]]+=tmp_out[j]
                mask[:, :,  tmp_pos[0]: tmp_pos[1], tmp_pos[2]: tmp_pos[3], tmp_pos[4]: tmp_pos[5]]+=1
        output /= mask
        if ned_pad:
            output=output[:,:,pad_d//2:-(pad_d//2+pad_d%2)]

        if softmax:
            output = output.softmax(1)
        return output
