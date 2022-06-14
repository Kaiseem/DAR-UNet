import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,affine=True):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight+1, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class ResBlk(nn.Module):
    expansion = 1
    def __init__(self, planes, norm_layer):
        super(ResBlk, self).__init__()

        self.conv1=nn.Conv2d(planes,planes,(3,3),(1,1),(1,1))
        self.norm1 = norm_layer(planes, affine=True)

        self.conv2 =nn.Conv2d(planes,planes,(3,3),(1,1),(1,1))
        self.norm2 = norm_layer(planes, affine=True)

        self.act = nn.LeakyReLU(0.2,True)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out += identity
        return out / math.sqrt(2)

class StyleEncoder(nn.Module):
    def __init__(self,  style_dim=8):
        super(StyleEncoder, self).__init__()
        nef=32
        act = nn.LeakyReLU(0.2,True)
        main = []
        main +=  [nn.Conv2d(1, nef, (7, 7), (1, 1), (3, 3)), nn.InstanceNorm2d(nef, affine=True), act]
        main += [nn.Conv2d(nef, nef * 2, (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(nef * 2, affine=True), act]
        main += [nn.Conv2d(nef * 2, nef * 4, (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(nef * 4, affine=True), act]
        main += [nn.Conv2d(nef * 4, nef * 4, (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(nef * 4, affine=True), act]
        main += [nn.Conv2d(nef * 4, nef * 4, (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(nef * 4, affine=True), act]
        main += [nn.Conv2d(nef * 4, nef * 4, (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(nef * 4, affine=True), act]
        main += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        main += [nn.Conv2d(nef * 4, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*main)

    def forward(self, x):
        return self.model(x).squeeze(-1).squeeze(-1)

class ContentEncoder(nn.Module):
    def __init__(self, nef=32, norm=nn.InstanceNorm2d):
        super(ContentEncoder, self).__init__()
        main = []
        main += [nn.Conv2d(1, nef, (7, 7), (1, 1), (3, 3)), nn.InstanceNorm2d(nef, affine=True),nn.LeakyReLU(0.2,True)]
        main += [nn.Conv2d(nef, nef * 2, (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(nef * 2, affine=True), nn.LeakyReLU(0.2,True)]
        main += [nn.Conv2d(nef * 2, nef * 4, (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(nef * 4, affine=True), nn.LeakyReLU(0.2,True)]
        main += [ResBlk(nef * 4, norm_layer=norm)]
        main += [ResBlk(nef * 4, norm_layer=norm)]
        main += [ResBlk(nef * 4, norm_layer=norm)]
        main += [ResBlk(nef * 4, norm_layer=norm)]
        self.main = nn.Sequential(*main)

    def forward(self, x):
        return self.main(x)

class Decoder(nn.Module):
    def __init__(self, ndf=32,style_dim=8):
        super(Decoder, self).__init__()
        main = []
        main += [ResBlk(ndf * 4, norm_layer=AdaptiveInstanceNorm2d)]
        main += [ResBlk(ndf * 4, norm_layer=AdaptiveInstanceNorm2d)]
        main += [ResBlk(ndf * 4, norm_layer=AdaptiveInstanceNorm2d)]
        main += [ResBlk(ndf * 4, norm_layer=AdaptiveInstanceNorm2d)]

        main += [AdaptiveInstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2,True)]
        main += [nn.Upsample(scale_factor=2), nn.Conv2d(ndf * 4, ndf * 2, (5, 5), (1, 1), (2, 2)),
                 AdaptiveInstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2,True)]
        main += [nn.Upsample(scale_factor=2), nn.Conv2d(ndf * 2, ndf * 1, (5, 5), (1, 1), (2, 2)),
                 AdaptiveInstanceNorm2d(ndf), nn.LeakyReLU(0.2,True)]
        main += [nn.Conv2d(ndf, 1, (7, 7), (1, 1), (3, 3)), nn.Tanh()]
        self.main = nn.Sequential(*main)

        mlp = []
        mlp += [nn.Linear(style_dim, 128),nn.ReLU(True)]
        mlp += [nn.Linear(128, 256), nn.ReLU(True)]
        mlp += [nn.Linear(256, self.get_num_adain_params(self.main))]
        self.mlp=nn.Sequential(*mlp)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def forward(self,c, s):
        adain_params=self.mlp(s)
        self.assign_adain_params(adain_params,self.main)
        return self.main(c)
