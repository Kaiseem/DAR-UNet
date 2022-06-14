import torch
import torch.nn as nn
import torch.nn.functional as F

class ZPool(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([x.max(self.dim, keepdim=True).values, x.mean(self.dim, keepdim=True)], dim=self.dim)

class DimAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.compress = nn.Sequential(
            ZPool(dim=1),
            nn.Conv3d(2, 1, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(1, eps=1e-5, momentum=0.01),
            nn.Sigmoid()
        )
        self.dim = dim

    def forward(self, x):
        if self.dim != 1:
            x = x.transpose(self.dim, 1).contiguous()
        out = x * self.compress(x)
        if self.dim != 1:
            out = out.transpose(self.dim, 1).contiguous()
        return out

class QAM(nn.Module):
    def __init__(self,dims=[1,2,3,4]) -> None:
        super().__init__()
        self.dims=dims
        self.branchs=nn.ModuleList()
        for d in dims:
            self.branchs.append(DimAttention(dim=d))

    def forward(self, x):
        y=0
        for b in self.branchs:
            y+=b(x)
        y/=len(self.dims)
        return y

class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, anisotropy_stride=False, from_image=False, anisotropy_dim=0):
        super(ResBlock, self).__init__()
        stride=[stride,stride,stride]

        if anisotropy_stride:
            stride[anisotropy_dim]=1

        if from_image:
            self.conv_block = nn.Sequential(
                nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),
                nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.GroupNorm(input_dim // 8, input_dim, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),

                nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

                nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            )

        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),
            nn.GroupNorm(output_dim // 8, output_dim, affine=True),
        )

        self.qam=QAM()

    def forward(self, x):
        return self.qam(self.conv_block(x) + self.conv_skip(x))

class VAM(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, output_dim, anisotropy_stride=False,anisotropy_dim=0):
        super(VAM, self).__init__()
        stride = [2, 2, 2]
        padding= [1,1,1]
        if anisotropy_stride:
            stride[anisotropy_dim]=1
            padding[anisotropy_dim]=0

        self.conv_encoder = nn.Sequential(
            nn.GroupNorm(encoder_dim//8,encoder_dim,affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(encoder_dim, output_dim, 3, padding=1),
            nn.MaxPool3d(stride,stride),
        )

        self.conv_decoder = nn.Sequential(
            nn.GroupNorm(decoder_dim//8,decoder_dim,affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(decoder_dim, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.GroupNorm(output_dim//8, output_dim,affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_dim, 1, 1),
        )

        self.upsample = nn.ConvTranspose3d(output_dim,output_dim, kernel_size=3, stride=stride, padding=1, output_padding=padding)

    def forward(self, x_e, x_d):
        out = self.conv_encoder(x_e) + self.conv_decoder(x_d)
        out = self.conv_attn(out)
        out = out * x_d
        out=self.upsample(out)
        return torch.cat([out,x_e],1)

def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:] ,mode='trilinear', align_corners=False)
    return src

class DARUnet(nn.Module):
    def __init__(self,num_classes=3):
        super(DARUnet, self).__init__()
        channels=[16,32,64,128,256]

        self.L1_fromimg = ResBlock(1,channels[0],stride=1,from_image=True)
        self.L2_down = ResBlock(channels[0], channels[1], stride=2, anisotropy_stride=True)
        self.L3_down = ResBlock(channels[1], channels[2], stride=2, anisotropy_stride=True)
        self.L4_down = ResBlock(channels[2], channels[3], stride=2)
        self.L5_down = ResBlock(channels[3], channels[4], stride=2)

        self.vam4=VAM(channels[3], channels[4], channels[4])
        self.L4_up=ResBlock(channels[3] + channels[4], channels[3])

        self.vam3 = VAM(channels[2], channels[3], channels[3])
        self.L3_up = ResBlock(channels[2] + channels[3], channels[2])

        self.vam2 = VAM(channels[1], channels[2], channels[2], anisotropy_stride=True)
        self.L2_up = ResBlock(channels[1] + channels[2], channels[1] ,anisotropy_stride=True)

        self.vam1 = VAM(channels[0], channels[1], channels[1], anisotropy_stride=True)
        self.L1_up = ResBlock(channels[0] + channels[1], channels[0], anisotropy_stride=True)

        self.sides=nn.ModuleList([nn.Conv3d(channels[0],num_classes,3,1,1),
                                  nn.Conv3d(channels[1], num_classes, 3, 1, 1),
                                  nn.Conv3d(channels[2], num_classes, 3, 1, 1),
                                  nn.Conv3d(channels[3], num_classes, 3, 1, 1),
                                  nn.Conv3d(channels[4], num_classes, 3, 1, 1),
        ])

        self.outconv = nn.Conv3d(num_classes*5, num_classes, 3, padding=1)

    def forward(self, x):
        x0_0 = self.L1_fromimg(x)
        x1_0 = self.L2_down(x0_0)
        x2_0 = self.L3_down(x1_0)
        x3_0 = self.L4_down(x2_0)
        x4_1 = self.L5_down(x3_0)

        x3_2=self.L4_up(self.vam4(x3_0, x4_1))
        x2_3=self.L3_up(self.vam3(x2_0, x3_2))
        x1_4=self.L2_up(self.vam2(x1_0,x2_3))
        x0_5=self.L1_up(self.vam1(x0_0,x1_4))

        d1= self.sides[0](x0_5)
        d2= self.sides[1](x1_4)
        d3= self.sides[2](x2_3)
        d4= self.sides[3](x3_2)
        d5= self.sides[4](x4_1)

        d2 = _upsample_like(d2,d1)
        d3 = _upsample_like(d3,d1)
        d4 = _upsample_like(d4,d1)
        d5 = _upsample_like(d5,d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
        return d0,d1,d2,d3,d4,d5


if __name__=="__main__":
    model=DARUnet().cuda()
    input=torch.rand((2,1,32,256,256)).cuda()
    print(model(input))

