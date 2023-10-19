import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
#import utils
import random

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y_fc1, y_fc2):
        eps = 1e-5
        mean_x = torch.mean(x, dim=[2,3])
		#mean_y = torch.mean(y, dim=[2,3])

        std_x = torch.std(x, dim=[2,3])
		#std_y = torch.std(y, dim=[2,3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
		#mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
		#std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps
        y_fc2 = y_fc2.unsqueeze(-1).unsqueeze(-1)
        y_fc1 = y_fc1.unsqueeze(-1).unsqueeze(-1)
        out = (x - mean_x)/ std_x * y_fc2 + y_fc1


        return out

def Block_S(inc,outc,kernel_size=3,stride=1,padding=1,norm_layer='LN',non_linerity_layer='relu'):
    norm=get_norm_layer(norm_layer)
    nll=get_non_linearity(layer_type=non_linerity_layer)
    model=[]
    if norm is not None:
        model.append(norm(inc))
    model+=[nll(),nn.ReflectionPad2d(padding),nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    return nn.Sequential(*model)
  
def padding_conv(inc,outc,kernel_size=3,stride=1,padding=1):
    model=[]
    model+=[nn.ReflectionPad2d(padding),nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    return nn.Sequential(*model)
  
def Block_E(inc,outc,kernel_size=3,stride=1,padding=1,norm_layer='LN',non_linerity_layer='relu'):
    norm=get_norm_layer(norm_layer)
    nll=get_non_linearity(layer_type=non_linerity_layer)
    model=[]
    model.append(nn.ReflectionPad2d(padding))
    model+=[nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    if norm is not None:
        model.append(norm(outc))
    model+=[nll()]
    return nn.Sequential(*model)

class resblock_conv(nn.Module):
    def __init__(self,inc,outc=None,hiddenc=None,norm_layer='LN',non_linerity_layer='relu'):
        super(resblock_conv,self).__init__()
        hiddenc=inc if hiddenc is None else hiddenc
        outc=inc if outc is None else outc
        norm=get_norm_layer(norm_layer)
        nll=get_non_linearity(layer_type=non_linerity_layer)
        model=[]
        if norm is not None:
            model.append(norm(inc))
        model+=[nll(),nn.ReflectionPad2d(1),nn.Conv2d(inc,hiddenc,3,1,padding=0)]
        if norm is not None:
            model.append(norm(hiddenc))
        model+=[nll(),nn.ReflectionPad2d(1),nn.Conv2d(hiddenc,outc,3,1,padding=0)]
        self.bp=False
        if outc!=inc:
            self.bp=True
            self.bypass=nn.Conv2d(inc,outc,1,1,0)
        self.model=nn.Sequential(*model)

    def forward(self,x):
        residual=x
        if self.bp:
            out=self.model(x)+self.bypass(residual)
        else:
            out=self.model(x)+residual
        return out

class resblock_transconv(nn.Module):
    def __init__(self,inc,outc,hiddenc=None,norm_layer='LN',non_linerity_layer='relu'):
        super(resblock_transconv,self).__init__()
        norm=get_norm_layer(norm_layer)
        nll=get_non_linearity(layer_type=non_linerity_layer)
        hiddenc=inc if hiddenc is None else hiddenc
        self.model=[]
        if norm is not None:
            self.model.append(norm(inc))
        self.model+=[nll(),
            nn.Conv2d(inc,hiddenc,3,1,1)]
        if norm is not None:
            self.model.append(norm(hiddenc))
        self.model+=[nll(),
            nn.ConvTranspose2d(hiddenc,outc,3,2,1,1)]
        self.model=nn.Sequential(*self.model)
        self.bypass=nn.Sequential(nn.ConvTranspose2d(inc,outc,3,2,1,1))

    def forward(self,x):
        residual=x
        out=self.model(x)+self.bypass(residual)
        return out
      
class Encoder_S(nn.Module):
    def __init__(self,inc=3,n_downsample=2,ndf=32,norm_layer='LN'):
        super().__init__()
        self.Inconv=padding_conv(inc,ndf,7,1,3)
        channel_in=ndf
        self.model=list()
        for _ in range(n_downsample):
            self.model+=[Block_S(channel_in,channel_in*2,4,2,1,norm_layer=norm_layer)]
            channel_in=channel_in*2
        for _ in range(2):
            self.model+=[resblock_conv(channel_in,norm_layer=norm_layer)]
        self.model=nn.Sequential(*self.model)
        self.Outconv=nn.Sequential(Block_S(channel_in,channel_in,1,1,0,norm_layer=norm_layer))
        self.outc=channel_in
        self.n_downsample=n_downsample
        #initialize_weights(self)

    def forward(self, x):
        y=self.Inconv(x)
        # ho=[]
        # for layer in self.model:
        #     y=layer(y)
        #     ho.append(y)
        y=self.model(y)
        y=self.Outconv(y)
        return y
      
class Encoder_FFT(nn.Module):
    def __init__(self,inc=3,n_downsample=4,outc=256,ndf=64,usekl=True):
        super().__init__()
        self.usekl=usekl
        self.conv1=Block_E(inc,ndf,7,1,3,norm_layer=None)
        self.downconv=[]
        channel_in=ndf
        for _ in range(2):
            self.downconv.append(Block_E(channel_in,channel_in*2,4,2,1,norm_layer=None))
            channel_in*=2
        for _ in range(n_downsample-2):
            self.downconv+=[Block_E(channel_in,channel_in,4,2,1,norm_layer=None)]
        self.downconv.append(nn.AdaptiveAvgPool2d(1))
        self.downconv=nn.Sequential(*self.downconv)
        if usekl:
            self.mean_fc =nn.Linear(channel_in, outc)
            self.var_fc= nn.Linear(channel_in, outc)
        else:
            self.fc_real= nn.Linear(channel_in, outc)
            self.fc_imag= nn.Linear(channel_in, outc)
        #initialize_weights(self)

    def forward(self, x):
        y = self.conv1(x)
        y=self.downconv(y)
        y = y.view(x.size(0), -1)
        if self.usekl:
            mean=self.mean_fc(y)
            var=self.var_fc(y)
            return mean,var
        else:
            y_real=self.fc_real(y)
            y_imag=self.fc_imag(y)
            return y_real, y_imag

class Decoder(nn.Module):
    def __init__(self,s_inc=128,e_inc=256,outc=3,n_upsample=2,norm_layer='LN'):
        super().__init__()
        inc=s_inc
        #self.adin=ADAIN(inc,e_inc)
        self.fc_mean = nn.Linear(e_inc, s_inc)
        self.fc_var = nn.Linear(e_inc, s_inc)
        self.adin=AdaIN()
        self.model=[]
        self.model+=[padding_conv(inc,inc,1,1,0),resblock_conv(inc,norm_layer=norm_layer)]
        channel_in=inc
        for _ in range(n_upsample):
            self.model+=[resblock_transconv(channel_in,channel_in//2,norm_layer=norm_layer)]
            channel_in=channel_in//2
        self.model=nn.Sequential(*self.model)
        self.outconv = nn.Sequential(
            block_nl_nll_conv(channel_in,outc,3,1,1,norm_layer=norm_layer),
        )
        #initialize_weights(self)

    def forward(self, x1,x2):
        #print(x2.shape)
        x2_mean = self.fc_mean(x2)
        x2_var = self.fc_var(x2)
        #x2 = x2.unsqueeze(-1).unsqueeze(-1)
        out = self.adin(x1, x2_mean, x2_var)
        out=self.model(out)
        out=self.outconv(out)
        #out=torch.tanh(out)
        return out
