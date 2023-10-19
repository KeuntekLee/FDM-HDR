import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
#import utils
import random

class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta

        return out
	    
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
def get_norm_layer(type):
    if type=='LN':
        return functools.partial(LayerNorm,affine=True)
    if type=='IN':
        return functools.partial(nn.InstanceNorm2d,affine=True)
    if type=='BN':
        return functools.partial(nn.BatchNorm2d,momentum=0.1,affine=True)
    return None

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU,0.1)
  elif layer_type == 'selu':
    nl_layer = functools.partial(nn.SELU)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer
	
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

class resblock_upbilin(nn.Module):
    def __init__(self,inc,outc=None,norm_layer='LN',non_linerity_layer='relu'):
        super(resblock_upbilin,self).__init__()
        norm=get_norm_layer(norm_layer)
        nll=get_non_linearity(layer_type=non_linerity_layer)
        outc=inc if outc is None else outc
        self.model=[]
        if norm is not None:
            self.model.append(norm(inc))
        self.model+=[nll(),
            nn.Conv2d(inc,outc,3,1,1)]
        if norm is not None:
            self.model.append(norm(outc))
        self.model+=[nll(),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        self.model=nn.Sequential(*self.model)
        self.bypass=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self,x):
        residual=x
        out=self.model(x)+self.bypass(residual)
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
