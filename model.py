
import torch
import torch.nn as nn
import einops
from einops import rearrange
import torch.nn.functional as F
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class MultiscaleAlign(nn.Module):
    def __init__(self, dim):
        super(MultiscaleAlign, self).__init__()
        self.act = nn.GELU()
        self.reduce = nn.Conv2d(256, dim,kernel_size=1)
        self.down1 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        #self.upsample_x2 = F.interpolate(scale_factor=2)
        #self.upsample_x4 = F.interpolate(scale_factor=4)
        #self.downsample_x2 = F.interpolate(scale_factor=0.5)
        #self.downsample_x4 = F.interpolate(scale_factor=0.25)

        #self.conv_inp_local = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_nonref = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_ref = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_mask_x4 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_mask_x2 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_mask_x1 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_x2 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_x1 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        #self.reduce_conv = nn.Conv2d(64,dim,kernel_size=1)
    def forward(self, inp, ref, local_feat_inp, local_feat_ref):
        local_feat_inp = self.reduce(local_feat_inp)
        local_feat_ref = self.reduce(local_feat_ref)
        inp_down_x2 = self.down1(inp)
        ref_down_x2 = self.down1(ref)
        inp_down_x4 = self.down2(self.act(inp_down_x2))
        ref_down_x4 = self.down2(self.act(ref_down_x2))

        x4_nonref = self.conv_nonref(torch.cat([inp_down_x4,local_feat_inp],dim=1))
        x4_ref = self.conv_ref(torch.cat([ref_down_x4,local_feat_ref],dim=1))

        x4_mask = self.conv_mask_x4(torch.cat([x4_ref,x4_nonref],dim=1))
        x4_mask = F.interpolate(x4_mask, scale_factor=2)

        x2_nonref = self.conv_x2(torch.cat([inp_down_x2, x4_mask],dim=1))
        x2_mask = self.conv_mask_x2(torch.cat([ref_down_x2,x2_nonref],dim=1))
        x2_mask = F.interpolate(x2_mask, scale_factor=2)

        x_nonref = self.conv_x1(torch.cat([inp, x2_mask],dim=1))
        x_mask = self.conv_mask_x1(torch.cat([ref,x_nonref],dim=1))


        feat = F.sigmoid(x_mask)
        return inp * feat

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
      
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class FreqMerging(nn.Module):
    def __init__(self,in_channels=256):
        super(FreqMerging, self).__init__()
        self.freq_real_fcs = nn.ModuleList([])
        self.dim = in_channels
        for i in range(3):
            self.freq_real_fcs.append(nn.Conv2d(256, 256, kernel_size=1))
        self.freq_imag_fcs = nn.ModuleList([])
        for i in range(3):
            self.freq_imag_fcs.append(nn.Conv2d(256, 256, kernel_size=1))

        self.softmax = nn.Softmax(dim=1)
    def forward(self, lumi_feats_real, lumi_feats_imag):
        batch_size = lumi_feats_real[0].shape[0]
        freq_real_attention_vectors = [self.freq_real_fcs[0](lumi_feats_real[0].unsqueeze(-1).unsqueeze(-1)), 
        self.freq_real_fcs[1](lumi_feats_real[1].unsqueeze(-1).unsqueeze(-1)), 
        self.freq_real_fcs[2](lumi_feats_real[2].unsqueeze(-1).unsqueeze(-1))]
        
        #freq_attention_vectors = []
        freq_real_attention_vectors = torch.cat(freq_real_attention_vectors, dim=1)
        freq_real_attention_vectors = freq_real_attention_vectors.view(batch_size, 3, self.dim, 1, 1)
        freq_real_attention_vectors = self.softmax(freq_real_attention_vectors)
        #print(inp_feats.shape, attention_vectors.shape)
        
        #freq_real_merged = torch.sum(torch.stack([freq_1.real, freq_2.real, freq_3.real], dim=1) * freq_real_attention_vectors, dim=1)

        freq_imag_attention_vectors = [self.freq_imag_fcs[0](lumi_feats_imag[0].unsqueeze(-1).unsqueeze(-1)), 
        self.freq_imag_fcs[1](lumi_feats_imag[1].unsqueeze(-1).unsqueeze(-1)), 
        self.freq_imag_fcs[2](lumi_feats_imag[2].unsqueeze(-1).unsqueeze(-1))]
        
        #freq_attention_vectors = []
        freq_imag_attention_vectors = torch.cat(freq_imag_attention_vectors, dim=1)
        freq_imag_attention_vectors = freq_imag_attention_vectors.view(batch_size, 3, self.dim, 1, 1)
        freq_imag_attention_vectors = self.softmax(freq_imag_attention_vectors)
        #print(inp_feats.shape, attention_vectors.shape)
        
        #freq_imag_merged = torch.sum(torch.stack([freq_1.imag, freq_2.imag, freq_3.imag], dim=1) * freq_imag_attention_vectors, dim=1)
        #print(torch.stack(lumi_feats_real, dim=1).shape, freq_real_attention_vectors.shape)
        freq_real_attention_vec = torch.sum(torch.stack(lumi_feats_real, dim=1).unsqueeze(-1).unsqueeze(-1) * freq_real_attention_vectors, dim=1)
        freq_imag_attention_vec = torch.sum(torch.stack(lumi_feats_imag, dim=1).unsqueeze(-1).unsqueeze(-1) * freq_imag_attention_vectors, dim=1)
        return freq_real_attention_vec, freq_imag_attention_vec
      
class FreqModulate(nn.Module):
    def __init__(self,in_channels=256):
        super(FreqModulate, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.fconv_1 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=1)
        self.fconv_2 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=1)
        self.fc = nn.Linear(512, in_channels*2)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()
    def forward(self, x, real_vec, imag_vec):
        b,c,h,w = x.shape
        #batch_size = lumi_feats_real[0].shape[0]
        x_identity = x
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x_fft = torch.fft.rfft2(x)
        x_real = x_fft.real
        x_imag = x_fft.imag

        x_fft_cat = torch.cat([x_real, x_imag], dim=1)
        x_fft_cat = self.fconv_1(x_fft_cat)
        x_fft_cat = self.act(x_fft_cat)
        x_fft_cat = self.fconv_2(x_fft_cat)
        #print(real_vec.shape)
        modulate_vec = self.fc(torch.cat([real_vec.squeeze(-1).squeeze(-1), imag_vec.squeeze(-1).squeeze(-1)],dim=1))
        modulate_vec = self.sigmoid(modulate_vec)
        #print(x_fft_cat.shape, modulate_vec.shape)
        x_fft_cat = x_fft_cat * modulate_vec.unsqueeze(-1).unsqueeze(-1)

        x_real, x_imag = torch.chunk(x_fft_cat, 2, dim=1)
        
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x, s=(h, w))

        x = self.conv_3(x)
        x = self.act(x)
        x = self.conv_4(x)
        x = x + x_identity

        return x

class FDMNet(nn.Module):
    def __init__(self, 
        inp_channels=6, 
        dim = 32,
        num_blocks = [1,1,1,2], 
        num_refinement_blocks = 4,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(FDMNet, self).__init__()

        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed2 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed3 = OverlapPatchEmbed(inp_channels, dim)
        
        #self.sp_att1 = SpatialAttention(dim)
        #self.sp_att3 = SpatialAttention(dim)

        self.sp_att1 = MultiscaleAlign(dim)
        self.sp_att3 = MultiscaleAlign(dim)

        self.encoder_level1_1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_3 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce = nn.Conv2d(dim*3,dim,kernel_size=1)
        
        #self.freq_merging1 = TransformerBlock_Freq(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.freq_modulate1 = FreqModulate(dim)
        self.bottleneck1 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.vec_merge1 = FreqMerging(256)

        #self.freq_merging2 = TransformerBlock_Freq(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.freq_modulate2 = FreqModulate(dim)
        self.bottleneck2 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.vec_merge2 = FreqMerging(256)

        #self.freq_merging3 = TransformerBlock_Freq(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.freq_modulate3 = FreqModulate(dim)
        self.bottleneck3 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.vec_merge3 = FreqMerging(256)

        #self.freq_merging4 = TransformerBlock_Freq(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.freq_modulate4 = FreqModulate(dim)
        self.bottleneck4 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.vec_merge4 = FreqMerging(256)

        self.freq_modulate5 = FreqModulate(dim)
        self.bottleneck5 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.vec_merge5 = FreqMerging(256)

        self.freq_modulate6 = FreqModulate(dim)
        self.bottleneck6 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.vec_merge6 = FreqMerging(256)

        
        #self.refinement = nn.ModuleList([TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x1, x2, x3, lumi_feats_real, lumi_feats_imag, spatial_feats):
        #print()
        F1_1 = self.patch_embed1(x1)
        F2_1 = self.patch_embed2(x2)
        F3_1 = self.patch_embed3(x3)

        F1_1 = self.sp_att1(F1_1, F2_1, spatial_feats[0], spatial_feats[1])
        F3_1 = self.sp_att3(F3_1, F2_1, spatial_feats[2], spatial_feats[1])

        F1_1 = self.encoder_level1_1(F1_1)
        F2_1 = self.encoder_level1_2(F2_1)
        F3_1 = self.encoder_level1_3(F3_1)

        F_M = torch.cat([F1_1,F2_1,F3_1],dim=1)

        F_M = self.reduce(F_M)
        real_merged, imag_merged = self.vec_merge1(lumi_feats_real,lumi_feats_imag)
        F_M = self.freq_modulate1(F_M,real_merged,imag_merged)
        for bottleneck in self.bottleneck1:
            F_M = bottleneck(F_M)

        real_merged, imag_merged = self.vec_merge2(lumi_feats_real,lumi_feats_imag)
        F_M = self.freq_modulate2(F_M,real_merged,imag_merged)
        for bottleneck in self.bottleneck2:
            F_M = bottleneck(F_M)

        #real_merged, imag_merged = self.vec_merge3(lumi_feats_real,lumi_feats_imag)
        #F_M = self.freq_merging3(F_M,real_merged,imag_merged)
        real_merged, imag_merged = self.vec_merge3(lumi_feats_real,lumi_feats_imag)
        F_M = self.freq_modulate3(F_M,real_merged,imag_merged)
        for bottleneck in self.bottleneck3:
            F_M = bottleneck(F_M)

        #real_merged, imag_merged = self.vec_merge4(lumi_feats_real,lumi_feats_imag)
        #F_M = self.freq_merging4(F_M,real_merged,imag_merged)
        real_merged, imag_merged = self.vec_merge4(lumi_feats_real,lumi_feats_imag)
        F_M = self.freq_modulate4(F_M,real_merged,imag_merged)
        for bottleneck in self.bottleneck4:
            F_M = bottleneck(F_M)

        real_merged, imag_merged = self.vec_merge5(lumi_feats_real,lumi_feats_imag)
        F_M = self.freq_modulate5(F_M,real_merged,imag_merged)
        for bottleneck in self.bottleneck5:
            F_M = bottleneck(F_M)
        
        real_merged, imag_merged = self.vec_merge6(lumi_feats_real,lumi_feats_imag)
        F_M = self.freq_modulate6(F_M,real_merged,imag_merged)
        for bottleneck in self.bottleneck6:
            F_M = bottleneck(F_M)
        
        
        '''
        for refinement in self.refinement:
            F_M = refinement(F_M)
        '''
        out = self.output(F_M)
        out = F.tanh(out)
        return out
