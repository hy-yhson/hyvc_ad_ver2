import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(28 * 28, 768)
    
    def forward(self, x):
        size = x.shape[0]
        idx = torch.linspace(0, 28*28-1, steps=28*28, dtype=torch.long)
        idx = torch.repeat_interleave(idx.unsqueeze(0), repeats=size, dim=0)
        embed_x = self.embedding(idx.to('cuda'))
        x[:, :, 768:] = x[:, :, 768:] + embed_x

        return x

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]     

class feature_adaptor(nn.Module):
    def __init__(self):
        super(feature_adaptor, self).__init__()

        self.adaptor = nn.Sequential(
            nn.Linear(len_feature, len_feature), # feature adoptor
            nn.LeakyReLU(.2),
        )
    
    def forward(self, x):
        adapted_features = self.adaptor(x)
        return adapted_features

class localnet(nn.Module):
    def __init__(self, len_feature):
        super(localnet, self).__init__()
        # self.embedding = PositionalEmbedding()
        
        self.adaptor = nn.Sequential(
            nn.Linear(len_feature, len_feature), # feature adoptor
            nn.LeakyReLU(.2),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(len_feature, 1024),
            nn.LeakyReLU(.2),
            nn.Linear(1024, 128),
            nn.LeakyReLU(.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = self.embedding(x)
        adapted_features = self.adaptor(x)
        local_score = self.discriminator(adapted_features)
        # local_score = local_score.reshape(-1, 28, 28)
        return adapted_features, local_score.squeeze()

class globalnet(nn.Module):
    def __init__(self, len_feature):
        super(globalnet, self).__init__()

        # self.attention = SpatialGate()

        self.adaptor = nn.Sequential(
            nn.Linear(len_feature, len_feature), # feature adoptor
            nn.LeakyReLU(.2),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(len_feature, 1024),
            nn.LeakyReLU(.2),
            nn.Linear(1024, 128),
            nn.LeakyReLU(.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # x = x.reshape(-1, len_feature, 28, 28)
        # adapted_features = self.attention(x)
        # adapted_features = adapted_features + x
        # adapted_features = adapted_features.reshape(-1, len_feature, 784)
        # adapted_features = adapted_features.permute(0, 2, 1)
        # global_features = adapted_features.mean(axis=1).squeeze()
        # global_features = torch.logsumexp(adapted_features, dim=1)
        # global_score = self.discriminator(global_features)
        adapted_features = self.adaptor(x)
        global_score = self.discriminator(adapted_features)

        return global_score.squeeze()