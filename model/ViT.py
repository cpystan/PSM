
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings+1, embedding_dim)
        self.seq_length = seq_length+1

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]


        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src



class VisionTransformer_Encoder(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False
    ):
        super(VisionTransformer_Encoder, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * 3

        self.out_dim = patch_dim * patch_dim * 3

        self.no_pos = no_pos
        self.head = nn.Sequential(default_conv(3, 64, 3),
                                  nn.ReLU(),
                                  ResBlock(default_conv, 64, 3),
                                  nn.ReLU(),
                                  ResBlock(default_conv, 64, 3),
                                  nn.ReLU(),
                                  ResBlock(default_conv, 64, 3),
                                  nn.ReLU(),
                                  default_conv(64, 3, 3))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        if self.mlp == False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)

            #self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)


        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, 1)
        )


    def forward(self, x, con=False):
        x = self.head(x)

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).contiguous()

        if self.mlp == False:
            x = self.linear_encoding(x)
            x = self.dropout_layer1(x)

            #query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        #else:
            #query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x)

        x = torch.cat((self.cls_token.repeat(x.shape[0],1,1), x), dim=1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
        elif self.no_pos:
            x = self.encoder(x)
        else:
            x = self.encoder(x + pos)

        x = self.output1(x[:,-1,:])
        return x

class encoder(nn.Module):
    def __init__(self, n_colors,n_feats,patch_size,patch_dim,
                 num_heads,num_layers,num_queries,
                 dropout_rate,no_mlp,pos_every,
                 no_pos,no_norm,
                 conv=default_conv):
        super(encoder, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        self.body = VisionTransformer_Encoder(img_dim=patch_size, patch_dim=patch_dim, num_channels=n_feats,
                                      embedding_dim=n_feats * patch_dim , num_heads=num_heads,
                                      num_layers=num_layers,
                                      hidden_dim=n_feats * patch_dim  * 4,
                                      num_queries=num_queries, dropout_rate=dropout_rate, mlp=no_mlp,
                                      pos_every=pos_every, no_pos=no_pos, no_norm=no_norm)

    def forward(self, x):

        x = self.body(x)

        return x




class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()
        n_colors = args.n_colors
        n_feats = args.n_feats
        patch_size = args.patch_size
        patch_dim = args.patch_dim
        #num_channels = n_feats
        #embedding_dim = n_feats * args.patch_dim * args.patch_dim
        num_heads = args.num_heads
        num_layers = args.num_layers
        #hidden_dim = n_feats * args.patch_dim * args.patch_dim * 4
        num_queries = args.num_queries
        dropout_rate = args.dropout_rate
        no_mlp = args.no_mlp
        pos_every = args.pos_every
        no_pos = args.no_pos
        no_norm = args.no_norm
        self. input_feature = None

        self.encoder = encoder(n_colors,n_feats,patch_size,patch_dim,
                 num_heads,num_layers,num_queries,
                 dropout_rate,no_mlp,pos_every,
                 no_pos,no_norm,
                 conv=default_conv)


    def forward(self,x):
        return self.encoder(x)