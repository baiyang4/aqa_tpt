import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import copy
import pdb

class ActionDecoder(nn.Module):
    def __init__(self, num_cluster, d_model=256, d_ffn=512):
        super(ActionDecoder, self).__init__()
        self.query_embed = nn.Embedding(num_cluster, d_model)

        self.layers = nn.ModuleList([ActionDecoderLayer(d_model, d_ffn, isfirst=True), ActionDecoderLayer(d_model, d_ffn)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        bs, T, D = x.shape
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)
        tgt, att1 = self.layers[0](x, tgt, query_embed)
        tgt, att2 = self.layers[1](x, tgt, query_embed)
        att = torch.cat([att1, att2], dim=1)
        return tgt, att

class ActionDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, isfirst=False, dropout=0.1):
        super(ActionDecoderLayer, self).__init__()
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.key_embed = nn.Linear(1024, d_model)
        self.value_embed = nn.Linear(1024, d_model)
        self.query_embed = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.relu = nn.ReLU(True)
        if isfirst:
            self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout4 = nn.Dropout(dropout)
        self.isfirst = isfirst

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, memory, tgt, query_embedding):

        ### cross attention ###
        key = self.key_embed(memory)
        value = self.value_embed(memory)
        query = self.query_embed(self.with_pos_embed(tgt, query_embedding))

        v2l_graph_adj = torch.div(torch.matmul(key, query.transpose(-1,-2)), key.size(-1)) / self.temp
        v2l_graph_adj = F.softmax(v2l_graph_adj, dim=1) # bs x T x K
        tgt2 = torch.matmul(v2l_graph_adj.transpose(-1, -2), value)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout2(self.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)

        ### self-attention ###
        if self.isfirst:
            q = k = self.with_pos_embed(tgt, query_embedding)
            tgt2 = self.self_attn(q.permute(1,0,2).contiguous(), k.permute(1,0,2).contiguous(), value=tgt.permute(1,0,2).contiguous())[0]
            tgt2 = tgt2.permute(1,0,2).contiguous()
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm3(tgt)

        return tgt, v2l_graph_adj


class SAModule(nn.Module):
    def __init__(self, d_model, n_layers, dropout=0.1, pe=False):
        super(SAModule, self).__init__()
        self.sa_layers = clones(SelfAttentionLayer(d_model, dropout, pe), n_layers)

    def forward(self, inputs):
        output = inputs
        for layer in self.sa_layers:
            output = layer(output)

        return output


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, pe=False):
        super(SelfAttentionLayer, self).__init__()
        # self.attention = SelfAttentionBase()
        # self.qkv = nn.Linear(d_modal, d_modal * 3)
        self.attention = nn.MultiheadAttention(d_model, 1, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, 512)
        self.linear2 = nn.Linear(512, d_model)

        # self.pe = PositionalEncoding(d_model, max_len=10) if pe else None

    def forward(self, x):
        # q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        # if self.pe is not None:
        #     attention = self.attention(self.pe(q), self.pe(k), v)
        # else:
        #     attention = self.attention(q, k, v)
        x = x.transpose(0, 1)
        attention, _ = self.attention(x, x, x)
        #attention, _ = self.attention(self.pe(x), self.pe(x), x)
        # x.shape = L * B * D
        out = self.norm(x + self.dropout(attention))
        out2 = self.linear2(self.dropout1(self.activation(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm1(out)
        return out.transpose(0, 1)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        pe = Variable(self.pe[:, :x.size(0)],
                      requires_grad=False)

        x = x + pe
        return self.dropout(x)



class PairEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(PairEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x1, x2):
        # bs x k x d
        if x2.shape[1] > x1.shape[1]:
            x1 = x1.repeat(1,x2.shape[1],1,1)
        return self.mlp(torch.cat([x1, x2], dim=-1))


class RelativeScore(nn.Module):
    def __init__(self, hidden_size, weighted=False):
        super(RelativeScore, self).__init__()
        self.classify = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1)
            # nn.LayerNorm(hidden_size)
        )
        self.anchor_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.exemplar_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.weighted = weighted
        # if weighted:
        #     self.logits = nn.Linear(hidden_size, 1)

    def forward(self, v1, v2):
        scores = self.classify(self.anchor_embed(v1) - self.exemplar_embed(v2))
        if self.weighted:
            # logits = self.logits(scores)
            # scores = torch.matmul(logits.transpose(-1, -2), scores).squeeze(1)
            scores = scores.mean(1)
        return scores

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
