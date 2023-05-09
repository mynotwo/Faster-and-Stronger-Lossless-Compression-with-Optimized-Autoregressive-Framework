# coding=utf-8
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
import random
import math

class mlpCompressor(torch.nn.Module):

  def __init__(self, vocab_size, vocab_dim,  hidden_dim, n_layers, ffn_dim, n_heads,
               batch_size):
    super(mlpCompressor, self).__init__()

    self._vocab_size = vocab_size
    self._vocab_dim = vocab_dim
    self._hidden_dim = hidden_dim
    self._scale = hidden_dim // vocab_dim
    self.input_map = torch.nn.Embedding(vocab_size, vocab_dim)
    self.output_logit_map = torch.nn.Linear(hidden_dim, vocab_size)
    
    torch.nn.init.normal_(self.input_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.output_logit_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.output_logit_map.bias, 0, 0.01)
    
    self.layernorm1 = LayerNorm((batch_size, 1, hidden_dim))
    self.batch_size = batch_size 
    l = []
    
    l.append(SLiMPerformerBELayer(16, 16, 64, batch_size, [True, True]))
    l.append(SLiMPerformerOrgLayer(hidden_dim, 4096, hidden_dim, batch_size, [True, True]))
    l.append(SLiMPerformerBELayer(8, 32, 64, batch_size, [True, True]))
    l.append(SLiMPerformerOrgLayer(hidden_dim, 4096, hidden_dim, batch_size, [True, True]))
    l.append(SLiMPerformerBELayer(4, 64, 64, batch_size, [True, True]))
    l.append(SLiMPerformerOrgLayer(hidden_dim, 4096, hidden_dim, batch_size, [True, True]))
    l.append(SLiMPerformerBELayer(2, 128, 64, batch_size, [True, True]))
    l.append(SLiMPerformerOrgLayer(hidden_dim, 4096, hidden_dim, batch_size, [True, True]))
    
    self.layers = torch.nn.ModuleList(l)
    self.last = []
    
  def forward(self, x, last=False):
    """Naive full forward pass."""
    emb = torch.sigmoid(self.input_map(x))
    bs, seqlen, vlen = emb.shape
    x = emb.reshape(bs, seqlen // self._scale, vlen*self._scale) 
    for i, layer in enumerate(self.layers):
      x = layer.full_forward(x)
      
    x = self.output_logit_map(x)

    return x, emb

  def min_max(self, inp):
    min_, e = torch.min(inp, 1)
    max_, e = torch.max(inp, 1)
    
    out = (inp.squeeze(-1)-min_)/max_
    return out.unsqueeze(2)

  def full_loss(self,
                inputs,
                with_grad=True,
                nonpad_mask=None,
                return_acc=False):
    """Naive full loss and grad."""

    logits, emb = self.forward(inputs[:, :-1])
    logits = logits.transpose(1, 2)
    loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], inputs[:, -1], reduction='mean')
  
    if with_grad:
      loss.backward()

    return loss, logits
    

class dense_baens(nn.Module):
  def __init__(self, N=5, B=4, D1=3, D2=2):
    super(dense_baens, self).__init__()

    self.N = N
    self.B = B
    self.D1 = D1
    self.D2 = D2
    self.U = nn.Parameter(torch.normal(0, 0.01, (N, D1, D2)), requires_grad=True)
    self.bias = nn.Parameter(torch.normal(0, 0.01, (N, B, D2)), requires_grad=True)

  def forward(self, x):
    act = torch.bmm(x, self.U)
    act += self.bias
    return act

class SLiMPerformerBELayer(torch.nn.Module):

  def __init__(self, branch, vocab_dim, ffn_dim, batch_size, ea=[True, True], trans=False):

    super(SLiMPerformerBELayer, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.ffn_dim = ffn_dim
    self.batch_size = batch_size
    self.V_map = dense_baens(batch_size, branch, vocab_dim, vocab_dim)
    self.layernorm1 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[0])
    self.layernorm2 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[1])
    self.trans = trans

  def full_forward(self, x):
    x = x.reshape(self.batch_size, self.branch, self.vocab_dim)

    x = self.layernorm1(x)
    
    skip = x
    
    x = self.V_map(x)
    
    x = self.layernorm2(x)
    x = torch.nn.functional.gelu(x)
    x = (skip + x)/2
    x = x.reshape(self.batch_size, 1, self.branch*self.vocab_dim)

    return x

class SLiMPerformerOrgLayer(torch.nn.Module):
  def __init__(self, hidden_dim, ffn_dim, out_dim, n_heads, ea=[True, True]):

    super(SLiMPerformerOrgLayer, self).__init__()

    self.U_map = torch.nn.Linear(hidden_dim, ffn_dim, bias=True)
    torch.nn.init.normal_(self.U_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.U_map.bias, 0, 0.01)
    
    self.V_map = torch.nn.Linear(ffn_dim, out_dim, bias=True)
    torch.nn.init.normal_(self.V_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.V_map.bias, 0, 0.01)
    
    self.layernorm1 = torch.nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=ea[0])
    self.layernorm2 = torch.nn.LayerNorm(out_dim, eps=1e-05, elementwise_affine=ea[1])

  def full_forward(self, x):

    x = self._ffn(x)

    return x

  def _ffn(self, x):
    
    x = self.layernorm1(x)
    
    skip = x
    
    x = self.U_map(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map(x)
    
    x = self.layernorm2(x)
    x = torch.nn.functional.gelu(x)
    
    x = (skip + x)/2


    return x
