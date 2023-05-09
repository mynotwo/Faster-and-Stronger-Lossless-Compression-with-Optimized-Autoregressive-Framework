# coding=utf-8
import numpy as np
import torch
import time
import random
import math
from torch import nn

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
    self.beta = torch.nn.Parameter(torch.ones(batch_size, self._scale)*0.01, requires_grad=True)
    self.bias = torch.nn.Parameter(torch.zeros(batch_size, self._scale), requires_grad=True)
    
    l = []
    l.append(BELayer(self._scale//1, self._vocab_dim*1, 256, batch_size, [True, True]))
    l.append(LinearLayer(hidden_dim, ffn_dim, hidden_dim, batch_size, [True, True]))
    l.append(BELayer(self._scale//2, self._vocab_dim*2, 256, batch_size, [True, True]))
    l.append(LinearLayer(hidden_dim, ffn_dim, hidden_dim, batch_size, [True, True]))
    l.append(BELayer(self._scale//4, self._vocab_dim*4, 256, batch_size, [True, True]))
    l.append(LinearLayer(hidden_dim, ffn_dim, hidden_dim, batch_size, [True, True]))
    l.append(BELayer(self._scale//8, self._vocab_dim*8, 256, batch_size, [True, True]))
    l.append(LinearLayer(hidden_dim, ffn_dim, hidden_dim, batch_size, [True, True]))
    
    self.layers = torch.nn.ModuleList(l)
    self.Pi = 0.7
    self.ONE = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
    self.ONE_ = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False).unsqueeze(1).repeat(batch_size, 1).cuda()
    self.ONE_ = self.ONE_
    self.pi = self.Pi * torch.ones(self._scale)
    self.pz = (torch.cat([self.ONE, torch.cumprod(self.pi, dim=0)]) * torch.cat([1-self.pi, self.ONE])).repeat(batch_size, 1)
    self.pz = self.pz.cuda()
    
  def get_mask(self, beta, rep):
    beta = torch.sigmoid(torch.clamp(beta, -5, 5))
    sample = torch.nn.functional.gumbel_softmax(beta, tau=5, hard=False)
    ordered_mask = sample.cumsum(dim=1)
    ordered_mask = ordered_mask.repeat_interleave(rep, dim=1).unsqueeze(1)

    return ordered_mask, beta


  def forward(self, x, tau=10):
    """Naive full forward pass."""
    x = self.input_map(x)
    bs, seqlen, vlen = x.shape
    
    ordered_mask, beta = self.get_mask(self.beta, self._vocab_dim)
    
    x = x.reshape(bs, seqlen // self._scale, vlen*self._scale)#.squeeze(1) 
    x = x*ordered_mask+self.bias.repeat_interleave(256//self._scale, dim=1).unsqueeze(1)
    for i, layer in enumerate(self.layers):
      x = layer.full_forward(x)
    x = self.output_logit_map(x)

    return x, beta

  def full_loss(self,
                inputs,
                with_grad=True,
                nonpad_mask=None,
                return_acc=False):
    """Naive full loss and grad."""
    tau=5
    logits, beta = self.forward(inputs[:, :-1], tau)
    logits = logits.squeeze(1)
    
    loss = torch.nn.functional.cross_entropy(
            logits, inputs[:, -1], reduction='mean')
     
    pz = self.pz.detach()
    ONE_ = self.ONE_.detach()
    
    qz = torch.cat([ONE_, torch.cumprod(beta, dim=1)], dim=-1) * torch.cat([1-beta, ONE_], dim=-1)
    log_frac_qz_pz = torch.log(1e-8 + qz / pz)
    kl = torch.diagonal(torch.mm(qz, log_frac_qz_pz.T), 0)
    kl = kl.mean()
    
    loss_ = loss + kl
    
    if with_grad:
      loss_.backward()

    return loss, logits
    
  def _concat_pos_embs(self, x, start_index):

    pos_emb_size = self._vocab_dim // 2

    positions = torch.arange(
        start_index, start_index + x.shape[1], dtype=x.dtype, device=x.device)
    freqs = torch.exp(
        torch.arange(0, pos_emb_size, 2, dtype=x.dtype, device=x.device) *
        (-np.log(10000) / pos_emb_size))
    args = positions[None, :, None] * freqs[None, None, :]
    sin_pos_embs = torch.sin(args) * torch.ones_like(x[:, :1, :1])
    cos_pos_embs = torch.cos(args) * torch.ones_like(x[:, :1, :1])
    return torch.cat([x, sin_pos_embs, cos_pos_embs], 2)

class dense_baens(nn.Module):
  def __init__(self, N=5, D1=3, D2=2):
    super(dense_baens, self).__init__()

    self.N = N
    self.D1 = D1
    self.D2 = D2
    self.U = nn.Parameter(torch.normal(0, 0.01, (N, D1, D2)), requires_grad=True)
    self.bias = nn.Parameter(torch.normal(0, 0.01, (N, D2)), requires_grad=True)

  def forward(self, x):
    act = torch.bmm(x.permute(1,0,2), self.U).permute(1,0,2)
    act += self.bias
    
    return act

class BELayer(torch.nn.Module):

  def __init__(self, branch, vocab_dim, ffn_dim, batch_size, ea):

    super(BELayer, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.ffn_dim = ffn_dim
    self.batch_size = batch_size
    self.U_map = dense_baens(branch, vocab_dim, ffn_dim)
    self.V_map = dense_baens(branch, ffn_dim, vocab_dim)
    self.layernorm2 = torch.nn.LayerNorm(vocab_dim, elementwise_affine=True)

  def full_forward(self, x):
    
    x = x.reshape(self.batch_size, self.branch, self.vocab_dim)
    skip = x
    
    x = self.U_map(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map(x)
    
    x = self.layernorm2(x)
    x = torch.nn.functional.gelu(x)
    x = skip + x
    x = x.reshape(self.batch_size, 1, self.branch*self.vocab_dim)

    return x

class LinearLayer(torch.nn.Module):

  def __init__(self, hidden_dim, ffn_dim, out_dim, n_heads, ea):

    super(LinearLayer, self).__init__()

    self.U_map = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
    self.V_map = torch.nn.Linear(ffn_dim, out_dim, bias=False)
    self.layernorm2 = torch.nn.LayerNorm(out_dim, elementwise_affine=ea[1])
    self.ln1 = 1
  
  def full_forward(self, x):

    x = self._ffn(x)

    return x

  def _ffn(self, x):
    
    skip = x
    
    x = self.U_map(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map(x)
    
    x = self.layernorm2(x)
    x = torch.nn.functional.gelu(x)
    
    x = skip + x


    return x
