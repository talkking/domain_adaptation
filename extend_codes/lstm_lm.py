import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr.data.field import Field
from asr.model import Model, add_model


logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

@add_model('LSTMLM')
class LSTMLM(Model):

    def __init__(self, nhid, nproj, nvocab, nlayers, max_norm):
        super().__init__()
        dec_in = nproj
        checkpoint = "/mnt/lustre02/jiangsu/aispeech/home/hs418/domain_adaptation/exp/trans20L-lstm2L-MWER-LMadapt/checkpoint"
        model = torch.load(checkpoint)
        self.embed = nn.Embedding(nvocab, nproj)
        # weight = model['model']['decoder.embed.weight']
        # self.embed.load_state_dict({'weight': weight})
        #self.embed.weight.requires_grad = False

        self.rnn = nn.ModuleList(
            [nn.LSTMCell(dec_in if i==0 else nhid, nhid) for i in range(nlayers)]
        )
        self.linear = nn.Linear(nhid, nvocab)
        self.max_norm = max_norm
        self.nlayers = nlayers
        self.nhid = nhid
        self.nproj = nproj
        self.nvocab = nvocab
        self.dropout = 0.0

    def rnn_forward(self, x, state):
        if state is None:
            h = [torch.zeros(x.size(0), self.nhid, device=device) for _ in range(self.nlayers)]
            state = {'h': h}
            c = [torch.zeros(x.size(0), self.nhid, device=device) for _ in range(self.nlayers)]
            state['c'] = c

        h = [None] * self.nlayers
        c = [None] * self.nlayers
        for i in range(self.nlayers):
            # dropout applied only 
            inp = x if i == 0 else F.dropout( h[i-1], self.dropout, self.training)
            
            h[i], c[i] = self.rnn[i](inp, (state['h'][i], state['c'][i]))
        state = {'h': h, 'c': c}
        return state

    def forward(self, batch):
        label = batch['label'].tensor.to(device)          # (B, L)
        label = label.masked_fill(label == -1, 0)
        #if (label.size(0) * label.size(1) > 20000):
        #    return Field(torch.zeros(label.size(0), label.size(1), self.nvocab, device='cpu'), batch['label'].length)
        bs = label.size(0)
        #import time 
        #t1 = time.time()
        eys = self.embed(label)
        #t2 = time.time()
        state = None
        h_hat = []
        for i in range(eys.size(1)):
            inp = eys[:,i,:]
            state = self.rnn_forward(inp, state)
            hdec = state['h'][-1].view(bs,1,-1)
            h_hat.append(hdec)
        h_hat = torch.cat(h_hat, dim=1)
        y_hat = self.linear(h_hat)[:,:-1,:]
        batch['label'].tensor = batch['label'].tensor[:, 1:]
        batch['label'].length -= 1
        return Field(y_hat, batch['label'].length)

    def forward_one_step(self, state, ys):
        eys = self.embed(ys)
        state = self.rnn_forward(eys, state)
        return state, self.linear(state['h'][-1])

    def grad_post_processing(self):
        """Clip the accumulated norm of all gradients to max_norm"""
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        if norm >= self.max_norm:
            logger.debug(f'Norm overflow: {norm}')
        if math.isnan(norm) or math.isinf(norm):
            self.zero_grad()
            logger.debug(f'Norm is abnormal: {norm}')
