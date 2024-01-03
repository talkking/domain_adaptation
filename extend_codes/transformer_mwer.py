import logging
import math

import torch
import torch.nn as nn

from asr.model.transformer import Transformer
from asr.model import add_model
from asr.data.field import Field
from asr.model.transformer.vgg import VGG2L
logger = logging.getLogger(__name__)

@add_model('TRANSMWER')
class TRANSMWER(Transformer):
    # mode can be ce, ctc, mt, mmi, las
    def __init__(self, ninp, nhid=2048, nproj=512, nctc=2966, natt=6979,
                 nlayer=12, nhid_dec=1024, ndecode=2, max_norm=1000, activation='relu6', dec='lstm', dropout=0.1, pos_emb=True, beam=4, beta=0):
        super().__init__(ninp, nhid, nproj, nctc, natt, nlayer, nhid_dec, ndecode, max_norm, activation, dec, dropout, pos_emb, 'las')
        self.conv = VGG2L(1, 128, keepbn=True)
        self.beam = beam
        self.beta = beta

    def forward(self, batch):
        xs = batch['feat'].tensor.cuda()
        length = batch['feat'].length
        label = batch['label'].tensor.cuda()            # (B, L)

        xs, length = self.conv(xs, length)
        xs = self.proj(xs).transpose(0, 1)
        cu_length = length.cuda()
        memory_key_padding_mask = cu_length.unsqueeze(1) <= torch.arange(0, xs.size(0), device='cuda').unsqueeze(0)
        for i in range(len(self.att)):
            xs = self.att[i](xs, memory_key_padding_mask)

        xs = xs.detach()
        max_length = (cu_length + 5) // 2
        nbest = self.decoder.decode_e2e(xs, memory_key_padding_mask, max_length, self.beam, self.beam, 1)

        label = label.masked_fill(label == -1, 0)
        ss = self.beta
        if not self.training and ss > 0:
            ss=1
        output = self.decoder(xs, memory_key_padding_mask, label, sampling_prob=ss)[:, :-1, :]
        batch['label'].tensor = batch['label'].tensor[:, 1:]
        batch['label'].length -= 1
        
        return Field(output, batch['label'].length), nbest

    def grad_post_processing(self):
        """Clip the accumulated norm of all gradients to max_norm"""
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        if norm >= self.max_norm:
            logger.debug(f'Norm overflow: {norm}')
        if math.isnan(norm) or math.isinf(norm):
            self.zero_grad()
            logger.debug(f'Norm is abnormal: {norm}')
