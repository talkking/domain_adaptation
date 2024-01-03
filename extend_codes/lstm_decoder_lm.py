from copy import deepcopy
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import kenlm

from asr.model.transformer.decoder.lstm_decoder import LSTMDecoder
from asr.utils.dynload import build_model
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
class LSTMDecoderNGRAM(LSTMDecoder):

    def __init__(self, nhid_dec, nproj, natt, ndecode):
        super().__init__(nhid_dec, nproj, natt, ndecode)
        self.nnlm1 = None
        self.nnlm2 = None
        self.penalty = 0
        self.ngram = None
        charlist = {}
        i = 0
        with open("/mnt/lustre02/jiangsu/aispeech/home/hs418/LM-Adaptation/transform_e2e/dict.txt", "r") as f:
            for line in f:
                charlist[str(i)] = line.split()[0]
                i += 1
        self.chardict =  charlist

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
            #inp = x if i == 0 else self.dropout(h[i - 1])
            if i == 0:
                inp = x
            else:
                if self.training:
                    inp = self.dropout(h[i - 1])
                else:
                    inp = h[i - 1]
            h[i], c[i] = self.rnn[i](inp, (state['h'][i], state['c'][i]))
        state = {'h': h, 'c': c}
        return state

    def load_nnlm1(self, path):
        logger.warning(f'Build language model from {path}')
        ckpt = torch.load(path, map_location=device)
        model = build_model(ckpt['hparams'])
        model.load_state_dict(ckpt['model'])
        self.nnlm1 = model.to(device)

    def load_nnlm2(self, path):
        logger.warning(f'Build language model from {path}')
        ckpt = torch.load(path, map_location=device)
        model = build_model(ckpt['hparams'])
        model.load_state_dict(ckpt['model'])
        self.nnlm2 = model.to(device)


    def set_penalty(self, penalty):
        self.penalty = penalty

    def set_T(self, T):
        self.T = T

    def set_penalty1(self, penalty):
        self.penalty1 = penalty

    def set_penalty2(self, penalty):
        self.penalty2 = penalty

    def forward_one_step(self, state, tgt, enc, en_mask):
        if state is None:
            state = {'e2e': None, 'nnlm2': None, 'ngram': None, 'interlm': None, 'ngram1': None, 'ngram2': None}
        
        
        am_out, am_state = super().forward_one_step(state['e2e'], tgt, enc, en_mask)
        
        inter_lm_out, inter_lm_state = super().forward_one_step(state['interlm'], tgt, enc, en_mask, True)


        nnlm2_state = state['nnlm2']
        nnlm2_state, nnlm2_out = self.nnlm2.forward_one_step(nnlm2_state, tgt)
        nnlm2_out = nnlm2_out.log_softmax(dim=-1)        


        LLR = nnlm2_out - inter_lm_out
        mask = LLR.lt(self.T)
        DR = nnlm2_out - inter_lm_out
        #DR.masked_fill_(mask, 0)
        out = am_out + DR * self.penalty
        out = am_out
        out[:, 0] = am_out[:, 0] 
        return out, {'e2e': am_state, 'nnlm2': nnlm2_state, 'interlm': inter_lm_state}

    def update_state(self, state, vidx, B, beam ):
        am_state = super().update_state(state['e2e'],vidx,B,beam)
        
        inter_lm_state = super().update_state1(state['interlm'],vidx,B,beam)

        # external LM
        nnlm2_state = state['nnlm2']
        
        if vidx is None:
            for j in range(self.nlayers):
                ## before nnlm1_state['h'][j].shape = [1, 1024]
                nnlm2_state['h'][j] = nnlm2_state['h'][j].view(B, 1, -1).repeat(1, beam, 1).view(B * beam, -1)
                ## after nnlm1_stae['h'][j].shape = [8, 1024]
        else:
            for j in range(self.nlayers):
                nnlm2_state['h'][j] = nnlm2_state['h'][j][vidx, :]
        return {'e2e': am_state, 'nnlm2': nnlm2_state, 'interlm': inter_lm_state}
