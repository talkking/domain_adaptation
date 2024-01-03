import logging

from asr.model.transformer import Transformer
from asr.model import add_model
logger = logging.getLogger(__name__)

@add_model('TransNGRAM')
class TransNGRAM(Transformer):
    def __init__( self, ninp, nhid=2048, nproj=512, nctc=2966, natt=6979,
                  nlayer=12, nhead=8, nhid_dec=1024, ndecode=2, max_norm=1000, activation='relu6', dec='lstm', dropout=0.1, pos_emb=True, mode='mt', zerocontext=False):
        super().__init__(ninp, nhid, nproj, nctc, natt, nlayer, nhead, nhid_dec, ndecode, max_norm, activation, dec, dropout, pos_emb, mode, zerocontext=zerocontext)
        from .lstm_decoder_lm import LSTMDecoderNGRAM
        self.decoder = LSTMDecoderNGRAM(nhid_dec, nproj, natt, ndecode)

    def load_nnlm1(self, path):
        self.decoder.load_nnlm1(path)
    def load_nnlm2(self, path):
        self.decoder.load_nnlm2(path)
    def load_ngram(self, path):
        self.decoder.load_ngram(path)

    def load_ngram1(self, path):
        self.decoder.load_ngram1(path)
    def load_ngram2(self, path):
        self.decoder.load_ngram2(path)

    def set_T(self, T):
        self.T = T
    
