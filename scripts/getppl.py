from extend_codes.lstm_decoder_lm import LSTMDecoderNGRAM
from asr.utils.dynload import build_model
import torch
import torch.nn.functional as F
import math
import numpy as np
from asr.utils.checkpoint import Checkpoint
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

class CaculatePPL:
    def __init__(self, model, text, quan=False, nn_type="lstm", device="cuda"):
        self.path = model
        self.file = text
        self.quan = quan
        self.nn_type = nn_type       
        self.device = device
 
    def load_nnlm(self, path):
        ckpt = torch.load(path, map_location=self.device)
        model = build_model(ckpt['hparams'])
        model.load_state_dict(ckpt['model'])
        if self.quan:
           print("model quantization")
           model.cpu()
           from torch.quantization.qconfig import QConfigDynamic,default_dynamic_quant_observer, default_weight_observer, default_per_channel_weight_observer
           default_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer, weight=default_weight_observer)
           default_dynamic_qconfig_channel = QConfigDynamic(activation=default_dynamic_quant_observer, weight=default_per_channel_weight_observer)
           if self.nn_type == "lstm":
              NNtype = torch.nn.LSTMCell
           else:
              NNtype = torch.nn.GRUCell
           quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear: default_dynamic_qconfig_channel, NNtype: default_dynamic_qconfig}, dtype=torch.qint8)
           self.nnlm = quantized_model
        else:
           self.nnlm = model.to(self.device)

    def load_model(self, path):
        model = Checkpoint.load_model_from_checkpoint(path)
        model.eval()
        self.interlm = model.decoder

    def caculateppl(self):
        self.load_nnlm(self.path)
        #self.load_model(self.path)
        total_nll = 0
        total_ntoken = 0
        total_time = 0
        total_step = 0
        embedding_time = 0
        rnn_time = 0
        with torch.no_grad():
            with open(self.file, "r") as f: 
                for line in f:
                    name = line.split()[0]
                    textid = line.split()[1:]
                    beam = 8
                    tgt = torch.zeros(beam, device=self.device).long()
                    state = None
                    utt_nll = 0
                    utt_ntoken = 0
                    for i in range(len(textid)):
                        #nnlm_out, state = self.interlm.forward_one_step(state, tgt, None, None, True)
                        t1 = time.time()
                        state, nnlm_out, sta = self.nnlm.forward_one_step(state, tgt)
                        t2 = time.time()
                        total_time += t2 - t1
                        total_step += 1
                        embedding_time += sta['embedding_time']
                        rnn_time += sta['rnn_time']
                        tgt = torch.tensor(data=[int(textid[i])], device=self.device).long().repeat(beam,)
                        
                        nll = F.cross_entropy(nnlm_out, tgt, reduction="none")
                        # nll.shape = [B]
                        
                        # nll = nnlm_out.log_softmax(dim=-1)
                        #print(nnlm_out.shape)
                        # nll, _ = torch.max(nll, dim=-1)
                        utt_nll += nll.sum(-1)
                        utt_ntoken += nll.size(-1)
                    total_nll += utt_nll
                    total_ntoken += utt_ntoken
                    #utt_ppl = np.exp(utt_nll / utt_ntoken)
                    #print(name, utt_ppl)
                total_ppl = math.exp(total_nll / total_ntoken)
                print("total_ppl=", total_ppl)
                print(f"total_time={total_time}")
                print("total_step", total_step)
                print("avg time:=", total_time / total_step)
                print("embedding_time=", embedding_time / total_step)
                print("rnn_time=", rnn_time / total_step)

import sys
from distutils.util import strtobool

def main():
    model_path = "/mnt/lustre02/jiangsu/aispeech/home/hs418/transform_e2e/exp/e2e_newtool/exp/lstmlm2L-adapt_LSLoss/checkpoint"
    #model_path = "/mnt/lustre02/jiangsu/aispeech/home/hs418/transform_e2e/exp/e2e_newtool/exp/lstmlm2L-adapt-aviage_LSLoss/checkpoint"
    model_path = "/mnt/lustre02/jiangsu/aispeech/home/hs418/LM-Adaptation/transform_e2e/exp/LSTMLM_LSLoss/checkpoint"
    #model_path ="/mnt/lustre02/jiangsu/aispeech/home/hs418/LM-Adaptation/transform_e2e/exp/trans20L-lstm2L_LMadapt_ILME/checkpoint"
    model_path = "exp/lstmlm2L_tv/checkpoint"
    text_path = sys.argv[1]
    model_path = sys.argv[2]
    quan = strtobool(sys.argv[3]) #True#bool(sys.argv[3])
    nn_type = sys.argv[4]
    device = sys.argv[5]
    CPPL = CaculatePPL(model_path, text_path, quan, nn_type, device)
    CPPL.caculateppl()


if __name__ == "__main__":
    main()
