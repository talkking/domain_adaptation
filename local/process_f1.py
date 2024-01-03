from sklearn.metrics import f1_score
import pdb

import sys

ref = sys.argv[1]
hyp = sys.argv[2]
ref_file = open(ref,'r')
hyp_file = open(hyp,'r')
dic = {}
#punc_list = "。 ！ ， ？ 、".split()
punc_list = "？ ， 。 ！ 、".split()
punc_list = {punc:i+1 for i,punc in enumerate(punc_list)}

def textseq_to_punc_lab(text_list):
    ref_id = []
    for i, token in enumerate(text_list):
        if i < len(text_list)-1:
            if text_list[i+1] in punc_list:
                #ref_id.append(punc_list[text_list[i+1]]) 
                ref_id.append(1) 
            else:
                if text_list[i] in punc_list:
                    continue
                else:
                    ref_id.append(0)
        else:
            if text_list[i] not in punc_list:
                ref_id.append(0)
    return ref_id

for ref in ref_file:
    ref = ref.rstrip()
    ref = ref.split()
    ref_id = textseq_to_punc_lab(ref[1:])
    dic[ref[0]] = ref_id

refs = []
hyps = []
for hyp in hyp_file:
    hyp = hyp.rstrip()
    hyp = hyp.split()

    if hyp[0] not in dic:
        continue
    ref_seq = dic[hyp[0]]
    hyp_seq = textseq_to_punc_lab(hyp[1:])
    a = len(ref_seq)
    b = len(hyp_seq)
    if a != b :
        continue
    refs += ref_seq
    hyps += hyp_seq

print(f1_score(refs, hyps, average='binary'))

ref_file.close()
hyp_file.close()

