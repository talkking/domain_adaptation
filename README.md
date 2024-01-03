> form hang.shao_sx

本例子是领域自适应，只用传入目标领域text，以及目标领域或源域测试集，就可以在该测试集上自适应

##数据准备
将目标域文本放在data/target_domain_text目录下用于nnlm训练
将目标域测试集数据集放在data/test下
如果目标域文本数量小于100w条就将它repeat到100w级别
feats.scp为伪造feats
对应的脚本是scripts/data_process.sh

##nnlm训练
目标领域nnlm training
```
bash ./scripts/train.sh --expdir exp/grulm2L_target_domain_aviage2L128h --conf conf/grulm2L128h.yaml

```
##解码测试集
./scripts/decode_e2e_ngram.sh --dir ${exp} --data ${dir} --sets ${dataset} --suffix "_${expname}" --nj ${job} --pp ${pp} --target_nnlm exp/lstmlm2L_target_domain_tv/checkpoint

dir为存放模型checkpoint的实验目录，data为数据目录，sets为数据目录里面具体的子集如test，target_nnlm为目标域lm的路径


##experiment result
1.tts合成的target_domain_tv:
| system   | source(aitrans) |  tv  | 
| baseline |  8.92           | 7.89 |
| ours     |  9.33           | 0.41 |
2.aviage航空领域:
| system   | source(aitrans) | aviage |
| baseline | 8.92            | 57.58  |
| ours     | 9.71            | 28.73  |
| ours(lambda0.6) |  9.29    | 33.61  |
baseline为直接在source model下的解码
可以看到目标域的性能改善非常明显，但是源域几乎不降(rel 5%以内)

##Usage Example
准备数据：
bash run.sh --stage -1 --target_domain_text target_domain_textdir/tv_text --stop_stage -1 --setname tv

目标域nnlm训练：
bash run.sh --stage 0 --target_domain_text target_domain_textdir/tv_text --stop_stage 0 --setname tv 

解码测试集：
decode source domain when target domain is tv
'''
bash run.sh --setname aitrans --stage 1 --target_domain tv
'''
decode target domain tv
'''
bash run.sh --setname tv --stage 1 --target_domain tv
'''

总的流程：
bash run.sh --target_domain_text target_domain_textdir/aviage_text --target_domain aviage 

target_domain指定当前目标域，target_domain_text指定目标域文本


Version Update: 2022-11-30
更新了更轻量级的语言模型gru2L128，经过压缩量化后在CPU的推理速度和之前的lstm2L1024在GPU上的速度差不多


