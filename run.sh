target_domain_text=
target_domain=
stage=-1
stop_stage=2
pp=0.8
suffix=
target_nnlm=
T=0
expname=
nj=32
conf=conf/grulm.yaml
dir=data/test
exp=exp/trans20L-lstm2L-MWER-6983
lang=data/chn_3K_200319/lang_decode_e2e_6983

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

target_nnlm=exp/grulm2L_target_domain_${target_domain}/checkpoint
expname=${target_domain}_${pp}

testset="${target_domain} aitrans "

target_nnlm=exp/grulm2L_target_domain_navi/checkpoint
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
   echo "Stage -1"
   echo "data preprocess, ${target_domain_text} to token_id"
   ./scripts/data_preprocess.sh ${target_domain_text} ${target_domain} 
fi
### target nnlm training
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0"
  echo "nnlm training, train_set=$PWD/data/target_domain_${target_domain}/tr/text"
  bash ./scripts/train.sh --nj 1 --expdir exp/grulm2L_target_domain_${target_domain} --conf ${conf} hparams.exp_name=grulm2L_target_domain_${target_domain} data.data_dir=data/target_domain_${target_domain} || exit 1;
fi
### decode testset
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1"
  for test in ${testset}; do
    echo "decoding ${test} ${nj}"
    ./scripts/decode_e2e_ngram.sh --dir ${exp} --data ${dir} --sets ${test} --suffix "_${expname}" --nj ${nj} --pp ${pp} --T ${T} --target_nnlm ${target_nnlm} --lang ${lang} || exit 1;
  done
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Show Result
  echo "Source CER"
  grep WER ${exp}/decode_${target_domain}/aitrans/cer
  echo "Target CER"
  grep WER ${exp}/decode_${target_domain}/${target_domain}/cer
fi
