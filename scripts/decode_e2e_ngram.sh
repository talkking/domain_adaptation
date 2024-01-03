#!/bin/bash

. ./path.sh
. cmd.sh

dir=
data=
beam=8
lang=data/chn_3K_200319/lang_decode_e2e_6979

nj=10
suffix=
sets=""
pp=0.6
T=0
target_nnlm=
beta=0.99

. utils/parse_options.sh

for setname in $sets; do
    test_set=${data}/${setname}
    sdata="${test_set}/split${nj}"
    [[ ! -d $sdata && ${data}/${setname}/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data/${setname} $nj || exit 1;
    echo $nj > $dir/num_jobs

    nnet_forward="python -u -m extend_codes.decode_e2e_ngram --dir $dir --testset $sdata/JOB --beam $beam --penalty $pp --T ${T} --beta ${beta} --target_nnlm ${target_nnlm} --stream-size 1"
    
    decode_dir=${dir}/decode$suffix/$( basename $setname )
    decode_e2e.sh --nnet_forward "${nnet_forward}" \
            --nj ${nj} \
            $test_set $lang $decode_dir &
done
wait
