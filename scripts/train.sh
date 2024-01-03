#!/bin/bash
. ./path.sh
. cmd.sh

nj=16
expdir=
conf=

. utils/parse_options.sh

if [ -z $expdir ]; then
    echo 'must set expdir'
    exit 1;
fi

if [ -z $conf ]; then
    echo 'must set config'
    exit 1
fi


$cuda_cmd JOB=1:$nj ${expdir}/log/train.JOB.log python -u -m asr.launch \
    -c $conf \
    $@  || exit 1
