text=$1
setname=$2
nj=1


. ./path.sh


if [ $# -lt 2 ]; then
  echo "usage: target_domain_text setname!"
  exit 1;
fi
if [ -d data/target_domain_${setname} ]; then
  rm -rf data/target_domain_${setname}/tr
  rm -rf data/target_domain_${setname}/cv
else
  mkdir data/target_domain_${setname}
fi
mkdir data/target_domain_${setname}/tr

## split Chinese char
#awk '{printf("%s", $1); for(j=2;j<=NF;j++){num=split($j,ss,"");for(i=1;i<=num;i++){c=ss[i];if(c~/[\000-\177]/){s=s""c; if(i==num){printf(" %s",s);s="";}} else if(c!~/[\000-\177]/){if(s!=""){printf(" %s",s);s="";}printf(" %s",c);}} } printf("\n");}' tmp > tmp1



sed -e 's/ //g' ${text} > tmp1
#python scripts/add_space.py tmp1 tmp
#awk '{print NR " " $0}' tmp > tmp1
awk '{printf("%s ", NR); for(j=1;j<=NF;j++){num=split($j,ss,"");for(i=1;i<=num;i++){c=ss[i];if(c~/[\000-\177]/){
s=s""c; if(i==num){printf(" %s",s);s="";}} else if(c!~/[\000-\177]/){if(s!=""){printf(" %s",s);s="";}printf(" %c%c%c",c,ss[i+1],ss[i+2]);i+=2;}} } printf("\n");}' tmp1 > tmp
mv tmp tmp1

## map English word into bpe token
lexicon=data/dictionary/english_lexicon.txt 

awk '{print(toupper($0))}' tmp1 | awk 'NR==FNR{name=$1;$1="";a[name]=$0}NR>FNR{for(i=1;i<=NF;i++) if($i in a) $i=a[$i];print($0);}' ${lexicon} - > token

## map all token into token_id
dict=data/chn_3K_200319/lang_decode_e2e_6983/units.txt
#data/dictionary/dict.txt

awk 'NR==FNR{a[$1]=$2}NR>FNR{for(i=2;i<=NF;i++) if($i in a) $i=a[$i]; else $i=""; print($0)}' ${dict} token > textid


## repeat text 100w scale
num_line=`cat textid | wc -l`
if [ $num_line -lt 1000000 ]; then
  repeat_nums=$(((1000000-1)/$num_line+1))
  while [ $repeat_nums -gt 10 ]; do
    bash scripts/repeat_text.sh textid textid 10
    repeat_nums=$(($(($repeat_nums-1))/10+1))
  done
  bash scripts/repeat_text.sh textid textid $repeat_nums
fi

#awk -v n=$3 '{for (i=1; i<=n; ++i) print $0}' textid > tmp


## renumber
awk '{$1=NR; print $0}' textid > tmp

#paste -d" " line tmp > tmp1
#mv tmp1 textid
mv tmp textid

## add eos/sos
cut -f 1 -d" " textid > line
cut -f 2- -d" " textid > content

awk '{print "0 " $0 " 0"}' content > tmp2
paste -d" " line tmp2 > text

mv text data/target_domain_${setname}/tr

## fake feats.scp

awk '{print $1 " /mnt/lustre02/jiangsu/aispeech/home/hs418/LM-Adaptation/transform_e2e/target_domain_tv/test/data/raw_fbank_test.1.ark:2"}' line > data/target_domain_${setname}/tr/feats.scp

## utt2spk
awk '{print $1 " " $1}' data/target_domain_${setname}/tr/text > data/target_domain_${setname}/tr/utt2spk

## split job
#nj=1
./utils/split_data.sh --per-utt data/target_domain_${setname}/tr ${nj}
mv data/target_domain_${setname}/tr/split${nj}/${nj} data/target_domain_${setname}/tr/split${nj}/0

## tr and cv are same
cp -rf data/target_domain_${setname}/tr data/target_domain_${setname}/cv

rm line tmp1 tmp2 textid token content 
