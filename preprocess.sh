#!/bin/bash
usage() {
    echo "Usage:$0 dataset_name [src_vocab_size] [tgt_vocab_size]"
    exit 1
}
if [ $# -lt 1 ];then
    usage;
fi



dataset=$1
src_vocab_size=$2
tgt_vocab_size=$3
if [ -z $src_vocab_size ];then
  src_vocab_size=50000
fi
if [ -z $tgt_vocab_size ];then
  tgt_vocab_size=50000
fi
suffix=$(($src_vocab_size/1000))k-$(($tgt_vocab_size/1000))k
data_dir=dataset/$dataset/processed.kytea-moses

if [ ! -e $data_dir/fairseq.$suffix ]; then
    mkdir -p $data_dir/fairseq.$suffix
fi


case $dataset in
    aspec-je)
	;;
    jesc-je)
	;;
    * ) echo "invalid name"
        exit 1 ;;
esac

python preprocess.py \
       --source-lang en --target-lang ja \
       --trainpref $data_dir/train \
       --validpref $data_dir/dev \
       --testpref $data_dir/test \
       --destdir $data_dir/fairseq.$suffix \
       --nwordssrc $src_vocab_size \
       --nwordstgt $tgt_vocab_size \
       --workers 4

