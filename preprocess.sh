#!/bin/bash
usage() {
    echo "Usage:$0 dataset_name [aspec-je|jesc-je]"
    exit 1
}
if [ $# -lt 1 ];then
    usage;
fi



dataset=$1

data_dir=dataset/$dataset/processed.kytea-moses
if [ ! -e $data_dir/fairseq ]; then
    mkdir -p $data_dir/fairseq
fi

case $dataset in
    aspec-je)
	fairseq-preprocess --source-lang en --target-lang ja \
			   --trainpref $data_dir/train \
			   --validpref $data_dir/dev \
			   --testpref $data_dir/test \
			   --destdir $data_dir/fairseq
 
	;;
    * ) echo "invalid name"
        exit 1 ;;
esac


