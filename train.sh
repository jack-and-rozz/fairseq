#!/bin/bash
usage() {
    echo "Usage:$0 target_dir mode"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

target_dir=$1
mode=$2

if [ ! -e $target_dir ];then
    mkdir -p $target_dir
    mkdir -p $target_dir/tests
    mkdir -p $target_dir/checkpoints
    mkdir -p $target_dir/tensorboard
fi

shared_settings="-layers 6 -rnn_size 512 -word_vec_size 512 \
	        -transformer_ff 2048 -heads 8  \
		-encoder_type transformer -decoder_type transformer \
		-position_encoding \
		-train_steps 200000 -max_generator_batches 2 -dropout 0.1 \
		-batch_size 4096 -batch_type tokens \
		-normalization tokens  -accum_count 2 \
		-optim adam -adam_beta2 0.998 \
		-decay_method noam -warmup_steps 8000 \
		-learning_rate 2 \
		-max_grad_norm 0 -param_init 0  -param_init_glorot \
		-label_smoothing 0.1 -valid_steps 10000 \
		-save_checkpoint_steps 10000 \
		-tensorboard \
		-tensorboard_log_dir $target_dir/runs
		-save_model $target_dir/checkpoints 
		-log_file $target_dir/train.log
		"

case $mode in
    aspec.baseline)
	data_dir=dataset/aspec-je/processed.kytea-moses
	data_options="$data_dir/fairseq.50k-50k \
       	              --encoder-embed-path $data_dir/word2vec.en.512d \
       	              --decoder-embed-path $data_dir/word2vec.ja.512d"
	task_options="--task translation \
		      --source-lang en \
		      --target-lang ja"
	;;
    jesc.baseline)
	data_dir=dataset/jesc-je/processed.kytea-moses
	data_options="$data_dir/fairseq.50k-50k \
       	              --encoder-embed-path $data_dir/word2vec.en.512d \
       	              --decoder-embed-path $data_dir/word2vec.ja.512d"
	task_options="--task translation \
		      --source-lang en \
		      --target-lang ja"
	;;

    * ) echo "invalid mode"
        exit 1 ;;
esac

# Start training.
python train.py \
       --ddp-backend=no_c10d \
       --log-interval 50 --log-format simple \
       --save-dir $target_dir/checkpoints \
       --tensorboard-logdir $target_dir/tensorboard \
       --arch transformer \
       $task_options \
       $data_options \
       --update-freq 2 \
       --num-workers 2 \
       --max-tokens 4096 \
       --max-update 200000 \
       --optimizer adam --adam-betas '(0.9, 0.98)' \
       --lr 1e-03 --min-lr 1e-09 \
       --lr-scheduler inverse_sqrt \
       --warmup-init-lr 1e-07 --warmup-updates 4000 \
       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
       --dropout 0.1 \
       --encoder-layers 6 --decoder-layers 6 \
       --encoder-attention-heads 8 --decoder-attention-heads 8 \
       --encoder-ffn-embed-dim 2048 \
       --decoder-ffn-embed-dim 2048 \
       --share-decoder-input-output-embed \
       >> $target_dir/train.log 
