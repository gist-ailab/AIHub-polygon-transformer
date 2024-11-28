#!/bin/bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6093
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4


########################## Evaluate Refcoco+ ##########################
user_dir=../../polyformer_module
bpe_dir=../../utils/BPE
selected_cols=0,5,6,2,4,3


model='polyformer_b'
num_bins=64
batch_size=16

dataset='aihub_manufact_test_1120'
# ckpt_path=../finetune/polyformer_b_aihub_manufact_checkpoints/100_5e-5_512/checkpoint_last.pt
# ckpt_path=../finetune/polyformer_b_aihub_manufact_80_uniq_checkpoints/100_5e-5_512/checkpoint_best.pt
ckpt_path=../finetune/polyformer_b_aihub_manufact_80_uniq_logs/checkpoint_best.pt
# dataset='refcocog'
# ckpt_path=../../weights/polyformer_b_refcocog.pt

# for split in 'refcocog_val' 'refcocog_test'
# for split in 'aihub_manufact_val' 'aihub_manufact_test'
for split in 'aihub_manufact_test'
do
# data=../../datasets/finetune/${dataset}/${split}.tsv
data=../../datasets/finetune/aihub_manufact_test_1121/aihub_manufact_test.tsv

result_path=../../results_${model}/${dataset}/
vis_dir=${result_path}/vis/${split}
# vis_dir=/media/sblee/e0289bbd-f18a-4b52-a657-9079fe07ec70/polyformer_manufact_vis/vis/${split}
result_dir=${result_path}/result/${split}
log_file=aihub_manufact_rec.txt

echo "Command: bash evaluate_polyformer_b_aihub_manufact.sh" > ${log_file}

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${ckpt_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=${batch_size} \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --num-bins=${num_bins} \
    --vis_dir=${vis_dir} \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" \
    --vis >> ${log_file} 2>&1
done