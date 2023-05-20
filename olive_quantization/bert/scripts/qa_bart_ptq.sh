#!/bin/bash

task_name=${1:-"squad"}
size=${2:-"base"}
gpu_num=${3:-0}
batch_size=${4:-"64"}
mode=${5:-"ant-int-flint"}
dataset_name=squad
if [ "$size" == "base" ] ; then
  path="ModelTC/bart-base-$task_name "
fi

if [ "$task_name" == "squad2" ] ; then
  dataset_name="squad_v2 --version_2_with_negative"
fi


mkdir -p ./log/bart_${size}_ptq/$task_name

export CUDA_VISIBLE_DEVICES=$gpu_num

python run_qa.py \
  --do_eval \
  --model_name_or_path $path \
  --max_seq_length 384 \
  --dataset_name $dataset_name \
  --quantize_batch_size $batch_size \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --output_dir ./log/bart_${size}_ptq/$task_name/ \
  --mode $mode \
  --abit 4 \
  --wbit 4 \
  -wu 250 \
  -wl 75  \
  -au 250 \
  -al 75 
  # > ./log/bart_${size}_ptq/$task_name/${batch_size}.out