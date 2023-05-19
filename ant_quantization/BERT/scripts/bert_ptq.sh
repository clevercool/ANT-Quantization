#!/bin/bash

task_name=${1:-"cola"}
size=${2:-"base"}
gpu_num=${3:-0}
mode=${5:-"ant-int-flint"}

if [ "$size" == "base" ] ; then
  path="ModelTC/bert-base-uncased-$task_name "
  batch_size=${4:-"128"}
fi

if [ "$size" == "large" ] ; then
  path="yoshitomo-matsubara/bert-large-uncased-$task_name "
  batch_size=${4:-"64"}
fi

mkdir -p ./log/bert_${size}_ptq/$task_name

export CUDA_VISIBLE_DEVICES=$gpu_num
python run_glue.py \
  --do_eval \
  --model_name_or_path $path \
  --task_name $task_name \
  --max_length 128 \
  --quantize_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --output_dir ./log/bert_${size}_ptq/$task_name/ \
  --mode $mode \
  --abit 4 \
  --wbit 4 \
  -wu 250 \
  -wl 75  \
  -au 250 \
  -al 75 
  # > ./log/bert_${size}_ptq/$task_name/${batch_size}.out