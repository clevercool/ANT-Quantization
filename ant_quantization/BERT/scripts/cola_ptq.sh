#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

task_name="CoLA"

q_mode=${1:-"int"}
n8=${2:-"0"}
gpu_num=${3:-0}
port=${4:-46666}
q_bit=${5:-"4"}
epochs=${6:-"3"}
precision=${7:-"fp32"}
batch_size=${8:-"64"}
learning_rate=${9:-"2e-5"}
seed=${10:-2}
vocab_file=${11:-"./vocab.txt"}
CONFIG_FILE=${12:-"./bert_config.json"}
mode=${13:-"train eval"}
init_checkpoint=${14:-"./model/${task_name}/pytorch_model.bin"}
max_steps=${15:-"-1.0"} # if < 0, has no effect
warmup_proportion=${16:-"0.01"}

TASK_DIR=./glue/$task_name
OUT_DIR=./results/${task_name}_${q_mode}_8bit_layer_${n8}

mkdir -p $OUT_DIR

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16 "
fi

export CUDA_VISIBLE_DEVICES=$gpu_num
mpi_command=" -m torch.distributed.launch --nproc_per_node=1 --master_port $port"

CMD="python -u $mpi_command run_glue.py "
CMD+="--task_name $task_name "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [ "$mode" == "eval" ] ; then
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi
if [ "$mode" == "train eval" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--do_lower_case "
CMD+="--data_dir $TASK_DIR "
CMD+="--bert_model bert-base-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$CONFIG_FILE "
CMD+="--output_dir $OUT_DIR "
CMD+="$use_fp16"


CMD+="--mode=$q_mode "
CMD+="--wbit=$q_bit "
CMD+="--abit=$q_bit "
CMD+="-n8=$n8 "
CMD+="-wu=150 "
CMD+="-wl=80 "
CMD+="-au=150 "
CMD+="-al=80 "
# CMD+="--disable_input_quantization "

LOGFILE=$OUT_DIR/logfile

echo $CMD
$CMD |& tee $LOGFILE

