transformer_model=${1:-"gpt2"}
dataset=${2:-"wikitext"}
dataset_config=${3:-"wikitext-103-raw-v1"}
q_mode=${4:-"ant-int-flint"}
q_bit=${5:-"4"}
batch_size=${6:-"8"}
port=${7:-46666}
desc=${8:-""}
n8=${9:-"0"}

mkdir -p ./log
mkdir -p ./log/bigscience
mkdir -p ./log/facebook

log_name=""
if [ "$dataset" = "wikitext" ] ; then
  log_name=$transformer_model"_"$dataset_config"_"$q_bit"bit_batch"$batch_size"_"$desc
else
  log_name=$transformer_model"_"$dataset"_"$q_bit"bit_batch"$batch_size"_"$desc
fi

python -u -m torch.distributed.launch --nproc_per_node=1 --master_port $port run_clm.py \
  --model_name_or_path $transformer_model \
  --dataset_name $dataset --dataset_config_name $dataset_config \
  --output_dir checkpoints/$transformer_model \
  --do_eval \
  --mode=$q_mode --wbit=$q_bit --abit=$q_bit --a_low=75 --a_up=250 --w_low=75 --w_up=250 --layer_8bit_n=$n8 \
  --eval_batch_size=$batch_size --train_batch_size=$batch_size --quantize_batch_size=$batch_size \
  2>&1 | tee ./log/${log_name}.log \