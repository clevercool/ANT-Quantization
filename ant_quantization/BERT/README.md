# ANT quantization for BERT

## GLUE data

Before running this artifact you must download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](./download_glue_data.py)
and unpack it to some directory `./glue`.
```shell
python download_glue_data.py    #  305 MB
```

## Models
You need to download the pre-trained models from
[here](https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/bert_model.tar.gz)
and extract them.
```shell
wget https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/bert_model.tar.gz
tar -zxvf ./bert_model.tar.gz

# bert_model    - 1.13 GB
```
## Results with fine-tuning (Figure 12).

You can run the following scripts to reproduce the results with fine-tuning (Figure 12) for BERT model with CoLA, SST-2 and MNLI.
All the benchmarks have the same setting for each model.
The result may have a little random error ($\pm$ 0.5%) due to the rounding CUDA kernel.
```shell
./scripts/eval_cola.sh      # It takes about 25 mins.
./scripts/eval_sst2.sh      # It takes about 2h.
./scripts/eval_mnli.sh      # It takes about 3 mins to pre-process, and 15h to fine-tune.
```