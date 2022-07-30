# ANT quantization for BERT

## GLUE data

Before running this artifact you must download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](./download_glue_data.py)
and unpack it to some directory `./glue`.
```
python download_glue_data.py
```

## Models
You need to download the pre-trained models from
[here](https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/bert_model.tar.gz)
and extract them.
```
wget https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/bert_model.tar.gz
tar -zxvf ./bert_model.tar.gz
```
## Results with fine-tuning (Figure 12).

You can run the following scripts to reproduce the results with fine-tuning (Figure 12) for BERT model with CoLA, SST-2 and MNLI.
All the benchmarks have the same setting for each model.
The result may have a little error ($\pm$ 0.5%). But the trend between different modes (Int, IP, and IP-F) is the same as the results in the paper.
```
./scripts/eval_cola.sh
./scripts/eval_sst2.sh
./scripts/eval_mnli.sh
```

