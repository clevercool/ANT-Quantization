# OliVe Quantization
## Environment
```bash
conda create -n OliVe python=3.8
conda activate OliVe

conda install pytorch=1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

pip install ./quant
```

## Paper's Hardware Configuration

+ AMD EPYC 7302 16-Core Processor
+ NVIDIA A40 GPU (48GB)

## BERT / BART

We adopt the BERT and BART models for the NLP task with five datasets, MNLI, CoLA, SST-2, QQP and MRPC.

For reproducing the results in the paper, please refer to `./bert`.

## Large Language Models

We adopt the GPT-2, OPT and Bloom models for the NLP task with two datasets, wikitext and C4.

For reproducing the results in the paper, please refer to `./llm`.