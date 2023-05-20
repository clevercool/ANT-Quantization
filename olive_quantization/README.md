# OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization [[paper](https://arxiv.org/abs/2304.07493)]

![](figures/intro_victor.png)

## Abstract

Transformer-based large language models (LLMs) have achieved great success with the growing model size. LLMs’ size grows by 240× every two years, which outpaces the hardware progress and makes model inference increasingly costly. Model quantization is a promising approach to mitigate the widening gap between LLM size and hardware capacity. However, the existence of outliers, values with significant magnitudes, in LLMs makes existing quantization methods less effective. Prior outlier-aware quantization schemes adopt sparsity encoding techniques to separate outliers from nor- mal values where the process requires global coordination (e.g., a global sparsity coordination list). This incurs complex encod- ing/decoding hardware logics and an extra orchestration controller for the computation between outlier and normal values. As such, it is not hardware-efficient and hence only achieves sub-optimal quantization benefits.

We propose OliVe, an algorithm/architecture co-designed so- lution that adopts an outlier-victim pair (OVP) quantization and handles outlier values locally with low hardware overheads and high performance gains. The key insight of OliVe is that outliers are important while the normal values next to them are not. Thus those normal values (called victims) can be sacrificed to accommodate outliers. This enables a memory-aligned OVP encoding scheme, which can be efficiently integrated to the existing hardware accel- erators like systolic array and tensor core. As a result, OliVe-based accelerator surpasses the existing outlier-aware accelerator, GOBO, by 4.5× speedup and 4.0× energy reduction, respectively, with a superior model accuracy.

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

## Usage
### BERT / BART

We adopt the BERT and BART models for the NLP task with five datasets, MNLI, CoLA, SST-2, QQP and MRPC.

For reproducing the results in the paper, please refer to `./bert`.

### Large Language Models

We adopt the GPT-2, OPT and Bloom models for the NLP task with two datasets, wikitext and C4.

For reproducing the results in the paper, please refer to `./llm`.