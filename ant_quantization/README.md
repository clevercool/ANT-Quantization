# ANT Quantization
We evaluate the results with models in image classification and NLP. 
## Paper's Hardware Configuration

+ Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
+ NVIDIA A100 GPU (40GB)
+ 4 * NVIDIA A10 GPUs (24GB)

## Environment
```
# PyTorch 1.11
conda create -n ant_quant python=3.8 
conda activate ant_quant
conda install  pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# Quantization CUDA kernel
pip install ./quant

#ImageNet
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

#BERT
pip install -r BERT/requirements.txt
```


## ImageNet
The image classification tasks include five models, i.e., VGG16, ResNet18, ResNet50, Inception-V3, and ViT. 

For reproducing the results in Table V and Figure 12, please refer to `./ImageNet`.

## BERT
We adopt the BERT model for the NLP task with three datasets, MNLI, CoLA, and SST-2. 

For reproducing the results in Figure 12, please refer to `./BERT`.

## Results
The file `./result/ANT-quantization.xlsx` contains the results and template for Figure 12.