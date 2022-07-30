# ANT quantization for image classification.
## ImageNet data

Please prepare your dataset with [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) and set your dataset path using "--dataset_path /your/imagenet_path".

## Evaluation 

### Results of 6-bit quantization without fine-tuning (Talbe V).

```shell
./scripts/quant_6bit_ptq.sh
```
The accuracy results under our configuration are listed in the following table. 

| Model | ANT  | Model | ANT | 
| :----:| :----: | :----: | :----: | 
| AlexNet | **55.850%** | VGG16 | **72.798%** | 
| ResNet50 | **75.082%** | ResNet152 | **77.302%** |

Results of 6-bit quantization can be reproduced with slight random error. 

There are relatively large errors and unacceptable accuracy losses (10%-50%) in the results of 4-bit quantization without fine-tuning.

### Results of 4-bit quantization with fine-tuning (Figure 12).
Recommend executing the following script on `NVIDIA A100`. Some subcases require 40GB GPU memory in training. It takes 1-10h in `NVIDIA A100`.
Mode IP, IP-F, FIP, and FIP-F have the same settings.
For ANT4-8, the log file will print the data type chosen result, and we can analyze it for type ratio.

```shell
./scripts/vgg16_qat.sh
./scripts/resnet18_qat.sh
./scripts/resnet50_qat.sh
./scripts/inceptionv3_qat.sh
./scripts/vit_qat.sh
```
Notice that the complete fine-tuning process will take dozen hours for all five models. 
We recommend the **fast evaluation** with checkpoints.
## Fast Evaluation

We have collected the checkpoints for the models. You can download them by running [this script](./scripts/download_checkpoint.sh). 

```shell 
# options: resnet18, resnet50, inceptionv3, vgg16, vit
# download for resnet18
./scripts/download_checkpoint.sh resnet18
# download for all models
./scripts/download_checkpoint.sh all
```
You can run the following scripts to reproduce the results with fine-tuning (Figure 12) for CNNs (ResNet18, ResNet50, VGG16, InceptionV3) and ViT. The result may have a little random error (< 0.1%) due to the CUDA rounding implementation.
```shell
# run
./scripts/eval_resnet18.sh
./scripts/eval_resnet50.sh
./scripts/eval_inceptionv3.sh
./scripts/eval_vgg16.sh
./scripts/eval_vit.sh

```

The accuracy results are listed in the following table. 
| Network | Int  | IP | FIP | IP-F | FIP-F | ANT4-8 |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: |
| VGG16 | 68.57% | 70.06% | 71.08% | 71.50% | 71.56% | 73.52% |
| ResNet18 | 66.28% | 66.35% | 67.28% | 67.86% | 67.85% | 69.64% |
| ResNet50 | 73.04% | 73.17% | 73.97% | 74.83% | 74.91% | 75.95% |
| InceptionV3 | 72.05% | 73.24% | 73.74% | 74.48% | 74.41% | 77.19% |
| ViT | 72.19% | 77.93% | 77.93% | 78.33% | 78.33% | 80.02% |

Then, you can fill it in `../result/ANT-quantization.xlsx` to produce Figure 12 in the paper.