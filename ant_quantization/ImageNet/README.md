# ANT quantization for image classification.
## ImageNet data

Please prepare your dataset with [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) and set your dataset path using "--dataset_path /your/imagenet_path".

You can also fill your ImageNet path into the default setting in [line 28](https://github.com/clevercool/ANT_Micro22/blob/main/ant_quantization/ImageNet/main.py#L28) of the `main.py`.

## Evaluation 

### Results of 6-bit quantization without fine-tuning (Table V).
---

```shell
./scripts/quant_6bit_ptq.sh         # About 3 minutes
```
The accuracy results under our configuration are listed in the following table. 

| Model | ANT  | Model | ANT | 
| :----:| :----: | :----: | :----: | 
| AlexNet | **55.850%** | VGG16 | **72.798%** | 
| ResNet50 | **75.082%** | ResNet152 | **77.302%** |

Results of 6-bit quantization can be reproduced with slight random error. 

There are relatively large errors and unacceptable accuracy losses (10%-50%) in the results of 4-bit quantization without fine-tuning.

### Results of 4-bit quantization with fine-tuning (Figure 12).
---
We fine-tune the CV models with two types of server configuration.
- The server equipped with a single `NVIDIA A100 (40GB)` GPU is for models:
    - VGG-16;
    - Inception-V3.
- The server equipped with four `NVIDIA A10 (24GB)` GPUs is for the models:
    - ResNet-18;
    - ResNet-50 (is distributed on 4 GPUs);
    - ViT (is distributed on 4 GPUs.).

Note that you can reconfigure the batch size to reduce the memory requirement to run on a server with less memory, but this will impact the model accuracy results due to different batch sizes.
To conduct a fair comparison, we set Mode IP, IP-F, FIP, and FIP-F with the same settings.
For ANT4-8, the log file will print the data type chosen result, and we can analyze it for type ratio.

You can exploit the following scripts to fine-tune all models. We provide the approximate execution time for each script.


You can find the log files in the directory `./log`.

```shell
./scripts/vgg16_qat.sh          # About 20 hours
./scripts/resnet18_qat.sh       # About 4  hours
./scripts/resnet50_qat.sh       # About 9  hours
./scripts/inceptionv3_qat.sh    # About 17 hours
./scripts/vit_qat.sh            # About 12 hours
```
Notice that the complete fine-tuning process will take dozens of hours for all five models. 

We highly recommend the **fast evaluation** with checkpoints.
## Fast Evaluation

We have collected the checkpoints for the models. You can download them by running [this script](./scripts/download_checkpoint.sh). 

```shell 
# options: resnet18, resnet50, inceptionv3, vgg16, vit
# download for resnet18
./scripts/download_checkpoint.sh resnet18
# download for all models
./scripts/download_checkpoint.sh all

# resnet18      - 0.54 GB
# resnet50      - 1.18 GB
# inceptionv3   - 1.10 GB
# vgg16         - 6.18 GB
# vit           - 3.97 GB
```
You can run the following scripts to reproduce the results with fine-tuning (Figure 12) for CNNs (ResNet18, ResNet50, VGG16, InceptionV3) and ViT. The result may have a little random error (< 0.1%) due to the CUDA rounding implementation.

If it occurs the error "RuntimeError: CUDA out of memory.", you can reduce the batch size.

```shell
# run
./scripts/eval_resnet18.sh      # About 7 minutes
./scripts/eval_resnet50.sh      # About 15 minutes
./scripts/eval_inceptionv3.sh   # About 20 minutes
./scripts/eval_vgg16.sh         # About 18 minutes
./scripts/eval_vit.sh           # About 22 minutes

```

The accuracy results are listed in the following table. You can use `./script/print_result.sh` to get this table.

```shell
./script/print_result.sh
```

| Network | Int  | IP | FIP | IP-F | FIP-F | ANT4-8 |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: |
| VGG16 | 68.57% | 70.06% | 71.08% | 71.50% | 71.56% | 73.52% |
| ResNet18 | 66.28% | 66.35% | 67.28% | 67.86% | 67.85% | 69.64% |
| ResNet50 | 73.04% | 73.17% | 73.97% | 74.83% | 74.91% | 75.95% |
| InceptionV3 | 72.05% | 73.24% | 73.74% | 74.48% | 74.41% | 77.19% |
| ViT | 72.19% | 77.93% | 77.93% | 78.33% | 78.33% | 80.02% |

You can fill it in `../result/ANT-quantization.xlsx` to produce Figure 12 in the paper.