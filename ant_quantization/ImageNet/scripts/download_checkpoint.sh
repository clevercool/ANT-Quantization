#!/bin/bash
mkdir -p checkpoints
if [ "$1" = "inceptionv3" ];then
    echo "----------------- Download inceptionv3 -----------------"
    echo "----------------- Download inceptionv3 -----------------"
    echo "----------------- Download inceptionv3 -----------------"
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_ip_f.pth 
elif [ "$1" = "resnet18" ]; then 
    echo "----------------- Download resnet18 -----------------"
    echo "----------------- Download resnet18 -----------------"
    echo "----------------- Download resnet18 -----------------"
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ip_f.pth 
elif [ "$1" = "resnet50" ]; then 
    echo "----------------- Download resnet50 -----------------"
    echo "----------------- Download resnet50 -----------------"
    echo "----------------- Download resnet50 -----------------"
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_ip_f.pth 
elif [ "$1" = "vgg16" ]; then 
    echo "----------------- Download vgg16 -----------------"
    echo "----------------- Download vgg16 -----------------"
    echo "----------------- Download vgg16 -----------------"
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_ip_f.pth 
elif [ "$1" = "vit" ]; then 
    echo "----------------- Download vit -----------------"
    echo "----------------- Download vit -----------------"
    echo "----------------- Download vit -----------------"
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_ip_f.pth 
elif [ "$1" = "all" ]; then 
    echo "----------------- Download All -----------------"
    echo "----------------- Download All -----------------"
    echo "----------------- Download All -----------------"
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/inceptionv3_ip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet50_ip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vgg16_ip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/vit_ip_f.pth 
else
    echo "----------------- By default, only download resnet18 -----------------"
    echo "----------------- By default, only download resnet18 -----------------"
    echo "----------------- By default, only download resnet18 -----------------"
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ant48.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_fip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_fip_f.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_int.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ip.pth 
    wget -nc -P ./checkpoints/ https://github.com/clevercool/ANT_Micro22/releases/download/v0.1/resnet18_ip_f.pth 
fi