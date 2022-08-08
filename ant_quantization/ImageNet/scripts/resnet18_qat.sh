mkdir -p log

# Run on the server with 4 NVIDIA A10 GPUs.

#Int
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --dataset=imagenet --model=resnet18 --epoch=2 --mode=int --wbit=4 --abit=4 --batch_size=256 --lr=0.00005 --train > ./log/resnet18_Int.log 2>&1

#IP
CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46667 main.py --dataset=imagenet --model=resnet18 --epoch=2 --mode=ant-int-pot --wbit=4 --abit=4 --batch_size=256 --lr=0.00005 --train > ./log/resnet18_IP.log 2>&1

#FIP
CUDA_VISIBLE_DEVICES=2 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46668 main.py --dataset=imagenet --model=resnet18 --epoch=2 --mode=ant-int-pot-float --wbit=4 --abit=4 --batch_size=256 --lr=0.00005 --train > ./log/resnet18_FIP.log 2>&1

#IP-F
CUDA_VISIBLE_DEVICES=3 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46669 main.py --dataset=imagenet --model=resnet18 --epoch=2 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=256 --lr=0.00005 --train > ./log/resnet18_IP-F.log 2>&1

#FIP-F
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46670 main.py --dataset=imagenet --model=resnet18 --epoch=2 --mode=ant-int-pot-float-flint --wbit=4 --abit=4 --batch_size=256 --lr=0.00005 --train > ./log/resnet18_FIP-F.log 2>&1

#ANT4-8
CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46671 main.py --dataset=imagenet --model=resnet18 --epoch=3 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=256 --lr=0.0005 --train -l8=0,20 > ./log/resnet18_ANT4-8.log 2>&1
