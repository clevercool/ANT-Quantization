mkdir -p log

# Run on the server with 4 NVIDIA A10 GPUs.

#Int
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46666 main.py --dataset=imagenet --model=resnet50 --epoch=2 --mode=int --wbit=4 --abit=4 --batch_size=64 --lr=0.00005 --train > ./log/resnet50_Int.log 2>&1

#IP
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46667 main.py --dataset=imagenet --model=resnet50 --epoch=2 --mode=ant-int-pot --wbit=4 --abit=4 --batch_size=64 --lr=0.00005 --train > ./log/resnet50_IP.log 2>&1

#FIP
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46668 main.py --dataset=imagenet --model=resnet50 --epoch=2 --mode=ant-int-pot-float --wbit=4 --abit=4 --batch_size=64 --lr=0.00005 --train > ./log/resnet50_FIP.log 2>&1

#IP-F
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46669 main.py --dataset=imagenet --model=resnet50 --epoch=2 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=64 --lr=0.00005 --train > ./log/resnet50_IP-F.log 2>&1

#FIP-F
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46670 main.py --dataset=imagenet --model=resnet50 --epoch=2 --mode=ant-int-pot-float-flint --wbit=4 --abit=4 --batch_size=64 --lr=0.00005 --train > ./log/resnet50_FIP-F.log 2>&1

#ANT4-8
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46671 main.py --dataset=imagenet --model=resnet50 --epoch=3 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=64 --lr=0.0005 --train -l8=0,53 > ./log/resnet50_ANT4-8.log 2>&1
