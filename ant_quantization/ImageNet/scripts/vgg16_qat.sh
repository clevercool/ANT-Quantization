mkdir -p log

# Run on NVIDIA A100.

#Int
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --dataset=imagenet --model=vgg16_bn --epoch=2 --mode=int --wbit=4 --abit=4 --batch_size=128 --lr=0.0001 --train > ./log/vgg16_bn_Int.log 2>&1

#IP
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46667 main.py --dataset=imagenet --model=vgg16_bn --epoch=2 --mode=ant-int-pot --wbit=4 --abit=4 --batch_size=128 --lr=0.0001 --train > ./log/vgg16_bn_IP.log 2>&1

#FIP
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46668 main.py --dataset=imagenet --model=vgg16_bn --epoch=2 --mode=ant-int-pot-float --wbit=4 --abit=4 --batch_size=128 --lr=0.0001 --train > ./log/vgg16_bn_FIP.log 2>&1

#IP-F
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46669 main.py --dataset=imagenet --model=vgg16_bn --epoch=2 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=128 --lr=0.0001 --train > ./log/vgg16_bn_IP-F.log 2>&1

#FIP-F
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46670 main.py --dataset=imagenet --model=vgg16_bn --epoch=2 --mode=ant-int-pot-float-flint --wbit=4 --abit=4 --batch_size=128 --lr=0.0001 --train > ./log/vgg16_bn_FIP-F.log 2>&1

#ANT4-8
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46671 main.py --dataset=imagenet --model=vgg16_bn --epoch=3 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=64 --lr=0.0001 --train -l8=0,15 > ./log/vgg16_bn_ANT4-8.log 2>&1
