mkdir -p log

# Run on NVIDIA A100.

#Int
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --dataset=imagenet --model=inception_v3 --epoch=2 --mode=int --wbit=4 --abit=4 --batch_size=160 --lr=5e-05 --train > ./log/inception_v3_Int.log 2>&1

#IP
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46667 main.py --dataset=imagenet --model=inception_v3 --epoch=2 --mode=ant-int-pot --wbit=4 --abit=4 --batch_size=64 --lr=1e-05 -al=50 --train > ./log/inception_v3_IP.log 2>&1

#FIP
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46668 main.py --dataset=imagenet --model=inception_v3 --epoch=2 --mode=ant-int-pot-float --wbit=4 --abit=4 --batch_size=64 --lr=1e-05 -al=50 --train > ./log/inception_v3_FIP.log 2>&1

#IP-F
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46669 main.py --dataset=imagenet --model=inception_v3 --epoch=2 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=64 --lr=1e-05 -al=50 --train > ./log/inception_v3_IP-F.log 2>&1

#FIP-F
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46670 main.py --dataset=imagenet --model=inception_v3 --epoch=2 --mode=ant-int-pot-float-flint --wbit=4 --abit=4 --batch_size=64 --lr=1e-05 -al=50 --train > ./log/inception_v3_FIP-F.log 2>&1

#ANT4-8
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port 46671 main.py --dataset=imagenet --model=inception_v3 --epoch=2 --mode=ant-int-pot-flint --wbit=4 --abit=4 --batch_size=64 --lr=5e-05 --train -l8=0,1,2,3,94 > ./log/inception_v3_ANT4-8.log 2>&1
