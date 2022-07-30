import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models
import sys
import os
import argparse
import pickle
import numpy as np
import copy
sys.path.append("../antquant")
from quant_model import *
from quant_utils import *
from dataloader import get_dataloader, get_imagenet_dataloader


parser = argparse.ArgumentParser(description='PyTorch Adaptive Numeric DataType Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--ckpt_path', default=None, type=str,
                    help='checkpoint path')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    help='dataset name')
parser.add_argument('--dataset_path', default='/nvme/imagenet', type=str, 
                    help='dataset path')
parser.add_argument('--model', default='resnet18', type=str, 
                    help='model name')
parser.add_argument('--train', default=False, action='store_true', 
                    help='train')
parser.add_argument('--epoch', default=3, type=int, 
                    help='epoch num')
parser.add_argument('--batch_size', default=256, type=int, 
                    help='batch_size num')
parser.add_argument('--tag', default='', type=str, 
                    help='tag checkpoint')
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
                    
parser.add_argument('--mode', default='base', type=str,
                    help='quantizer mode')
parser.add_argument('--wbit', '-wb', default='8', type=int, 
                    help='weight bit width')
parser.add_argument('--abit', '-ab', default='8', type=int, 
                    help='activation bit width')
parser.add_argument('--search', default=False, action='store_true', 
                    help='search alpha')
parser.add_argument('--w_up', '-wu', default='150', type=int, 
                    help='weight search upper bound')
parser.add_argument('--a_up', '-au', default='150', type=int, 
                    help='activation search upper bound')
parser.add_argument('--w_low', '-wl', default='75', type=int, 
                    help='weight search lower bound')
parser.add_argument('--a_low', '-al', default='75', type=int, 
                    help='activation search lower bound')
parser.add_argument('--percent', '-p', default='100', type=int, 
                    help='percent for outlier')
parser.add_argument('--ptq', default=False, action='store_true', 
                    help='post training quantization')
parser.add_argument('--disable_quant', default=False, action='store_true', 
                    help='disable quantization')
parser.add_argument('--disable_input_quantization', default=False, action='store_true', 
                    help='disable input quantization')
parser.add_argument('--layer_8bit_n', '-n8', default='0', type=int, 
                    help='number of 8-bit layers')
parser.add_argument('--layer_8bit_l', '-l8', default=None, type=str, 
                    help='list of 8-bit layers')
args = parser.parse_args()

print(args)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

dist.init_process_group(backend='nccl')
local_rank = args.local_rank
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cudnn.benchmark = True

# output path
output_path = get_ckpt_path(args)

# logging setting
set_util_logging(output_path + "/training.log")
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(output_path + "/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(output_path)
logger.info(args)

# Data
logger.info('==> Preparing data..')
trainloader, testloader = get_dataloader(args.dataset, args.batch_size, args.dataset_path, args.model)

# Set Quantizer
logger.info('==> Setting quantizer..')
set_quantizer(args)
print(args)
logger.info(args)

# Model
logger.info('==> Building model..')
model = get_model(args)

model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = quantize_model(model)
set_first_last_layer(model)
if not args.disable_quant and args.mode != 'base':
    enable_quantization(model)
else:
    disable_quantization(model)
if args.disable_input_quantization:
    disable_input_quantization(model)

model.to(device)

args.lr = args.lr * float(args.batch_size * dist.get_world_size()) / 256.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)

"""
### LR scheduler
# Train resnet-50: decay the learning rate by a factor of 10 at the 30th, 48th, and 58th epochs.
# Using such a schedule, we reach 75% single crop top-1 accuracy on ImageNet 
# in just 50 epochs and reach 75.5% top-1 accuracy in 60 epochs. 
# https://arxiv.org/pdf/1907.08610v2.pdf
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [2,3,4], gamma = 0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.ckpt_path, map_location='cuda')
    check = {}
    for key, item in checkpoint['model'].items():
        check[key[7:]] = item

    ## Load for ANT quantizaton
    load_ant_state_dict(model, check)

    model.load_state_dict(check, strict=True)
    start_epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])


model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

def reduce_ave_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
def reduce_sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

# Training
def train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0 
    total = 0
    best_acc = 0

    for batch_idx, data in enumerate(trainloader):
        inputs = data[0]["data"]
        targets = data[0]["label"].squeeze(-1).long()

        if batch_idx == 0 and epoch == 0  and args.layer_8bit_n != 0:
            model(inputs)
            set_8_bit_layer_n(model, args.layer_8bit_n)
        if batch_idx == 0 and epoch == 0  and args.layer_8bit_l != None:
            model(inputs)
            set_8_bit_layer_l(model, args.layer_8bit_l)
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        correct_sum = reduce_sum_tensor(torch.tensor(correct).to(device)).item()
        total_sum = reduce_sum_tensor(torch.tensor(total).to(device)).item()
        acc = 100.*correct_sum/total_sum

        if local_rank == 0:
            if batch_idx % 100 == 0:
                logger.info('test: [epoch: %d | batch: %d/%d ] | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), acc, correct_sum, total_sum))
        
        # if batch_idx == 1000:
        #     break

        step = batch_idx + epoch * len(trainloader)

    scheduler.step()
    if epoch % 1 == 0:
        if local_rank == 0:
            logger.info('Saving Checkpoint')
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, get_ckpt_filename(output_path, epoch))
    trainloader.reset()

# Post-Training Quantization
def ptq_init():
    model.eval()
    data = trainloader.next()
    inputs = data[0]["data"]
    model(inputs)
    del(inputs)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0     
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            inputs = data[0]["data"]
            targets = data[0]["label"].squeeze(-1).long()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            _, predicted_5 = outputs.topk(5, 1, True, True)
            predicted_5 = predicted_5.t()
            correct_ = predicted_5.eq(targets.view(1, -1).expand_as(predicted_5))
            correct_5 += correct_[:5].reshape(-1).float().sum(0, keepdim=True).item()

            correct_sum = reduce_sum_tensor(torch.tensor(correct).to(device)).item()
            correct_5_sum = reduce_sum_tensor(torch.tensor(correct_5).to(device)).item()
            total_sum = reduce_sum_tensor(torch.tensor(total).to(device)).item()
            if local_rank == 0:
                if batch_idx % 10 == 0 or batch_idx == len(testloader) - 1:
                    logger.info('test: [batch: %d/%d ] | Loss: %.3f | Acc: %.3f%% (%d/%d)/ %.3f%% (%d/%d)'
                                % (batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct_sum/total_sum, correct_sum, total_sum, 100.*correct_5_sum/total_sum, correct_5_sum, total_sum))

            ave_loss = test_loss/total

    acc = 100.*correct/total
    testloader.reset()

    acc_ave = reduce_ave_tensor(torch.tensor(acc).to(device)).item()
    if local_rank == 0:
        logger.info("Final accuracy: %.3f" % acc_ave)

if args.train:
    for epoch in range(start_epoch, start_epoch + args.epoch):
        logger.info(scheduler.state_dict())
        train(epoch)
        test()
else:
    test()
