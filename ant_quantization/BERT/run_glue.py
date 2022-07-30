# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import logging
import os
import random
import json
import time
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

import modeling
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from sklearn.metrics import matthews_corrcoef, f1_score
from utils import (is_main_process, mkdir_by_main_process, format_step,
                   get_world_size)
from processors.glue import PROCESSORS, convert_examples_to_features

import sys
sys.path.append("../antquant")
from quant_model import *
from quant_utils import *

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def parse_args(parser=argparse.ArgumentParser()):
    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data "
        "files) for the task.",
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, "
        "bert-base-multilingual-uncased, bert-base-multilingual-cased, "
        "bert-base-chinese.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str.lower,
        required=True,
        choices=PROCESSORS.keys(),
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints "
        "will be written.",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        required=True,
        help="The checkpoint file from pretraining",
    )

    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece "
        "tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to get model-task performance on the dev"
                        "set by running eval.")
    parser.add_argument("--do_find",
                        action='store_true')
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to output prediction results on the dev ")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1.0,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup "
        "for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a "
        "backward/update pass.")
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Mixed precision training",
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help="Mixed precision training",
    )
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when "
        "fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--vocab_file',
                        type=str,
                        default=None,
                        required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")

    ## For quantization
    parser.add_argument('--mode', default='source', type=str,
                    help='quantizer mode')
    parser.add_argument('--wbit', '-wb', default='8', type=int, 
                        help='weight bit width')
    parser.add_argument('--abit', '-ab', default='8', type=int, 
                        help='activation bit width')
    parser.add_argument('--percent', '-p', default='100', type=int, 
                        help='percent')
    parser.add_argument('--sigma', '-s', default='0', type=float, 
                        help='Init activation range with Batchnorm Sigma')
    parser.add_argument('--disable_quant', default=False, action='store_true', 
                        help='disable quant')
    parser.add_argument('--disable_input_quantization', default=False, action='store_true', 
                        help='quant_input')
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
    parser.add_argument('--layer_8bit_n', '-n8', default='0', type=int, 
                        help='number of 8-bit layers')
    parser.add_argument('--layer_8bit_l', '-l8', default=None, type=str, 
                        help='list of 8-bit layers')
    args = parser.parse_args()

    return parser.parse_args()

args = parse_args()
# set logging
set_util_logging(os.path.join(args.output_dir, "debug.log"))
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, "debug.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def init_optimizer_and_amp(model, learning_rate, loss_scale, warmup_proportion,
                           num_train_optimization_steps, use_fp16):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer, scheduler = None, None
    logger.info("using fp32")
    if num_train_optimization_steps is not None:
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            warmup=warmup_proportion,
            t_total=num_train_optimization_steps,
        )
    return model, optimizer, scheduler


def gen_tensor_dataset(features):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long,
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in features],
        dtype=torch.long,
    )
    return TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )


def get_train_features(data_dir, bert_model, max_seq_length, do_lower_case,
                       local_rank, train_batch_size,
                       gradient_accumulation_steps, num_train_epochs, tokenizer,
                       processor):
    cached_train_features_file = os.path.join(
        data_dir,
        '{0}_{1}_{2}'.format(
            list(filter(None, bert_model.split('/'))).pop(),
            str(max_seq_length),
            str(do_lower_case),
        ),
    )
    train_features = None
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
        logger.info("Loaded pre-processed features from {}".format(
            cached_train_features_file))
    except:
        logger.info("Did not find pre-processed features from {}".format(
            cached_train_features_file))
        train_examples = processor.get_train_examples(data_dir)
        train_features, _ = convert_examples_to_features(
            train_examples,
            processor.get_labels(),
            max_seq_length,
            tokenizer,
        )
        if is_main_process():
            logger.info("  Saving train features into cached file %s",
                        cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
    return train_features


def main(args):
    logger.info(args)
    ## Quantization setting
    logger.info('==> Setting quantizer..')

    args.fp16 = args.fp16 or args.amp
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port),
            redirect_output=True,
        )
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device,
                    n_gpu,
                    bool(args.local_rank != -1),
                    args.fp16,
                ))

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or "
                         "`do_predict` must be True.")

    if is_main_process():
        if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and
                args.do_train):
            logger.warning("Output directory ({}) already exists and is not "
                           "empty.".format(args.output_dir))
    mkdir_by_main_process(args.output_dir)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                             args.gradient_accumulation_steps))
    if args.gradient_accumulation_steps > args.train_batch_size:
        raise ValueError("gradient_accumulation_steps ({}) cannot be larger "
                         "train_batch_size ({}) - there cannot be a fraction "
                         "of one sample.".format(
                             args.gradient_accumulation_steps,
                             args.train_batch_size,
                         ))
    args.train_batch_size = (args.train_batch_size //
                             args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # tokenizer = BertTokenizer(
    #     args.vocab_file,
    #     do_lower_case=args.do_lower_case,
    #     max_len=512,
    # )  # for bert large

    num_train_optimization_steps = None
    # if args.do_train:
    train_features = get_train_features(
        args.data_dir,
        args.bert_model,
        args.max_seq_length,
        args.do_lower_case,
        args.local_rank,
        args.train_batch_size,
        args.gradient_accumulation_steps,
        args.num_train_epochs,
        tokenizer,
        processor,
    )
    num_train_optimization_steps = int(
        len(train_features) / args.train_batch_size /
        args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = (num_train_optimization_steps //
                                        torch.distributed.get_world_size())

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    # if config.vocab_size % 8 != 0:
    #     config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    # modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu
    model = modeling.BertForSequenceClassification(
        config,
        num_labels=num_labels,
    )
    logger.info("USING CHECKPOINT from {}".format(args.init_checkpoint))
    state_dict = torch.load(args.init_checkpoint, map_location='cpu')['model']
    # print(check.keys())
    
    # model.from_pretrained(args.init_checkpoint, strict=True)

    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'LayerNorm.gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'LayerNorm.beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    model.load_state_dict(
        state_dict,
        strict=False,
    )
    logger.info("USED CHECKPOINT from {}".format(args.init_checkpoint))
    # set_percentile(model)

    # for param_tensor in model.state_dict():
    #     if "beta" not in param_tensor:
    #         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    #         print(model.state_dict()[param_tensor].requires_grad)


    # Set Quantizer
    set_quantizer(args)
    model = quantize_model(model)    
    model.to(device)
    if not args.disable_quant and args.mode != 'source':
        enable_quantization(model)
    else:
        disable_quantization(model)
    if args.disable_input_quantization:
        disable_input_quantization(model)

    # Prepare optimizer
    model, optimizer, scheduler = init_optimizer_and_amp(
        model,
        args.learning_rate,
        args.loss_scale,
        args.warmup_proportion,
        num_train_optimization_steps,
        args.fp16,
    )
    # if optimizer is not None:
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [1,2], gamma = 0.1, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [100, 200, 300], gamma = 0.1, last_epoch=-1)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use "
                              "distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    loss_fct = torch.nn.CrossEntropyLoss()
    results = {}

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        train_data = gen_tensor_dataset(train_features)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
        )

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        latency_train = 0.0
        nb_tr_examples = 0
        model.train()
        tic_train = time.perf_counter()
        loss_a = []
        for e in range(args.num_train_epochs):
            model.train()
            tr_loss, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):

                if step == 0 and e == 0  and args.layer_8bit_n != 0:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    model(input_ids, segment_ids, input_mask)
                    set_8_bit_layer_n(model, args.layer_8bit_n)
                if step == 0 and e == 0  and args.layer_8bit_l != None:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    model(input_ids, segment_ids, input_mask)
                    set_8_bit_layer_l(model, args.layer_8bit_l)
                
                if args.max_steps > 0 and global_step > args.max_steps:
                    break
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids, segment_ids, input_mask)
                loss = loss_fct(
                    logits.view(-1, num_labels),
                    label_ids.view(-1),
                )
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()

                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                out_label_ids = label_ids.detach().cpu().numpy()
                train_result = compute_metrics(args.task_name, preds, out_label_ids)

                if step % 10 == 0:
                    logger.info("epoch: {0} [{1}/{2}]: loss: {3}, acc: {4}".format(e, step, len(train_dataloader), loss.item(), train_result))
                tr_loss += loss.item()
                loss_a.append(loss.item())
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # eval
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features, label_map = convert_examples_to_features(
                eval_examples,
                processor.get_labels(),
                args.max_seq_length,
                tokenizer,
            )
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_data = gen_tensor_dataset(eval_features)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size,
            )

            model.eval()
            preds = None
            out_label_ids = None
            eval_loss = 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(eval_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)
                    if args.do_eval:
                        eval_loss += loss_fct(
                            logits.view(-1, num_labels),
                            label_ids.view(-1),
                        ).mean().item()

                nb_eval_steps += 1
                nb_eval_examples += input_ids.size(0)
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids,
                        label_ids.detach().cpu().numpy(),
                        axis=0,
                    )

            preds = np.argmax(preds, axis=1)
            results['eval:loss'] = eval_loss / nb_eval_steps
            eval_result = compute_metrics(args.task_name, preds, out_label_ids)
            results.update(eval_result)
            logger.info(eval_result)

        tr_loss = tr_loss / nb_tr_steps
        

        
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(
            {"model": model_to_save.state_dict()},
            os.path.join(args.output_dir, modeling.WEIGHTS_NAME),
        )
        with open(
                os.path.join(args.output_dir, modeling.CONFIG_NAME),
                'w',
        ) as f:
            f.write(model_to_save.config.to_json_string())

    if args.do_eval:
        # eval
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features, label_map = convert_examples_to_features(
            eval_examples,
            processor.get_labels(),
            args.max_seq_length,
            tokenizer,
        )
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_data = gen_tensor_dataset(eval_features)
        # eval_data = gen_tensor_dataset(train_features)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
        )

        model.eval()
        preds = None
        out_label_ids = None
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
                if args.do_eval:
                    eval_loss += loss_fct(
                        logits.view(-1, num_labels),
                        label_ids.view(-1),
                    ).mean().item()

            nb_eval_steps += 1
            nb_eval_examples += input_ids.size(0)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    label_ids.detach().cpu().numpy(),
                    axis=0,
                )            

            preds1 = np.argmax(preds, axis=1)
            results['eval:loss'] = eval_loss / nb_eval_steps
            eval_result = compute_metrics(args.task_name, preds1, out_label_ids)
            results.update(eval_result)
            logger.info(eval_result)
            # exit()

    logger.info("***** Results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    with open(os.path.join(args.output_dir, "results.txt"), "w") as writer:
        json.dump(results, writer)

    return results


if __name__ == "__main__":
    main(args)
