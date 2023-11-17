---
license: other
tags:
- generated_from_trainer
datasets:
- wikitext
model-index:
- name: opt-125m
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# opt-125m

This model is a fine-tuned version of [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- eval_loss: 4.4711
- eval_accuracy: 0.2692
- eval_runtime: 37.8287
- eval_samples_per_second: 6.424
- eval_steps_per_second: 3.225
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.26.1
- Pytorch 1.11.0
- Datasets 2.15.0
- Tokenizers 0.13.3
