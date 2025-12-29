#!/bin/bash

MODEL="lang=en,bert=frozen,charlm=no,pretrain=no"

# === Force Stanza to read the file as UTF-8 ===
export PYTHONUTF8=1

# === Find the training scripts `stanza.utils.training.run_constituency` ===
export PYTHONPATH="/root/autodl-tmp/stanza"

# === Training data dir ===
export DEPPARSE_DATA_DIR="/root/autodl-tmp/recasting/data/${DATA}"

# === Let Stanza know where the resources are to avoid downloading ===
export STANZA_RESOURCES_DIR="/root/autodl-tmp/stanza_resources"

# === Force Transformers into offline mode ===
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Options: 
# Frozen BERT [--use-bert] [--bert_model]
# Finetuning BERT[--bert_finetune][--no_charlm][--no_pretrain]
# charlm [--no_charlm]
# Pretrain [--no_pretrain]

python3 -m stanza.utils.training.run_depparse en_ptb \
    --use_bert \
    --bert_model /root/autodl-tmp/roberta-base \
    --no_charlm \
    --no_pretrain \
    --save_dir /root/autodl-tmp/recasting/depparse-models/$MODEL \
    --seed 42