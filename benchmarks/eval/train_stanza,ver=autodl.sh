#!/bin/bash

DATA="lang=pl"
MODEL="${DATA},bert=finetune,charlm=no,pretrain=yes,epochs=100"

# === Force Stanza to read the file as UTF-8 ===
export PYTHONUTF8=1

# === Let Stanza training scripts know where the root directory lives ===
export CONSTITUENCY_BASE="/root/autodl-tmp/recasting"

# === Training data dir ===
export CONSTITUENCY_DATA_DIR="/root/autodl-tmp/recasting/data/${DATA}"

# === Find the training scripts `stanza.utils.training.run_constituency` ===
export PYTHONPATH="/root/autodl-tmp/stanza"

# === Access to CoreNLP utilies ===
export CLASSPATH="/root/autodl-tmp/stanford-corenlp-4.5.10/*"
export CORENLP_HOME="/root/autodl-tmp/stanford-corenlp-4.5.10"

# === Let Stanza know where the resources are to avoid downloading ===
export STANZA_RESOURCES_DIR="/root/autodl-tmp/stanza_resources"

# === Force Transformers into offline mode ===
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Options:
# [--bert_finetune]
# [--bert_model /root/autodl-tmp/roberta-base] 

python -m stanza.utils.training.run_constituency pl_ \
    --use_bert \
    --bert_model /root/autodl-tmp/roberta-base/ \
    --bert_finetune \
    --no_charlm \
    --no_retag \
    --epochs 100 \
    --save_dir /root/autodl-tmp/recasting/stanza-models/$MODEL \
    --seed 1234