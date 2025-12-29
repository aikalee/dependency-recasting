#!/bin/bash

# Paths
MY_PROJECT="/root/autodl-tmp/recasting"
BNP_REPO="/root/autodl-tmp/self-attentive-parser"
DATA="en-penn-ud,filter=none,method=most-crossed,pos=upos"

# Force UTF-8
export PYTHONUTF8=1

# BNP code lives here
export PYTHONPATH="$BNP_REPO"

# Run BNP from BNP directory
cd "$BNP_REPO" || exit 1

python -m src.main train \
  --test-path "$MY_PROJECT/data/$DATA/en__test.mrg" \
  --evalb-dir "$BNP_REPO/EVALB" \
  --subbatch-max-tokens 3000 \
  --numpy-seed 1234 \
  --use-pretrained \
  --pretrained-model /root/autodl-tmp/roberta-base \
  --model-path-base "$MY_PROJECT/bnp-models/$DATA"

