DATA="en-penn-ud,filter=none,method=most-crossed,pos=upos"

# === Force Stanza to read the file as UTF-8 ===
export PYTHONUTF8=1

# === Find the training scripts `src.main` ===
export PYTHONPATH="/root/autodl-tmp/self-attentive-parser"

python -m src.main \
--model-path-base bnp-models/$DATA \
--train-path data/processed/$DATA/en__train.mrg \
--dev-path data/processed/$DATA/dev__train.mrg \
--subbatch-max-tokens 3000 \
--batch-size 16 \
--numpy-seed 1234 \
--use-pretrained \
--pretrained-model /root/autodl-tmp/roberta-base
