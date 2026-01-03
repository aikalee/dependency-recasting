#!/bin/bash

PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")

# === Add project root to PATH (similar to Python sys.path.insert) ===
export PATH="$PROJECT_ROOT:$PATH"

# === Change working directory for file operations ===
cd "$PROJECT_ROOT" || exit
echo "Now running in: $(pwd)"

# EPOCH=("20" "100")

LANG="Polish"

declare -A ptb_abbr
ptb_abbr["English"]="en"
ptb_abbr["Polish"]="pl"

declare -A treebank
treebank["English"]="Penn"
treebank["Polish"]="LFG"

OUTPUT_DIR="results/stanza"
mkdir -p "$OUTPUT_DIR"

# for EP in "${EPOCH[@]}"; do
    
# lowercase the treebank name (Penn → penn)
TBLOWER="${treebank[$LANG],,}"

MODELNAME="lang=${ptb_abbr[$LANG]},bert=finetune,charlm=no,pretrain=yes,epochs=100,deprojz=yes"

SYSFILE="predictions/stanza/${MODELNAME}.conllu"
GOLDFILE="data/raw/UD_${LANG}-${treebank[$LANG]}/${ptb_abbr[$LANG]}_${TBLOWER}-ud-test.conllu"

OUTPUT="$OUTPUT_DIR/validation,${MODELNAME}.txt"

echo "Validating $SYSFILE against $GOLDFILE..."
python benchmarks/eval/eval.py --verbose "$SYSFILE" "$GOLDFILE" > "$OUTPUT"

# done

echo "All validations done. Logs saved in $OUTPUT_DIR/"
