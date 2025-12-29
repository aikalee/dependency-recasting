#!/bin/bash

PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")

# === Add project root to PATH (similar to Python sys.path.insert) ===
export PATH="$PROJECT_ROOT:$PATH"

# === Change working directory for file operations ===
cd "$PROJECT_ROOT" || exit
echo "Now running in: $(pwd)"

# EPOCH=("20" "100")

LANG="English"

declare -A ptb_abbr
ptb_abbr["English"]="en"

declare -A treebank
treebank["English"]="Penn"

OUTPUT_DIR="results/bnp"
mkdir -p "$OUTPUT_DIR"

# for EP in "${EPOCH[@]}"; do
    
# lowercase the treebank name (Penn → penn)
TBLOWER="${treebank[$LANG],,}"

# MODELNAME="${ptb_abbr[$LANG]}-${TBLOWER}-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=${EP}"
MODELNAME="${ptb_abbr[$LANG]}-${TBLOWER}-ud,filter=none,method=most-crossed-deprojz,pos=upos"

# SYSFILE="predictions/stanza/${MODELNAME},matched=yes.conllu"
SYSFILE="predictions/bnp/${MODELNAME},matched=yes.conllu"
# GOLDFILE="data/processed/gold/UD_${LANG}-${treebank[$LANG]}/en_penn-ud-test,epoch=${EP},matched=yes.conllu"
GOLDFILE="data/processed/gold/UD_${LANG}-${treebank[$LANG]}/en_penn-ud-test,matched=yes.conllu"

OUTPUT="$OUTPUT_DIR/validation,${MODELNAME}.txt"

echo "Validating $SYSFILE against $GOLDFILE..."
python benchmarks/eval/eval.py --verbose "$SYSFILE" "$GOLDFILE" > "$OUTPUT"

# done

echo "All validations done. Logs saved in $OUTPUT_DIR/"
