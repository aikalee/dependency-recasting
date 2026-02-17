#!/bin/bash

PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")

# === Add project root to PATH (similar to Python sys.path.insert) ===
export PATH="$PROJECT_ROOT:$PATH"

# === Change working directory for file operations ===
cd "$PROJECT_ROOT" || exit
echo "Now running in: $(pwd)"

# EPOCH=("20" "100")

MODE="rule_based"
LANG="Ancient_Greek"

declare -A ud_abbr
ud_abbr["Chinese"]="zh"
ud_abbr["English"]="en"
ud_abbr["Polish"]="pl"
ud_abbr["Ancient_Greek"]="grc"

declare -A stnz_abbr
stnz_abbr["Chinese"]="zh-hans"
stnz_abbr["English"]="en"
stnz_abbr["Polish"]="pl"
stnz_abbr["Ancient_Greek"]="grc"

declare -A treebank
treebank["Chinese"]="Penn"
treebank["English"]="Penn"
treebank["Polish"]="LFG"
treebank["Ancient_Greek"]="Perseus"

OUTPUT_DIR="results/stanza/${MODE}"
mkdir -p "$OUTPUT_DIR"

# for EP in "${EPOCH[@]}"; do
    
# lowercase the treebank name (Penn → penn)
TBLOWER="${treebank[$LANG],,}"

MODELNAME="lang=${stnz_abbr[$LANG]},pos=upos,epochs=100"

SYSFILE="predictions/stanza/${MODE}/${MODELNAME},deprojz=yes.conllu"
GOLDFILE="data/raw/UD_${LANG}-${treebank[$LANG]}/${ud_abbr[$LANG]}_${TBLOWER}-ud-test.conllu"

OUTPUT="$OUTPUT_DIR/validation,${MODELNAME}.txt"

echo "Validating $SYSFILE against $GOLDFILE..."
python benchmarks/eval/eval.py --verbose "$SYSFILE" "$GOLDFILE" > "$OUTPUT"

# done

echo "All validations done. Logs saved in $OUTPUT_DIR/"
