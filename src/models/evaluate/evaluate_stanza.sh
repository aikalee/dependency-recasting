#!/bin/bash

PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")")

# === Add project root to PATH (similar to Python sys.path.insert) ===
export PATH="$PROJECT_ROOT:$PATH"

# === Change working directory for file operations ===
cd "$PROJECT_ROOT" || exit
echo "Now running in: $(pwd)"

# EPOCH=("20" "100")

MODE="neural"
LANG="English"
abbr="en"
dir_abbr="en-penn"
treebank="Penn"
treebank_lower="penn"
pos="upos"

# declare -A ud_abbr
# ud_abbr["Chinese"]="zh"
# ud_abbr["English"]="en"
# ud_abbr["Finnish"]="fi"
# ud_abbr["Polish"]="pl"
# ud_abbr["Ancient_Greek"]="grc"

# declare -A stnz_abbr
# stnz_abbr["Chinese"]="zh-hans"
# stnz_abbr["English"]="en"
# stnz_abbr["Finnish"]="fi"
# stnz_abbr["Polish"]="pl"
# stnz_abbr["Ancient_Greek"]="grc"

# declare -A treebank
# treebank["Chinese"]="Penn"
# treebank["English"]="Penn"
# treebank["Finnish"]="TDT"
# treebank["Polish"]="LFG"
# treebank["Ancient_Greek"]="Perseus"

OUTPUT_DIR="results/stanza/${MODE}"
mkdir -p "$OUTPUT_DIR"

# for EP in "${EPOCH[@]}"; do
    
# lowercase the treebank name (Penn → penn)
# TBLOWER="${treebank,,}"

# MODELNAME="lang=${dir_abbr},pos=${pos},epochs=100"
MODELNAME="lang=${dir_abbr},pos=${pos},gate=yes"

SYSFILE="predictions/${MODE}/${MODELNAME},deprojz=yes.conllu"
GOLDFILE="data/raw/UD_${LANG}-${treebank}/${abbr}_${treebank_lower}-ud-test.conllu"

OUTPUT="$OUTPUT_DIR/validation,${MODELNAME}.txt"

echo "Validating $SYSFILE against $GOLDFILE..."
python src/models/evaluate/eval.py --verbose "$SYSFILE" "$GOLDFILE" 2>&1 | tee "$OUTPUT"
# python models/eval.py --verbose "$SYSFILE" "$GOLDFILE" > "$OUTPUT"

# done

echo "All validations done. Logs saved in $OUTPUT_DIR/"
