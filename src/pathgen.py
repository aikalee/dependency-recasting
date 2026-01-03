import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PREDICTION_DIR = BASE_DIR / "predictions"

UD_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",   
    "Polish": "pl",
    "Russian": "ru"
    }

# language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
PTB_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "Polish": "pl",
    "Russian": "ru"
    }

TREEBANK_LOOKUP = {
    "Ancient_Greek": "Perseus",
    "Czech": "PDT",
    "Dutch": "Alpino",
    "English": "Penn",
    "Polish": "LFG",
    "Russian": "SynTagRus"
    }
     

def get_projz_file_path(lang, split):
    """
    file name: lang_method_split
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")

    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
   
    read_path = DATA_DIR / "raw" /f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-{split}.conllu"
    output_dir = DATA_DIR / "processed" / "projectivized" / f"UD_{lang}-{treebank}"
    write_path = output_dir / f"{ud_abbr}__{split}.conllu"
    
    if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_deprojz_file_path(lang, model, bert="frozen", charlm="yes", pretrain="yes", epochs=20):
    """
    file name: lang_method_split_(pseudo)
    """
    if model not in ["stanza", "bnp"]:
        raise ValueError("Model must be either `stanza` or `bnp`.")
    ptb_abbr = PTB_ABBR_LOOKUP[lang]
    

    if model == "stanza":
        read_write_dir = PREDICTION_DIR / "stanza" 
        epoch_info = f",epochs={epochs}"
    elif model == "bnp":
        read_write_dir = PREDICTION_DIR / "bnp"
        epoch_info = ""

    read_path = read_write_dir / f"lang={ptb_abbr},bert={bert},charlm={charlm},pretrain={pretrain}{epoch_info}.conllu"
    write_path = read_write_dir / f"lang={ptb_abbr},bert={bert},charlm={charlm},pretrain={pretrain}{epoch_info},deprojz=yes.conllu"
  
    
    if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
  
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_tree_file_path(lang="Chinese", split="train"):
    """
    Input file name: lang_method_split
    Output dir name: lang=en
    Output file name: en__train.mrg
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")
    

    ud_abbr = UD_ABBR_LOOKUP[lang]
    ptb_abbr = PTB_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
   
    read_path = DATA_DIR / "processed" / "projectivized" / f"UD_{lang}-{treebank}/{ud_abbr}__{split}.conllu"
    output_dir = DATA_DIR / "processed" / "constituentized" / f"lang={ptb_abbr}"     
      
    if not read_path.exists():
        raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  
    write_path = output_dir / f"{ptb_abbr}__{split}.mrg"
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_conllu_file_path(lang, model, bert="frozen", charlm="yes", pretrain="yes", epochs=20):
    """
    Input file name: lang=en,bert=frozen,charlm=yes,pretrain=yes,epochs=20.mrg
    Output file name: lang=en,bert=frozen,charlm=yes,pretrain=yes,epochs=20.conllu
    """
    if model not in ["stanza", "bnp"]:
        raise ValueError("Model must be either `stanza` or `bnp`.")
    
    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

    if model == "stanza":
        read_write_dir = PREDICTION_DIR / "stanza" 
        epoch_info = f",epochs={epochs}"
    elif model == "bnp":
        read_write_dir = PREDICTION_DIR / "bnp"
        epoch_info = ""
    
    read_tree_path = read_write_dir / f"lang={ud_abbr},bert={bert},charlm={charlm},pretrain={pretrain}{epoch_info}.mrg"
    read_orig_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_path = read_write_dir / f"lang={ud_abbr},bert={bert},charlm={charlm},pretrain={pretrain}{epoch_info}.conllu"
            
    if not read_tree_path.exists():
        raise FileNotFoundError(f"The file '{read_tree_path}' does not exist.")
    if not read_orig_path.exists():
        raise FileNotFoundError(f"The file '{read_orig_path}' does not exist.")
    
    # if not os.path.exists(write_path):
    #     os.makedirs(write_path, exist_ok=True)
    #     print(f"Created: {write_path}")

    print(f"Loading from {read_tree_path}...")
    print(f"Writing into {write_path}...")
    
    return read_tree_path, read_orig_path, write_path

def get_matched_file_path(lang, model, bert="frozen", charlm="yes", pretrain="yes", epochs=20):
    """
    Input file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20.conllu
    Output file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20,matched=yes.conllu
    """
    if model not in ["stanza", "bnp"]:
        raise ValueError("Model must be either `stanza` or `bnp`.")
    
    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

    if model == "stanza":
        read_write_dir = PREDICTION_DIR / "stanza" 
        epoch_info = f",epochs={epochs}"
    elif model == "bnp":
        read_write_dir = PREDICTION_DIR / "bnp"
        epoch_info = ""

    read_system_path = read_write_dir / f"lang={ud_abbr},bert={bert},charlm={charlm},pretrain={pretrain}{epoch_info},deprojz=yes.conllu"
    read_gold_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}/{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_system_path = read_write_dir / f"lang={ud_abbr},bert={bert},charlm={charlm},pretrain={pretrain}{epoch_info},deprojz=yes,matched=yes.conllu"
    write_gold_path = DATA_DIR / "processed" / "gold" / f"UD_{lang}-{treebank}/lang={ud_abbr},bert={bert},charlm={charlm},pretrain={pretrain}{epoch_info},matched=yes.conllu"
            
    for read_path in [read_system_path, read_gold_path]:
        if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
        
    output_dir = DATA_DIR / "processed" / "gold" / f"UD_{lang}-{treebank}/"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  

    print(f"Loading from {read_system_path} and {read_gold_path}...")
    print(f"Writing into {write_system_path} and {write_gold_path}...")
    
    return read_system_path, read_gold_path, write_system_path, write_gold_path

    