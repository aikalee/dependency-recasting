import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PREDICTION_DIR = BASE_DIR / "predictions"

UD_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",   
    "Polish": "pl",
    "Russian": "ru"
    }

# language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
STNZ_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh-hans",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "Polish": "pl",
    "Russian": "ru"
    }

TREEBANK_LOOKUP = {
    "Ancient_Greek": "Perseus",
    "Chinese": "Penn",
    "Czech": "PDT",
    "Dutch": "Alpino",
    "English": "Penn",
    "Polish": "LFG",
    "Russian": "SynTagRus"
    }
     

def get_projz_file_path(lang, split, is_upstream=True):
    """
    file name: lang_method_split
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")

    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    task = "upstream" if is_upstream else "downstream"
   
    read_path = DATA_DIR / "raw" /f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-{split}.conllu"
    output_dir = DATA_DIR / task / "projectivized" / f"UD_{lang}-{treebank}"
    write_path = output_dir / f"{ud_abbr}__{split}.conllu"
    
    if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_dep2const_file_path(lang="Chinese", split="train", pos="XPOS", is_upstream=True):
    """
    Input file name: lang_method_split
    Output dir name: lang=en
    Output file name: en__train.mrg
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")
    

    ud_abbr = UD_ABBR_LOOKUP[lang]
    stnz_abbr = STNZ_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    task = "upstream" if is_upstream else "downstream"
   
    read_path = DATA_DIR / task / "projectivized" / f"UD_{lang}-{treebank}/{ud_abbr}__{split}.conllu"
    output_dir = DATA_DIR / task / "constituentized" / f"lang={stnz_abbr},pos={pos.lower()}"     
      
    if not read_path.exists():
        raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  
    write_path = output_dir / f"{stnz_abbr}__{split}.mrg"
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_const2dep_file_path(lang, pos="XPOS", epochs=20, is_neural=True):
    """
    Input file name: lang=en,bert=frozen,charlm=yes,pretrain=yes,epochs=20.mrg
    Output file name: lang=en,bert=frozen,charlm=yes,pretrain=yes,epochs=20.conllu
    """
    
    ud_abbr = UD_ABBR_LOOKUP[lang]
    stnz_abbr = STNZ_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

    read_write_dir = PREDICTION_DIR / "stanza" 
    epoch_info = f",epochs={epochs}"
    postprocess = "neural" if is_neural else "rule_based"

    
    read_tree_path = read_write_dir / "raw" / f"lang={ud_abbr}{epoch_info}.mrg"
    read_orig_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_path = read_write_dir / postprocess / f"lang={stnz_abbr},pos={pos.lower()}{epoch_info}.conllu"
            
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

def get_deprojz_file_path(lang, pos="XPOS", epochs=20, is_neural=True):
    """
    file name: lang_method_split_(pseudo)
    """
    
    stnz_abbr = STNZ_ABBR_LOOKUP[lang]
    epoch_info = f",epochs={epochs}"
    postprocess = "neural" if is_neural else "rule_based"

    read_write_dir = PREDICTION_DIR / "stanza" / postprocess
   
    read_path = read_write_dir / f"lang={stnz_abbr},pos={pos.lower()}{epoch_info}.conllu"
    write_path = read_write_dir / f"lang={stnz_abbr},pos={pos.lower()}{epoch_info},deprojz=yes.conllu"
  
    
    if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
  
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_matched_file_path(lang, epochs=20):
    """
    Input file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20.conllu
    Output file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20,matched=yes.conllu
    """
    
    ud_abbr = UD_ABBR_LOOKUP[lang]
    stnz_abbr = STNZ_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

   
    read_write_dir = PREDICTION_DIR / "stanza" 
    epoch_info = f",epochs={epochs}"

    read_system_path = read_write_dir / f"lang={stnz_abbr}{epoch_info},deprojz=yes.conllu"
    read_gold_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}/{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_system_path = read_write_dir / f"lang={stnz_abbr}{epoch_info},deprojz=yes,matched=yes.conllu"
    write_gold_path = DATA_DIR / "upstream" / "gold" / f"UD_{lang}-{treebank}/lang={stnz_abbr}{epoch_info},matched=yes.conllu"
            
    for read_path in [read_system_path, read_gold_path]:
        if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
        
    output_dir = DATA_DIR / "upstream" / "gold" / f"UD_{lang}-{treebank}/"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  

    print(f"Loading from {read_system_path} and {read_gold_path}...")
    print(f"Writing into {write_system_path} and {write_gold_path}...")
    
    return read_system_path, read_gold_path, write_system_path, write_gold_path

def get_linearization_file_path(lang, split="train", pos="XPOS", is_tgt=False, epochs=20):

    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

    if is_tgt:
        read_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-{split}.conllu"
    else:
        read_path = PREDICTION_DIR / "stanza" / "raw" /  f"lang={ud_abbr},pos={pos.lower()},epochs={epochs}.mrg"

    if not read_path.exists():
        raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    output_dir = DATA_DIR / "downstrem" / f"lang={ud_abbr}"
    write_path = output_dir / f"train.{"tgt" if is_tgt else "src"}.txt"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
    
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path


    