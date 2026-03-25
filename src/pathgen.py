import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PREDICTION_DIR = BASE_DIR / "predictions"

UD_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh",
    "English-Penn": "en",
    "English-EWT": "en",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Russian": "ru",
    "Tamil": "ta",
    "Uyghur": "ug",
    "Wolof": "wo"
    }

# language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
STNZ_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh-hans",
    "English-Penn": "en",
    "English-EWT": "en",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Russian": "ru",
    "Tamil": "ta",
    "Uyghur": "ug",
    "Wolof": "wo"
    }

DIR_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh-hans",
    "English-Penn": "en-penn",
    "English-EWT": "en-ewt",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Russian": "ru",
    "Tamil": "ta",
    "Uyghur": "ug",
    "Wolof": "wo"
    }


TREEBANK_LOOKUP = {
    "Ancient_Greek": "Perseus",
    "Chinese": "Penn",
    "English-Penn": "Penn",
    "English-EWT": "EWT",
    "Finnish": "TDT",
    "French": "GSD",
    "Hebrew": "HTB",
    "Russian": "GSD",
    "Tamil": "TTB",
    "Uyghur": "UDT",
    "Wolof": "WTB"
    }
     

def get_projz_file_path(lang, split, is_common=True):
    """
    file name: lang_method_split
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")

    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    task = "common" if is_common else "downstream"
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]

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

def get_dep2const_file_path(lang="Chinese", split="train", pos="XPOS", is_common=True):
    """
    Input file name: lang_method_split
    Output dir name: lang=en
    Output file name: en__train.mrg
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")
    
    ud_abbr = UD_ABBR_LOOKUP[lang]
    stnz_abbr = STNZ_ABBR_LOOKUP[lang]
    dir_abbr = DIR_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    task = "common" if is_common else "downstream"
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]

   
    read_path = DATA_DIR / task / "projectivized" / f"UD_{lang}-{treebank}/{ud_abbr}__{split}.conllu"
    output_dir = DATA_DIR / task / "constituentized" / f"lang={dir_abbr},pos={pos.lower()}"     
      
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
    dir_abbr = DIR_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]


    read_write_dir = PREDICTION_DIR  
    epoch_info = f",epochs={epochs}"
    postprocess = "neural" if is_neural else "rule_based"

    if is_neural:
        read_tree_path = read_write_dir / "neural" / f"lang={dir_abbr},pos={pos.lower()}{epoch_info}.mrg"
    else:
        read_tree_path = read_write_dir / "raw" / f"lang={dir_abbr},split=test,pos={pos.lower()}{epoch_info}.mrg"
    read_orig_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_path = read_write_dir / postprocess / f"lang={dir_abbr},pos={pos.lower()}{epoch_info}.conllu"
            
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
    dir_abbr = DIR_ABBR_LOOKUP[lang]
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]

    epoch_info = f",epochs={epochs}"
    postprocess = "neural" if is_neural else "rule_based"

    read_write_dir = PREDICTION_DIR / postprocess
   
    read_path = read_write_dir / f"lang={dir_abbr},pos={pos.lower()}{epoch_info}.conllu"
    write_path = read_write_dir / f"lang={dir_abbr},pos={pos.lower()}{epoch_info},deprojz=yes.conllu"
  
    
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
    dir_abbr = DIR_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]


   
    read_write_dir = PREDICTION_DIR
    epoch_info = f",epochs={epochs}"

    read_system_path = read_write_dir / f"lang={dir_abbr}{epoch_info},deprojz=yes.conllu"
    read_gold_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}/{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_system_path = read_write_dir / f"lang={dir_abbr}{epoch_info},deprojz=yes,matched=yes.conllu"
    write_gold_path = DATA_DIR / "common" / "gold" / f"UD_{lang}-{treebank}/lang={dir_abbr}{epoch_info},matched=yes.conllu"
            
    for read_path in [read_system_path, read_gold_path]:
        if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
        
    output_dir = DATA_DIR / "common" / "gold" / f"UD_{lang}-{treebank}/"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  

    print(f"Loading from {read_system_path} and {read_gold_path}...")
    print(f"Writing into {write_system_path} and {write_gold_path}...")
    
    return read_system_path, read_gold_path, write_system_path, write_gold_path

def get_mrg2txt_file_path(lang, split="train", pos="XPOS", epochs=20, is_target=False):

    stnz_abbr = STNZ_ABBR_LOOKUP[lang]
    dir_abbr = DIR_ABBR_LOOKUP[lang]
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]

    output_dir = DATA_DIR / "downstream" / "linearized" / f"lang={dir_abbr},pos={pos.lower()}"

    if is_target:
        read_path = DATA_DIR / "common" / "constituentized" / f"lang={dir_abbr},pos={pos.lower()}" / f"{stnz_abbr}__{split}.mrg"
    else:
        read_path = DATA_DIR / "downstream" / "upstream_outputs" / f"lang={dir_abbr},pos={pos.lower()}" / f"lang={dir_abbr},split={split},pos={pos.lower()},epochs={epochs}.mrg"

    if not read_path.exists():
        raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    tgt_or_src = "tgt" if is_target else "src"
    
    write_path = output_dir / f"{split}.{tgt_or_src}.txt"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
    
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_txt2mrg_file_path(lang, pos="XPOS", epochs=20):
    ud_abbr = UD_ABBR_LOOKUP[lang]
    dir_abbr = DIR_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]


    read_write_dir = PREDICTION_DIR / "neural"
    read_linearized_path = read_write_dir / f"lang={dir_abbr},pos={pos.lower()},epochs={epochs}.txt"
    read_source_path = PREDICTION_DIR / "raw" /  f"lang={dir_abbr},split=test,pos={pos.lower()},epochs=100.mrg"
    read_orig_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_path = read_write_dir / f"lang={dir_abbr},pos={pos.lower()},epochs={epochs}.mrg"

    print(f"Loading from {read_linearized_path} and {read_source_path}...")
    print(f"Writing into {write_path}...")

    return read_linearized_path, read_source_path, read_orig_path, write_path

def get_upsteam_inference_file_path(lang, split):
    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]

    read_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}" / f"{ud_abbr}_{treebank.lower()}-ud-{split}.conllu"
    output_dir = DATA_DIR / "upstream" / f"UD_{lang}-{treebank}" 
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
    
    write_path = output_dir / f"{ud_abbr}_{treebank.lower()}-ud-{split}.conllu"

    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")

    return read_path, write_path
