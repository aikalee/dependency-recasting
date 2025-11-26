import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PREDICTION_DIR = BASE_DIR / "predictions"

UD_ABBR_LOOKUP = {
        "Ancient_Greek": "grc",
        "Chinese": "zh",
        "Danish": "da",
        "English": "en",
        "Korean": "ko",
        "Latin": "la",
        "Old_East_Slavic": "orv",
        "Urdu": "ur"
    }

# language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
PTB_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh-hant",
    "Danish": "da",
    "English": "en",
    "Korean": "ko",
    "Latin": "la",
    "Old_East_Slavic": "orv",
    "Urdu": "ur"
}

TREEBANK_LOOKUP = {
        "Ancient_Greek": "Perseus",
        "Chinese": "GSD",
        "Danish": "DDT",
        "English": "Penn",
        "Korean": "GSD",
        "Latin": "UDante",
        "Old_East_Slavic": "RNC",
        "Urdu": "UDTB"
    }
     

def get_projz_file_path(lang, split, pseudo_filter=False):
    """
    file name: lang_method_split_(pseudo)
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")

    ud_abbr = UD_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]


   
    projz = f"{treebank.lower()}-ud-"
    pseudo = "_pseudo" if pseudo_filter else ""

    
    output_dir = DATA_DIR / "processed" / "ud" / f"UD_{lang}-{treebank}"
    read_path = DATA_DIR / "raw" /f"UD_{lang}-{treebank}" / f"{ud_abbr}_{projz}{split}.conllu"
    write_path = output_dir / f"{ud_abbr}_most-crossed_{split}{pseudo}.conllu"
    
    if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_deprojz_file_path(lang, epoch, pseudo_filter=False):
    """
    file name: lang_method_split_(pseudo)
    """
    ptb_abbr = PTB_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

    pseudo = "pseudo" if pseudo_filter else "none"
    pos = "upos"
    
    read_path = PREDICTION_DIR / f"{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method=most-crossed,pos={pos},epoch={epoch}.conllu"
    write_path = PREDICTION_DIR / f"{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method=most-crossed-deprojz,pos={pos},epoch={epoch}.conllu"
    
    if not read_path.exists():
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
  
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_tree_file_path(lang="Chinese", split="train", pos="UPOS", pseudo_filter=True):
    """
    Input file name: lang_method_split_(pseudo)
    Output dir name: lang=en-penn-ud,filter=none,method=most-crossed,pos=upos
    Output file name: en__train.mrg
    """

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")
    

    ud_abbr = UD_ABBR_LOOKUP[lang]
    ptb_abbr = PTB_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang].lower()


    projz = "most-crossed"
    pos = pos.lower()

    if pseudo_filter:
        pseudo_in = "_pseudo" 
        pseudo_out = "pseudo"
    else:
        pseudo_in = ""
        pseudo_out = "none"
    
   
    read_path = DATA_DIR / "processed" / "ud" / f"UD_{lang}-{treebank}/{ud_abbr}_{projz}_{split}{pseudo_in}.conllu"
    output_dir = DATA_DIR / "processed" / "stanza" / f"{ud_abbr}-{treebank}-ud,filter={pseudo_out},method={projz},pos={pos}"     
      
    if not read_path.exists():
        raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")
  
    write_path = output_dir / f"{ptb_abbr}__{split}.mrg"
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_conllu_file_path(lang, epoch, pos="UPOS", pseudo_filter=True):
    """
    Input file name: lang=en-penn-ud,filter=none,method=most-crossed,pos=upos.mrg
    Output file name: lang=en-penn-ud,filter=none,method=most-crossed,pos=upos.conllu
    """
    ptb_abbr = PTB_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang].lower()

    projz = "most-crossed"
    pos = pos.lower()
    pseudo = "pseudo" if pseudo_filter else "none"

    read_path = PREDICTION_DIR / f"{ptb_abbr}-{treebank}-ud,filter={pseudo},method={projz},pos={pos},epoch={epoch}.mrg"
    write_path = PREDICTION_DIR / f"{ptb_abbr}-{treebank}-ud,filter={pseudo},method={projz},pos={pos},epoch={epoch}.conllu"
            
    if not read_path.exists():
        raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    # if not os.path.exists(write_path):
    #     os.makedirs(write_path, exist_ok=True)
    #     print(f"Created: {write_path}")

    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path

def get_matched_file_path(lang, epoch, pos="UPOS", pseudo_filter=False):
    """
    Input file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20.conllu
    Output file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20,matched=yes.conllu
    """
    ud_abbr = UD_ABBR_LOOKUP[lang]
    ptb_abbr = PTB_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]

    projz = "most-crossed"
    pos = pos.lower()
    pseudo = "pseudo" if pseudo_filter else "none"

    read_system_path = PREDICTION_DIR / f"{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method={projz}-deprojz,pos={pos},epoch={epoch}.conllu"
    read_gold_path = DATA_DIR / "raw" / f"UD_{lang}-{treebank}/{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_system_path = PREDICTION_DIR / f"{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method={projz}-deprojz,pos={pos},epoch={epoch},matched=yes.conllu"
    write_gold_path = DATA_DIR / "processed" / "gold" / f"UD_{lang}-{treebank}/{ud_abbr}_{treebank.lower()}-ud-test,epoch={epoch},matched=yes.conllu"
            
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

    