import os
import sys

from data_loader import read_conllu_file
from dependency2constituency import sentence2tree
from itertools import product

def conllu_to_mrg(read_path, write_path):

    with open(write_path, "w", encoding="utf-8") as f:
        pass

    for sentence in read_conllu_file(read_path):

        ptb = sentence2tree(sentence)

        with open(write_path, "a", encoding="utf-8") as f:
            f.write(ptb.pformat(margin=float('inf')) + "\n")


def construct_tree_file_path(lang="Chinese", split="train", pos="UPOS", pseudo_filter=True):

    # Input file name: lang_(method)_split_(pseudo)_(deprojz)
    # Output dir name: lang=en-penn-ud,filter=none,method=most-crossed,pos=upos
    # Output file name: en__train.mrg

    if split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")
    

    ud_abbr_lookup = {
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
    ptb_abbr_lookup = {
        "Ancient_Greek": "grc",
        "Chinese": "zh-hant",
        "Danish": "da",
        "English": "en",
        "Korean": "ko",
        "Latin": "la",
        "Old_East_Slavic": "orv",
        "Urdu": "ur"
    }

    treebank_lookup = {
        "Ancient_Greek": "Perseus",
        "Chinese": "GSD",
        "Danish": "DDT",
        "English": "Penn",
        "Korean": "GSD",
        "Latin": "UDante",
        "Old_East_Slavic": "RNC",
        "Urdu": "UDTB"

    }

    ud_abbr = ud_abbr_lookup[lang]
    ptb_abbr = ptb_abbr_lookup[lang]
    treebank = treebank_lookup[lang].lower()


    projz = "most-crossed"
    pos = pos.lower()

    if pseudo_filter:
        pseudo_in = "_pseudo" 
        pseudo_out = "pseudo"
    else:
        pseudo_in = ""
        pseudo_out = "none"
    
 
   
    read_path = fr"data/processed/ud/UD_{lang}-{treebank}/{ud_abbr}_{projz}_{split}{pseudo_in}.conllu"
    output_dir = f"data/processed/stanza/{ud_abbr}-{treebank}-ud,filter={pseudo_out},method={projz},pos={pos}/"
            
    if not os.path.exists(read_path):
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created: {output_dir}")
  
    write_path = f"{output_dir}/{ptb_abbr}__{split}.mrg"
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path


def main():
    # lang_name = ["Chinese"] 
    # split_name = ["train", "dev", "test"]
    # pos = ["UPOS"]
    # pseudo_flags = [True, False]

    # paras = list(product(lang_name, split_name, pos, pseudo_flags))
    
        
    # for para in paras:
    #     # _, _, pseudo_flag = para
    #     read_path, write_path = construct_tree_file_path(*para)
    #     write_mrg(read_path, write_path)
    read_path = "data/debug.conllu"
    write_path = "data/debug.output.mrg"
    conllu_to_mrg(read_path, write_path)

if __name__ == "__main__":
    main()
  